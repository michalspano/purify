package main

import (
	"math"
	"runtime"
	"slices"
	"sync"
)

// params controls the binarization pipeline.
type params struct {
	negative   bool    // white ink on black paper: process the negative
	scale      int     // supersampling factor (>1 gives smoother contours)
	threshold  float64 // how much darker than the paper (0..1) a stroke must be
	cleanEdges bool    // remove scan-border artifacts touching the image edge
}

// binarize turns an 8-bit grayscale scan (w×h) into an ink mask (1 = ink,
// 0 = paper) of size (w·scale)×(h·scale).
//
// Pipeline: flatten lighting → light denoise → supersample → hysteresis
// threshold → remove scan-border artifacts → drop specks / fill pinholes.
func binarize(pix []uint8, w, h int, p params) (mask []uint8, W, H int) {
	minDim := float64(min(w, h))
	img := make([]float32, len(pix))
	for i, v := range pix {
		if p.negative {
			v = 255 - v
		}
		img[i] = float32(v)
	}
	flatten(img, w, h, max(16, int(minDim)/24))
	img = blur(img, w, h, max(0.6, minDim/1500))

	W, H = w*p.scale, h*p.scale
	strong := float32(255 * (1 - p.threshold))   // definitely ink: seeds
	weak := float32(255 * (1 - p.threshold*0.4)) // ink only if connected to a seed
	mask = classify(img, w, h, p.scale, strong, weak)
	hysteresis(mask, W, H)

	s := float64(p.scale)
	if p.cleanEdges {
		cleanEdges(mask, W, H, max(2, int(0.02*minDim*s)), 2*p.scale)
	}
	speck := int(12 * (minDim / 1000) * (minDim / 1000) * s * s)
	removeSmall(mask, W, H, 1, speck, true, false) // isolated dots of grain
	removeSmall(mask, W, H, 0, speck, false, true) // pinholes inside strokes
	return mask, W, H
}

// flatten divides out uneven lighting/paper tone so that paper becomes 255.
// The paper level is the 90th percentile of t×t tiles, median-filtered to
// reject ink-dense tiles, and bilinearly interpolated between tile centres.
func flatten(img []float32, w, h, t int) {
	gw, gh := (w+t-1)/t, (h+t-1)/t
	grid := make([]float32, gw*gh)
	parallel(gh, func(lo, hi int) {
		var hist [256]int
		for gy := lo; gy < hi; gy++ {
			for gx := range gw {
				clear(hist[:])
				x0, y0, x1, y1 := gx*t, gy*t, min(gx*t+t, w), min(gy*t+t, h)
				for y := y0; y < y1; y++ {
					for _, v := range img[y*w+x0 : y*w+x1] {
						hist[int(v)]++
					}
				}
				rank, c, v := (x1-x0)*(y1-y0)*9/10, 0, 0
				for ; v < 255; v++ {
					if c += hist[v]; c > rank {
						break
					}
				}
				grid[gy*gw+gx] = float32(max(v, 1))
			}
		}
	})
	grid = median3(grid, gw, gh)

	tf := float32(t)
	xa, xb, xf := lerpTable(w, tf, gw)
	parallel(h, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			ya, yb, fy := lerpIdx(y, tf, gh)
			g0, g1 := grid[ya*gw:], grid[yb*gw:]
			row := img[y*w : y*w+w]
			for x := range row {
				a, b, fx := xa[x], xb[x], xf[x]
				bg := (g0[a]+(g0[b]-g0[a])*fx)*(1-fy) + (g1[a]+(g1[b]-g1[a])*fx)*fy
				row[x] = min(row[x]*255/bg, 255)
			}
		}
	})
}

// median3 applies a 3×3 median filter to a small grid.
func median3(g []float32, w, h int) []float32 {
	out := make([]float32, len(g))
	var nb [9]float32
	for y := range h {
		for x := range w {
			k := 0
			for yy := max(y-1, 0); yy <= min(y+1, h-1); yy++ {
				for xx := max(x-1, 0); xx <= min(x+1, w-1); xx++ {
					nb[k] = g[yy*w+xx]
					k++
				}
			}
			slices.Sort(nb[:k])
			out[y*w+x] = nb[k/2]
		}
	}
	return out
}

// blur is a separable Gaussian blur; it reuses src for the result.
func blur(src []float32, w, h int, sigma float64) []float32 {
	r := int(math.Ceil(3 * sigma))
	k := make([]float32, 2*r+1)
	var sum float64
	for i := -r; i <= r; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		k[i+r], sum = float32(v), sum+v
	}
	for i := range k {
		k[i] /= float32(sum)
	}
	tmp := make([]float32, len(src))
	parallel(h, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			in, out := src[y*w:y*w+w], tmp[y*w:y*w+w]
			for x := range out {
				var acc float32
				for j, kv := range k {
					acc += kv * in[min(max(x+j-r, 0), w-1)]
				}
				out[x] = acc
			}
		}
	})
	parallel(h, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			out := src[y*w : y*w+w]
			clear(out)
			for j, kv := range k {
				yy := min(max(y+j-r, 0), h-1)
				for x, v := range tmp[yy*w : yy*w+w] {
					out[x] += kv * v
				}
			}
		}
	})
	return src
}

// classify upsamples img bilinearly by s and labels every output pixel:
// 2 = strong ink, 1 = weak ink, 0 = paper.
func classify(img []float32, w, h, s int, strong, weak float32) []uint8 {
	W, H := w*s, h*s
	m := make([]uint8, W*H)
	sf := float32(s)
	xa, xb, xf := lerpTable(W, sf, w)
	parallel(H, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			ya, yb, fy := lerpIdx(y, sf, h)
			r0, r1 := img[ya*w:ya*w+w], img[yb*w:yb*w+w]
			out := m[y*W : y*W+W]
			for x := range out {
				a, b, fx := xa[x], xb[x], xf[x]
				v := (r0[a]+(r0[b]-r0[a])*fx)*(1-fy) + (r1[a]+(r1[b]-r1[a])*fx)*fy
				if v < strong {
					out[x] = 2
				} else if v < weak {
					out[x] = 1
				}
			}
		}
	})
	return m
}

// lerpIdx maps destination index i onto the two source samples (cells of
// size t, n of them) that bracket it, plus the interpolation weight.
func lerpIdx(i int, t float32, n int) (int, int, float32) {
	f := (float32(i)+0.5)/t - 0.5
	if f <= 0 {
		return 0, 0, 0
	}
	a := int(f)
	if a >= n-1 {
		return n - 1, n - 1, 0
	}
	return a, a + 1, f - float32(a)
}

func lerpTable(size int, t float32, n int) ([]int, []int, []float32) {
	a, b, f := make([]int, size), make([]int, size), make([]float32, size)
	for i := range size {
		a[i], b[i], f[i] = lerpIdx(i, t, n)
	}
	return a, b, f
}

// set is a lookup table of mask values that belong to the foreground.
type set [256]bool

var (
	anyInk = set{1: true, 2: true}
	isZero = set{0: true}
	isOne  = set{1: true}
)

// label finds connected components of pixels whose value is in fg using a
// two-pass union-find scan. It returns a component id per pixel (0 = not
// foreground), the size of each component, and whether it touches the border.
func label(m []uint8, W, H int, fg *set, conn8 bool) (lab, size []int32, edge []bool) {
	lab = make([]int32, len(m))
	parent := []int32{0}
	find := func(x int32) int32 {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	join := func(l, n int32) int32 {
		if n == 0 || n == l {
			return l
		}
		if l == 0 {
			return n
		}
		a, b := find(l), find(n)
		if a > b {
			a, b = b, a
		}
		parent[b] = a
		return a
	}
	for y := range H {
		row := y * W
		for x := range W {
			i := row + x
			if !fg[m[i]] {
				continue
			}
			var l int32
			if x > 0 {
				l = join(l, lab[i-1])
			}
			if y > 0 {
				l = join(l, lab[i-W])
				if conn8 && x > 0 {
					l = join(l, lab[i-W-1])
				}
				if conn8 && x < W-1 {
					l = join(l, lab[i-W+1])
				}
			}
			if l == 0 {
				l = int32(len(parent))
				parent = append(parent, l)
			}
			lab[i] = l
		}
	}
	size, edge = make([]int32, len(parent)), make([]bool, len(parent))
	for i, l := range lab {
		if l != 0 {
			l = find(l)
			lab[i] = l
			size[l]++
		}
	}
	for x := range W {
		edge[lab[x]], edge[lab[(H-1)*W+x]] = true, true
	}
	for y := range H {
		edge[lab[y*W]], edge[lab[y*W+W-1]] = true, true
	}
	edge[0] = false
	return lab, size, edge
}

// hysteresis keeps weak ink only where it is 8-connected to strong ink, so
// faint parts of real strokes survive while faint paper grain/smudges do not.
// On return m holds 1 = ink, 0 = paper.
func hysteresis(m []uint8, W, H int) {
	lab, size, _ := label(m, W, H, &anyInk, true)
	keep := make([]bool, len(size))
	for i, v := range m {
		if v == 2 {
			keep[lab[i]] = true
		}
	}
	for i, l := range lab {
		m[i] = 0
		if keep[l] {
			m[i] = 1
		}
	}
}

// removeSmall flips components of val (0 or 1) smaller than minArea to the
// opposite value. With keepEdge, components touching the border stay.
func removeSmall(m []uint8, W, H int, val uint8, minArea int, conn8, keepEdge bool) {
	fg := &isZero
	if val == 1 {
		fg = &isOne
	}
	lab, size, edge := label(m, W, H, fg, conn8)
	for i, l := range lab {
		if l != 0 && int(size[l]) < minArea && !(keepEdge && edge[l]) {
			m[i] = 1 - val
		}
	}
}

// cleanEdges removes typical scan-border artifacts: whatever lies beyond the
// paper (table, neighbouring page, shadows) that phone scanners leave along
// the image edge. r is the margin width; anything within e pixels of the
// border counts as touching it (tolerates thin light lines at the very edge).
func cleanEdges(m []uint8, W, H, r, e int) {
	// 1. Thick blobs touching the border (e.g. the table showing past a
	//    rounded paper corner): remove the morphological opening, with a disc
	//    of radius r, of ink that reaches the border. Thin strokes are kept.
	//    Runs on a grid of f×f cells (majority vote) for speed.
	f := max(1, r/8)
	cw, ch, rc := (W+f-1)/f, (H+f-1)/f, r/f
	cell := make([]int32, W)
	for x := range W {
		cell[x] = int32(x / f)
	}
	cnt := make([]int32, cw*ch)
	for y := range H {
		c := cnt[y/f*cw:]
		for x, v := range m[y*W : y*W+W] {
			c[cell[x]] += int32(v)
		}
	}
	src := make([]uint8, cw*ch) // paper cells away from the very edge
	for i, n := range cnt {
		cx, cy := i%cw, i/cw
		if area := min(f, W-cx*f) * min(f, H-cy*f); 2*int(n) < area {
			src[i] = 1
		}
	}
	eachInFrame(cw, ch, (e+f-1)/f, func(i int) { src[i] = 0 })
	for i, v := range chamfer(cw, ch, src) {
		src[i] = 0
		if v >= int32(3*rc) {
			src[i] = 1 // core: deep inside a thick ink region
		}
	}
	lab, _, edge := label(src, cw, ch, &isOne, true)
	found := false
	for i, l := range lab {
		src[i] = 0
		if edge[l] {
			src[i], found = 1, true
		}
	}
	if found {
		lim := int32(3 * (rc + 1))
		gone := make([]bool, cw*ch)
		for i, v := range chamfer(cw, ch, src) {
			gone[i] = v <= lim
		}
		for y := range H {
			g := gone[y/f*cw:]
			row := m[y*W : y*W+W]
			for x := range row {
				if g[cell[x]] {
					row[x] = 0
				}
			}
		}
	}

	// 2. Thin slivers along the border (paper edge, shadow lines): clear ink
	//    that is reachable from the border without leaving the r-wide margin.
	stack := make([]int32, 0, 4096)
	eachInFrame(W, H, e, func(i int) {
		if m[i] == 1 {
			m[i] = 0
			stack = append(stack, int32(i))
		}
	})
	for len(stack) > 0 {
		j := int(stack[len(stack)-1])
		stack = stack[:len(stack)-1]
		x, y := j%W, j/W
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				xx, yy := x+dx, y+dy
				if xx < 0 || yy < 0 || xx >= W || yy >= H || (xx >= r && yy >= r && xx < W-r && yy < H-r) {
					continue
				}
				if k := yy*W + xx; m[k] == 1 {
					m[k] = 0
					stack = append(stack, int32(k))
				}
			}
		}
	}
}

// eachInFrame calls fn for every pixel index within b pixels of the border.
func eachInFrame(W, H, b int, fn func(i int)) {
	for y := range H {
		if y < b || y >= H-b {
			for x := range W {
				fn(y*W + x)
			}
			continue
		}
		for x := range min(b, W) {
			fn(y*W + x)
			fn(y*W + W - 1 - x)
		}
	}
}

// chamfer returns the 3-4 chamfer distance (≈ 3× Euclidean) from every pixel
// to the nearest pixel where src is non-zero.
func chamfer(W, H int, src []uint8) []int32 {
	const inf = math.MaxInt32 / 2
	d := make([]int32, W*H)
	for i, v := range src {
		if v == 0 {
			d[i] = inf
		}
	}
	for y := range H {
		for x := range W {
			i := y*W + x
			v := d[i]
			if v == 0 {
				continue
			}
			if x > 0 {
				v = min(v, d[i-1]+3)
			}
			if y > 0 {
				v = min(v, d[i-W]+3)
				if x > 0 {
					v = min(v, d[i-W-1]+4)
				}
				if x < W-1 {
					v = min(v, d[i-W+1]+4)
				}
			}
			d[i] = v
		}
	}
	for y := H - 1; y >= 0; y-- {
		for x := W - 1; x >= 0; x-- {
			i := y*W + x
			v := d[i]
			if v == 0 {
				continue
			}
			if x < W-1 {
				v = min(v, d[i+1]+3)
			}
			if y < H-1 {
				v = min(v, d[i+W]+3)
				if x < W-1 {
					v = min(v, d[i+W+1]+4)
				}
				if x > 0 {
					v = min(v, d[i+W-1]+4)
				}
			}
			d[i] = v
		}
	}
	return d
}

// parallel splits [0,n) into contiguous chunks processed concurrently.
func parallel(n int, fn func(lo, hi int)) {
	workers := min(runtime.NumCPU(), n)
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, min(lo+chunk, n))
	}
	wg.Wait()
}
