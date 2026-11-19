package main

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
)

// SVG export: the ink mask is traced into closed outlines along pixel edges,
// simplified (Douglas–Peucker) and smoothed with quadratic Béziers, giving
// the clean, scalable look of a drawing made digitally.

const (
	svgTolerance = 0.7 // max deviation from the pixel outline when simplifying (px)
	svgCorner    = 3.0 // rounding radius kept at sharp corners (px)
	svgSharp     = 0.5 // turns sharper than acos(0.5) = 60° count as corners
	svgSmooth    = 4   // passes of the [1 2 1] filter that removes pixel jitter
)

type pt struct{ x, y float64 }

// hex returns the SVG colour of mask value v.
func (s style) hex(v uint8) string {
	if s.white(v) {
		return "#fff"
	}
	return "#000"
}

func writeSVG(path string, mask []uint8, w, h int, dpi float64, st style) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	b := bufio.NewWriterSize(f, 1<<20)
	fmt.Fprintf(b, `<svg xmlns="http://www.w3.org/2000/svg" width="%.2fpt" height="%.2fpt" viewBox="0 0 %d %d">`+"\n",
		float64(w)*72/dpi, float64(h)*72/dpi, w, h)
	if !st.transparent {
		fmt.Fprintf(b, `<rect width="%d" height="%d" fill="%s"/>`+"\n", w, h, st.hex(0))
	}
	// Path coordinates are in 1/10 px (hence the scale) to keep numbers short.
	fmt.Fprintf(b, `<path fill="%s" transform="scale(.1)" d="`, st.hex(1))
	var buf []byte
	for _, c := range traceOutlines(mask, w, h) {
		buf = appendOutline(buf[:0], simplify(smooth(c, svgSmooth), svgTolerance))
		b.Write(buf)
	}
	b.WriteString("\"/>\n</svg>\n")
	return errors.Join(b.Flush(), f.Close())
}

// Directions, clockwise on screen: right, down, left, up.
var steps = [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}

// traceOutlines follows the pixel edges between ink and paper with the ink on
// the right-hand side and returns every closed outline as the midpoints of its
// unit edges. Outer outlines run clockwise and holes counter-clockwise, so
// SVG's default nonzero fill rule renders holes. Ink pixels that touch only
// diagonally are joined, matching the 8-connectivity used by binarize.
func traceOutlines(mask []uint8, w, h int) [][]pt {
	cw := w + 1
	// Per pixel corner: bit d = a boundary edge leaves it in direction d;
	// bit 4+d = that edge has been traced.
	edges := make([]uint8, cw*(h+1))
	ink := func(x, y int) bool { return x >= 0 && y >= 0 && x < w && y < h && mask[y*w+x] == 1 }
	for y := range h {
		for x := range w {
			if mask[y*w+x] == 0 {
				continue
			}
			if !ink(x, y-1) {
				edges[y*cw+x] |= 1 // top edge, heading right
			}
			if !ink(x+1, y) {
				edges[y*cw+x+1] |= 2 // right edge, heading down
			}
			if !ink(x, y+1) {
				edges[(y+1)*cw+x+1] |= 4 // bottom edge, heading left
			}
			if !ink(x-1, y) {
				edges[(y+1)*cw+x] |= 8 // left edge, heading up
			}
		}
	}
	var out [][]pt
	for c, e := range edges {
		if e == 0 {
			continue
		}
		for d := range 4 {
			if e&(1<<d) == 0 || edges[c]&(16<<d) != 0 {
				continue
			}
			var poly []pt
			x, y, dir := c%cw, c/cw, d
			for i := c; edges[i]&(16<<dir) == 0; i = y*cw + x {
				edges[i] |= 16 << dir
				s := steps[dir]
				poly = append(poly, pt{float64(x) + float64(s[0])/2, float64(y) + float64(s[1])/2})
				x, y = x+s[0], y+s[1]
				next := edges[y*cw+x]
				for _, turn := range [3]int{3, 0, 1} { // left (joins diagonal ink), straight, right
					if nd := (dir + turn) % 4; next&(1<<nd) != 0 {
						dir = nd
						break
					}
				}
			}
			out = append(out, poly)
		}
	}
	return out
}

// smooth low-pass filters a closed outline (a [1 2 1] kernel per pass),
// removing the pixel staircase and scan grain while keeping the shape.
func smooth(p []pt, passes int) []pt {
	n := len(p)
	if n < 8 {
		return p
	}
	q := make([]pt, n)
	for range passes {
		for i, b := range p {
			a, c := p[(i+n-1)%n], p[(i+1)%n]
			q[i] = pt{(a.x + 2*b.x + c.x) / 4, (a.y + 2*b.y + c.y) / 4}
		}
		p, q = q, p
	}
	return p
}

// simplify reduces a closed polygon with Douglas–Peucker: it keeps only the
// vertices needed to stay within tol of the original outline.
func simplify(p []pt, tol float64) []pt {
	n := len(p)
	if n < 8 {
		return p
	}
	far, best := 0, 0.0
	for i, q := range p {
		if d := (q.x-p[0].x)*(q.x-p[0].x) + (q.y-p[0].y)*(q.y-p[0].y); d > best {
			far, best = i, d
		}
	}
	keep := make([]bool, n)
	keep[0], keep[far] = true, true
	stack := [][2]int{{0, far}, {far, n}} // index n wraps to 0
	for len(stack) > 0 {
		a, b := stack[len(stack)-1][0], stack[len(stack)-1][1]
		stack = stack[:len(stack)-1]
		A, B := p[a], p[b%n]
		dx, dy := B.x-A.x, B.y-A.y
		idx, worst := -1, tol*tol*(dx*dx+dy*dy) // compares squared cross products
		for i := a + 1; i < b; i++ {
			if c := (p[i].x-A.x)*dy - (p[i].y-A.y)*dx; c*c > worst {
				idx, worst = i, c*c
			}
		}
		if idx >= 0 {
			keep[idx] = true
			stack = append(stack, [2]int{a, idx}, [2]int{idx, b})
		}
	}
	out := make([]pt, 0, n/4)
	for i, k := range keep {
		if k {
			out = append(out, p[i])
		}
	}
	if len(out) < 3 {
		return p
	}
	return out
}

// appendOutline writes a closed polygon as SVG path data (1/10 px units,
// relative commands). Each vertex becomes a quadratic curve with the vertex as
// control point: gentle turns are smoothed across half of each adjacent
// segment, sharp corners only within svgCorner so they stay crisp.
func appendOutline(b []byte, v []pt) []byte {
	n := len(v)
	if n < 3 {
		return b
	}
	at := func(i int) pt { return v[(i+n)%n] }
	arms := func(i int) (in, out pt) { // where the curve around vertex i starts and ends
		p, c, q := at(i-1), at(i), at(i+1)
		ax, ay, bx, by := c.x-p.x, c.y-p.y, q.x-c.x, q.y-c.y
		la, lb := math.Hypot(ax, ay), math.Hypot(bx, by)
		ta, tb := la/2, lb/2
		if ax*bx+ay*by < svgSharp*la*lb {
			ta, tb = min(ta, svgCorner), min(tb, svgCorner)
		}
		return pt{c.x - ax/la*ta, c.y - ay/la*ta}, pt{c.x + bx/lb*tb, c.y + by/lb*tb}
	}
	tenth := func(f float64) int { return int(math.Round(f * 10)) }
	num := func(i int) {
		if last := len(b) - 1; i >= 0 && last >= 0 && b[last] >= '0' && b[last] <= '9' {
			b = append(b, ' ') // a minus sign separates numbers by itself
		}
		b = strconv.AppendInt(b, int64(i), 10)
	}
	cmd := byte(0)
	op := func(c byte) { // emit a command letter unless it repeats implicitly
		if c != cmd {
			b, cmd = append(b, c), c
		}
	}

	_, start := arms(0)
	cx, cy := tenth(start.x), tenth(start.y)
	b = append(b, 'M')
	num(cx)
	num(cy)
	cmd = 'M'
	for i := 1; i <= n; i++ {
		in, out := arms(i)
		if x, y := tenth(in.x), tenth(in.y); x != cx || y != cy {
			op('l')
			num(x - cx)
			num(y - cy)
			cx, cy = x, y
		}
		op('q')
		c := at(i)
		num(tenth(c.x) - cx)
		num(tenth(c.y) - cy)
		x, y := tenth(out.x), tenth(out.y)
		num(x - cx)
		num(y - cy)
		cx, cy = x, y
	}
	return append(b, 'z')
}
