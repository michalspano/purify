// digitalize turns scanned drawings (PDF/PNG/JPG) into pure black & white
// images: background exactly #fff (or #000), ink exactly #000 (or #fff).
package main

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const usageText = `digitalize: turn scanned drawings into pure black & white images.

Usage:
  digitalize -paper white|black [options] <scan.pdf|.png|.jpg> ...

  -paper white   black ink on white paper -> #000 strokes on #fff
  -paper black   white ink on black paper -> #fff strokes on #000

Options:
`

const examplesText = `
Examples:
  digitalize -paper white images/sample.pdf          # -> images/sample-bw.png
  digitalize -paper black -format pdf -o out/ *.pdf  # one PDF per input in out/
  digitalize -paper white -t 0.3 drawing.pdf         # ignore faint pencil marks
  digitalize -paper white -transparent drawing.pdf   # black ink on transparent PNG
  digitalize -paper white -transparent -invert d.pdf # white ink on transparent PNG

PDF input needs ghostscript (gs) or poppler (pdftoppm) on PATH; poppler's
pdfimages is used to detect the scan's native resolution.
PNG and PDF output is exactly two-tone; JPEG is lossy, so stroke edges carry
slight compression noise.
`

type config struct {
	params
	style
	format string
	out    string
	outDir bool
	dpi    int
}

// style says how the ink mask (1 = ink, 0 = paper) is drawn.
type style struct {
	inkWhite    bool // #fff ink on #000 paper (else #000 on #fff)
	transparent bool // paper is fully transparent (PNG only)
}

// white reports whether mask value v is drawn white.
func (s style) white(v uint8) bool { return (v == 1) == s.inkWhite }

// palette returns the colours of mask values 0 (paper) and 1 (ink).
func (s style) palette() color.Palette {
	ink, paper := color.Color(color.Black), color.Color(color.White)
	if s.inkWhite {
		ink, paper = paper, ink
	}
	if s.transparent {
		paper = color.Transparent
	}
	return color.Palette{paper, ink}
}

type page struct {
	pix  []uint8
	w, h int
	dpi  float64
}

func main() {
	fs := flag.NewFlagSet("digitalize", flag.ExitOnError)
	paper := fs.String("paper", "", "paper colour: `white` (black ink) or black (white ink); required")
	format := fs.String("format", "", "output `format`: png, jpg or pdf (default: from -o extension, else png)")
	out := fs.String("o", "", "output `path`; a directory when there are several inputs (default: <input>-bw.<format>)")
	scale := fs.Int("scale", 2, "supersampling `factor`; >1 gives smoother edges and a larger image")
	thr := fs.Float64("t", 0.2, "ink `threshold`: how much darker than the paper a stroke must be (0-1); lower keeps fainter lines")
	dpi := fs.Int("dpi", 0, "PDF render resolution (default: the scan's native resolution)")
	keepEdges := fs.Bool("keep-edges", false, "keep dark regions touching the image border (by default they are treated as scan artifacts)")
	invert := fs.Bool("invert", false, "swap output colours: white ink on black, or black ink on white for -paper black")
	transparent := fs.Bool("transparent", false, "make the background transparent, keeping only the ink (PNG only)")
	fs.Usage = func() {
		fmt.Fprint(os.Stderr, usageText)
		fs.PrintDefaults()
		fmt.Fprint(os.Stderr, examplesText)
	}

	// Allow flags anywhere, e.g. `digitalize scan.pdf -paper black`.
	var inputs []string
	for args := os.Args[1:]; ; {
		fs.Parse(args)
		if args = fs.Args(); len(args) == 0 {
			break
		}
		inputs, args = append(inputs, args[0]), args[1:]
	}

	cfg := config{params: params{scale: *scale, threshold: *thr, cleanEdges: !*keepEdges},
		style: style{transparent: *transparent}, out: *out, dpi: *dpi}
	switch strings.ToLower(*paper) {
	case "white", "w":
	case "black", "b":
		cfg.negative = true
	case "":
		die("missing -paper white|black (run with -h for help)")
	default:
		die("invalid -paper %q: use white or black", *paper)
	}
	cfg.inkWhite = cfg.negative != *invert // ink keeps its colour unless inverted
	if len(inputs) == 0 {
		fs.Usage()
		os.Exit(2)
	}
	if cfg.scale < 1 || cfg.scale > 8 {
		die("-scale must be between 1 and 8")
	}
	if cfg.threshold <= 0 || cfg.threshold >= 1 {
		die("-t must be between 0 and 1")
	}

	cfg.format = strings.ToLower(strings.TrimPrefix(*format, "."))
	if st, err := os.Stat(cfg.out); cfg.out != "" && (len(inputs) > 1 || strings.HasSuffix(cfg.out, "/") || (err == nil && st.IsDir())) {
		cfg.outDir = true
		if err := os.MkdirAll(cfg.out, 0o755); err != nil {
			die("%v", err)
		}
	}
	if cfg.format == "" && cfg.out != "" && !cfg.outDir {
		cfg.format = strings.ToLower(strings.TrimPrefix(filepath.Ext(cfg.out), "."))
	}
	switch cfg.format {
	case "":
		cfg.format = "png"
	case "jpeg":
		cfg.format = "jpg"
	case "png", "jpg", "pdf":
	default:
		die("unsupported output format %q: use png, jpg or pdf", cfg.format)
	}
	if cfg.transparent && cfg.format != "png" {
		die("-transparent needs PNG output (got %s)", cfg.format)
	}

	failed := false
	for _, in := range inputs {
		if err := run(in, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "digitalize: %s: %v\n", in, err)
			failed = true
		}
	}
	if failed {
		os.Exit(1)
	}
}

func die(f string, a ...any) {
	fmt.Fprintf(os.Stderr, "digitalize: "+f+"\n", a...)
	os.Exit(2)
}

func run(in string, cfg config) error {
	start := time.Now()
	if _, err := os.Stat(in); err != nil {
		return errors.Unwrap(err) // "no such file or directory" etc.
	}
	pages, err := load(in, cfg.dpi)
	if err != nil {
		return err
	}
	if len(pages) == 0 {
		return errors.New("no pages found")
	}
	paths := outputPaths(in, cfg, len(pages))
	for _, p := range paths {
		if same(p, in) {
			return fmt.Errorf("refusing to overwrite the input file; choose another -o")
		}
	}

	pdfPages := make([]pdfPage, len(pages))
	errs := make([]error, len(pages))
	sem := make(chan struct{}, max(1, min(4, runtime.NumCPU()/2)))
	var wg sync.WaitGroup
	for i := range pages {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			pg := pages[i]
			pages[i].pix = nil
			mask, w, h := binarize(pg.pix, pg.w, pg.h, cfg.params)
			switch cfg.format {
			case "pdf":
				pdfPages[i] = encodePDFPage(mask, w, h, pg.dpi*float64(cfg.scale), cfg.style)
			default:
				errs[i] = writeImage(paths[i], cfg.format, mask, w, h, cfg.style)
			}
		}()
	}
	wg.Wait()
	if cfg.format == "pdf" {
		errs = append(errs, writePDF(paths[0], pdfPages))
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	for _, p := range paths {
		fmt.Println(p)
	}
	fmt.Fprintf(os.Stderr, "%s: %d page(s) in %.2fs\n", in, len(pages), time.Since(start).Seconds())
	return nil
}

// outputPaths returns one path per output file (a single one for PDF).
func outputPaths(in string, cfg config, n int) []string {
	ext := "." + cfg.format
	stem := strings.TrimSuffix(filepath.Base(in), filepath.Ext(in)) + "-bw"
	base := filepath.Join(filepath.Dir(in), stem+ext)
	switch {
	case cfg.outDir:
		base = filepath.Join(cfg.out, stem+ext)
	case cfg.out != "":
		base = cfg.out
	}
	if cfg.format == "pdf" || n == 1 {
		return []string{base}
	}
	root := strings.TrimSuffix(base, filepath.Ext(base))
	digits := len(strconv.Itoa(n))
	paths := make([]string, n)
	for i := range paths {
		paths[i] = fmt.Sprintf("%s-%0*d%s", root, digits, i+1, ext)
	}
	return paths
}

func same(a, b string) bool {
	sa, errA := os.Stat(a)
	sb, errB := os.Stat(b)
	return errA == nil && errB == nil && os.SameFile(sa, sb)
}

// ---- input ----------------------------------------------------------------

func load(path string, dpi int) ([]page, error) {
	if strings.EqualFold(filepath.Ext(path), ".pdf") {
		return loadPDF(path, dpi)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(bufio.NewReader(f))
	if errors.Is(err, image.ErrFormat) {
		return nil, errors.New("unsupported input: expected PDF, PNG or JPEG")
	} else if err != nil {
		return nil, err
	}
	return []page{toGray(img)}, nil
}

func toGray(img image.Image) page {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	pix := make([]uint8, w*h)
	switch m := img.(type) {
	case *image.YCbCr: // JPEG: luma is the gray value
		for y := range h {
			copy(pix[y*w:], m.Y[m.YOffset(b.Min.X, b.Min.Y+y):][:w])
		}
	case *image.Gray:
		for y := range h {
			copy(pix[y*w:], m.Pix[m.PixOffset(b.Min.X, b.Min.Y+y):][:w])
		}
	default:
		for y := range h {
			for x := range w {
				pix[y*w+x] = color.GrayModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.Gray).Y
			}
		}
	}
	return page{pix, w, h, 72}
}

// loadPDF rasterizes every page to 8-bit gray at the scan's native resolution.
func loadPDF(path string, dpi int) ([]page, error) {
	if dpi <= 0 {
		dpi = nativeDPI(path)
	}
	r := strconv.Itoa(dpi)
	var cmd *exec.Cmd
	if _, err := exec.LookPath("gs"); err == nil {
		cmd = exec.Command("gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sstdout=%stderr",
			"-sDEVICE=pgmraw", "-r"+r, "-sOutputFile=-", path)
	} else if _, err := exec.LookPath("pdftoppm"); err == nil {
		cmd = exec.Command("pdftoppm", "-gray", "-r", r, path)
	} else {
		return nil, errors.New("PDF input needs ghostscript (brew install ghostscript) or poppler (brew install poppler)")
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	var pages []page
	br := bufio.NewReaderSize(stdout, 1<<20)
	for {
		if _, err := br.Peek(1); err != nil {
			break
		}
		pg, err := readPGM(br)
		if err != nil {
			cmd.Process.Kill()
			cmd.Wait()
			return nil, fmt.Errorf("reading rasterized page: %w", err)
		}
		pg.dpi = float64(dpi)
		pages = append(pages, pg)
	}
	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("%s: %v %s", filepath.Base(cmd.Path), err, strings.TrimSpace(stderr.String()))
	}
	return pages, nil
}

// nativeDPI returns the resolution of the largest raster image in the PDF
// (a phone scan is one image per page), so rendering neither blurs nor
// invents pixels. Falls back to 150 dpi.
func nativeDPI(path string) int {
	out, err := exec.Command("pdfimages", "-list", path).Output()
	if err != nil {
		return 150
	}
	best, area := 0.0, 0
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 14 || f[2] != "image" {
			continue
		}
		w, _ := strconv.Atoi(f[3])
		h, _ := strconv.Atoi(f[4])
		xp, _ := strconv.ParseFloat(f[12], 64)
		yp, _ := strconv.ParseFloat(f[13], 64)
		if w*h > area {
			best, area = max(xp, yp), w*h
		}
	}
	if best <= 0 {
		return 150
	}
	return min(max(int(math.Round(best)), 50), 1200)
}

// readPGM reads one binary (P5, 8-bit) PGM image.
func readPGM(r *bufio.Reader) (page, error) {
	var hdr [4]int
	magic := make([]byte, 2)
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != "P5" {
		return page{}, errors.New("not a binary PGM stream")
	}
	for i := 1; i < 4; i++ {
		c, err := r.ReadByte()
		for err == nil && (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '#') {
			if c == '#' {
				_, err = r.ReadString('\n')
			}
			if err == nil {
				c, err = r.ReadByte()
			}
		}
		for err == nil && c >= '0' && c <= '9' {
			hdr[i] = hdr[i]*10 + int(c-'0')
			c, err = r.ReadByte()
		}
		if err != nil {
			return page{}, err
		}
	}
	w, h := hdr[1], hdr[2]
	if hdr[3] != 255 || w <= 0 || h <= 0 {
		return page{}, fmt.Errorf("unsupported PGM (%dx%d, maxval %d)", w, h, hdr[3])
	}
	pix := make([]uint8, w*h)
	if _, err := io.ReadFull(r, pix); err != nil {
		return page{}, err
	}
	return page{pix: pix, w: w, h: h}, nil
}

// ---- output ---------------------------------------------------------------

func writeImage(path, format string, mask []uint8, w, h int, st style) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	rect := image.Rect(0, 0, w, h)
	if format == "png" {
		// Mask values index a 2-colour palette: Go writes a compact 1-bit PNG
		// (with a tRNS chunk when the paper is transparent).
		err = png.Encode(bw, &image.Paletted{Pix: mask, Stride: w, Rect: rect, Palette: st.palette()})
	} else {
		img := &image.Gray{Pix: make([]uint8, w*h), Stride: w, Rect: rect}
		for i, v := range mask {
			if st.white(v) {
				img.Pix[i] = 255
			}
		}
		err = jpeg.Encode(bw, img, &jpeg.Options{Quality: 100})
	}
	return errors.Join(err, bw.Flush(), f.Close())
}

type pdfPage struct {
	w, h       int
	wPt, hPt   float64
	compressed []byte
}

// encodePDFPage packs the mask into a Flate-compressed 1-bit DeviceGray image.
// The page keeps the physical size of the scan (dpi = output pixels per inch).
func encodePDFPage(mask []uint8, w, h int, dpi float64, st style) pdfPage {
	stride := (w + 7) / 8
	bits := make([]byte, stride*h)
	for y := range h {
		row := bits[y*stride:]
		for x, v := range mask[y*w : y*w+w] {
			if st.white(v) {
				row[x>>3] |= 0x80 >> (x & 7)
			}
		}
	}
	var buf bytes.Buffer
	zw := zlib.NewWriter(&buf)
	zw.Write(bits)
	zw.Close()
	return pdfPage{w, h, float64(w) * 72 / dpi, float64(h) * 72 / dpi, buf.Bytes()}
}

func writePDF(path string, pages []pdfPage) error {
	var b bytes.Buffer
	offsets := make([]int, 3+3*len(pages))
	obj := func(id int, dict string, stream []byte) {
		offsets[id] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\n", id, dict)
		if stream != nil {
			b.WriteString("stream\n")
			b.Write(stream)
			b.WriteString("\nendstream\n")
		}
		b.WriteString("endobj\n")
	}
	b.WriteString("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
	kids := make([]string, len(pages))
	for i := range pages {
		kids[i] = fmt.Sprintf("%d 0 R", 3+3*i)
	}
	obj(1, "<< /Type /Catalog /Pages 2 0 R >>", nil)
	obj(2, fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(pages)), nil)
	for i, p := range pages {
		id := 3 + 3*i
		content := fmt.Sprintf("q %.3f 0 0 %.3f 0 0 cm /Im0 Do Q", p.wPt, p.hPt)
		obj(id, fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.3f %.3f] "+
			"/Resources << /XObject << /Im0 %d 0 R >> >> /Contents %d 0 R >>", p.wPt, p.hPt, id+2, id+1), nil)
		obj(id+1, fmt.Sprintf("<< /Length %d >>", len(content)), []byte(content))
		obj(id+2, fmt.Sprintf("<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "+
			"/BitsPerComponent 1 /Filter /FlateDecode /Length %d >>", p.w, p.h, len(p.compressed)), p.compressed)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(offsets))
	for _, off := range offsets[1:] {
		fmt.Fprintf(&b, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(offsets), xref)
	return os.WriteFile(path, b.Bytes(), 0o644)
}
