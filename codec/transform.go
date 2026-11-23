package codec

// Spatial pre-transform for image-like blocks.
//
// A byte-level model sees an image as a one-dimensional stream, so vertical
// redundancy (the same pixel column in the next row) is invisible to it. Coding
// the residual against the byte one row back turns a smooth gradient into runs
// of zeros, which every predictor in the codec handles well.
//
// The row stride is discovered from the data rather than parsed from a header:
// a block in the middle of a file has no header, and the same trick works for
// any fixed-stride raster (BMP, PPM, raw frames, audio channels).

// DeltaEncode replaces p[i] with p[i]-p[i-stride] (modulo 256), in place.
func DeltaEncode(p []byte, stride int) {
	if stride <= 0 || stride >= len(p) {
		return
	}
	for i := len(p) - 1; i >= stride; i-- {
		p[i] -= p[i-stride]
	}
}

// DeltaDecode reverses DeltaEncode, in place, and must run forwards.
func DeltaDecode(p []byte, stride int) {
	if stride <= 0 || stride >= len(p) {
		return
	}
	for i := stride; i < len(p); i++ {
		p[i] += p[i-stride]
	}
}

// StrideMinMatch is the fraction of bytes that must agree at a candidate lag
// before the data is treated as a raster. Noise sits near 1/256; a raster row
// is usually far above this.
const StrideMinMatch = 0.15

// DetectStride returns the lag with the strongest byte agreement, or 0 when the
// block does not look like a fixed-stride raster.
//
// It samples at most 2 KiB of positions for each candidate lag, so the search
// costs a few million byte comparisons regardless of block size.
func DetectStride(b []byte) int {
	const (
		maxStrideCap = 3072
		sampleCap    = 1 << 11 // 2048 positions per lag
	)
	n := len(b)
	if n < 4096 {
		return 0 // too short to trust a stride estimate
	}
	maxStride := n / 4
	if maxStride > maxStrideCap {
		maxStride = maxStrideCap
	}
	sample := sampleCap
	if sample > n-maxStride {
		sample = n - maxStride
	}
	// Use the middle of the block so a header at offset 0 does not skew it.
	base := n/2 - sample/2
	if base < 0 {
		base = 0
	}
	if base+sample > n {
		base = n - sample
	}
	probe := b[base : base+sample]

	best, bestScore := 0, 0.0
	// Step 1 keeps the scan cheap; step 4 catches every realistic raster width
	// (which is a multiple of the pixel size or padded to 4 bytes), and a short
	// refinement pass recovers the exact lag.
	for stride := 8; stride <= maxStride; stride += 4 {
		var match int
		for i := 0; i < sample; i++ {
			if probe[i] == b[base+stride+i] {
				match++
			}
		}
		score := float64(match) / float64(sample)
		if score > bestScore {
			best, bestScore = stride, score
		}
	}
	if best == 0 || bestScore < StrideMinMatch {
		return 0
	}
	// Refine around the coarse winner.
	lo := best - 4
	if lo < 2 {
		lo = 2
	}
	for stride := lo; stride <= best+4 && base+stride+sample <= n; stride++ {
		var match int
		for i := 0; i < sample; i++ {
			if probe[i] == b[base+stride+i] {
				match++
			}
		}
		if score := float64(match) / float64(sample); score > bestScore {
			best, bestScore = stride, score
		}
	}
	return best
}

// PaethEncode replaces p[i] with the residual against the Paeth predictor of
// its left, upper and upper-left neighbours. Unlike a plain row delta it
// predicts smooth gradients well, which is what makes photographs compressible.
func PaethEncode(p []byte, stride int) {
	if stride <= 0 || stride >= len(p) {
		return
	}
	for i := len(p) - 1; i >= stride; i-- {
		p[i] -= paeth(p, i, stride)
	}
}

// PaethDecode reverses PaethEncode; it must run forwards.
func PaethDecode(p []byte, stride int) {
	if stride <= 0 || stride >= len(p) {
		return
	}
	for i := stride; i < len(p); i++ {
		p[i] += paeth(p, i, stride)
	}
}

// paeth returns the PNG Paeth prediction for position i. Callers guarantee
// i >= stride, so the upper neighbour exists.
func paeth(p []byte, i, stride int) byte {
	up := int(p[i-stride])
	if i < stride+1 || i == stride {
		return byte(up)
	}
	left := int(p[i-1])
	upLeft := int(p[i-stride-1])
	pa := absInt(up - upLeft)
	pb := absInt(left - upLeft)
	pc := absInt(left + up - 2*upLeft)
	if pa <= pb && pa <= pc {
		return byte(left)
	}
	if pb <= pc {
		return byte(up)
	}
	return byte(upLeft)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// Transform identifies a spatial pre-transform.
type Transform uint8

const (
	TransformNone  Transform = 0
	TransformDelta Transform = 1
	TransformPaeth Transform = 2
)

// Valid reports whether t is known.
func (t Transform) Valid() bool { return t <= TransformPaeth }

// Apply encodes the transform in place.
func (t Transform) Apply(p []byte, stride int) {
	switch t {
	case TransformDelta:
		DeltaEncode(p, stride)
	case TransformPaeth:
		PaethEncode(p, stride)
	}
}

// Undo reverses Apply in place.
func (t Transform) Undo(p []byte, stride int) {
	switch t {
	case TransformDelta:
		DeltaDecode(p, stride)
	case TransformPaeth:
		PaethDecode(p, stride)
	}
}

// ChooseTransform picks the spatial predictor that minimises the order-0
// entropy of the residual, or returns (TransformNone, 0) when no predictor
// earns its keep.
//
// This is a measured choice rather than a rule of thumb: on a linear gradient a
// row delta wins, on a smooth photograph a Paeth predictor wins, and on noise
// neither does. The decision is encoder-side only — the decoder reads it from
// the frame header — so the floating-point entropy estimate cannot affect
// determinism.
func ChooseTransform(b []byte) (Transform, int) {
	const (
		scanSample = 1 << 12 // 4 KiB for the coarse scan
		bestSample = 1 << 16 // 64 KiB to confirm the winner
		maxStride  = 2048
	)
	n := len(b)
	if n < 4*4096 {
		return TransformNone, 0
	}
	base := entropyBits(b, scanSample)

	bestT, bestStride := TransformNone, 0
	bestE := base
	try := func(t Transform, stride int, sample int) {
		if stride <= 0 || stride >= n {
			return
		}
		w := append([]byte(nil), b[:minInt(sample, n)]...)
		t.Apply(w, stride)
		if e := entropyBits(w, 0); e < bestE {
			bestE, bestT, bestStride = e, t, stride
		}
	}

	for stride := 4; stride <= maxStride; stride += 4 {
		try(TransformDelta, stride, scanSample)
		try(TransformPaeth, stride, scanSample)
	}
	if bestT == TransformNone {
		return TransformNone, 0
	}
	// Confirm and refine around the winner on a larger sample.
	lo := bestStride - 4
	if lo < 2 {
		lo = 2
	}
	for stride := lo; stride <= bestStride+4; stride++ {
		try(TransformDelta, stride, bestSample)
		try(TransformPaeth, stride, bestSample)
	}
	// Require a real win, otherwise the transform is pure overhead.
	if bestE > base*0.97 {
		return TransformNone, 0
	}
	return bestT, bestStride
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
