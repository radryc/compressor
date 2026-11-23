package codec

import "math"

// entropyBits estimates the order-0 entropy of up to sample bytes of b, in bits
// per byte. It is used only to *choose* a transform, on the encoder side, so the
// floating point here never influences what the decoder does.
func entropyBits(b []byte, sample int) float64 {
	if len(b) == 0 {
		return 0
	}
	if sample > 0 && len(b) > sample {
		b = b[:sample]
	}
	var hist [256]int
	for _, c := range b {
		hist[c]++
	}
	n := float64(len(b))
	var h float64
	for _, c := range hist {
		if c == 0 {
			continue
		}
		p := float64(c) / n
		h -= p * math.Log2(p)
	}
	return h
}
