package stream

import "github.com/radryc/compressor/codec"

// Content classification.
//
// The codec configuration is chosen per block from cheap byte statistics, not
// from a file name or a magic number: a block in the middle of a stream has
// neither. The decision is stored in the frame header, so the decoder applies
// exactly what the encoder used and nothing has to be guessed on read.

// Decision is the per-block profile choice.
type Decision struct {
	Profile   codec.Profile
	Transform codec.Transform
	Stride    int // >0 only when a transform is applied
}

// Classify inspects up to 64 KiB of a block and picks a profile.
func Classify(b []byte) Decision {
	const maxSample = 1 << 16
	n := len(b)
	if n == 0 {
		return Decision{Profile: codec.ProfileTelemetry}
	}
	if n > maxSample {
		// Sample the middle: a block boundary can land inside a header, a table
		// or a burst of markup, and either end is then unrepresentative.
		mid := n / 2
		return Classify(b[mid-maxSample/2 : mid+maxSample/2])
	}

	var high, printable, space, structur int
	for _, c := range b {
		switch {
		case c >= 0x80:
			high++
		case c == ' ':
			space++
			printable++
		case c == '\t' || c == '\n' || c == '\r':
			printable++
		case c >= 0x20 && c <= 0x7e:
			printable++
		}
		switch c {
		case '"', '{', '}', ':', ',', '[', ']', '<', '>', '=':
			structur++
		}
	}
	f := func(x, total int) float64 { return float64(x) / float64(total) }

	// Binary with a large non-ASCII share: a raster, or already-compressed
	// data. Only a raster gets a spatial predictor, and only when the predictor
	// measurably lowers the residual entropy.
	if f(high, n) > 0.25 {
		if tr, stride := codec.ChooseTransform(b); tr != codec.TransformNone {
			return Decision{Profile: codec.ProfileImage, Transform: tr, Stride: stride}
		}
		return Decision{Profile: codec.ProfileTelemetry}
	}

	// Mostly printable ASCII: prose (spaces, little markup) or structured data.
	if f(printable, n) >= 0.90 {
		// Word spacing is the discriminator between prose and minified
		// structured data; markup density is not, because Wikipedia is full of
		// XML tags and still wants the deeper context.
		if f(space, n) >= 0.08 {
			return Decision{Profile: codec.ProfileText}
		}
		if f(structur, n) >= 0.02 {
			return Decision{Profile: codec.ProfileTelemetry}
		}
		return Decision{Profile: codec.ProfileText}
	}
	return Decision{Profile: codec.ProfileTelemetry}
}
