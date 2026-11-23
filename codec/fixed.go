package codec

// Deterministic fixed-point math.
//
// Every value in this file is produced with integer operations only. Go
// guarantees exact results for integer arithmetic on every architecture, so
// the encoder and the decoder derive bit-identical probabilities whether they
// run on amd64, arm64 or anything else. This is what makes a compressed
// archive portable: floating point (and in particular FMA contraction, which
// Go performs on arm64 but not on amd64) is never allowed into the hot path.
//
// Number formats used across the codec:
//
//	prob   uint32  Q16  probability in [1, 65535]
//	logit  int32   Q7   natural-log units, 128 per nat, range +-16 nats
//	act    int16   Q11  activations in [-1, 1] -> +-2048
//	weight int16   Q11  weights, |w| <= 16
//	acc    int32   Q22  sum of H weight*act products
//	bias   int32   Q16  layer biases, |b| <= 2048
//	inproj int32   Q16  pre-fused input projections

const (
	// ProbBits is the number of bits used for probabilities.
	ProbBits = 16
	// ProbOne is one in Q16.
	ProbOne = 1 << ProbBits

	// LogitScale is the number of Q7 units in one natural-log unit.
	LogitScale = 128
	// LogitLimit bounds the Q7 logit index range to +-16 nats.
	LogitLimit = 2048

	// ActOne is one in Q11.
	ActOne = 1 << 11
	// ActShift converts a Q22 accumulator into Q11.
	ActShift = 11
	// AccShift converts a Q22 accumulator into Q16 (pre-activation scale).
	AccShift = 6
)

const ln2Q32 = 2977044471 // round(ln2 * 2^32)

var (
	// squashTab maps a Q7 logit offset (-LogitLimit..LogitLimit-1) to Q16.
	squashTab [LogitLimit * 2]uint16
	// stretchTab maps a Q4 probability index (0..4095) to a Q7 logit.
	stretchTab [LogitLimit * 2]int16
)

func init() {
	for i := 0; i < len(squashTab); i++ {
		// index i corresponds to logit (i - LogitLimit) in Q7.
		squashTab[i] = uint16(sigmoidQ16(int32(i-LogitLimit) * (ProbOne / LogitScale)))
	}

	// stretchTab is the numeric inverse of squashTab, built with integer
	// comparisons only so it is reproducible everywhere.
	j := 0
	for i := 0; i < len(stretchTab); i++ {
		target := (i << 4) + 8
		for j < len(squashTab)-1 && int(squashTab[j+1]) <= target {
			j++
		}
		stretchTab[i] = int16(j - LogitLimit)
	}
}

// expNeg returns exp(-x) in Q30 for x >= 0 expressed in Q16.
func expNeg(x int64) int64 {
	if x <= 0 {
		return 1 << 30
	}
	k := (x << 16) / ln2Q32
	r := x - ((k*ln2Q32 + (1 << 15)) >> 16) // r in [0, ln2) in Q16, rounded
	one := int64(1) << 30
	term := one
	sum := one
	// exp(-r) = sum_n (-r)^n / n!
	for n := int64(1); n <= 9; n++ {
		term = (term * -r) / (n << 16)
		sum += term
	}
	if k >= 31 {
		return 0
	}
	if k == 0 {
		return sum
	}
	return (sum + 1<<uint(k-1)) >> uint(k)
}

// sigmoidQ16 returns 1/(1+exp(-x)) in Q16. x is a natural-log logit in Q16.
func sigmoidQ16(x int32) uint32 {
	neg := x < 0
	ax := int64(x)
	if neg {
		ax = -ax
	}
	if ax > 16*ProbOne {
		ax = 16 * ProbOne
	}
	e := expNeg(ax) // Q30, in (0, 1<<30]
	den := (int64(1) << 30) + e
	// (1<<46)/den is Q16 because den is Q30.
	p := uint32((int64(1) << 46) / den)
	if neg {
		p = ProbOne - p
	}
	return clampProb(p)
}

// squash converts a Q7 logit to a Q16 probability.
func squash(logit int32) uint32 {
	if logit < -LogitLimit {
		logit = -LogitLimit
	} else if logit > LogitLimit-1 {
		logit = LogitLimit - 1
	}
	return uint32(squashTab[logit+LogitLimit])
}

// stretch converts a Q16 probability to a Q7 logit.
func stretch(p uint32) int32 {
	idx := int(p >> 4)
	if idx >= len(stretchTab) {
		idx = len(stretchTab) - 1
	}
	return int32(stretchTab[idx])
}

// tanhQ11 returns tanh(x) in Q11 where x is a Q16 natural-log value.
//
// This goes through the same squash LUT the coder uses, so the gate
// non-linearity costs one table lookup instead of an exp() evaluation and is
// bit-identical on every platform. The LUT resolves the exponent at 1/128 nat,
// which is far finer than a Q11 activation can represent.
func tanhQ11(x int32) int16 {
	// tanh(x) = 2*sigmoid(2x) - 1
	p := squash((x * 2) >> 9) // Q16
	v := (int32(p) - ProbOne/2) >> 4
	if v > ActOne {
		v = ActOne
	} else if v < -ActOne {
		v = -ActOne
	}
	return int16(v)
}

// sigmoidQ11 returns sigmoid(x) in Q11.
func sigmoidQ11(x int32) int16 {
	return int16(squash(x>>9) >> 5)
}

func clampProb(p uint32) uint32 {
	if p < 1 {
		return 1
	}
	if p > ProbOne-1 {
		return ProbOne - 1
	}
	return p
}

func clamp16(v int32, lo, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
