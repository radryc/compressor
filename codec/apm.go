package codec

// Adaptive probability map (SSE).
//
// The mixer produces a probability from predictors that are pooled across a
// context; the map then corrects it *for that context*. It is a learned
// piecewise-linear remapping of the mixer's output: if the mixer is
// systematically over-confident whenever a long match is running, the map
// learns to pull those probabilities back.
//
// Two stages are chained, lpaq-style, and each blend is 3:1 against the input
// so a stage that has not learned anything yet cannot hurt.

const (
	apmPoints = 33
	// apmContexts is (bit-tree node, match state).
	apmContexts = mixerNodes * 4
	// apmRate controls how fast a cell moves; 6 was stable across all corpora.
	apmRate = 6
)

type apm struct {
	t [apmContexts][apmPoints]int32
}

func (a *apm) Reset() {
	for c := 0; c < apmContexts; c++ {
		for i := 0; i < apmPoints; i++ {
			a.t[c][i] = int32(i) * ProbOne / (apmPoints - 1)
		}
	}
}

func apmIndex(node, mstate int) int {
	return (node << 2) | (mstate & 3)
}

// split returns the interpolation cell and fraction for a probability.
func split(p uint32) (int, int32) {
	x := int32(p) * (apmPoints - 1) // 0 .. 32*65536
	i := int(x >> 16)
	if i >= apmPoints-1 {
		i = apmPoints - 2
	}
	return i, x & 0xffff
}

// Prob returns the mapped probability for p in context ctx.
func (a *apm) Prob(ctx int, p uint32) uint32 {
	i, frac := split(p)
	lo := a.t[ctx][i]
	hi := a.t[ctx][i+1]
	// int64: lo can be 65536 and the weight 65536, so the product overflows
	// int32.
	v := int32((int64(lo)*int64(ProbOne-frac) + int64(hi)*int64(frac)) >> ProbBits)
	if v < 0 {
		v = 0
	}
	if v > int32(ProbOne) {
		v = int32(ProbOne)
	}
	return clampProb(uint32(v))
}

// Update moves the two adjacent cells toward the observed bit.
func (a *apm) Update(ctx int, p uint32, bit uint32) {
	i, frac := split(p)
	target := int32(0)
	if bit == 1 {
		target = ProbOne
	}
	lo := &a.t[ctx][i]
	hi := &a.t[ctx][i+1]
	*lo += int32((int64(target-*lo) * int64(ProbOne-frac)) >> (16 + apmRate))
	*hi += int32((int64(target-*hi) * int64(frac)) >> (16 + apmRate))
}
