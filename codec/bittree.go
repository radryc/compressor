package codec

// Bit-tree context modelling.
//
// The 256-byte alphabet is coded as eight binary decisions. Each internal node
// of the 8-level binary tree (nodes 1..255) carries:
//
//   - adaptive mixer weights that combine three predictors, and
//   - an adaptive bit statistic indexed by (node, low nibble of previous byte).
//
// Both are integer-only and are updated identically by encoder and decoder, so
// they are part of the deterministic model state.

const (
	mixerNodes  = 256 // bit-tree nodes 1..255 (index 0 unused)
	mixerInputs = 8   // GRU, match, order-2/3/4, word, run-length, bias

	// Mixer weights are Q12: 4096 == 1.0. Q12 leaves enough low bits that a
	// single integer gradient step does not truncate to zero, which is what
	// makes convergence stable without a floating-point learning rate.
	mixerQ12   = 1 << 12
	mixerMax   = 4 * mixerQ12
	mixerShift = 19 // (err Q16 * s Q7) >> 19 -> Q12 step
)

// Mixer is a logistic mixer in the log-odds domain.
//
// Weights are indexed by (bit-tree node, run-length bucket). A single weight per
// node cannot serve contexts as different as "inside a 32-character hex id"
// (where the run predictor is near-unbeatable) and "the run just ended" (where
// it is dangerous), so the run bucket is part of the weight address.
type Mixer struct {
	w [mixerContexts * mixerInputs]int16
}

const (
	mixerRunBits  = 5 // 4 classes x 8 run lengths
	mixerContexts = mixerNodes << mixerRunBits << 2
)

// mixCtx folds the run-length bucket and the match state into the mixer
// address. Without the match bucket a single weight has to serve both "no
// match at all" and "a 200-byte match is running", which are opposite regimes.
func mixCtx(node, run, mstate int) int {
	return ((node<<4)|(run&0x0f))<<2 | (mstate & 3)
}

func (mx *Mixer) Reset() {
	for n := 0; n < mixerContexts; n++ {
		base := n * mixerInputs
		mx.w[base+0] = mixerQ12 / 2 // GRU
		mx.w[base+1] = mixerQ12 / 2 // match
		mx.w[base+2] = mixerQ12 / 2 // order-2
		mx.w[base+3] = mixerQ12 / 2 // order-3 (text profile only)
		mx.w[base+4] = 0            // order-4 (must earn its weight)
		mx.w[base+5] = 0            // word
		mx.w[base+6] = mixerQ12 / 2 // run-length
		mx.w[base+7] = 0            // bias
	}
}

// Mix combines the stretched inputs into a probability. s values are Q7
// logits; the result is Q16.
func (mx *Mixer) Mix(ctx int, s *[mixerInputs]int32) uint32 {
	base := ctx * mixerInputs
	var dot int32
	for i := 0; i < mixerInputs; i++ {
		dot += int32(mx.w[base+i]) * s[i]
	}
	return squash(clampLogit(dot >> 12))
}

// Update applies one logistic-gradient step for the observed bit.
//
// The gradient of the log-likelihood with respect to a weight is
// (bit - p) * s, so the weight moves by +lr*(bit-p)*s. err carries that sign;
// getting it backwards makes the mixer learn to invert its inputs.
func (mx *Mixer) Update(ctx int, s *[mixerInputs]int32, p uint32, bit uint32) {
	base := ctx * mixerInputs
	var err int32
	if bit == 1 {
		err = ProbOne - int32(p)
	} else {
		err = -int32(p)
	}
	for i := 0; i < mixerInputs; i++ {
		d := (err * s[i]) >> mixerShift
		w := int32(mx.w[base+i]) + d
		mx.w[base+i] = int16(clamp16(w, -mixerMax, mixerMax))
	}
}

// bitStat is an adaptive probability with a count-based learning rate.
type bitStat struct {
	p int32  // Q16 in [0, ProbOne]
	n uint16 // update count
}

const (
	statLimit = 60
	statInit  = ProbOne / 2
)

// statRate[n] == 65536/(n+2) in Q16, so the adaptive update needs a multiply
// instead of an integer division (the division was ~25 cycles per bit).
var statRate = func() (r [statLimit + 2]int32) {
	for n := range r {
		r[n] = int32((1 << 16) / (n + 2))
	}
	return
}()

const (
	// Table sizes are a *function of the profile*, never of the machine, so
	// encoder and decoder always agree even when they run with different
	// worker counts.
	statBitsDefault = 18
	statBitsText    = 20 // deep contexts only: collision pressure, not capacity
)

type statTable struct {
	st    []bitStat
	shift uint
}

func newStatTable(bits int) *statTable {
	t := &statTable{st: make([]bitStat, 1<<uint(bits)), shift: uint(32 - bits)}
	t.Reset()
	return t
}

func (t *statTable) Reset() {
	for i := range t.st {
		t.st[i] = bitStat{p: statInit}
	}
}

// statIndex hashes the bit-tree node with the previous two bytes (order-2).
// An order-1 context (just the previous byte) was measurably worse: on JSON the
// extra byte is what distinguishes "inside a key name" from "inside a value".
func (t *statTable) index(node int, prevByte, prev2 byte) int {
	return int(ctxHash(node, uint32(prevByte)<<8|uint32(prev2)) >> t.shift)
}

func (t *statTable) Prob(node int, prevByte, prev2 byte) uint32 {
	return clampProb(uint32(t.st[t.index(node, prevByte, prev2)].p))
}

// runStat models "how long have we been inside a run of this character class".
//
// It is the antidote to context dilution: inside a 32-character hex identifier
// the surrounding byte contexts are all unique and useless, but the run length
// is 1..32 and a model indexed by it learns both the uniform nibble distribution
// and where the run has to stop.
//
// The class is part of the address because identifiers (0-9a-f), decimal
// timestamps (0-9) and words (a-z) share the same run counter but have very
// different nibble distributions; mixing them was costing bits on both.
type runStat struct {
	st [runClasses * runLens * mixerNodes]bitStat
}

const (
	runClasses = 4 // other, decimal, hex letter, alpha
	runLens    = 8 // saturated run length
)

// runClass classifies a byte for the run model.
func runClass(b byte) int {
	switch {
	case b >= '0' && b <= '9':
		return 1
	case b >= 'a' && b <= 'f', b >= 'A' && b <= 'F':
		return 2
	case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z':
		return 3
	}
	return 0
}

func runIndex(node, class, run int) int {
	if run > runLens-1 {
		run = runLens - 1
	}
	return ((node*runClasses+class)*runLens + run)
}

func (t *runStat) Reset() {
	for i := range t.st {
		t.st[i] = bitStat{p: statInit}
	}
}

func (t *runStat) Prob(node, class, run int) uint32 {
	return clampProb(uint32(t.st[runIndex(node, class, run)].p))
}

func (t *runStat) Update(node, class, run int, bit uint32) {
	s := &t.st[runIndex(node, class, run)]
	target := int32(0)
	if bit == 1 {
		target = ProbOne
	}
	s.p += ((target - s.p) * statRate[s.n]) >> 16
	if s.n < statLimit {
		s.n++
	}
}

// hashTable is a hashed adaptive context. Text profiles allocate three of
// them (order-3, order-4 and a word model); every other profile allocates none,
// because on structured JSON the deeper contexts only fragment the statistics.
type hashTable struct {
	st    []bitStat
	shift uint
}

func newHashTable(bits int) *hashTable {
	t := &hashTable{st: make([]bitStat, 1<<uint(bits)), shift: uint(32 - bits)}
	t.Reset()
	return t
}

func (t *hashTable) Reset() {
	for i := range t.st {
		t.st[i] = bitStat{p: statInit}
	}
}

func ctxHash(node int, x uint32) uint32 {
	return x*2654435761 ^ uint32(node)*2246822519
}

func idx3(node int, shift uint, b1, b2, b3 byte) int {
	return int(ctxHash(node, uint32(b1)<<16|uint32(b2)<<8|uint32(b3)) >> shift)
}

func idx4(node int, shift uint, b1, b2, b3, b4 byte) int {
	return int(ctxHash(node, uint32(b1)<<24|uint32(b2)<<16|uint32(b3)<<8|uint32(b4)) >> shift)
}

func idxWord(node int, shift uint, h uint32) int {
	return int(ctxHash(node, h*0x9e3779b1) >> shift)
}

func (t *hashTable) Prob(idx int) uint32 {
	return clampProb(uint32(t.st[idx].p))
}

// ProbConf returns the probability and a Q16 confidence that grows with the
// number of observations behind this context.
//
// A high-order context is often seen once or twice; without this signal the
// mixer has no way to tell it apart from a context with thousands of samples,
// so it either trusts both or neither. Attenuating the input by confidence
// makes a new context nearly silent and lets it earn influence as it fills in.
func (t *hashTable) ProbConf(idx int) (uint32, int32) {
	e := &t.st[idx]
	p := clampProb(uint32(e.p))
	n := int32(e.n)
	if n > hashWarmup {
		n = hashWarmup
	}
	return p, n * (ProbOne / hashWarmup)
}

// hashWarmup is how many observations a context needs before it is fully
// trusted.
const hashWarmup = 8

func (t *hashTable) Update(idx int, bit uint32) {
	s := &t.st[idx]
	target := int32(0)
	if bit == 1 {
		target = ProbOne
	}
	s.p += ((target - s.p) * statRate[s.n]) >> 16
	if s.n < statLimit {
		s.n++
	}
}

func (t *statTable) Update(node int, prevByte, prev2 byte, bit uint32) {
	s := &t.st[t.index(node, prevByte, prev2)]
	target := int32(0)
	if bit == 1 {
		target = ProbOne
	}
	d := ((target - s.p) * statRate[s.n]) >> 16
	s.p += d
	if s.n < statLimit {
		s.n++
	}
}
