package codec

// MatchModel is an LZ-style predictor. It indexes the most recent occurrence of
// the last 4, 8 and 16 bytes and predicts that the following byte repeats.
//
// Three context lengths matter because reliability grows with context: a 4-byte
// hit is a hint, a 16-byte hit is nearly a certainty. The mixer receives one
// confidence, taken from the longest verified context, and the confidence keeps
// rising while a run continues to be predicted correctly.
//
// Everything is integer, so encoder and decoder reconstruct identical
// predictions from identical history.

const (
	matchLens = 3 // 4, 8 and 16 byte contexts
	// matchMaxBack bounds the backward verification loop so a long repetitive
	// region cannot make candidate selection quadratic.
	matchMaxBack = 64
)

const hashBase = 0x100000001B3 // FNV-1a 64-bit prime

var hashPow [17]uint64

func init() {
	hashPow[0] = 1
	for i := 1; i < len(hashPow); i++ {
		hashPow[i] = hashPow[i-1] * hashBase
	}
}

type matchTable struct {
	head []int32 // 0 = empty, otherwise position+1
	mask uint32
}

func newMatchTable(bits int) matchTable {
	n := 1 << uint(bits)
	return matchTable{
		head: make([]int32, n),
		mask: uint32(n) - 1,
	}
}

func (t *matchTable) reset() { clear(t.head) }

type matchCand struct {
	pos  int32 // position of the predicted byte
	blen int32 // verified backward context length
}

type MatchModel struct {
	hist []byte
	tabs [matchLens]matchTable
	lens [matchLens]int
	roll [matchLens]uint64

	maxHist  int
	matchPos int32
	matchLen int32
	pred     int32 // predicted next byte, -1 when inactive
	conf     int32 // Q7 logit confidence in pred
}

// NewMatchModel creates a match model. maxHist bounds retained history (0 =
// unbounded); tableBits sizes each hash table.
func NewMatchModel(maxHist, tableBits int) *MatchModel {
	if tableBits < 8 {
		tableBits = 8
	}
	if tableBits > 20 {
		tableBits = 20
	}
	m := &MatchModel{
		maxHist:  maxHist,
		lens:     [matchLens]int{4, 8, 16},
		matchPos: -1,
		pred:     -1,
	}
	for i := 0; i < matchLens; i++ {
		m.tabs[i] = newMatchTable(tableBits)
	}
	m.hist = make([]byte, 0, 1<<16)
	return m
}

func (m *MatchModel) Reset() {
	m.hist = m.hist[:0]
	for i := range m.tabs {
		m.tabs[i].reset()
		m.roll[i] = 0
	}
	m.matchPos = -1
	m.matchLen = 0
	m.pred = -1
	m.conf = 0
}

// roll advances the polynomial hash for a window of length L ending at n.
func (m *MatchModel) rollWindow(i, L, n int, b byte) {
	if n+1 < L {
		return
	}
	if n+1 == L {
		var v uint64
		for k := 0; k < L; k++ {
			v = v*hashBase + uint64(m.hist[n-L+1+k])
		}
		m.roll[i] = v
		return
	}
	// H(n) = H(n-1)*B + b - hist[n-L]*B^L
	m.roll[i] = m.roll[i]*hashBase + uint64(b) - uint64(m.hist[n-L])*hashPow[L]
}

// verify checks one stored position against the current context and returns
// the prediction with its verified backward length.
func (m *MatchModel) verify(p0, L, n int) matchCand {
	if p0 < L-1 || p0 >= n {
		return matchCand{}
	}
	// Reject hash collisions by comparing the whole context.
	for k := 0; k < L; k++ {
		if m.hist[p0-k] != m.hist[n-k] {
			return matchCand{}
		}
	}
	// Extend backwards to measure how strong the evidence really is.
	bl := L
	for bl < matchMaxBack && p0-bl >= 0 && n-bl >= 0 && m.hist[p0-bl] == m.hist[n-bl] {
		bl++
	}
	return matchCand{pos: int32(p0 + 1), blen: int32(bl)}
}

// probe installs this position and returns the verified candidate.
//
// A second candidate per slot was measured and did not pay: the most recent
// occurrence is almost always the best predictor, so it cost 2x table memory for
// nothing. Single slot, most recent wins.
func (m *MatchModel) probe(i, L, n int) matchCand {
	t := &m.tabs[i]
	idx := uint32(m.roll[i]>>32) & t.mask
	prev := t.head[idx]
	t.head[idx] = int32(n + 1)
	if prev == 0 {
		return matchCand{}
	}
	return m.verify(int(prev-1), L, n)
}

// Update appends the coded byte and prepares the prediction for the next one.
func (m *MatchModel) Update(b byte) {
	// Continue or drop the active run.
	if m.matchPos >= 0 && int(m.matchPos) < len(m.hist) {
		if m.hist[m.matchPos] == b {
			m.matchPos++
			m.matchLen++
		} else {
			m.matchPos = -1
			m.matchLen = 0
		}
	}

	n := len(m.hist)
	m.hist = append(m.hist, b)

	// Refresh the index and collect verified candidates for the new context.
	var best matchCand
	for i := 0; i < matchLens; i++ {
		L := m.lens[i]
		m.rollWindow(i, L, n, b)
		if n+1 < L {
			continue
		}
		if c := m.probe(i, L, n); c.blen >= best.blen {
			// Ties prefer the longer context, which is the more reliable one.
			best = c
		}
	}

	// Start a fresh run only when none is active (a running match has the
	// strongest evidence of all: every additional byte confirms it).
	if best.blen > 0 && (m.matchPos < 0 || int(m.matchPos) >= len(m.hist)) {
		m.matchPos = best.pos
		m.matchLen = best.blen
	}

	if m.matchPos >= 0 && int(m.matchPos) < len(m.hist) {
		m.pred = int32(m.hist[m.matchPos])
	} else {
		m.pred = -1
		m.matchLen = 0
	}
	m.conf = matchConf(m.matchLen)

	if m.maxHist > 0 && len(m.hist) > m.maxHist {
		m.trim()
	}
}

// matchConf converts a verified match length into a Q7 logit confidence.
//
// p = L^2/(L^2+4) was measured against p = L^2/(L^2+k) for k=1..64: the result
// is flat within +-0.5%, because the mixer's per-node weight rescales the input.
// All that matters is that it is monotonic.
func matchConf(ml int32) int32 {
	if ml <= 0 {
		return 0
	}
	if ml > 4096 {
		ml = 4096
	}
	num := int64(ml) * int64(ml)
	p := uint32(num * ProbOne / (num + 4))
	return stretch(clampProb(p))
}

// State buckets the match model's reliability for the mixer address:
// 0 = inactive, 1 = short match, 2 = medium, 3 = long.
func (m *MatchModel) State() int {
	if m.pred < 0 {
		return 0
	}
	switch {
	case m.matchLen >= 24:
		return 3
	case m.matchLen >= 8:
		return 2
	default:
		return 1
	}
}

// BitStretch returns the Q7 logit this model contributes to the given binary
// decision inside the byte being coded, or 0 when it has no information.
//
// node is the bit-tree node (its low bitpos bits are the prefix coded so far).
func (m *MatchModel) BitStretch(node, bitpos int) int32 {
	if m.pred < 0 || m.conf == 0 {
		return 0
	}
	if bitpos > 0 {
		mask := (1 << uint(bitpos)) - 1
		if node&mask != (int(m.pred)>>uint(8-bitpos))&mask {
			return 0
		}
	}
	if int(m.pred)>>uint(7-bitpos)&1 == 1 {
		return m.conf
	}
	return -m.conf
}

// trim evicts the older half of the history and rebases table positions so
// long-lived sessions keep bounded memory.
func (m *MatchModel) trim() {
	keep := m.maxHist / 2
	drop := len(m.hist) - keep
	if drop <= 0 {
		return
	}
	copy(m.hist, m.hist[drop:])
	m.hist = m.hist[:keep]
	if m.matchPos >= 0 {
		m.matchPos -= int32(drop)
		if m.matchPos < 0 {
			m.matchPos = -1
			m.matchLen = 0
			m.pred = -1
		}
	}
	for i := range m.tabs {
		t := &m.tabs[i]
		for j, v := range t.head {
			if v > int32(drop) {
				t.head[j] = v - int32(drop)
			} else {
				t.head[j] = 0
			}
		}
	}
}

// History exposes the retained history (used by keyframe serialisation).
func (m *MatchModel) History() []byte { return m.hist }

// Rebuild replays a history window from scratch. Keyframe resumption uses this
// so both sides derive an identical match state from the same bytes.
func (m *MatchModel) Rebuild(hist []byte) {
	m.Reset()
	for _, b := range hist {
		m.Update(b)
	}
}
