package codec

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/radryc/compressor/model"
)

// Blend selects which predictors feed the mixer. Disabling one is what makes
// the ablation study in `nzip ablate` possible.
type Blend struct {
	GRU   bool
	Match bool
	Table bool
	Run   bool
}

// Full enables every predictor.
var Full = Blend{GRU: true, Match: true, Table: true, Run: true}

func (b Blend) flags() byte {
	var f byte
	if b.GRU {
		f |= 1
	}
	if b.Match {
		f |= 2
	}
	if b.Table {
		f |= 4
	}
	if b.Run {
		f |= 8
	}
	return f
}

func blendFromFlags(f byte) Blend {
	return Blend{GRU: f&1 != 0, Match: f&2 != 0, Table: f&4 != 0, Run: f&8 != 0}
}

// Profile selects the codec configuration for a block's content.
//
// The distinction exists because one configuration cannot serve every kind of
// data: natural language wants a deeper context, structured telemetry is worse
// with it, and images want a spatial transform rather than a different model.
type Profile uint8

const (
	// ProfileTelemetry is the default: order-2 context, run model, match model.
	ProfileTelemetry Profile = iota
	// ProfileText adds an order-3 context, which pays on natural language.
	ProfileText
	// ProfileImage applies a row-delta transform before coding.
	ProfileImage
)

// Valid reports whether p is a known profile.
func (p Profile) Valid() bool { return p <= ProfileImage }

// String renders a profile for logs.
func (p Profile) String() string {
	switch p {
	case ProfileText:
		return "text"
	case ProfileImage:
		return "image"
	default:
		return "telemetry"
	}
}

// Config tunes a BlockEngine.
type Config struct {
	Blend     Blend
	TableBits int // match-model hash table size, 8..20
	MaxHist   int // match history cap, 0 = unbounded (block mode)
	Profile   Profile
	StatBits  int // order-2 table, 0 = default (18)
	DeepBits  int // order-3/4/word tables, 0 = none
}

// DefaultConfig matches the production defaults.
func DefaultConfig() Config {
	return Config{Blend: Full, TableBits: 16}
}

// ConfigForProfile sizes the match table for a block length and selects the
// context depth for a profile.
func ConfigForProfile(p Profile, blockSize int) Config {
	cfg := ConfigForBlock(blockSize)
	cfg.Profile = p
	cfg.StatBits = statBitsDefault
	if p == ProfileText {
		cfg.DeepBits = statBitsText
	}
	return cfg
}

// EngineBytes estimates the per-engine working set, used to clamp the worker
// count against a memory budget. Table sizes themselves are never budget-driven:
// the decoder must derive them from the profile alone.
func (c Config) EngineBytes() int {
	const entry = 6 // bitStat: int32 p + uint16 n
	statBits := c.StatBits
	if statBits == 0 {
		statBits = statBitsDefault
	}
	n := (1 << uint(statBits)) * entry
	if c.DeepBits > 0 {
		n += 3 * (1 << uint(c.DeepBits)) * entry
	}
	n += 256 * 16 * entry // run-length table
	n += 4096 * mixerInputs * 2
	n += 1024 * 33 * 4 // APM
	return n
}

// ConfigForBlock sizes the match table for a block length.
func ConfigForBlock(blockSize int) Config {
	cfg := DefaultConfig()
	bits := 12
	for blockSize > 1<<uint(bits) && bits < 16 {
		bits++
	}
	cfg.TableBits = bits
	return cfg
}

// BlockEngine codes one block (or one session frame) at a time. It is not safe
// for concurrent use: allocate one per worker.
type BlockEngine struct {
	gru   GRUState
	match *MatchModel
	mixer Mixer
	stats statTable
	// Allocated only for the text profile.
	o3, o4, word *hashTable
	wordHash     uint32
	prev4        byte

	run  runStat
	apm  apm
	prof Profile

	// runLen is the length of the current same-class run, saturated, and
	// runCls is the class of that run.
	runLen uint8
	runCls uint8
	// prev2 is the byte before prev, used by the order-2 context.
	prev2 byte
	prev3 byte

	enc ArithmeticEncoder
	dec ArithmeticDecoder

	blend Blend
	s     [mixerInputs]int32
	prev  int
}

func NewBlockEngine(m *model.Model, cfg Config) *BlockEngine {
	statBits := cfg.StatBits
	if statBits == 0 {
		statBits = statBitsDefault
	}
	e := &BlockEngine{
		match: NewMatchModel(cfg.MaxHist, cfg.TableBits),
		stats: *newStatTable(statBits),
		blend: cfg.Blend,
		prof:  cfg.Profile,
	}
	if cfg.DeepBits > 0 {
		e.o3 = newHashTable(cfg.DeepBits)
		e.o4 = newHashTable(cfg.DeepBits)
		e.word = newHashTable(cfg.DeepBits)
	}
	e.Reset(m)
	return e
}

// Reset returns the engine to the start-of-stream state for model m.
func (e *BlockEngine) Reset(m *model.Model) {
	e.gru.Reset(m)
	e.match.Reset()
	e.mixer.Reset()
	e.stats.Reset()
	if e.o3 != nil {
		e.o3.Reset()
		e.o4.Reset()
		e.word.Reset()

	}
	e.wordHash = 0
	e.prev4 = 0
	e.run.Reset()
	e.apm.Reset()
	e.runLen = 0
	e.runCls = 0
	e.prev = 0
	e.prev2 = 0
	e.prev3 = 0
}

// ==========================================
// Byte loop
// ==========================================

// byteLoop runs the model over one byte. getBit returns the coded bit given
// the predicted probability for that bit position.
func (e *BlockEngine) byteLoop(prevByte int, getBit func(bitpos int, p uint32) uint32) byte {
	if e.blend.GRU {
		e.gru.Step(prevByte)
	}
	node := 1
	var out byte
	for bitpos := 0; bitpos < 8; bitpos++ {
		// The deeper context must include the bit-tree node, so it is indexed
		// per bit, not per byte.
		var i3, i4, iw int
		if e.o3 != nil {
			i3 = idx3(node, e.o3.shift, byte(prevByte), e.prev2, e.prev3)
			i4 = idx4(node, e.o4.shift, byte(prevByte), e.prev2, e.prev3, e.prev4)
			iw = idxWord(node, e.word.shift, e.wordHash)
		}
		e.s[0] = 0
		e.s[1] = 0
		e.s[2] = 0
		e.s[3] = 0
		e.s[4] = 0
		if e.blend.GRU {
			e.s[0] = e.gru.bitLogit[bitpos]
		}
		if e.blend.Match {
			e.s[1] = e.match.BitStretch(node, bitpos)
		}
		if e.blend.Table {
			e.s[2] = stretch(e.stats.Prob(node, byte(prevByte), e.prev2))
			if e.o3 != nil {
				e.s[3] = stretch(e.o3.Prob(i3))
				e.s[4] = stretch(e.o4.Prob(i4))
				e.s[5] = stretch(e.word.Prob(iw))
			}
		}
		if e.blend.Run {
			e.s[6] = stretch(e.run.Prob(node, int(e.runCls), int(e.runLen)))
		}
		e.s[7] = 128

		ctx := mixCtx(node, int(e.runCls)*runLens+int(e.runLen), e.match.State())
		p0 := e.mixer.Mix(ctx, &e.s)
		// SSE: correct the mixer's output for this (node, match state), then
		// blend 3:1 so an untrained map is harmless.
		actx := apmIndex(node, e.match.State())
		p := (e.apm.Prob(actx, p0)*3 + p0) >> 2
		bit := getBit(bitpos, p)
		e.apm.Update(actx, p0, bit)
		e.mixer.Update(ctx, &e.s, p0, bit)
		if e.blend.Table {
			e.stats.Update(node, byte(prevByte), e.prev2, bit)
			if e.o3 != nil {
				e.o3.Update(i3, bit)
				e.o4.Update(i4, bit)
				e.word.Update(iw, bit)
			}
		}
		if e.blend.Run {
			e.run.Update(node, int(e.runCls), int(e.runLen), bit)
		}
		out = out<<1 | byte(bit)
		node = node<<1 | int(bit)
	}
	e.match.Update(out)
	e.prev4 = e.prev3
	e.prev3 = e.prev2
	e.prev2 = byte(prevByte)
	if isWordByte(out) {
		e.wordHash = e.wordHash*0x01000193 ^ uint32(out)
	} else {
		e.wordHash = 0
	}
	if c := runClass(out); c != 0 && int(e.runCls) == c {
		if e.runLen < runLens-1 {
			e.runLen++
		}
	} else {
		e.runCls = uint8(c)
		e.runLen = 0
	}
	return out
}

// CompressFrame codes src into dst using a fresh arithmetic coder but the
// engine's current model state.
func (e *BlockEngine) CompressFrame(src, dst []byte) []byte {
	e.enc.Reset(dst)
	for _, b := range src {
		bb := b
		e.byteLoop(e.prev, func(bitpos int, p uint32) uint32 {
			bit := uint32(bb>>uint(7-bitpos)) & 1
			e.enc.EncodeBit(p, bit)
			return bit
		})
		e.prev = int(bb)
	}
	return e.enc.Finish()
}

// DecompressFrame decodes n bytes from src, growing dst as needed.
func (e *BlockEngine) DecompressFrame(src []byte, n int, dst []byte) []byte {
	if cap(dst) < n {
		dst = make([]byte, 0, n)
	}
	dst = dst[:0]
	e.dec.Reset(src)
	for i := 0; i < n; i++ {
		b := e.byteLoop(e.prev, func(bitpos int, p uint32) uint32 {
			return e.dec.DecodeBit(p)
		})
		dst = append(dst, b)
		e.prev = int(b)
	}
	return dst
}

// Prime advances the model over data exactly as if it had been coded, without
// producing output. Dictionaries and passthrough frames both use this so the
// decoder can reach an identical state.
func (e *BlockEngine) Prime(data []byte) {
	for _, b := range data {
		bb := b
		e.byteLoop(e.prev, func(bitpos int, p uint32) uint32 {
			return uint32(bb>>uint(7-bitpos)) & 1
		})
		e.prev = int(bb)
	}
}

// ResetWithDict resets the engine and primes it with a dictionary. Priming
// feeds the bytes through the model without producing output, so both sides
// reach the same state.
func (e *BlockEngine) ResetWithDict(m *model.Model, dict []byte) {
	e.Reset(m)
	if len(dict) > 0 {
		e.Prime(dict)
	}
}

// CompressBlockWithDict compresses one independent block, optionally primed
// with a dictionary. If the coded output is not smaller than the input it
// returns a copy of src and passthrough=true.
func (e *BlockEngine) CompressBlockWithDict(m *model.Model, dict, src, dst []byte) ([]byte, bool) {
	e.ResetWithDict(m, dict)
	out := e.CompressFrame(src, dst)
	if len(out) >= len(src) {
		return append(dst[:0:len(dst)], src...), true
	}
	return out, false
}

// DecompressBlockWithDict decodes one independent block.
func (e *BlockEngine) DecompressBlockWithDict(m *model.Model, dict, src []byte, origLen int, dst []byte) []byte {
	e.ResetWithDict(m, dict)
	return e.DecompressFrame(src, origLen, dst)
}

// CompressBlock compresses one independent block with no dictionary.
func (e *BlockEngine) CompressBlock(m *model.Model, src, dst []byte) ([]byte, bool) {
	return e.CompressBlockWithDict(m, nil, src, dst)
}

// DecompressBlock decodes one independent block with no dictionary.
func (e *BlockEngine) DecompressBlock(m *model.Model, src []byte, origLen int, dst []byte) []byte {
	return e.DecompressBlockWithDict(m, nil, src, origLen, dst)
}

// ==========================================
// Keyframe state serialisation
// ==========================================

const snapshotVersion = 2

// SnapshotHeader is the fixed part of a serialised codec state.
const SnapshotHeader = 16

// SnapshotSize returns the encoded size of a snapshot carrying histLen bytes
// of history.
func SnapshotSize(hiddenDim, histLen int) int {
	return SnapshotHeader + 2*hiddenDim + 4 + histLen
}

// Snapshot serialises the resumable part of the state: the hidden vector, the
// previous byte, and the history window the decoder replays to rebuild the
// match model.
//
// The adaptive tables (bit statistics, run statistics, mixer weights) are
// deliberately *not* serialised. They are large and they reconverge within a few
// kilobytes, so a keyframe that reset them is far cheaper than one that carries
// them: 65 KiB instead of 520 KiB. The encoder therefore reloads its own
// snapshot after writing it, which makes encoder and decoder converge on exactly
// the same state by construction.
func (e *BlockEngine) Snapshot(dst []byte) []byte {
	h := e.gru.m.HiddenDim
	hist := e.match.History()
	size := SnapshotSize(h, len(hist))
	if cap(dst) < size {
		dst = make([]byte, size)
	}
	dst = dst[:size]
	binary.LittleEndian.PutUint16(dst[0:2], snapshotVersion)
	binary.LittleEndian.PutUint16(dst[2:4], uint16(h))
	dst[4] = e.blend.flags()
	dst[5] = byte(e.prof)
	binary.LittleEndian.PutUint32(dst[8:12], uint32(int32(e.prev)))
	binary.LittleEndian.PutUint32(dst[12:16], uint32(len(hist)))
	off := SnapshotHeader
	for i := 0; i < h; i++ {
		binary.LittleEndian.PutUint16(dst[off:], uint16(e.gru.h[i]))
		off += 2
	}
	copy(dst[off:], hist)
	// Adopt the snapshot ourselves: the serialised form omits the adaptive
	// tables, so the writer must rewind to the same state the reader will
	// reconstruct. Doing it here makes that impossible to forget.
	_ = e.Restore(e.gru.m, dst)
	return dst
}

// KeyframeHistory is how much raw history a keyframe carries.
const KeyframeHistory = 64 << 10

// TrimHistory rewinds the match model to the last n bytes of history.
func (e *BlockEngine) TrimHistory(n int) {
	h := e.match.History()
	if len(h) <= n {
		return
	}
	e.match.Rebuild(h[len(h)-n:])
}

// Restore loads a snapshot produced by Snapshot. The adaptive tables are reset,
// matching what Snapshot leaves out.
func (e *BlockEngine) Restore(m *model.Model, src []byte) error {
	if len(src) < SnapshotHeader {
		return errors.New("codec: snapshot too short")
	}
	if binary.LittleEndian.Uint16(src[0:2]) != snapshotVersion {
		return fmt.Errorf("codec: unsupported snapshot version %d", binary.LittleEndian.Uint16(src[0:2]))
	}
	h := int(binary.LittleEndian.Uint16(src[2:4]))
	if h != m.HiddenDim {
		return fmt.Errorf("codec: snapshot hidden_dim=%d does not match model %d", h, m.HiddenDim)
	}
	if blend := blendFromFlags(src[4]); blend != e.blend {
		return fmt.Errorf("codec: snapshot blend %v does not match engine %v", blend, e.blend)
	}
	if p := Profile(src[5]); p != e.prof {
		return fmt.Errorf("codec: snapshot profile %v does not match engine %v", p, e.prof)
	}
	e.Reset(m)
	histLen := int(binary.LittleEndian.Uint32(src[12:16]))
	if need := SnapshotSize(h, histLen); len(src) < need {
		return fmt.Errorf("codec: snapshot truncated (%d < %d)", len(src), need)
	}
	e.prev = int(int32(binary.LittleEndian.Uint32(src[8:12])))
	off := SnapshotHeader
	for i := 0; i < h; i++ {
		e.gru.h[i] = int16(binary.LittleEndian.Uint16(src[off:]))
		off += 2
	}
	e.match.Rebuild(src[off : off+histLen])
	return nil
}

// History returns the match model's retained history.
func (e *BlockEngine) History() []byte { return e.match.History() }

// Profile reports the engine's profile.
func isWordByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b >= 0x80
}

func (e *BlockEngine) Profile() Profile { return e.prof }

// Model returns the engine's model.
func (e *BlockEngine) Model() *model.Model { return e.gru.m }
