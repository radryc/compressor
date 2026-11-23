package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"sync"
	"sync/atomic"
)

// Magic identifies an NGRM model artifact.
var Magic = [4]byte{'N', 'G', 'R', 'M'}

// Version is the on-disk model format version. Version 2 replaces the 256-way
// output softmax with eight binary bit heads and stores every weight in a
// fixed-point format that the codec can consume deterministically.
const Version = 2

// MaxHidden bounds the hidden dimension. The quantised GRU accumulates in
// int32 with H * |w|max * |h|max = H * 8192 * 2048, which stays below 2^31 for
// H <= 96. Hidden dims above that would wrap (deterministically, but with a
// large ratio loss), so they are rejected at load time.
const MaxHidden = 96

// maxWeightQ11 is the +-4.0 clamp applied to Q11 weights. Together with
// MaxHidden it keeps every dot product inside int32.
const maxWeightQ11 = 8192

// NumFloats returns the number of float32 weights an artifact with hidden
// dimension h stores.
func NumFloats(h int) int {
	return 256*3*h + 3*h*h + 3*h + 8*h + 8
}

// Model holds read-only quantised weights. It is safe for concurrent use.
//
// Fixed-point layout (see codec/fixed.go):
//
//	InProj  int32 Q16  [256*3H]  pre-fused input projection per byte
//	WHH     int16 Q11  [3H*H]    recurrent weights, gates ordered r,z,n
//	BHH     int32 Q16  [3H]
//	WBit    int16 Q11  [8*H]     bit-head weights, one row per bit position
//	BBit    int32 Q7   [8]       bit-head biases
type Model struct {
	InProj []int32
	WHH    []int16
	BHH    []int32
	WBit   []int16
	BBit   []int32

	HiddenDim int
	ID        uint32
	Hash      uint32 // crc32c of the artifact bytes
}

// Weights is a floating-point artifact prior to quantisation. It is what
// trainers produce before calling Encode.
type Weights struct {
	InProj []float32 // [256*3H]
	WHH    []float32 // [3H*H]
	BHH    []float32 // [3H]
	WBit   []float32 // [8*H]
	BBit   []float32 // [8]

	HiddenDim int
	ID        uint32
}

// NewWeights allocates zeroed float weights for hidden dimension h.
func NewWeights(h, id int) *Weights {
	return &Weights{
		InProj:    make([]float32, 256*3*h),
		WHH:       make([]float32, 3*h*h),
		BHH:       make([]float32, 3*h),
		WBit:      make([]float32, 8*h),
		BBit:      make([]float32, 8),
		HiddenDim: h,
		ID:        uint32(id),
	}
}

// Encode serialises the weights as a version-2 NGRM artifact.
func (w *Weights) Encode() ([]byte, error) {
	h := w.HiddenDim
	if h <= 0 || h > MaxHidden {
		return nil, fmt.Errorf("invalid hidden_dim=%d", h)
	}
	if len(w.InProj) != 256*3*h || len(w.WHH) != 3*h*h || len(w.BHH) != 3*h ||
		len(w.WBit) != 8*h || len(w.BBit) != 8 {
		return nil, fmt.Errorf("weight slices do not match hidden_dim=%d", h)
	}
	out := make([]byte, 16+NumFloats(h)*4)
	copy(out[0:4], Magic[:])
	binary.LittleEndian.PutUint16(out[4:6], Version)
	binary.LittleEndian.PutUint16(out[6:8], uint16(h))
	binary.LittleEndian.PutUint32(out[8:12], w.ID)
	binary.LittleEndian.PutUint32(out[12:16], 0) // reserved
	off := 16
	for _, s := range [][]float32{w.InProj, w.WHH, w.BHH, w.WBit, w.BBit} {
		for _, v := range s {
			binary.LittleEndian.PutUint32(out[off:], math.Float32bits(v))
			off += 4
		}
	}
	return out, nil
}

// Parse decodes and quantises an NGRM artifact.
func Parse(raw []byte) (*Model, error) {
	if len(raw) < 16 || !bytes.Equal(raw[:4], Magic[:]) {
		return nil, fmt.Errorf("invalid NGRM header")
	}
	ver := binary.LittleEndian.Uint16(raw[4:6])
	if ver != Version {
		return nil, fmt.Errorf("unsupported NGRM version %d (want %d)", ver, Version)
	}
	h := int(binary.LittleEndian.Uint16(raw[6:8]))
	if h <= 0 || h > MaxHidden {
		return nil, fmt.Errorf("invalid hidden_dim=%d", h)
	}
	want := 16 + NumFloats(h)*4
	if len(raw) != want {
		return nil, fmt.Errorf("unexpected NGRM size %d (want %d)", len(raw), want)
	}
	id := binary.LittleEndian.Uint32(raw[8:12])

	off := 16
	next := func() float32 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[off:]))
		off += 4
		return v
	}

	m := &Model{
		InProj:    make([]int32, 256*3*h),
		WHH:       make([]int16, 3*h*h),
		BHH:       make([]int32, 3*h),
		WBit:      make([]int16, 8*h),
		BBit:      make([]int32, 8),
		HiddenDim: h,
		ID:        id,
		Hash:      crc32.Checksum(raw, crc32.MakeTable(crc32.Castagnoli)),
	}
	for i := range m.InProj {
		m.InProj[i] = q16(next())
	}
	for i := range m.WHH {
		m.WHH[i] = q11(next())
	}
	for i := range m.BHH {
		m.BHH[i] = q16(next())
	}
	for i := range m.WBit {
		m.WBit[i] = q11(next())
	}
	for i := range m.BBit {
		m.BBit[i] = q7(next())
	}
	return m, nil
}

// q16 quantises to Q16 with saturation. Multiplying by a power of two and
// rounding with math.Round is exact, so the result is arch-independent.
func q16(f float32) int32 {
	v := math.Round(float64(f) * 65536)
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

func q11(f float32) int16 {
	v := math.Round(float64(f) * 2048)
	if v > maxWeightQ11 {
		return maxWeightQ11
	}
	if v < -maxWeightQ11 {
		return -maxWeightQ11
	}
	return int16(v)
}

func q7(f float32) int32 {
	v := math.Round(float64(f) * 128)
	return int32(clamp(v, -1<<20, 1<<20))
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ==========================================
// Registry
// ==========================================

// Loader fetches raw NGRM bytes for a model id.
type Loader interface {
	Load(ctx context.Context, id uint32) ([]byte, error)
}

// HashLoader is an optional Loader extension that can resolve an exact
// content-addressed artifact. Archives record the model hash, so a retrain of
// model 101 no longer makes yesterday's archives undecodable.
type HashLoader interface {
	LoadHash(ctx context.Context, id, hash uint32) ([]byte, error)
}

type key struct {
	id   uint32
	hash uint32
}

// Registry caches immutable models by content, and holds one hot-swappable
// active model per dataset.
type Registry struct {
	loader Loader

	loadMu sync.Mutex // serialises cache misses (single-flight)

	mu     sync.RWMutex
	models map[key]*Model
	active map[string]*atomic.Pointer[Model]
}

func NewRegistry(loader Loader) *Registry {
	return &Registry{
		loader: loader,
		models: make(map[key]*Model),
		active: make(map[string]*atomic.Pointer[Model]),
	}
}

// Register adds an immutable model to the cache.
func (r *Registry) Register(m *Model) {
	r.mu.Lock()
	r.models[key{m.ID, m.Hash}] = m
	r.mu.Unlock()
}

// HotSwap atomically replaces the active compression model for a dataset.
func (r *Registry) HotSwap(dataset string, m *Model) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models[key{m.ID, m.Hash}] = m
	ptr, ok := r.active[dataset]
	if !ok {
		ptr = &atomic.Pointer[Model]{}
		r.active[dataset] = ptr
	}
	// Store while holding the lock so concurrent swaps are totally ordered:
	// the store that wins the lock last is the one that stays visible.
	ptr.Store(m)
}

// Active returns the current compression model for a dataset.
func (r *Registry) Active(dataset string) (*Model, bool) {
	r.mu.RLock()
	ptr, ok := r.active[dataset]
	r.mu.RUnlock()
	if !ok {
		return nil, false
	}
	m := ptr.Load()
	return m, m != nil
}

// Get resolves a historical model by id and, when hash is non-zero, verifies
// its content hash. A hash mismatch is an error, never a silent fallback.
func (r *Registry) Get(ctx context.Context, id, hash uint32) (*Model, error) {
	k := key{id, hash}
	r.mu.RLock()
	m, ok := r.models[k]
	r.mu.RUnlock()
	if ok {
		return m, nil
	}

	r.loadMu.Lock()
	defer r.loadMu.Unlock()
	r.mu.RLock()
	m, ok = r.models[k]
	r.mu.RUnlock()
	if ok {
		return m, nil
	}
	if r.loader == nil {
		return nil, fmt.Errorf("model_id=%d hash=%08x not found in registry", id, hash)
	}

	var raw []byte
	var err error
	if hl, ok := r.loader.(HashLoader); ok && hash != 0 {
		raw, err = hl.LoadHash(ctx, id, hash)
	} else {
		raw, err = r.loader.Load(ctx, id)
	}
	if err != nil {
		return nil, fmt.Errorf("load model_id=%d hash=%08x: %w", id, hash, err)
	}
	m, err = Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("parse model_id=%d: %w", id, err)
	}
	if hash != 0 && m.Hash != hash {
		return nil, fmt.Errorf("model content mismatch for id=%d: artifact hash %08x, archive expects %08x", id, m.Hash, hash)
	}
	// Cache under both the requested key and the real hash.
	r.mu.Lock()
	r.models[k] = m
	r.models[key{m.ID, m.Hash}] = m
	r.mu.Unlock()
	return m, nil
}

// Count reports the number of cached models (useful for tests and metrics).
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}
