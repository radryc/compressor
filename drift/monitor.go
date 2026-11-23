package drift

import (
	"context"
	"sync"
	"time"
)

// Event describes a detected compression degradation for a dataset.
type Event struct {
	Timestamp   time.Time
	Dataset     string
	BaselineBPB float64
	CurrentBPB  float64
	ModelID     uint32
}

// Trigger handles drift notifications (for example, enqueueing a retrain).
type Trigger interface {
	OnDrift(ctx context.Context, ev Event) error
}

type Option func(*Monitor)

// WithWindowLen sets how many blocks are aggregated into one measurement.
func WithWindowLen(n int) Option {
	return func(m *Monitor) {
		if n > 0 {
			m.windowLen = n
		}
	}
}

// WithCooldown sets the minimum spacing between alerts for one dataset.
func WithCooldown(d time.Duration) Option {
	return func(m *Monitor) {
		if d > 0 {
			m.cooldown = d
		}
	}
}

// WithClock injects a clock (tests).
func WithClock(now func() time.Time) Option {
	return func(m *Monitor) {
		if now != nil {
			m.now = now
		}
	}
}

// WithAdaptation sets the EWMA rate (1/scale). A higher scale adapts slower.
func WithAdaptation(scale int) Option {
	return func(m *Monitor) {
		if scale > 0 {
			m.adapt = scale
		}
	}
}

type sample struct {
	rawBytes  int64
	compBytes int64
}

type datasetWindow struct {
	ring  []sample
	head  int
	count int

	// ewma is the rolling baseline bits-per-byte.
	ewma      float64
	lastAlert time.Time
	everAlert bool
}

// Monitor implements stream.Observer and tracks rolling bits-per-byte per
// dataset.
//
// The baseline is an EWMA, not a single frozen sample: a model whose ratio
// legitimately settles at a new level must stop alerting, and a retrain that
// improves the ratio must lower the bar. Upward adaptation is deliberately
// slower than downward adaptation so a genuine regression is still reported.
type Monitor struct {
	trigger   Trigger
	now       func() time.Time
	threshold float64 // bits per byte above the baseline
	cooldown  time.Duration
	windowLen int
	adapt     int

	mu      sync.Mutex
	windows map[string]*datasetWindow
}

// NewMonitor creates a drift monitor. thresholdBPB is the regression, in bits
// per byte, that counts as drift.
func NewMonitor(trigger Trigger, thresholdBPB float64, opts ...Option) *Monitor {
	m := &Monitor{
		trigger:   trigger,
		now:       time.Now,
		threshold: thresholdBPB,
		cooldown:  15 * time.Minute,
		windowLen: 32,
		adapt:     8,
		windows:   make(map[string]*datasetWindow),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// RecordBlock satisfies stream.Observer.
func (m *Monitor) RecordBlock(dataset string, modelID uint32, rawBytes, compBytes int) {
	if rawBytes <= 0 {
		return
	}
	now := m.now()

	var ev *Event
	m.mu.Lock()
	w, ok := m.windows[dataset]
	if !ok {
		w = &datasetWindow{ring: make([]sample, m.windowLen)}
		m.windows[dataset] = w
	}
	w.ring[w.head] = sample{rawBytes: int64(rawBytes), compBytes: int64(compBytes)}
	w.head = (w.head + 1) % m.windowLen
	if w.count < m.windowLen {
		w.count++
	}

	if w.count == m.windowLen {
		var totRaw, totComp int64
		for i := 0; i < w.count; i++ {
			totRaw += w.ring[i].rawBytes
			totComp += w.ring[i].compBytes
		}
		if totRaw > 0 {
			cur := float64(totComp) * 8 / float64(totRaw)
			switch {
			case w.ewma == 0:
				w.ewma = cur
			case cur > w.ewma+m.threshold:
				// Regression: adapt slowly so a permanent shift converges, but
				// alert immediately if the cooldown allows.
				if now.Sub(w.lastAlert) >= m.cooldown || !w.everAlert {
					w.lastAlert = now
					w.everAlert = true
					ev = &Event{
						Timestamp:   now,
						Dataset:     dataset,
						BaselineBPB: w.ewma,
						CurrentBPB:  cur,
						ModelID:     modelID,
					}
				}
				w.ewma += (cur - w.ewma) / float64(m.adapt*4)
			default:
				// Same or better: follow the improvement.
				w.ewma += (cur - w.ewma) / float64(m.adapt)
			}
		}
	}
	m.mu.Unlock()

	if ev != nil && m.trigger != nil {
		_ = m.trigger.OnDrift(context.Background(), *ev)
	}
}

// Baseline reports the current baseline for a dataset (tests and metrics).
func (m *Monitor) Baseline(dataset string) (float64, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	w, ok := m.windows[dataset]
	if !ok || w.ewma == 0 {
		return 0, false
	}
	return w.ewma, true
}

// Reset drops all history for a dataset (for example after promoting a new
// model, so the fresh baseline is not compared against the old one).
func (m *Monitor) Reset(dataset string) {
	m.mu.Lock()
	delete(m.windows, dataset)
	m.mu.Unlock()
}
