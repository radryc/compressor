package drift

import (
	"context"
	"testing"
	"time"
)

type recorder struct{ events []Event }

func (r *recorder) OnDrift(_ context.Context, ev Event) error {
	r.events = append(r.events, ev)
	return nil
}

func clock(start time.Time) func() time.Time {
	now := start
	return func() time.Time { return now }
}

func feed(m *Monitor, raw, comp int, n int) {
	for i := 0; i < n; i++ {
		m.RecordBlock("d", 1, raw, comp)
	}
}

func TestAlertOnRegression(t *testing.T) {
	rec := &recorder{}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	m := NewMonitor(rec, 0.5, WithWindowLen(4), WithClock(clock(now)))
	feed(m, 1000, 100, 8) // baseline ~0.8 bpb
	if len(rec.events) != 0 {
		t.Fatalf("unexpected alert: %+v", rec.events)
	}
	feed(m, 1000, 400, 8) // 3.2 bpb, a big regression
	if len(rec.events) != 1 {
		t.Fatalf("expected one alert, got %d", len(rec.events))
	}
	ev := rec.events[0]
	if ev.Dataset != "d" || ev.CurrentBPB <= ev.BaselineBPB {
		t.Fatalf("bad event: %+v", ev)
	}
}

func TestCooldownSuppressesRepeats(t *testing.T) {
	rec := &recorder{}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	m := NewMonitor(rec, 0.5, WithWindowLen(2), WithCooldown(time.Hour), WithClock(clock(now)))
	feed(m, 1000, 100, 4)
	feed(m, 1000, 900, 20)
	if len(rec.events) != 1 {
		t.Fatalf("expected 1 alert during cooldown, got %d", len(rec.events))
	}
}

func TestBaselineAdaptsOnImprovement(t *testing.T) {
	rec := &recorder{}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	m := NewMonitor(rec, 0.5, WithWindowLen(4), WithClock(clock(now)))
	feed(m, 1000, 400, 8) // 3.2 bpb baseline
	base1, _ := m.Baseline("d")
	feed(m, 1000, 100, 40) // much better
	base2, _ := m.Baseline("d")
	if base2 >= base1 {
		t.Fatalf("baseline did not adapt down: %f -> %f", base1, base2)
	}
	// And the improved level must not trigger alerts.
	if len(rec.events) != 0 {
		t.Fatalf("unexpected alert: %+v", rec.events)
	}
}

func TestPermanentShiftConverges(t *testing.T) {
	rec := &recorder{}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	m := NewMonitor(rec, 0.2, WithWindowLen(4), WithCooldown(time.Minute), WithClock(clock(now)))
	feed(m, 1000, 100, 8)
	feed(m, 1000, 300, 400) // permanently worse
	// It may alert a few times but must not alert on every window forever.
	if len(rec.events) > 4 {
		t.Fatalf("too many alerts for a stable shift: %d", len(rec.events))
	}
	base, _ := m.Baseline("d")
	if base < 2.0 {
		t.Fatalf("baseline did not move toward the new level: %f", base)
	}
}

func TestReset(t *testing.T) {
	m := NewMonitor(nil, 0.5, WithWindowLen(2))
	feed(m, 1000, 100, 4)
	if _, ok := m.Baseline("d"); !ok {
		t.Fatal("no baseline")
	}
	m.Reset("d")
	if _, ok := m.Baseline("d"); ok {
		t.Fatal("baseline survived reset")
	}
}
