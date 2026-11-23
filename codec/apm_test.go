package codec

import "testing"

func TestAPMIdentityThenLearns(t *testing.T) {
	var a apm
	a.Reset()
	// Untrained: a near-identity map.
	for _, p := range []uint32{1, 1000, 32768, 60000, 65535} {
		got := a.Prob(0, p)
		d := int64(got) - int64(p)
		if d < -300 || d > 300 {
			t.Fatalf("untrained APM(0,%d)=%d", p, got)
		}
	}
	// After always observing 1 in context 7, probabilities must rise.
	before := a.Prob(7, 20000)
	for i := 0; i < 400; i++ {
		a.Update(7, 20000, 1)
	}
	after := a.Prob(7, 20000)
	if after <= before {
		t.Fatalf("APM did not learn: %d -> %d", before, after)
	}
	if after < 40000 {
		t.Fatalf("APM learned too slowly: %d", after)
	}
	// A different context must be unaffected.
	if got := a.Prob(8, 20000); got > 25000 || got < 15000 {
		t.Fatalf("APM leaked across contexts: %d", got)
	}
}
