//go:build amd64 && !purego

package codec

// dotI16Kernel is implemented in dot_amd64.s using SSE2 PMADDWD.
// n must be a multiple of 8; both slices must be at least n long.
func dotI16Kernel(a, b *int16, n int) int32

// dotI16 returns sum(a[i]*b[i]) as an int32.
//
// Overflow safety: with act/weight quantisation at Q11 the largest possible
// product is 32768*32768 = 2^30, but real activations are bounded by +-2048
// and weights by +-32768, so |a*b| <= 2^26 and a sum of H<=256 terms stays
// below 2^34... which is why the contract is |a| <= 2048 (Q11 activation) and
// |b| <= 32768 (Q11 weight), giving |sum| <= 256*2^26 = 2^34. Callers clamp
// activations to +-2048 and weights are clamped at load time, so a sum never
// exceeds 2^31 for H <= 256. Guarded by TestDotI16Overflow.
func dotI16(a, b []int16) int32 {
	n := len(b)
	if len(a) < n {
		n = len(a)
	}
	var acc int32
	k := 0
	if n >= 8 {
		m := n &^ 7
		acc = dotI16Kernel(&a[0], &b[0], m)
		k = m
	}
	for ; k < n; k++ {
		acc += int32(a[k]) * int32(b[k])
	}
	return acc
}
