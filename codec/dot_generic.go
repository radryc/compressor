//go:build !amd64 || purego

package codec

// dotI16 returns sum(a[i]*b[i]) as an int32. The pure-Go path is
// bit-identical to the SIMD path because it performs the same exact integer
// arithmetic.
func dotI16(a, b []int16) int32 {
	n := len(b)
	if len(a) < n {
		n = len(a)
	}
	var acc int32
	for k := 0; k < n; k++ {
		acc += int32(a[k]) * int32(b[k])
	}
	return acc
}
