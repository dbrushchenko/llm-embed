//go:build amd64

package ggml

// dotQ8_0AVX2 computes fused dequant+dot for Q8_0 quantized data using AVX2 SIMD.
// data: pointer to Q8_0 block data (34 bytes per block)
// x: pointer to float32 input vector
// n: number of elements (must be multiple of 32)
// Returns the dot product as float32.
//
// Implemented in dot_amd64.s
//
//go:noescape
func dotQ8_0AVX2(data *byte, x *float32, n int) float32

// dotF32AVX2 computes dot product of two float32 vectors using AVX2 FMA.
// a, b: pointers to float32 arrays. n: number of elements (must be multiple of 8).
//
//go:noescape
func dotF32AVX2(a *float32, b *float32, n int) float32

// dotQ6K32AVX2 computes fused dequant+dot for one 32-element Q6_K sub-group.
// ql: 32 bytes (low nibbles, one per byte after pre-processing by caller for high-nibble sub-groups).
// qh: 32 bytes (high 2-bit fields, 4 per byte).
// qhShift: right-shift amount for qh (0, 2, 4, or 6).
// scale: pre-computed (d * sub_scale).
// x: float32 input (32 floats).
//
//go:noescape
func dotQ6K32AVX2(ql *byte, qh *byte, qhShift uint64, scale float32, x *float32) float32

// dotQ6K32AVX2v2 is like dotQ6K32AVX2 but takes TWO scales: scale1 for elements 0-15,
// scale2 for elements 16-31. This matches Q6_K's per-16-element scale granularity.
//
//go:noescape
func dotQ6K32AVX2v2(ql *byte, qh *byte, qhShift uint64, scale1 float32, scale2 float32, x *float32) float32

// dotQ4K32AVX2 computes fused dequant+dot for one 32-element Q4_K sub-group.
// qs: 32 packed nibble bytes. isHigh: 0=low nibble, 1=high nibble.
// scale: d * sub_scale. minVal: dmin * sub_min. x: float32 input (32 floats).
//
//go:noescape
func dotQ4K32AVX2(qs *byte, isHigh uint64, scale float32, minVal float32, x *float32) float32

// dotQ5K32AVX2 computes fused dequant+dot for one 32-element Q5_K sub-group.
// qs: packed nibble bytes. qh: high-bit bytes (1 bit per element, in byte lanes).
// qhBitOffset: which bit in each qh byte. isHigh: 0=low nibble, 1=high.
// scale: d * sub_scale. minVal: dmin * sub_min. x: float32 input (32 floats).
//
//go:noescape
func dotQ5K32AVX2(qs *byte, qh *byte, qhBitOffset uint64, isHigh uint64, scale float32, minVal float32, x *float32) float32

// hasAVX2 reports whether the CPU supports AVX2 and FMA3.
var hasAVX2 = detectAVX2()

func detectAVX2() bool {
	eax1, _, ecx1, _ := cpuid(1, 0)
	_ = eax1
	hasFMA := ecx1&(1<<12) != 0
	hasAVX := ecx1&(1<<28) != 0
	hasOSXSAVE := ecx1&(1<<27) != 0

	if !hasAVX || !hasFMA || !hasOSXSAVE {
		return false
	}

	xcr0 := xgetbv(0)
	if xcr0&0x6 != 0x6 {
		return false
	}

	_, ebx7, _, _ := cpuid(7, 0)
	hasAVX2flag := ebx7&(1<<5) != 0

	return hasAVX2flag
}
