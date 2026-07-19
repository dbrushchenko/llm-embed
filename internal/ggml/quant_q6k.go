package ggml

import "fmt"

// Q6_K dequantization — pure Go implementation.
// Format: 256 elements per block, 210 bytes per block.
//
// Block layout (210 bytes):
//   [128] ql     — low 4 bits of 6-bit quants (256 values, 2 per byte via nibble packing)
//   [64]  qh     — high 2 bits of 6-bit quants (256 values, 4 per byte)
//   [16]  scales — per-group scale (int8, 16 groups of 16 elements)
//   [2]   d      — f16 super-block scale
//
// Dequant formula:
//   6-bit value = (ql_nibble) | ((qh_2bits) << 4)  → range [0, 63], then subtract 32 → [-32, 31]
//   float = d * scales[group] * (6bit_value - 32)

const (
	Q6_K_BytesPerBlock = 210 // 128 + 64 + 16 + 2
)

// DequantQ6_K dequantizes Q6_K data into float32 values.
func DequantQ6_K(data []byte, numElements int) ([]float32, error) {
	numBlocks := numElements / QK_K
	if len(data) < numBlocks*Q6_K_BytesPerBlock {
		return nil, fmt.Errorf("ggml: Q6_K data too short: have %d, need %d",
			len(data), numBlocks*Q6_K_BytesPerBlock)
	}

	out := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q6_K_BytesPerBlock:]

		ql := block[0:128]         // low 4 bits
		qh := block[128:192]       // high 2 bits
		scales := block[192:208]   // int8 scales
		dBits := block[208:210]    // f16 super-block scale

		d := f16ToF32(uint16(dBits[0]) | uint16(dBits[1])<<8)
		baseOut := b * QK_K

		// Process in groups of 128 (matching the C implementation's outer loop)
		for n := 0; n < QK_K; n += 128 {
			qlOff := n / 2  // 128 bytes covers 256 values (2 per byte), so offset is n/2
			qhOff := n / 4  // 64 bytes covers 256 values (4 per byte), so offset is n/4
			scOff := (n / 16) // 16 scales per 128-group → n/128 * 8 scales

			for l := 0; l < 32; l++ {
				is := l / 16 // sub-group index within this 128-chunk

				// Reconstruct 6-bit values from ql (4 low bits) and qh (2 high bits)
				q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int8((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32

				sc0 := int8(scales[scOff+is+0])
				sc2 := int8(scales[scOff+is+2])
				sc4 := int8(scales[scOff+is+4])
				sc6 := int8(scales[scOff+is+6])

				out[baseOut+n+l] = d * float32(sc0) * float32(q1)
				out[baseOut+n+l+32] = d * float32(sc2) * float32(q2)
				out[baseOut+n+l+64] = d * float32(sc4) * float32(q3)
				out[baseOut+n+l+96] = d * float32(sc6) * float32(q4)
			}

			// Advance ql/qh offsets for next 128-element chunk
			// (ql uses 64 bytes per 128 elements, qh uses 32 bytes per 128 elements)
		}
	}

	return out, nil
}

// DequantQ6_KSlice is a convenience wrapper.
func DequantQ6_KSlice(data []byte, numElements int) ([]float32, error) {
	return DequantQ6_K(data, numElements)
}

// Q6_K_RowSize returns the byte size of one row of Q6_K data.
func Q6_K_RowSize(numElements int) int {
	return (numElements / QK_K) * Q6_K_BytesPerBlock
}

// DotQ6_K computes the fused dequant+dot product for Q6_K data. Zero allocation.
func DotQ6_K(data []byte, x []float32, n int) float32 {
	numBlocks := n / QK_K
	var total float32

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q6_K_BytesPerBlock:]

		ql := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		dBits := block[208:210]

		d := f16ToF32(uint16(dBits[0]) | uint16(dBits[1])<<8)
		baseX := b * QK_K

		for n := 0; n < QK_K; n += 128 {
			qlOff := n / 2
			qhOff := n / 4
			scOff := (n / 16)

			for l := 0; l < 32; l++ {
				is := l / 16

				q1 := int8((ql[qlOff+l]&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
				q2 := int8((ql[qlOff+l+32]&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
				q3 := int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32

				sc0 := int8(scales[scOff+is+0])
				sc2 := int8(scales[scOff+is+2])
				sc4 := int8(scales[scOff+is+4])
				sc6 := int8(scales[scOff+is+6])

				total += d * float32(sc0) * float32(q1) * x[baseX+n+l]
				total += d * float32(sc2) * float32(q2) * x[baseX+n+l+32]
				total += d * float32(sc4) * float32(q3) * x[baseX+n+l+64]
				total += d * float32(sc6) * float32(q4) * x[baseX+n+l+96]
			}
		}
	}

	return total
}
