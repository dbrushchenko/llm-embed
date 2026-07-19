package ggml

import "fmt"

// Q4_K dequantization — pure Go implementation.
// Format: 256 elements per block, 144 bytes per block.
//
// Block layout (144 bytes total):
//   [2] d     — f16 super-block scale
//   [2] dmin  — f16 super-block minimum
//   [12] scales — packed 6-bit scale/min pairs for 8 sub-blocks (32 elements each)
//   [128] qs   — 256 x 4-bit quantized values (2 per byte, low nibble first)
//
// Dequant formula per sub-block of 32:
//   value[i] = d * sub_scale * quant[i] - dmin * sub_min
//
// Each super-block has 8 sub-blocks of 32 elements.
// Processed in groups of 64 (2 sub-blocks): first 32 use low nibble, next 32 use high nibble.

const (
	QK_K          = 256 // elements per K-quant block
	Q4_K_BytesPerBlock = 144 // 2 + 2 + 12 + 128
)

// DequantQ4_K dequantizes a Q4_K block into float32 values.
// data must be at least numElements/256 * 144 bytes.
func DequantQ4_K(data []byte, numElements int) ([]float32, error) {
	numBlocks := numElements / QK_K
	if len(data) < numBlocks*Q4_K_BytesPerBlock {
		return nil, fmt.Errorf("ggml: Q4_K data too short: have %d bytes, need %d",
			len(data), numBlocks*Q4_K_BytesPerBlock)
	}

	out := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q4_K_BytesPerBlock:]

		// Read d and dmin (f16)
		d := f16ToF32(uint16(block[0]) | uint16(block[1])<<8)
		dmin := f16ToF32(uint16(block[2]) | uint16(block[3])<<8)

		scales := block[4:16]  // 12 bytes of packed scales
		qs := block[16:144]    // 128 bytes of 4-bit quants

		baseOut := b * QK_K
		is := 0   // sub-block scale index
		qOff := 0 // offset into qs

		// Process 4 groups of 64 elements
		for j := 0; j < QK_K; j += 64 {
			// Get scale and min for first sub-block of 32
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			dm1 := dmin * float32(m1)

			// Get scale and min for second sub-block of 32
			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			dm2 := dmin * float32(m2)

			// First 32: low nibble
			for l := 0; l < 32; l++ {
				out[baseOut+j+l] = d1*float32(qs[qOff+l]&0xF) - dm1
			}
			// Second 32: high nibble
			for l := 0; l < 32; l++ {
				out[baseOut+j+32+l] = d2*float32(qs[qOff+l]>>4) - dm2
			}

			qOff += 32
			is += 2
		}
	}

	return out, nil
}

// DequantQ4_KSlice dequantizes a raw Q4_K byte slice (convenience wrapper).
func DequantQ4_KSlice(data []byte, numElements int) ([]float32, error) {
	return DequantQ4_K(data, numElements)
}

// DotQ4_K computes the fused dequant+dot product of a Q4_K quantized vector
// with a float32 vector. Zero allocation.
// data: raw Q4_K bytes, x: float32 input, n: number of elements.
func DotQ4_K(data []byte, x []float32, n int) float32 {
	numBlocks := n / QK_K
	var total float32

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q4_K_BytesPerBlock:]

		d := f16ToF32(uint16(block[0]) | uint16(block[1])<<8)
		dmin := f16ToF32(uint16(block[2]) | uint16(block[3])<<8)
		scales := block[4:16]
		qs := block[16:144]

		baseX := b * QK_K
		is := 0
		qOff := 0

		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			dm1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			dm2 := dmin * float32(m2)

			// Fused: dequant + multiply + accumulate
			for l := 0; l < 32; l++ {
				val := d1*float32(qs[qOff+l]&0xF) - dm1
				total += val * x[baseX+j+l]
			}
			for l := 0; l < 32; l++ {
				val := d2*float32(qs[qOff+l]>>4) - dm2
				total += val * x[baseX+j+32+l]
			}

			qOff += 32
			is += 2
		}
	}

	return total
}

// getScaleMinK4 unpacks a 6-bit scale and 6-bit min from the packed scales array.
// j: sub-block index (0-7), scales: 12-byte packed array.
// Returns (scale, min) as uint8 values (0-63).
func getScaleMinK4(j int, scales []byte) (uint8, uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	sc := (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
	m := (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	return sc, m
}

// Q4_K_RowSize returns the byte size of one row of Q4_K data.
func Q4_K_RowSize(numElements int) int {
	return (numElements / QK_K) * Q4_K_BytesPerBlock
}
