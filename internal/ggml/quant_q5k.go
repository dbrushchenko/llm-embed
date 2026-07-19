package ggml

import "fmt"

// Q5_K dequantization — pure Go implementation.
// Format: 256 elements per block, 176 bytes per block.
//
// Block layout (176 bytes total):
//   [2]   d      — f16 super-block scale
//   [2]   dmin   — f16 super-block minimum
//   [12]  scales — packed 6-bit scale/min pairs for 8 sub-blocks (32 elements each)
//   [32]  qh     — high bits (1 bit per element: 256 bits = 32 bytes)
//   [128] qs     — low 4 bits per element (256/2 = 128 bytes, nibble packed)
//
// High-bit layout (matches llama.cpp):
//   qh is 32 bytes. qh[l] (l=0..31) contains 8 bits — one per sub-group.
//   Bit s of qh[l] is the high bit for position l in sub-group s.
//   Sub-groups are numbered 0-7 across 4 iterations of j (2 sub-groups per j):
//     j=0: sub-group 0 (low nibble), sub-group 1 (high nibble)
//     j=1: sub-group 2 (low nibble), sub-group 3 (high nibble)
//     j=2: sub-group 4 (low nibble), sub-group 5 (high nibble)
//     j=3: sub-group 6 (low nibble), sub-group 7 (high nibble)

const (
	Q5_K_BytesPerBlock = 176 // 2 + 2 + 12 + 32 + 128
)

// DequantQ5_K dequantizes Q5_K data into float32 values.
func DequantQ5_K(data []byte, numElements int) ([]float32, error) {
	numBlocks := numElements / QK_K
	if len(data) < numBlocks*Q5_K_BytesPerBlock {
		return nil, fmt.Errorf("ggml: Q5_K data too short: have %d bytes, need %d",
			len(data), numBlocks*Q5_K_BytesPerBlock)
	}

	out := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q5_K_BytesPerBlock:]

		d := f16ToF32(uint16(block[0]) | uint16(block[1])<<8)
		dmin := f16ToF32(uint16(block[2]) | uint16(block[3])<<8)

		scales := block[4:16]  // 12 bytes packed scales
		qh := block[16:48]     // 32 bytes high bits
		qs := block[48:176]    // 128 bytes low 4 bits

		baseOut := b * QK_K
		is := 0    // sub-block scale index
		qOff := 0  // offset into qs
		qhBit := uint(0) // which bit of qh[l] to test (0-7, increments per sub-group)

		// Process 4 groups of 64 elements (each group = 2 sub-blocks of 32)
		for j := 0; j < QK_K; j += 64 {
			// Get scale and min for first sub-block of 32
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			dm1 := dmin * float32(m1)

			// Get scale and min for second sub-block of 32
			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			dm2 := dmin * float32(m2)

			// First 32 elements: low nibble of qs + high bit from qh
			// High bit = bit qhBit of qh[l], where l is position 0-31
			for l := 0; l < 32; l++ {
				low4 := qs[qOff+l] & 0xF
				high1 := (qh[l] >> qhBit) & 1
				q := low4 | (high1 << 4)
				out[baseOut+j+l] = d1*float32(q) - dm1
			}
			qhBit++

			// Second 32 elements: high nibble of qs + high bit from qh
			// High bit = bit qhBit of qh[l], where l is position 0-31
			for l := 0; l < 32; l++ {
				low4 := qs[qOff+l] >> 4
				high1 := (qh[l] >> qhBit) & 1
				q := low4 | (high1 << 4)
				out[baseOut+j+32+l] = d2*float32(q) - dm2
			}
			qhBit++

			qOff += 32
			is += 2
		}
	}

	return out, nil
}

// Q5_K_RowSize returns the byte size of one row of Q5_K data.
func Q5_K_RowSize(numElements int) int {
	return (numElements / QK_K) * Q5_K_BytesPerBlock
}

// DotQ5_K computes the fused dequant+dot product for Q5_K data. Zero allocation.
// data: raw Q5_K bytes, x: float32 input, n: number of elements.
func DotQ5_K(data []byte, x []float32, n int) float32 {
	numBlocks := n / QK_K
	var total float32

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q5_K_BytesPerBlock:]

		d := f16ToF32(uint16(block[0]) | uint16(block[1])<<8)
		dmin := f16ToF32(uint16(block[2]) | uint16(block[3])<<8)

		scales := block[4:16]
		qh := block[16:48]
		qs := block[48:176]

		baseX := b * QK_K
		is := 0
		qOff := 0
		qhBit := uint(0)

		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			d1 := d * float32(sc1)
			dm1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			d2 := d * float32(sc2)
			dm2 := dmin * float32(m2)

			// First 32: low nibble + high bit
			for l := 0; l < 32; l++ {
				low4 := qs[qOff+l] & 0xF
				high1 := (qh[l] >> qhBit) & 1
				q := low4 | (high1 << 4)
				val := d1*float32(q) - dm1
				total += val * x[baseX+j+l]
			}
			qhBit++

			// Second 32: high nibble + high bit
			for l := 0; l < 32; l++ {
				low4 := qs[qOff+l] >> 4
				high1 := (qh[l] >> qhBit) & 1
				q := low4 | (high1 << 4)
				val := d2*float32(q) - dm2
				total += val * x[baseX+j+32+l]
			}
			qhBit++

			qOff += 32
			is += 2
		}
	}

	return total
}
