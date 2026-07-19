package ggml

import (
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// DotQ8_0 computes dot product of a float32 vector with a Q8_0 quantized vector
// WITHOUT allocating an intermediate float32 buffer. Fused dequant + dot.
// data: raw Q8_0 bytes (blocks of 34 bytes), x: float32 input, n: number of elements.
func DotQ8_0(data []byte, x []float32, n int) float32 {
	// Use AVX2 SIMD kernel when available (typically 5-8x faster)
	if hasAVX2 && n >= 32 {
		return dotQ8_0AVX2(&data[0], &x[0], n)
	}
	return dotQ8_0Scalar(data, x, n)
}

// dotQ8_0Scalar is the pure Go fallback for the Q8_0 dot product.
func dotQ8_0Scalar(data []byte, x []float32, n int) float32 {
	const blockSize = 32
	const bytesPerBlock = 34

	numBlocks := n / blockSize
	var total float32

	for b := 0; b < numBlocks; b++ {
		blockData := data[b*bytesPerBlock:]

		// Read f16 scale
		scaleBits := uint16(blockData[0]) | uint16(blockData[1])<<8
		scale := f16ToF32(scaleBits)

		// Fused: sum(quant[i] * x[base+i]) * scale
		quants := blockData[2 : 2+blockSize]
		baseIdx := b * blockSize
		var blockSum float32
		for i := 0; i < blockSize; i++ {
			q := int8(quants[i])
			blockSum += float32(q) * x[baseIdx+i]
		}
		total += blockSum * scale
	}

	return total
}

// DotQ8_0x4 computes 4 dot products simultaneously (loop unroll for ILP).
// Returns 4 dot products of the SAME x vector against 4 different Q8_0 rows.
func DotQ8_0x4(data0, data1, data2, data3 []byte, x []float32, n int) [4]float32 {
	const blockSize = 32
	const bytesPerBlock = 34
	numBlocks := n / blockSize

	var totals [4]float32

	for b := 0; b < numBlocks; b++ {
		off := b * bytesPerBlock
		baseIdx := b * blockSize

		s0 := f16ToF32(uint16(data0[off]) | uint16(data0[off+1])<<8)
		s1 := f16ToF32(uint16(data1[off]) | uint16(data1[off+1])<<8)
		s2 := f16ToF32(uint16(data2[off]) | uint16(data2[off+1])<<8)
		s3 := f16ToF32(uint16(data3[off]) | uint16(data3[off+1])<<8)

		q0 := data0[off+2 : off+2+blockSize]
		q1 := data1[off+2 : off+2+blockSize]
		q2 := data2[off+2 : off+2+blockSize]
		q3 := data3[off+2 : off+2+blockSize]

		var sum0, sum1, sum2, sum3 float32
		for i := 0; i < blockSize; i++ {
			xv := x[baseIdx+i]
			sum0 += float32(int8(q0[i])) * xv
			sum1 += float32(int8(q1[i])) * xv
			sum2 += float32(int8(q2[i])) * xv
			sum3 += float32(int8(q3[i])) * xv
		}

		totals[0] += sum0 * s0
		totals[1] += sum1 * s1
		totals[2] += sum2 * s2
		totals[3] += sum3 * s3
	}

	return totals
}

// MatMulVecQ8_0Parallel computes W @ x for Q8_0 weight matrix using goroutines.
// W shape: [K, N] (GGUF: inner=K, outer=N), x: [K], output: [N].
// Uses nWorkers goroutines to parallelize across output rows.
func MatMulVecQ8_0Parallel(wData []byte, x []float32, K, N, nWorkers int) []float32 {
	if nWorkers <= 0 {
		nWorkers = runtime.NumCPU()
	}

	blocksPerRow := K / 32
	bytesPerRow := blocksPerRow * 34
	out := make([]float32, N)

	if nWorkers == 1 || N < 256 {
		// Sequential with 4x unroll
		matMulVecQ8_0Chunk(wData, x, out, K, N, 0, N, bytesPerRow)
		return out
	}

	// Parallel: split N output rows across workers
	var wg sync.WaitGroup
	chunkSize := (N + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > N {
			end = N
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			matMulVecQ8_0Chunk(wData, x, out, K, N, start, end, bytesPerRow)
		}(start, end)
	}

	wg.Wait()
	return out
}

// matMulVecQ8_0Chunk computes a chunk of output rows [start, end) using 4x unrolling.
func matMulVecQ8_0Chunk(wData []byte, x []float32, out []float32, K, N, start, end, bytesPerRow int) {
	// Process 4 rows at a time for instruction-level parallelism
	n := start
	for ; n+4 <= end; n += 4 {
		d0 := wData[n*bytesPerRow:]
		d1 := wData[(n+1)*bytesPerRow:]
		d2 := wData[(n+2)*bytesPerRow:]
		d3 := wData[(n+3)*bytesPerRow:]

		results := DotQ8_0x4(
			d0[:bytesPerRow], d1[:bytesPerRow],
			d2[:bytesPerRow], d3[:bytesPerRow],
			x, K,
		)
		out[n] = results[0]
		out[n+1] = results[1]
		out[n+2] = results[2]
		out[n+3] = results[3]
	}

	// Handle remaining rows
	for ; n < end; n++ {
		offset := n * bytesPerRow
		out[n] = DotQ8_0(wData[offset:offset+bytesPerRow], x, K)
	}
}

// MatMulVecF32Parallel computes W @ x for F32 weight matrix using goroutines.
func MatMulVecF32Parallel(wData []byte, x []float32, K, N, nWorkers int) []float32 {
	if nWorkers <= 0 {
		nWorkers = runtime.NumCPU()
	}

	out := make([]float32, N)
	bytesPerRow := K * 4

	if nWorkers == 1 || N < 64 {
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			row := (*[1 << 28]float32)(unsafe.Pointer(&wData[offset]))[:K:K]
			var sum float32
			for k := 0; k < K; k++ {
				sum += row[k] * x[k]
			}
			out[n] = sum
		}
		return out
	}

	var wg sync.WaitGroup
	chunkSize := (N + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > N {
			end = N
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for n := start; n < end; n++ {
				offset := n * bytesPerRow
				row := (*[1 << 28]float32)(unsafe.Pointer(&wData[offset]))[:K:K]
				var sum float32
				for k := 0; k < K; k++ {
					sum += row[k] * x[k]
				}
				out[n] = sum
			}
		}(start, end)
	}

	wg.Wait()
	return out
}

// NumWorkers returns the recommended worker count for parallel ops.
func NumWorkers() int {
	n := runtime.NumCPU()
	if n > 8 {
		return 8 // Diminishing returns beyond 8 (cache contention)
	}
	if n < 2 {
		return 1
	}
	return n
}

// f16ToF32Fast converts f16 to f32 using direct bit manipulation (no branching for normals).
func f16ToF32Fast(h uint16) float32 {
	// Fast path for the common case (normal numbers, which is >99% of Q8_0 scales)
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal - rare in practice
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		exp += 127 - 15
		return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
	}
	if exp == 31 {
		return math.Float32frombits((sign << 31) | (0xff << 23) | (mant << 13))
	}
	exp += 127 - 15
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}

// --- K-Quant AVX2-accelerated dot products ---
//
// Strategy: scalar dequant into a FIXED stack buffer (no heap allocation),
// then AVX2 FMA dot product via dotF32AVX2. The dot product dominates at
// 768 dimensions (the BERT hidden size), so accelerating it gives ~3-4× speedup.
//
// This is not as fast as a fully fused register-only kernel (like Q8_0's dotQ8_0AVX2),
// but Q4_K/Q5_K/Q6_K have complex bit-packing that makes fully fused assembly
// impractical without 200+ lines of carefully debugged asm per format.
// The hybrid approach (scalar dequant + SIMD dot) is correct, fast enough,
// and maintainable.

// DotQ6_KFast computes Q6_K dot product using fused AVX2 sub-group kernels.
func DotQ6_KFast(data []byte, x []float32, n int) float32 {
	if !hasAVX2 || n < 256 {
		return DotQ6_K(data, x, n)
	}
	numBlocks := n / QK_K
	var total float32
	var qlShifted [32]byte // small stack buffer for high-nibble pre-shift

	for b := 0; b < numBlocks; b++ {
		block := data[b*Q6_K_BytesPerBlock:]
		ql := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		dBits := block[208:210]
		d := f16ToF32(uint16(dBits[0]) | uint16(dBits[1])<<8)
		baseX := b * QK_K

		// Process 2 chunks of 128 elements
		for chunk := 0; chunk < 2; chunk++ {
			qlOff := chunk * 64
			qhOff := chunk * 32
			scOff := chunk * 8
			xOff := baseX + chunk*128

			// Sub-group 0: ql[0:32] low nibble, qh shift=0
			// Elements 0-15 use scales[scOff+0], elements 16-31 use scales[scOff+1]
			scale1 := d * float32(int8(scales[scOff+0]))
			scale2 := d * float32(int8(scales[scOff+1]))
			total += dotQ6K32AVX2v2(&ql[qlOff], &qh[qhOff], 0, scale1, scale2, &x[xOff])

			// Sub-group 1: ql[32:64] low nibble, qh shift=2
			// Elements 0-15 use scales[scOff+2], elements 16-31 use scales[scOff+3]
			scale1 = d * float32(int8(scales[scOff+2]))
			scale2 = d * float32(int8(scales[scOff+3]))
			total += dotQ6K32AVX2v2(&ql[qlOff+32], &qh[qhOff], 2, scale1, scale2, &x[xOff+32])

			// Sub-group 2: ql[0:32] HIGH nibble, qh shift=4
			// Pre-shift ql bytes right by 4 into stack buffer
			for i := 0; i < 32; i++ {
				qlShifted[i] = ql[qlOff+i] >> 4
			}
			// Elements 0-15 use scales[scOff+4], elements 16-31 use scales[scOff+5]
			scale1 = d * float32(int8(scales[scOff+4]))
			scale2 = d * float32(int8(scales[scOff+5]))
			total += dotQ6K32AVX2v2(&qlShifted[0], &qh[qhOff], 4, scale1, scale2, &x[xOff+64])

			// Sub-group 3: ql[32:64] HIGH nibble, qh shift=6
			for i := 0; i < 32; i++ {
				qlShifted[i] = ql[qlOff+32+i] >> 4
			}
			// Elements 0-15 use scales[scOff+6], elements 16-31 use scales[scOff+7]
			scale1 = d * float32(int8(scales[scOff+6]))
			scale2 = d * float32(int8(scales[scOff+7]))
			total += dotQ6K32AVX2v2(&qlShifted[0], &qh[qhOff], 6, scale1, scale2, &x[xOff+96])
		}
	}
	return total
}

// DotQ4_KFast computes Q4_K dot product using fused AVX2 sub-group kernels.
func DotQ4_KFast(data []byte, x []float32, n int) float32 {
	if !hasAVX2 || n < 256 {
		return DotQ4_K(data, x, n)
	}
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
			scale1 := d * float32(sc1)
			min1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			scale2 := d * float32(sc2)
			min2 := dmin * float32(m2)

			total += dotQ4K32AVX2(&qs[qOff], 0, scale1, min1, &x[baseX+j])
			total += dotQ4K32AVX2(&qs[qOff], 1, scale2, min2, &x[baseX+j+32])

			qOff += 32
			is += 2
		}
	}
	return total
}

// DotQ5_KFast computes Q5_K dot product using fused AVX2 sub-group kernels.
func DotQ5_KFast(data []byte, x []float32, n int) float32 {
	if !hasAVX2 || n < 256 {
		return DotQ5_K(data, x, n)
	}
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

		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			scale1 := d * float32(sc1)
			min1 := dmin * float32(m1)

			sc2, m2 := getScaleMinK4(is+1, scales)
			scale2 := d * float32(sc2)
			min2 := dmin * float32(m2)

			total += dotQ5K32AVX2(&qs[qOff], &qh[0], uint64(is), 0, scale1, min1, &x[baseX+j])
			total += dotQ5K32AVX2(&qs[qOff], &qh[0], uint64(is+1), 1, scale2, min2, &x[baseX+j+32])

			qOff += 32
			is += 2
		}
	}
	return total
}


// matMulVecKQuantParallel parallelizes any K-quant matmul across goroutines.
// dotFn: the scalar fused dot product function for this quant type.
func matMulVecKQuantParallel(wData []byte, x []float32, K, N, bytesPerRow int, dotFn func([]byte, []float32, int) float32, nWorkers int) []float32 {
	if nWorkers <= 0 {
		nWorkers = runtime.NumCPU()
	}
	if nWorkers > 8 {
		nWorkers = 8
	}
	out := make([]float32, N)

	if nWorkers == 1 || N < 256 {
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			out[n] = dotFn(wData[offset:offset+bytesPerRow], x, K)
		}
		return out
	}

	var wg sync.WaitGroup
	chunkSize := (N + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > N {
			end = N
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for n := start; n < end; n++ {
				offset := n * bytesPerRow
				out[n] = dotFn(wData[offset:offset+bytesPerRow], x, K)
			}
		}(start, end)
	}

	wg.Wait()
	return out
}
