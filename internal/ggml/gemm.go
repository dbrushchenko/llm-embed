//go:build amd64

package ggml

import (
	"sync"
	"unsafe"
)

// GEMM micro-kernel parameters (BLIS-style for AVX2)
const (
	MR = 6  // rows of A per micro-kernel
	NR = 8  // cols of B per micro-kernel (one YMM width)
	KC = 768 // inner dimension block (full hidden dim fits in L2 for our use case)
	// MC and NC are chosen so working sets fit in cache:
	// MC × KC × 4 bytes should fit in L2 (~256KB): MC=48 → 48×768×4 = 147KB ✓
	// KC × NC × 4 bytes should fit in L3: NC=2304 (full output) ✓
	MC = 48
)

// gemmMicroKernel6x8AVX2 computes C[6×8] += A[6×KC] × B[KC×8]
// A is packed as KC groups of 6 contiguous floats (panel format)
// B is packed as KC groups of 8 contiguous floats (panel format)
//
//go:noescape
func gemmMicroKernel6x8AVX2(a, b, c unsafe.Pointer, KC, ldc int)

// packBPanelAVX2 packs 8 rows of B into panel format using AVX2.
// b points to B[jStart][0], stride is row length in floats.
//go:noescape
func packBPanelAVX2(b, packB unsafe.Pointer, K, stride int)

// gemmMicroKernel6x16AVX2 computes C[6x16] += A[6xKC] x B[KCx16]
// Uses all 16 YMM registers: 12 accumulators + 2 B loads + 1 broadcast + 1 spare.
//go:noescape
func gemmMicroKernel6x16AVX2(a, b, c unsafe.Pointer, KC, ldc int)

// packAPanelAVX2 packs 6 rows into panel format with 4x unrolled scalar loads.
//go:noescape
func packAPanelAVX2(r0, r1, r2, r3, r4, r5, dst unsafe.Pointer, K int)

// GemmBatch computes C = A × B where:
//   A: [M × K] (M = seqLen, K = hidden_dim, row-major as [][]float32)
//   B: quantized weight tensor [K × N] (in GGUF quant format)
//   Returns: [M × N] as [][]float32
//
// This is the BLIS/GotoBLAS approach: tile the multiply into cache-friendly
// blocks and process each block with the fused micro-kernel.
func GemmBatch(w *Tensor, inputs [][]float32) ([][]float32, error) {
	M := len(inputs)
	if M == 0 {
		return nil, nil
	}
	K := w.Shape[0]
	N := 1
	for _, d := range w.Shape[1:] {
		N *= d
	}

	// Allocate flat C matrix (row-major, contiguous for cache efficiency)
	C := make([]float32, M*N)

	// Dequantize weight into float32 panel buffer
	// For our model: K=768, N=2304 max → 768×2304×4 = 7MB (fits in L3)
	// We dequant in KC×NR tiles to keep working set in L1/L2
	B := DequantWeightMatrix(w, K, N)

	// Pack and compute using BLIS 5-loop structure
	nWorkers := NumWorkers()
	gemmTiled(inputs, B, C, M, N, K, nWorkers)

	// Convert flat C back to [][]float32
	outputs := make([][]float32, M)
	for i := 0; i < M; i++ {
		outputs[i] = C[i*N : (i+1)*N]
	}
	return outputs, nil
}

// gemmTiled implements the BLIS 5-loop pattern with parallel MC blocks
func gemmTiled(A [][]float32, B []float32, C []float32, M, N, K, nWorkers int) {
	// For our use case (M=8-12, K=768, N≤3072), the parallelism is over N blocks
	// since M is too small to parallelize over rows effectively.

	if M <= MR && N >= NR {
		// Fast path: M fits in one micro-kernel row-block
		// Parallelize over N dimension
		gemmParallelN(A, B, C, M, N, K, nWorkers)
		return
	}

	// General path: tile over M and N
	var wg sync.WaitGroup
	// Split N into chunks for workers
	nChunks := nWorkers
	colsPerChunk := ((N/NR + nChunks - 1) / nChunks) * NR // round up to NR boundary

	for w := 0; w < nChunks; w++ {
		jStart := w * colsPerChunk
		jEnd := jStart + colsPerChunk
		if jEnd > N {
			jEnd = N
		}
		if jStart >= N {
			break
		}
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
			gemmBlock(A, B, C, M, N, K, 0, M, jStart, jEnd)
		}(jStart, jEnd)
	}
	wg.Wait()
}

// gemmParallelN parallelizes over N columns (best for small M)
func gemmParallelN(A [][]float32, B []float32, C []float32, M, N, K, nWorkers int) {
	var wg sync.WaitGroup
	colsPerWorker := ((N/NR + nWorkers - 1) / nWorkers) * NR

	for w := 0; w < nWorkers; w++ {
		jStart := w * colsPerWorker
		jEnd := jStart + colsPerWorker
		if jEnd > N {
			jEnd = N
		}
		if jStart >= N {
			break
		}
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
			gemmBlock(A, B, C, M, N, K, 0, M, jStart, jEnd)
		}(jStart, jEnd)
	}
	wg.Wait()
}

// gemmBlock computes a sub-block of C[iStart:iEnd][jStart:jEnd]
func gemmBlock(A [][]float32, B []float32, C []float32, M, N, K, iStart, iEnd, jStart, jEnd int) {
	// Packed buffers (reused across iterations — avoids GC pressure)
	packA := make([]float32, MR*K)
	packB := make([]float32, K*NR)

	for i := iStart; i < iEnd; i += MR {
		mr := MR
		if i+mr > iEnd {
			mr = iEnd - i
		}

		// Pack A[i:i+mr][0:K] into panel format (k-major: for each k, mr values)
		packAPanel(A, packA, i, mr, K)

		for j := jStart; j < jEnd; j += NR {
			nr := NR
			if j+nr > jEnd {
				nr = jEnd - j
			}

			if mr == MR && nr == NR {
				// Full micro-kernel: 6×8
				gemmMicroKernelPacked(packA[:MR*K], packB, B, C, i, j, K, N)
			} else {
				// Edge case: partial tile, use scalar
				gemmScalarBlock(A, B, C, N, K, i, i+mr, j, j+nr)
			}
		}
	}
}

// gemmMicroKernelPacked handles the micro-kernel call with proper B packing
func gemmMicroKernelPacked(packA []float32, packB []float32, B []float32, C []float32, iStart, jStart, K, N int) {
	// Pack B[jStart:jStart+NR][0:K] into panel format: K groups of NR contiguous floats
	for k := 0; k < K; k++ {
		for n := 0; n < NR; n++ {
			// B is stored as N rows of K: B[(jStart+n)*K + k]
			packB[k*NR+n] = B[(jStart+n)*K+k]
		}
	}

	aPtr := unsafe.Pointer(&packA[0])
	bPtr := unsafe.Pointer(&packB[0])
	cPtr := unsafe.Pointer(&C[iStart*N+jStart])
	gemmMicroKernel6x8AVX2(aPtr, bPtr, cPtr, K, N)
}

// packAPanel packs rows of A into panel format for the micro-kernel.
// Panel format: for each k in [0,K), store MR values A[i+0][k], A[i+1][k], ..., A[i+MR-1][k]
func packAPanel(A [][]float32, pack []float32, iStart, mr, K int) {
	idx := 0
	for k := 0; k < K; k++ {
		for m := 0; m < MR; m++ {
			if iStart+m < len(A) {
				pack[idx] = A[iStart+m][k]
			} else {
				pack[idx] = 0
			}
			idx++
		}
	}
}

// gemmScalarBlock handles edge tiles that don't fill a full MR×NR micro-kernel
func gemmScalarBlock(A [][]float32, B []float32, C []float32, N, K, iStart, iEnd, jStart, jEnd int) {
	for i := iStart; i < iEnd; i++ {
		for j := jStart; j < jEnd; j++ {
			var sum float32
			// B is stored row-major as [N][K] (transposed weight)
			bRow := B[j*K : (j+1)*K]
			aRow := A[i]
			for k := 0; k < K; k++ {
				sum += aRow[k] * bRow[k]
			}
			C[i*N+j] += sum
		}
	}
}

// dequantWeightMatrix dequantizes the full weight tensor into [N][K] float32
// (transposed so each output neuron is a contiguous row for cache-friendly dot products)
func DequantWeightMatrix(w *Tensor, K, N int) []float32 {
	B := make([]float32, N*K)

	switch w.Type {
	case 8: // Q8_0
		blocksPerRow := K / 32
		bytesPerRow := blocksPerRow * 34
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			rowData := w.Data[offset : offset+bytesPerRow]
			dequantQ8_0Row(rowData, B[n*K:(n+1)*K], K)
		}
	case 12: // Q4_K
		bytesPerRow := Q4_K_RowSize(K)
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			rowData := w.Data[offset : offset+bytesPerRow]
			row, _ := DequantQ4_K(rowData, K)
			copy(B[n*K:(n+1)*K], row)
		}
	case 13, 15, 16: // Q5_K
		bytesPerRow := Q5_K_RowSize(K)
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			rowData := w.Data[offset : offset+bytesPerRow]
			row, _ := DequantQ5_K(rowData, K)
			copy(B[n*K:(n+1)*K], row)
		}
	case 14: // Q6_K
		bytesPerRow := Q6_K_RowSize(K)
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			rowData := w.Data[offset : offset+bytesPerRow]
			row, _ := DequantQ6_K(rowData, K)
			copy(B[n*K:(n+1)*K], row)
		}
	case 0: // F32
		bytesPerRow := K * 4
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			row, _ := DequantF32Slice(w.Data[offset:offset+bytesPerRow], K)
			copy(B[n*K:(n+1)*K], row)
		}
	}

	return B
}

// dequantQ8_0Row dequantizes one row of Q8_0 data into float32
func dequantQ8_0Row(data []byte, out []float32, K int) {
	blocks := K / 32
	for b := 0; b < blocks; b++ {
		blockOff := b * 34
		// First 2 bytes: f16 scale
		scale := f16ToF32Fast(uint16(data[blockOff]) | uint16(data[blockOff+1])<<8)
		// Next 32 bytes: int8 quants
		for i := 0; i < 32; i++ {
			q := int8(data[blockOff+2+i])
			out[b*32+i] = float32(q) * scale
		}
	}
}

// GemmFloat32 computes C[M×N] += A[M×K] × B[N×K]^T using tiled micro-kernels.
// B is stored as [N][K] (each output neuron is a row of K).
// C is pre-zeroed by caller, stored row-major [M×N].
func GemmFloat32(A [][]float32, B []float32, C []float32, M, N, K int) {
	gemmTiled(A, B, C, M, N, K, NumWorkers())
}



// PrePackBPanel converts a [N][K] row-major weight matrix to panel format: [N/NR panels][K][NR].
// Each panel of NR columns has K groups of NR contiguous floats.
// The micro-kernel can then read B sequentially with zero runtime packing.
func PrePackBPanel(B []float32, K, N int) []float32 {
	panels := (N + NR - 1) / NR
	packed := make([]float32, panels*K*NR)

	for p := 0; p < panels; p++ {
		jStart := p * NR
		panelOff := p * K * NR
		for k := 0; k < K; k++ {
			base := panelOff + k*NR
			for n := 0; n < NR; n++ {
				col := jStart + n
				if col < N {
					packed[base+n] = B[col*K+k]
				}
			}
		}
	}
	return packed
}
// GemmPrePacked computes C[M×N] = A[M×K] × B_packed (already in panel format).
// B_packed layout: [N/NR panels][K][NR] — micro-kernel reads sequentially, zero packing needed.
func GemmPrePacked(A [][]float32, Bpacked []float32, C []float32, M, N, K int) {
	nWorkers := NumWorkers()
	// Parallelize over N (panels)
	panels := (N + NR - 1) / NR
	var wg sync.WaitGroup
	panelsPerWorker := (panels + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		pStart := w * panelsPerWorker
		pEnd := pStart + panelsPerWorker
		if pEnd > panels {
			pEnd = panels
		}
		if pStart >= panels {
			break
		}
		wg.Add(1)
		go func(pStart, pEnd int) {
			defer wg.Done()
			packA := make([]float32, MR*K)
			for i := 0; i < M; i += MR {
				mr := MR
				if i+mr > M {
					mr = M - i
				}
				packAPanel(A, packA, i, mr, K)

				for p := pStart; p < pEnd; p++ {
					j := p * NR
					nr := NR
					if j+nr > N {
						nr = N - j
					}
					if mr == MR && nr == NR {
						// B panel is at offset p*K*NR, already in [K][NR] format
						bPtr := unsafe.Pointer(&Bpacked[p*K*NR])
						aPtr := unsafe.Pointer(&packA[0])
						cPtr := unsafe.Pointer(&C[i*N+j])
						gemmMicroKernel6x8AVX2(aPtr, bPtr, cPtr, K, N)
					} else {
						// Edge: scalar fallback
						for ii := i; ii < i+mr && ii < M; ii++ {
							for jj := j; jj < j+nr && jj < N; jj++ {
								var sum float32
								panelOff := p * K * NR
								for k := 0; k < K; k++ {
									sum += A[ii][k] * Bpacked[panelOff+k*NR+(jj-j)]
								}
								C[ii*N+jj] += sum
							}
						}
					}
				}
			}
		}(pStart, pEnd)
	}
	wg.Wait()
}

// PackAPanelExported is an exported wrapper for benchmarking.
func PackAPanelExported(A [][]float32, pack []float32, iStart, mr, K int) {
	packAPanel(A, pack, iStart, mr, K)
}


// GemmQuantTiled performs GEMM on quantized weight tensors by dequanting NR columns
// at a time into a small float32 buffer, then running the micro-kernel.
// This gives K-quants the GEMM benefit (amortize weight read across MR rows of A)
// without requiring full N×K pre-dequantization (~50MB).
//
// Memory overhead: only K×NR×4 = 768×8×4 = 24KB per worker (fits in L1 cache).
func GemmQuantTiled(w *Tensor, inputs [][]float32) ([][]float32, error) {
	M := len(inputs)
	if M == 0 {
		return nil, nil
	}
	K := w.Shape[0]
	N := 1
	for _, d := range w.Shape[1:] {
		N *= d
	}

	// Allocate flat output
	C := make([]float32, M*N)

	nWorkers := NumWorkers()
	panels := (N + NR - 1) / NR

	// Determine row size based on quant type
	var bytesPerRow int
	var dequantRow func(data []byte, out []float32, K int)

	switch w.Type {
	case 8: // Q8_0
		bytesPerRow = (K / 32) * 34
		dequantRow = dequantQ8_0Row
	case 12: // Q4_K
		bytesPerRow = Q4_K_RowSize(K)
		dequantRow = func(data []byte, out []float32, K int) {
			row, _ := DequantQ4_K(data, K)
			copy(out, row)
		}
	case 13, 15, 16: // Q5_K
		bytesPerRow = Q5_K_RowSize(K)
		dequantRow = func(data []byte, out []float32, K int) {
			row, _ := DequantQ5_K(data, K)
			copy(out, row)
		}
	case 14: // Q6_K
		bytesPerRow = Q6_K_RowSize(K)
		dequantRow = func(data []byte, out []float32, K int) {
			row, _ := DequantQ6_K(data, K)
			copy(out, row)
		}
	default:
		// Unsupported type — fallback to MatMulBatch
		return MatMulBatch(w, inputs)
	}

	// Parallel over panels (N dimension), each worker dequants its own tile
	var wg sync.WaitGroup
	panelsPerWorker := (panels + nWorkers - 1) / nWorkers

	for wk := 0; wk < nWorkers; wk++ {
		pStart := wk * panelsPerWorker
		pEnd := pStart + panelsPerWorker
		if pEnd > panels {
			pEnd = panels
		}
		if pStart >= panels {
			break
		}
		wg.Add(1)
		go func(pStart, pEnd int) {
			defer wg.Done()
			// Per-worker buffers (L1-sized)
			tileB := make([]float32, K*NR)  // dequanted tile: NR rows of K floats
			packB := make([]float32, K*NR)  // packed for micro-kernel: K groups of NR
			packA := make([]float32, MR*K)  // packed A panel

			for p := pStart; p < pEnd; p++ {
				jStart := p * NR
				nr := NR
				if jStart+nr > N {
					nr = N - jStart
				}

				// Dequant NR rows of the weight matrix into tileB
				for n := 0; n < nr; n++ {
					colIdx := jStart + n
					offset := colIdx * bytesPerRow
					rowData := w.Data[offset : offset+bytesPerRow]
					dequantRow(rowData, tileB[n*K:(n+1)*K], K)
				}
				// Zero any padding columns
				for n := nr; n < NR; n++ {
					for k := 0; k < K; k++ {
						tileB[n*K+k] = 0
					}
				}

				// Pack B tile into panel format [K][NR]
				for k := 0; k < K; k++ {
					for n := 0; n < NR; n++ {
						packB[k*NR+n] = tileB[n*K+k]
					}
				}

				// Process all rows of A against this B panel
				for i := 0; i < M; i += MR {
					mr := MR
					if i+mr > M {
						mr = M - i
					}

					if mr == MR && nr == NR {
						packAPanel(inputs, packA, i, mr, K)
						aPtr := unsafe.Pointer(&packA[0])
						bPtr := unsafe.Pointer(&packB[0])
						cPtr := unsafe.Pointer(&C[i*N+jStart])
						gemmMicroKernel6x8AVX2(aPtr, bPtr, cPtr, K, N)
					} else {
						// Edge case
						for ii := i; ii < i+mr && ii < M; ii++ {
							for jj := 0; jj < nr; jj++ {
								var sum float32
								for k := 0; k < K; k++ {
									sum += inputs[ii][k] * tileB[jj*K+k]
								}
								C[ii*N+jStart+jj] += sum
							}
						}
					}
				}
			}
		}(pStart, pEnd)
	}
	wg.Wait()

	// Convert flat C to [][]float32
	outputs := make([][]float32, M)
	for i := 0; i < M; i++ {
		outputs[i] = C[i*N : (i+1)*N]
	}
	return outputs, nil
}

const NR16 = 16

// PrePackBPanel16 converts [N][K] to panel format with NR=16.
func PrePackBPanel16(B []float32, K, N int) []float32 {
	panels := (N + NR16 - 1) / NR16
	packed := make([]float32, panels*K*NR16)
	for p := 0; p < panels; p++ {
		jStart := p * NR16
		panelOff := p * K * NR16
		for k := 0; k < K; k++ {
			base := panelOff + k*NR16
			for n := 0; n < NR16; n++ {
				col := jStart + n
				if col < N {
					packed[base+n] = B[col*K+k]
				}
			}
		}
	}
	return packed
}

// GemmPrePacked16 uses the 6×16 micro-kernel for 2× throughput per iteration.
func GemmPrePacked16(A [][]float32, Bpacked []float32, C []float32, M, N, K int) {
	nWorkers := NumWorkers()
	panels := (N + NR16 - 1) / NR16

	// KC blocking: process K in chunks that keep micro-kernel working set in L1.
	// OpenBLAS Haswell uses KC=320 for float. With MR=6, NR=16:
	//   A chunk: MR×KC×4 = 6×256×4 = 6KB
	//   B chunk: KC×NR16×4 = 256×16×4 = 16KB
	//   Total: 22KB — fits in 32KB L1 cache
	const KC = 128

	var wg sync.WaitGroup
	panelsPerWorker := (panels + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		pStart := w * panelsPerWorker
		pEnd := pStart + panelsPerWorker
		if pEnd > panels {
			pEnd = panels
		}
		if pStart >= panels {
			break
		}
		wg.Add(1)
		go func(pStart, pEnd int) {
			defer wg.Done()
			packA := make([]float32, MR*KC)

			// 3-level tiling: N(panels) → K(KC blocks) → M
			// B panel KC-chunk stays in L2 while A rotates through L1 for all M rows
			for p := pStart; p < pEnd; p++ {
				j := p * NR16
				nr := NR16
				if j+nr > N {
					nr = N - j
				}

				for kc := 0; kc < K; kc += KC {
					kcLen := KC
					if kc+kcLen > K {
						kcLen = K - kc
					}

					// B chunk pointer (stays in L2 for all M iterations below)
					bOff := p*K*NR16 + kc*NR16

					for i := 0; i < M; i += MR {
						mr := MR
						if i+mr > M {
							mr = M - i
						}

						if mr == MR && nr == NR16 {
							// Pack A[i:i+MR][kc:kc+kcLen]
							idx := 0
							for kk := kc; kk < kc+kcLen; kk++ {
								for m := 0; m < MR; m++ {
									packA[idx] = A[i+m][kk]
									idx++
								}
							}

							bPtr := unsafe.Pointer(&Bpacked[bOff])
							aPtr := unsafe.Pointer(&packA[0])
							cPtr := unsafe.Pointer(&C[i*N+j])
							gemmMicroKernel6x16AVX2(aPtr, bPtr, cPtr, kcLen, N)
						} else {
							// Edge: scalar
							for ii := i; ii < i+mr && ii < M; ii++ {
								for jj := 0; jj < nr && j+jj < N; jj++ {
									var sum float32
									panelOff := p * K * NR16
									for k := kc; k < kc+kcLen; k++ {
										sum += A[ii][k] * Bpacked[panelOff+k*NR16+jj]
									}
									C[ii*N+j+jj] += sum
								}
							}
						}
					}
				}
			}
		}(pStart, pEnd)
	}
	wg.Wait()
}
