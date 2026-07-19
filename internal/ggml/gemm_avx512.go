//go:build amd64

package ggml

import (
	"sync"
	"unsafe"
)

const NR32 = 32

// gemmMicroKernel6x32AVX512 computes C[6×32] += A[6×KC] × B[KC×32]
// Uses AVX-512 ZMM registers for 2× throughput vs AVX2.
//
//go:noescape
func gemmMicroKernel6x32AVX512(a, b, c unsafe.Pointer, KC, ldc int)

// HasAVX512 reports whether the CPU supports AVX-512F.
// Detected at init time via CPUID.
var HasAVX512 = detectAVX512()

func detectAVX512() bool {
	// CPUID EAX=7, ECX=0 → EBX bit 16 = AVX-512F
	return hasAVX512F()
}

// hasAVX512F detects AVX-512 Foundation via CPUID + XGETBV
func hasAVX512F() bool {
	// First check AVX2 prerequisites
	if !hasAVX2 {
		return false
	}

	// CPUID(7, 0).EBX bit 16 = AVX-512F
	_, ebx7, _, _ := cpuid(7, 0)
	if ebx7&(1<<16) == 0 {
		return false
	}

	// OS must support ZMM state saving: XCR0 bits 5,6,7 must be set
	// (bit 5 = opmask, bit 6 = ZMM_Hi256, bit 7 = Hi16_ZMM)
	xcr0 := xgetbv(0)
	return xcr0&0xE0 == 0xE0
}

// PrePackBPanel32 converts [N][K] to panel format with NR=32 for AVX-512.
func PrePackBPanel32(B []float32, K, N int) []float32 {
	panels := (N + NR32 - 1) / NR32
	packed := make([]float32, panels*K*NR32)
	for p := 0; p < panels; p++ {
		jStart := p * NR32
		panelOff := p * K * NR32
		for k := 0; k < K; k++ {
			base := panelOff + k*NR32
			for n := 0; n < NR32; n++ {
				col := jStart + n
				if col < N {
					packed[base+n] = B[col*K+k]
				}
			}
		}
	}
	return packed
}

// GemmPrePacked32 uses the 6×32 AVX-512 micro-kernel for 2× throughput.
func GemmPrePacked32(A [][]float32, Bpacked []float32, C []float32, M, N, K int) {
	nWorkers := NumWorkers()
	panels := (N + NR32 - 1) / NR32

	// KC for AVX-512: smaller than AVX2 because NR32 doubles B panel width.
	// B chunk: KC512×NR32×4 = 128×32×4 = 16KB (fits in L1 with A = 3KB)
	// Without this, B chunk = 256×32×4 = 32KB (fills entire L1, evicts A)
	const KC512 = 128

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
			packA := make([]float32, MR*KC512)

			// 3-level tiling: N→K→M (same as AVX2 path)
			for p := pStart; p < pEnd; p++ {
				j := p * NR32
				nr := NR32
				if j+nr > N {
					nr = N - j
				}

				for kc := 0; kc < K; kc += KC512 {
					kcLen := KC512
					if kc+kcLen > K {
						kcLen = K - kc
					}

					bOff := p*K*NR32 + kc*NR32

					for i := 0; i < M; i += MR {
						mr := MR
						if i+mr > M {
							mr = M - i
						}

						if mr == MR && nr == NR32 {
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
							gemmMicroKernel6x32AVX512(aPtr, bPtr, cPtr, kcLen, N)
						} else {
							// Edge: scalar fallback
							for ii := i; ii < i+mr && ii < M; ii++ {
								for jj := 0; jj < nr && j+jj < N; jj++ {
									var sum float32
									panelOff := p * K * NR32
									for k := kc; k < kc+kcLen; k++ {
										sum += A[ii][k] * Bpacked[panelOff+k*NR32+jj]
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
