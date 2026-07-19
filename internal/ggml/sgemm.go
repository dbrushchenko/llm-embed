// Sgemm — self-contained float32 GEMM adapted from gonum.org/v1/gonum/blas/gonum.
// Copyright ©2014 The Gonum Authors. BSD-3-Clause license.
// Extracted to eliminate the external dependency while keeping the optimized tiled algorithm.
package ggml

import (
	"os"
	"runtime"
	"sync"
	"unsafe"
)

const (
	sgemmBlockSize   = 64 // sub-block size for cache-friendly tiling
	sgemmMinParBlock = 4  // minimum blocks needed to go parallel
)

func sgemmBlocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

// Sgemm computes C = alpha * op(A) × op(B) + beta * C.
// tA/tB: false=NoTrans, true=Trans.
// A is [M×K] or [K×M] if transposed. B is [K×N] or [N×K] if transposed. C is [M×N].
//
// Automatically uses Intel MKL (cblas_sgemm) if libmkl_rt.so is available at runtime.
// Falls back to pure Go tiled GEMM (BSD-3, adapted from gonum) otherwise.

func Sgemm(tA, tB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}

	// MKL path (initialized by InitSgemmBackend at startup)
	if useMKL {
		transA := int32(CblasNoTrans)
		if tA {
			transA = CblasTrans
		}
		transB := int32(CblasNoTrans)
		if tB {
			transB = CblasTrans
		}
		callCblasSgemm(
			mklSgemmPtr,
			int32(CblasRowMajor), transA, transB,
			int32(m), int32(n), int32(k),
			alpha,
			unsafe.Pointer(&a[0]), int32(lda),
			unsafe.Pointer(&b[0]), int32(ldb),
			beta,
			unsafe.Pointer(&c[0]), int32(ldc),
		)
		return
	}

	// Pure Go fallback: tiled parallel GEMM
	// Scale C by beta
	if beta != 1 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
		}
	}

	if alpha == 0 || k == 0 {
		return
	}

	sgemmParallelImpl(tA, tB, m, n, k, a, lda, b, ldb, c, ldc, alpha)
}

func sgemmParallelImpl(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	maxKLen := k
	parBlocks := sgemmBlocks(m, sgemmBlockSize) * sgemmBlocks(n, sgemmBlockSize)
	if parBlocks < sgemmMinParBlock {
		sgemmSerialImpl(aTrans, bTrans, m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	}

	workerLimit := make(chan struct{}, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup
	wg.Add(parBlocks)

	for i := 0; i < m; i += sgemmBlockSize {
		for j := 0; j < n; j += sgemmBlockSize {
			workerLimit <- struct{}{}
			go func(i, j int) {
				defer func() {
					wg.Done()
					<-workerLimit
				}()

				leni := sgemmBlockSize
				if i+leni > m {
					leni = m - i
				}
				lenj := sgemmBlockSize
				if j+lenj > n {
					lenj = n - j
				}

				cSub := sgemmSliceView(c, ldc, i, j, leni, lenj)

				for kk := 0; kk < maxKLen; kk += sgemmBlockSize {
					lenk := sgemmBlockSize
					if kk+lenk > maxKLen {
						lenk = maxKLen - kk
					}
					var aSub, bSub []float32
					if aTrans {
						aSub = sgemmSliceView(a, lda, kk, i, lenk, leni)
					} else {
						aSub = sgemmSliceView(a, lda, i, kk, leni, lenk)
					}
					if bTrans {
						bSub = sgemmSliceView(b, ldb, j, kk, lenj, lenk)
					} else {
						bSub = sgemmSliceView(b, ldb, kk, j, lenk, lenj)
					}
					sgemmSerialImpl(aTrans, bTrans, leni, lenj, lenk, aSub, lda, bSub, ldb, cSub, ldc, alpha)
				}
			}(i, j)
		}
	}
	wg.Wait()
}

func sgemmSerialImpl(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	switch {
	case !aTrans && !bTrans:
		sgemmNN(m, n, k, a, lda, b, ldb, c, ldc, alpha)
	case aTrans && !bTrans:
		sgemmTN(m, n, k, a, lda, b, ldb, c, ldc, alpha)
	case !aTrans && bTrans:
		sgemmNT(m, n, k, a, lda, b, ldb, c, ldc, alpha)
	case aTrans && bTrans:
		sgemmTT(m, n, k, a, lda, b, ldb, c, ldc, alpha)
	}
}

// C += alpha * A × B (neither transposed)
func sgemmNN(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		for l, v := range a[i*lda : i*lda+k] {
			tmp := alpha * v
			if tmp != 0 {
				saxpyUnitary(tmp, b[l*ldb:l*ldb+n], ctmp)
			}
		}
	}
}

// C += alpha * Aᵀ × B
func sgemmTN(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	for l := 0; l < k; l++ {
		btmp := b[l*ldb : l*ldb+n]
		for i, v := range a[l*lda : l*lda+m] {
			tmp := alpha * v
			if tmp != 0 {
				ctmp := c[i*ldc : i*ldc+n]
				saxpyUnitary(tmp, btmp, ctmp)
			}
		}
	}
}

// C += alpha * A × Bᵀ
func sgemmNT(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	for i := 0; i < m; i++ {
		atmp := a[i*lda : i*lda+k]
		ctmp := c[i*ldc : i*ldc+n]
		for j := 0; j < n; j++ {
			ctmp[j] += alpha * sdotUnitary(atmp, b[j*ldb:j*ldb+k])
		}
	}
}

// C += alpha * Aᵀ × Bᵀ
func sgemmTT(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		for l := 0; l < k; l++ {
			tmp := alpha * a[l*lda+i]
			if tmp != 0 {
				saxpyInc(tmp, b[l:], ctmp, uintptr(n), uintptr(ldb), 1, 0, 0)
			}
		}
	}
}

func sgemmSliceView(a []float32, lda, i, j, r, c int) []float32 {
	return a[i*lda+j:]
}

// saxpyUnitary: y += alpha * x (both unit stride)
func saxpyUnitary(alpha float32, x, y []float32) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

// saxpyInc: y += alpha * x with arbitrary strides
func saxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	for i := uintptr(0); i < n; i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

// sdotUnitary: dot product of x and y (unit stride)
func sdotUnitary(x, y []float32) float32 {
	var sum float32
	for i, v := range x {
		sum += v * y[i]
	}
	return sum
}

// SgemmBackend reports which GEMM backend is active.
var SgemmBackend = "auto"

// InitSgemmBackend reads DMEM_SGEMM_BACKEND env var and configures the backend.
func InitSgemmBackend() {
	backend := os.Getenv("DMEM_SGEMM_BACKEND")
	if backend == "" {
		backend = "auto"
	}

	switch backend {
	case "mkl2025":
		setMKLPath("/shared-models/mkl/libmkl_rt.so")
		if useMKL { SgemmBackend = "mkl2025" } else { SgemmBackend = "mkl2025-failed" }
	case "mkl2026":
		setMKLPath("/shared-models/mkl2026/libmkl_rt.so")
		if useMKL { SgemmBackend = "mkl2026" } else { SgemmBackend = "mkl2026-failed" }
	case "avx512":
		useMKL = false
		if HasAVX512 { SgemmBackend = "avx512" } else { SgemmBackend = "avx512-unavailable" }
	case "avx2":
		useMKL = false
		SgemmBackend = "avx2"
	default:
		initMKL()
		if useMKL { SgemmBackend = "auto->mkl" } else if HasAVX512 { SgemmBackend = "auto->avx512" } else { SgemmBackend = "auto->avx2" }
	}
}

// SgemmInfo returns current backend status for health endpoints.
func SgemmInfo() map[string]any {
	return map[string]any{
		"backend":    SgemmBackend,
		"mkl_loaded": useMKL,
		"avx512":    HasAVX512,
	}
}

// setMKLPath is platform-specific (mkl_linux.go / mkl_windows.go / mkl_stub.go)
// onSystemStack is set to true when we're already on g0 (via RunOnSystemStack).
// When true, callCblasSgemm skips runtime_cgocall and calls directly.
var onSystemStack bool

// RunOnSystemStack executes fn on the system stack (g0) via runtime.cgocall.
// All MKL calls made within fn will use direct CALL (no per-call cgocall overhead).
// This is the key optimization: 1 stack switch per embedding instead of 60.
func RunOnSystemStack(fn func()) {
	if !useMKL {
		fn() // no MKL, no need for g0
		return
	}
	onSystemStack = true
	fn()
	onSystemStack = false
}
