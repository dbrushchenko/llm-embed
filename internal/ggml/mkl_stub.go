//go:build (!linux && !windows) || !amd64

// MKL is only available on Linux amd64. Stubs for other platforms.
package ggml

import "unsafe"

// MKL constants (needed by Sgemm even when MKL unavailable)
const (
	CblasRowMajor = 101
	CblasNoTrans  = 111
	CblasTrans    = 112
)

// MKLAvailable always returns false on non-Linux platforms.
func MKLAvailable() bool { return false }

// initMKL is a no-op on non-Linux.
func initMKL() {}

// useMKL is always false on non-Linux.
var useMKL = false
var mklSgemmPtr uintptr

// callCblasSgemm is a no-op stub on non-Linux (never called because useMKL=false).
func callCblasSgemm(fn uintptr, order, transA, transB, m, n, k int32, alpha float32, a unsafe.Pointer, lda int32, b unsafe.Pointer, ldb int32, beta float32, c unsafe.Pointer, ldc int32) {
}


func setMKLPath(path string) {}
