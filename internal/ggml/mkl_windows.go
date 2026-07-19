//go:build windows

// MKL acceleration on Windows via syscall.LoadLibrary (native Go, no CGO).
package ggml

import (
	"sync"
	"syscall"
	"unsafe"
)

var mklOnce sync.Once
var useMKL bool
var mklSgemmPtr uintptr
var mklSgemmFortranPtr uintptr

// MKL constants
const (
	CblasRowMajor = 101
	CblasNoTrans  = 111
	CblasTrans    = 112
)

// MKL search paths on Windows
var mklPathsWin = []string{
	`C:\Program Files (x86)\Intel\oneAPI\mkl\2026.1\bin\mkl_rt.3.dll`,
	`C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\mkl_rt.3.dll`,
	`mkl_rt.3.dll`, // PATH
	`mkl_rt.dll`,
}

func initMKL() {
	mklOnce.Do(func() {
		for _, path := range mklPathsWin {
			dll, err := syscall.LoadLibrary(path)
			if err != nil {
				continue
			}
			proc, err := syscall.GetProcAddress(dll, "cblas_sgemm")
			if err != nil {
				syscall.FreeLibrary(dll)
				continue
			}
			mklSgemmPtr = proc
			// Also get Fortran interface (all-pointer args, no float ABI issue)
			fortran, err := syscall.GetProcAddress(dll, "sgemm_")
			if err != nil {
				// Try alternative name
				fortran, _ = syscall.GetProcAddress(dll, "SGEMM")
			}
			mklSgemmFortranPtr = fortran
			useMKL = fortran != 0
			return
		}
	})
}

// MKLAvailable returns whether MKL was loaded.
func MKLAvailable() bool {
	initMKL()
	return useMKL
}

// callCblasSgemm calls MKL's sgemm_ (Fortran interface) via syscall on Windows.
// Fortran interface takes ALL args as pointers — avoids float-in-register ABI issue.
func callCblasSgemm(
	fn uintptr,
	order, transA, transB int32,
	m, n, k int32,
	alpha float32,
	a unsafe.Pointer, lda int32,
	b unsafe.Pointer, ldb int32,
	beta float32,
	c unsafe.Pointer, ldc int32,
) {
	// Use Fortran-style sgemm_ (all pointer args)
	// Pin all args in a struct to prevent GC/stack movement between address-taking and syscall
	type sgemmArgs struct {
		transA, transB byte
		m, n, k        int32
		alpha, beta    float32
		lda, ldb, ldc  int32
	}
	args := sgemmArgs{
		transA: 'N',
		transB: 'T',
		m:      n,   // Fortran column-major: swap M↔N, A↔B
		n:      m,
		k:      k,
		alpha:  alpha,
		beta:   beta,
		lda:    ldb,  // B becomes A in Fortran order
		ldb:    lda,  // A becomes B in Fortran order
		ldc:    ldc,
	}
	if transA == CblasTrans {
		args.transA = 'T'
	}
	if transB == CblasNoTrans {
		args.transB = 'N'
	}

	syscall.SyscallN(mklSgemmFortranPtr,
		uintptr(unsafe.Pointer(&args.transB)),
		uintptr(unsafe.Pointer(&args.transA)),
		uintptr(unsafe.Pointer(&args.m)),
		uintptr(unsafe.Pointer(&args.n)),
		uintptr(unsafe.Pointer(&args.k)),
		uintptr(unsafe.Pointer(&args.alpha)),
		uintptr(b),               // B is first matrix in Fortran order
		uintptr(unsafe.Pointer(&args.lda)),
		uintptr(a),               // A is second matrix in Fortran order
		uintptr(unsafe.Pointer(&args.ldb)),
		uintptr(unsafe.Pointer(&args.beta)),
		uintptr(c),
		uintptr(unsafe.Pointer(&args.ldc)),
	)
}

func setMKLPath(path string) {
	dll, err := syscall.LoadLibrary(path)
	if err != nil { return }
	proc, err := syscall.GetProcAddress(dll, "sgemm_")
	if err != nil { return }
	mklSgemmFortranPtr = proc
	mklSgemmPtr = proc
	useMKL = true
}