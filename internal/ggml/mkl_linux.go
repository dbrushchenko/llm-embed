//go:build linux && amd64

// MKL acceleration for Sgemm — optional runtime accelerator.
// Loads libmkl_rt.so via dlopen at init time (no CGO needed).
// Falls back to pure Go Sgemm if MKL is not available.
//
// Uses Go's //go:cgo_import_dynamic to link dlopen/dlsym from libdl.so.2
// and a minimal assembly trampoline to call cblas_sgemm via C ABI.
//
// Adapted from ebitengine/purego (Apache-2.0) — minimal subset for our single function.
package ggml

import (
	"sync"
	"unsafe"
)

// MKL constants (from cblas.h)
const (
	CblasRowMajor = 101
	CblasNoTrans  = 111
	CblasTrans    = 112
)

// useMKL is true if libmkl_rt.so was loaded successfully.
var useMKL bool

// mklSgemmPtr holds the resolved cblas_sgemm function pointer.
var mklSgemmPtr uintptr

// MKL search paths (shared-models PVC, system locations)
// Priority: 2025.0 first (tested), then 2026.1 (newer), then system
var mklPaths = []string{
	"/shared-models/mkl/libmkl_rt.so",       // MKL 2025.0 on PVC
	"/shared-models/mkl2026/libmkl_rt.so",   // MKL 2026.1 on PVC
	"/shared-models/libmkl_rt.so",           // top-level symlink
	"/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so", // node-installed
	"/usr/lib/x86_64-linux-gnu/libmkl_rt.so",
	"libmkl_rt.so", // LD_LIBRARY_PATH
}

var mklOnce sync.Once

// initMKL attempts to load MKL at first use.
func initMKL() {
	mklOnce.Do(func() {
		for _, path := range mklPaths {
			handle := dlopen_wrapper(path)
			if handle == 0 {
				continue
			}
			ptr := dlsym_wrapper(handle, "cblas_sgemm")
			if ptr == 0 {
				continue
			}
			mklSgemmPtr = ptr
			useMKL = true
			return
		}
	})
}

// MKLAvailable returns whether MKL was loaded.
func MKLAvailable() bool {
	return useMKL
}


// dlopen_trampoline and dlsym_trampoline are called on g0 via asmcgocall.
func dlopen_trampoline()
func dlsym_trampoline()

// asmcgocall switches to g0 stack and calls fn(arg).
// Required for dlopen because glibc's ELF loader needs deep stack + TLS.
//
//go:linkname asmcgocall runtime.asmcgocall
//go:noescape
func asmcgocall(fn, arg unsafe.Pointer) int32

// dlopen args passed to trampoline via asmcgocall.
type dlopenArgs struct {
	path uintptr // *byte (null-terminated)
	mode uintptr
	ret  uintptr // return value
}

// dlsym args passed to trampoline via asmcgocall.
type dlsymArgs struct {
	handle uintptr
	name   uintptr // *byte (null-terminated)
	ret    uintptr // return value
}

//go:nosplit
func funcPC(fn func()) uintptr {
	return **(**uintptr)(unsafe.Pointer(&fn))
}

// dlopen_wrapper loads a shared library by path (switches to g0 stack).
func dlopen_wrapper(path string) uintptr {
	cpath := append([]byte(path), 0)
	args := dlopenArgs{
		path: uintptr(unsafe.Pointer(&cpath[0])),
		mode: 2, // RTLD_NOW
	}
	asmcgocall(unsafe.Pointer(funcPC(dlopen_trampoline)), unsafe.Pointer(&args))
	return args.ret
}

// dlsym_wrapper resolves a symbol from a library handle (switches to g0 stack).
func dlsym_wrapper(handle uintptr, name string) uintptr {
	cname := append([]byte(name), 0)
	args := dlsymArgs{
		handle: handle,
		name:   uintptr(unsafe.Pointer(&cname[0])),
	}
	asmcgocall(unsafe.Pointer(funcPC(dlsym_trampoline)), unsafe.Pointer(&args))
	return args.ret
}

// sgemmCallArgs is the packed argument struct for the assembly trampoline.
// All fields are uintptr-sized for simple, predictable layout (no padding issues).
type sgemmCallArgs struct {
	fn     uintptr
	order  uintptr
	transA uintptr
	transB uintptr
	m      uintptr
	n      uintptr
	k      uintptr
	alpha  uintptr // float32 bits stored in low 32 bits
	a      uintptr // pointer
	lda    uintptr
	b      uintptr // pointer
	ldb    uintptr
	beta   uintptr // float32 bits stored in low 32 bits
	c      uintptr // pointer
	ldc    uintptr
}

// callCblasSgemm calls MKL's cblas_sgemm via direct assembly trampoline.
// Safe because: MKL threading uses OS pthreads (not goroutine stack) and
// the calling frame is 64KB (enough for MKL's dispatcher code).
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
	args := sgemmCallArgs{
		fn:     fn,
		order:  uintptr(order),
		transA: uintptr(transA),
		transB: uintptr(transB),
		m:      uintptr(m),
		n:      uintptr(n),
		k:      uintptr(k),
		alpha:  uintptr(*(*uint32)(unsafe.Pointer(&alpha))),
		a:      uintptr(a),
		lda:    uintptr(lda),
		b:      uintptr(b),
		ldb:    uintptr(ldb),
		beta:   uintptr(*(*uint32)(unsafe.Pointer(&beta))),
		c:      uintptr(c),
		ldc:    uintptr(ldc),
	}
	callCblasSgemmDirect(unsafe.Pointer(&args))
}

// callCblasSgemmDirect calls the trampoline directly (when already on g0).
// This is just a Go wrapper that calls the assembly.
func callCblasSgemmDirect(args unsafe.Pointer)

func setMKLPath(path string) {
	handle := dlopen_wrapper(path)
	if handle == 0 { return }
	ptr := dlsym_wrapper(handle, "cblas_sgemm")
	if ptr == 0 { return }
	mklSgemmPtr = ptr
	useMKL = true
}