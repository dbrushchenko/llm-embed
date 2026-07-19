//go:build linux && amd64

#include "textflag.h"

// func dlopen_raw(path *byte, mode int32) uintptr
TEXT ·dlopen_raw(SB), NOSPLIT, $0-24
	MOVQ path+0(FP), DI
	MOVL mode+8(FP), SI
	CALL purego_dlopen(SB)
	MOVQ AX, ret+16(FP)
	RET

// func dlsym_raw(handle uintptr, name *byte) uintptr
TEXT ·dlsym_raw(SB), NOSPLIT, $0-24
	MOVQ handle+0(FP), DI
	MOVQ name+8(FP), SI
	CALL purego_dlsym(SB)
	MOVQ AX, ret+16(FP)
	RET

// func callCblasSgemmDirect(args unsafe.Pointer)
// Direct call to cblas_sgemm via C ABI. 64KB frame for MKL dispatcher stack.
// MKL threading uses OS pthreads (own stacks), so goroutine stack only needs
// enough for the dispatch code.
//
// sgemmCallArgs layout (all uintptr = 8 bytes):
//   0: fn, 8: order, 16: transA, 24: transB, 32: m, 40: n, 48: k
//   56: alpha (float32 bits), 64: a, 72: lda, 80: b, 88: ldb
//   96: beta (float32 bits), 104: c, 112: ldc
//
TEXT ·callCblasSgemmDirect(SB), $65584-8
	MOVQ args+0(FP), R11

	// Load function pointer
	MOVQ 0(R11), R10

	// Float args into XMM (separate sequence from integers)
	MOVL 56(R11), AX
	MOVD AX, X0               // XMM0 = alpha
	MOVL 96(R11), AX
	MOVD AX, X1               // XMM1 = beta

	// Integer args 1-6 into registers
	MOVQ 8(R11), DI           // Order
	MOVQ 16(R11), SI          // TransA
	MOVQ 24(R11), DX          // TransB
	MOVQ 32(R11), CX          // M
	MOVQ 40(R11), R8          // N
	MOVQ 48(R11), R9          // K

	// Integer args 7-12 onto stack
	MOVQ 64(R11), AX
	MOVQ AX, 0(SP)            // A pointer
	MOVQ 72(R11), AX
	MOVQ AX, 8(SP)            // lda
	MOVQ 80(R11), AX
	MOVQ AX, 16(SP)           // B pointer
	MOVQ 88(R11), AX
	MOVQ AX, 24(SP)           // ldb
	MOVQ 104(R11), AX
	MOVQ AX, 32(SP)           // C pointer
	MOVQ 112(R11), AX
	MOVQ AX, 40(SP)           // ldc

	MOVB $2, AX               // AL = number of XMM args used
	CALL R10

	RET
