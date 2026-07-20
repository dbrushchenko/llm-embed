//go:build linux && amd64

#include "textflag.h"

// dlopen_trampoline: called on g0 stack via runtime.asmcgocall.
// DI = *dlopenArgs{path uintptr, mode uintptr, ret uintptr}
TEXT ·dlopen_trampoline(SB), NOSPLIT, $0
	MOVQ DI, BX            // save args struct pointer
	MOVQ 8(BX), SI         // SI = mode (2nd C arg)
	MOVQ 0(BX), DI         // DI = path (1st C arg)
	CALL dlopen_sym(SB)
	MOVQ AX, 16(BX)       // store return value in args.ret
	RET

// dlsym_trampoline: called on g0 stack via runtime.asmcgocall.
// DI = *dlsymArgs{handle uintptr, name uintptr, ret uintptr}
TEXT ·dlsym_trampoline(SB), NOSPLIT, $0
	MOVQ DI, BX            // save args struct pointer
	MOVQ 8(BX), SI         // SI = name (2nd C arg)
	MOVQ 0(BX), DI         // DI = handle (1st C arg)
	CALL dlsym_sym(SB)
	MOVQ AX, 16(BX)       // store return value in args.ret
	RET

// callCblasSgemmDirect: called directly (64KB frame, proven to work).
// MKL's sgemm uses pthreads internally (doesn't recurse on our stack).
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
