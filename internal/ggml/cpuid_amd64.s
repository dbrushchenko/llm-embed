//go:build amd64

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv(index uint32) uint64
TEXT ·xgetbv(SB), NOSPLIT, $0-12
	MOVL index+0(FP), CX
	XGETBV
	// Result in EDX:EAX
	MOVL AX, R8
	MOVL DX, R9
	SHLQ $32, R9
	ORQ  R8, R9
	MOVQ R9, ret+8(FP)
	RET
