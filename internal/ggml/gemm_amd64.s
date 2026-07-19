//go:build amd64

#include "textflag.h"

// func gemmMicroKernel6x8AVX2(
//     a *float32,    // packed A panel: MR×KC contiguous (6×KC)
//     b *float32,    // packed B panel: KC×NR contiguous (KC×8)
//     c *float32,    // output C tile: MR rows, stride=ldc floats (6 rows × ldc stride)
//     KC int,        // inner dimension loop count
//     ldc int,       // C row stride in floats
// )
//
// Computes: C[6×8] += A[6×KC] × B[KC×8]
//
// Register allocation:
//   Y0-Y5:  6 accumulators (one per row of C)
//   Y6:     B[k][0:8] — current B row
//   Y7:     broadcast of A[m][k]
//   SI:     pointer to A (advances by 4 bytes per k, wraps every 6 rows)
//   DI:     pointer to B (advances by 32 bytes per k)
//   R8:     pointer to C
//   CX:     KC loop counter
//   R9:     ldc * 4 (byte stride for C rows)
//
TEXT ·gemmMicroKernel6x8AVX2(SB), NOSPLIT, $0-40
	MOVQ a+0(FP), SI        // A pointer
	MOVQ b+8(FP), DI        // B pointer
	MOVQ c+16(FP), R8       // C pointer
	MOVQ KC+24(FP), CX      // loop count
	MOVQ ldc+32(FP), R9     // C stride in floats
	SHLQ $2, R9             // R9 = ldc * 4 (byte stride)

	// Zero accumulators
	VXORPS Y0, Y0, Y0       // C[0][0:8]
	VXORPS Y1, Y1, Y1       // C[1][0:8]
	VXORPS Y2, Y2, Y2       // C[2][0:8]
	VXORPS Y3, Y3, Y3       // C[3][0:8]
	VXORPS Y4, Y4, Y4       // C[4][0:8]
	VXORPS Y5, Y5, Y5       // C[5][0:8]

	// Check KC > 0
	TESTQ CX, CX
	JZ store

	// Main loop: for k = 0..KC-1
loop:
	// Load B[k][0:8] — 8 floats from packed B
	VMOVUPS (DI), Y6         // Y6 = B[k][0:8]
	ADDQ $32, DI             // advance B pointer by 8 floats

	// Row 0: C[0] += A[0][k] * B[k]
	VBROADCASTSS (SI), Y7
	VFMADD231PS Y7, Y6, Y0

	// Row 1: C[1] += A[1][k] * B[k]
	VBROADCASTSS 4(SI), Y7
	VFMADD231PS Y7, Y6, Y1

	// Row 2: C[2] += A[2][k] * B[k]
	VBROADCASTSS 8(SI), Y7
	VFMADD231PS Y7, Y6, Y2

	// Row 3: C[3] += A[3][k] * B[k]
	VBROADCASTSS 12(SI), Y7
	VFMADD231PS Y7, Y6, Y3

	// Row 4: C[4] += A[4][k] * B[k]
	VBROADCASTSS 16(SI), Y7
	VFMADD231PS Y7, Y6, Y4

	// Row 5: C[5] += A[5][k] * B[k]
	VBROADCASTSS 20(SI), Y7
	VFMADD231PS Y7, Y6, Y5

	// Advance A pointer by MR=6 floats (24 bytes)
	ADDQ $24, SI

	DECQ CX
	JNZ loop

store:
	// Store accumulators to C (additive: C += result)
	// Row 0
	VADDPS (R8), Y0, Y0
	VMOVUPS Y0, (R8)
	// Row 1
	LEAQ (R8)(R9*1), R10
	VADDPS (R10), Y1, Y1
	VMOVUPS Y1, (R10)
	// Row 2
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y2, Y2
	VMOVUPS Y2, (R10)
	// Row 3
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y3, Y3
	VMOVUPS Y3, (R10)
	// Row 4
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y4, Y4
	VMOVUPS Y4, (R10)
	// Row 5
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y5, Y5
	VMOVUPS Y5, (R10)

	VZEROUPPER
	RET

// func packBPanelAVX2(
//     b *float32,    // start of B[jStart][0] — first of 8 rows
//     packB *float32, // output: K groups of 8 contiguous floats
//     K int,         // inner dimension
//     stride int,    // row stride in floats (= K for our layout)
// )
//
// Packs 8 rows of B (each K floats, stride apart) into panel format.
// For each k: packB[k*8:k*8+8] = B[row0][k], B[row1][k], ..., B[row7][k]
//
// Uses VGATHERDPS to load 8 floats from 8 different addresses in one instruction.
//
TEXT ·packBPanelAVX2(SB), NOSPLIT, $0-32
	MOVQ b+0(FP), SI         // B base pointer (first row, column 0)
	MOVQ packB+8(FP), DI     // output pointer
	MOVQ K+16(FP), CX        // loop count
	MOVQ stride+24(FP), R8   // stride in floats
	SHLQ $2, R8              // R8 = stride in bytes

	// Build index vector for VGATHERDPS: [0, stride, 2*stride, ..., 7*stride] in bytes
	// We'll use scalar loads instead — VGATHERDPS is slow on many CPUs (2+ cycles/element)
	// 8 scalar loads + 1 store is actually faster on Haswell/Broadwell/Skylake

	// Precompute row pointers
	MOVQ SI, R9              // row0
	LEAQ (SI)(R8*1), R10     // row1
	LEAQ (R10)(R8*1), R11    // row2
	LEAQ (R11)(R8*1), R12    // row3
	LEAQ (R12)(R8*1), R13    // row4
	LEAQ (R13)(R8*1), R14    // row5
	LEAQ (R14)(R8*1), R15    // row6
	LEAQ (R15)(R8*1), BX     // row7

	TESTQ CX, CX
	JZ packb_done

packb_loop:
	// Load one float from each of 8 rows at current k position
	// Build a YMM register from 8 scattered floats
	VMOVSS (R9), X0
	VINSERTPS $0x10, (R10), X0, X0
	VINSERTPS $0x20, (R11), X0, X0
	VINSERTPS $0x30, (R12), X0, X0
	VMOVSS (R13), X1
	VINSERTPS $0x10, (R14), X1, X1
	VINSERTPS $0x20, (R15), X1, X1
	VINSERTPS $0x30, (BX), X1, X1
	VINSERTI128 $1, X1, Y0, Y0

	// Store 8 floats contiguously
	VMOVUPS Y0, (DI)

	// Advance all row pointers by 4 bytes (1 float)
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11
	ADDQ $4, R12
	ADDQ $4, R13
	ADDQ $4, R14
	ADDQ $4, R15
	ADDQ $4, BX
	ADDQ $32, DI            // packB advances by 8 floats (32 bytes)

	DECQ CX
	JNZ packb_loop

packb_done:
	VZEROUPPER
	RET

// func gemmMicroKernel6x16AVX2(
//     a *float32,    // packed A panel: 6×KC (k-major: for each k, 6 values)
//     b *float32,    // packed B panel: KC×16 (k-major: for each k, 16 values)
//     c *float32,    // output C tile: 6 rows, stride=ldc floats
//     KC int,        // inner dimension loop count
//     ldc int,       // C row stride in floats
// )
//
// Computes: C[6×16] += A[6×KC] × B[KC×16]
// Uses ALL 16 YMM registers:
//   Y0-Y5:   accumulators for columns 0-7 (6 rows)
//   Y6-Y11:  accumulators for columns 8-15 (6 rows)
//   Y12:     B[k][0:8]
//   Y13:     B[k][8:16]
//   Y14:     broadcast A[m][k]
//   Y15:     (unused/spare)
//
TEXT ·gemmMicroKernel6x16AVX2(SB), NOSPLIT, $0-40
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ c+16(FP), R8
	MOVQ KC+24(FP), CX
	MOVQ ldc+32(FP), R9
	SHLQ $2, R9              // byte stride

	// Zero all 12 accumulators
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

	TESTQ CX, CX
	JZ store16

	// K=2 unrolled loop: process 2 k-values per iteration (OpenBLAS KERNEL_k2 pattern)
	// Halves branch overhead, allows CPU to overlap k+1 loads with k FMAs
	MOVQ CX, DX
	SHRQ $1, DX              // DX = KC/2 (number of k2 iterations)
	ANDQ $1, CX             // CX = KC%2 (remainder)

	TESTQ DX, DX
	JZ loop16_remainder

loop16_k2:
	// Prefetch
	PREFETCHT0 512(SI)
	PREFETCHT0 128(DI)

	// === k+0 ===
	VMOVUPS (DI), Y12         // B[k][0:8]
	VMOVUPS 32(DI), Y13       // B[k][8:16]

	VBROADCASTSS (SI), Y14
	VFMADD231PS Y14, Y12, Y0
	VFMADD231PS Y14, Y13, Y6

	VBROADCASTSS 4(SI), Y14
	VFMADD231PS Y14, Y12, Y1
	VFMADD231PS Y14, Y13, Y7

	VBROADCASTSS 8(SI), Y14
	VFMADD231PS Y14, Y12, Y2
	VFMADD231PS Y14, Y13, Y8

	VBROADCASTSS 12(SI), Y14
	VFMADD231PS Y14, Y12, Y3
	VFMADD231PS Y14, Y13, Y9

	VBROADCASTSS 16(SI), Y14
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y14, Y13, Y10

	VBROADCASTSS 20(SI), Y14
	VFMADD231PS Y14, Y12, Y5
	VFMADD231PS Y14, Y13, Y11

	// === k+1 ===
	VMOVUPS 64(DI), Y12       // B[k+1][0:8]
	VMOVUPS 96(DI), Y13       // B[k+1][8:16]

	VBROADCASTSS 24(SI), Y14
	VFMADD231PS Y14, Y12, Y0
	VFMADD231PS Y14, Y13, Y6

	VBROADCASTSS 28(SI), Y14
	VFMADD231PS Y14, Y12, Y1
	VFMADD231PS Y14, Y13, Y7

	VBROADCASTSS 32(SI), Y14
	VFMADD231PS Y14, Y12, Y2
	VFMADD231PS Y14, Y13, Y8

	VBROADCASTSS 36(SI), Y14
	VFMADD231PS Y14, Y12, Y3
	VFMADD231PS Y14, Y13, Y9

	VBROADCASTSS 40(SI), Y14
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y14, Y13, Y10

	VBROADCASTSS 44(SI), Y14
	VFMADD231PS Y14, Y12, Y5
	VFMADD231PS Y14, Y13, Y11

	// Advance pointers by 2 iterations
	ADDQ $128, DI             // B: 2 × 16 floats × 4 bytes = 128
	ADDQ $48, SI              // A: 2 × 6 floats × 4 bytes = 48

	DECQ DX
	JNZ loop16_k2

loop16_remainder:
	// Handle odd KC (0 or 1 remaining iteration)
	TESTQ CX, CX
	JZ store16

	// Single k iteration
	VMOVUPS (DI), Y12
	VMOVUPS 32(DI), Y13
	ADDQ $64, DI

	VBROADCASTSS (SI), Y14
	VFMADD231PS Y14, Y12, Y0
	VFMADD231PS Y14, Y13, Y6

	VBROADCASTSS 4(SI), Y14
	VFMADD231PS Y14, Y12, Y1
	VFMADD231PS Y14, Y13, Y7

	VBROADCASTSS 8(SI), Y14
	VFMADD231PS Y14, Y12, Y2
	VFMADD231PS Y14, Y13, Y8

	VBROADCASTSS 12(SI), Y14
	VFMADD231PS Y14, Y12, Y3
	VFMADD231PS Y14, Y13, Y9

	VBROADCASTSS 16(SI), Y14
	VFMADD231PS Y14, Y12, Y4
	VFMADD231PS Y14, Y13, Y10

	VBROADCASTSS 20(SI), Y14
	VFMADD231PS Y14, Y12, Y5
	VFMADD231PS Y14, Y13, Y11

	ADDQ $24, SI

store16:
	// Store 6 rows × 16 cols (two 8-wide stores per row)
	// Row 0
	VADDPS (R8), Y0, Y0
	VMOVUPS Y0, (R8)
	VADDPS 32(R8), Y6, Y6
	VMOVUPS Y6, 32(R8)
	// Row 1
	LEAQ (R8)(R9*1), R10
	VADDPS (R10), Y1, Y1
	VMOVUPS Y1, (R10)
	VADDPS 32(R10), Y7, Y7
	VMOVUPS Y7, 32(R10)
	// Row 2
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y2, Y2
	VMOVUPS Y2, (R10)
	VADDPS 32(R10), Y8, Y8
	VMOVUPS Y8, 32(R10)
	// Row 3
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y3, Y3
	VMOVUPS Y3, (R10)
	VADDPS 32(R10), Y9, Y9
	VMOVUPS Y9, 32(R10)
	// Row 4
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y4, Y4
	VMOVUPS Y4, (R10)
	VADDPS 32(R10), Y10, Y10
	VMOVUPS Y10, 32(R10)
	// Row 5
	LEAQ (R10)(R9*1), R10
	VADDPS (R10), Y5, Y5
	VMOVUPS Y5, (R10)
	VADDPS 32(R10), Y11, Y11
	VMOVUPS Y11, 32(R10)

	VZEROUPPER
	RET

// func packAPanelAVX2(
//     r0, r1, r2, r3, r4, r5 *float32,  // 6 row pointers
//     dst *float32,                       // output: K groups of 6 floats
//     K int,                              // number of columns
// )
//
// Packs 6 rows into panel format: for each k, stores [r0[k], r1[k], r2[k], r3[k], r4[k], r5[k]]
// Uses scalar loads (6 scattered pointers can't use SIMD gather effectively)
// but unrolls 4 iterations of k for reduced loop overhead.
//
TEXT ·packAPanelAVX2(SB), NOSPLIT, $0-64
	MOVQ r0+0(FP), R8
	MOVQ r1+8(FP), R9
	MOVQ r2+16(FP), R10
	MOVQ r3+24(FP), R11
	MOVQ r4+32(FP), R12
	MOVQ r5+40(FP), R13
	MOVQ dst+48(FP), DI
	MOVQ K+56(FP), CX

	// Process 4 k-values per iteration (unrolled)
	MOVQ CX, DX
	SHRQ $2, DX              // DX = K/4
	ANDQ $3, CX             // CX = K%4 remainder

	TESTQ DX, DX
	JZ packa_remainder

packa_loop4:
	// k+0
	MOVSS (R8), X0
	MOVSS (R9), X1
	MOVSS (R10), X2
	MOVSS (R11), X3
	MOVSS (R12), X4
	MOVSS (R13), X5
	MOVSS X0, (DI)
	MOVSS X1, 4(DI)
	MOVSS X2, 8(DI)
	MOVSS X3, 12(DI)
	MOVSS X4, 16(DI)
	MOVSS X5, 20(DI)

	// k+1
	MOVSS 4(R8), X0
	MOVSS 4(R9), X1
	MOVSS 4(R10), X2
	MOVSS 4(R11), X3
	MOVSS 4(R12), X4
	MOVSS 4(R13), X5
	MOVSS X0, 24(DI)
	MOVSS X1, 28(DI)
	MOVSS X2, 32(DI)
	MOVSS X3, 36(DI)
	MOVSS X4, 40(DI)
	MOVSS X5, 44(DI)

	// k+2
	MOVSS 8(R8), X0
	MOVSS 8(R9), X1
	MOVSS 8(R10), X2
	MOVSS 8(R11), X3
	MOVSS 8(R12), X4
	MOVSS 8(R13), X5
	MOVSS X0, 48(DI)
	MOVSS X1, 52(DI)
	MOVSS X2, 56(DI)
	MOVSS X3, 60(DI)
	MOVSS X4, 64(DI)
	MOVSS X5, 68(DI)

	// k+3
	MOVSS 12(R8), X0
	MOVSS 12(R9), X1
	MOVSS 12(R10), X2
	MOVSS 12(R11), X3
	MOVSS 12(R12), X4
	MOVSS 12(R13), X5
	MOVSS X0, 72(DI)
	MOVSS X1, 76(DI)
	MOVSS X2, 80(DI)
	MOVSS X3, 84(DI)
	MOVSS X4, 88(DI)
	MOVSS X5, 92(DI)

	// Advance pointers
	ADDQ $16, R8             // 4 floats
	ADDQ $16, R9
	ADDQ $16, R10
	ADDQ $16, R11
	ADDQ $16, R12
	ADDQ $16, R13
	ADDQ $96, DI             // 4 * 6 floats = 24 floats = 96 bytes

	DECQ DX
	JNZ packa_loop4

packa_remainder:
	TESTQ CX, CX
	JZ packa_done

packa_rem_loop:
	MOVSS (R8), X0
	MOVSS (R9), X1
	MOVSS (R10), X2
	MOVSS (R11), X3
	MOVSS (R12), X4
	MOVSS (R13), X5
	MOVSS X0, (DI)
	MOVSS X1, 4(DI)
	MOVSS X2, 8(DI)
	MOVSS X3, 12(DI)
	MOVSS X4, 16(DI)
	MOVSS X5, 20(DI)
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11
	ADDQ $4, R12
	ADDQ $4, R13
	ADDQ $24, DI

	DECQ CX
	JNZ packa_rem_loop

packa_done:
	RET
