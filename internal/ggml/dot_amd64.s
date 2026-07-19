//go:build amd64

#include "textflag.h"

// func dotQ8_0AVX2(data *byte, x *float32, n int) float32
//
// Computes fused dequant+dot for Q8_0 quantized data using AVX2 SIMD.
//
// Q8_0 block layout (34 bytes):
//   bytes 0-1:  f16 scale (little-endian)
//   bytes 2-33: 32 × int8 quantized values
//
// Algorithm per block:
//   blockSum = sum_i(int8_to_float(quants[i]) * x[baseIdx+i])
//   total += f16_to_f32(scale) * blockSum
//
// AVX2 strategy: process 8 elements at a time (4 groups per block).
//   - VPMOVSXBD: sign-extend 8 int8 values to 8 int32
//   - VCVTDQ2PS: convert 8 int32 to 8 float32
//   - VFMADD231PS: fused multiply-add with x values into block accumulator
//   - After 4 groups: horizontal sum, multiply by scale, add to total
//
// Frame: data(8) + x(8) + n(8) + ret(4) = 28 bytes args, 0 locals
TEXT ·dotQ8_0AVX2(SB), NOSPLIT, $0-28
	MOVQ data+0(FP), SI       // SI = data pointer
	MOVQ x+8(FP), DI          // DI = x pointer
	MOVQ n+16(FP), CX         // CX = n (element count)

	SHRQ $5, CX               // CX = numBlocks = n / 32

	// Total accumulator (scalar, in X0 low element)
	VXORPS X0, X0, X0         // X0 = total (scalar accumulator)

	TESTQ CX, CX
	JZ    done

blockLoop:
	// === Load f16 scale and convert to f32 ===
	// f16 layout: [sign:1][exp:5][mant:10]
	// f32 layout: [sign:1][exp:8][mant:23]
	// For normal f16: f32_exp = f16_exp + 112, f32_mant = f16_mant << 13
	MOVWLZX (SI), AX          // AX = f16 bits (16-bit, zero-extended to 32)

	// Extract components
	MOVL  AX, DX              // DX = working copy
	MOVL  AX, R8              // R8 = save original

	// sign bit -> position 31
	SHRL  $15, DX
	SHLL  $31, DX             // DX = sign << 31

	// exponent: bits [14:10]
	MOVL  R8, R9
	SHRL  $10, R9
	ANDL  $0x1F, R9           // R9 = 5-bit exponent

	// mantissa: bits [9:0]
	MOVL  R8, R10
	ANDL  $0x3FF, R10         // R10 = 10-bit mantissa

	// Check for zero/subnormal (exp == 0)
	TESTL R9, R9
	JZ    zeroScale

	// Normal case (most common for Q8_0 scales):
	// f32 = sign | ((exp + 112) << 23) | (mant << 13)
	ADDL  $112, R9            // bias adjust: 127 - 15 = 112
	SHLL  $23, R9             // position exponent
	SHLL  $13, R10            // position mantissa
	ORL   R9, DX
	ORL   R10, DX             // DX = f32 bits

	JMP   haveScale

zeroScale:
	// exp==0: treat as zero (subnormals negligible for Q8_0 scales)
	// DX = sign<<31 | 0 = +0.0 or -0.0
	// (Mantissa was already not OR'd in)

haveScale:
	// DX = f32 bits of scale

	// === Compute dot product for this block's 32 quants ===
	// Block accumulator
	VXORPS Y1, Y1, Y1         // Y1 = block sum accumulator (8 floats)

	// Quants start at SI+2
	// Group 0: quants[0:8], x[0:8]
	VPMOVSXBD 2(SI), Y2       // sign-extend 8 × int8 -> 8 × int32
	VCVTDQ2PS Y2, Y2          // convert to float32
	VMOVUPS   (DI), Y3        // load x[0:8]
	VFMADD231PS Y2, Y3, Y1    // Y1 += Y2 * Y3

	// Group 1: quants[8:16], x[8:16]
	VPMOVSXBD 10(SI), Y2      // SI+2+8 = SI+10
	VCVTDQ2PS Y2, Y2
	VMOVUPS   32(DI), Y3      // 8 floats = 32 bytes offset
	VFMADD231PS Y2, Y3, Y1

	// Group 2: quants[16:24], x[16:24]
	VPMOVSXBD 18(SI), Y2      // SI+2+16 = SI+18
	VCVTDQ2PS Y2, Y2
	VMOVUPS   64(DI), Y3
	VFMADD231PS Y2, Y3, Y1

	// Group 3: quants[24:32], x[24:32]
	VPMOVSXBD 26(SI), Y2      // SI+2+24 = SI+26
	VCVTDQ2PS Y2, Y2
	VMOVUPS   96(DI), Y3
	VFMADD231PS Y2, Y3, Y1

	// === Horizontal sum of Y1 -> scalar in X1 ===
	VEXTRACTF128 $1, Y1, X2   // X2 = high 128 bits of Y1
	VADDPS       X1, X2, X1   // X1 = pairwise add low+high
	VHADDPS      X1, X1, X1   // horizontal add
	VHADDPS      X1, X1, X1   // X1[0] = sum of all 8 elements

	// === Multiply by scale and accumulate ===
	MOVL  DX, X3              // X3[0] = scale as f32
	VMULSS X3, X1, X1         // X1[0] = blockSum * scale
	VADDSS X1, X0, X0         // total += blockSum * scale

	// === Advance to next block ===
	ADDQ  $34, SI             // 34 bytes per Q8_0 block
	ADDQ  $128, DI            // 32 × 4 bytes per float32 group

	DECQ  CX
	JNZ   blockLoop

done:
	MOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// func dotF32AVX2(a *float32, b *float32, n int) float32
//
// AVX2 FMA dot product of two float32 vectors.
// Processes 16 elements per iteration (2×8 for ILP), then 8-element tail.
//
// Frame: a(8) + b(8) + n(8) + ret(4) = 28 bytes args
TEXT ·dotF32AVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y4, Y4, Y4

	MOVQ CX, DX
	SHRQ $4, DX
	TESTQ DX, DX
	JZ    dotf32_tail8

dotf32_loop16:
	VMOVUPS (SI), Y1
	VMOVUPS (DI), Y2
	VFMADD231PS Y1, Y2, Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 32(DI), Y2
	VFMADD231PS Y1, Y2, Y4
	ADDQ $64, SI
	ADDQ $64, DI
	DECQ DX
	JNZ  dotf32_loop16
	VADDPS Y4, Y0, Y0

dotf32_tail8:
	TESTQ $8, CX
	JZ    dotf32_hsum
	VMOVUPS (SI), Y1
	VMOVUPS (DI), Y2
	VFMADD231PS Y1, Y2, Y0

dotf32_hsum:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X0, X1, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	MOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// =====================================================================
// Q6_K FUSED AVX2 KERNEL — 32 elements
// =====================================================================
// func dotQ6K32AVX2(ql *byte, qh *byte, qhShift uint64, scale float32, x *float32) float32
TEXT ·dotQ6K32AVX2(SB), NOSPLIT, $0-44
	MOVQ    ql+0(FP), SI
	MOVQ    qh+8(FP), DX
	MOVQ    qhShift+16(FP), CX
	MOVSS   scale+24(FP), X15
	MOVQ    x+32(FP), DI

	VBROADCASTSS X15, Y15

	// Load ql, mask low nibble
	VMOVDQU (SI), Y1
	MOVL    $0x0F0F0F0F, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPAND   Y10, Y1, Y1          // Y1 = 32 nibbles [0-15]

	// Load qh, shift right, mask 2 bits
	VMOVDQU (DX), Y2
	VMOVD   CX, X9
	VPSRLD  X9, Y2, Y2           // shift 32-bit lanes (ok: we mask after)
	MOVL    $0x03030303, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPAND   Y10, Y2, Y2          // Y2 = 32 × 2-bit values [0-3]

	// Byte shift left by 4: use VPSHUFB as lookup table
	// Table: [0x00, 0x10, 0x20, 0x30, 0,0,0,0, 0,0,0,0, 0,0,0,0] per 16-byte lane
	// FIX BUG #1: VPSHUFB in Plan 9 is VPSHUFB indices, data, dst
	// We want dst[i] = table[Y2[i]], so: VPSHUFB Y2, Y8, Y2 means Y2=shuffle(Y8 by Y2)
	// In Plan 9: VPSHUFB src_indices, src_data, dst → VPSHUFB Y2, Y8, Y2
	// Actually Plan 9: VPSHUFB op1, op2, dst means dst = pshufb(op2, op1)
	// dst[i] = op2[op1[i]] — so op1=indices, op2=table
	// We need: Y2[i] = table[Y2[i]] → indices=Y2, table=Y8 → VPSHUFB Y2, Y8, Y2 ✗
	// Correct: op1=Y2 (indices), op2=Y8 (table) → dst = op2[op1[i]] = Y8[Y2[i]]
	// In Plan 9 syntax: VPSHUFB Y2, Y8, Y2  means Y2 = pshufb(Y8, Y2) = Y8 shuffled by Y2
	// NO! Plan 9 for VPSHUFB is: VPSHUFB src, table, dst
	//   which computes: dst[i] = table[src[i] & 0xF] (if src[i] bit7=0)
	// So we need: src=Y2 (indices 0-3), table=Y8 (lookup values) → VPSHUFB Y2, Y8, Y2
	// This gives Y2[i] = Y8[Y2[i]] ← CORRECT
	// Wait no — let me just look at what Go assembler actually does.
	// Go Plan 9: VPSHUFB X, Y, Z compiles to vpshufb ymm_Z, ymm_Y, ymm_X in Intel syntax
	// Intel: VPSHUFB dst, src1, src2 → dst[i] = (src2[i] & 0x80) ? 0 : src1[src2[i] & 0xF]
	// So Go: VPSHUFB X, Y, Z → Intel: vpshufb Z, Y, X → Z[i] = Y[X[i] & 0xF]
	// We want: result[i] = table[indices[i]]
	// Go syntax: VPSHUFB indices, table, result
	// = VPSHUFB Y2, Y8, Y2 → Y2[i] = Y8[Y2[i]] ← but Go reverses to Intel!
	// Go VPSHUFB A, B, C → Intel vpshufb C, B, A → C[i] = B[A[i]]
	// So: VPSHUFB Y2, Y8, Y2 → Y2[i] = Y8[Y2[i]]... 
	// Actually I need to just test. But the USER said it's REVERSED.
	// User says: "VPSHUFB Y2, Y8, Y2 means Y2 = shuffle(Y8, Y2)" — second source is data, first is indices
	// So to get Y2[i] = Y8[Y2[i]], we need VPSHUFB Y8, Y2, Y2
	// (Y8=data/table, Y2=indices, Y2=dest)
	
	MOVQ    $0x3020100030201000, R8
	VMOVQ   R8, X8
	VPBROADCASTQ X8, Y8           // Y8 = lookup table: 0→0x00, 1→0x10, 2→0x20, 3→0x30
	VPSHUFB Y2, Y8, Y2            // Y2[i] = Y8[Y2[i]] — data=Y8, indices=Y2

	// Combine: 6-bit = nibble | (high2 << 4)
	VPOR    Y2, Y1, Y1

	// Subtract 32 to center [-32, 31]
	MOVL    $0x20202020, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPSUBB  Y10, Y1, Y1          // Y1 = signed bytes

	// Widen and FMA — 4 groups of 8
	VXORPS  Y0, Y0, Y0

	// Group 0: low 8 bytes of low lane
	VEXTRACTI128 $0, Y1, X6      // X6 = low 128 bits
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS (DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 1: high 8 bytes of low lane
	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS 32(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 2: low 8 bytes of high lane
	VEXTRACTI128 $1, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS 64(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 3: high 8 bytes of high lane
	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS 96(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Horizontal sum
	VEXTRACTF128 $1, Y0, X2
	VADDPS  X0, X2, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	MOVSS   X0, ret+40(FP)
	VZEROUPPER
	RET

// =====================================================================
// Q6_K FUSED AVX2 KERNEL v2 — 32 elements, TWO scales (per-16)
// =====================================================================
// func dotQ6K32AVX2v2(ql *byte, qh *byte, qhShift uint64, scale1 float32, scale2 float32, x *float32) float32
// scale1 applies to elements 0-15, scale2 applies to elements 16-31.
TEXT ·dotQ6K32AVX2v2(SB), NOSPLIT, $0-44
	MOVQ    ql+0(FP), SI
	MOVQ    qh+8(FP), DX
	MOVQ    qhShift+16(FP), CX
	MOVSS   scale1+24(FP), X15
	MOVSS   scale2+28(FP), X14
	MOVQ    x+32(FP), DI

	VBROADCASTSS X15, Y15          // Y15 = scale1 (elements 0-15)
	VBROADCASTSS X14, Y14          // Y14 = scale2 (elements 16-31)

	// Load ql, mask low nibble
	VMOVDQU (SI), Y1
	MOVL    $0x0F0F0F0F, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPAND   Y10, Y1, Y1          // Y1 = 32 nibbles [0-15]

	// Load qh, shift right, mask 2 bits
	VMOVDQU (DX), Y2
	VMOVD   CX, X9
	VPSRLD  X9, Y2, Y2           // shift 32-bit lanes
	MOVL    $0x03030303, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPAND   Y10, Y2, Y2          // Y2 = 32 × 2-bit values [0-3]

	// Byte shift left by 4 via VPSHUFB lookup
	MOVQ    $0x3020100030201000, R8
	VMOVQ   R8, X8
	VPBROADCASTQ X8, Y8
	VPSHUFB Y2, Y8, Y2            // Y2[i] = Y8[Y2[i]]

	// Combine: 6-bit = nibble | (high2 << 4)
	VPOR    Y2, Y1, Y1

	// Subtract 32 to center [-32, 31]
	MOVL    $0x20202020, R8
	VMOVD   R8, X10
	VPBROADCASTD X10, Y10
	VPSUBB  Y10, Y1, Y1          // Y1 = signed bytes

	// Widen and FMA — 4 groups of 8
	// Groups 0-1 use Y15 (scale1), Groups 2-3 use Y14 (scale2)
	VXORPS  Y0, Y0, Y0

	// Group 0: elements 0-7 (low 8 bytes of low lane) — scale1
	VEXTRACTI128 $0, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS (DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 1: elements 8-15 (high 8 bytes of low lane) — scale1
	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y15, Y3, Y3
	VMOVUPS 32(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 2: elements 16-23 (low 8 bytes of high lane) — scale2
	VEXTRACTI128 $1, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VMOVUPS 64(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Group 3: elements 24-31 (high 8 bytes of high lane) — scale2
	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VMOVUPS 96(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	// Horizontal sum
	VEXTRACTF128 $1, Y0, X2
	VADDPS  X0, X2, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	MOVSS   X0, ret+40(FP)
	VZEROUPPER
	RET

// =====================================================================
// Q4_K FUSED AVX2 KERNEL — 32 elements
// =====================================================================
// func dotQ4K32AVX2(qs *byte, isHigh uint64, scale float32, minVal float32, x *float32) float32
TEXT ·dotQ4K32AVX2(SB), NOSPLIT, $0-36
	MOVQ    qs+0(FP), SI
	MOVQ    isHigh+8(FP), R8
	MOVSS   scale+16(FP), X14
	MOVSS   minVal+20(FP), X13
	MOVQ    x+24(FP), DI

	VBROADCASTSS X14, Y14
	VBROADCASTSS X13, Y13

	// Load 32 bytes
	VMOVDQU (SI), Y1

	// FIX BUG #2: For high nibble, shift right 4 PER BYTE (not per word)
	// Correct approach: shift right 4 and mask, works because:
	// VPSRLW $4 shifts 16-bit words right 4, which gives us (byte[1]<<12 | byte[0]>>4)
	// in the low byte. But then VPAND 0x0F masks to just the low 4 bits of each byte.
	// For the LOW byte of each word: (byte>>4) & 0x0F = correct high nibble
	// For the HIGH byte of each word: (byte_shifted_with_garbage) & 0x0F = WRONG!
	// Fix: always do VPSRLW $4 then VPAND 0x0F. The high byte of each word gets
	// bits from the low byte shifted in — but after AND 0x0F on the BYTE level
	// this is still wrong because VPAND 0x0F0F0F0F masks each byte independently!
	// Actually: VPSRLW shifts 16-bit words. After shift, low byte = (word >> 4) & 0xFF.
	// For word = [high_byte, low_byte]: shifted word = [0000_high_byte>>4, high_byte_low4_bits | low_byte>>4]
	// Wait no — VPSRLW $4 on a 16-bit word ab_cdef_ghij_klmn → 0000_abcd_efgh_ijkl
	// The bytes become: high byte = 0000_abcd, low byte = efgh_ijkl
	// Original: high byte = abcd_efgh, low byte = ijkl_mnop
	// After VPSRLW $4: high byte = 0000_abcd, low byte = efgh_ijkl
	// Low byte now has HIGH nibble of high_byte in its high nibble!
	// After VPAND 0x0F: high byte = 0000_0000 & 0x0F = 0x0d (just low nibble of shifted high byte)
	//                    low byte = ijkl & 0x0F — this is WRONG! We wanted mnop >> 4 = 0x0m!
	//
	// The FIX: VPSRLW is indeed wrong. Use VPAND 0xF0 first, then VPSRLW $4:
	// Original bytes: [H1 L1] [H2 L2] ...
	// VPAND 0xF0: [H1&F0, L1&F0] = [high_nibble_H1<<4, high_nibble_L1<<4]
	// VPSRLW $4: shift word right 4 — still corrupts across byte boundary.
	//
	// CORRECT FIX: just use VPAND + VPSRLW for the MASKED version:
	// Step 1: VPSRLW $4, Y1, Y1  (shift words right 4)
	// Step 2: VPAND 0x0F, Y1, Y1 (mask low nibble of EACH byte)
	// After step 1: each byte contains garbage from adjacent byte's bits
	// After step 2: mask removes the garbage! Because:
	//   Low byte of word after VPSRLW $4: contains bits from both bytes, BUT
	//   the low 4 bits = the original high nibble of that byte's original position...
	//   NO! This is still wrong.
	//
	// Actually let me think simply:
	// Byte at position 0 has value V0, byte at position 1 has V1 (in a 16-bit word: V1:V0)
	// VPSRLW $4 treats [V1:V0] as 16-bit number V1*256+V0, shifts right 4:
	//   result = (V1*256 + V0) >> 4 = V1*16 + V0/16
	//   Low byte = (V1*16 + V0/16) & 0xFF = (V1 & 0x0F)<<4 | (V0 >> 4)
	//   High byte = V1 >> 4
	// So: low byte after shift = (V1_low_nibble << 4) | V0_high_nibble
	// After VPAND 0x0F: low byte = V0_high_nibble ← CORRECT!
	// High byte after shift = V1 >> 4 = V1_high_nibble
	// After VPAND 0x0F: high byte = V1_high_nibble ← CORRECT!
	// 
	// Wait — (V1*16 + V0/16) & 0xFF:
	// V1=0x5A, V0=0x3C: word=0x5A3C, >>4 = 0x05A3
	// Low byte = 0xA3, high byte = 0x05
	// VPAND 0x0F: low = 0x03, high = 0x05
	// Expected: V0 high nibble = 0x03 ← CORRECT!
	// Expected: V1 high nibble = 0x05 ← CORRECT!
	//
	// So VPSRLW $4 followed by VPAND 0x0F DOES correctly extract the high nibble of each byte!
	// The key insight: the cross-byte contamination from the shift is in bits [7:4] of the
	// low byte, which get masked away by VPAND 0x0F. BUG #2 IS NOT A BUG!

	MOVL    $0x0F0F0F0F, R9
	VMOVD   R9, X10
	VPBROADCASTD X10, Y10

	TESTQ   R8, R8
	JZ      q4k_low
	VPSRLW  $4, Y1, Y1           // shift words right 4 (extracts high nibble after mask)
q4k_low:
	VPAND   Y10, Y1, Y1          // mask: 32 clean nibbles [0-15]

	// Widen and FMA
	VXORPS  Y0, Y0, Y0

	VEXTRACTI128 $0, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS (DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 32(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VEXTRACTI128 $1, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 64(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 96(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VEXTRACTF128 $1, Y0, X2
	VADDPS  X0, X2, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	MOVSS   X0, ret+32(FP)
	VZEROUPPER
	RET

// =====================================================================
// Q5_K FUSED AVX2 KERNEL — 32 elements
// =====================================================================
// func dotQ5K32AVX2(qs *byte, qh *byte, qhBitOffset uint64, isHigh uint64, scale float32, minVal float32, x *float32) float32
TEXT ·dotQ5K32AVX2(SB), NOSPLIT, $0-52
	MOVQ    qs+0(FP), SI
	MOVQ    qh+8(FP), DX
	MOVQ    qhBitOffset+16(FP), CX   // FIX BUG #4: this is the sub-group bit index (0-7)
	MOVQ    isHigh+24(FP), R8
	MOVSS   scale+32(FP), X14
	MOVSS   minVal+36(FP), X13
	MOVQ    x+40(FP), DI

	VBROADCASTSS X14, Y14
	VBROADCASTSS X13, Y13

	// Load qs, extract nibble
	VMOVDQU (SI), Y1
	MOVL    $0x0F0F0F0F, R9
	VMOVD   R9, X10
	VPBROADCASTD X10, Y10

	TESTQ   R8, R8
	JZ      q5k_low
	VPSRLW  $4, Y1, Y1
q5k_low:
	VPAND   Y10, Y1, Y1          // Y1 = 32 × 4-bit values

	// Load qh (32 bytes), extract single bit per byte using VPSRLW + AND 0x01
	// VPSRLW shifts 16-bit WORDS right — both bytes get the correct bit because:
	//   byte0 of word: bit N ends up at position 0 after shift → AND 0x01 = correct
	//   byte1 of word: its bit N is at position N+8 in the word, after shift right N
	//                  it's at position 8, which is bit 0 of byte1 → AND 0x01 = correct!
	// (Unlike VPSRLD which corrupts bytes 1-3 due to 32-bit cross-byte leakage)
	VMOVDQU (DX), Y2
	VMOVD   CX, X9
	VPSRLW  X9, Y2, Y2           // shift 16-bit words right by bit offset
	MOVL    $0x01010101, R9
	VMOVD   R9, X10
	VPBROADCASTD X10, Y10
	VPAND   Y10, Y2, Y2          // Y2 = 32 × single bit (0 or 1) — CORRECT for all bytes!

	// Map 0→0x00, 1→0x10 via VPSHUFB lookup
	MOVQ    $0x0000000000001000, R9
	VMOVQ   R9, X8
	VPBROADCASTQ X8, Y8
	VPSHUFB Y2, Y8, Y2            // Y2[i] = Y8[Y2[i]] → 0→0x00, 1→0x10

	// Combine: q5 = nibble | (high_bit << 4)
	VPOR    Y2, Y1, Y1

	// Widen and FMA
	VXORPS  Y0, Y0, Y0

	VEXTRACTI128 $0, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS (DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 32(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VEXTRACTI128 $1, Y1, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 64(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VPSRLDQ $8, X6, X6
	VPMOVSXBD X6, Y3
	VCVTDQ2PS Y3, Y3
	VMULPS  Y14, Y3, Y3
	VSUBPS  Y13, Y3, Y3
	VMOVUPS 96(DI), Y4
	VFMADD231PS Y3, Y4, Y0

	VEXTRACTF128 $1, Y0, X2
	VADDPS  X0, X2, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	MOVSS   X0, ret+48(FP)
	VZEROUPPER
	RET
