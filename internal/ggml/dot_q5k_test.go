package ggml

import (
	"math"
	"math/rand"
	"testing"
)

// makeQ5_KBlock creates a single Q5_K block (176 bytes) with known values.
// d, dmin are float32 super-block scale/min.
// scales: 8 sub-blocks, each with (scale, min) 6-bit values.
// quants: 256 5-bit values [0..31] to encode.
func makeQ5_KBlock(d, dmin float32, subScales [8]uint8, subMins [8]uint8, quants [256]uint8) []byte {
	block := make([]byte, Q5_K_BytesPerBlock)

	// d (f16, 2 bytes)
	hd := f32ToF16(d)
	block[0] = byte(hd)
	block[1] = byte(hd >> 8)

	// dmin (f16, 2 bytes)
	hm := f32ToF16(dmin)
	block[2] = byte(hm)
	block[3] = byte(hm >> 8)

	// Pack scales (12 bytes at offset 4)
	// For simplicity, only use 6-bit values (< 64) so we can use the simple j<4 path
	scales := block[4:16]
	for i := 0; i < 4; i++ {
		scales[i] = subScales[i] & 63
		scales[i+4] = subMins[i] & 63
	}
	// For sub-blocks 4-7, pack into bytes 8-11 with high bits from bytes 0-3
	for i := 4; i < 8; i++ {
		sc := subScales[i]
		m := subMins[i]
		// scales[j+4] stores low nibbles: sc low in low 4, m low in high 4
		scales[i+4] = (sc & 0xF) | ((m & 0xF) << 4)
		// High 2 bits go into bits 6-7 of scales[i-4]
		scales[i-4] |= (sc >> 4) << 6
		scales[i] |= (m >> 4) << 6
	}

	// qh (32 bytes at offset 16) — high bits
	qh := block[16:48]
	// qs (128 bytes at offset 48) — low 4 bits, nibble packed
	qs := block[48:176]

	// Encode 256 quants into qs and qh
	// Layout (matching llama.cpp):
	//   4 groups of 64, each group has 2 sub-groups of 32
	//   Sub-group s (0-7): position l (0-31)
	//     low4 stored in qs: first 32 in low nibble, second 32 in high nibble
	//     high1 stored in qh[l] bit s
	qOff := 0
	subGroup := 0
	for j := 0; j < 256; j += 64 {
		// First 32: low nibble of qs[qOff+l]
		for l := 0; l < 32; l++ {
			q := quants[j+l]
			qs[qOff+l] = (qs[qOff+l] & 0xF0) | (q & 0xF) // low nibble
			if q&0x10 != 0 {
				qh[l] |= 1 << uint(subGroup)
			}
		}
		subGroup++

		// Second 32: high nibble of qs[qOff+l]
		for l := 0; l < 32; l++ {
			q := quants[j+32+l]
			qs[qOff+l] = (qs[qOff+l] & 0x0F) | ((q & 0xF) << 4) // high nibble
			if q&0x10 != 0 {
				qh[l] |= 1 << uint(subGroup)
			}
		}
		subGroup++

		qOff += 32
	}

	return block
}

func TestDequantQ5_KKnownValues(t *testing.T) {
	// Create a block where all quants = 15 (low4=15, high=0)
	// d=1.0, dmin=0.0, all sub-block scales=1, mins=0
	var quants [256]uint8
	for i := range quants {
		quants[i] = 15 // 5-bit value: 01111
	}
	var subScales [8]uint8
	var subMins [8]uint8
	for i := range subScales {
		subScales[i] = 1
	}

	block := makeQ5_KBlock(1.0, 0.0, subScales, subMins, quants)
	out, err := DequantQ5_K(block, 256)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: d * scale * q - dmin * min = 1.0 * 1 * 15 - 0 = 15.0
	for i, v := range out {
		if math.Abs(float64(v)-15.0) > 0.01 {
			t.Fatalf("element %d: expected 15.0, got %f", i, v)
		}
	}
}

func TestDequantQ5_KHighBit(t *testing.T) {
	// Test that the high bit is correctly applied.
	// Set all quants to 16 (low4=0, high=1): binary 10000
	var quants [256]uint8
	for i := range quants {
		quants[i] = 16
	}
	var subScales [8]uint8
	var subMins [8]uint8
	for i := range subScales {
		subScales[i] = 1
	}

	block := makeQ5_KBlock(1.0, 0.0, subScales, subMins, quants)
	out, err := DequantQ5_K(block, 256)
	if err != nil {
		t.Fatal(err)
	}

	// Expected: d * scale * q - dmin * min = 1.0 * 1 * 16 - 0 = 16.0
	for i, v := range out {
		if math.Abs(float64(v)-16.0) > 0.01 {
			t.Fatalf("element %d: expected 16.0, got %f", i, v)
		}
	}
}

func TestDequantQ5_KMixed(t *testing.T) {
	// Test mixed values: quants alternate between 0 and 31
	var quants [256]uint8
	for i := range quants {
		if i%2 == 0 {
			quants[i] = 0
		} else {
			quants[i] = 31
		}
	}
	var subScales [8]uint8
	var subMins [8]uint8
	for i := range subScales {
		subScales[i] = 2
		subMins[i] = 1
	}

	block := makeQ5_KBlock(0.5, 0.25, subScales, subMins, quants)
	out, err := DequantQ5_K(block, 256)
	if err != nil {
		t.Fatal(err)
	}

	// Expected for q=0: 0.5*2*0 - 0.25*1 = -0.25
	// Expected for q=31: 0.5*2*31 - 0.25*1 = 31.0 - 0.25 = 30.75
	for i, v := range out {
		var expected float64
		if i%2 == 0 {
			expected = -0.25
		} else {
			expected = 30.75
		}
		if math.Abs(float64(v)-expected) > 0.05 {
			t.Fatalf("element %d: expected %f, got %f", i, expected, v)
		}
	}
}

func TestDequantQ5_KHighBitPattern(t *testing.T) {
	// Test that high bits land in the correct position for each sub-group.
	// Set specific elements: only element 0 in each sub-group gets high bit set.
	var quants [256]uint8
	// Sub-group 0: elements 0-31 (first 32 of j=0 group)
	quants[0] = 16 // high bit set for element 0, sub-group 0
	// Sub-group 1: elements 32-63 (second 32 of j=0 group)
	quants[32] = 16 // high bit set for element 32 (position 0 in sub-group 1)
	// Sub-group 2: elements 64-95
	quants[64] = 16
	// Sub-group 3: elements 96-127
	quants[96] = 16
	// Sub-group 4: elements 128-159
	quants[128] = 16
	// Sub-group 5: elements 160-191
	quants[160] = 16
	// Sub-group 6: elements 192-223
	quants[192] = 16
	// Sub-group 7: elements 224-255
	quants[224] = 16

	var subScales [8]uint8
	var subMins [8]uint8
	for i := range subScales {
		subScales[i] = 1
	}

	block := makeQ5_KBlock(1.0, 0.0, subScales, subMins, quants)
	out, err := DequantQ5_K(block, 256)
	if err != nil {
		t.Fatal(err)
	}

	// Each marked element should be 16.0, all others should be 0.0
	specialElements := map[int]bool{0: true, 32: true, 64: true, 96: true, 128: true, 160: true, 192: true, 224: true}
	for i, v := range out {
		if specialElements[i] {
			if math.Abs(float64(v)-16.0) > 0.01 {
				t.Errorf("element %d: expected 16.0 (high bit), got %f", i, v)
			}
		} else {
			if math.Abs(float64(v)) > 0.01 {
				t.Errorf("element %d: expected 0.0, got %f", i, v)
			}
		}
	}
}

func TestDotQ5_KvsDeqDot(t *testing.T) {
	// Verify DotQ5_K matches DequantQ5_K followed by manual dot product.
	rng := rand.New(rand.NewSource(42))

	// Generate random Q5_K block
	var quants [256]uint8
	for i := range quants {
		quants[i] = uint8(rng.Intn(32)) // 5-bit values
	}
	var subScales [8]uint8
	var subMins [8]uint8
	for i := range subScales {
		subScales[i] = uint8(rng.Intn(60) + 1)
		subMins[i] = uint8(rng.Intn(60))
	}

	block := makeQ5_KBlock(0.05, 0.01, subScales, subMins, quants)

	// Random x vector
	x := make([]float32, 256)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// Method 1: DequantQ5_K then dot product
	dequant, err := DequantQ5_K(block, 256)
	if err != nil {
		t.Fatal(err)
	}
	var expected float32
	for i := 0; i < 256; i++ {
		expected += dequant[i] * x[i]
	}

	// Method 2: DotQ5_K (fused)
	got := DotQ5_K(block, x, 256)

	relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-10)
	if relErr > 1e-5 {
		t.Errorf("DotQ5_K mismatch: expected=%f, got=%f, relErr=%e", expected, got, relErr)
	}
	t.Logf("DotQ5_K vs manual dot: expected=%f, got=%f, relErr=%e", expected, got, relErr)
}

func TestDotQ5_KMultiBlock(t *testing.T) {
	// Test with multiple blocks (512 elements = 2 blocks)
	rng := rand.New(rand.NewSource(99))

	n := 512
	numBlocks := n / QK_K

	// Build raw Q5_K data for 2 blocks
	data := make([]byte, numBlocks*Q5_K_BytesPerBlock)
	for b := 0; b < numBlocks; b++ {
		var quants [256]uint8
		for i := range quants {
			quants[i] = uint8(rng.Intn(32))
		}
		var subScales [8]uint8
		var subMins [8]uint8
		for i := range subScales {
			subScales[i] = uint8(rng.Intn(60) + 1)
			subMins[i] = uint8(rng.Intn(60))
		}
		block := makeQ5_KBlock(0.03, 0.005, subScales, subMins, quants)
		copy(data[b*Q5_K_BytesPerBlock:], block)
	}

	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// Dequant + dot
	dequant, err := DequantQ5_K(data, n)
	if err != nil {
		t.Fatal(err)
	}
	var expected float32
	for i := 0; i < n; i++ {
		expected += dequant[i] * x[i]
	}

	got := DotQ5_K(data, x, n)

	relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-10)
	if relErr > 1e-5 {
		t.Errorf("multi-block: expected=%f, got=%f, relErr=%e", expected, got, relErr)
	}
	t.Logf("Multi-block DotQ5_K: expected=%f, got=%f", expected, got)
}


func TestDotQ5K32AVX2Direct(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Test the AVX2 kernel directly against manual scalar computation.
	// This isolates the kernel from the DotQ5_KFast wrapper.
	rng := rand.New(rand.NewSource(7))

	// Generate test data
	var qs [32]byte // 32 packed nibble bytes
	var qh [32]byte // 32 high-bit bytes
	var x [32]float32

	for i := range qs {
		qs[i] = byte(rng.Intn(256)) // random packed nibbles
	}
	for i := range qh {
		qh[i] = byte(rng.Intn(256)) // random high bits
	}
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// Test all 8 qhBitOffset values × 2 isHigh values
	for qhBit := uint64(0); qhBit < 8; qhBit++ {
		for isHigh := uint64(0); isHigh <= 1; isHigh++ {
			scale := float32(0.05)
			minVal := float32(0.01)

			// Compute expected result manually (same as scalar code)
			var expected float32
			for l := 0; l < 32; l++ {
				var low4 byte
				if isHigh == 0 {
					low4 = qs[l] & 0x0F
				} else {
					low4 = qs[l] >> 4
				}
				high1 := (qh[l] >> qhBit) & 1
				q := low4 | (high1 << 4)
				val := scale*float32(q) - minVal
				expected += val * x[l]
			}

			got := dotQ5K32AVX2(&qs[0], &qh[0], qhBit, isHigh, scale, minVal, &x[0])

			relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-6)
			if relErr > 1e-4 {
				t.Errorf("qhBit=%d isHigh=%d: expected=%f, got=%f, relErr=%e",
					qhBit, isHigh, expected, got, relErr)
			}
		}
	}
}

func TestDotQ5K32AVX2HighBitOnly(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Isolate high-bit extraction: all low nibbles = 0, x = 1.0
	// If high bit correctly extracted, result = scale * 16 * count_of_set_bits - minVal * 32
	var qs [32]byte // all zeros (both nibbles)
	var qh [32]byte
	var x [32]float32

	// Set bit 3 of every qh byte (test qhBitOffset=3)
	for i := range qh {
		qh[i] = 0x08 // bit 3 set
	}
	for i := range x {
		x[i] = 1.0
	}

	scale := float32(1.0)
	minVal := float32(0.0)

	// All elements: low4=0, high1=1, q=16
	// expected = sum(1.0 * 16 * 1.0) = 16 * 32 = 512
	expected := float32(512.0)
	got := dotQ5K32AVX2(&qs[0], &qh[0], 3, 0, scale, minVal, &x[0])

	if math.Abs(float64(got-expected)) > 0.01 {
		t.Errorf("High bit only (qhBit=3): expected=%f, got=%f", expected, got)
	}

	// Same but with bit NOT set (qhBitOffset=4, but bit 3 is set)
	// All elements: low4=0, high1=0, q=0
	// expected = sum(1.0 * 0 - 0) = 0
	got2 := dotQ5K32AVX2(&qs[0], &qh[0], 4, 0, scale, minVal, &x[0])
	if math.Abs(float64(got2)) > 0.01 {
		t.Errorf("No high bit (qhBit=4): expected=0, got=%f", got2)
	}
}

func TestDotQ5_KFastVsScalar(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Test DotQ5_KFast against DotQ5_K with random data
	rng := rand.New(rand.NewSource(123))

	for trial := 0; trial < 10; trial++ {
		n := 256 * (1 + rng.Intn(4)) // 1-4 blocks
		numBlocks := n / QK_K

		data := make([]byte, numBlocks*Q5_K_BytesPerBlock)
		for b := 0; b < numBlocks; b++ {
			var quants [256]uint8
			for i := range quants {
				quants[i] = uint8(rng.Intn(32))
			}
			var subScales [8]uint8
			var subMins [8]uint8
			for i := range subScales {
				subScales[i] = uint8(rng.Intn(60) + 1)
				subMins[i] = uint8(rng.Intn(60))
			}
			block := makeQ5_KBlock(0.03+rng.Float32()*0.05, 0.005+rng.Float32()*0.01, subScales, subMins, quants)
			copy(data[b*Q5_K_BytesPerBlock:], block)
		}

		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}

		expected := DotQ5_K(data, x, n)
		got := DotQ5_KFast(data, x, n)

		relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-10)
		if relErr > 1e-4 {
			t.Errorf("trial %d (n=%d): scalar=%f, AVX2=%f, relErr=%e",
				trial, n, expected, got, relErr)
		}
	}
}
