package ggml

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestDotQ6KFastCorrectness verifies that DotQ6_KFast matches the scalar DotQ6_K.
func TestDotQ6KFastCorrectness(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	rng := rand.New(rand.NewSource(42))

	// Test with 256 elements (1 block) and 768 elements (3 blocks)
	for _, n := range []int{256, 768} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			numBlocks := n / QK_K
			data := make([]byte, numBlocks*Q6_K_BytesPerBlock)

			// Fill with random but valid Q6_K data
			for i := range data {
				data[i] = byte(rng.Intn(256))
			}

			// Make d (super-block scale) a reasonable f16 value
			for b := 0; b < numBlocks; b++ {
				off := b*Q6_K_BytesPerBlock + 208
				// Encode f16 for ~0.01
				data[off] = 0x11   // f16 bits for a small positive value
				data[off+1] = 0x24 // roughly 0.01
			}

			// Random float32 input
			x := make([]float32, n)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}

			// Compare scalar vs fast
			expected := DotQ6_K(data, x, n)
			got := DotQ6_KFast(data, x, n)

			relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-10)
			if relErr > 1e-5 {
				t.Errorf("DotQ6_KFast mismatch: expected %g, got %g (relErr=%g)", expected, got, relErr)
			} else {
				t.Logf("OK: expected=%g, got=%g, relErr=%g", expected, got, relErr)
			}
		})
	}
}

// TestDotQ6KFastKnownScales tests the scale-per-16 behavior specifically.
// Uses data where scales differ between the two halves of each 32-element group.
func TestDotQ6KFastKnownScales(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	n := 256
	numBlocks := 1
	data := make([]byte, numBlocks*Q6_K_BytesPerBlock)

	// Set all ql to constant value (e.g., 0x11 = nibbles 1,1)
	for i := 0; i < 128; i++ {
		data[i] = 0x11
	}

	// Set qh to zero (no high bits)
	for i := 128; i < 192; i++ {
		data[i] = 0x00
	}

	// Set scales: make them ALL DIFFERENT to detect scale misassignment
	// scales[0] = 10, scales[1] = 20, scales[2] = 30, etc.
	for i := 0; i < 16; i++ {
		data[192+i] = byte(int8(10 + i*5))
	}

	// Set d to f16 for 1.0 (0x3C00)
	data[208] = 0x00
	data[209] = 0x3C

	// Input x: all 1.0
	x := make([]float32, n)
	for i := range x {
		x[i] = 1.0
	}

	expected := DotQ6_K(data, x, n)
	got := DotQ6_KFast(data, x, n)

	relErr := math.Abs(float64(got-expected)) / math.Max(math.Abs(float64(expected)), 1e-10)
	if relErr > 1e-5 {
		t.Errorf("Scale test: expected %g, got %g (relErr=%g)", expected, got, relErr)
	} else {
		t.Logf("Scale test OK: expected=%g, got=%g, relErr=%g", expected, got, relErr)
	}
}

// BenchmarkDotQ6_K benchmarks scalar vs fast implementations.
func BenchmarkDotQ6_K(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	n := 768
	numBlocks := n / QK_K
	data := make([]byte, numBlocks*Q6_K_BytesPerBlock)
	for i := range data {
		data[i] = byte(rng.Intn(256))
	}
	for bl := 0; bl < numBlocks; bl++ {
		off := bl*Q6_K_BytesPerBlock + 208
		data[off] = 0x11
		data[off+1] = 0x24
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	b.Run("Scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = DotQ6_K(data, x, n)
		}
	})

	b.Run("Fast", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = DotQ6_KFast(data, x, n)
		}
	})
}
