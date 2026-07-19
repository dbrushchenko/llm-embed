package ggml

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"
)

// makeQ8_0Block creates a single Q8_0 block (34 bytes) with given scale and quants.
func makeQ8_0Block(scale float32, quants [32]int8) []byte {
	block := make([]byte, 34)
	// Convert f32 scale to f16 bits (simple: just store as-is using the reverse of f16ToF32)
	h := f32ToF16(scale)
	block[0] = byte(h)
	block[1] = byte(h >> 8)
	for i, q := range quants {
		block[2+i] = byte(q)
	}
	return block
}

// f32ToF16 converts float32 to float16 bits (for test data generation).
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	if exp == 0 {
		return uint16(sign << 15)
	}
	if exp == 0xFF {
		return uint16((sign << 15) | 0x7C00 | (mant >> 13))
	}

	// Rebias exponent: f32 bias=127, f16 bias=15
	newExp := int(exp) - 127 + 15
	if newExp >= 31 {
		return uint16((sign << 15) | 0x7C00) // overflow -> inf
	}
	if newExp <= 0 {
		return uint16(sign << 15) // underflow -> zero
	}

	return uint16((sign << 15) | (uint32(newExp) << 10) | (mant >> 13))
}

// generateTestData creates random Q8_0 data and matching float32 vector.
func generateTestData(n int, rng *rand.Rand) ([]byte, []float32) {
	numBlocks := n / 32
	data := make([]byte, numBlocks*34)
	x := make([]float32, n)

	for b := 0; b < numBlocks; b++ {
		// Random scale in reasonable range [0.001, 0.1]
		scale := rng.Float32()*0.099 + 0.001
		h := f32ToF16(scale)
		off := b * 34
		data[off] = byte(h)
		data[off+1] = byte(h >> 8)
		// Random int8 quants
		for i := 0; i < 32; i++ {
			data[off+2+i] = byte(int8(rng.Intn(256) - 128))
		}
	}

	// Random float32 input vector
	for i := range x {
		x[i] = rng.Float32()*2 - 1 // [-1, 1]
	}

	return data, x
}

func TestHasAVX2(t *testing.T) {
	t.Logf("hasAVX2 = %v", hasAVX2)
	t.Logf("GOARCH = %s", runtime.GOARCH)
	if runtime.GOARCH == "amd64" && !hasAVX2 {
		t.Log("WARNING: AVX2 not detected on amd64 - benchmarks will use scalar fallback")
	}
}

func TestDotQ8_0AVX2Correctness(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	rng := rand.New(rand.NewSource(42))

	// Test various sizes (all multiples of 32)
	sizes := []int{32, 64, 128, 256, 768, 1024, 3072}

	for _, n := range sizes {
		t.Run(func() string {
			return "n=" + itoa(n)
		}(), func(t *testing.T) {
			data, x := generateTestData(n, rng)

			// Compute with scalar
			scalarResult := dotQ8_0Scalar(data, x, n)

			// Compute with AVX2
			avx2Result := dotQ8_0AVX2(&data[0], &x[0], n)

			// Compare: allow small relative error due to float32 associativity
			if scalarResult == 0 && avx2Result == 0 {
				return // both zero is fine
			}

			relErr := float64(math.Abs(float64(avx2Result-scalarResult))) / math.Max(float64(math.Abs(float64(scalarResult))), 1e-10)
			if relErr > 1e-5 {
				t.Errorf("n=%d: scalar=%f, avx2=%f, relErr=%e", n, scalarResult, avx2Result, relErr)
			}
		})
	}
}

func TestDotQ8_0AVX2KnownValues(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Create a simple known-value test
	// 1 block: scale=1.0, quants all = 1, x all = 1.0
	// Expected: 1.0 * (32 * 1.0 * 1) = 32.0
	var quants [32]int8
	for i := range quants {
		quants[i] = 1
	}
	data := makeQ8_0Block(1.0, quants)
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1.0
	}

	scalar := dotQ8_0Scalar(data, x, 32)
	avx2 := dotQ8_0AVX2(&data[0], &x[0], 32)

	if math.Abs(float64(scalar-32.0)) > 0.01 {
		t.Errorf("scalar expected ~32.0, got %f", scalar)
	}
	if math.Abs(float64(avx2-32.0)) > 0.01 {
		t.Errorf("avx2 expected ~32.0, got %f", avx2)
	}
	t.Logf("Known values: scalar=%f, avx2=%f", scalar, avx2)
}

func TestDotQ8_0AVX2NegativeQuants(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Test with negative quants to verify sign extension
	var quants [32]int8
	for i := range quants {
		quants[i] = -1
	}
	data := makeQ8_0Block(0.5, quants)
	x := make([]float32, 32)
	for i := range x {
		x[i] = 2.0
	}

	// Expected: 0.5 * sum(-1 * 2.0 for 32 elements) = 0.5 * (-64) = -32.0
	scalar := dotQ8_0Scalar(data, x, 32)
	avx2 := dotQ8_0AVX2(&data[0], &x[0], 32)

	if math.Abs(float64(scalar-(-32.0))) > 0.01 {
		t.Errorf("scalar expected ~-32.0, got %f", scalar)
	}
	if math.Abs(float64(avx2-(-32.0))) > 0.01 {
		t.Errorf("avx2 expected ~-32.0, got %f", avx2)
	}
}

func TestDotQ8_0AVX2MultipleBlocks(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	rng := rand.New(rand.NewSource(123))

	// 768 elements = 24 blocks (typical BERT hidden size)
	n := 768
	data, x := generateTestData(n, rng)

	scalar := dotQ8_0Scalar(data, x, n)
	avx2 := dotQ8_0AVX2(&data[0], &x[0], n)

	relErr := math.Abs(float64(avx2-scalar)) / math.Max(math.Abs(float64(scalar)), 1e-10)
	if relErr > 1e-5 {
		t.Errorf("768-elem: scalar=%f, avx2=%f, relErr=%e", scalar, avx2, relErr)
	}
	t.Logf("768 elements: scalar=%f, avx2=%f, relErr=%e", scalar, avx2, relErr)
}

func TestDotQ8_0AVX2Alignment(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Test that unaligned data pointers still work correctly
	// (VMOVUPS handles unaligned loads, but data pointer from slice might be unaligned)
	rng := rand.New(rand.NewSource(99))

	// Add extra byte at front to force misalignment
	n := 256
	rawData, x := generateTestData(n, rng)

	// Create misaligned copy
	padded := make([]byte, len(rawData)+1)
	copy(padded[1:], rawData)
	misaligned := padded[1:]

	scalar := dotQ8_0Scalar(misaligned, x, n)
	avx2 := dotQ8_0AVX2(&misaligned[0], &x[0], n)

	relErr := math.Abs(float64(avx2-scalar)) / math.Max(math.Abs(float64(scalar)), 1e-10)
	if relErr > 1e-5 {
		t.Errorf("misaligned: scalar=%f, avx2=%f, relErr=%e", scalar, avx2, relErr)
	}
}

// BenchmarkDotQ8_0Scalar benchmarks the scalar Go implementation.
func BenchmarkDotQ8_0Scalar(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	n := 768
	data, x := generateTestData(n, rng)

	b.SetBytes(int64(len(data) + n*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotQ8_0Scalar(data, x, n)
	}
}

// BenchmarkDotQ8_0AVX2 benchmarks the AVX2 assembly implementation.
func BenchmarkDotQ8_0AVX2(b *testing.B) {
	if !hasAVX2 {
		b.Skip("AVX2 not available")
	}

	rng := rand.New(rand.NewSource(42))
	n := 768
	data, x := generateTestData(n, rng)

	b.SetBytes(int64(len(data) + n*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotQ8_0AVX2(&data[0], &x[0], n)
	}
}

// BenchmarkDotQ8_0Dispatched benchmarks the dispatching DotQ8_0 (picks best path).
func BenchmarkDotQ8_0Dispatched(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	n := 768
	data, x := generateTestData(n, rng)

	b.SetBytes(int64(len(data) + n*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotQ8_0(data, x, n)
	}
}

// BenchmarkDotQ8_0Large benchmarks with a typical large dimension (3072 = FFN intermediate).
func BenchmarkDotQ8_0Large(b *testing.B) {
	if !hasAVX2 {
		b.Skip("AVX2 not available")
	}

	rng := rand.New(rand.NewSource(42))
	n := 3072
	data, x := generateTestData(n, rng)

	b.Run("Scalar", func(b *testing.B) {
		b.SetBytes(int64(len(data) + n*4))
		for i := 0; i < b.N; i++ {
			dotQ8_0Scalar(data, x, n)
		}
	})

	b.Run("AVX2", func(b *testing.B) {
		b.SetBytes(int64(len(data) + n*4))
		for i := 0; i < b.N; i++ {
			dotQ8_0AVX2(&data[0], &x[0], n)
		}
	})
}

// BenchmarkMatMulVecQ8_0 benchmarks full matrix-vector multiply (realistic workload).
func BenchmarkMatMulVecQ8_0(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	K := 768
	N := 768 // typical BERT hidden->hidden projection
	blocksPerRow := K / 32
	bytesPerRow := blocksPerRow * 34

	// Generate weight matrix
	wData := make([]byte, N*bytesPerRow)
	for i := range wData {
		wData[i] = byte(rng.Intn(256))
	}
	// Fix up scale bytes to be valid f16
	for row := 0; row < N; row++ {
		for blk := 0; blk < blocksPerRow; blk++ {
			off := row*bytesPerRow + blk*34
			s := f32ToF16(rng.Float32()*0.1 + 0.001)
			wData[off] = byte(s)
			wData[off+1] = byte(s >> 8)
		}
	}

	// Generate input vector
	x := make([]float32, K)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	b.SetBytes(int64(len(wData) + K*4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulVecQ8_0Parallel(wData, x, K, N, NumWorkers())
	}
}

// itoa is a simple int to string for test names (avoids strconv import).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	buf := make([]byte, 20)
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// Verify the test helper f32ToF16 round-trips correctly.
func TestF32ToF16RoundTrip(t *testing.T) {
	values := []float32{0.0, 1.0, 0.5, 0.001, 0.1, -0.5, -1.0}
	for _, v := range values {
		h := f32ToF16(v)
		back := f16ToF32(h)
		relErr := math.Abs(float64(back-v)) / math.Max(math.Abs(float64(v)), 1e-10)
		// f16 has limited precision, allow ~0.1% error
		if v != 0 && relErr > 0.002 {
			t.Errorf("f32ToF16 round-trip: %f -> 0x%04x -> %f (relErr=%e)", v, h, back, relErr)
		}
	}
}

// Ensure unsafe.Pointer conversions compile (sanity check).
func TestPointerConversions(t *testing.T) {
	data := make([]byte, 34)
	x := make([]float32, 32)
	_ = unsafe.Pointer(&data[0])
	_ = unsafe.Pointer(&x[0])
}
