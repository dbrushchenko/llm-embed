package embed

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

const testModelPath = "../../models/nomic-embed-text-v1.5.Q8_0.gguf"

func TestLoad(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if e == nil {
		t.Fatal("Load returned nil embedder")
	}
	if e.Dim() != 768 {
		t.Errorf("expected dim 768, got %d", e.Dim())
	}
}

func TestEmbed(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	vec, err := e.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	// Check dimension
	if len(vec) != 768 {
		t.Errorf("expected 768-dim vector, got %d", len(vec))
	}

	// Check not all zeros
	allZero := true
	for _, v := range vec {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("embedding is all zeros")
	}

	// Check no NaN values
	for i, v := range vec {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at position %d", i)
			break
		}
		if math.IsInf(float64(v), 0) {
			t.Errorf("Inf at position %d", i)
			break
		}
	}

	// Check L2 norm is approximately 1.0 (since we normalize)
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("expected L2 norm ~1.0, got %f", norm)
	}

	// Check values are in reasonable range
	for i, v := range vec {
		if v > 1.0 || v < -1.0 {
			t.Errorf("value out of [-1,1] range at position %d: %f", i, v)
			break
		}
	}

	t.Logf("Embedding first 5 values: %v", vec[:5])
}

func TestSetTargetDim(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	e.SetTargetDim(256)
	if e.Dim() != 256 {
		t.Errorf("expected dim 256 after SetTargetDim, got %d", e.Dim())
	}

	vec, err := e.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed with truncation failed: %v", err)
	}
	if len(vec) != 256 {
		t.Errorf("expected 256-dim vector after truncation, got %d", len(vec))
	}

	// Truncated vector should still be normalized
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 0.01 {
		t.Errorf("truncated vector L2 norm should be ~1.0, got %f", norm)
	}
}

func TestEmbedBatchConsistency(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Single embed
	single, err := e.Embed("hello world")
	if err != nil {
		t.Fatalf("single Embed failed: %v", err)
	}

	// Batch of 1
	batch, err := e.EmbedBatch([]string{"hello world"})
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(batch) != 1 {
		t.Fatalf("expected 1 result, got %d", len(batch))
	}

	// Compare: should be identical
	for i := range single {
		if single[i] != batch[0][i] {
			t.Errorf("mismatch at position %d: single=%f, batch=%f", i, single[i], batch[0][i])
			break
		}
	}
}

func TestTokenizer(t *testing.T) {
	tok, err := LoadTokenizer(testModelPath)
	if err != nil {
		t.Fatalf("LoadTokenizer failed: %v", err)
	}

	if tok.VocabSize() != 30522 {
		t.Errorf("expected vocab size 30522, got %d", tok.VocabSize())
	}

	ids, mask := tok.Encode("hello world", 512)
	if len(ids) < 3 { // [CLS] + at least 1 token + [SEP]
		t.Errorf("expected at least 3 tokens, got %d", len(ids))
	}
	if ids[0] != 101 {
		t.Errorf("expected [CLS] (101) at start, got %d", ids[0])
	}
	if ids[len(ids)-1] != 102 {
		t.Errorf("expected [SEP] (102) at end, got %d", ids[len(ids)-1])
	}
	if len(mask) != len(ids) {
		t.Errorf("mask length %d != ids length %d", len(mask), len(ids))
	}
	for _, m := range mask {
		if m != 1 {
			t.Error("expected all mask values to be 1 for non-padded input")
			break
		}
	}

	t.Logf("'hello world' tokenized to %d tokens: %v", len(ids), ids)
}


func BenchmarkEmbedFull(b *testing.B) {
	e, err := Load(testModelPath)
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := e.Embed("search_query: what is machine learning")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbedShort(b *testing.B) {
	e, err := Load(testModelPath)
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := e.Embed("hello world")
		if err != nil {
			b.Fatal(err)
		}
	}
}


func TestEmbedBatchParallel(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	texts := []string{
		"hello world",
		"what is machine learning",
		"the cat sat on the mat",
		"how to cook pasta",
		"quantum computing basics",
		"go programming language",
		"neural network architecture",
		"climate change effects",
	}

	// Run batch (parallel)
	results, err := e.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(results) != len(texts) {
		t.Fatalf("expected %d results, got %d", len(texts), len(results))
	}

	// Verify each result matches individual Embed call
	for i, text := range texts {
		single, err := e.Embed(text)
		if err != nil {
			t.Fatalf("single Embed failed for %q: %v", text, err)
		}

		for j := range single {
			if single[j] != results[i][j] {
				t.Errorf("text %q: mismatch at dim %d: single=%f, batch=%f", text, j, single[j], results[i][j])
				break
			}
		}
	}
}

func TestEmbedConcurrentSafety(t *testing.T) {
	e, err := Load(testModelPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Hammer the embedder from multiple goroutines simultaneously
	// This tests that the model struct is safe for concurrent reads
	const nGoroutines = 16
	const nIter = 5

	texts := []string{"hello", "world", "test", "embedding"}
	errs := make(chan error, nGoroutines*nIter)

	var wg sync.WaitGroup
	for g := 0; g < nGoroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < nIter; i++ {
				text := texts[(g+i)%len(texts)]
				vec, err := e.Embed(text)
				if err != nil {
					errs <- err
					return
				}
				if len(vec) != 768 {
					errs <- fmt.Errorf("goroutine %d iter %d: expected 768 dim, got %d", g, i, len(vec))
					return
				}
				// Verify normalized
				var norm float64
				for _, v := range vec {
					norm += float64(v) * float64(v)
				}
				norm = math.Sqrt(norm)
				if math.Abs(norm-1.0) > 0.01 {
					errs <- fmt.Errorf("goroutine %d iter %d: norm=%f", g, i, norm)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

func BenchmarkEmbedBatch8(b *testing.B) {
	e, err := Load(testModelPath)
	if err != nil {
		b.Fatalf("Load failed: %v", err)
	}

	texts := []string{
		"hello world",
		"what is machine learning",
		"the cat sat on the mat",
		"how to cook pasta",
		"quantum computing basics",
		"go programming language",
		"neural network architecture",
		"climate change effects",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := e.EmbedBatch(texts)
		if err != nil {
			b.Fatal(err)
		}
	}
}
