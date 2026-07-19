package embed

import (
	"fmt"
	"github.com/dbrushchenko/llm-embed/internal/ggml"
	"runtime"
	"sync"
)

// Embedder is the public API for producing text embeddings.
type Embedder struct {
	model      *BertModel
	tokenizer  *WordPieceTokenizer
	targetDim  int
	TaskPrefix string // e.g. "search_query: " or "search_document: "
}

// Load loads the BERT model and tokenizer from a GGUF file.
func Load(modelPath string) (*Embedder, error) {
	model, err := LoadBertModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("embed: load model: %w", err)
	}

	// Pre-dequant all weights for GEMM acceleration (~50MB extra RAM, 2× faster inference)
	model.PreDequant()

	// Precompute RoPE cos/sin table (eliminates 36K+ trig calls per forward pass)
	model.ropeTable = ggml.NewRoPETable(model.MaxSeqLen, model.HeadDim, model.RoPEBase)

	tokenizer, err := LoadTokenizer(modelPath)
	if err != nil {
		return nil, fmt.Errorf("embed: load tokenizer: %w", err)
	}

	return &Embedder{
		model:      model,
		tokenizer:  tokenizer,
		targetDim:  model.HiddenDim,
		TaskPrefix: "search_query: ",
	}, nil
}

// Embed produces a normalized embedding vector for the given text.
// The TaskPrefix is prepended to the text before tokenization.
func (e *Embedder) Embed(text string) ([]float32, error) {
	// Prepend task prefix
	fullText := e.TaskPrefix + text

	// Tokenize
	ids, mask := e.tokenizer.Encode(fullText, e.model.MaxSeqLen)

	// Forward pass
	hiddens, err := e.model.Forward(ids, mask)
	if err != nil {
		return nil, fmt.Errorf("embed: forward: %w", err)
	}

	// Mean pooling over non-masked positions
	pooled := MeanPool(hiddens, mask)

	// L2 normalize
	normalized := L2Normalize(pooled)

	// Matryoshka truncation if target dim is smaller than model dim
	if e.targetDim < len(normalized) {
		normalized = L2Normalize(normalized[:e.targetDim])
	}

	return normalized, nil
}

// EmbedBatch produces normalized embedding vectors for multiple texts using parallel workers.
// Concurrency-safe: each goroutine runs an independent forward pass (no shared mutable state).
func (e *Embedder) EmbedBatch(texts []string) ([][]float32, error) {
	n := len(texts)
	if n == 0 {
		return nil, nil
	}
	if n == 1 {
		vec, err := e.Embed(texts[0])
		if err != nil {
			return nil, err
		}
		return [][]float32{vec}, nil
	}

	results := make([][]float32, n)
	errs := make([]error, n)

	nWorkers := runtime.NumCPU()
	if nWorkers > n {
		nWorkers = n
	}
	if nWorkers > 8 {
		nWorkers = 8 // diminishing returns beyond 8 (cache contention on weight tensors)
	}

	var wg sync.WaitGroup
	ch := make(chan int, n)

	// Feed work items
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)

	// Launch workers
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				vec, err := e.Embed(texts[i])
				if err != nil {
					errs[i] = err
				} else {
					results[i] = vec
				}
			}
		}()
	}

	wg.Wait()

	// Return first error encountered
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("embed: batch item %d: %w", i, err)
		}
	}
	return results, nil
}

// Dim returns the current output embedding dimension.
func (e *Embedder) Dim() int {
	return e.targetDim
}

// SetTargetDim sets the target output dimension for Matryoshka truncation.
// Valid values for nomic-embed-text-v1.5: 768, 512, 256, 128, 64.
func (e *Embedder) SetTargetDim(dim int) {
	if dim > e.model.HiddenDim {
		dim = e.model.HiddenDim
	}
	if dim < 1 {
		dim = 1
	}
	e.targetDim = dim
}

// Model returns the underlying BertModel for testing/benchmarking.
func (e *Embedder) Model() *BertModel { return e.model }
