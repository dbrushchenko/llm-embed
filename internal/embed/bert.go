package embed

import (
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/dbrushchenko/llm-embed/internal/ggml"
	"github.com/dbrushchenko/llm-embed/internal/gguf"
)

// BertModel holds all weights for the nomic-bert embedding model.
type BertModel struct {
	// Model params
	NumLayers  int
	HiddenDim  int
	NumHeads   int
	HeadDim    int
	FFNDim     int
	MaxSeqLen  int
	LNEps      float32
	RoPEBase   float32

	// Memory-mapped backing (keeps mmap alive while tensors reference it)
	mapped *ggml.MmapedFile

	// Embeddings
	TokenEmbd     *ggml.Tensor // [768, 30522] q8_0
	TokenTypes    *ggml.Tensor // [768, 2] f32
	EmbdNormW     []float32   // [768]
	EmbdNormB     []float32   // [768]

	// Transformer layers
	Layers []BertLayer

	// Pre-allocated scratch buffers for GEMM (avoid per-call allocation)
	scratchA []float32 // [MaxSeqLen × HiddenDim]
	scratchC []float32 // [MaxSeqLen × max(2304, 3072)]

	// Precomputed RoPE cos/sin table (eliminates 36K+ trig calls per forward pass)
	ropeTable *ggml.RoPETable
}

// Close releases the memory-mapped file. Call when done with the model.
func (m *BertModel) Close() {
	if m.mapped != nil {
		m.mapped.Close()
		m.mapped = nil
	}
}

// BertLayer holds weights for a single transformer layer.
type BertLayer struct {
	// Combined QKV projection: [768, 2304]
	AttnQKV *ggml.Tensor

	// Attention output projection: [768, 768]
	AttnOutput *ggml.Tensor

	// Attention output layer norm
	AttnNormW []float32 // [768]
	AttnNormB []float32 // [768]

	// FFN (gated): up [768, 3072], gate [768, 3072], down [3072, 768]
	FFNUp   *ggml.Tensor
	FFNGate *ggml.Tensor
	FFNDown *ggml.Tensor

	// Layer output layer norm
	LayerNormW []float32 // [768]
	LayerNormB []float32 // [768]

	// Pre-dequanted float32 weights for GEMM (populated by PreDequant)
	dqQKV    []float32 // [N×K] transposed
	dqOutput []float32
	dqUp     []float32
	dqGate   []float32
	dqDown   []float32

	// Pre-packed B panels (panel format for zero-cost GEMM B access)
	ppQKV    []float32
	ppOutput []float32
	ppUp     []float32
	ppGate   []float32
	ppDown   []float32
}

// LoadBertModel loads the BERT model weights from a GGUF file.
// Uses memory-mapped I/O: weights are backed by OS page cache, not heap.
func LoadBertModel(path string) (*BertModel, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("embed: open model: %w", err)
	}

	// Read model params from metadata
	numLayers, _ := gf.GetUint32("nomic-bert.block_count")
	hiddenDim, _ := gf.GetUint32("nomic-bert.embedding_length")
	ffnDim, _ := gf.GetUint32("nomic-bert.feed_forward_length")
	numHeads, _ := gf.GetUint32("nomic-bert.attention.head_count")
	maxSeqLen, _ := gf.GetUint32("nomic-bert.context_length")
	lnEps, _ := gf.GetFloat32("nomic-bert.attention.layer_norm_epsilon")
	ropeBase, _ := gf.GetFloat32("nomic-bert.rope.freq_base")

	if numLayers == 0 || hiddenDim == 0 {
		return nil, fmt.Errorf("embed: missing model params in GGUF metadata")
	}

	model := &BertModel{
		NumLayers: int(numLayers),
		HiddenDim: int(hiddenDim),
		NumHeads:  int(numHeads),
		HeadDim:   int(hiddenDim) / int(numHeads),
		FFNDim:    int(ffnDim),
		MaxSeqLen: int(maxSeqLen),
		LNEps:     lnEps,
		RoPEBase:  ropeBase,
		Layers:    make([]BertLayer, numLayers),
	}

	// Memory-map the entire file — tensors reference slices of the mapped region
	mapped, err := ggml.MmapOpen(path)
	if err != nil {
		// Fallback to traditional file I/O if mmap fails
		return loadBertModelFallback(path, gf, model)
	}
	// NOTE: mapped must live as long as the model (tensors point into it)
	model.mapped = mapped

	// Helper to load a tensor by name (zero-copy slice from mmap)
	loadTensor := func(name string) (*ggml.Tensor, error) {
		info := gf.GetTensor(name)
		if info == nil {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		size := info.ByteSize()
		offset := int(gf.DataOffset + info.Offset)
		data := mapped.Data[offset : offset+int(size)]
		return ggml.NewTensorFromGGUF(info, data), nil
	}
	// Helper to load a small f32 tensor and return as []float32
	loadF32Vec := func(name string) ([]float32, error) {
		t, err := loadTensor(name)
		if err != nil {
			return nil, err
		}
		return t.Dequantize()
	}

	// Load embeddings
	model.TokenEmbd, err = loadTensor("token_embd.weight")
	if err != nil {
		return nil, err
	}
	model.TokenTypes, err = loadTensor("token_types.weight")
	if err != nil {
		return nil, err
	}
	model.EmbdNormW, err = loadF32Vec("token_embd_norm.weight")
	if err != nil {
		return nil, err
	}
	model.EmbdNormB, err = loadF32Vec("token_embd_norm.bias")
	if err != nil {
		return nil, err
	}

	// Load transformer layers
	for i := 0; i < int(numLayers); i++ {
		prefix := fmt.Sprintf("blk.%d.", i)

		model.Layers[i].AttnQKV, err = loadTensor(prefix + "attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].AttnOutput, err = loadTensor(prefix + "attn_output.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].FFNUp, err = loadTensor(prefix + "ffn_up.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].FFNGate, err = loadTensor(prefix + "ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].FFNDown, err = loadTensor(prefix + "ffn_down.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].AttnNormW, err = loadF32Vec(prefix + "attn_output_norm.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].AttnNormB, err = loadF32Vec(prefix + "attn_output_norm.bias")
		if err != nil {
			return nil, err
		}
		model.Layers[i].LayerNormW, err = loadF32Vec(prefix + "layer_output_norm.weight")
		if err != nil {
			return nil, err
		}
		model.Layers[i].LayerNormB, err = loadF32Vec(prefix + "layer_output_norm.bias")
		if err != nil {
			return nil, err
		}
	}

	return model, nil
}

// Forward runs the BERT forward pass on token IDs and attention mask.
// Returns hidden states: [seqLen][hiddenDim].
func (m *BertModel) Forward(tokenIDs []int, attentionMask []int) ([][]float32, error) {
	seqLen := len(tokenIDs)
	dim := m.HiddenDim

	// 1. Compute embeddings: token_embd + token_type_embd
	// GGUF tensor convention: shape[0] = inner (contiguous) dim = 768
	// token_embd.weight [768, 30522] → 30522 vectors of 768 floats
	// token_types.weight [768, 2] → 2 vectors of 768 floats
	// Use getEmbedding helper which uses shape[0] as stride

	// Pre-fetch token type 0 embedding (all tokens use segment 0)
	typeEmbd, err := getEmbedding(m.TokenTypes, 0, dim)
	if err != nil {
		return nil, fmt.Errorf("token type embedding: %w", err)
	}

	hiddens := make([][]float32, seqLen)
	for i, tok := range tokenIDs {
		// Token embedding
		tokEmbd, err := getEmbedding(m.TokenEmbd, tok, dim)
		if err != nil {
			return nil, fmt.Errorf("token embedding %d: %w", tok, err)
		}

		// Sum embeddings
		h := make([]float32, dim)
		for j := 0; j < dim; j++ {
			h[j] = tokEmbd[j] + typeEmbd[j]
		}

		// Apply embedding LayerNorm
		ggml.LayerNormInPlace(h, m.EmbdNormW, m.EmbdNormB, m.LNEps)
		hiddens[i] = h
	}

	// 2. Transformer layers
	for layerIdx := 0; layerIdx < m.NumLayers; layerIdx++ {
		layer := &m.Layers[layerIdx]

		// Multi-head attention with RoPE
		attnOut, err := m.multiHeadAttention(hiddens, attentionMask, layer)
		if err != nil {
			return nil, fmt.Errorf("layer %d attention: %w", layerIdx, err)
		}

		// Residual + LayerNorm
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dim; j++ {
				hiddens[i][j] += attnOut[i][j]
			}
			ggml.LayerNormInPlace(hiddens[i], layer.AttnNormW, layer.AttnNormB, m.LNEps)
		}

		// FFN (gated)
		ffnOut, err := m.ffnForward(hiddens, layer)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn: %w", layerIdx, err)
		}

		// Residual + LayerNorm
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dim; j++ {
				hiddens[i][j] += ffnOut[i][j]
			}
			ggml.LayerNormInPlace(hiddens[i], layer.LayerNormW, layer.LayerNormB, m.LNEps)
		}
	}

	return hiddens, nil
}

// getEmbedding retrieves a single embedding vector from a GGUF embedding tensor.
// GGUF convention: shape[0] = inner dim (stride), shape[1..] = outer dims.
// So for token_embd [768, 30522], each embedding is 768 floats, and there are 30522.
func getEmbedding(t *ggml.Tensor, idx int, dim int) ([]float32, error) {
	switch t.Type {
	case 0: // F32
		offset := idx * dim * 4
		data := t.Data[offset : offset+dim*4]
		return ggml.DequantF32Slice(data, dim)
	case 8: // Q8_0
		blocksPerRow := dim / 32
		bytesPerRow := blocksPerRow * 34
		offset := idx * bytesPerRow
		return ggml.DequantQ8_0Slice(t.Data[offset:offset+bytesPerRow], dim)
	case 12: // Q4_K
		bytesPerRow := ggml.Q4_K_RowSize(dim)
		offset := idx * bytesPerRow
		return ggml.DequantQ4_K(t.Data[offset:offset+bytesPerRow], dim)
	case 13: // Q5_K
		bytesPerRow := ggml.Q5_K_RowSize(dim)
		offset := idx * bytesPerRow
		return ggml.DequantQ5_K(t.Data[offset:offset+bytesPerRow], dim)
	case 14: // Q6_K
		bytesPerRow := ggml.Q6_K_RowSize(dim)
		offset := idx * bytesPerRow
		return ggml.DequantQ6_K(t.Data[offset:offset+bytesPerRow], dim)
	default:
		return nil, fmt.Errorf("unsupported embedding type: %v", t.Type)
	}
}

// multiHeadAttention computes bidirectional multi-head attention with RoPE.
func (m *BertModel) multiHeadAttention(hiddens [][]float32, mask []int, layer *BertLayer) ([][]float32, error) {
	seqLen := len(hiddens)
	dim := m.HiddenDim
	numHeads := m.NumHeads
	headDim := m.HeadDim

	// Project to QKV for all positions — BATCHED (weight matrix read once for all tokens)
	// attn_qkv.weight: [768, 2304] → all positions [seqLen][768] → [seqLen][2304]
	qkv, err := gemmOrBatch(layer.dqQKV, layer.ppQKV, layer.AttnQKV, hiddens)
	if err != nil {
		return nil, err
	}

	// Split into Q, K, V and apply RoPE to Q, K
	// QKV layout: [Q0..Q767 | K768..K1535 | V1536..V2303]
	// Apply RoPE per head
	Q := make([][]float32, seqLen) // [seqLen][dim]
	K := make([][]float32, seqLen)
	V := make([][]float32, seqLen)

	for i := 0; i < seqLen; i++ {
		Q[i] = make([]float32, dim)
		K[i] = make([]float32, dim)
		V[i] = make([]float32, dim)

		copy(Q[i], qkv[i][0:dim])
		copy(K[i], qkv[i][dim:2*dim])
		copy(V[i], qkv[i][2*dim:3*dim])

		// Apply RoPE to each head in Q and K (using precomputed cos/sin table)
		for h := 0; h < numHeads; h++ {
			offset := h * headDim
			m.ropeTable.ApplyInPlace(Q[i][offset:offset+headDim], i)
			m.ropeTable.ApplyInPlace(K[i][offset:offset+headDim], i)
		}
	}

	// Compute attention per head (parallelized — heads are independent)
	output := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		output[i] = make([]float32, dim)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	var wg sync.WaitGroup
	for h := 0; h < numHeads; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()
			offset := h * headDim
			scores := make([]float32, seqLen) // reused across all query positions

			// For each query position
			for qi := 0; qi < seqLen; qi++ {
				// Compute attention scores for this query against all keys
				for ki := 0; ki < seqLen; ki++ {
					qRow := Q[qi][offset : offset+headDim]
					kRow := K[ki][offset : offset+headDim]
					scores[ki] = ggml.DotF32(qRow, kRow) * scale
				}

				// Apply attention mask (mask out padding positions)
				for ki := 0; ki < seqLen; ki++ {
					if mask[ki] == 0 {
						scores[ki] = -1e9
					}
				}

				// Softmax
				ggml.SoftmaxInPlace(scores)

				// Weighted sum of values
				for d := 0; d < headDim; d++ {
					var sum float32
					for vi := 0; vi < seqLen; vi++ {
						sum += scores[vi] * V[vi][offset+d]
					}
					output[qi][offset+d] = sum
				}
			}
		}(h)
	}
	wg.Wait()

	// Output projection — batched
	result, err := gemmOrBatch(layer.dqOutput, layer.ppOutput, layer.AttnOutput, output)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// ffnForward computes the gated FFN for all positions.
// FFN(x) = down(gelu(gate(x)) * up(x))
func (m *BertModel) ffnForward(hiddens [][]float32, layer *BertLayer) ([][]float32, error) {
	seqLen := len(hiddens)

	// Batched up projection: all tokens at once
	ups, err := gemmOrBatch(layer.dqUp, layer.ppUp, layer.FFNUp, hiddens)
	if err != nil {
		return nil, err
	}

	// Batched gate projection: all tokens at once
	gates, err := gemmOrBatch(layer.dqGate, layer.ppGate, layer.FFNGate, hiddens)
	if err != nil {
		return nil, err
	}

	// Apply GeLU to gates, multiply with ups (per-token, cheap)
	for i := 0; i < seqLen; i++ {
		ggml.GeLUInPlace(gates[i])
		for j := range gates[i] {
			gates[i][j] *= ups[i][j]
		}
	}

	// Batched down projection: all tokens at once
	result, err := gemmOrBatch(layer.dqDown, layer.ppDown, layer.FFNDown, gates)
	if err != nil {
		return nil, err
	}

	return result, nil
}


// loadBertModelFallback uses traditional file I/O when mmap is unavailable.
func loadBertModelFallback(path string, gf *gguf.File, model *BertModel) (*BertModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	loadTensor := func(name string) (*ggml.Tensor, error) {
		info := gf.GetTensor(name)
		if info == nil {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		size := info.ByteSize()
		data := make([]byte, size)
		offset := int64(gf.DataOffset + info.Offset)
		if _, err := f.ReadAt(data, offset); err != nil {
			return nil, fmt.Errorf("reading tensor %q: %w", name, err)
		}
		return ggml.NewTensorFromGGUF(info, data), nil
	}

	loadF32Vec := func(name string) ([]float32, error) {
		t, err := loadTensor(name)
		if err != nil {
			return nil, err
		}
		return t.Dequantize()
	}

	var errL error
	model.TokenEmbd, errL = loadTensor("token_embd.weight")
	if errL != nil {
		return nil, errL
	}
	model.TokenTypes, errL = loadTensor("token_types.weight")
	if errL != nil {
		return nil, errL
	}
	model.EmbdNormW, errL = loadF32Vec("token_embd_norm.weight")
	if errL != nil {
		return nil, errL
	}
	model.EmbdNormB, errL = loadF32Vec("token_embd_norm.bias")
	if errL != nil {
		return nil, errL
	}

	for i := 0; i < model.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		model.Layers[i].AttnQKV, errL = loadTensor(prefix + "attn_qkv.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].AttnOutput, errL = loadTensor(prefix + "attn_output.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].FFNUp, errL = loadTensor(prefix + "ffn_up.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].FFNGate, errL = loadTensor(prefix + "ffn_gate.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].FFNDown, errL = loadTensor(prefix + "ffn_down.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].AttnNormW, errL = loadF32Vec(prefix + "attn_output_norm.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].AttnNormB, errL = loadF32Vec(prefix + "attn_output_norm.bias")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].LayerNormW, errL = loadF32Vec(prefix + "layer_output_norm.weight")
		if errL != nil {
			return nil, errL
		}
		model.Layers[i].LayerNormB, errL = loadF32Vec(prefix + "layer_output_norm.bias")
		if errL != nil {
			return nil, errL
		}
	}

	return model, nil
}

// PreDequant pre-dequantizes all weight matrices for GEMM acceleration.
// Call once after loading. Uses ~50MB extra RAM for Q8_0 model but enables
// 2× faster inference via tiled GEMM with float32 micro-kernels.
func (m *BertModel) PreDequant() {
	// Pre-dequant ALL models for Sgemm/GemmPrePacked acceleration (~50MB extra RAM)
	K := m.HiddenDim
	for i := range m.Layers {
		l := &m.Layers[i]
		qkvN := 1
		for _, d := range l.AttnQKV.Shape[1:] { qkvN *= d }
		l.dqQKV = ggml.DequantWeightMatrix(l.AttnQKV, K, qkvN)
		if ggml.HasAVX512 { l.ppQKV = ggml.PrePackBPanel32(l.dqQKV, K, qkvN) } else { l.ppQKV = ggml.PrePackBPanel16(l.dqQKV, K, qkvN) }

		outN := 1
		for _, d := range l.AttnOutput.Shape[1:] { outN *= d }
		l.dqOutput = ggml.DequantWeightMatrix(l.AttnOutput, K, outN)
		if ggml.HasAVX512 { l.ppOutput = ggml.PrePackBPanel32(l.dqOutput, K, outN) } else { l.ppOutput = ggml.PrePackBPanel16(l.dqOutput, K, outN) }

		upN := 1
		for _, d := range l.FFNUp.Shape[1:] { upN *= d }
		l.dqUp = ggml.DequantWeightMatrix(l.FFNUp, K, upN)
		if ggml.HasAVX512 { l.ppUp = ggml.PrePackBPanel32(l.dqUp, K, upN) } else { l.ppUp = ggml.PrePackBPanel16(l.dqUp, K, upN) }

		gateN := 1
		for _, d := range l.FFNGate.Shape[1:] { gateN *= d }
		l.dqGate = ggml.DequantWeightMatrix(l.FFNGate, K, gateN)
		if ggml.HasAVX512 { l.ppGate = ggml.PrePackBPanel32(l.dqGate, K, gateN) } else { l.ppGate = ggml.PrePackBPanel16(l.dqGate, K, gateN) }

		downK := m.FFNDim
		downN := 1
		for _, d := range l.FFNDown.Shape[1:] { downN *= d }
		l.dqDown = ggml.DequantWeightMatrix(l.FFNDown, downK, downN)
		if ggml.HasAVX512 { l.ppDown = ggml.PrePackBPanel32(l.dqDown, downK, downN) } else { l.ppDown = ggml.PrePackBPanel16(l.dqDown, downK, downN) }
	}
	// Allocate scratch buffers (reused across all gemmOrBatch calls)
	m.scratchA = make([]float32, m.MaxSeqLen*m.HiddenDim)
	m.scratchC = make([]float32, m.MaxSeqLen*3072) // max output dim
}

// gemmOrBatch uses pre-dequanted GEMM if available, otherwise falls back to MatMulBatch.
func gemmOrBatch(dq []float32, pp []float32, w *ggml.Tensor, inputs [][]float32) ([][]float32, error) {
	if dq != nil {
		M := len(inputs)
		K := w.Shape[0]
		N := 1
		for _, d := range w.Shape[1:] { N *= d }

		// Priority 1: MKL cblas_sgemm (fastest if available)
		if ggml.MKLAvailable() {
			A := make([]float32, M*K)
			for i := 0; i < M; i++ {
				copy(A[i*K:(i+1)*K], inputs[i])
			}
			C := make([]float32, M*N)
			ggml.Sgemm(false, true, M, N, K, 1.0, A, K, dq, K, 0.0, C, N)
			outputs := make([][]float32, M)
			for i := 0; i < M; i++ {
				outputs[i] = C[i*N : (i+1)*N]
			}
			return outputs, nil
		}

		// Priority 2: AVX-512 GemmPrePacked32 (if hardware supports + pre-packed available)
		if ggml.HasAVX512 && pp != nil {
			C := make([]float32, M*N)
			ggml.GemmPrePacked32(inputs, pp, C, M, N, K)
			outputs := make([][]float32, M)
			for i := 0; i < M; i++ {
				outputs[i] = C[i*N : (i+1)*N]
			}
			return outputs, nil
		}

		// Priority 3: AVX2 GemmPrePacked16 (pre-packed available)
		if pp != nil {
			C := make([]float32, M*N)
			ggml.GemmPrePacked16(inputs, pp, C, M, N, K)
			outputs := make([][]float32, M)
			for i := 0; i < M; i++ {
				outputs[i] = C[i*N : (i+1)*N]
			}
			return outputs, nil
		}

		// Priority 4: Sgemm with tiled pure Go (no pre-packing)
		A := make([]float32, M*K)
		for i := 0; i < M; i++ {
			copy(A[i*K:(i+1)*K], inputs[i])
		}
		C := make([]float32, M*N)
		ggml.Sgemm(false, true, M, N, K, 1.0, A, K, dq, K, 0.0, C, N)
		outputs := make([][]float32, M)
		for i := 0; i < M; i++ {
			outputs[i] = C[i*N : (i+1)*N]
		}
		return outputs, nil
	}
	// K-quant path: fused AVX2 kernels in MatMulBatch are faster than dequant+GEMM
	return ggml.MatMulBatch(w, inputs)
}
