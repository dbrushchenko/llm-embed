// Package embed implements bidirectional multi-head attention for BERT.
//
// Architecture notes for nomic-embed-text-v1.5:
//   - 12 heads, head_dim=64, total dim=768
//   - Combined QKV weight: [768, 2304] (Q|K|V concatenated)
//   - RoPE positional encoding (freq_base=1000), NOT absolute position embeddings
//   - Bidirectional: NO causal mask, all positions attend to all positions
//   - Attention mask is used only to zero out padding positions
//   - Output projection: [768, 768]
//
// The multi-head attention implementation is in bert.go (multiHeadAttention method).
// This file provides utility functions for attention computation.
package embed

import "math"

// ComputeRoPEFreqs precomputes the RoPE frequency table for a given base and dimension.
// Returns [headDim/2] frequencies.
func ComputeRoPEFreqs(headDim int, base float32) []float32 {
	nRot := headDim / 2
	freqs := make([]float32, nRot)
	for i := 0; i < nRot; i++ {
		freqs[i] = float32(1.0) / float32(math.Pow(float64(base), float64(2*i)/float64(headDim)))
	}
	return freqs
}

// ScaledDotProduct computes attention scores between a query and all keys.
// q: [headDim], keys: [seqLen][headDim]
// Returns [seqLen] unnormalized scores.
func ScaledDotProduct(q []float32, keys [][]float32, headDim int) []float32 {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	seqLen := len(keys)
	scores := make([]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		var dot float32
		for d := 0; d < headDim; d++ {
			dot += q[d] * keys[i][d]
		}
		scores[i] = dot * scale
	}
	return scores
}

// ApplyAttentionMask masks out padding positions by setting their scores to -inf.
func ApplyAttentionMask(scores []float32, mask []int) {
	for i, m := range mask {
		if m == 0 {
			scores[i] = -1e9
		}
	}
}
