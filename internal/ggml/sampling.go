package ggml

import (
	"math"
	"math/rand"
	"sort"
)

// SampleParams configures token sampling behavior.
type SampleParams struct {
	Temperature float32 // 0 = greedy, >0 = random. Default: 1.0
	TopP        float32 // Nucleus sampling cutoff. Default: 0.95
	TopK        int     // Top-K filter before sampling. Default: 40 (0 = disabled)
	MinP        float32 // Minimum probability relative to top. Default: 0.05
	Seed        int64   // RNG seed (0 = use default)
}

// DefaultSampleParams returns production-quality sampling parameters.
func DefaultSampleParams() SampleParams {
	return SampleParams{
		Temperature: 1.0,
		TopP:        0.95,
		TopK:        40,
		MinP:        0.05,
	}
}

// GreedyParams returns params for greedy (deterministic) decoding.
func GreedyParams() SampleParams {
	return SampleParams{Temperature: 0}
}

// Sample selects a token from logits using the configured sampling strategy.
func Sample(logits []float32, params SampleParams, rng *rand.Rand) int {
	if params.Temperature <= 0 {
		return ArgMax(logits)
	}

	n := len(logits)

	// Apply temperature
	scaled := make([]float32, n)
	invTemp := 1.0 / params.Temperature
	for i := range logits {
		scaled[i] = logits[i] * invTemp
	}

	// Softmax to get probabilities
	probs := Softmax(scaled)

	// Top-K filter
	if params.TopK > 0 && params.TopK < n {
		probs = topKFilter(probs, params.TopK)
	}

	// Min-P filter (relative to top probability)
	if params.MinP > 0 {
		probs = minPFilter(probs, params.MinP)
	}

	// Top-P (nucleus) filter
	if params.TopP > 0 && params.TopP < 1.0 {
		probs = topPFilter(probs, params.TopP)
	}

	// Renormalize
	var sum float32
	for _, p := range probs {
		sum += p
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	} else {
		// Fallback: uniform over all tokens
		for i := range probs {
			probs[i] = 1.0 / float32(n)
		}
	}

	// Sample from distribution
	r := rng.Float32()
	var cumsum float32
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return n - 1 // shouldn't reach here
}

// topKFilter zeroes out all probabilities except the top-K.
func topKFilter(probs []float32, k int) []float32 {
	type idxVal struct {
		idx int
		val float32
	}

	// Find the k-th largest value
	sorted := make([]idxVal, len(probs))
	for i, p := range probs {
		sorted[i] = idxVal{i, p}
	}
	sort.Slice(sorted, func(a, b int) bool {
		return sorted[a].val > sorted[b].val
	})

	threshold := sorted[k-1].val

	// Zero out everything below threshold
	result := make([]float32, len(probs))
	for i, p := range probs {
		if p >= threshold {
			result[i] = p
		}
	}
	return result
}

// topPFilter implements nucleus sampling: keep smallest set of tokens
// whose cumulative probability exceeds topP.
func topPFilter(probs []float32, topP float32) []float32 {
	type idxVal struct {
		idx int
		val float32
	}

	sorted := make([]idxVal, 0, len(probs))
	for i, p := range probs {
		if p > 0 {
			sorted = append(sorted, idxVal{i, p})
		}
	}
	sort.Slice(sorted, func(a, b int) bool {
		return sorted[a].val > sorted[b].val
	})

	// Find cutoff
	var cumsum float32
	cutoffIdx := len(sorted)
	for i, sv := range sorted {
		cumsum += sv.val
		if cumsum >= topP {
			cutoffIdx = i + 1
			break
		}
	}

	// Zero out everything below cutoff
	result := make([]float32, len(probs))
	for i := 0; i < cutoffIdx; i++ {
		result[sorted[i].idx] = sorted[i].val
	}
	return result
}

// minPFilter removes tokens with probability less than minP * max_prob.
func minPFilter(probs []float32, minP float32) []float32 {
	maxProb := float32(0)
	for _, p := range probs {
		if p > maxProb {
			maxProb = p
		}
	}
	threshold := maxProb * minP

	result := make([]float32, len(probs))
	for i, p := range probs {
		if p >= threshold {
			result[i] = p
		}
	}
	return result
}

// RepetitionPenalty applies a penalty to tokens that appeared in context.
func RepetitionPenalty(logits []float32, contextTokens []int, penalty float32) []float32 {
	if penalty == 1.0 || len(contextTokens) == 0 {
		return logits
	}
	result := make([]float32, len(logits))
	copy(result, logits)
	for _, tok := range contextTokens {
		if tok >= 0 && tok < len(result) {
			if result[tok] > 0 {
				result[tok] /= penalty
			} else {
				result[tok] *= penalty
			}
		}
	}
	return result
}

// LogitsToProbs converts raw logits to probabilities (softmax with temperature).
func LogitsToProbs(logits []float32, temperature float32) []float32 {
	if temperature <= 0 {
		// Greedy: one-hot on argmax
		probs := make([]float32, len(logits))
		probs[ArgMax(logits)] = 1.0
		return probs
	}
	scaled := make([]float32, len(logits))
	for i := range logits {
		scaled[i] = logits[i] / temperature
	}
	return Softmax(scaled)
}

// Entropy computes the Shannon entropy of a probability distribution.
func Entropy(probs []float32) float32 {
	var h float32
	for _, p := range probs {
		if p > 0 {
			h -= p * float32(math.Log2(float64(p)))
		}
	}
	return h
}
