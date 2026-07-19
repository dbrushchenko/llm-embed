package embed

import "math"

// MeanPool averages hidden states over non-masked positions.
// hiddens: [seqLen][dim], mask: [seqLen] (1 for real tokens, 0 for padding)
// Returns: [dim] averaged vector.
func MeanPool(hiddens [][]float32, mask []int) []float32 {
	if len(hiddens) == 0 {
		return nil
	}
	dim := len(hiddens[0])
	sum := make([]float32, dim)
	var count float32

	for i, m := range mask {
		if m == 1 {
			for j := 0; j < dim; j++ {
				sum[j] += hiddens[i][j]
			}
			count++
		}
	}

	if count == 0 {
		return sum
	}

	invCount := 1.0 / count
	for j := 0; j < dim; j++ {
		sum[j] *= invCount
	}
	return sum
}

// L2Normalize normalizes a vector to unit length (L2 norm = 1).
func L2Normalize(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v
	}

	out := make([]float32, len(v))
	invNorm := float32(1.0 / norm)
	for i, x := range v {
		out[i] = x * invNorm
	}
	return out
}
