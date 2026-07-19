package ggml

import (
	"fmt"
	"sync"
	"math"
)

// MatMul performs matrix multiplication: C = A @ B^T
// A: [M, K], B: [N, K] (B stored row-major, so each row is a "column" of the logical B^T)
// Result C: [M, N]
// This is the standard GGML convention: weights are stored [out_features, in_features]
// and we compute output = input @ weight^T
func MatMul(a *Tensor, b *Tensor) (*Tensor, error) {
	// a is [M, K] f32 (activations)
	// b is [N, K] quantized or f32 (weights)
	// result is [M, N] f32
	M := a.Rows()
	K := a.Cols()
	N := b.Rows() // output dim = number of rows in weight matrix

	if b.Cols() != K {
		panic("ggml: MatMul dimension mismatch: a.Cols != b.Cols")
	}

	// Get a as float32
	aData, err := a.Float32s()
	if err != nil {
		return nil, err
	}

	// Result buffer
	out := make([]float32, M*N)

	// For each row of A, compute dot product with each row of B
	for m := 0; m < M; m++ {
		aRow := aData[m*K : (m+1)*K]
		for n := 0; n < N; n++ {
			bRow, err := b.Row(n)
			if err != nil {
				return nil, err
			}
			var sum float32
			for k := 0; k < K; k++ {
				sum += aRow[k] * bRow[k]
			}
			out[m*N+n] = sum
		}
	}

	return FromFloat32(out, M, N), nil
}

// MatMulVec performs matrix-vector multiplication: y = W @ x
// For GGUF tensors, shape is [in_features, out_features] (inner dim = input).
// W: [K, N] where K=input dim (shape[0]), N=output dim (shape[1])
// x: [K] (input vector)
// Result y: [N] (output vector)
// This is the hot path for single-token decode.
func MatMulVec(w *Tensor, x []float32) ([]float32, error) {
	// GGUF convention: shape[0] = inner (input) dim, shape[1] = outer (output) dim
	K := w.Shape[0]  // input features (inner/contiguous dimension)
	N := 1
	for _, d := range w.Shape[1:] {
		N *= d
	}
	if N == 1 && len(w.Shape) == 1 {
		N = K
		K = 1
	}

	if len(x) != K {
		panic(fmt.Sprintf("ggml: MatMulVec dimension mismatch: weight inner=%d, input len=%d", K, len(x)))
	}

	out := make([]float32, N)

	// Each "row" in memory is K contiguous elements (one output neuron's weights)
	// There are N such rows
	switch w.Type {
	case 8: // GGML_TYPE_Q8_0
		blocksPerRow := K / 32
		bytesPerRow := blocksPerRow * 34
		// Use parallel for large matrices, sequential for small
		if N >= 256 {
			return MatMulVecQ8_0Parallel(w.Data, x, K, N, NumWorkers()), nil
		}
		// Small matrix: sequential with fused dot
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			out[n] = DotQ8_0(w.Data[offset:offset+bytesPerRow], x, K)
		}
	case 0: // GGML_TYPE_F32
		for n := 0; n < N; n++ {
			offset := n * K * 4
			row, err := DequantF32Slice(w.Data[offset:offset+K*4], K)
			if err != nil {
				return nil, err
			}
			var sum float32
			for k := 0; k < K; k++ {
				sum += row[k] * x[k]
			}
			out[n] = sum
		}
	case 12: // GGML_TYPE_Q4_K
		bytesPerRow := Q4_K_RowSize(K)
		if N >= 256 {
			return matMulVecKQuantParallel(w.Data, x, K, N, bytesPerRow, DotQ4_KFast, NumWorkers()), nil
		}
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			out[n] = DotQ4_KFast(w.Data[offset:offset+bytesPerRow], x, K)
		}
	case 13, 15, 16: // GGML_TYPE_Q5_K, Q5_K_S, Q5_K_M
		bytesPerRow := Q5_K_RowSize(K)
		if N >= 256 {
			return matMulVecKQuantParallel(w.Data, x, K, N, bytesPerRow, DotQ5_KFast, NumWorkers()), nil
		}
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			out[n] = DotQ5_KFast(w.Data[offset:offset+bytesPerRow], x, K)
		}
	case 14: // GGML_TYPE_Q6_K
		bytesPerRow := Q6_K_RowSize(K)
		if N >= 256 {
			return matMulVecKQuantParallel(w.Data, x, K, N, bytesPerRow, DotQ6_KFast, NumWorkers()), nil
		}
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			out[n] = DotQ6_KFast(w.Data[offset:offset+bytesPerRow], x, K)
		}
	default:
		// Fallback: use Row()
		for n := 0; n < N; n++ {
			row, err := w.Row(n)
			if err != nil {
				return nil, err
			}
			if len(row) != K {
				// Row returned wrong size - use what we can
				actualK := len(row)
				if actualK > K {
					actualK = K
				}
				var sum float32
				for k := 0; k < actualK; k++ {
					sum += row[k] * x[k]
				}
				out[n] = sum
			} else {
				var sum float32
				for k := 0; k < K; k++ {
					sum += row[k] * x[k]
				}
				out[n] = sum
			}
		}
	}

	return out, nil
}

// RMSNorm applies Root Mean Square Layer Normalization.
// x: input vector [n], weight: scale vector [n], eps: epsilon
// output[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
func RMSNorm(x []float32, weight []float32, eps float32) []float32 {
	n := len(x)
	out := make([]float32, n)

	// Compute sum of squares
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = ss/float32(n) + eps
	rms := float32(1.0 / math.Sqrt(float64(ss)))

	// Normalize and scale
	for i := 0; i < n; i++ {
		out[i] = x[i] * rms * weight[i]
	}
	return out
}

// RoPE applies Rotary Position Embedding to query/key vectors.
// x: [head_dim] vector for a single head
// pos: token position
// freqBase: base frequency (e.g., 1000000.0 for Gemma 4)
// nRot: number of dimensions to rotate (usually head_dim)
func RoPE(x []float32, pos int, freqBase float32, nRot int) []float32 {
	out := make([]float32, len(x))
	copy(out, x)

	for i := 0; i < nRot/2; i++ {
		// Frequency for this dimension pair
		freq := float32(1.0) / float32(math.Pow(float64(freqBase), float64(2*i)/float64(nRot)))
		theta := freq * float32(pos)

		cosT := float32(math.Cos(float64(theta)))
		sinT := float32(math.Sin(float64(theta)))

		// Rotate pair (x[2i], x[2i+1])
		x0 := x[2*i]
		x1 := x[2*i+1]
		out[2*i] = x0*cosT - x1*sinT
		out[2*i+1] = x0*sinT + x1*cosT
	}

	return out
}

// RoPEWithFreqs applies RoPE using precomputed frequency tensor.
// freqs: [n_rot/2] precomputed frequency values
func RoPEWithFreqs(x []float32, pos int, freqs []float32) []float32 {
	out := make([]float32, len(x))
	copy(out, x)

	nRot := len(freqs) * 2
	if nRot > len(x) {
		nRot = len(x)
	}

	for i := 0; i < nRot/2; i++ {
		theta := freqs[i] * float32(pos)
		cosT := float32(math.Cos(float64(theta)))
		sinT := float32(math.Sin(float64(theta)))

		x0 := x[2*i]
		x1 := x[2*i+1]
		out[2*i] = x0*cosT - x1*sinT
		out[2*i+1] = x0*sinT + x1*cosT
	}

	return out
}

// SiLU applies the SiLU (Swish) activation: x * sigmoid(x)
func SiLU(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * sigmoid(v)
	}
	return out
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// GeLU applies the Gaussian Error Linear Unit activation (approximate).
func GeLU(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		// Approximate GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		out[i] = 0.5 * v * (1.0 + float32(math.Tanh(float64(0.7978845608*v*(1.0+0.044715*v*v)))))
	}
	return out
}

// Softmax applies softmax to a vector (in-place safe).
func Softmax(x []float32) []float32 {
	n := len(x)
	out := make([]float32, n)

	// Find max for numerical stability
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	// exp(x - max) and sum
	var sum float32
	for i := 0; i < n; i++ {
		out[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += out[i]
	}

	// Normalize
	invSum := 1.0 / sum
	for i := 0; i < n; i++ {
		out[i] *= invSum
	}
	return out
}

// Add performs element-wise addition: out = a + b
func Add(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// AddInPlace performs element-wise addition in place: a += b
func AddInPlace(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// Mul performs element-wise multiplication: out = a * b
func Mul(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

// Scale multiplies every element by a scalar.
func Scale(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * s
	}
	return out
}

// ScaleInPlace multiplies every element by a scalar in place.
func ScaleInPlace(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}

// Embedding looks up a token embedding from the embedding weight matrix.
// weight: [n_vocab, n_embd] quantized, tokenID: token index
// Returns [n_embd] float32.
func Embedding(weight *Tensor, tokenID int) ([]float32, error) {
	return weight.Row(tokenID)
}

// ArgMax returns the index of the maximum value.
func ArgMax(x []float32) int {
	maxIdx := 0
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// TopK returns the indices of the top-k largest values.
func TopK(x []float32, k int) []int {
	if k >= len(x) {
		indices := make([]int, len(x))
		for i := range indices {
			indices[i] = i
		}
		return indices
	}

	// Simple selection for small k
	indices := make([]int, k)
	used := make([]bool, len(x))

	for j := 0; j < k; j++ {
		maxIdx := -1
		maxVal := float32(math.Inf(-1))
		for i := range x {
			if !used[i] && x[i] > maxVal {
				maxVal = x[i]
				maxIdx = i
			}
		}
		indices[j] = maxIdx
		used[maxIdx] = true
	}
	return indices
}


// LayerNorm applies Layer Normalization.
// x: input vector [n], weight: scale vector [n], bias: offset vector [n], eps: epsilon
// output[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]
func LayerNorm(x []float32, weight []float32, bias []float32, eps float32) []float32 {
	n := len(x)
	out := make([]float32, n)

	// Compute mean
	var mean float32
	for i := 0; i < n; i++ {
		mean += x[i]
	}
	mean /= float32(n)

	// Compute variance
	var variance float32
	for i := 0; i < n; i++ {
		d := x[i] - mean
		variance += d * d
	}
	variance /= float32(n)

	// Normalize, scale, and shift
	invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))
	for i := 0; i < n; i++ {
		out[i] = (x[i]-mean)*invStd*weight[i] + bias[i]
	}
	return out
}

// MatMulBatch computes W × X for multiple input vectors simultaneously.
// W shape: [K, N] (same as MatMulVec). inputs: [][]float32, each of length K.
// Returns [][]float32, each of length N.
//
// This is 3-5× faster than calling MatMulVec in a loop because the weight
// matrix is loaded from memory ONCE for all input vectors (cache reuse).
// ONNX Runtime uses this approach — it's why ONNX was 60ms vs our 360ms.
func MatMulBatch(w *Tensor, inputs [][]float32) ([][]float32, error) {
	K := w.Shape[0]
	N := 1
	for _, d := range w.Shape[1:] {
		N *= d
	}
	if N == 1 && len(w.Shape) == 1 {
		N = K
		K = 1
	}

	seqLen := len(inputs)
	if seqLen == 0 {
		return nil, nil
	}
	if seqLen == 1 {
		out, err := MatMulVec(w, inputs[0])
		if err != nil {
			return nil, err
		}
		return [][]float32{out}, nil
	}

	// Allocate output
	outputs := make([][]float32, seqLen)
	for i := range outputs {
		outputs[i] = make([]float32, N)
	}

	// Strategy: iterate over output rows (weight rows), and for each row,
	// compute the dot product against ALL input vectors.
	// This reads each weight row ONCE and applies it to all inputs.
	nWorkers := NumWorkers()

	switch w.Type {
	case 8: // Q8_0 — use per-token parallel path (4x unroll is faster than batch for Q8_0)
		for s := 0; s < seqLen; s++ {
			outputs[s] = MatMulVecQ8_0Parallel(w.Data, inputs[s], K, N, nWorkers)
		}
	case 12: // Q4_K
		bytesPerRow := Q4_K_RowSize(K)
		if N >= 256 && nWorkers > 1 {
			matMulBatchParallel(w.Data, inputs, outputs, K, N, bytesPerRow, DotQ4_KFast, nWorkers)
		} else {
			for n := 0; n < N; n++ {
				offset := n * bytesPerRow
				rowData := w.Data[offset : offset+bytesPerRow]
				for s := 0; s < seqLen; s++ {
					outputs[s][n] = DotQ4_KFast(rowData, inputs[s], K)
				}
			}
		}
	case 13, 15, 16: // Q5_K
		bytesPerRow := Q5_K_RowSize(K)
		if N >= 256 && nWorkers > 1 {
			matMulBatchParallel(w.Data, inputs, outputs, K, N, bytesPerRow, DotQ5_KFast, nWorkers)
		} else {
			for n := 0; n < N; n++ {
				offset := n * bytesPerRow
				rowData := w.Data[offset : offset+bytesPerRow]
				for s := 0; s < seqLen; s++ {
					outputs[s][n] = DotQ5_KFast(rowData, inputs[s], K)
				}
			}
		}
	case 14: // Q6_K
		bytesPerRow := Q6_K_RowSize(K)
		if N >= 256 && nWorkers > 1 {
			matMulBatchParallel(w.Data, inputs, outputs, K, N, bytesPerRow, DotQ6_KFast, nWorkers)
		} else {
			for n := 0; n < N; n++ {
				offset := n * bytesPerRow
				rowData := w.Data[offset : offset+bytesPerRow]
				for s := 0; s < seqLen; s++ {
					outputs[s][n] = DotQ6_KFast(rowData, inputs[s], K)
				}
			}
		}
	case 0: // F32
		bytesPerRow := K * 4
		for n := 0; n < N; n++ {
			offset := n * bytesPerRow
			row, _ := DequantF32Slice(w.Data[offset:offset+bytesPerRow], K)
			for s := 0; s < seqLen; s++ {
				var sum float32
				for k := 0; k < K; k++ {
					sum += row[k] * inputs[s][k]
				}
				outputs[s][n] = sum
			}
		}
	default:
		// Fallback: sequential MatMulVec per input
		for s := range inputs {
			out, err := MatMulVec(w, inputs[s])
			if err != nil {
				return nil, err
			}
			outputs[s] = out
		}
	}

	return outputs, nil
}

// matMulBatchParallel splits output rows across workers, each worker processes
// ALL sequence positions for its assigned rows (maximizes weight data reuse).
func matMulBatchParallel(wData []byte, inputs [][]float32, outputs [][]float32, K, N, bytesPerRow int, dotFn func([]byte, []float32, int) float32, nWorkers int) {
	seqLen := len(inputs)
	var wg sync.WaitGroup
	chunkSize := (N + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > N {
			end = N
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for n := start; n < end; n++ {
				offset := n * bytesPerRow
				rowData := wData[offset : offset+bytesPerRow]
				for s := 0; s < seqLen; s++ {
					outputs[s][n] = dotFn(rowData, inputs[s], K)
				}
			}
		}(start, end)
	}
	wg.Wait()
}
