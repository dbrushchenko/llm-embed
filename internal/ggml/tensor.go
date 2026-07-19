// Package ggml provides pure Go tensor operations for GGML-compatible inference.
package ggml

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/dbrushchenko/llm-embed/internal/gguf"
)

// Tensor represents a multi-dimensional array with optional quantization.
type Tensor struct {
	Name  string
	Shape []int    // dimensions (row-major: last dim is contiguous)
	Type  gguf.GGMLType
	Data  []byte  // raw quantized/float data
}

// NumElements returns the total number of logical elements.
func (t *Tensor) NumElements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Rows returns the number of rows (product of all dims except last).
func (t *Tensor) Rows() int {
	if len(t.Shape) < 2 {
		return 1
	}
	n := 1
	for _, d := range t.Shape[:len(t.Shape)-1] {
		n *= d
	}
	return n
}

// Cols returns the innermost dimension (number of columns).
func (t *Tensor) Cols() int {
	if len(t.Shape) == 0 {
		return 0
	}
	return t.Shape[len(t.Shape)-1]
}

// Dequantize converts quantized tensor data to float32.
// Returns a new float32 slice with NumElements() values.
func (t *Tensor) Dequantize() ([]float32, error) {
	switch t.Type {
	case gguf.GGML_TYPE_F32:
		return dequantF32(t.Data, t.NumElements())
	case gguf.GGML_TYPE_Q8_0:
		return dequantQ8_0(t.Data, t.NumElements())
	case gguf.GGML_TYPE_Q4_K:
		return DequantQ4_K(t.Data, t.NumElements())
	case gguf.GGML_TYPE_Q5_K:
		return DequantQ5_K(t.Data, t.NumElements())
	case gguf.GGML_TYPE_Q6_K:
		return DequantQ6_K(t.Data, t.NumElements())
	default:
		return nil, fmt.Errorf("ggml: dequantize not implemented for type %s", t.Type)
	}
}

// Row returns a single row as float32 values (dequantized).
// For a 2D tensor [rows, cols], returns cols float32 values for the given row index.
func (t *Tensor) Row(idx int) ([]float32, error) {
	cols := t.Cols()
	switch t.Type {
	case gguf.GGML_TYPE_F32:
		offset := idx * cols * 4
		return dequantF32(t.Data[offset:offset+cols*4], cols)
	case gguf.GGML_TYPE_Q8_0:
		// Q8_0: 32 elements per block, 34 bytes per block
		blocksPerRow := cols / 32
		bytesPerRow := blocksPerRow * 34
		offset := idx * bytesPerRow
		return dequantQ8_0(t.Data[offset:offset+bytesPerRow], cols)
	case gguf.GGML_TYPE_Q4_K:
		bytesPerRow := Q4_K_RowSize(cols)
		offset := idx * bytesPerRow
		return DequantQ4_K(t.Data[offset:offset+bytesPerRow], cols)
	case gguf.GGML_TYPE_Q5_K:
		bytesPerRow := Q5_K_RowSize(cols)
		offset := idx * bytesPerRow
		return DequantQ5_K(t.Data[offset:offset+bytesPerRow], cols)
	case gguf.GGML_TYPE_Q6_K:
		bytesPerRow := Q6_K_RowSize(cols)
		offset := idx * bytesPerRow
		return DequantQ6_K(t.Data[offset:offset+bytesPerRow], cols)
	default:
		return nil, fmt.Errorf("ggml: Row not implemented for type %s", t.Type)
	}
}

// --- F32 ---

// DequantF32Slice converts raw f32 bytes to float32 slice.
func DequantF32Slice(data []byte, n int) ([]float32, error) {
	return dequantF32(data, n)
}

func dequantF32(data []byte, n int) ([]float32, error) {
	if len(data) < n*4 {
		return nil, fmt.Errorf("ggml: f32 data too short: have %d bytes, need %d", len(data), n*4)
	}
	// Zero-copy reinterpret if aligned
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := *(*uint32)(unsafe.Pointer(&data[i*4]))
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// --- Q8_0 ---
// Format: blocks of 34 bytes each
//   - 2 bytes: f16 scale (delta)
//   - 32 bytes: 32 x int8 quantized values
// Dequant: value[i] = quant[i] * delta

// DequantQ8_0Slice converts raw Q8_0 bytes to float32 slice.
func DequantQ8_0Slice(data []byte, numElements int) ([]float32, error) {
	return dequantQ8_0(data, numElements)
}

func dequantQ8_0(data []byte, numElements int) ([]float32, error) {
	const blockSize = 32
	const bytesPerBlock = 34 // 2 (f16 scale) + 32 (int8 data)

	numBlocks := numElements / blockSize
	if len(data) < numBlocks*bytesPerBlock {
		return nil, fmt.Errorf("ggml: q8_0 data too short: have %d bytes, need %d",
			len(data), numBlocks*bytesPerBlock)
	}

	out := make([]float32, numElements)

	for b := 0; b < numBlocks; b++ {
		blockData := data[b*bytesPerBlock:]

		// Read f16 scale
		scaleBits := uint16(blockData[0]) | uint16(blockData[1])<<8
		scale := f16ToF32(scaleBits)

		// Dequantize 32 int8 values
		quants := blockData[2 : 2+blockSize]
		baseIdx := b * blockSize
		for i := 0; i < blockSize; i++ {
			q := int8(quants[i])
			out[baseIdx+i] = float32(q) * scale
		}
	}

	return out, nil
}

// f16ToF32 converts an IEEE 754 half-precision float to float32.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch {
	case exp == 0:
		if mant == 0 {
			// Zero
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		fallthrough
	case exp < 31:
		// Normal
		exp += 127 - 15
		bits := (sign << 31) | (exp << 23) | (mant << 13)
		return math.Float32frombits(bits)
	default:
		// Inf or NaN
		bits := (sign << 31) | (0xff << 23) | (mant << 13)
		return math.Float32frombits(bits)
	}
}

// NewTensorFromGGUF creates a Tensor from GGUF tensor info and raw data.
func NewTensorFromGGUF(info *gguf.TensorInfo, data []byte) *Tensor {
	shape := make([]int, len(info.Dims))
	for i, d := range info.Dims {
		shape[i] = int(d)
	}
	return &Tensor{
		Name:  info.Name,
		Shape: shape,
		Type:  info.Type,
		Data:  data,
	}
}

// Zeros creates a zero-filled F32 tensor with the given shape.
func Zeros(shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return &Tensor{
		Shape: shape,
		Type:  gguf.GGML_TYPE_F32,
		Data:  make([]byte, n*4),
	}
}

// FromFloat32 creates an F32 tensor from a float32 slice.
func FromFloat32(data []float32, shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(data) != n {
		panic(fmt.Sprintf("ggml: FromFloat32 shape mismatch: %d elements for shape %v", len(data), shape))
	}
	raw := make([]byte, n*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		*(*uint32)(unsafe.Pointer(&raw[i*4])) = bits
	}
	return &Tensor{
		Shape: shape,
		Type:  gguf.GGML_TYPE_F32,
		Data:  raw,
	}
}

// Float32s returns the tensor data as float32 (dequantizing if needed).
func (t *Tensor) Float32s() ([]float32, error) {
	return t.Dequantize()
}
