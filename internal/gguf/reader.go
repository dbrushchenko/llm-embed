// Package gguf implements a pure Go reader for the GGUF file format.
// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Magic bytes for GGUF files
const Magic = 0x46554747 // "GGUF" as little-endian uint32

// GGUF value types
type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

func (t ValueType) String() string {
	names := [...]string{
		"uint8", "int8", "uint16", "int16", "uint32", "int32",
		"float32", "bool", "string", "array", "uint64", "int64", "float64",
	}
	if int(t) < len(names) {
		return names[t]
	}
	return fmt.Sprintf("unknown(%d)", t)
}

// GGML tensor types (quantization formats)
type GGMLType uint32

const (
	GGML_TYPE_F32  GGMLType = 0
	GGML_TYPE_F16  GGMLType = 1
	GGML_TYPE_Q4_0 GGMLType = 2
	GGML_TYPE_Q4_1 GGMLType = 3
	GGML_TYPE_Q5_0 GGMLType = 6
	GGML_TYPE_Q5_1 GGMLType = 7
	GGML_TYPE_Q8_0 GGMLType = 8
	GGML_TYPE_Q8_1 GGMLType = 9
	GGML_TYPE_Q2_K GGMLType = 10
	GGML_TYPE_Q3_K GGMLType = 11
	GGML_TYPE_Q4_K GGMLType = 12
	GGML_TYPE_Q5_K GGMLType = 13
	GGML_TYPE_Q6_K GGMLType = 14
	GGML_TYPE_Q8_K GGMLType = 15
	GGML_TYPE_I8   GGMLType = 24
	GGML_TYPE_I16  GGMLType = 25
	GGML_TYPE_I32  GGMLType = 26
	GGML_TYPE_I64  GGMLType = 27
)

func (t GGMLType) String() string {
	switch t {
	case GGML_TYPE_F32:
		return "f32"
	case GGML_TYPE_F16:
		return "f16"
	case GGML_TYPE_Q4_0:
		return "q4_0"
	case GGML_TYPE_Q4_1:
		return "q4_1"
	case GGML_TYPE_Q5_0:
		return "q5_0"
	case GGML_TYPE_Q5_1:
		return "q5_1"
	case GGML_TYPE_Q8_0:
		return "q8_0"
	case GGML_TYPE_Q8_1:
		return "q8_1"
	case GGML_TYPE_Q2_K:
		return "q2_k"
	case GGML_TYPE_Q3_K:
		return "q3_k"
	case GGML_TYPE_Q4_K:
		return "q4_k"
	case GGML_TYPE_Q5_K:
		return "q5_k"
	case GGML_TYPE_Q6_K:
		return "q6_k"
	case GGML_TYPE_Q8_K:
		return "q8_k"
	case GGML_TYPE_I32:
		return "i32"
	default:
		return fmt.Sprintf("type_%d", t)
	}
}

// BlockSize returns the number of elements per quantization block.
func (t GGMLType) BlockSize() int {
	switch t {
	case GGML_TYPE_F32:
		return 1
	case GGML_TYPE_F16:
		return 1
	case GGML_TYPE_Q4_0:
		return 32
	case GGML_TYPE_Q4_1:
		return 32
	case GGML_TYPE_Q5_0:
		return 32
	case GGML_TYPE_Q5_1:
		return 32
	case GGML_TYPE_Q8_0:
		return 32
	case GGML_TYPE_Q8_1:
		return 32
	case GGML_TYPE_Q4_K:
		return 256
	case GGML_TYPE_Q5_K:
		return 256
	case GGML_TYPE_Q6_K:
		return 256
	case GGML_TYPE_I32:
		return 1
	default:
		return 1
	}
}

// TypeSize returns bytes per block for a given GGML type.
func (t GGMLType) TypeSize() int {
	switch t {
	case GGML_TYPE_F32:
		return 4
	case GGML_TYPE_F16:
		return 2
	case GGML_TYPE_Q4_0:
		return 2 + 16 // scale (f16) + 16 bytes for 32 4-bit values
	case GGML_TYPE_Q4_1:
		return 2 + 2 + 16 // scale + min + data
	case GGML_TYPE_Q5_0:
		return 2 + 4 + 16 // scale + high bits + data
	case GGML_TYPE_Q5_1:
		return 2 + 2 + 4 + 16
	case GGML_TYPE_Q8_0:
		return 2 + 32 // scale (f16) + 32 bytes for 32 int8 values = 34 bytes/block
	case GGML_TYPE_Q8_1:
		return 4 + 4 + 32 // scale + sum + data
	case GGML_TYPE_Q4_K:
		return 144 // complex format
	case GGML_TYPE_Q5_K:
		return 176
	case GGML_TYPE_Q6_K:
		return 210
	case GGML_TYPE_I32:
		return 4
	default:
		return 0
	}
}

// KVPair is a key-value metadata entry in the GGUF file.
type KVPair struct {
	Key   string
	Type  ValueType
	Value interface{}
}

// TensorInfo describes a tensor stored in the GGUF file.
type TensorInfo struct {
	Name   string
	NDims  uint32
	Dims   []uint64
	Type   GGMLType
	Offset uint64
}

// NumElements returns the total number of elements in the tensor.
func (t *TensorInfo) NumElements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// ByteSize returns the total size in bytes for this tensor's data.
func (t *TensorInfo) ByteSize() uint64 {
	nel := t.NumElements()
	bs := uint64(t.Type.BlockSize())
	ts := uint64(t.Type.TypeSize())
	if bs == 0 || ts == 0 {
		return 0
	}
	return (nel / bs) * ts
}

// File represents a parsed GGUF file.
type File struct {
	Version    uint32
	NumTensors uint64
	NumKV      uint64
	KV         []KVPair
	Tensors    []TensorInfo
	DataOffset uint64 // byte offset where tensor data begins
	Path       string
}

// Open parses a GGUF file and returns the metadata and tensor info.
// It does NOT load tensor data into memory.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %s: %w", path, err)
	}
	defer f.Close()

	r := &reader{r: f, order: binary.LittleEndian}

	// Header
	magic := r.u32()
	if magic != Magic {
		return nil, fmt.Errorf("gguf: invalid magic: 0x%08x (expected 0x%08x)", magic, Magic)
	}

	version := r.u32()
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version: %d", version)
	}

	numTensors := r.u64()
	numKV := r.u64()

	if r.err != nil {
		return nil, fmt.Errorf("gguf: reading header: %w", r.err)
	}

	// KV pairs
	kvs := make([]KVPair, 0, numKV)
	for i := uint64(0); i < numKV; i++ {
		kv := r.readKV()
		if r.err != nil {
			return nil, fmt.Errorf("gguf: reading KV pair %d: %w", i, r.err)
		}
		kvs = append(kvs, kv)
	}

	// Tensor infos
	tensors := make([]TensorInfo, 0, numTensors)
	for i := uint64(0); i < numTensors; i++ {
		ti := r.readTensorInfo()
		if r.err != nil {
			return nil, fmt.Errorf("gguf: reading tensor %d: %w", i, r.err)
		}
		tensors = append(tensors, ti)
	}

	// Data offset: aligned to 32 bytes after the metadata
	pos, _ := f.Seek(0, io.SeekCurrent)
	alignment := uint64(32)
	dataOffset := (uint64(pos) + alignment - 1) & ^(alignment - 1)

	return &File{
		Version:    version,
		NumTensors: numTensors,
		NumKV:      numKV,
		KV:         kvs,
		Tensors:    tensors,
		DataOffset: dataOffset,
		Path:       path,
	}, nil
}

// GetKV looks up a KV pair by key name. Returns nil if not found.
func (f *File) GetKV(key string) *KVPair {
	for i := range f.KV {
		if f.KV[i].Key == key {
			return &f.KV[i]
		}
	}
	return nil
}

// GetUint32 retrieves a uint32 KV value by key.
func (f *File) GetUint32(key string) (uint32, bool) {
	kv := f.GetKV(key)
	if kv == nil {
		return 0, false
	}
	switch v := kv.Value.(type) {
	case uint32:
		return v, true
	default:
		return 0, false
	}
}

// GetString retrieves a string KV value by key.
func (f *File) GetString(key string) (string, bool) {
	kv := f.GetKV(key)
	if kv == nil {
		return "", false
	}
	switch v := kv.Value.(type) {
	case string:
		return v, true
	default:
		return "", false
	}
}

// GetFloat32 retrieves a float32 KV value by key.
func (f *File) GetFloat32(key string) (float32, bool) {
	kv := f.GetKV(key)
	if kv == nil {
		return 0, false
	}
	switch v := kv.Value.(type) {
	case float32:
		return v, true
	default:
		return 0, false
	}
}

// GetTensor looks up a tensor by name. Returns nil if not found.
func (f *File) GetTensor(name string) *TensorInfo {
	for i := range f.Tensors {
		if f.Tensors[i].Name == name {
			return &f.Tensors[i]
		}
	}
	return nil
}

// --- internal reader ---

type reader struct {
	r     io.ReadSeeker
	order binary.ByteOrder
	err   error
}

func (r *reader) u8() uint8 {
	if r.err != nil {
		return 0
	}
	var v uint8
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) u16() uint16 {
	if r.err != nil {
		return 0
	}
	var v uint16
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) u32() uint32 {
	if r.err != nil {
		return 0
	}
	var v uint32
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) u64() uint64 {
	if r.err != nil {
		return 0
	}
	var v uint64
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) i32() int32 {
	if r.err != nil {
		return 0
	}
	var v int32
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) f32() float32 {
	if r.err != nil {
		return 0
	}
	var v float32
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) f64() float64 {
	if r.err != nil {
		return 0
	}
	var v float64
	r.err = binary.Read(r.r, r.order, &v)
	return v
}

func (r *reader) str() string {
	if r.err != nil {
		return ""
	}
	length := r.u64()
	if r.err != nil {
		return ""
	}
	buf := make([]byte, length)
	_, r.err = io.ReadFull(r.r, buf)
	return string(buf)
}

func (r *reader) readKV() KVPair {
	key := r.str()
	vtype := ValueType(r.u32())
	value := r.readValue(vtype)
	return KVPair{Key: key, Type: vtype, Value: value}
}

func (r *reader) readValue(vtype ValueType) interface{} {
	if r.err != nil {
		return nil
	}
	switch vtype {
	case TypeUint8:
		return r.u8()
	case TypeInt8:
		return int8(r.u8())
	case TypeUint16:
		return r.u16()
	case TypeInt16:
		return int16(r.u16())
	case TypeUint32:
		return r.u32()
	case TypeInt32:
		return r.i32()
	case TypeFloat32:
		return r.f32()
	case TypeBool:
		return r.u8() != 0
	case TypeString:
		return r.str()
	case TypeArray:
		return r.readArray()
	case TypeUint64:
		return r.u64()
	case TypeInt64:
		return int64(r.u64())
	case TypeFloat64:
		return r.f64()
	default:
		r.err = fmt.Errorf("unknown value type: %d", vtype)
		return nil
	}
}

func (r *reader) readArray() interface{} {
	elemType := ValueType(r.u32())
	length := r.u64()
	if r.err != nil {
		return nil
	}

	// For large arrays (tokenizer data), skip reading into memory
	// Just record the type and length
	switch elemType {
	case TypeString:
		if length > 10000 {
			// Skip large string arrays (tokenizer tokens/merges)
			for i := uint64(0); i < length; i++ {
				_ = r.str()
				if r.err != nil {
					return nil
				}
			}
			return fmt.Sprintf("[%d strings]", length)
		}
		arr := make([]string, length)
		for i := uint64(0); i < length; i++ {
			arr[i] = r.str()
		}
		return arr
	case TypeFloat32:
		if length > 10000 {
			// Skip large float arrays (tokenizer scores)
			buf := make([]byte, length*4)
			_, r.err = io.ReadFull(r.r, buf)
			return fmt.Sprintf("[%d float32s]", length)
		}
		arr := make([]float32, length)
		for i := uint64(0); i < length; i++ {
			arr[i] = r.f32()
		}
		return arr
	case TypeInt32:
		if length > 10000 {
			buf := make([]byte, length*4)
			_, r.err = io.ReadFull(r.r, buf)
			return fmt.Sprintf("[%d int32s]", length)
		}
		arr := make([]int32, length)
		for i := uint64(0); i < length; i++ {
			arr[i] = r.i32()
		}
		return arr
	case TypeUint32:
		if length > 10000 {
			buf := make([]byte, length*4)
			_, r.err = io.ReadFull(r.r, buf)
			return fmt.Sprintf("[%d uint32s]", length)
		}
		arr := make([]uint32, length)
		for i := uint64(0); i < length; i++ {
			arr[i] = r.u32()
		}
		return arr
	case TypeBool:
		arr := make([]bool, length)
		for i := uint64(0); i < length; i++ {
			arr[i] = r.u8() != 0
		}
		return arr
	default:
		// Skip unknown array types
		for i := uint64(0); i < length; i++ {
			_ = r.readValue(elemType)
		}
		return fmt.Sprintf("[%d %s]", length, elemType)
	}
}

func (r *reader) readTensorInfo() TensorInfo {
	name := r.str()
	ndims := r.u32()
	dims := make([]uint64, ndims)
	for i := uint32(0); i < ndims; i++ {
		dims[i] = r.u64()
	}
	dtype := GGMLType(r.u32())
	offset := r.u64()
	return TensorInfo{
		Name:   name,
		NDims:  ndims,
		Dims:   dims,
		Type:   dtype,
		Offset: offset,
	}
}
