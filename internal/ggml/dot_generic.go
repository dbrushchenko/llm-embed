//go:build !amd64

package ggml

// hasAVX2 is always false on non-amd64 platforms.
var hasAVX2 = false

// dotQ8_0AVX2 is not available on non-amd64 platforms.
// The caller should check hasAVX2 before calling this.
func dotQ8_0AVX2(data *byte, x *float32, n int) float32 {
	panic("ggml: dotQ8_0AVX2 called on non-amd64 platform")
}
