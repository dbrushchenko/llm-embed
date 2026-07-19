// Command inspect parses a GGUF file and prints metadata and tensor info.
// Usage: go run ./cmd/inspect <path-to-gguf>
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/dbrushchenko/llm-embed/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <file.gguf>\n", os.Args[0])
		os.Exit(1)
	}

	path := os.Args[1]
	f, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("GGUF v%d: %d tensors, %d KV pairs\n", f.Version, f.NumTensors, f.NumKV)
	fmt.Printf("Data offset: %d bytes\n\n", f.DataOffset)

	// Print KV pairs
	fmt.Println("=== Metadata ===")
	for i, kv := range f.KV {
		val := fmt.Sprintf("%v", kv.Value)
		if len(val) > 80 {
			val = val[:77] + "..."
		}
		fmt.Printf("  %3d: %-55s %-8s = %s\n", i, kv.Key, kv.Type, val)
	}

	// Print tensor info
	fmt.Printf("\n=== Tensors (%d) ===\n", f.NumTensors)
	var totalBytes uint64
	for _, t := range f.Tensors {
		dims := make([]string, len(t.Dims))
		for j, d := range t.Dims {
			dims[j] = fmt.Sprintf("%d", d)
		}
		size := t.ByteSize()
		totalBytes += size
		fmt.Printf("  %-50s [%s] %-5s %8.2f MiB  offset=%d\n",
			t.Name,
			strings.Join(dims, ", "),
			t.Type,
			float64(size)/1024/1024,
			t.Offset,
		)
	}
	fmt.Printf("\nTotal tensor data: %.2f MiB\n", float64(totalBytes)/1024/1024)

	// Print key model parameters
	fmt.Println("\n=== Model Parameters ===")
	if arch, ok := f.GetString("general.architecture"); ok {
		fmt.Printf("  Architecture: %s\n", arch)
	}
	if name, ok := f.GetString("general.name"); ok {
		fmt.Printf("  Name: %s\n", name)
	}

	arch, _ := f.GetString("general.architecture")
	prefix := arch + "."

	if v, ok := f.GetUint32(prefix + "block_count"); ok {
		fmt.Printf("  Layers: %d\n", v)
	}
	if v, ok := f.GetUint32(prefix + "embedding_length"); ok {
		fmt.Printf("  Embedding dim: %d\n", v)
	}
	if v, ok := f.GetUint32(prefix + "feed_forward_length"); ok {
		fmt.Printf("  FFN dim: %d\n", v)
	}
	if v, ok := f.GetUint32(prefix + "attention.head_count"); ok {
		fmt.Printf("  Heads: %d\n", v)
	}
	if v, ok := f.GetUint32(prefix + "attention.head_count_kv"); ok {
		fmt.Printf("  KV Heads: %d\n", v)
	}
	if v, ok := f.GetUint32(prefix + "context_length"); ok {
		fmt.Printf("  Context length: %d\n", v)
	}
}
