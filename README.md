# llm-embed

**15ms text embeddings in pure Go.** No Python. No ONNX. No CGO. Just `go build`.

A high-performance BERT embedding engine with hand-written AVX2/AVX-512 assembly kernels
that matches Intel MKL speed — from a single static binary with zero external dependencies.

### Highlights

- ⚡ **15ms per embedding** on Xeon (with optional MKL), **44ms pure Go**
- 🔧 **Zero dependencies** — `CGO_ENABLED=0`, fully static, `FROM scratch` Docker
- 🖥️ **Cross-platform** — Linux and Windows (amd64)
- 🧮 **4 quantization levels** — Q8_0, Q6_K, Q5_K_M, Q4_K_M (all under 42ms)
- 🏎️ **14× faster than ONNX Runtime** on the same hardware
- 🔬 **Hand-written Plan 9 assembly** — AVX2 6×16 + AVX-512 6×32 micro-kernels
- 📦 **Optional Intel MKL** — loaded at runtime via `dlopen`, falls back gracefully
- 🎯 **Production-tested** — deployed and serving live traffic

## Goal

A zero-dependency pure Go text embedding engine that matches or exceeds
ONNX Runtime performance via custom AVX2/AVX-512 assembly kernels and
optional Intel MKL acceleration at runtime.

- **Zero CGO** — `CGO_ENABLED=0`, fully static binary
- **Zero dependencies** — no ONNX, no Python, no shared libraries required
- **Optional MKL** — loads Intel MKL via `dlopen` at runtime for ~10× speedup (falls back to pure Go if unavailable)
- **`FROM scratch` compatible** — minimal Docker images

## Supported Models

**Nomic-embed-text-v1.5** (GGUF quantized)
- Architecture: BERT encoder (12 layers, 768-dim, 12 heads)
- Vocab: 30522 (WordPiece)
- Output: 768-dim normalized embeddings (Matryoshka: truncatable to 64-768)

| Model | Size | Quality (cosine vs F32) |
|-------|------|------------------------|
| Q8_0 | 139MB | 1.0000 (lossless) |
| Q6_K | 108MB | 0.9822 |
| Q5_K_M | 95MB | 0.9762 |
| Q4_K_M | 80MB | 0.9336 |

Download from [HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF).

## Architecture

```
text → WordPiece tokenize → [CLS] + tokens + [SEP]
  → Token embedding + position embedding + segment embedding
  → 12× BERT encoder layers:
      LayerNorm → MultiHeadAttention (12 heads, 64-dim each)
      → residual → LayerNorm → FFN (768→3072→768, GeLU)
      → residual
  → Mean pool (over attention_mask) → L2 normalize → [768] float32
```

## Performance

All measurements use nomic-embed-text-v1.5 GGUF models (~8 token input).

### Intel MKL (optional, via dlopen)

| Model | Laptop (AVX2) | Xeon Gold 6442Y | Cosine vs Q8_0 |
|-------|---------------|-----------------|----------------|
| Q8_0 (139MB) | 41ms | 15ms | 1.0000 |
| Q6_K (108MB) | 34ms | ~15ms | 0.9822 |
| Q5_K_M (95MB) | 35ms | ~15ms | 0.9762 |
| Q4_K_M (80MB) | 38ms | ~15ms | 0.9336 |

### Pure Go (zero dependencies)

| Model | Laptop (AVX2) | Xeon Gold 6442Y | Cosine vs Q8_0 |
|-------|---------------|-----------------|----------------|
| Q8_0 (139MB) | 44ms | ~140ms | 1.0000 |
| Q6_K (108MB) | 34ms | ~140ms | 0.9822 |
| Q5_K_M (95MB) | 35ms | ~140ms | 0.9762 |
| Q4_K_M (80MB) | 38ms | ~140ms | 0.9336 |

### Comparison

| Runtime | Latency (server) | Dependencies | Binary |
|---------|-----------------|--------------|--------|
| **llm-embed + MKL** | **15ms** | libmkl_rt.so (optional) | Static Go |
| **llm-embed (pure Go)** | **140ms** | **None** | Static Go |
| ONNX Runtime | 210-280ms | CGO + libonnxruntime (522MB) | Dynamic |

## Key Files

```
internal/
  embed/
    embedder.go        — Public API: Embed(text) → []float32
    bert.go            — BERT encoder forward pass (12 layers)
    attention.go       — Multi-head self-attention (batched Q·K^T·V)
    tokenizer.go       — WordPiece tokenizer (from GGUF vocab)
    pool.go            — Mean pooling + L2 normalize
  ggml/
    sgemm.go           — Tiled parallel GEMM (KC blocking, N→K→M ordering)
    gemm.go            — GemmPrePacked16 (AVX2 path)
    gemm_avx512.go     — GemmPrePacked32 (AVX-512 path)
    gemm_amd64.s       — AVX2 6×16 micro-kernel (Plan 9 assembly)
    gemm_avx512_amd64.s — AVX-512 6×32 micro-kernel
    dot_amd64.s        — Fused AVX2 dot kernels (Q4_K, Q5_K, Q6_K, Q8_0)
    mkl_linux.go       — Optional MKL via dlopen (no CGO)
    ops_simd.go        — In-place LayerNorm, GeLU, Softmax, RoPE table
    tensor.go          — Tensor struct, quantization dequant
  gguf/
    reader.go          — GGUF file parser
cmd/
  bench/main.go        — Benchmark tool
  inspect/main.go      — Model inspection tool
```

## API

```go
import "github.com/dbrushchenko/llm-embed/internal/embed"

e, err := embed.Load("nomic-embed-text-v1.5.Q8_0.gguf")
vec, err := e.Embed("search_document: Your text here")
// vec is []float32 of length 768, L2-normalized
```

## License

Apache-2.0. See [NOTICE](NOTICE) for third-party attributions.
