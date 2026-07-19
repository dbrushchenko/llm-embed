//go:build ignore

package main

import (
	"fmt"
	"math"
	"time"

	"github.com/dbrushchenko/llm-embed/internal/embed"
)

func main() {
	models := []string{"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}
	var vecs [][]float32
	for _, m := range models {
		e, err := embed.Load("models/nomic-embed-text-v1.5." + m + ".gguf")
		if err != nil {
			fmt.Printf("%-7s FAILED: %v\n", m, err)
			vecs = append(vecs, nil)
			continue
		}
		e.Embed("warmup")
		start := time.Now()
		var v []float32
		for i := 0; i < 3; i++ {
			v, _ = e.Embed("search_document: Kubernetes pods are scheduled by kube-scheduler")
		}
		vecs = append(vecs, v)
		fmt.Printf("%-7s %v/embed  dim=%d\n", m, time.Since(start)/3, len(v))
	}
	if vecs[3] != nil {
		fmt.Println("\nCosine vs Q8_0:")
		for i, m := range models[:3] {
			if vecs[i] == nil {
				continue
			}
			fmt.Printf("  %-7s %.4f\n", m, cosine(vecs[i], vecs[3]))
		}
	}
}

func cosine(a, b []float32) float32 {
	var d, na, nb float32
	for i := range a {
		d += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return d / (float32(math.Sqrt(float64(na))) * float32(math.Sqrt(float64(nb))))
}
