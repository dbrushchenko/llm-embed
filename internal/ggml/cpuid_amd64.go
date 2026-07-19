//go:build amd64

package ggml

// cpuid executes the CPUID instruction with the given leaf (eaxArg) and sub-leaf (ecxArg).
//
//go:noescape
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

// xgetbv reads the extended control register specified by index.
//
//go:noescape
func xgetbv(index uint32) uint64
