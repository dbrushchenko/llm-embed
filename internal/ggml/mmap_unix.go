//go:build linux || darwin

package ggml

import (
	"fmt"
	"os"
	"syscall"
)

// MmapedFile represents a memory-mapped file. The Data slice is backed by the OS page cache.
type MmapedFile struct {
	Data []byte
	size int
}

// MmapOpen memory-maps a file for read-only access.
// The returned Data slice is valid until Close() is called.
// Benefits: zero-copy load, OS manages paging, no heap allocation for model weights.
func MmapOpen(path string) (*MmapedFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap: open: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("mmap: stat: %w", err)
	}

	size := int(fi.Size())
	if size == 0 {
		return nil, fmt.Errorf("mmap: file is empty: %s", path)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: mmap: %w", err)
	}

	return &MmapedFile{Data: data, size: size}, nil
}

// Close unmaps the file from memory.
func (m *MmapedFile) Close() error {
	if m.Data == nil {
		return nil
	}
	err := syscall.Munmap(m.Data)
	m.Data = nil
	return err
}

// Size returns the file size in bytes.
func (m *MmapedFile) Size() int {
	return m.size
}
