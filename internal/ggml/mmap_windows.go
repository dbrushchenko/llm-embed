//go:build windows

package ggml

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// MmapedFile represents a memory-mapped file. The Data slice is backed by the OS page cache.
type MmapedFile struct {
	Data   []byte
	size   int
	handle syscall.Handle
}

// MmapOpen memory-maps a file for read-only access.
// The returned Data slice is valid until Close() is called.
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

	// Create file mapping
	sizeHigh := uint32(fi.Size() >> 32)
	sizeLow := uint32(fi.Size() & 0xFFFFFFFF)
	handle, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil,
		syscall.PAGE_READONLY, sizeHigh, sizeLow, nil)
	if err != nil {
		return nil, fmt.Errorf("mmap: CreateFileMapping: %w", err)
	}

	// Map view
	ptr, err := syscall.MapViewOfFile(handle, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		syscall.CloseHandle(handle)
		return nil, fmt.Errorf("mmap: MapViewOfFile: %w", err)
	}

	// Create slice backed by the mapped memory
	data := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), size)

	return &MmapedFile{Data: data, size: size, handle: handle}, nil
}

// Close unmaps the file from memory.
func (m *MmapedFile) Close() error {
	if m.Data == nil {
		return nil
	}
	err := syscall.UnmapViewOfFile(uintptr(unsafe.Pointer(&m.Data[0])))
	syscall.CloseHandle(m.handle)
	m.Data = nil
	return err
}

// Size returns the file size in bytes.
func (m *MmapedFile) Size() int {
	return m.size
}
