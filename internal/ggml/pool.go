package ggml

import "sync"

// BufferPool provides reusable float32 slices to reduce GC pressure.
// Used for intermediate computation results in the forward pass.
type BufferPool struct {
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// GlobalPool is the default buffer pool for inference operations.
var GlobalPool = NewBufferPool()

// NewBufferPool creates a new buffer pool.
func NewBufferPool() *BufferPool {
	return &BufferPool{
		pools: make(map[int]*sync.Pool),
	}
}

// Get returns a float32 slice of the given length from the pool.
// The slice contents are NOT zeroed — caller must fill or zero as needed.
func (bp *BufferPool) Get(n int) []float32 {
	bp.mu.RLock()
	pool, ok := bp.pools[n]
	bp.mu.RUnlock()

	if !ok {
		bp.mu.Lock()
		pool, ok = bp.pools[n]
		if !ok {
			pool = &sync.Pool{
				New: func() interface{} {
					buf := make([]float32, n)
					return &buf
				},
			}
			bp.pools[n] = pool
		}
		bp.mu.Unlock()
	}

	ptr := pool.Get().(*[]float32)
	return *ptr
}

// Put returns a float32 slice to the pool for reuse.
func (bp *BufferPool) Put(buf []float32) {
	n := cap(buf)
	bp.mu.RLock()
	pool, ok := bp.pools[n]
	bp.mu.RUnlock()

	if ok {
		b := buf[:n]
		pool.Put(&b)
	}
}

// GetBuf gets a buffer from the global pool.
func GetBuf(n int) []float32 {
	return GlobalPool.Get(n)
}

// PutBuf returns a buffer to the global pool.
func PutBuf(buf []float32) {
	GlobalPool.Put(buf)
}
