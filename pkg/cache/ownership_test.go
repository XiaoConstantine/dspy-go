package cache

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestMemoryCache(t *testing.T, maxSize int64) *MemoryCache {
	t.Helper()
	cache, err := NewMemoryCache(CacheConfig{
		Type:    "memory",
		MaxSize: maxSize,
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	})
	require.NoError(t, err)
	t.Cleanup(func() { cache.Close() })
	return cache
}

func newTestSQLiteCache(t *testing.T, maxSize int64) *SQLiteCache {
	t.Helper()
	cache, err := NewSQLiteCache(CacheConfig{
		Type:    "sqlite",
		MaxSize: maxSize,
		SQLiteConfig: SQLiteConfig{
			Path: t.TempDir() + "/cache.db",
		},
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, cache.Close()) })
	return cache
}

func sqliteStoredSize(t *testing.T, cache *SQLiteCache) int64 {
	t.Helper()
	var size int64
	require.NoError(t, cache.db.QueryRow(`SELECT COALESCE(SUM(size), 0) FROM cache_entries`).Scan(&size))
	return size
}

func TestMemoryCache_ValueIsolation(t *testing.T) {
	cache := newTestMemoryCache(t, 1024)
	ctx := context.Background()

	original := []byte("immutable")
	require.NoError(t, cache.Set(ctx, "key", original, 0))

	// Mutating the slice passed to Set must not affect the cached value.
	original[0] = 'X'

	got, found, err := cache.Get(ctx, "key")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, []byte("immutable"), got)

	// Mutating the slice returned by Get must not affect later reads.
	got[0] = 'Y'

	again, found, err := cache.Get(ctx, "key")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, []byte("immutable"), again)
}

func TestMemoryCache_ReplacementRespectsMaxSize(t *testing.T) {
	cache := newTestMemoryCache(t, 100)
	ctx := context.Background()

	require.NoError(t, cache.Set(ctx, "a", make([]byte, 40), 0))
	require.NoError(t, cache.Set(ctx, "b", make([]byte, 40), 0))

	// Growing "b" to 90 bytes must evict "a" rather than exceed MaxSize.
	require.NoError(t, cache.Set(ctx, "b", make([]byte, 90), 0))

	stats := cache.Stats()
	assert.LessOrEqual(t, stats.Size, int64(100), "cache size must stay within MaxSize after replacement")

	got, found, err := cache.Get(ctx, "b")
	require.NoError(t, err)
	require.True(t, found, "the updated entry must survive its own growth eviction")
	assert.Len(t, got, 90)
}

func TestSQLiteCache_RejectsValueLargerThanMaxSize(t *testing.T) {
	cache := newTestSQLiteCache(t, 100)
	ctx := context.Background()
	require.NoError(t, cache.Set(ctx, "existing", make([]byte, 40), 0))

	err := cache.Set(ctx, "existing", make([]byte, 101), 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds max cache size")

	value, found, getErr := cache.Get(ctx, "existing")
	require.NoError(t, getErr)
	require.True(t, found)
	assert.Len(t, value, 40)
	assert.Equal(t, int64(40), cache.Stats().Size)
	assert.Equal(t, int64(40), sqliteStoredSize(t, cache))
}

func TestSQLiteCache_ReplacementEvictsOtherEntries(t *testing.T) {
	cache := newTestSQLiteCache(t, 100)
	ctx := context.Background()
	require.NoError(t, cache.Set(ctx, "updated", make([]byte, 40), 0))
	require.NoError(t, cache.Set(ctx, "other", make([]byte, 40), 0))

	require.NoError(t, cache.Set(ctx, "updated", make([]byte, 90), 0))

	value, found, err := cache.Get(ctx, "updated")
	require.NoError(t, err)
	require.True(t, found, "an entry must not evict itself while it is being replaced")
	assert.Len(t, value, 90)
	_, found, err = cache.Get(ctx, "other")
	require.NoError(t, err)
	assert.False(t, found)
	assert.Equal(t, int64(90), cache.Stats().Size)
	assert.Equal(t, int64(90), sqliteStoredSize(t, cache))
}

func TestSQLiteCache_ConcurrentSetsRespectMaxSize(t *testing.T) {
	cache := newTestSQLiteCache(t, 100)
	ctx := context.Background()
	start := make(chan struct{})
	const writers = 20
	errs := make(chan error, writers)

	var wg sync.WaitGroup
	for i := range writers {
		wg.Go(func() {
			<-start
			errs <- cache.Set(ctx, fmt.Sprintf("key-%d", i), make([]byte, 30), 0)
		})
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	assert.LessOrEqual(t, cache.Stats().Size, int64(100))
	assert.LessOrEqual(t, sqliteStoredSize(t, cache), int64(100))
}

func TestSQLiteCache_ImportRespectsMaxSize(t *testing.T) {
	cache := newTestSQLiteCache(t, 100)
	entries := []CacheEntry{
		{Key: "one", Value: make([]byte, 60)},
		{Key: "two", Value: make([]byte, 60)},
		{Key: "three", Value: make([]byte, 60)},
	}

	require.NoError(t, cache.Import(context.Background(), entries))
	assert.LessOrEqual(t, cache.Stats().Size, int64(100))
	assert.LessOrEqual(t, sqliteStoredSize(t, cache), int64(100))
}

func TestMemoryCache_ExportValueIsolation(t *testing.T) {
	cache := newTestMemoryCache(t, 1024)
	ctx := context.Background()

	require.NoError(t, cache.Set(ctx, "key", []byte("immutable"), 0))

	// Mutating the exported value must not affect the cached bytes.
	require.NoError(t, cache.Export(ctx, func(entry CacheEntry) error {
		entry.Value[0] = 'X'
		return nil
	}))

	got, found, err := cache.Get(ctx, "key")
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, []byte("immutable"), got)
}

func TestMemoryCache_ExportWriterMayReenterCache(t *testing.T) {
	cache := newTestMemoryCache(t, 1024)
	ctx := context.Background()

	require.NoError(t, cache.Set(ctx, "key", []byte("value"), 0))

	// The writer runs outside the cache lock, so calling back into the
	// cache must not deadlock.
	done := make(chan error, 1)
	go func() {
		done <- cache.Export(ctx, func(entry CacheEntry) error {
			if _, _, err := cache.Get(ctx, entry.Key); err != nil {
				return err
			}
			return cache.Set(ctx, "reentrant-"+entry.Key, entry.Value, 0)
		})
	}()

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("Export writer re-entering the cache deadlocked")
	}

	_, found, err := cache.Get(ctx, "reentrant-key")
	require.NoError(t, err)
	assert.True(t, found)
}

func TestMemoryCache_CloseIsIdempotent(t *testing.T) {
	cache := newTestMemoryCache(t, 1024)
	require.NoError(t, cache.Close())
	require.NotPanics(t, func() {
		require.NoError(t, cache.Close())
	})
}

func TestSQLiteCache_CloseIsIdempotent(t *testing.T) {
	cache, err := NewSQLiteCache(CacheConfig{
		Type: "sqlite",
		SQLiteConfig: SQLiteConfig{
			Path: t.TempDir() + "/cache.db",
		},
	})
	require.NoError(t, err)
	require.NoError(t, cache.Close())
	require.NotPanics(t, func() {
		require.NoError(t, cache.Close())
	})
}

// closeConcurrently calls Close from several goroutines released by a common
// start barrier and reports every returned error.
func closeConcurrently(t *testing.T, closeFn func() error) []error {
	t.Helper()

	const callers = 8
	start := make(chan struct{})
	errs := make([]error, callers)

	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errs[i] = closeFn()
		}(i)
	}
	close(start)
	wg.Wait()
	return errs
}

func TestMemoryCache_ConcurrentClose(t *testing.T) {
	cache := newTestMemoryCache(t, 1024)
	for _, err := range closeConcurrently(t, cache.Close) {
		assert.NoError(t, err)
	}
}

func TestSQLiteCache_ConcurrentClose(t *testing.T) {
	cache, err := NewSQLiteCache(CacheConfig{
		Type: "sqlite",
		SQLiteConfig: SQLiteConfig{
			Path: t.TempDir() + "/cache.db",
		},
	})
	require.NoError(t, err)

	// Every concurrent caller must observe the same (first) close result.
	for _, err := range closeConcurrently(t, cache.Close) {
		assert.NoError(t, err)
	}
}
