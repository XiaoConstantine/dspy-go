package cache

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryCache_BasicOperations(t *testing.T) {
	config := CacheConfig{
		Type:       "memory",
		MaxSize:    1024,
		DefaultTTL: time.Hour,
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("Set and Get", func(t *testing.T) {
		key := "test-key"
		value := []byte("test-value")

		// Set value
		err := cache.Set(ctx, key, value, 0)
		assert.NoError(t, err)

		// Get value
		retrieved, found, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)
	})

	t.Run("Get non-existent key", func(t *testing.T) {
		retrieved, found, err := cache.Get(ctx, "non-existent")
		assert.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, retrieved)
	})

	t.Run("Delete", func(t *testing.T) {
		key := "delete-key"
		value := []byte("delete-value")

		// Set and verify
		err := cache.Set(ctx, key, value, 0)
		assert.NoError(t, err)

		_, found, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)

		// Delete
		err = cache.Delete(ctx, key)
		assert.NoError(t, err)

		// Verify deleted
		_, found, err = cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.False(t, found)
	})

	t.Run("Clear", func(t *testing.T) {
		// Set multiple values
		for i := 0; i < 5; i++ {
			key := "clear-key-" + string(rune('0'+i))
			value := []byte("clear-value-" + string(rune('0'+i)))
			err := cache.Set(ctx, key, value, 0)
			assert.NoError(t, err)
		}

		// Clear all
		err := cache.Clear(ctx)
		assert.NoError(t, err)

		// Verify all cleared
		for i := 0; i < 5; i++ {
			key := "clear-key-" + string(rune('0'+i))
			retrieved, found, err := cache.Get(ctx, key)
			assert.NoError(t, err)
			assert.False(t, found)
			assert.Nil(t, retrieved)
		}
	})
}

func TestMemoryCache_TTL(t *testing.T) {
	config := CacheConfig{
		Type:       "memory",
		MaxSize:    1024,
		DefaultTTL: 100 * time.Millisecond,
		MemoryConfig: MemoryConfig{
			CleanupInterval: 50 * time.Millisecond,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("TTL expiration", func(t *testing.T) {
		key := "ttl-key"
		value := []byte("ttl-value")

		// Set with short TTL
		err := cache.Set(ctx, key, value, 50*time.Millisecond)
		assert.NoError(t, err)

		// Should be available immediately
		retrieved, found, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)

		// Wait for expiration
		time.Sleep(100 * time.Millisecond)

		// Should be expired
		retrieved, found, err = cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, retrieved)
	})

	t.Run("Default TTL", func(t *testing.T) {
		key := "default-ttl-key"
		value := []byte("default-ttl-value")

		// Set with no TTL (should use default)
		err := cache.Set(ctx, key, value, 0)
		assert.NoError(t, err)

		// Should be available
		retrieved, found, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)

		// Wait for default TTL to expire
		time.Sleep(150 * time.Millisecond)

		// Should be expired
		retrieved, found, err = cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, retrieved)
	})

	t.Run("No TTL", func(t *testing.T) {
		// Create cache with no default TTL
		configNoTTL := CacheConfig{
			Type:    "memory",
			MaxSize: 1024,
			MemoryConfig: MemoryConfig{
				CleanupInterval: 50 * time.Millisecond,
			},
		}

		cacheNoTTL, err := NewMemoryCache(configNoTTL)
		require.NoError(t, err)
		defer cacheNoTTL.Close()

		key := "no-ttl-key"
		value := []byte("no-ttl-value")

		// Set with no TTL
		err = cacheNoTTL.Set(ctx, key, value, 0)
		assert.NoError(t, err)

		// Wait and verify still available
		time.Sleep(100 * time.Millisecond)

		retrieved, found, err := cacheNoTTL.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, value, retrieved)
	})
}

func TestMemoryCache_SizeLimit(t *testing.T) {
	config := CacheConfig{
		Type:    "memory",
		MaxSize: 100, // Small size for testing
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("Value too large", func(t *testing.T) {
		key := "large-key"
		value := make([]byte, 200) // Larger than max size

		err := cache.Set(ctx, key, value, 0)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds max cache size")
	})

	t.Run("LRU eviction", func(t *testing.T) {
		// Add values that will exceed the size limit
		key1 := "key1"
		value1 := make([]byte, 40)
		err := cache.Set(ctx, key1, value1, 0)
		assert.NoError(t, err)

		key2 := "key2"
		value2 := make([]byte, 40)
		err = cache.Set(ctx, key2, value2, 0)
		assert.NoError(t, err)

		// This should trigger eviction of key1 (LRU)
		key3 := "key3"
		value3 := make([]byte, 40)
		err = cache.Set(ctx, key3, value3, 0)
		assert.NoError(t, err)

		// key1 should be evicted
		_, found, err := cache.Get(ctx, key1)
		assert.NoError(t, err)
		assert.False(t, found)

		// key2 and key3 should still be there
		_, found, err = cache.Get(ctx, key2)
		assert.NoError(t, err)
		assert.True(t, found)

		_, found, err = cache.Get(ctx, key3)
		assert.NoError(t, err)
		assert.True(t, found)
	})

	t.Run("Update existing key", func(t *testing.T) {
		key := "update-key"
		value1 := []byte("small")
		value2 := make([]byte, 50)

		// Set initial value
		err := cache.Set(ctx, key, value1, 0)
		assert.NoError(t, err)

		// Update with larger value
		err = cache.Set(ctx, key, value2, 0)
		assert.NoError(t, err)

		// Verify updated
		retrieved, found, err := cache.Get(ctx, key)
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, value2, retrieved)
	})
}

func TestMemoryCache_Stats(t *testing.T) {
	config := CacheConfig{
		Type:    "memory",
		MaxSize: 1024,
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Initial stats
	stats := cache.Stats()
	assert.Equal(t, int64(0), stats.Hits)
	assert.Equal(t, int64(0), stats.Misses)
	assert.Equal(t, int64(0), stats.Sets)
	assert.Equal(t, int64(0), stats.Deletes)
	assert.Equal(t, int64(0), stats.Size)
	assert.Equal(t, int64(1024), stats.MaxSize)

	// Set operation
	key := "stats-key"
	value := []byte("stats-value")
	err = cache.Set(ctx, key, value, 0)
	assert.NoError(t, err)

	stats = cache.Stats()
	assert.Equal(t, int64(1), stats.Sets)
	assert.Equal(t, int64(len(value)), stats.Size)

	// Hit
	_, found, err := cache.Get(ctx, key)
	assert.NoError(t, err)
	assert.True(t, found)

	stats = cache.Stats()
	assert.Equal(t, int64(1), stats.Hits)

	// Miss
	_, found, err = cache.Get(ctx, "non-existent")
	assert.NoError(t, err)
	assert.False(t, found)

	stats = cache.Stats()
	assert.Equal(t, int64(1), stats.Misses)

	// Delete
	err = cache.Delete(ctx, key)
	assert.NoError(t, err)

	stats = cache.Stats()
	assert.Equal(t, int64(1), stats.Deletes)
	assert.Equal(t, int64(0), stats.Size)
}

func TestMemoryCache_LRU(t *testing.T) {
	config := CacheConfig{
		Type:    "memory",
		MaxSize: 200,
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Add entries
	key1 := "lru1"
	value1 := make([]byte, 50)
	err = cache.Set(ctx, key1, value1, 0)
	assert.NoError(t, err)

	key2 := "lru2"
	value2 := make([]byte, 50)
	err = cache.Set(ctx, key2, value2, 0)
	assert.NoError(t, err)

	key3 := "lru3"
	value3 := make([]byte, 50)
	err = cache.Set(ctx, key3, value3, 0)
	assert.NoError(t, err)

	// Access key1 to make it recently used
	_, found, err := cache.Get(ctx, key1)
	assert.NoError(t, err)
	assert.True(t, found)

	// Add another entry that should evict key2 (least recently used)
	key4 := "lru4"
	value4 := make([]byte, 70)
	err = cache.Set(ctx, key4, value4, 0)
	assert.NoError(t, err)

	// key2 should be evicted
	_, found, err = cache.Get(ctx, key2)
	assert.NoError(t, err)
	assert.False(t, found)

	// key1, key3, key4 should still be there
	_, found, err = cache.Get(ctx, key1)
	assert.NoError(t, err)
	assert.True(t, found)

	_, found, err = cache.Get(ctx, key3)
	assert.NoError(t, err)
	assert.True(t, found)

	_, found, err = cache.Get(ctx, key4)
	assert.NoError(t, err)
	assert.True(t, found)
}

func TestMemoryCache_ExportImport(t *testing.T) {
	config := CacheConfig{
		Type:    "memory",
		MaxSize: 1024,
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Set some values
	testData := map[string][]byte{
		"export1": []byte("value1"),
		"export2": []byte("value2"),
		"export3": []byte("value3"),
	}

	for key, value := range testData {
		err := cache.Set(ctx, key, value, time.Hour)
		assert.NoError(t, err)
	}

	t.Run("Export", func(t *testing.T) {
		var exported []CacheEntry
		err := cache.Export(ctx, func(entry CacheEntry) error {
			exported = append(exported, entry)
			return nil
		})
		assert.NoError(t, err)
		assert.Len(t, exported, 3)

		// Verify exported data
		exportedMap := make(map[string][]byte)
		for _, entry := range exported {
			exportedMap[entry.Key] = entry.Value
			assert.False(t, entry.ExpiresAt.IsZero())
			assert.False(t, entry.CreatedAt.IsZero())
			assert.True(t, entry.Size > 0)
		}

		for key, expectedValue := range testData {
			actualValue, exists := exportedMap[key]
			assert.True(t, exists)
			assert.Equal(t, expectedValue, actualValue)
		}
	})

	t.Run("Import", func(t *testing.T) {
		// Create new cache
		newCache, err := NewMemoryCache(config)
		require.NoError(t, err)
		defer newCache.Close()

		// Prepare import data
		importData := []CacheEntry{
			{
				Key:        "import1",
				Value:      []byte("import-value1"),
				ExpiresAt:  time.Now().Add(time.Hour),
				CreatedAt:  time.Now(),
				Size:       int64(len([]byte("import-value1"))),
			},
			{
				Key:        "import2",
				Value:      []byte("import-value2"),
				ExpiresAt:  time.Now().Add(time.Hour),
				CreatedAt:  time.Now(),
				Size:       int64(len([]byte("import-value2"))),
			},
			{
				Key:       "expired",
				Value:     []byte("expired-value"),
				ExpiresAt: time.Now().Add(-time.Hour), // Already expired
				CreatedAt: time.Now(),
				Size:      int64(len([]byte("expired-value"))),
			},
		}

		// Import
		err = newCache.Import(ctx, importData)
		assert.NoError(t, err)

		// Verify imported data (except expired entry)
		retrieved, found, err := newCache.Get(ctx, "import1")
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, []byte("import-value1"), retrieved)

		retrieved, found, err = newCache.Get(ctx, "import2")
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, []byte("import-value2"), retrieved)

		// Expired entry should not be imported
		_, found, err = newCache.Get(ctx, "expired")
		assert.NoError(t, err)
		assert.False(t, found)
	})
}

func TestMemoryCache_Cleanup(t *testing.T) {
	config := CacheConfig{
		Type:       "memory",
		MaxSize:    1024,
		DefaultTTL: 50 * time.Millisecond,
		MemoryConfig: MemoryConfig{
			CleanupInterval: 25 * time.Millisecond,
		},
	}

	cache, err := NewMemoryCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Set values with short TTL
	err = cache.Set(ctx, "cleanup1", []byte("value1"), 40*time.Millisecond)
	assert.NoError(t, err)

	err = cache.Set(ctx, "cleanup2", []byte("value2"), 40*time.Millisecond)
	assert.NoError(t, err)

	// Values should be available initially
	_, found, err := cache.Get(ctx, "cleanup1")
	assert.NoError(t, err)
	assert.True(t, found)

	// Wait for cleanup to happen
	time.Sleep(100 * time.Millisecond)

	// Values should be cleaned up
	_, found, err = cache.Get(ctx, "cleanup1")
	assert.NoError(t, err)
	assert.False(t, found)

	_, found, err = cache.Get(ctx, "cleanup2")
	assert.NoError(t, err)
	assert.False(t, found)
}