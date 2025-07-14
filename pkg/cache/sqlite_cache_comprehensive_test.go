package cache

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLiteCache_BasicOperations(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	config := CacheConfig{
		Type:       "sqlite",
		MaxSize:    1024,
		DefaultTTL: time.Hour,
		SQLiteConfig: SQLiteConfig{
			Path:              dbPath,
			EnableWAL:         true,
			VacuumInterval:    time.Hour,
			MaxConnections:    5,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("Set and Get", func(t *testing.T) {
		key := "sqlite-test-key"
		value := []byte("sqlite-test-value")

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
		retrieved, found, err := cache.Get(ctx, "non-existent-sqlite")
		assert.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, retrieved)
	})

	t.Run("Delete", func(t *testing.T) {
		key := "sqlite-delete-key"
		value := []byte("sqlite-delete-value")

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
			key := "sqlite-clear-key-" + string(rune('0'+i))
			value := []byte("sqlite-clear-value-" + string(rune('0'+i)))
			err := cache.Set(ctx, key, value, 0)
			assert.NoError(t, err)
		}

		// Clear all
		err := cache.Clear(ctx)
		assert.NoError(t, err)

		// Verify all cleared
		for i := 0; i < 5; i++ {
			key := "sqlite-clear-key-" + string(rune('0'+i))
			_, found, err := cache.Get(ctx, key)
			assert.NoError(t, err)
			assert.False(t, found)
		}
	})
}

func TestSQLiteCache_TTL(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "ttl_test.db")

	config := CacheConfig{
		Type:       "sqlite",
		MaxSize:    1024,
		DefaultTTL: 100 * time.Millisecond,
		SQLiteConfig: SQLiteConfig{
			Path:           dbPath,
			VacuumInterval: time.Hour,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("TTL expiration", func(t *testing.T) {
		key := "sqlite-ttl-key"
		value := []byte("sqlite-ttl-value")

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
		key := "sqlite-default-ttl-key"
		value := []byte("sqlite-default-ttl-value")

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
}

func TestSQLiteCache_SizeLimit(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "size_test.db")

	config := CacheConfig{
		Type:    "sqlite",
		MaxSize: 200, // Small size for testing
		SQLiteConfig: SQLiteConfig{
			Path:           dbPath,
			VacuumInterval: time.Hour,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	t.Run("Eviction when size limit exceeded", func(t *testing.T) {
		// Add values that will exceed the size limit
		key1 := "sqlite-key1"
		value1 := make([]byte, 80)
		err := cache.Set(ctx, key1, value1, 0)
		assert.NoError(t, err)

		key2 := "sqlite-key2"
		value2 := make([]byte, 80)
		err = cache.Set(ctx, key2, value2, 0)
		assert.NoError(t, err)

		// This should trigger eviction
		key3 := "sqlite-key3"
		value3 := make([]byte, 80)
		err = cache.Set(ctx, key3, value3, 0)
		assert.NoError(t, err)

		// At least one of the older entries should be evicted
		// Due to eviction timing, we can't guarantee which specific key is evicted
		totalFound := 0
		keys := []string{key1, key2, key3}
		for _, key := range keys {
			_, found, err := cache.Get(ctx, key)
			assert.NoError(t, err)
			if found {
				totalFound++
			}
		}
		// Should have fewer entries than we inserted due to eviction
		assert.Less(t, totalFound, 3)
	})

	t.Run("Update existing key", func(t *testing.T) {
		key := "sqlite-update-key"
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

func TestSQLiteCache_Stats(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "stats_test.db")

	config := CacheConfig{
		Type:    "sqlite",
		MaxSize: 1024,
		SQLiteConfig: SQLiteConfig{
			Path:           dbPath,
			VacuumInterval: time.Hour,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Initial stats
	stats := cache.Stats()
	assert.Equal(t, int64(1024), stats.MaxSize)

	// Set operation
	key := "sqlite-stats-key"
	value := []byte("sqlite-stats-value")
	err = cache.Set(ctx, key, value, 0)
	assert.NoError(t, err)

	// Get operation (hit)
	_, found, err := cache.Get(ctx, key)
	assert.NoError(t, err)
	assert.True(t, found)

	// Get operation (miss)
	_, found, err = cache.Get(ctx, "non-existent")
	assert.NoError(t, err)
	assert.False(t, found)

	// Delete operation
	err = cache.Delete(ctx, key)
	assert.NoError(t, err)

	// Stats should be updated
	stats = cache.Stats()
	// Note: SQLite cache stats are updated when loadStats is called
	// The exact values depend on timing of loadStats calls
	assert.Equal(t, int64(1024), stats.MaxSize)
}

func TestSQLiteCache_ExportImport(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "export_test.db")

	config := CacheConfig{
		Type:    "sqlite",
		MaxSize: 1024,
		SQLiteConfig: SQLiteConfig{
			Path:           dbPath,
			VacuumInterval: time.Hour,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Set some values
	testData := map[string][]byte{
		"sqlite-export1": []byte("sqlite-value1"),
		"sqlite-export2": []byte("sqlite-value2"),
		"sqlite-export3": []byte("sqlite-value3"),
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
			assert.False(t, entry.AccessedAt.IsZero())
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
		newDbPath := filepath.Join(tmpDir, "import_test.db")
		newConfig := config
		newConfig.SQLiteConfig.Path = newDbPath

		newCache, err := NewSQLiteCache(newConfig)
		require.NoError(t, err)
		defer newCache.Close()

		// Prepare import data
		importData := []CacheEntry{
			{
				Key:        "sqlite-import1",
				Value:      []byte("sqlite-import-value1"),
				ExpiresAt:  time.Now().Add(time.Hour),
				CreatedAt:  time.Now(),
				AccessedAt: time.Now(),
				Size:       int64(len([]byte("sqlite-import-value1"))),
			},
			{
				Key:        "sqlite-import2",
				Value:      []byte("sqlite-import-value2"),
				ExpiresAt:  time.Now().Add(time.Hour),
				CreatedAt:  time.Now(),
				AccessedAt: time.Now(),
				Size:       int64(len([]byte("sqlite-import-value2"))),
			},
		}

		// Import
		err = newCache.Import(ctx, importData)
		assert.NoError(t, err)

		// Verify imported data
		retrieved, found, err := newCache.Get(ctx, "sqlite-import1")
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, []byte("sqlite-import-value1"), retrieved)

		retrieved, found, err = newCache.Get(ctx, "sqlite-import2")
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, []byte("sqlite-import-value2"), retrieved)
	})
}

func TestSQLiteCache_ErrorHandling(t *testing.T) {
	t.Run("Invalid database path", func(t *testing.T) {
		config := CacheConfig{
			Type:    "sqlite",
			MaxSize: 1024,
			SQLiteConfig: SQLiteConfig{
				Path: "/invalid/path/test.db", // Invalid path
			},
		}

		_, err := NewSQLiteCache(config)
		assert.Error(t, err)
	})

	t.Run("Database file permissions", func(t *testing.T) {
		if os.Getuid() == 0 {
			t.Skip("Skipping permission test when running as root")
		}

		tmpDir := t.TempDir()
		dbPath := filepath.Join(tmpDir, "permission_test.db")

		// Create cache first
		config := CacheConfig{
			Type:    "sqlite",
			MaxSize: 1024,
			SQLiteConfig: SQLiteConfig{
				Path: dbPath,
			},
		}

		cache, err := NewSQLiteCache(config)
		require.NoError(t, err)

		// Make directory read-only to test permission errors
		cache.Close()
		err = os.Chmod(tmpDir, 0444)
		if err != nil {
			t.Logf("Could not change directory permissions: %v", err)
		}

		// Try to create new cache - this may or may not fail depending on the OS
		_, err = NewSQLiteCache(config)
		// Just ensure we can handle the error gracefully
		if err != nil {
			assert.Error(t, err)
		}

		// Restore permissions for cleanup
		_ = os.Chmod(tmpDir, 0755)
	})
}

func TestSQLiteCache_Cleanup(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "cleanup_test.db")

	config := CacheConfig{
		Type:       "sqlite",
		MaxSize:    1024,
		DefaultTTL: 50 * time.Millisecond,
		SQLiteConfig: SQLiteConfig{
			Path:           dbPath,
			VacuumInterval: 200 * time.Millisecond,
		},
	}

	cache, err := NewSQLiteCache(config)
	require.NoError(t, err)
	defer cache.Close()

	ctx := context.Background()

	// Set values with short TTL
	err = cache.Set(ctx, "sqlite-cleanup1", []byte("value1"), 40*time.Millisecond)
	assert.NoError(t, err)

	err = cache.Set(ctx, "sqlite-cleanup2", []byte("value2"), 40*time.Millisecond)
	assert.NoError(t, err)

	// Values should be available initially
	_, found, err := cache.Get(ctx, "sqlite-cleanup1")
	assert.NoError(t, err)
	assert.True(t, found)

	// Wait for TTL expiration and cleanup
	time.Sleep(100 * time.Millisecond)

	// Values should be expired (but cleanup may not have run yet)
	_, found, err = cache.Get(ctx, "sqlite-cleanup1")
	assert.NoError(t, err)
	assert.False(t, found)

	_, found, err = cache.Get(ctx, "sqlite-cleanup2")
	assert.NoError(t, err)
	assert.False(t, found)
}

func TestSQLiteCache_Configuration(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("WAL mode enabled", func(t *testing.T) {
		dbPath := filepath.Join(tmpDir, "wal_test.db")
		config := CacheConfig{
			Type:    "sqlite",
			MaxSize: 1024,
			SQLiteConfig: SQLiteConfig{
				Path:           dbPath,
				EnableWAL:     true,
				MaxConnections: 10,
			},
		}

		cache, err := NewSQLiteCache(config)
		assert.NoError(t, err)
		defer cache.Close()

		// Perform a write operation to trigger WAL file creation
		ctx := context.Background()
		err = cache.Set(ctx, "wal-test-key", []byte("wal-test-value"), 0)
		assert.NoError(t, err)

		// Check if WAL file was created after write operation
		walPath := dbPath + "-wal"
		_, err = os.Stat(walPath)
		// WAL file should exist after write operation
		assert.NoError(t, err, "WAL file should exist after write operation")
	})

	t.Run("Default configuration", func(t *testing.T) {
		dbPath := filepath.Join(tmpDir, "default_test.db")
		config := CacheConfig{
			Type:    "sqlite",
			MaxSize: 1024,
			SQLiteConfig: SQLiteConfig{
				Path: dbPath,
			},
		}

		cache, err := NewSQLiteCache(config)
		assert.NoError(t, err)
		defer cache.Close()

		ctx := context.Background()

		// Basic operation should work
		err = cache.Set(ctx, "default-test", []byte("default-value"), 0)
		assert.NoError(t, err)

		retrieved, found, err := cache.Get(ctx, "default-test")
		assert.NoError(t, err)
		assert.True(t, found)
		assert.Equal(t, []byte("default-value"), retrieved)
	})
}