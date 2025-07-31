package cache

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCacheEntry_IsExpired(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		entry     CacheEntry
		expected  bool
	}{
		{
			name: "Entry not expired",
			entry: CacheEntry{
				Key:       "test",
				Value:     []byte("data"),
				ExpiresAt: now.Add(time.Hour),
				CreatedAt: now,
			},
			expected: false,
		},
		{
			name: "Entry expired",
			entry: CacheEntry{
				Key:       "test",
				Value:     []byte("data"),
				ExpiresAt: now.Add(-time.Hour),
				CreatedAt: now.Add(-2 * time.Hour),
			},
			expected: true,
		},
		{
			name: "Entry with zero expiration time (never expires)",
			entry: CacheEntry{
				Key:       "test",
				Value:     []byte("data"),
				ExpiresAt: time.Time{},
				CreatedAt: now,
			},
			expected: false,
		},
		{
			name: "Entry exactly at expiration time",
			entry: CacheEntry{
				Key:       "test",
				Value:     []byte("data"),
				ExpiresAt: now.Add(-time.Nanosecond),
				CreatedAt: now.Add(-time.Hour),
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.entry.IsExpired()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestNewCache(t *testing.T) {
	tests := []struct {
		name           string
		config         CacheConfig
		expectedType   string
		expectError    bool
	}{
		{
			name: "Memory cache",
			config: CacheConfig{
				Type:       "memory",
				MaxSize:    1024 * 1024,
				DefaultTTL: time.Hour,
				MemoryConfig: MemoryConfig{
					CleanupInterval: time.Minute,
				},
			},
			expectedType: "*cache.MemoryCache",
			expectError:  false,
		},
		{
			name: "SQLite cache",
			config: CacheConfig{
				Type:       "sqlite",
				MaxSize:    1024 * 1024,
				DefaultTTL: time.Hour,
				SQLiteConfig: SQLiteConfig{
					Path:           ":memory:",
					EnableWAL:      false,
					MaxConnections: 5,
				},
			},
			expectedType: "*cache.SQLiteCache",
			expectError:  false,
		},
		{
			name: "Default to memory cache for unknown type",
			config: CacheConfig{
				Type:       "unknown",
				MaxSize:    1024 * 1024,
				DefaultTTL: time.Hour,
				MemoryConfig: MemoryConfig{
					CleanupInterval: time.Minute,
				},
			},
			expectedType: "*cache.MemoryCache",
			expectError:  false,
		},
		{
			name: "Empty type defaults to memory",
			config: CacheConfig{
				Type:       "",
				MaxSize:    1024 * 1024,
				DefaultTTL: time.Hour,
				MemoryConfig: MemoryConfig{
					CleanupInterval: time.Minute,
				},
			},
			expectedType: "*cache.MemoryCache",
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache, err := NewCache(tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, cache)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cache)
				assert.Equal(t, tt.expectedType, fmt.Sprintf("%T", cache))
			}

			if cache != nil {
				cache.Close()
			}
		})
	}
}

func TestCacheStats(t *testing.T) {
	stats := CacheStats{
		Hits:       100,
		Misses:     50,
		Sets:       75,
		Deletes:    25,
		Size:       1024,
		MaxSize:    2048,
		LastAccess: time.Now(),
	}

	assert.Equal(t, int64(100), stats.Hits)
	assert.Equal(t, int64(50), stats.Misses)
	assert.Equal(t, int64(75), stats.Sets)
	assert.Equal(t, int64(25), stats.Deletes)
	assert.Equal(t, int64(1024), stats.Size)
	assert.Equal(t, int64(2048), stats.MaxSize)
	assert.False(t, stats.LastAccess.IsZero())
}

func TestCacheEntry(t *testing.T) {
	now := time.Now()
	entry := CacheEntry{
		Key:        "test-key",
		Value:      []byte("test-value"),
		ExpiresAt:  now.Add(time.Hour),
		CreatedAt:  now,
		AccessedAt: now,
		Size:       10,
	}

	assert.Equal(t, "test-key", entry.Key)
	assert.Equal(t, []byte("test-value"), entry.Value)
	assert.Equal(t, now.Add(time.Hour), entry.ExpiresAt)
	assert.Equal(t, now, entry.CreatedAt)
	assert.Equal(t, now, entry.AccessedAt)
	assert.Equal(t, int64(10), entry.Size)
}

func TestCacheConfig(t *testing.T) {
	config := CacheConfig{
		Type:       "memory",
		MaxSize:    1024 * 1024,
		DefaultTTL: time.Hour,
		SQLiteConfig: SQLiteConfig{
			Path:           "/tmp/test.db",
			EnableWAL:      true,
			VacuumInterval: time.Hour * 24,
			MaxConnections: 10,
		},
		MemoryConfig: MemoryConfig{
			EvictionPolicy:  "lru",
			CleanupInterval: time.Minute,
			ShardCount:      16,
		},
	}

	assert.Equal(t, "memory", config.Type)
	assert.Equal(t, int64(1024*1024), config.MaxSize)
	assert.Equal(t, time.Hour, config.DefaultTTL)
	assert.Equal(t, "/tmp/test.db", config.SQLiteConfig.Path)
	assert.True(t, config.SQLiteConfig.EnableWAL)
	assert.Equal(t, time.Hour*24, config.SQLiteConfig.VacuumInterval)
	assert.Equal(t, 10, config.SQLiteConfig.MaxConnections)
	assert.Equal(t, "lru", config.MemoryConfig.EvictionPolicy)
	assert.Equal(t, time.Minute, config.MemoryConfig.CleanupInterval)
	assert.Equal(t, 16, config.MemoryConfig.ShardCount)
}
