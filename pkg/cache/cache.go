package cache

import (
	"context"
	"time"
)

// Cache defines the interface for caching LLM responses.
type Cache interface {
	// Get retrieves a cached value by key.
	Get(ctx context.Context, key string) ([]byte, bool, error)

	// Set stores a value with the given key and TTL.
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Delete removes a cached value by key.
	Delete(ctx context.Context, key string) error

	// Clear removes all cached values.
	Clear(ctx context.Context) error

	// Stats returns cache statistics.
	Stats() CacheStats

	// Close releases any resources held by the cache.
	Close() error
}

// CacheStats contains cache performance statistics.
type CacheStats struct {
	Hits       int64     `json:"hits"`
	Misses     int64     `json:"misses"`
	Sets       int64     `json:"sets"`
	Deletes    int64     `json:"deletes"`
	Size       int64     `json:"size"`
	MaxSize    int64     `json:"max_size"`
	LastAccess time.Time `json:"last_access"`
}

// CacheEntry represents a cached item.
type CacheEntry struct {
	Key        string    `json:"key"`
	Value      []byte    `json:"value"`
	ExpiresAt  time.Time `json:"expires_at"`
	CreatedAt  time.Time `json:"created_at"`
	AccessedAt time.Time `json:"accessed_at"`
	Size       int64     `json:"size"`
}

// IsExpired checks if the cache entry has expired.
func (e *CacheEntry) IsExpired() bool {
	return !e.ExpiresAt.IsZero() && time.Now().After(e.ExpiresAt)
}

// CacheConfig holds cache configuration.
type CacheConfig struct {
	// Type of cache: "memory" or "sqlite"
	Type string `json:"type" yaml:"type"`

	// Maximum cache size in bytes (0 = unlimited)
	MaxSize int64 `json:"max_size" yaml:"max_size"`

	// Default TTL for cache entries (0 = no expiration)
	DefaultTTL time.Duration `json:"default_ttl" yaml:"default_ttl"`

	// SQLite specific configuration
	SQLiteConfig SQLiteConfig `json:"sqlite_config,omitempty" yaml:"sqlite_config,omitempty"`

	// Memory cache specific configuration
	MemoryConfig MemoryConfig `json:"memory_config,omitempty" yaml:"memory_config,omitempty"`
}

// SQLiteConfig holds SQLite-specific configuration.
type SQLiteConfig struct {
	// Path to the SQLite database file
	Path string `json:"path" yaml:"path"`

	// Enable WAL mode for better concurrent performance
	EnableWAL bool `json:"enable_wal" yaml:"enable_wal"`

	// Vacuum interval for database maintenance
	VacuumInterval time.Duration `json:"vacuum_interval" yaml:"vacuum_interval"`

	// Maximum number of connections
	MaxConnections int `json:"max_connections" yaml:"max_connections"`
}

// MemoryConfig holds memory cache specific configuration.
type MemoryConfig struct {
	// Eviction policy: "lru", "lfu", "fifo"
	EvictionPolicy string `json:"eviction_policy" yaml:"eviction_policy"`

	// Cleanup interval for expired entries
	CleanupInterval time.Duration `json:"cleanup_interval" yaml:"cleanup_interval"`

	// Number of shards for concurrent access
	ShardCount int `json:"shard_count" yaml:"shard_count"`
}

// NewCache creates a new cache instance based on the configuration.
func NewCache(config CacheConfig) (Cache, error) {
	switch config.Type {
	case "memory":
		return NewMemoryCache(config)
	case "sqlite":
		return NewSQLiteCache(config)
	default:
		// Default to memory cache
		return NewMemoryCache(config)
	}
}