package cache

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
)

// 3. Smart defaults (lowest priority).
func LoadCacheConfig(fileConfig *config.CachingConfig) CacheConfig {
	// Start with smart defaults
	cacheConfig := CacheConfig{
		Type:       "memory",
		DefaultTTL: time.Hour,
		MaxSize:    100 * 1024 * 1024, // 100MB
		MemoryConfig: MemoryConfig{
			CleanupInterval: time.Minute,
			ShardCount:      16,
		},
	}

	// Apply config file settings if provided
	if fileConfig != nil {
		applyFileConfig(&cacheConfig, fileConfig)
	}

	// Apply environment variables (highest priority)
	applyEnvConfig(&cacheConfig)

	return cacheConfig
}

// applyFileConfig applies settings from configuration file.
func applyFileConfig(cacheConfig *CacheConfig, fileConfig *config.CachingConfig) {
	if !fileConfig.Enabled {
		cacheConfig.Type = "disabled"
		return
	}

	if fileConfig.Type != "" {
		cacheConfig.Type = fileConfig.Type
	}

	if fileConfig.TTL > 0 {
		cacheConfig.DefaultTTL = fileConfig.TTL
	}

	if fileConfig.MaxSize > 0 {
		cacheConfig.MaxSize = fileConfig.MaxSize
	}

	// Apply SQLite config
	if fileConfig.Type == "sqlite" {
		cacheConfig.SQLiteConfig = SQLiteConfig{
			Path:           fileConfig.SQLiteConfig.Path,
			EnableWAL:      fileConfig.SQLiteConfig.EnableWAL,
			VacuumInterval: fileConfig.SQLiteConfig.VacuumInterval,
			MaxConnections: fileConfig.SQLiteConfig.MaxConnections,
		}

		// Set default path if not specified
		if cacheConfig.SQLiteConfig.Path == "" {
			homeDir, _ := os.UserHomeDir()
			cacheConfig.SQLiteConfig.Path = filepath.Join(homeDir, ".dspy", "cache.db")
		}

		// Set default values
		if cacheConfig.SQLiteConfig.MaxConnections == 0 {
			cacheConfig.SQLiteConfig.MaxConnections = 10
		}
		if cacheConfig.SQLiteConfig.VacuumInterval == 0 {
			cacheConfig.SQLiteConfig.VacuumInterval = 24 * time.Hour
		}
	}

	// Apply memory config
	if fileConfig.Type == "memory" {
		if fileConfig.MemoryConfig.CleanupInterval > 0 {
			cacheConfig.MemoryConfig.CleanupInterval = fileConfig.MemoryConfig.CleanupInterval
		}
		if fileConfig.MemoryConfig.ShardCount > 0 {
			cacheConfig.MemoryConfig.ShardCount = fileConfig.MemoryConfig.ShardCount
		}
	}
}

// applyEnvConfig applies environment variable settings (highest priority).
func applyEnvConfig(cacheConfig *CacheConfig) {
	// Check if caching is enabled
	if enabled := os.Getenv("DSPY_CACHE_ENABLED"); enabled != "" {
		if enabled == "false" || enabled == "0" {
			cacheConfig.Type = "disabled"
			return
		}
	}

	// Cache type
	if cacheType := os.Getenv("DSPY_CACHE_TYPE"); cacheType != "" {
		cacheConfig.Type = cacheType
	}

	// Cache TTL
	if ttlStr := os.Getenv("DSPY_CACHE_TTL"); ttlStr != "" {
		if ttl, err := time.ParseDuration(ttlStr); err == nil {
			cacheConfig.DefaultTTL = ttl
		}
	}

	// Cache max size (support human-readable formats like "100MB")
	if maxSizeStr := os.Getenv("DSPY_CACHE_MAX_SIZE"); maxSizeStr != "" {
		if size := parseSize(maxSizeStr); size > 0 {
			cacheConfig.MaxSize = size
		}
	}

	// SQLite specific environment variables
	if cacheConfig.Type == "sqlite" {
		if path := os.Getenv("DSPY_CACHE_PATH"); path != "" {
			cacheConfig.SQLiteConfig.Path = expandPath(path)
		}

		if walStr := os.Getenv("DSPY_CACHE_WAL"); walStr != "" {
			cacheConfig.SQLiteConfig.EnableWAL = walStr == "true" || walStr == "1"
		}

		if vacuumStr := os.Getenv("DSPY_CACHE_VACUUM_INTERVAL"); vacuumStr != "" {
			if interval, err := time.ParseDuration(vacuumStr); err == nil {
				cacheConfig.SQLiteConfig.VacuumInterval = interval
			}
		}

		if maxConnStr := os.Getenv("DSPY_CACHE_MAX_CONNECTIONS"); maxConnStr != "" {
			if maxConn, err := strconv.Atoi(maxConnStr); err == nil {
				cacheConfig.SQLiteConfig.MaxConnections = maxConn
			}
		}
	}

	// Memory cache specific environment variables
	if cacheConfig.Type == "memory" {
		if cleanupStr := os.Getenv("DSPY_CACHE_CLEANUP_INTERVAL"); cleanupStr != "" {
			if interval, err := time.ParseDuration(cleanupStr); err == nil {
				cacheConfig.MemoryConfig.CleanupInterval = interval
			}
		}

		if shardStr := os.Getenv("DSPY_CACHE_SHARD_COUNT"); shardStr != "" {
			if shardCount, err := strconv.Atoi(shardStr); err == nil {
				cacheConfig.MemoryConfig.ShardCount = shardCount
			}
		}
	}
}

// parseSize parses human-readable size strings like "100MB", "1GB", etc.
func parseSize(s string) int64 {
	s = strings.ToUpper(strings.TrimSpace(s))

	var multiplier int64 = 1
	if strings.HasSuffix(s, "KB") {
		multiplier = 1024
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "MB") {
		multiplier = 1024 * 1024
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "GB") {
		multiplier = 1024 * 1024 * 1024
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "B") {
		s = s[:len(s)-1]
	}

	if num, err := strconv.ParseInt(s, 10, 64); err == nil {
		return num * multiplier
	}

	return 0
}

// expandPath expands ~ to user home directory.
func expandPath(path string) string {
	if strings.HasPrefix(path, "~/") {
		if homeDir, err := os.UserHomeDir(); err == nil {
			return filepath.Join(homeDir, path[2:])
		}
	}
	return path
}

// IsEnabled checks if caching is enabled based on configuration.
func IsEnabled(config CacheConfig) bool {
	return config.Type != "disabled" && config.Type != ""
}

// GetDefaultCacheConfig returns the default cache configuration with environment overrides.
func GetDefaultCacheConfig() CacheConfig {
	return LoadCacheConfig(nil)
}

// Environment variable documentation:
//
// DSPY_CACHE_ENABLED=true|false         - Enable/disable caching (default: true)
// DSPY_CACHE_TYPE=memory|sqlite          - Cache backend type (default: memory)
// DSPY_CACHE_TTL=1h|30m|24h             - Cache entry TTL (default: 1h)
// DSPY_CACHE_MAX_SIZE=100MB|1GB         - Maximum cache size (default: 100MB)
//
// SQLite specific:
// DSPY_CACHE_PATH=~/.dspy/cache.db      - SQLite database path
// DSPY_CACHE_WAL=true|false             - Enable WAL mode (default: true)
// DSPY_CACHE_VACUUM_INTERVAL=24h        - Vacuum interval (default: 24h)
// DSPY_CACHE_MAX_CONNECTIONS=10         - Max DB connections (default: 10)
//
// Memory cache specific:
// DSPY_CACHE_CLEANUP_INTERVAL=1m        - Cleanup interval (default: 1m)
// DSPY_CACHE_SHARD_COUNT=16             - Number of shards (default: 16)
