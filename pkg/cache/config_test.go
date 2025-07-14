package cache

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/stretchr/testify/assert"
)

func TestLoadCacheConfig(t *testing.T) {
	// Clear environment variables before each test
	envVars := []string{
		"DSPY_CACHE_ENABLED",
		"DSPY_CACHE_TYPE",
		"DSPY_CACHE_TTL",
		"DSPY_CACHE_MAX_SIZE",
		"DSPY_CACHE_PATH",
		"DSPY_CACHE_WAL",
		"DSPY_CACHE_VACUUM_INTERVAL",
		"DSPY_CACHE_MAX_CONNECTIONS",
		"DSPY_CACHE_CLEANUP_INTERVAL",
		"DSPY_CACHE_SHARD_COUNT",
	}

	for _, envVar := range envVars {
		os.Unsetenv(envVar)
	}

	t.Run("Default config with nil file config", func(t *testing.T) {
		cfg := LoadCacheConfig(nil)
		
		assert.Equal(t, "memory", cfg.Type)
		assert.Equal(t, time.Hour, cfg.DefaultTTL)
		assert.Equal(t, int64(100*1024*1024), cfg.MaxSize)
		assert.Equal(t, time.Minute, cfg.MemoryConfig.CleanupInterval)
		assert.Equal(t, 16, cfg.MemoryConfig.ShardCount)
	})

	t.Run("File config overrides defaults", func(t *testing.T) {
		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "sqlite",
			TTL:     2 * time.Hour,
			MaxSize: 200 * 1024 * 1024,
			SQLiteConfig: config.SQLiteCacheConfig{
				Path:           "/custom/path.db",
				EnableWAL:      true,
				VacuumInterval: 12 * time.Hour,
				MaxConnections: 20,
			},
		}

		cfg := LoadCacheConfig(fileConfig)
		
		assert.Equal(t, "sqlite", cfg.Type)
		assert.Equal(t, 2*time.Hour, cfg.DefaultTTL)
		assert.Equal(t, int64(200*1024*1024), cfg.MaxSize)
		assert.Equal(t, "/custom/path.db", cfg.SQLiteConfig.Path)
		assert.True(t, cfg.SQLiteConfig.EnableWAL)
		assert.Equal(t, 12*time.Hour, cfg.SQLiteConfig.VacuumInterval)
		assert.Equal(t, 20, cfg.SQLiteConfig.MaxConnections)
	})

	t.Run("Disabled cache", func(t *testing.T) {
		fileConfig := &config.CachingConfig{
			Enabled: false,
		}

		cfg := LoadCacheConfig(fileConfig)
		assert.Equal(t, "disabled", cfg.Type)
	})

	t.Run("SQLite config with defaults", func(t *testing.T) {
		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "sqlite",
		}

		cfg := LoadCacheConfig(fileConfig)
		
		assert.Equal(t, "sqlite", cfg.Type)
		homeDir, _ := os.UserHomeDir()
		expectedPath := filepath.Join(homeDir, ".dspy", "cache.db")
		assert.Equal(t, expectedPath, cfg.SQLiteConfig.Path)
		assert.Equal(t, 10, cfg.SQLiteConfig.MaxConnections)
		assert.Equal(t, 24*time.Hour, cfg.SQLiteConfig.VacuumInterval)
	})

	t.Run("Memory config with custom values", func(t *testing.T) {
		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "memory",
			MemoryConfig: config.MemoryCacheConfig{
				CleanupInterval: 30 * time.Second,
				ShardCount:      32,
			},
		}

		cfg := LoadCacheConfig(fileConfig)
		
		assert.Equal(t, "memory", cfg.Type)
		assert.Equal(t, 30*time.Second, cfg.MemoryConfig.CleanupInterval)
		assert.Equal(t, 32, cfg.MemoryConfig.ShardCount)
	})
}

func TestApplyEnvConfig(t *testing.T) {
	// Clear environment variables before each test
	envVars := []string{
		"DSPY_CACHE_ENABLED",
		"DSPY_CACHE_TYPE",
		"DSPY_CACHE_TTL",
		"DSPY_CACHE_MAX_SIZE",
		"DSPY_CACHE_PATH",
		"DSPY_CACHE_WAL",
		"DSPY_CACHE_VACUUM_INTERVAL",
		"DSPY_CACHE_MAX_CONNECTIONS",
		"DSPY_CACHE_CLEANUP_INTERVAL",
		"DSPY_CACHE_SHARD_COUNT",
	}

	for _, envVar := range envVars {
		os.Unsetenv(envVar)
	}

	t.Run("Cache disabled by environment", func(t *testing.T) {
		os.Setenv("DSPY_CACHE_ENABLED", "false")
		defer os.Unsetenv("DSPY_CACHE_ENABLED")

		cfg := LoadCacheConfig(nil)
		assert.Equal(t, "disabled", cfg.Type)
	})

	t.Run("Cache disabled by environment with 0", func(t *testing.T) {
		os.Setenv("DSPY_CACHE_ENABLED", "0")
		defer os.Unsetenv("DSPY_CACHE_ENABLED")

		cfg := LoadCacheConfig(nil)
		assert.Equal(t, "disabled", cfg.Type)
	})

	t.Run("Environment overrides", func(t *testing.T) {
		os.Setenv("DSPY_CACHE_TYPE", "sqlite")
		os.Setenv("DSPY_CACHE_TTL", "2h")
		os.Setenv("DSPY_CACHE_MAX_SIZE", "500MB")
		os.Setenv("DSPY_CACHE_PATH", "/tmp/cache.db")
		os.Setenv("DSPY_CACHE_WAL", "true")
		os.Setenv("DSPY_CACHE_VACUUM_INTERVAL", "12h")
		os.Setenv("DSPY_CACHE_MAX_CONNECTIONS", "25")
		
		defer func() {
			for _, envVar := range envVars {
				os.Unsetenv(envVar)
			}
		}()

		cfg := LoadCacheConfig(nil)
		
		assert.Equal(t, "sqlite", cfg.Type)
		assert.Equal(t, 2*time.Hour, cfg.DefaultTTL)
		assert.Equal(t, int64(500*1024*1024), cfg.MaxSize)
		assert.Equal(t, "/tmp/cache.db", cfg.SQLiteConfig.Path)
		assert.True(t, cfg.SQLiteConfig.EnableWAL)
		assert.Equal(t, 12*time.Hour, cfg.SQLiteConfig.VacuumInterval)
		assert.Equal(t, 25, cfg.SQLiteConfig.MaxConnections)
	})

	t.Run("Memory cache environment overrides", func(t *testing.T) {
		os.Setenv("DSPY_CACHE_TYPE", "memory")
		os.Setenv("DSPY_CACHE_CLEANUP_INTERVAL", "30s")
		os.Setenv("DSPY_CACHE_SHARD_COUNT", "32")
		
		defer func() {
			for _, envVar := range envVars {
				os.Unsetenv(envVar)
			}
		}()

		cfg := LoadCacheConfig(nil)
		
		assert.Equal(t, "memory", cfg.Type)
		assert.Equal(t, 30*time.Second, cfg.MemoryConfig.CleanupInterval)
		assert.Equal(t, 32, cfg.MemoryConfig.ShardCount)
	})

	t.Run("Invalid environment values are ignored", func(t *testing.T) {
		os.Setenv("DSPY_CACHE_TTL", "invalid")
		os.Setenv("DSPY_CACHE_MAX_SIZE", "invalid")
		os.Setenv("DSPY_CACHE_MAX_CONNECTIONS", "invalid")
		os.Setenv("DSPY_CACHE_SHARD_COUNT", "invalid")
		
		defer func() {
			for _, envVar := range envVars {
				os.Unsetenv(envVar)
			}
		}()

		cfg := LoadCacheConfig(nil)
		
		// Should use default values
		assert.Equal(t, time.Hour, cfg.DefaultTTL)
		assert.Equal(t, int64(100*1024*1024), cfg.MaxSize)
		assert.Equal(t, 0, cfg.SQLiteConfig.MaxConnections) // Environment parsing failed, so keeps 0
		assert.Equal(t, 16, cfg.MemoryConfig.ShardCount)
	})
}

func TestParseSize(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int64
	}{
		{"Bytes", "1024", 1024},
		{"Bytes with B suffix", "1024B", 1024},
		{"Kilobytes", "1KB", 1024},
		{"Megabytes", "1MB", 1024 * 1024},
		{"Gigabytes", "1GB", 1024 * 1024 * 1024},
		{"Lowercase", "1mb", 1024 * 1024},
		{"With spaces", " 1MB ", 1024 * 1024},
		{"Invalid format", "invalid", 0},
		{"Empty string", "", 0},
		{"Zero", "0", 0},
		{"Large number", "500MB", 500 * 1024 * 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseSize(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExpandPath(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Regular path",
			input:    "/tmp/cache.db",
			expected: "/tmp/cache.db",
		},
		{
			name:     "Relative path",
			input:    "./cache.db",
			expected: "./cache.db",
		},
		{
			name:     "Home directory expansion",
			input:    "~/cache.db",
			expected: func() string {
				homeDir, _ := os.UserHomeDir()
				return filepath.Join(homeDir, "cache.db")
			}(),
		},
		{
			name:     "Home directory with nested path",
			input:    "~/.dspy/cache.db",
			expected: func() string {
				homeDir, _ := os.UserHomeDir()
				return filepath.Join(homeDir, ".dspy", "cache.db")
			}(),
		},
		{
			name:     "Tilde without slash",
			input:    "~",
			expected: "~",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := expandPath(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsEnabled(t *testing.T) {
	tests := []struct {
		name     string
		config   CacheConfig
		expected bool
	}{
		{
			name:     "Enabled memory cache",
			config:   CacheConfig{Type: "memory"},
			expected: true,
		},
		{
			name:     "Enabled sqlite cache",
			config:   CacheConfig{Type: "sqlite"},
			expected: true,
		},
		{
			name:     "Disabled cache",
			config:   CacheConfig{Type: "disabled"},
			expected: false,
		},
		{
			name:     "Empty type",
			config:   CacheConfig{Type: ""},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsEnabled(tt.config)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetDefaultCacheConfig(t *testing.T) {
	// Clear environment variables
	envVars := []string{
		"DSPY_CACHE_ENABLED",
		"DSPY_CACHE_TYPE",
		"DSPY_CACHE_TTL",
		"DSPY_CACHE_MAX_SIZE",
	}

	for _, envVar := range envVars {
		os.Unsetenv(envVar)
	}

	cfg := GetDefaultCacheConfig()
	
	assert.Equal(t, "memory", cfg.Type)
	assert.Equal(t, time.Hour, cfg.DefaultTTL)
	assert.Equal(t, int64(100*1024*1024), cfg.MaxSize)
	assert.Equal(t, time.Minute, cfg.MemoryConfig.CleanupInterval)
	assert.Equal(t, 16, cfg.MemoryConfig.ShardCount)
}

func TestApplyFileConfig(t *testing.T) {
	t.Run("Apply file config to default config", func(t *testing.T) {
		cfg := CacheConfig{
			Type:       "memory",
			DefaultTTL: time.Hour,
			MaxSize:    100 * 1024 * 1024,
		}

		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "sqlite",
			TTL:     2 * time.Hour,
			MaxSize: 200 * 1024 * 1024,
		}

		applyFileConfig(&cfg, fileConfig)

		assert.Equal(t, "sqlite", cfg.Type)
		assert.Equal(t, 2*time.Hour, cfg.DefaultTTL)
		assert.Equal(t, int64(200*1024*1024), cfg.MaxSize)
	})

	t.Run("Disabled file config", func(t *testing.T) {
		cfg := CacheConfig{
			Type: "memory",
		}

		fileConfig := &config.CachingConfig{
			Enabled: false,
		}

		applyFileConfig(&cfg, fileConfig)

		assert.Equal(t, "disabled", cfg.Type)
	})

	t.Run("File config with zero values uses defaults", func(t *testing.T) {
		cfg := CacheConfig{
			Type:       "memory",
			DefaultTTL: time.Hour,
			MaxSize:    100 * 1024 * 1024,
		}

		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "memory",
			TTL:     0, // Should not override
			MaxSize: 0, // Should not override
		}

		applyFileConfig(&cfg, fileConfig)

		assert.Equal(t, "memory", cfg.Type)
		assert.Equal(t, time.Hour, cfg.DefaultTTL)  // Should keep default
		assert.Equal(t, int64(100*1024*1024), cfg.MaxSize)  // Should keep default
	})
}