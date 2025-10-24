package context

import (
	"fmt"
	"time"
)

// Config defines the configuration for agent context management.
// Inspired by Manus's context engineering patterns for production-grade agents.
type Config struct {
	// Session identification
	SessionID string `json:"session_id"`
	AgentID   string `json:"agent_id"`
	BaseDir   string `json:"base_dir"`

	// Feature toggles for Manus-inspired patterns
	EnableCacheOptimization bool `json:"enable_cache_optimization"`
	EnableFileSystemMemory  bool `json:"enable_filesystem_memory"`
	EnableTodoManagement    bool `json:"enable_todo_management"`
	EnableErrorRetention    bool `json:"enable_error_retention"`
	EnableLogitMasking      bool `json:"enable_logit_masking"`
	EnableCompression       bool `json:"enable_compression"`
	EnableDiversification   bool `json:"enable_diversification"`

	// Performance tuning
	CompressionThreshold int64   `json:"compression_threshold"` // Compress content above this size (bytes)
	MaxMemorySize        int64   `json:"max_memory_size"`       // Maximum filesystem memory usage
	CacheHitTarget       float64 `json:"cache_hit_target"`      // Target cache hit rate (0.0-1.0)
	MaxErrorRetention    int     `json:"max_error_retention"`   // Maximum number of errors to retain

	// Cache configuration
	Cache CacheConfig `json:"cache"`

	// Memory configuration
	Memory MemoryConfig `json:"memory"`

	// Todo management configuration
	Todo TodoConfig `json:"todo"`
}

// CacheConfig controls KV-cache optimization patterns.
type CacheConfig struct {
	StablePrefix       string `json:"stable_prefix"`        // Stable prompt prefix for cache hits
	MaxPrefixSize      int    `json:"max_prefix_size"`      // Maximum size of stable prefix
	BreakpointInterval int    `json:"breakpoint_interval"`  // Token interval for cache breakpoints
	EnableMetrics      bool   `json:"enable_metrics"`       // Track cache hit/miss metrics
	TimestampGranularity string `json:"timestamp_granularity"` // "day", "hour", "minute" - NEVER "second"
}

// MemoryConfig controls filesystem-based memory patterns.
type MemoryConfig struct {
	MaxFileSize      int64             `json:"max_file_size"`      // Maximum size per memory file
	RetentionPeriod  time.Duration     `json:"retention_period"`   // How long to keep memory files
	CompressionLevel int               `json:"compression_level"`  // 0=none, 1=fast, 9=best
	FilePatterns     map[string]string `json:"file_patterns"`      // Naming patterns for different content types
}

// TodoConfig controls todo.md attention manipulation patterns.
type TodoConfig struct {
	UpdateInterval    time.Duration `json:"update_interval"`     // How often to rewrite todo.md
	MaxActiveTasks    int           `json:"max_active_tasks"`    // Maximum tasks marked as active
	MaxPendingTasks   int           `json:"max_pending_tasks"`   // Maximum pending tasks to show
	MaxCompletedTasks int           `json:"max_completed_tasks"` // Maximum completed tasks to keep
	EnableEmojis      bool          `json:"enable_emojis"`       // Use emoji markers for attention
}

// DefaultConfig returns a production-ready configuration based on Manus patterns.
func DefaultConfig() Config {
	return Config{
		SessionID: "default_session",
		AgentID:   "default_agent",
		BaseDir:   "./agent_memory",

		// Enable core Manus patterns
		EnableCacheOptimization: true,
		EnableFileSystemMemory:  true,
		EnableTodoManagement:    true,
		EnableErrorRetention:    true,
		EnableLogitMasking:      false, // Opt-in for tool-heavy agents
		EnableCompression:       true,
		EnableDiversification:   true,

		// Performance settings optimized for cost reduction
		CompressionThreshold: 50000,  // 50KB threshold
		MaxMemorySize:        1 << 30, // 1GB default
		CacheHitTarget:       0.90,   // 90% cache hit target
		MaxErrorRetention:    10,     // Keep last 10 errors

		Cache: CacheConfig{
			StablePrefix:         "You are a helpful AI assistant.",
			MaxPrefixSize:        8192, // 8K tokens for stable prefix
			BreakpointInterval:   1000, // Cache breakpoints every 1K tokens
			EnableMetrics:        true,
			TimestampGranularity: "day", // CRITICAL: Day-level only for cache hits
		},

		Memory: MemoryConfig{
			MaxFileSize:      10 * 1024 * 1024, // 10MB per file
			RetentionPeriod:  24 * time.Hour,   // 24 hour retention
			CompressionLevel: 6,                // Balanced compression
			FilePatterns: map[string]string{
				"todo":         "todo.md",
				"context":      "context_%d.json",
				"observations": "observations_%s.txt",
				"errors":       "errors.log",
				"plan":         "plan.md",
				"memory":       "memory_%s.json",
			},
		},

		Todo: TodoConfig{
			UpdateInterval:    5 * time.Second, // Frequent updates for attention
			MaxActiveTasks:    3,               // Focus on 3 active tasks
			MaxPendingTasks:   10,              // Show up to 10 pending
			MaxCompletedTasks: 5,               // Keep 5 completed for context
			EnableEmojis:      true,            // Use emojis for visual attention
		},
	}
}

// DevelopmentConfig returns a configuration suitable for development/testing.
func DevelopmentConfig() Config {
	config := DefaultConfig()

	// Reduce thresholds for easier testing
	config.CompressionThreshold = 1000 // 1KB for testing
	config.MaxMemorySize = 100 * 1024 * 1024 // 100MB for development
	config.Cache.MaxPrefixSize = 1024 // 1K tokens for testing
	config.Memory.RetentionPeriod = 1 * time.Hour // 1 hour retention
	config.Todo.UpdateInterval = 1 * time.Second // Faster updates for testing

	return config
}

// ProductionConfig returns a configuration optimized for production cost efficiency.
func ProductionConfig() Config {
	config := DefaultConfig()

	// Optimize for maximum cost reduction
	config.CacheHitTarget = 0.95 // 95% cache hit target
	config.Cache.StablePrefix = "You are a production AI assistant. Today's date: "
	config.Cache.MaxPrefixSize = 16384 // 16K tokens for large stable prefix
	config.Memory.CompressionLevel = 9 // Maximum compression
	config.Todo.UpdateInterval = 10 * time.Second // Less frequent updates

	return config
}

// Validate checks if the configuration is valid and returns detailed errors.
func (c *Config) Validate() error {
	if c.SessionID == "" {
		return fmt.Errorf("session_id cannot be empty")
	}

	if c.AgentID == "" {
		return fmt.Errorf("agent_id cannot be empty")
	}

	if c.BaseDir == "" {
		return fmt.Errorf("base_dir cannot be empty")
	}

	if c.CompressionThreshold < 0 {
		return fmt.Errorf("compression_threshold must be non-negative")
	}

	if c.MaxMemorySize <= 0 {
		return fmt.Errorf("max_memory_size must be positive")
	}

	if c.CacheHitTarget < 0 || c.CacheHitTarget > 1 {
		return fmt.Errorf("cache_hit_target must be between 0.0 and 1.0")
	}

	// Validate cache config
	if c.Cache.MaxPrefixSize <= 0 {
		return fmt.Errorf("cache.max_prefix_size must be positive")
	}

	if c.Cache.TimestampGranularity == "second" {
		return fmt.Errorf("cache.timestamp_granularity cannot be 'second' - breaks KV-cache optimization")
	}

	validGranularities := map[string]bool{
		"day": true, "hour": true, "minute": true,
	}
	if !validGranularities[c.Cache.TimestampGranularity] {
		return fmt.Errorf("cache.timestamp_granularity must be one of: day, hour, minute")
	}

	// Validate memory config
	if c.Memory.MaxFileSize <= 0 {
		return fmt.Errorf("memory.max_file_size must be positive")
	}

	if c.Memory.CompressionLevel < 0 || c.Memory.CompressionLevel > 9 {
		return fmt.Errorf("memory.compression_level must be between 0 and 9")
	}

	// Validate todo config
	if c.Todo.UpdateInterval <= 0 {
		return fmt.Errorf("todo.update_interval must be positive")
	}

	if c.Todo.MaxActiveTasks <= 0 {
		return fmt.Errorf("todo.max_active_tasks must be positive")
	}

	return nil
}

// GetTimestampFormat returns the appropriate timestamp format for cache optimization.
func (c *Config) GetTimestampFormat() string {
	switch c.Cache.TimestampGranularity {
	case "day":
		return "2006-01-02"
	case "hour":
		return "2006-01-02 15"
	case "minute":
		return "2006-01-02 15:04"
	default:
		return "2006-01-02" // Safe fallback
	}
}
