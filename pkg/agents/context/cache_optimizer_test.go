package context

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNewCacheOptimizer(t *testing.T) {
	config := CacheConfig{
		StablePrefix:         "Test prefix",
		MaxPrefixSize:        1024,
		BreakpointInterval:   500,
		EnableMetrics:        true,
		TimestampGranularity: "day",
	}

	optimizer := NewCacheOptimizer(config)

	assert.NotNil(t, optimizer)
	assert.Equal(t, config.StablePrefix, optimizer.stablePrefix)
	assert.Equal(t, len(config.StablePrefix), optimizer.prefixSize)
	assert.Equal(t, config.EnableMetrics, optimizer.enableMetrics)
	assert.NotEmpty(t, optimizer.prefixHash)
}

func TestOptimizePrompt(t *testing.T) {
	config := CacheConfig{
		StablePrefix:         "You are a helpful AI assistant.",
		TimestampGranularity: "day",
		EnableMetrics:        true,
	}

	optimizer := NewCacheOptimizer(config)

	tests := []struct {
		name      string
		prompt    string
		timestamp time.Time
		validate  func(t *testing.T, result string)
	}{
		{
			name:      "basic optimization",
			prompt:    "Please help me with a task.",
			timestamp: time.Date(2024, 1, 15, 14, 30, 0, 0, time.UTC),
			validate: func(t *testing.T, result string) {
				assert.Contains(t, result, "You are a helpful AI assistant.")
				assert.Contains(t, result, "2024-01-15")
				assert.Contains(t, result, "Please help me with a task.")
				// Should NOT contain time components for day granularity
				assert.NotContains(t, result, "14:30")
			},
		},
		{
			name:      "hour granularity",
			prompt:    "Analyze this data.",
			timestamp: time.Date(2024, 1, 15, 14, 30, 45, 0, time.UTC),
			validate: func(t *testing.T, result string) {
				assert.Contains(t, result, "2024-01-15 14")
				// Should NOT contain minutes or seconds
				assert.NotContains(t, result, ":30")
				assert.NotContains(t, result, ":45")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "hour granularity" {
				// Temporarily change granularity for this test
				originalGranularity := optimizer.config.TimestampGranularity
				optimizer.config.TimestampGranularity = "hour"
				defer func() { optimizer.config.TimestampGranularity = originalGranularity }()
			}

			result := optimizer.OptimizePrompt(tt.prompt, tt.timestamp)
			assert.NotEmpty(t, result)
			tt.validate(t, result)
		})
	}
}

func TestOptimizeForAppendOnly(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "Test prefix",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	basePrompt := "Here is the base content."
	newContent := "This is additional content."

	result := optimizer.OptimizeForAppendOnly(basePrompt, newContent)

	assert.Contains(t, result, basePrompt)
	assert.Contains(t, result, newContent)
	assert.True(t, len(result) > len(basePrompt))
	assert.True(t, len(result) > len(newContent))

	// Should append, not modify existing content
	baseIndex := strings.Index(result, basePrompt)
	newIndex := strings.Index(result, newContent)
	assert.True(t, newIndex > baseIndex, "New content should come after base content")
}

func TestEstimateTokens(t *testing.T) {
	optimizer := NewCacheOptimizer(CacheConfig{})

	tests := []struct {
		content  string
		expected int
	}{
		{"", 0},
		{"word", 1},
		{"four word test string", 4},
		{"this is a longer string with more words", 8},
	}

	for _, tt := range tests {
		result := optimizer.EstimateTokens(tt.content)
		assert.Equal(t, tt.expected, result)
	}
}

func TestRecordCacheHit(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "Test prefix for caching",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Record cache hits
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheHit(150)

	metrics := optimizer.GetMetrics()

	assert.Equal(t, int64(2), metrics.Hits)
	assert.Equal(t, int64(0), metrics.Misses)
	assert.Equal(t, float64(1.0), metrics.HitRate)
	assert.Greater(t, metrics.TokensSaved, int64(0))
	assert.Greater(t, metrics.CostSavings, 0.0)
}

func TestRecordCacheMiss(t *testing.T) {
	config := CacheConfig{
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Record cache misses
	optimizer.RecordCacheMiss()
	optimizer.RecordCacheMiss()
	optimizer.RecordCacheMiss()

	metrics := optimizer.GetMetrics()

	assert.Equal(t, int64(0), metrics.Hits)
	assert.Equal(t, int64(3), metrics.Misses)
	assert.Equal(t, float64(0.0), metrics.HitRate)
	assert.Equal(t, int64(0), metrics.TokensSaved)
}

func TestMixedHitsAndMisses(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "Test prefix",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Record mixed hits and misses
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheMiss()
	optimizer.RecordCacheHit(200)
	optimizer.RecordCacheMiss()
	optimizer.RecordCacheHit(150)

	metrics := optimizer.GetMetrics()

	assert.Equal(t, int64(3), metrics.Hits)
	assert.Equal(t, int64(2), metrics.Misses)
	assert.Equal(t, float64(0.6), metrics.HitRate) // 3/5 = 0.6
	assert.Greater(t, metrics.TokensSaved, int64(0))
}

func TestGetHitRate(t *testing.T) {
	optimizer := NewCacheOptimizer(CacheConfig{EnableMetrics: true})

	// Initially no data
	assert.Equal(t, float64(0.0), optimizer.GetHitRate())

	// After some hits and misses
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheMiss()

	assert.Equal(t, float64(2.0/3.0), optimizer.GetHitRate())
}

func TestGetTokensSaved(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "This is a test prefix",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Initially no tokens saved
	assert.Equal(t, int64(0), optimizer.GetTokensSaved())

	// After cache hits
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheHit(200)

	savedTokens := optimizer.GetTokensSaved()
	assert.Greater(t, savedTokens, int64(0))
}

func TestGetCostSavings(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "Test prefix for cost calculation",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Initially no cost savings
	assert.Equal(t, float64(0.0), optimizer.GetCostSavings())

	// After cache hits
	optimizer.RecordCacheHit(1000) // Larger prompt size for measurable savings

	costSavings := optimizer.GetCostSavings()
	assert.Greater(t, costSavings, 0.0)
}

func TestAddCacheBreakpoint(t *testing.T) {
	optimizer := NewCacheOptimizer(CacheConfig{})

	// Add breakpoints
	optimizer.AddCacheBreakpoint(1000)
	optimizer.AddCacheBreakpoint(2000)
	optimizer.AddCacheBreakpoint(3000)

	breakpoints := optimizer.GetCacheBreakpoints()

	assert.Len(t, breakpoints, 3)
	assert.Contains(t, breakpoints, 1000)
	assert.Contains(t, breakpoints, 2000)
	assert.Contains(t, breakpoints, 3000)
}

func TestAnalyzePromptForCacheability(t *testing.T) {
	config := CacheConfig{
		StablePrefix: "You are a helpful assistant.",
	}

	optimizer := NewCacheOptimizer(config)

	tests := []struct {
		name     string
		prompt   string
		validate func(t *testing.T, analysis CacheAnalysis)
	}{
		{
			name:   "well optimized prompt",
			prompt: "You are a helpful assistant.\n\nPlease help me with this task.",
			validate: func(t *testing.T, analysis CacheAnalysis) {
				assert.True(t, analysis.HasStablePrefix)
				assert.False(t, analysis.HasTimestampIssues)
				assert.Greater(t, analysis.CacheabilityScore, 0.7)
				assert.Contains(t, analysis.Recommendations, "Excellent cache optimization - no changes needed")
			},
		},
		{
			name:   "prompt with timestamp issues",
			prompt: "Current time: 2024-01-15T14:30:45Z\nPlease help me.",
			validate: func(t *testing.T, analysis CacheAnalysis) {
				assert.False(t, analysis.HasStablePrefix)
				assert.True(t, analysis.HasTimestampIssues)
				assert.Less(t, analysis.CacheabilityScore, 0.5)
				assert.NotEmpty(t, analysis.Recommendations)
			},
		},
		{
			name:   "prompt with variable content",
			prompt: "Session ID: abc123\nUser UUID: def456\nPlease process this request.",
			validate: func(t *testing.T, analysis CacheAnalysis) {
				assert.True(t, analysis.HasVariableContent)
				assert.Less(t, analysis.CacheabilityScore, 0.8)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := optimizer.AnalyzePromptForCacheability(tt.prompt)

			assert.Equal(t, tt.prompt, analysis.Prompt)
			assert.Greater(t, analysis.EstimatedTokens, 0)
			assert.GreaterOrEqual(t, analysis.CacheabilityScore, 0.0)
			assert.LessOrEqual(t, analysis.CacheabilityScore, 1.0)
			assert.NotEmpty(t, analysis.Recommendations)

			tt.validate(t, analysis)
		})
	}
}

func TestResetMetrics(t *testing.T) {
	config := CacheConfig{
		StablePrefix:  "Test prefix",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	// Generate some metrics
	optimizer.RecordCacheHit(100)
	optimizer.RecordCacheHit(200)
	optimizer.RecordCacheMiss()

	// Verify metrics exist
	metrics := optimizer.GetMetrics()
	assert.Greater(t, metrics.Hits, int64(0))
	assert.Greater(t, metrics.Misses, int64(0))

	// Reset metrics
	optimizer.ResetMetrics()

	// Verify metrics are cleared
	metricsAfterReset := optimizer.GetMetrics()
	assert.Equal(t, int64(0), metricsAfterReset.Hits)
	assert.Equal(t, int64(0), metricsAfterReset.Misses)
	assert.Equal(t, int64(0), metricsAfterReset.TokensSaved)
	assert.Equal(t, float64(0.0), metricsAfterReset.CostSavings)
}

func TestTimestampGranularityHandling(t *testing.T) {
	timestamp := time.Date(2024, 1, 15, 14, 30, 45, 123456789, time.UTC)

	tests := []struct {
		granularity string
		expected    string
	}{
		{"day", "2024-01-15"},
		{"hour", "2024-01-15 14"},
		{"minute", "2024-01-15 14:30"},
		{"invalid", "2024-01-15"}, // Should fall back to day
	}

	for _, tt := range tests {
		t.Run(tt.granularity, func(t *testing.T) {
			config := CacheConfig{
				StablePrefix:         "Test prefix",
				TimestampGranularity: tt.granularity,
			}

			optimizer := NewCacheOptimizer(config)
			result := optimizer.OptimizePrompt("test prompt", timestamp)

			assert.Contains(t, result, tt.expected)

			// Make sure we don't have more precision than expected
			if tt.granularity == "day" {
				assert.NotContains(t, result, "14:")
			}
			if tt.granularity == "hour" {
				assert.NotContains(t, result, ":30")
			}
			if tt.granularity == "minute" {
				assert.NotContains(t, result, ":45")
			}
		})
	}
}

// Benchmark tests for performance

func BenchmarkOptimizePrompt(b *testing.B) {
	config := CacheConfig{
		StablePrefix:         "You are a helpful AI assistant with extensive knowledge.",
		TimestampGranularity: "day",
		EnableMetrics:        false, // Disable for pure optimization benchmark
	}

	optimizer := NewCacheOptimizer(config)
	prompt := "Please analyze the following data and provide insights on market trends, consumer behavior, and potential business opportunities."
	timestamp := time.Now()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = optimizer.OptimizePrompt(prompt, timestamp)
	}
}

func BenchmarkRecordCacheHit(b *testing.B) {
	config := CacheConfig{
		StablePrefix:  "Benchmark prefix",
		EnableMetrics: true,
	}

	optimizer := NewCacheOptimizer(config)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		optimizer.RecordCacheHit(100)
	}
}

func BenchmarkAnalyzePromptForCacheability(b *testing.B) {
	config := CacheConfig{
		StablePrefix: "You are a helpful assistant.",
	}

	optimizer := NewCacheOptimizer(config)
	prompt := "You are a helpful assistant.\n\nPlease analyze this complex prompt with multiple sections, timestamps, and variable content to determine its cacheability score and provide optimization recommendations."

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = optimizer.AnalyzePromptForCacheability(prompt)
	}
}
