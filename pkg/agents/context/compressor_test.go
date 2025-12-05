package context

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCompressor(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)

	assert.NotNil(t, compressor)
	assert.Equal(t, memory, compressor.memory)
	assert.Equal(t, config.CompressionThreshold, compressor.compressionThreshold)
}

func TestCompressContent_SmallContent(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	// Content below threshold should not be compressed
	content := CompressibleContent{
		Content:     "Small content that doesn't need compression",
		ContentType: "text",
		Priority:    PriorityMedium,
	}

	result, compressionResult, err := compressor.CompressContent(ctx, content)
	assert.NoError(t, err)
	assert.Equal(t, content.Content, result)
	assert.Equal(t, "none", compressionResult.Method)
	assert.Equal(t, float64(1.0), compressionResult.CompressionRatio)
}

func TestCompressContent_Code(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 100 // Lower threshold for testing
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	tests := []struct {
		name     string
		content  string
		priority CompressionPriority
		validate func(t *testing.T, compressed string, result CompressionResult)
	}{
		{
			name: "high priority code - minimal compression",
			content: `package main

import "fmt"

// This is a comment
func main() {
    fmt.Println("Hello, World!")
    // Another comment
    x := 42
}`,
			priority: PriorityHigh,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				assert.Contains(t, result.Method, "whitespace")
				assert.LessOrEqual(t, result.CompressionRatio, 1.0)
			},
		},
		{
			name: "medium priority code - remove comments",
			content: `package main

import "fmt"

// This is a comment that should be removed
func main() {
    fmt.Println("Hello, World!")
    // Another comment to remove
    x := 42
}`,
			priority: PriorityMedium,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				// Comments should be removed
				assert.NotContains(t, compressed, "// This is a comment")
				assert.Less(t, result.CompressionRatio, 1.0)
			},
		},
		{
			name: "low priority code - create summary",
			content: `package main

import "fmt"

func hello() string {
    return "Hello"
}

func world() string {
    return "World"
}

func main() {
    fmt.Println(hello() + world())
}`,
			priority: PriorityLow,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				assert.Equal(t, "summary", result.Method)
				assert.Contains(t, compressed, "Code Summary")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := CompressibleContent{
				Content:     tt.content,
				ContentType: "code",
				Priority:    tt.priority,
			}

			compressed, result, err := compressor.CompressContent(ctx, content)
			assert.NoError(t, err)
			tt.validate(t, compressed, result)
		})
	}
}

func TestCompressContent_JSON(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	tests := []struct {
		name     string
		content  string
		priority CompressionPriority
		validate func(t *testing.T, compressed string, result CompressionResult)
	}{
		{
			name: "high priority JSON - compact",
			content: `{
  "name": "test",
  "value": 42,
  "nested": {
    "key": "value"
  }
}`,
			priority: PriorityHigh,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				assert.Equal(t, "compact", result.Method)
				// Should remove whitespace
				assert.NotContains(t, compressed, "\n  ")
			},
		},
		{
			name: "low priority JSON - summary",
			content: `{
  "users": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ],
  "count": 2
}`,
			priority: PriorityLow,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				assert.Equal(t, "summary", result.Method)
				assert.Contains(t, compressed, "JSON Summary")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := CompressibleContent{
				Content:     tt.content,
				ContentType: "json",
				Priority:    tt.priority,
			}

			compressed, result, err := compressor.CompressContent(ctx, content)
			assert.NoError(t, err)
			tt.validate(t, compressed, result)
		})
	}
}

func TestCompressContent_Logs(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	logContent := `2024-01-15 10:00:00 INFO Starting application
2024-01-15 10:00:01 DEBUG Loading configuration
2024-01-15 10:00:02 INFO Configuration loaded
2024-01-15 10:00:03 ERROR Failed to connect to database
2024-01-15 10:00:04 WARN Retrying connection
2024-01-15 10:00:05 INFO Connection successful
2024-01-15 10:00:06 DEBUG Processing request
2024-01-15 10:00:07 ERROR Invalid input received
2024-01-15 10:00:08 INFO Request completed`

	tests := []struct {
		name     string
		priority CompressionPriority
		validate func(t *testing.T, compressed string)
	}{
		{
			name:     "medium priority - keep important logs",
			priority: PriorityMedium,
			validate: func(t *testing.T, compressed string) {
				assert.Contains(t, compressed, "ERROR")
				assert.Contains(t, compressed, "WARN")
				assert.Contains(t, compressed, "INFO")
			},
		},
		{
			name:     "low priority - keep only critical",
			priority: PriorityLow,
			validate: func(t *testing.T, compressed string) {
				assert.Contains(t, compressed, "ERROR")
				// Should prioritize errors
				lines := strings.Split(compressed, "\n")
				errorCount := 0
				for _, line := range lines {
					if strings.Contains(line, "ERROR") {
						errorCount++
					}
				}
				assert.Greater(t, errorCount, 0)
			},
		},
		{
			name:     "minimal priority - summary only",
			priority: PriorityMinimal,
			validate: func(t *testing.T, compressed string) {
				assert.Contains(t, compressed, "Log Summary")
				assert.Contains(t, compressed, "errors")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := CompressibleContent{
				Content:     logContent,
				ContentType: "logs",
				Priority:    tt.priority,
			}

			compressed, result, err := compressor.CompressContent(ctx, content)
			assert.NoError(t, err)
			assert.NotEmpty(t, compressed)
			assert.Less(t, result.CompressionRatio, 1.0)
			tt.validate(t, compressed)
		})
	}
}

func TestCompressContent_Text(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	textContent := `This is a test document with multiple sentences.
The document contains important information about testing.
Testing is critical for software quality. Testing is critical for software quality.
We should test everything thoroughly.`

	tests := []struct {
		name     string
		priority CompressionPriority
		validate func(t *testing.T, compressed string, result CompressionResult)
	}{
		{
			name:     "high priority - normalize whitespace",
			priority: PriorityHigh,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				// Should contain the content
				assert.Contains(t, compressed, "test document")
			},
		},
		{
			name:     "medium priority - remove redundancy",
			priority: PriorityMedium,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				// Redundant sentence should appear only once
				count := strings.Count(compressed, "Testing is critical for software quality")
				assert.Equal(t, 1, count)
			},
		},
		{
			name:     "low priority - key sentences",
			priority: PriorityLow,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				// Should extract important sentences
				assert.Contains(t, compressed, "important")
			},
		},
		{
			name:     "minimal priority - summary",
			priority: PriorityMinimal,
			validate: func(t *testing.T, compressed string, result CompressionResult) {
				assert.NotEmpty(t, compressed)
				assert.Contains(t, compressed, "Text Summary")
				assert.Contains(t, compressed, "words")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := CompressibleContent{
				Content:     textContent,
				ContentType: "text",
				Priority:    tt.priority,
			}

			compressed, result, err := compressor.CompressContent(ctx, content)
			assert.NoError(t, err)
			tt.validate(t, compressed, result)
		})
	}
}

func TestCompressContent_Observations(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	observationContent := `Observation 1: System is running normally.
Observation 2: CPU usage at 45%.
Observation 3: Memory usage at 60%.
Observation 4: No errors detected.`

	content := CompressibleContent{
		Content:     observationContent,
		ContentType: "observations",
		Priority:    PriorityMedium,
	}

	compressed, result, err := compressor.CompressContent(ctx, content)
	assert.NoError(t, err)
	assert.NotEmpty(t, compressed)
	assert.Less(t, result.CompressionRatio, 1.0)
	assert.Equal(t, "observation_structured", result.Method)
}

func TestCompressAndStore(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 100
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	// Large content that should be stored
	largeContent := strings.Repeat("This is a large content block that should be compressed and stored. ", 100)

	content := CompressibleContent{
		Content:     largeContent,
		ContentType: "text",
		Priority:    PriorityMedium,
	}

	result, err := compressor.CompressAndStore(ctx, "test-large", content)
	assert.NoError(t, err)
	assert.NotEmpty(t, result)
}

func TestDecompressContent(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	tests := []struct {
		name       string
		compressed string
		method     string
		validate   func(t *testing.T, decompressed string, err error)
	}{
		{
			name:       "summary method",
			compressed: "This is a summary of the content",
			method:     "summary",
			validate: func(t *testing.T, decompressed string, err error) {
				assert.NoError(t, err)
				assert.Contains(t, decompressed, "summary")
				assert.Contains(t, decompressed, "original content not recoverable")
			},
		},
		{
			name:       "structured method",
			compressed: "Structured content",
			method:     "structured",
			validate: func(t *testing.T, decompressed string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "Structured content", decompressed)
			},
		},
		{
			name:       "unknown method",
			compressed: "Unknown compression",
			method:     "unknown",
			validate: func(t *testing.T, decompressed string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "Unknown compression", decompressed)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decompressed, err := compressor.DecompressContent(ctx, tt.compressed, tt.method)
			tt.validate(t, decompressed, err)
		})
	}
}

func TestGetCompressionStats(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	// Perform several compressions
	for i := 0; i < 3; i++ {
		content := CompressibleContent{
			Content:     strings.Repeat("Test content for compression. ", 50),
			ContentType: "text",
			Priority:    PriorityMedium,
		}

		_, _, err := compressor.CompressContent(ctx, content)
		assert.NoError(t, err)
	}

	stats := compressor.GetCompressionStats()
	assert.NotNil(t, stats)

	// Verify expected keys
	assert.Contains(t, stats, "total_compressions")
	assert.Contains(t, stats, "total_bytes_original")
	assert.Contains(t, stats, "total_bytes_compressed")
	assert.Contains(t, stats, "total_bytes_saved")
	assert.Contains(t, stats, "average_compression_ratio")

	// Verify values
	assert.Equal(t, int64(3), stats["total_compressions"])
	assert.Greater(t, stats["total_bytes_original"], int64(0))
	assert.Greater(t, stats["total_bytes_compressed"], int64(0))
}

func TestCompressWithGzip(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 50
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	// Use generic type to trigger gzip compression
	largeContent := strings.Repeat("This content will be gzip compressed. ", 100)

	content := CompressibleContent{
		Content:     largeContent,
		ContentType: "unknown_type",
		Priority:    PriorityMedium,
	}

	compressed, result, err := compressor.CompressContent(ctx, content)
	assert.NoError(t, err)
	assert.NotEmpty(t, compressed)
	assert.Equal(t, "gzip", result.Method)
	assert.True(t, strings.HasPrefix(compressed, "GZIP:"))
}

func TestDecompressGzip(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	// Compress content first
	originalContent := strings.Repeat("Test content for gzip compression and decompression. ", 50)

	content := CompressibleContent{
		Content:     originalContent,
		ContentType: "unknown_type",
		Priority:    PriorityMedium,
	}

	compressed, _, err := compressor.CompressContent(ctx, content)
	require.NoError(t, err)

	// Now decompress it
	decompressed, err := compressor.DecompressContent(ctx, compressed, "gzip")
	assert.NoError(t, err)
	assert.Equal(t, originalContent, decompressed)
}

// Benchmark tests

func BenchmarkCompressContent_Text(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 100
	memory, err := NewFileSystemMemory(tempDir, "bench-session", "bench-agent", config.Memory)
	require.NoError(b, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	content := CompressibleContent{
		Content:     strings.Repeat("Benchmark text content for compression testing. ", 100),
		ContentType: "text",
		Priority:    PriorityMedium,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _, err := compressor.CompressContent(ctx, content)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCompressContent_JSON(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	config.CompressionThreshold = 100
	memory, err := NewFileSystemMemory(tempDir, "bench-session", "bench-agent", config.Memory)
	require.NoError(b, err)

	compressor := NewCompressor(memory, config)
	ctx := context.Background()

	jsonContent := `{"users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}, {"id": 2, "name": "Bob", "email": "bob@example.com"}], "count": 2, "timestamp": "2024-01-15T10:00:00Z"}`

	content := CompressibleContent{
		Content:     jsonContent,
		ContentType: "json",
		Priority:    PriorityMedium,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _, err := compressor.CompressContent(ctx, content)
		if err != nil {
			b.Fatal(err)
		}
	}
}
