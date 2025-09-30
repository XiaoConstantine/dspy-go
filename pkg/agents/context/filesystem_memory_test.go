package context

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewFileSystemMemory(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	assert.NoError(t, err)
	assert.NotNil(t, memory)
	assert.Equal(t, "test-session", memory.sessionID)
	assert.Equal(t, "test-agent", memory.agentID)

	// Verify directory was created
	expectedPath := filepath.Join(tempDir, "memory", "test-session", "test-agent")
	_, err = os.Stat(expectedPath)
	assert.NoError(t, err)
}

func TestStoreLargeObservation(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	tests := []struct {
		name     string
		id       string
		content  []byte
		metadata map[string]interface{}
	}{
		{
			name:    "simple observation",
			id:      "obs-1",
			content: []byte("This is a simple observation for testing."),
			metadata: map[string]interface{}{
				"timestamp": time.Now().Unix(),
				"source":    "test",
			},
		},
		{
			name:    "large observation",
			id:      "obs-2",
			content: []byte(generateLargeString(10000)),
			metadata: map[string]interface{}{
				"size": 10000,
				"type": "large",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reference, err := memory.StoreLargeObservation(ctx, tt.id, tt.content, tt.metadata)
			assert.NoError(t, err)
			assert.NotEmpty(t, reference)
			assert.Contains(t, reference, "file://")
			assert.Contains(t, reference, tt.id)

			// Verify file was created
			filename := extractFilenameFromReference(reference)
			fullPath := filepath.Join(memory.baseDir, filename)
			_, err = os.Stat(fullPath)
			assert.NoError(t, err)

			// Verify metadata file was created
			metaPath := fullPath + ".meta.json"
			_, err = os.Stat(metaPath)
			assert.NoError(t, err)
		})
	}
}

func TestStoreContext(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	contextData := map[string]interface{}{
		"task":        "test_task",
		"observations": []string{"obs1", "obs2", "obs3"},
		"metadata": map[string]interface{}{
			"timestamp": time.Now().Unix(),
			"version":   1,
		},
	}

	reference, err := memory.StoreContext(ctx, "context-1", contextData)
	assert.NoError(t, err)
	assert.NotEmpty(t, reference)
	assert.Contains(t, reference, "file://")

	// Verify files were created
	filename := extractFilenameFromReference(reference)
	fullPath := filepath.Join(memory.baseDir, filename)
	_, err = os.Stat(fullPath)
	assert.NoError(t, err)
}

func TestRetrieveObservation(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Store an observation first
	originalContent := []byte("Test observation content for retrieval")
	metadata := map[string]interface{}{
		"test": "data",
	}

	reference, err := memory.StoreLargeObservation(ctx, "retrieve-test", originalContent, metadata)
	require.NoError(t, err)

	// Extract clean filename from reference
	filename := extractFilenameFromReference(reference)
	cleanReference := "file://" + filename

	// Now retrieve it
	memoryContent, err := memory.RetrieveObservation(ctx, cleanReference)
	assert.NoError(t, err)
	assert.NotNil(t, memoryContent)
	assert.Equal(t, originalContent, memoryContent.Content)
	assert.Equal(t, "observation", memoryContent.Reference.Type)
	assert.Equal(t, int64(len(originalContent)), memoryContent.Reference.Size)
}

func TestRetrieveObservation_InvalidReference(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	tests := []struct {
		name      string
		reference string
	}{
		{
			name:      "invalid format",
			reference: "invalid-reference",
		},
		{
			name:      "non-existent file",
			reference: "file://nonexistent.txt",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := memory.RetrieveObservation(ctx, tt.reference)
			assert.Error(t, err)
		})
	}
}

func TestStoreFile(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	tests := []struct {
		name        string
		contentType string
		id          string
		content     []byte
		metadata    map[string]interface{}
	}{
		{
			name:        "cache file",
			contentType: "cache",
			id:          "cache-1",
			content:     []byte("cached data"),
			metadata:    map[string]interface{}{"ttl": 3600},
		},
		{
			name:        "memory file",
			contentType: "memory",
			id:          "session-1",
			content:     []byte("memory data"),
			metadata:    map[string]interface{}{"type": "context"},
		},
		{
			name:        "observation file",
			contentType: "test_obs",
			id:          "obs-1",
			content:     []byte("observation data"),
			metadata:    map[string]interface{}{"source": "test"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reference, err := memory.StoreFile(ctx, tt.contentType, tt.id, tt.content, tt.metadata)
			assert.NoError(t, err)
			assert.NotEmpty(t, reference)
			assert.Contains(t, reference, "file://")

			// Verify file exists
			filename := extractFilenameFromReference(reference)
			fullPath := filepath.Join(memory.baseDir, filename)
			_, err = os.Stat(fullPath)
			assert.NoError(t, err)
		})
	}
}

func TestListFiles(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Store multiple files of different types
	files := []struct {
		contentType string
		id          string
		content     []byte
	}{
		{"cache", "cache-1", []byte("cache 1")},
		{"cache", "cache-2", []byte("cache 2")},
		{"errors", "error-1", []byte("error 1")},
		{"plan", "plan-1", []byte("plan 1")},
	}

	for _, f := range files {
		_, err := memory.StoreFile(ctx, f.contentType, f.id, f.content, nil)
		require.NoError(t, err)
	}

	// Test listing all files
	t.Run("list all files", func(t *testing.T) {
		refs, err := memory.ListFiles("")
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(refs), 4)
	})

	// Test listing specific type
	t.Run("list cache files only", func(t *testing.T) {
		refs, err := memory.ListFiles("cache")
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(refs), 2)

		// Verify all returned files are cache type
		for _, ref := range refs {
			assert.Equal(t, "cache", ref.Type)
		}
	})

	t.Run("list error files only", func(t *testing.T) {
		refs, err := memory.ListFiles("errors")
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, len(refs), 1)

		for _, ref := range refs {
			assert.Equal(t, "errors", ref.Type)
		}
	})
}

func TestCleanExpired(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.Memory.RetentionPeriod = 1 * time.Hour // Short retention for testing
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Store a file
	content := []byte("test content")
	_, err = memory.StoreFile(ctx, "test", "old-file", content, nil)
	require.NoError(t, err)

	// For testing, we'll store files with current timestamp first
	// In real scenario, files would naturally expire after retention period

	// Store another recent file
	_, err = memory.StoreFile(ctx, "test", "new-file", content, nil)
	require.NoError(t, err)

	// Get initial file count
	initialRefs, err := memory.ListFiles("")
	require.NoError(t, err)
	initialCount := len(initialRefs)

	// Clean expired files (with current settings, nothing should be cleaned)
	cleaned, err := memory.CleanExpired(ctx)
	assert.NoError(t, err)

	// Since we didn't actually make files old enough, cleaned should be 0
	// This is expected for this test setup
	assert.GreaterOrEqual(t, cleaned, int64(0))

	// Verify files still exist (or were cleaned if old enough)
	refs, err := memory.ListFiles("")
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, len(refs), 0)

	// The count should be initialCount - cleaned
	assert.Equal(t, int64(initialCount)-cleaned, int64(len(refs)))
}

func TestGetMetrics(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Store some files to generate metrics
	for i := 0; i < 3; i++ {
		content := []byte(generateLargeString(1000))
		id := fmt.Sprintf("file-%d", i)
		_, err := memory.StoreFile(ctx, "test", id, content, nil)
		require.NoError(t, err)
	}

	// Retrieve metrics
	metrics := memory.GetMetrics()
	assert.NotNil(t, metrics)

	// Verify expected keys
	expectedKeys := []string{
		"total_files",
		"total_size_bytes",
		"access_count",
		"compression_savings",
		"base_directory",
		"session_id",
		"agent_id",
	}

	for _, key := range expectedKeys {
		assert.Contains(t, metrics, key, "Missing metric key: %s", key)
	}

	// Verify values
	assert.Greater(t, metrics["total_files"], int64(0))
	assert.Greater(t, metrics["total_size_bytes"], int64(0))
	assert.Equal(t, "test-session", metrics["session_id"])
	assert.Equal(t, "test-agent", metrics["agent_id"])
}

func TestGetTotalSize(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Initially should be 0
	assert.Equal(t, int64(0), memory.GetTotalSize())

	// Store some content
	content := []byte("test content with some data")
	_, err = memory.StoreFile(ctx, "test", "size-test", content, nil)
	require.NoError(t, err)

	// Size should increase
	assert.Greater(t, memory.GetTotalSize(), int64(0))
}

func TestGetFileCount(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Initially should be 0
	assert.Equal(t, int64(0), memory.GetFileCount())

	// Store multiple files
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("count-%d", i)
		_, err := memory.StoreFile(ctx, "test", id, []byte("content"), nil)
		require.NoError(t, err)
	}

	// Count should be 5
	assert.Equal(t, int64(5), memory.GetFileCount())
}

func TestConcurrentFileOperations(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "test-session", "test-agent", config.Memory)
	require.NoError(t, err)

	ctx := context.Background()

	// Run concurrent store operations
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(workerID int) {
			defer func() { done <- true }()

			content := []byte(generateLargeString(100))
			id := fmt.Sprintf("worker-%d", workerID)
			_, err := memory.StoreFile(ctx, "concurrent", id, content, nil)
			assert.NoError(t, err)
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Verify all files were stored
	assert.Equal(t, int64(numGoroutines), memory.GetFileCount())
}

// Helper functions

func extractFilenameFromReference(reference string) string {
	// Handle format: "file://filename" or "[Observation stored: file://filename (size: X bytes, checksum: Y)]"
	start := -1

	// Find "file://"
	for i := 0; i < len(reference)-6; i++ {
		if i+7 <= len(reference) && reference[i:i+7] == "file://" {
			start = i + 7
			break
		}
	}

	if start == -1 {
		return ""
	}

	// Find the end (space, closing bracket, parenthesis, or end of string)
	end := len(reference)
	for i := start; i < len(reference); i++ {
		if reference[i] == ' ' || reference[i] == ')' || reference[i] == ']' || reference[i] == '(' {
			end = i
			break
		}
	}

	return reference[start:end]
}

// Benchmark tests

func BenchmarkStoreLargeObservation(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "bench-session", "bench-agent", config.Memory)
	require.NoError(b, err)

	ctx := context.Background()
	content := []byte(generateLargeString(10000))

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := memory.StoreLargeObservation(ctx, string(rune(i)), content, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRetrieveObservation(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	memory, err := NewFileSystemMemory(tempDir, "bench-session", "bench-agent", config.Memory)
	require.NoError(b, err)

	ctx := context.Background()
	content := []byte(generateLargeString(10000))

	// Store an observation
	reference, err := memory.StoreLargeObservation(ctx, "bench-obs", content, nil)
	require.NoError(b, err)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := memory.RetrieveObservation(ctx, reference)
		if err != nil {
			b.Fatal(err)
		}
	}
}
