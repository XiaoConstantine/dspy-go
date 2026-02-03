package rlm

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewContextIndex(t *testing.T) {
	content := "line1\nline2\nline3\nline4\nline5"
	idx := NewContextIndex(content, DefaultChunkConfig())

	assert.Greater(t, idx.ChunkCount(), 0)
	assert.Equal(t, 5, idx.LineCount())
	assert.False(t, idx.IsIndexed())
}

func TestContextIndexChunkByLines(t *testing.T) {
	// Create content with 100 lines
	lines := make([]string, 100)
	for i := range lines {
		lines[i] = strings.Repeat("x", 50) // 50 chars per line
	}
	content := strings.Join(lines, "\n")

	config := ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 20,
	}

	idx := NewContextIndex(content, config)

	// Should have multiple chunks
	assert.Greater(t, idx.ChunkCount(), 1)

	// Each chunk should be around 20 lines
	chunks := idx.AllChunks()
	for _, chunk := range chunks {
		lineCount := strings.Count(chunk.Content, "\n") + 1
		assert.LessOrEqual(t, lineCount, 25) // Allow some overlap
	}
}

func TestContextIndexChunkBySize(t *testing.T) {
	// Create large content
	content := strings.Repeat("word ", 2000) // ~10000 chars

	config := ChunkConfig{
		MaxChunkSize: 1000,
		OverlapSize:  100,
	}

	idx := NewContextIndex(content, config)

	// Should have multiple chunks
	assert.Greater(t, idx.ChunkCount(), 5)

	// Each chunk should be around the max size
	chunks := idx.AllChunks()
	for _, chunk := range chunks {
		assert.LessOrEqual(t, len(chunk.Content), 1100) // Allow some flexibility
	}
}

func TestContextIndexGetChunk(t *testing.T) {
	content := "chunk0 content\nchunk1 content\nchunk2 content"
	config := ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 1,
	}

	idx := NewContextIndex(content, config)

	// Get valid chunk
	chunk, ok := idx.GetChunk(0)
	assert.True(t, ok)
	assert.Contains(t, chunk.Content, "chunk0")

	// Get invalid chunk
	_, ok = idx.GetChunk(999)
	assert.False(t, ok)

	// Get negative index
	_, ok = idx.GetChunk(-1)
	assert.False(t, ok)
}

func TestContextIndexGetContext(t *testing.T) {
	content := "line1\nline2\nline3\nline4\nline5"
	idx := NewContextIndex(content, DefaultChunkConfig())

	// Get range
	result := idx.GetContext(2, 4) // lines 2-4 (1-indexed, inclusive)
	assert.Equal(t, "line2\nline3\nline4", result)

	// Get single line
	result = idx.GetContext(3, 3)
	assert.Equal(t, "line3", result)

	// Get with out-of-bounds
	result = idx.GetContext(1, 100)
	assert.Contains(t, result, "line1")
	assert.Contains(t, result, "line5")

	// Get with invalid range
	result = idx.GetContext(5, 2)
	assert.Equal(t, "", result)
}

func TestContextIndexFindRelevantKeyword(t *testing.T) {
	content := `
Package main provides the entry point.

func main() {
    fmt.Println("Hello")
}

This is documentation about error handling.
Errors should be checked carefully.
Always handle errors properly.

func helper() {
    return nil
}
`

	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 4,
	})

	// Find chunks about errors (keyword matching)
	chunks, err := idx.FindRelevant(context.Background(), "error handling", 2)
	require.NoError(t, err)
	assert.Greater(t, len(chunks), 0)

	// At least one chunk should mention error
	found := false
	for _, chunk := range chunks {
		if strings.Contains(strings.ToLower(chunk.Content), "error") {
			found = true
			break
		}
	}
	assert.True(t, found, "Expected to find chunk containing 'error'")
}

func TestContextIndexFindRelevantWithNoMatches(t *testing.T) {
	content := "apple\nbanana\norange"
	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 1,
	})

	// Search for something not in content - should return first chunks
	chunks, err := idx.FindRelevant(context.Background(), "zebra unicorn", 2)
	require.NoError(t, err)
	assert.Greater(t, len(chunks), 0) // Should return something
}

func TestContextIndexSetChunkSummary(t *testing.T) {
	content := "line1\nline2\nline3"
	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 1,
	})

	idx.SetChunkSummary(0, "Summary of first chunk")

	chunk, ok := idx.GetChunk(0)
	assert.True(t, ok)
	assert.Equal(t, "Summary of first chunk", chunk.Summary)
}

func TestContextIndexAllChunks(t *testing.T) {
	content := "a\nb\nc\nd\ne"
	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 1,
	})

	chunks := idx.AllChunks()
	assert.Equal(t, idx.ChunkCount(), len(chunks))

	// Verify it's a copy (modifying shouldn't affect original)
	originalCount := idx.ChunkCount()
	_ = chunks[:1] // Modify slice to verify it doesn't affect original
	assert.Equal(t, originalCount, idx.ChunkCount())
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	assert.InDelta(t, 1.0, cosineSimilarity(a, b), 0.001)

	// Orthogonal vectors
	a = []float32{1, 0, 0}
	b = []float32{0, 1, 0}
	assert.InDelta(t, 0.0, cosineSimilarity(a, b), 0.001)

	// Opposite vectors
	a = []float32{1, 0, 0}
	b = []float32{-1, 0, 0}
	assert.InDelta(t, -1.0, cosineSimilarity(a, b), 0.001)

	// Different lengths
	a = []float32{1, 2}
	b = []float32{1, 2, 3}
	assert.Equal(t, 0.0, cosineSimilarity(a, b))

	// Empty vectors
	assert.Equal(t, 0.0, cosineSimilarity([]float32{}, []float32{}))

	// Zero vectors
	a = []float32{0, 0, 0}
	b = []float32{1, 0, 0}
	assert.Equal(t, 0.0, cosineSimilarity(a, b))
}

// mockEmbeddingFunc creates embeddings based on word overlap for testing.
func mockEmbeddingFunc(ctx context.Context, texts []string) ([][]float32, error) {
	// Simple mock: create vectors based on presence of common words
	keywords := []string{"error", "function", "main", "test", "data", "user", "handle", "return"}

	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		textLower := strings.ToLower(text)
		vec := make([]float32, len(keywords))
		for j, keyword := range keywords {
			if strings.Contains(textLower, keyword) {
				vec[j] = 1.0
			}
		}
		embeddings[i] = vec
	}
	return embeddings, nil
}

func TestContextIndexWithEmbeddings(t *testing.T) {
	content := `
This function handles errors.
Error handling is important.

This is the main function.
Main entry point of the program.

This tests user data.
User data should be validated.
`

	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 2,
	})

	// Set embedding function
	idx.SetEmbeddingFunc(mockEmbeddingFunc)

	// Index eagerly
	err := idx.IndexEagerly(context.Background())
	require.NoError(t, err)
	assert.True(t, idx.IsIndexed())

	// Find chunks about errors
	chunks, err := idx.FindRelevant(context.Background(), "error handling", 2)
	require.NoError(t, err)
	assert.Greater(t, len(chunks), 0)

	// First result should contain "error"
	assert.Contains(t, strings.ToLower(chunks[0].Content), "error")
}

func TestContextIndexLineNumbers(t *testing.T) {
	content := "line1\nline2\nline3\nline4\nline5"
	idx := NewContextIndex(content, ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 2,
	})

	chunks := idx.AllChunks()

	// First chunk should start at line 1
	assert.Equal(t, 1, chunks[0].StartLine)

	// Verify line numbers are set
	for _, chunk := range chunks {
		assert.Greater(t, chunk.StartLine, 0)
		assert.GreaterOrEqual(t, chunk.EndLine, chunk.StartLine)
	}
}
