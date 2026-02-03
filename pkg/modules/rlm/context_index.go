package rlm

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
)

// Chunk represents a segment of the context with metadata.
type Chunk struct {
	ID        int      // Unique identifier
	Content   string   // The actual text content
	StartLine int      // Starting line number (1-indexed)
	EndLine   int      // Ending line number (1-indexed)
	Summary   string   // Optional summary of the chunk
	Embedding []float32 // Embedding vector for semantic search
}

// ContextIndex provides efficient access to context slices.
// It supports both line-based access and semantic search via embeddings.
type ContextIndex struct {
	mu         sync.RWMutex
	rawContent string            // Original full content
	lines      []string          // Content split by lines
	chunks     []Chunk           // Chunked segments
	embeddings [][]float32       // Chunk embeddings (parallel to chunks)
	summaries  map[int]string    // Chunk ID -> summary
	indexed    bool              // Whether embeddings have been computed
	embeddingFn EmbeddingFunc    // Function to compute embeddings
}

// EmbeddingFunc computes embeddings for text inputs.
type EmbeddingFunc func(ctx context.Context, texts []string) ([][]float32, error)

// ChunkConfig configures how content is chunked.
type ChunkConfig struct {
	// MaxChunkSize is the maximum number of characters per chunk
	MaxChunkSize int
	// OverlapSize is the number of characters to overlap between chunks
	OverlapSize int
	// ChunkByLines if true, chunks by line count instead of character count
	ChunkByLines bool
	// LinesPerChunk when ChunkByLines is true
	LinesPerChunk int
}

// DefaultChunkConfig returns sensible defaults for chunking.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{
		MaxChunkSize:  4000,  // ~1000 tokens
		OverlapSize:   200,   // ~50 tokens overlap
		ChunkByLines:  false,
		LinesPerChunk: 50,
	}
}

// NewContextIndex creates a new context index from raw content.
func NewContextIndex(content string, config ChunkConfig) *ContextIndex {
	idx := &ContextIndex{
		rawContent: content,
		lines:      strings.Split(content, "\n"),
		summaries:  make(map[int]string),
	}

	idx.buildChunks(config)
	return idx
}

// buildChunks splits the content into chunks based on configuration.
func (idx *ContextIndex) buildChunks(config ChunkConfig) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if config.ChunkByLines {
		idx.chunkByLines(config.LinesPerChunk, config.OverlapSize/10) // overlap in lines
	} else {
		idx.chunkBySize(config.MaxChunkSize, config.OverlapSize)
	}
}

func (idx *ContextIndex) chunkByLines(linesPerChunk, overlapLines int) {
	if linesPerChunk <= 0 {
		linesPerChunk = 50
	}
	if overlapLines < 0 {
		overlapLines = 5
	}

	idx.chunks = nil
	chunkID := 0

	for i := 0; i < len(idx.lines); {
		endLine := i + linesPerChunk
		if endLine > len(idx.lines) {
			endLine = len(idx.lines)
		}

		content := strings.Join(idx.lines[i:endLine], "\n")
		idx.chunks = append(idx.chunks, Chunk{
			ID:        chunkID,
			Content:   content,
			StartLine: i + 1, // 1-indexed
			EndLine:   endLine,
		})

		chunkID++
		i = endLine - overlapLines
		if i <= (endLine - linesPerChunk) {
			i = endLine // Prevent infinite loop
		}
	}
}

func (idx *ContextIndex) chunkBySize(maxSize, overlap int) {
	if maxSize <= 0 {
		maxSize = 4000
	}
	if overlap < 0 {
		overlap = 0
	}
	// Ensure overlap is smaller than maxSize
	if overlap >= maxSize {
		overlap = maxSize / 4
	}

	idx.chunks = nil
	chunkID := 0
	content := idx.rawContent

	if len(content) == 0 {
		return
	}

	pos := 0
	lineNum := 1

	for pos < len(content) {
		endPos := pos + maxSize
		if endPos > len(content) {
			endPos = len(content)
		}

		// Try to break at a newline
		if endPos < len(content) {
			lastNewline := strings.LastIndex(content[pos:endPos], "\n")
			if lastNewline > 0 {
				endPos = pos + lastNewline + 1
			}
		}

		chunkContent := content[pos:endPos]
		endLineNum := lineNum + strings.Count(chunkContent, "\n")

		idx.chunks = append(idx.chunks, Chunk{
			ID:        chunkID,
			Content:   chunkContent,
			StartLine: lineNum,
			EndLine:   endLineNum,
		})

		chunkID++
		lineNum = endLineNum

		// Calculate next position with overlap
		nextPos := endPos - overlap
		// Ensure we always make progress
		if nextPos <= pos {
			nextPos = endPos
		}
		pos = nextPos
	}
}

// SetEmbeddingFunc sets the function used to compute embeddings.
func (idx *ContextIndex) SetEmbeddingFunc(fn EmbeddingFunc) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.embeddingFn = fn
}

// IndexEagerly computes embeddings for all chunks immediately.
func (idx *ContextIndex) IndexEagerly(ctx context.Context) error {
	idx.mu.Lock()
	if idx.embeddingFn == nil {
		idx.mu.Unlock()
		return nil // No embedding function, skip indexing
	}
	chunks := idx.chunks
	fn := idx.embeddingFn
	idx.mu.Unlock()

	// Collect chunk contents
	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Content
	}

	// Compute embeddings
	embeddings, err := fn(ctx, texts)
	if err != nil {
		return err
	}

	// Store embeddings
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.embeddings = embeddings
	for i := range idx.chunks {
		if i < len(embeddings) {
			idx.chunks[i].Embedding = embeddings[i]
		}
	}
	idx.indexed = true

	return nil
}

// FindRelevant returns the top-k most relevant chunks for a query.
// If embeddings are not indexed, returns chunks based on keyword matching.
func (idx *ContextIndex) FindRelevant(ctx context.Context, query string, topK int) ([]Chunk, error) {
	idx.mu.RLock()
	indexed := idx.indexed
	chunks := idx.chunks
	fn := idx.embeddingFn
	idx.mu.RUnlock()

	if topK <= 0 {
		topK = 5
	}
	if topK > len(chunks) {
		topK = len(chunks)
	}

	if indexed && fn != nil {
		return idx.findByEmbedding(ctx, query, topK, fn)
	}

	// Fallback to keyword matching
	return idx.findByKeyword(query, topK), nil
}

func (idx *ContextIndex) findByEmbedding(ctx context.Context, query string, topK int, fn EmbeddingFunc) ([]Chunk, error) {
	// Compute query embedding
	queryEmbeddings, err := fn(ctx, []string{query})
	if err != nil {
		// Fallback to keyword matching on error
		return idx.findByKeyword(query, topK), nil
	}

	if len(queryEmbeddings) == 0 || len(queryEmbeddings[0]) == 0 {
		return idx.findByKeyword(query, topK), nil
	}

	queryVec := queryEmbeddings[0]

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Score chunks by cosine similarity
	type scored struct {
		chunk Chunk
		score float64
	}

	scores := make([]scored, 0, len(idx.chunks))
	for _, chunk := range idx.chunks {
		if len(chunk.Embedding) == 0 {
			continue
		}
		score := cosineSimilarity(queryVec, chunk.Embedding)
		scores = append(scores, scored{chunk: chunk, score: score})
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return top-k
	results := make([]Chunk, 0, topK)
	for i := 0; i < topK && i < len(scores); i++ {
		results = append(results, scores[i].chunk)
	}

	return results, nil
}

func (idx *ContextIndex) findByKeyword(query string, topK int) []Chunk {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	queryLower := strings.ToLower(query)
	queryWords := strings.Fields(queryLower)

	type scored struct {
		chunk Chunk
		score int
	}

	scores := make([]scored, 0, len(idx.chunks))
	for _, chunk := range idx.chunks {
		contentLower := strings.ToLower(chunk.Content)
		score := 0
		for _, word := range queryWords {
			score += strings.Count(contentLower, word)
		}
		if score > 0 {
			scores = append(scores, scored{chunk: chunk, score: score})
		}
	}

	// Sort by score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return top-k
	results := make([]Chunk, 0, topK)
	for i := 0; i < topK && i < len(scores); i++ {
		results = append(results, scores[i].chunk)
	}

	// If no matches, return first chunks
	if len(results) == 0 && len(idx.chunks) > 0 {
		for i := 0; i < topK && i < len(idx.chunks); i++ {
			results = append(results, idx.chunks[i])
		}
	}

	return results
}

// GetChunk returns a chunk by its ID.
func (idx *ContextIndex) GetChunk(id int) (Chunk, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if id < 0 || id >= len(idx.chunks) {
		return Chunk{}, false
	}
	return idx.chunks[id], true
}

// GetContext returns content between start and end lines (1-indexed, inclusive).
func (idx *ContextIndex) GetContext(startLine, endLine int) string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Convert to 0-indexed
	start := startLine - 1
	end := endLine

	if start < 0 {
		start = 0
	}
	if end > len(idx.lines) {
		end = len(idx.lines)
	}
	if start >= end {
		return ""
	}

	return strings.Join(idx.lines[start:end], "\n")
}

// ChunkCount returns the number of chunks.
func (idx *ContextIndex) ChunkCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.chunks)
}

// LineCount returns the number of lines.
func (idx *ContextIndex) LineCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.lines)
}

// IsIndexed returns whether embeddings have been computed.
func (idx *ContextIndex) IsIndexed() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.indexed
}

// SetChunkSummary sets a summary for a chunk.
func (idx *ContextIndex) SetChunkSummary(chunkID int, summary string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if chunkID >= 0 && chunkID < len(idx.chunks) {
		idx.chunks[chunkID].Summary = summary
		idx.summaries[chunkID] = summary
	}
}

// AllChunks returns all chunks.
func (idx *ContextIndex) AllChunks() []Chunk {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	result := make([]Chunk, len(idx.chunks))
	copy(result, idx.chunks)
	return result
}

// GetRawContent returns the raw content string.
func (idx *ContextIndex) GetRawContent() string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.rawContent
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
