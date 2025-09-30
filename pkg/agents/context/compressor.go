package context

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// Compressor implements Manus-inspired content compression for context efficiency.
// Instead of truncating content, we intelligently compress and summarize to maintain
// all important information while reducing token usage.
type Compressor struct {
	mu sync.RWMutex

	memory               *FileSystemMemory
	compressionThreshold int64
	maxCompressionRatio  float64

	// Compression statistics
	totalCompressions    int64
	totalBytesOriginal   int64
	totalBytesCompressed int64
	totalBytesSaved      int64

	// Configuration
	config Config
}

// CompressionResult provides detailed information about compression operations.
type CompressionResult struct {
	OriginalSize     int64   `json:"original_size"`
	CompressedSize   int64   `json:"compressed_size"`
	CompressionRatio float64 `json:"compression_ratio"`
	Method           string  `json:"method"`
	Summary          string  `json:"summary,omitempty"`
	Checksum         string  `json:"checksum"`
	Timestamp        time.Time `json:"timestamp"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// CompressibleContent represents content that can be compressed or summarized.
type CompressibleContent struct {
	Content     string                 `json:"content"`
	ContentType string                 `json:"content_type"`
	Priority    CompressionPriority    `json:"priority"`
	Metadata    map[string]interface{} `json:"metadata"`
	PreserveKey bool                   `json:"preserve_key"` // Keep even if large
}

// CompressionPriority determines how aggressively content should be compressed.
type CompressionPriority string

const (
	PriorityHigh   CompressionPriority = "high"   // Most important, compress minimally
	PriorityMedium CompressionPriority = "medium" // Balanced compression
	PriorityLow    CompressionPriority = "low"    // Aggressive compression/summarization
	PriorityMinimal CompressionPriority = "minimal" // Only keep essential summary
)

// NewCompressor creates a content compressor optimized for context efficiency.
func NewCompressor(memory *FileSystemMemory, config Config) *Compressor {
	return &Compressor{
		memory:               memory,
		compressionThreshold: config.CompressionThreshold,
		maxCompressionRatio:  0.8, // Don't compress if we can't achieve 20% reduction
		config:               config,
	}
}

// CompressContent intelligently compresses content based on type and priority.
// CRITICAL: This maintains information density while reducing token count.
func (c *Compressor) CompressContent(ctx context.Context, content CompressibleContent) (string, CompressionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	logger := logging.GetLogger()
	originalSize := int64(len(content.Content))

	// Skip compression if content is below threshold
	if originalSize < c.compressionThreshold {
		return content.Content, CompressionResult{
			OriginalSize:     originalSize,
			CompressedSize:   originalSize,
			CompressionRatio: 1.0,
			Method:           "none",
			Timestamp:        time.Now(),
		}, nil
	}

	// Choose compression strategy based on content type and priority
	var result CompressionResult
	var compressed string
	var err error

	switch content.ContentType {
	case "code":
		compressed, result, err = c.compressCode(content)
	case "json":
		compressed, result, err = c.compressJSON(content)
	case "logs":
		compressed, result, err = c.compressLogs(content)
	case "text", "markdown":
		compressed, result, err = c.compressText(content)
	case "observations":
		compressed, result, err = c.compressObservations(content)
	default:
		compressed, result, err = c.compressGeneric(content)
	}

	if err != nil {
		return content.Content, CompressionResult{}, err
	}

	// Update statistics
	c.totalCompressions++
	c.totalBytesOriginal += originalSize
	c.totalBytesCompressed += result.CompressedSize
	c.totalBytesSaved += (originalSize - result.CompressedSize)

	logger.Debug(ctx, "Compressed %s content: %d -> %d bytes (%.1f%% reduction)",
		content.ContentType, originalSize, result.CompressedSize,
		(1.0-result.CompressionRatio)*100)

	return compressed, result, nil
}

// CompressAndStore combines compression with filesystem storage for large content.
func (c *Compressor) CompressAndStore(ctx context.Context, id string, content CompressibleContent) (string, error) {
	compressed, result, err := c.CompressContent(ctx, content)
	if err != nil {
		return "", err
	}

	// Store compressed content if it's still large
	if result.CompressedSize > c.compressionThreshold {
		reference, err := c.memory.StoreFile(ctx, "compressed", id, []byte(compressed), map[string]interface{}{
			"original_size":     result.OriginalSize,
			"compressed_size":   result.CompressedSize,
			"compression_ratio": result.CompressionRatio,
			"method":           result.Method,
			"content_type":     content.ContentType,
			"priority":         content.Priority,
		})

		if err != nil {
			return "", err
		}

		// Return a short reference with summary
		summary := c.createContentSummary(content.ContentType, compressed, result)
		return fmt.Sprintf("%s\n\n[Full content stored: %s]", summary, reference), nil
	}

	return compressed, nil
}

// DecompressContent reverses compression to restore original content.
func (c *Compressor) DecompressContent(ctx context.Context, compressedContent string, method string) (string, error) {
	switch method {
	case "gzip":
		return c.decompressGzip(compressedContent)
	case "summary":
		// Summaries can't be perfectly decompressed, return as-is with note
		return compressedContent + "\n[Note: This is a compressed summary, original content not recoverable]", nil
	case "structured":
		return compressedContent, nil // Already in readable format
	default:
		return compressedContent, nil
	}
}

// GetCompressionStats returns detailed compression performance metrics.
func (c *Compressor) GetCompressionStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var avgRatio float64
	if c.totalCompressions > 0 {
		avgRatio = float64(c.totalBytesCompressed) / float64(c.totalBytesOriginal)
	}

	return map[string]interface{}{
		"total_compressions":     c.totalCompressions,
		"total_bytes_original":   c.totalBytesOriginal,
		"total_bytes_compressed": c.totalBytesCompressed,
		"total_bytes_saved":      c.totalBytesSaved,
		"average_compression_ratio": avgRatio,
		"total_savings_percent":  (1.0 - avgRatio) * 100,
		"threshold_bytes":        c.compressionThreshold,
	}
}

// Specialized compression methods

func (c *Compressor) compressCode(content CompressibleContent) (string, CompressionResult, error) {
	code := content.Content
	originalSize := int64(len(code))

	var compressed string
	var method string

	switch content.Priority {
	case PriorityHigh:
		// Minimal compression - just remove excessive whitespace
		compressed = c.removeExcessiveWhitespace(code)
		method = "whitespace_minimal"

	case PriorityMedium:
		// Remove comments and extra whitespace
		compressed = c.removeCommentsAndWhitespace(code)
		method = "comments_whitespace"

	case PriorityLow, PriorityMinimal:
		// Create structural summary
		compressed = c.createCodeSummary(code)
		method = "summary"
	}

	compressedSize := int64(len(compressed))
	ratio := float64(compressedSize) / float64(originalSize)

	// Fall back to gzip if other methods don't achieve good compression
	if ratio > c.maxCompressionRatio && content.Priority != PriorityHigh {
		gzipCompressed, err := c.compressWithGzip(code)
		if err == nil && int64(len(gzipCompressed)) < compressedSize {
			compressed = gzipCompressed
			compressedSize = int64(len(compressed))
			ratio = float64(compressedSize) / float64(originalSize)
			method = "gzip"
		}
	}

	return compressed, CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           method,
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

func (c *Compressor) compressJSON(content CompressibleContent) (string, CompressionResult, error) {
	jsonContent := content.Content
	originalSize := int64(len(jsonContent))

	// Parse and reformat JSON compactly
	var data interface{}
	if err := json.Unmarshal([]byte(jsonContent), &data); err != nil {
		// If not valid JSON, treat as text
		return c.compressText(content)
	}

	var compressed []byte
	var method string
	var err error

	switch content.Priority {
	case PriorityHigh:
		// Compact JSON without indentation
		compressed, err = json.Marshal(data)
		method = "compact"

	case PriorityMedium:
		// Remove null fields and compact
		cleanData := c.removeNullFields(data)
		compressed, err = json.Marshal(cleanData)
		method = "compact_cleaned"

	case PriorityLow, PriorityMinimal:
		// Create JSON structure summary
		summary := c.createJSONSummary(data)
		compressed = []byte(summary)
		method = "summary"
	}

	if err != nil {
		return jsonContent, CompressionResult{}, err
	}

	compressedSize := int64(len(compressed))
	ratio := float64(compressedSize) / float64(originalSize)

	return string(compressed), CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           method,
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

func (c *Compressor) compressLogs(content CompressibleContent) (string, CompressionResult, error) {
	logs := content.Content
	originalSize := int64(len(logs))

	lines := strings.Split(logs, "\n")
	var compressed []string

	switch content.Priority {
	case PriorityHigh:
		// Keep all lines but remove duplicate timestamps
		compressed = c.deduplicateLogTimestamps(lines)

	case PriorityMedium:
		// Keep errors, warnings, and important info
		compressed = c.filterImportantLogs(lines)

	case PriorityLow:
		// Keep only errors and critical messages
		compressed = c.filterCriticalLogs(lines)

	case PriorityMinimal:
		// Create log summary
		summary := c.createLogSummary(lines)
		compressed = []string{summary}
	}

	result := strings.Join(compressed, "\n")
	compressedSize := int64(len(result))
	ratio := float64(compressedSize) / float64(originalSize)

	return result, CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           "log_filtering",
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

func (c *Compressor) compressText(content CompressibleContent) (string, CompressionResult, error) {
	text := content.Content
	originalSize := int64(len(text))

	var compressed string
	var method string

	switch content.Priority {
	case PriorityHigh:
		// Just normalize whitespace
		compressed = c.normalizeWhitespace(text)
		method = "whitespace_normalized"

	case PriorityMedium:
		// Remove redundant sentences and normalize
		compressed = c.removeRedundantSentences(text)
		method = "redundancy_removed"

	case PriorityLow:
		// Extract key sentences
		compressed = c.extractKeySentences(text)
		method = "key_sentences"

	case PriorityMinimal:
		// Create bullet point summary
		compressed = c.createTextSummary(text)
		method = "summary"
	}

	compressedSize := int64(len(compressed))
	ratio := float64(compressedSize) / float64(originalSize)

	return compressed, CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           method,
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

func (c *Compressor) compressObservations(content CompressibleContent) (string, CompressionResult, error) {
	observations := content.Content
	originalSize := int64(len(observations))

	// Observations are often structured, try to preserve structure while compressing
	var compressed string

	switch content.Priority {
	case PriorityHigh:
		// Keep full observations but remove redundancy
		compressed = c.removeRedundantObservations(observations)

	case PriorityMedium, PriorityLow:
		// Extract key insights and patterns
		compressed = c.extractObservationInsights(observations)

	case PriorityMinimal:
		// Create high-level summary
		compressed = c.createObservationSummary(observations)
	}

	compressedSize := int64(len(compressed))
	ratio := float64(compressedSize) / float64(originalSize)

	return compressed, CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           "observation_structured",
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

func (c *Compressor) compressGeneric(content CompressibleContent) (string, CompressionResult, error) {
	// Fall back to gzip compression for unknown content types
	compressed, err := c.compressWithGzip(content.Content)
	if err != nil {
		return content.Content, CompressionResult{}, err
	}

	originalSize := int64(len(content.Content))
	compressedSize := int64(len(compressed))
	ratio := float64(compressedSize) / float64(originalSize)

	return compressed, CompressionResult{
		OriginalSize:     originalSize,
		CompressedSize:   compressedSize,
		CompressionRatio: ratio,
		Method:           "gzip",
		Checksum:         c.calculateChecksum(content.Content),
		Timestamp:        time.Now(),
		Metadata:         content.Metadata,
	}, nil
}

// Helper methods for compression techniques

func (c *Compressor) compressWithGzip(content string) (string, error) {
	var buf bytes.Buffer
	gzipWriter := gzip.NewWriter(&buf)

	if _, err := gzipWriter.Write([]byte(content)); err != nil {
		return "", err
	}

	if err := gzipWriter.Close(); err != nil {
		return "", err
	}

	// Return base64 encoded for text compatibility
	return fmt.Sprintf("GZIP:%s", hex.EncodeToString(buf.Bytes())), nil
}

func (c *Compressor) decompressGzip(compressedContent string) (string, error) {
	if !strings.HasPrefix(compressedContent, "GZIP:") {
		return compressedContent, nil
	}

	hexData := strings.TrimPrefix(compressedContent, "GZIP:")
	compressed, err := hex.DecodeString(hexData)
	if err != nil {
		return "", err
	}

	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return "", err
	}
	defer reader.Close()

	decompressed, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}

	return string(decompressed), nil
}

func (c *Compressor) calculateChecksum(content string) string {
	hasher := sha256.New()
	hasher.Write([]byte(content))
	return hex.EncodeToString(hasher.Sum(nil))[:16]
}

func (c *Compressor) createContentSummary(contentType, content string, result CompressionResult) string {
	return fmt.Sprintf("Content Summary (%s, %d->%d bytes, %.1f%% reduction):\n%s",
		contentType, result.OriginalSize, result.CompressedSize,
		(1.0-result.CompressionRatio)*100, content[:min(200, len(content))])
}

// Simple implementations of compression techniques
// These are basic implementations - in production you might want more sophisticated NLP

func (c *Compressor) removeExcessiveWhitespace(text string) string {
	// Replace multiple whitespace with single space
	lines := strings.Split(text, "\n")
	var result []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}

	return strings.Join(result, "\n")
}

func (c *Compressor) removeCommentsAndWhitespace(code string) string {
	lines := strings.Split(code, "\n")
	var result []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Remove lines that are just comments (basic detection)
		if trimmed != "" && !strings.HasPrefix(trimmed, "//") && !strings.HasPrefix(trimmed, "#") && !strings.HasPrefix(trimmed, "/*") {
			result = append(result, trimmed)
		}
	}

	return strings.Join(result, "\n")
}

func (c *Compressor) createCodeSummary(code string) string {
	lines := strings.Split(code, "\n")
	var functions, structs, imports []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") {
			functions = append(functions, trimmed)
		} else if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "struct") {
			structs = append(structs, trimmed)
		} else if strings.HasPrefix(trimmed, "import") || strings.HasPrefix(trimmed, "from ") {
			imports = append(imports, trimmed)
		}
	}

	summary := "Code Summary:\n"
	if len(imports) > 0 {
		summary += fmt.Sprintf("Imports: %d\n", len(imports))
	}
	if len(structs) > 0 {
		summary += fmt.Sprintf("Structs/Classes: %d\n", len(structs))
	}
	if len(functions) > 0 {
		summary += fmt.Sprintf("Functions: %d\n", len(functions))
		summary += "Key functions:\n"
		for i, fn := range functions {
			if i < 5 { // Show first 5 functions
				summary += fmt.Sprintf("- %s\n", fn)
			} else {
				summary += fmt.Sprintf("- ... and %d more\n", len(functions)-5)
				break
			}
		}
	}

	return summary
}

func (c *Compressor) normalizeWhitespace(text string) string {
	// Replace multiple spaces with single space
	return strings.Join(strings.Fields(text), " ")
}

func (c *Compressor) removeRedundantSentences(text string) string {
	sentences := strings.Split(text, ".")
	seen := make(map[string]bool)
	var result []string

	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		if trimmed != "" && !seen[trimmed] {
			seen[trimmed] = true
			result = append(result, trimmed)
		}
	}

	return strings.Join(result, ". ")
}

func (c *Compressor) extractKeySentences(text string) string {
	// Simple key sentence extraction based on keywords
	sentences := strings.Split(text, ".")
	keywords := []string{"important", "critical", "error", "warning", "success", "failed", "completed"}
	var result []string

	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		lower := strings.ToLower(trimmed)

		for _, keyword := range keywords {
			if strings.Contains(lower, keyword) {
				result = append(result, trimmed)
				break
			}
		}
	}

	if len(result) == 0 {
		// If no key sentences found, take first and last sentences
		if len(sentences) > 0 {
			result = append(result, strings.TrimSpace(sentences[0]))
		}
		if len(sentences) > 1 {
			result = append(result, strings.TrimSpace(sentences[len(sentences)-1]))
		}
	}

	return strings.Join(result, ". ")
}

func (c *Compressor) createTextSummary(text string) string {
	// Create a very basic summary
	wordCount := len(strings.Fields(text))
	sentences := strings.Split(text, ".")
	sentenceCount := len(sentences)

	summary := fmt.Sprintf("Text Summary: %d words, %d sentences\n", wordCount, sentenceCount)

	if len(sentences) > 0 {
		summary += fmt.Sprintf("First: %s\n", strings.TrimSpace(sentences[0]))
	}
	if len(sentences) > 1 {
		summary += fmt.Sprintf("Last: %s", strings.TrimSpace(sentences[len(sentences)-1]))
	}

	return summary
}

// Additional helper methods would be implemented similarly...
func (c *Compressor) removeNullFields(data interface{}) interface{} {
	// Simplified implementation - remove null/empty fields from JSON
	return data // Placeholder - would implement actual null field removal
}

func (c *Compressor) createJSONSummary(data interface{}) string {
	return fmt.Sprintf("JSON Summary: %d bytes of structured data", len(fmt.Sprintf("%v", data)))
}

func (c *Compressor) deduplicateLogTimestamps(lines []string) []string {
	return lines // Placeholder implementation
}

func (c *Compressor) filterImportantLogs(lines []string) []string {
	var filtered []string
	for _, line := range lines {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "error") || strings.Contains(lower, "warn") || strings.Contains(lower, "info") {
			filtered = append(filtered, line)
		}
	}
	return filtered
}

func (c *Compressor) filterCriticalLogs(lines []string) []string {
	var filtered []string
	for _, line := range lines {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "error") || strings.Contains(lower, "critical") || strings.Contains(lower, "fatal") {
			filtered = append(filtered, line)
		}
	}
	return filtered
}

func (c *Compressor) createLogSummary(lines []string) string {
	errorCount := 0
	warnCount := 0
	for _, line := range lines {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "error") {
			errorCount++
		} else if strings.Contains(lower, "warn") {
			warnCount++
		}
	}
	return fmt.Sprintf("Log Summary: %d total lines, %d errors, %d warnings", len(lines), errorCount, warnCount)
}

func (c *Compressor) removeRedundantObservations(observations string) string {
	return observations // Placeholder implementation
}

func (c *Compressor) extractObservationInsights(observations string) string {
	return "Key insights extracted from observations" // Placeholder
}

func (c *Compressor) createObservationSummary(observations string) string {
	return "Observation summary" // Placeholder
}
