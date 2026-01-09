// Package rlm provides context structure pre-analysis for RLM.
// The analyzer detects the structure of input context (JSON, markdown, code, etc.)
// and provides hints for optimal processing strategies.
package rlm

import (
	"encoding/json"
	"regexp"
	"strings"
	"unicode"
)

// ContextType represents the detected type of content.
type ContextType string

const (
	// TypeUnknown is used when the content type cannot be determined.
	TypeUnknown ContextType = "unknown"

	// TypeJSON indicates JSON-structured content.
	TypeJSON ContextType = "json"

	// TypeMarkdown indicates markdown-formatted content.
	TypeMarkdown ContextType = "markdown"

	// TypeCode indicates source code content.
	TypeCode ContextType = "code"

	// TypePlainText indicates plain text content.
	TypePlainText ContextType = "plaintext"

	// TypeCSV indicates CSV/tabular data.
	TypeCSV ContextType = "csv"

	// TypeXML indicates XML-structured content.
	TypeXML ContextType = "xml"

	// TypeLog indicates log file content.
	TypeLog ContextType = "log"

	// TypeMixed indicates mixed content with multiple types.
	TypeMixed ContextType = "mixed"
)

// ChunkStrategy represents the recommended chunking approach.
type ChunkStrategy string

const (
	// StrategyNone means no chunking needed - context is small enough.
	StrategyNone ChunkStrategy = "none"

	// StrategyFixed means split into fixed-size chunks.
	StrategyFixed ChunkStrategy = "fixed"

	// StrategyDelimiter means split on natural delimiters.
	StrategyDelimiter ChunkStrategy = "delimiter"

	// StrategyHierarchical means split based on document structure.
	StrategyHierarchical ChunkStrategy = "hierarchical"

	// StrategySemantic means split based on semantic units.
	StrategySemantic ChunkStrategy = "semantic"
)

// ContextAnalysis contains the analysis results for a context payload.
type ContextAnalysis struct {
	// Type is the detected primary content type.
	Type ContextType

	// SecondaryTypes are additional types detected in mixed content.
	SecondaryTypes []ContextType

	// Size is the total size in bytes.
	Size int

	// EstimatedTokens is the estimated token count (rough approximation).
	EstimatedTokens int

	// RecommendedStrategy is the suggested chunking strategy.
	RecommendedStrategy ChunkStrategy

	// RecommendedChunkSize is the suggested chunk size in bytes.
	RecommendedChunkSize int

	// Delimiters are natural delimiters found in the content.
	Delimiters []string

	// Structure provides hints about the content structure.
	Structure StructureHints

	// LLMHint is a string hint to prepend to LLM prompts about the context.
	LLMHint string
}

// StructureHints provides detailed structure information.
type StructureHints struct {
	// HasHeaders indicates if the content has section headers.
	HasHeaders bool

	// HeaderCount is the number of headers detected.
	HeaderCount int

	// HasCodeBlocks indicates if code blocks are present.
	HasCodeBlocks bool

	// CodeBlockCount is the number of code blocks detected.
	CodeBlockCount int

	// HasLists indicates if lists are present.
	HasLists bool

	// ListCount is the number of lists detected.
	ListCount int

	// HasTables indicates if tables are present.
	HasTables bool

	// NestingDepth is the maximum nesting depth for structured content.
	NestingDepth int

	// LineCount is the total number of lines.
	LineCount int

	// AvgLineLength is the average line length.
	AvgLineLength int
}

// Regular expressions for content detection.
var (
	// Markdown patterns.
	markdownHeaderRe    = regexp.MustCompile(`(?m)^#{1,6}\s+.+$`)
	markdownCodeBlockRe = regexp.MustCompile("(?s)```[a-zA-Z]*\\n.*?```")
	markdownListRe      = regexp.MustCompile(`(?m)^[\s]*[-*+]\s+.+$|^[\s]*\d+\.\s+.+$`)
	markdownTableRe     = regexp.MustCompile(`(?m)^\|.+\|$`)
	markdownLinkRe      = regexp.MustCompile(`\[.+?\]\(.+?\)`)

	// Code patterns.
	codePatternRe = regexp.MustCompile(`(?m)^\s*(func|def|class|function|import|package|const|var|let|public|private|#include)\s`)

	// Log patterns.
	logPatternRe = regexp.MustCompile(`(?m)^\[?\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}`)

	// XML patterns.
	xmlPatternRe = regexp.MustCompile(`(?s)^\s*<\?xml|^\s*<[a-zA-Z][a-zA-Z0-9]*(\s|>)`)
)

// AnalyzerConfig holds analyzer configuration options.
type AnalyzerConfig struct {
	// SmallContextThreshold is the size below which no chunking is needed.
	SmallContextThreshold int

	// TokenEstimateRatio is the character-to-token ratio estimate.
	TokenEstimateRatio float64
}

// DefaultAnalyzerConfig returns the default analyzer configuration.
func DefaultAnalyzerConfig() AnalyzerConfig {
	return AnalyzerConfig{
		SmallContextThreshold: 50000,
		TokenEstimateRatio:    4.0,
	}
}

// AnalyzeContext examines the context payload and returns analysis results.
func AnalyzeContext(payload any) *ContextAnalysis {
	return AnalyzeContextWithConfig(payload, DefaultAnalyzerConfig())
}

// AnalyzeContextWithConfig performs analysis with custom configuration.
func AnalyzeContextWithConfig(payload any, cfg AnalyzerConfig) *ContextAnalysis {
	var content string
	switch v := payload.(type) {
	case string:
		content = v
	case []byte:
		content = string(v)
	default:
		// Try JSON marshaling for complex types
		if bytes, err := json.Marshal(v); err == nil {
			content = string(bytes)
		}
	}

	analysis := &ContextAnalysis{
		Size:            len(content),
		EstimatedTokens: estimateTokens(content),
	}

	// Analyze content structure
	analysis.Structure = analyzeStructure(content)

	// Detect content type
	analysis.Type, analysis.SecondaryTypes = detectContentType(content)

	// Determine chunking strategy
	analysis.RecommendedStrategy, analysis.RecommendedChunkSize = determineChunkStrategy(analysis, cfg)

	// Find natural delimiters
	analysis.Delimiters = findDelimiters(content, analysis.Type)

	// Generate LLM hint
	analysis.LLMHint = generateLLMHint(analysis)

	return analysis
}

// estimateTokens provides a rough token count estimate.
// Uses the approximation that 1 token ~= 4 characters for English text.
func estimateTokens(content string) int {
	// More accurate estimation based on character type
	wordCount := 0
	inWord := false
	for _, r := range content {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if inWord {
				wordCount++
				inWord = false
			}
		} else {
			inWord = true
		}
	}
	if inWord {
		wordCount++
	}

	// Tokens are roughly 0.75 * words + 0.25 * special characters
	specialChars := 0
	for _, r := range content {
		if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			specialChars++
		}
	}

	return int(float64(wordCount)*1.3 + float64(specialChars)*0.5)
}

// analyzeStructure examines the detailed structure of content.
func analyzeStructure(content string) StructureHints {
	hints := StructureHints{}

	lines := strings.Split(content, "\n")
	hints.LineCount = len(lines)

	totalLen := 0
	for _, line := range lines {
		totalLen += len(line)
	}
	if hints.LineCount > 0 {
		hints.AvgLineLength = totalLen / hints.LineCount
	}

	// Check for markdown headers
	headers := markdownHeaderRe.FindAllString(content, -1)
	hints.HasHeaders = len(headers) > 0
	hints.HeaderCount = len(headers)

	// Check for code blocks
	codeBlocks := markdownCodeBlockRe.FindAllString(content, -1)
	hints.HasCodeBlocks = len(codeBlocks) > 0
	hints.CodeBlockCount = len(codeBlocks)

	// Check for lists
	lists := markdownListRe.FindAllString(content, -1)
	hints.HasLists = len(lists) > 0
	hints.ListCount = len(lists)

	// Check for tables
	tables := markdownTableRe.FindAllString(content, -1)
	hints.HasTables = len(tables) > 0

	// Calculate nesting depth for JSON/structured content
	hints.NestingDepth = calculateNestingDepth(content)

	return hints
}

// calculateNestingDepth determines the maximum nesting level in structured content.
func calculateNestingDepth(content string) int {
	maxDepth := 0
	currentDepth := 0

	for _, ch := range content {
		switch ch {
		case '{', '[':
			currentDepth++
			if currentDepth > maxDepth {
				maxDepth = currentDepth
			}
		case '}', ']':
			currentDepth--
			if currentDepth < 0 {
				currentDepth = 0
			}
		}
	}

	return maxDepth
}

// detectContentType determines the primary and secondary content types.
func detectContentType(content string) (ContextType, []ContextType) {
	content = strings.TrimSpace(content)
	if len(content) == 0 {
		return TypePlainText, nil
	}

	var types []ContextType

	// Check for JSON
	if isJSON(content) {
		types = append(types, TypeJSON)
	}

	// Check for XML
	if xmlPatternRe.MatchString(content) {
		types = append(types, TypeXML)
	}

	// Check for markdown indicators
	markdownScore := 0
	if markdownHeaderRe.MatchString(content) {
		markdownScore += 2
	}
	if markdownCodeBlockRe.MatchString(content) {
		markdownScore += 2
	}
	if markdownListRe.MatchString(content) {
		markdownScore++
	}
	if markdownLinkRe.MatchString(content) {
		markdownScore++
	}
	if markdownScore >= 2 {
		types = append(types, TypeMarkdown)
	}

	// Check for code patterns
	if codePatternRe.MatchString(content) {
		types = append(types, TypeCode)
	}

	// Check for log patterns
	if logPatternRe.MatchString(content) {
		types = append(types, TypeLog)
	}

	// Check for CSV
	if isCSV(content) {
		types = append(types, TypeCSV)
	}

	// Determine primary type
	if len(types) == 0 {
		return TypePlainText, nil
	}

	if len(types) == 1 {
		return types[0], nil
	}

	// Multiple types detected - return as mixed with first as primary
	return TypeMixed, types
}

// isJSON checks if content is valid JSON.
func isJSON(content string) bool {
	content = strings.TrimSpace(content)
	if len(content) == 0 {
		return false
	}

	// Quick check for JSON start characters
	if content[0] != '{' && content[0] != '[' {
		return false
	}

	var js any
	return json.Unmarshal([]byte(content), &js) == nil
}

// isCSV checks if content appears to be CSV formatted.
func isCSV(content string) bool {
	lines := strings.Split(content, "\n")
	if len(lines) < 2 {
		return false
	}

	// Check first few lines for consistent delimiter usage
	delimiterCounts := make(map[int]int)
	for i, line := range lines {
		if i >= 5 || line == "" {
			break
		}
		commaCount := strings.Count(line, ",")
		tabCount := strings.Count(line, "\t")
		if commaCount > 0 {
			delimiterCounts[commaCount]++
		}
		if tabCount > 0 {
			delimiterCounts[tabCount]++
		}
	}

	// Check if there's a consistent delimiter count
	for _, count := range delimiterCounts {
		if count >= 3 {
			return true
		}
	}
	return false
}

// determineChunkStrategy determines the optimal chunking approach.
func determineChunkStrategy(analysis *ContextAnalysis, cfg AnalyzerConfig) (ChunkStrategy, int) {
	// Small contexts don't need chunking
	if analysis.Size < cfg.SmallContextThreshold {
		return StrategyNone, 0
	}

	// Medium contexts might need simple chunking
	if analysis.Size < 200000 { // ~200KB
		switch analysis.Type {
		case TypeJSON:
			return StrategyHierarchical, 100000
		case TypeMarkdown:
			if analysis.Structure.HasHeaders {
				return StrategyDelimiter, 50000
			}
			return StrategyFixed, 50000
		case TypeCode:
			return StrategySemantic, 30000
		case TypeLog:
			return StrategyDelimiter, 50000
		case TypeCSV:
			return StrategyDelimiter, 100000
		default:
			return StrategyFixed, 50000
		}
	}

	// Large contexts need chunking based on type
	switch analysis.Type {
	case TypeJSON:
		return StrategyHierarchical, 200000
	case TypeMarkdown:
		return StrategyDelimiter, 100000
	case TypeCode:
		return StrategySemantic, 50000
	case TypeLog:
		return StrategyDelimiter, 100000
	case TypeCSV:
		return StrategyDelimiter, 200000
	default:
		return StrategyFixed, 200000
	}
}

// findDelimiters identifies natural splitting points in content.
func findDelimiters(content string, contentType ContextType) []string {
	var delimiters []string

	switch contentType {
	case TypeJSON:
		delimiters = []string{`"items":`, `"data":`, `"results":`}
	case TypeMarkdown:
		delimiters = []string{"# ", "## ", "---", "\n\n"}
	case TypeCode:
		delimiters = []string{"\nfunc ", "\ndef ", "\nclass ", "\n// ---"}
	case TypeLog:
		delimiters = []string{"\n[", "\n20"} // Common log line starts
	case TypeCSV:
		delimiters = []string{"\n"}
	case TypeXML:
		delimiters = []string{"</", "><"}
	default:
		delimiters = []string{"\n\n", ". ", "---"}
	}

	return delimiters
}

// generateLLMHint creates a context hint for the LLM.
func generateLLMHint(analysis *ContextAnalysis) string {
	var hints []string

	// Type hint
	switch analysis.Type {
	case TypeJSON:
		hints = append(hints, "The context is JSON-formatted data")
		if analysis.Structure.NestingDepth > 3 {
			hints = append(hints, "with deep nesting")
		}
	case TypeMarkdown:
		hints = append(hints, "The context is markdown-formatted")
		if analysis.Structure.HasHeaders {
			hints = append(hints, "with section headers")
		}
		if analysis.Structure.HasCodeBlocks {
			hints = append(hints, "containing code blocks")
		}
	case TypeCode:
		hints = append(hints, "The context contains source code")
	case TypeLog:
		hints = append(hints, "The context contains log entries")
	case TypeCSV:
		hints = append(hints, "The context is tabular/CSV data")
	case TypeXML:
		hints = append(hints, "The context is XML-formatted data")
	case TypeMixed:
		hints = append(hints, "The context contains mixed content types")
	}

	// Size hint
	if analysis.EstimatedTokens > 100000 {
		hints = append(hints, "- This is a VERY LARGE context; use QueryBatched for parallel processing")
	} else if analysis.EstimatedTokens > 50000 {
		hints = append(hints, "- This is a large context; consider chunking for efficiency")
	}

	// Structure hints
	if analysis.Structure.LineCount > 10000 {
		hints = append(hints, "- The content has many lines; use targeted searches rather than scanning")
	}

	if len(hints) == 0 {
		return ""
	}

	return "CONTEXT ANALYSIS:\n" + strings.Join(hints, "\n")
}

// IsLargeContext returns true if the context is considered large and would benefit from chunking.
func (a *ContextAnalysis) IsLargeContext() bool {
	return a.RecommendedStrategy != StrategyNone
}

// ShouldUseBatching returns true if the context is large enough to warrant batched queries.
func (a *ContextAnalysis) ShouldUseBatching() bool {
	return a.EstimatedTokens > 50000
}
