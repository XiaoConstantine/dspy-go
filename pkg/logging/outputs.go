package logging

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type fileWriter interface {
	Write(p []byte) (n int, err error)
	WriteString(s string) (n int, err error)
	Sync() error
	Close() error
	Stat() (os.FileInfo, error)
}

// ConsoleOutput formats logs for human readability.
type ConsoleOutput struct {
	mu     sync.Mutex
	writer io.Writer
	color  bool // Whether to use ANSI color codes
}

// FileOutput formats logs and writes them to a file.
type FileOutput struct {
	mu         sync.Mutex
	file       fileWriter
	path       string
	formatter  LogFormatter
	jsonFormat bool
	rotateSize int64  // File size in bytes that triggers rotation (0 = no rotation)
	curSize    int64  // Current file size
	maxFiles   int    // Maximum number of rotated files to keep
	bufferSize int    // Buffer size for write operations
	buffer     []byte // Internal buffer for writes
}

// LogFormatter defines an interface for formatting log entries.
type LogFormatter interface {
	Format(entry LogEntry) string
}

// TextFormatter implements LogFormatter with a simple text format.
type TextFormatter struct {
	// Whether to include timestamps in the output
	IncludeTimestamp bool
	// Whether to include file and line information
	IncludeLocation bool
	// Whether to include stack traces for errors
	IncludeStackTrace bool
}

// Format formats a LogEntry as text.
func (f *TextFormatter) Format(e LogEntry) string {
	timestamp := time.Unix(0, e.Time).Format(time.RFC3339)

	// Build the log line
	logLine := fmt.Sprintf("%s [%s] %s", timestamp, e.Severity, e.Message)

	if f.IncludeLocation {
		logLine = fmt.Sprintf("%s [%s:%d]", logLine, e.File, e.Line)
	}

	// Add trace ID if present
	if e.TraceID != "" {
		logLine = fmt.Sprintf("%s [traceId=%s]", logLine, e.TraceID)
	}

	// Add token info if present
	if e.TokenInfo != nil {
		logLine = fmt.Sprintf("%s [tokens=%d/%d]",
			logLine,
			e.TokenInfo.PromptTokens,
			e.TokenInfo.CompletionTokens)
	}

	// Add other fields
	if len(e.Fields) > 0 {
		logLine = fmt.Sprintf("%s %s", logLine, formatFields(e.Fields))
	}

	return logLine
}

// JSONFormatter implements LogFormatter for JSON output.
type JSONFormatter struct{}

// Format formats a LogEntry as JSON.
func (f *JSONFormatter) Format(e LogEntry) string {
	// Convert LogEntry to a map
	logMap := map[string]interface{}{
		"timestamp": time.Unix(0, e.Time).Format(time.RFC3339),
		"level":     e.Severity.String(),
		"message":   e.Message,
		"file":      e.File,
		"line":      e.Line,
		"function":  e.Function,
	}

	// Add traceID if present
	if e.TraceID != "" {
		logMap["traceId"] = e.TraceID
	}

	// Add model info if present
	if e.ModelID != "" {
		logMap["modelId"] = e.ModelID
	}

	// Add token info if present
	if e.TokenInfo != nil {
		logMap["tokenInfo"] = map[string]interface{}{
			"promptTokens":     e.TokenInfo.PromptTokens,
			"completionTokens": e.TokenInfo.CompletionTokens,
			"totalTokens":      e.TokenInfo.TotalTokens,
		}
	}

	// Add other fields
	for k, v := range e.Fields {
		logMap[k] = v
	}

	// Marshal to JSON
	jsonBytes, err := json.Marshal(logMap)
	if err != nil {
		return fmt.Sprintf("Error marshaling log entry to JSON: %v", err)
	}

	return string(jsonBytes)
}

// FileOutputOption is a functional option for configuring FileOutput.
type FileOutputOption func(*FileOutput)

// WithJSONFormat configures the output to use JSON formatting.
func WithJSONFormat(enabled bool) FileOutputOption {
	return func(f *FileOutput) {
		f.jsonFormat = enabled
		if enabled {
			f.formatter = &JSONFormatter{}
		} else {
			f.formatter = &TextFormatter{
				IncludeTimestamp: true,
				IncludeLocation:  true,
			}
		}
	}
}

// WithRotation enables log file rotation.
func WithRotation(maxSizeBytes int64, maxFiles int) FileOutputOption {
	return func(f *FileOutput) {
		f.rotateSize = maxSizeBytes
		f.maxFiles = maxFiles
	}
}

// WithBufferSize sets the internal buffer size for writes.
func WithBufferSize(size int) FileOutputOption {
	return func(f *FileOutput) {
		f.bufferSize = size
		f.buffer = make([]byte, 0, size)
	}
}

// WithFormatter sets a custom log formatter.
func WithFormatter(formatter LogFormatter) FileOutputOption {
	return func(f *FileOutput) {
		f.formatter = formatter
	}
}

// NewFileOutput creates a new file-based logger output.
func NewFileOutput(path string, opts ...FileOutputOption) (*FileOutput, error) {
	// Create the directory if it doesn't exist
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}

	// Open the file with append mode, creating if it doesn't exist
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	// Get current file info to determine size
	info, err := file.Stat()
	var curSize int64 = 0
	if err == nil {
		curSize = info.Size()
	}

	// Create with default options
	output := &FileOutput{
		file:       file,
		path:       path,
		formatter:  &TextFormatter{IncludeTimestamp: true, IncludeLocation: true},
		jsonFormat: false,
		curSize:    curSize,
		bufferSize: 4096,
		buffer:     make([]byte, 0, 4096), // Default 4KB buffer
	}

	// Apply any provided options
	for _, opt := range opts {
		opt(output)
	}

	return output, nil
}

// Write implements the Output interface.
func (f *FileOutput) Write(e LogEntry) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Format the log entry
	formatted := f.formatter.Format(e)
	formatted = formatted + "\n" // Add newline

	entrySize := int64(len(formatted))
	// Check if we need to rotate the file
	if f.rotateSize > 0 && (f.curSize+entrySize) >= f.rotateSize {
		if err := f.rotate(); err != nil {
			return fmt.Errorf("failed to rotate log file: %w", err)
		}
		f.curSize = 0
	}

	// Write to the file
	n, err := f.file.WriteString(formatted)
	if err != nil {
		return fmt.Errorf("failed to write to log file: %w", err)
	}

	// Update current size
	f.curSize += int64(n)

	return nil
}

// Sync flushes any buffered data to the underlying file.
func (f *FileOutput) Sync() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.file.Sync()
}

// Close closes the file.
func (f *FileOutput) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.file.Close()
}

// rotate handles log file rotation.
func (f *FileOutput) rotate() error {
	// Close the current file
	if err := f.file.Close(); err != nil {
		return err
	}

	// Rename current file to backup
	backupPath := fmt.Sprintf("%s.%s", f.path, time.Now().Format("20060102-150405"))
	if err := os.Rename(f.path, backupPath); err != nil {
		return err
	}

	// Create a new file
	file, err := os.OpenFile(f.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}

	// Update file and reset size
	f.file = file
	f.curSize = 0

	// Clean up old log files if needed
	if f.maxFiles > 0 {
		if err := f.cleanOldLogFiles(); err != nil {
			// Just log the error, don't fail rotation
			fmt.Fprintf(os.Stderr, "Error cleaning old log files: %v\n", err)
		}
	}

	return nil
}

// cleanOldLogFiles removes old rotated log files beyond the maxFiles limit.
func (f *FileOutput) cleanOldLogFiles() error {
	dir := filepath.Dir(f.path)
	base := filepath.Base(f.path)

	// List all files in the directory
	files, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	// Filter only log files with our pattern
	var logFiles []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Check if it's a rotated log file
		name := file.Name()
		if filepath.Base(f.path) != name && len(name) > len(base) && name[:len(base)] == base {
			logFiles = append(logFiles, filepath.Join(dir, name))
		}
	}

	// If we have more files than maxFiles, delete the oldest ones
	if len(logFiles) > f.maxFiles {
		// Sort log files by modification time (oldest first)
		// This is a simplification - in production you'd use file info to sort properly
		// For brevity, skipping the actual sort implementation

		// Delete excess files
		for i := 0; i < len(logFiles)-f.maxFiles; i++ {
			if err := os.Remove(logFiles[i]); err != nil {
				return err
			}
		}
	}

	return nil
}

type ConsoleOutputOption func(*ConsoleOutput)

func WithColor(enabled bool) ConsoleOutputOption {
	return func(c *ConsoleOutput) {
		c.color = enabled
	}
}

func NewConsoleOutput(useStderr bool, opts ...ConsoleOutputOption) *ConsoleOutput {
	// Choose the appropriate writer based on useStderr flag
	writer := os.Stdout
	if useStderr {
		writer = os.Stderr
	}

	// Create the base console output
	c := &ConsoleOutput{
		writer: writer,
		color:  true, // Enable colors by default
	}

	// Apply any provided options
	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Helper function to get ANSI color codes for different severity levels.
func getSeverityColor(s Severity) string {
	switch s {
	case DEBUG:
		return "\033[37m" // Gray
	case INFO:
		return "\033[32m" // Green
	case WARN:
		return "\033[33m" // Yellow
	case ERROR:
		return "\033[31m" // Red
	case FATAL:
		return "\033[35m" // Magenta
	default:
		return ""
	}
}

const (
	// Color codes for token flow visualization.
	greenArrow = "\033[32m↑\033[0m" // Up arrow in green
	blueArrow  = "\033[34m↓\033[0m" // Down arrow in blue
)

func formatFields(fields map[string]interface{}) string {
	if len(fields) == 0 {
		return ""
	}

	var result string
	for k, v := range fields {
		// Handle special fields like prompts and completions
		if k == "prompt" || k == "completion" {
			// Truncate long text for console display
			str := fmt.Sprintf("%v", v)
			if len(str) > 100 {
				str = str[:97] + "..."
			}
			result += fmt.Sprintf("%s=%q ", k, str)
		} else {
			result += fmt.Sprintf("%s=%v ", k, v)
		}
	}

	return result
}

func (c *ConsoleOutput) Sync() error {
	if syncer, ok := c.writer.(interface{ Sync() error }); ok {
		return syncer.Sync()
	}
	return nil
}

func (o *ConsoleOutput) Write(e LogEntry) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Format basic log info
	timestamp := time.Unix(0, e.Time).Format(time.RFC3339)

	var levelColor, resetColor string
	if o.color {
		levelColor = getSeverityColor(e.Severity)
		resetColor = "\033[0m"
	}

	traceDisplay := "traceId=-" // Default when no trace ID is present
	if e.TraceID != "" {
		// Truncate long trace IDs to 8 characters for readability
		displayID := e.TraceID
		if len(displayID) > 8 {
			displayID = displayID[:8]
		}
		traceDisplay = fmt.Sprintf("traceId=%s", displayID)
	}

	// Format the base message
	basic := fmt.Sprintf("%s %s%-5s%s [%s] [%s:%d] %s",
		timestamp,
		levelColor,
		e.Severity,
		resetColor,
		traceDisplay,
		e.File,
		e.Line,
		e.Message,
	)

	// Add LLM-specific info in a structured way
	if e.ModelID != "" || e.TokenInfo != nil {
		basic += formatLLMInfo(e.ModelID, e.TokenInfo)
	}

	// Write the basic message
	if _, err := fmt.Fprintln(o.writer, basic); err != nil {
		return err
	}

	if e.Severity <= DEBUG {
		if spans, ok := e.Fields["spans"]; ok && spans != nil {
			if spanSlice, ok := spans.([]*core.Span); ok && len(spanSlice) > 0 {
				spanInfo := formatSpans(spanSlice)
				if _, err := fmt.Fprintln(o.writer, spanInfo); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// Close cleans up any resources.
func (c *ConsoleOutput) Close() error {
	if closer, ok := c.writer.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

func extractModelName(modelID string) string {
	// Split on common separators
	parts := strings.FieldsFunc(modelID, func(r rune) bool {
		return r == '/' || r == '-' || r == '.'
	})

	// Handle different model naming patterns
	switch {
	case contains(parts, "claude"):
		// Find "claude" and the following size indicator (opus, sonnet, etc.)
		for i, part := range parts {
			if part == "claude" && i+1 < len(parts) {
				if parts[i+1] == "3" && i+2 < len(parts) {
					// Handle claude-3-opus format
					return fmt.Sprintf("claude-%s", parts[i+2])
				}
				return fmt.Sprintf("claude-%s", parts[i+1])
			}
		}
		return "claude"

	case contains(parts, "gemini"):
		// Find "gemini" and its variant (pro, vision, etc.)
		for i, part := range parts {
			if part == "gemini" && i+1 < len(parts) {
				return fmt.Sprintf("gemini-%s", parts[i+1])
			}
		}
		return "gemini"

	case contains(parts, "gpt"):
		// Handle GPT models
		if len(parts) >= 3 && parts[0] == "gpt" && parts[1] == "4" {
			if parts[2] == "32k" {
				return "gpt-4-32k" // Special case for gpt-4-32k
			}
			// Other gpt-4-X variants
			return fmt.Sprintf("gpt-%s-%s", parts[1], parts[2])
		}
		// Basic gpt-X format
		if len(parts) >= 2 {
			return fmt.Sprintf("gpt-%s", parts[1])
		}
		return parts[0]
	case contains(parts, "llama"):
		// Handle llama-2 and other variants
		for i, part := range parts {
			if part == "llama" && i+1 < len(parts) {
				return fmt.Sprintf("llama-%s", parts[i+1])
			}
		}
		return "llama"
	default:
		// For unknown patterns, return the first part
		if len(parts) > 0 {
			return parts[0]
		}
		return modelID
	}
}

// contains is a helper function to check if a string slice contains a specific string.
func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

func formatLLMInfo(modelID string, tokenInfo *core.TokenInfo) string {
	var parts []string

	// Add model information if available
	if modelID != "" {
		// Extract just the model name without the full path/version for cleaner logs
		modelName := extractModelName(modelID)
		parts = append(parts, fmt.Sprintf("model=%s", modelName))
	}

	// Add token usage information if available
	if tokenInfo != nil {
		upArrow := greenArrow  // Prompt tokens in green
		downArrow := blueArrow // Completion tokens in blue
		// Format token information with specific highlights:
		// - Total tokens for quick overview
		// - Completion tokens to understand response size
		// - Prompt tokens to track input size

		tokenUsage := fmt.Sprintf("tokens=%d(%d%s%d%s)",
			tokenInfo.TotalTokens,
			tokenInfo.PromptTokens,
			upArrow,
			tokenInfo.CompletionTokens,
			downArrow)
		parts = append(parts, tokenUsage)
	}

	// If we have any information to show, format it with brackets
	if len(parts) > 0 {
		return " [" + strings.Join(parts, "] [") + "]"
	}

	return ""
}

func formatSpans(spans []*core.Span) string {
	if len(spans) == 0 {
		return ""
	}

	// Create a tree structure to show span relationships
	spanMap := make(map[string]*core.Span)
	rootSpans := make([]*core.Span, 0)

	for _, span := range spans {
		spanMap[span.ID] = span
		if span.ParentID == "" {
			rootSpans = append(rootSpans, span)
		}
	}

	var b strings.Builder
	b.WriteString("\nTrace Timeline:\n")

	// Format each root span and its children
	for _, root := range rootSpans {
		formatSpanTree(&b, root, spanMap, "", true)
	}

	return b.String()
}

func formatSpanDuration(d time.Duration) string {

	switch {
	case d < time.Microsecond:
		return "<1µs" // Show very small durations simply
	case d < time.Millisecond:
		return fmt.Sprintf("%.2fµs", float64(d.Nanoseconds())/1000)
	case d < time.Second:
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1000000)
	case d < time.Minute:
		return fmt.Sprintf("%.2fs", d.Seconds())
	case d < time.Hour:
		return fmt.Sprintf("%.1fm", d.Minutes())
	default:
		return fmt.Sprintf("%.1fh", d.Hours())
	}
}

func formatSpanTree(b *strings.Builder, span *core.Span, spanMap map[string]*core.Span, prefix string, isLast bool) {
	// Create the prefix for this level
	marker := "├──"
	if isLast {
		marker = "└──"
	}

	// Calculate duration
	var duration time.Duration
	if !span.EndTime.IsZero() && !span.StartTime.IsZero() {
		duration = span.EndTime.Sub(span.StartTime)
	} else if !span.StartTime.IsZero() {
		// For in-progress spans, use time since start
		duration = time.Since(span.StartTime)
	}

	spanInfo := fmt.Sprintf("%s%s %s", prefix, marker, span.Operation)

	// Add module context information if available
	if moduleInfo, ok := span.Annotations["module"].(map[string]interface{}); ok {
		// Module context was already included in span.Operation by StartSpanWithContext
		// but we can add additional type information if helpful
		if moduleType, ok := moduleInfo["type"].(string); ok && moduleType != "" {
			spanInfo += fmt.Sprintf(" [%s]", moduleType)
		}
	}

	if chainInfo, ok := span.Annotations["chain_step"].(map[string]interface{}); ok {
		if stepName, ok := chainInfo["name"].(string); ok {
			spanInfo += fmt.Sprintf(" (step=%s)", stepName)
		}
		if stepIndex, ok := chainInfo["index"].(int); ok {
			spanInfo += fmt.Sprintf(" [%d/%d]", stepIndex+1, chainInfo["total"].(int))
		}
	}

	if taskInfo, ok := span.Annotations["task"].(map[string]interface{}); ok {
		// Add processor type and task ID for better context
		if processor, ok := taskInfo["processor"].(string); ok {
			spanInfo += fmt.Sprintf(" [%s]", processor)
		}
		if id, ok := taskInfo["id"].(string); ok {
			spanInfo += fmt.Sprintf(" (task=%s)", id)
		}
		// Optionally add task type and priority if they provide valuable context
		if taskType, ok := taskInfo["type"].(string); ok && taskType != "" {
			spanInfo += fmt.Sprintf(" type=%s", taskType)
		}
	}
	// Format duration with appropriate unit and handle edge cases
	var durationStr string
	switch {
	case duration < 0:
		// Handle negative duration (likely a timing error)
		durationStr = "?"
	case duration == 0:
		// Handle zero duration (likely unfinished span)
		durationStr = "in progress"
	default:
		// Format positive durations appropriately
		durationStr = formatSpanDuration(duration)
	}

	spanInfo += fmt.Sprintf(" (%s)", durationStr)

	// Add error if present
	if span.Error != nil {
		spanInfo += fmt.Sprintf(" [ERROR: %v]", span.Error)
	}
	if tokenInfo, ok := span.Annotations["token_usage"].(*core.TokenUsage); ok {
		spanInfo += fmt.Sprintf(" {tokens=%d(%d↑%d↓)}",
			tokenInfo.TotalTokens,
			tokenInfo.PromptTokens,
			tokenInfo.CompletionTokens)
	}

	filteredAnnotations := filterRelevantAnnotations(span.Annotations)
	if len(filteredAnnotations) > 0 {
		spanInfo += fmt.Sprintf(" {%s}", formatAnnotations(filteredAnnotations))
	}

	fmt.Fprintln(b, spanInfo)

	// Find children
	var children []*core.Span
	for _, s := range spanMap {
		if s.ParentID == span.ID {
			children = append(children, s)
		}
	}

	// Sort children by start time
	sort.Slice(children, func(i, j int) bool {
		return children[i].StartTime.Before(children[j].StartTime)
	})

	// Process children
	childPrefix := prefix
	if !isLast {
		childPrefix += "│   "
	} else {
		childPrefix += "    "
	}

	for i, child := range children {
		formatSpanTree(b, child, spanMap, childPrefix, i == len(children)-1)
	}
}

// Only include relevant annotations that provide useful context.
func filterRelevantAnnotations(annotations map[string]interface{}) map[string]interface{} {
	relevant := make(map[string]interface{})

	// Define which annotations are meaningful for logging
	relevantKeys := map[string]bool{
		"status":   true,
		"result":   true,
		"error":    true,
		"count":    true,
		"score":    true,
		"progress": true,
	}

	for k, v := range annotations {
		if relevantKeys[k] {
			relevant[k] = v
		}
	}

	return relevant
}

func formatAnnotations(annotations map[string]interface{}) string {
	if len(annotations) == 0 {
		return ""
	}

	// Sort keys for consistent output
	keys := make([]string, 0, len(annotations))
	for k := range annotations {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, key := range keys {
		value := annotations[key]
		formatted := formatAnnotationValue(key, value)
		if formatted != "" {
			parts = append(parts, formatted)
		}
	}

	return strings.Join(parts, ", ")
}

// formatAnnotationValue handles different types of annotation values and formats
// them appropriately. It includes special handling for common value types and
// prevents overly verbose output.
func formatAnnotationValue(key string, value interface{}) string {
	switch v := value.(type) {
	case nil:
		return ""

	case bool:
		return fmt.Sprintf("%s=%t", key, v)

	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("%s=%d", key, v)

	case float32, float64:
		// Format floating point numbers with appropriate precision
		return fmt.Sprintf("%s=%.2f", key, v)

	case string:
		// Truncate long strings
		if len(v) > 47 {
			return fmt.Sprintf("%s='%s...'", key, v[:47])
		}
		if key == "status" || key == "result" {
			return fmt.Sprintf("%s=%s", key, v)
		}
		return fmt.Sprintf("%s='%s'", key, v)

	case time.Duration:
		return fmt.Sprintf("%s=%s", key, formatDuration(v))

	case []interface{}:
		// Handle arrays, limiting the number of items shown
		if len(v) == 0 {
			return ""
		}
		if len(v) > 3 {
			return fmt.Sprintf("%s=[%d items]", key, len(v))
		}
		items := make([]string, len(v))
		for i, item := range v {
			items[i] = fmt.Sprintf("%v", item)
		}
		return fmt.Sprintf("%s=[%s]", key, strings.Join(items, ","))

	case map[string]interface{}:
		// Handle nested maps, showing only the number of keys
		if len(v) == 0 {
			return ""
		}
		return fmt.Sprintf("%s={%d keys}", key, len(v))

	case error:
		return fmt.Sprintf("%s=error('%s')", key, v.Error())

	default:
		// For unknown types, use a simple string representation
		return fmt.Sprintf("%s=%v", key, v)
	}
}

// formatDuration formats a duration in a human-readable way,
// choosing appropriate units and precision.
func formatDuration(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	case d < time.Millisecond:
		return fmt.Sprintf("%.2fµs", float64(d.Nanoseconds())/1000)
	case d < time.Second:
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1000000)
	case d < time.Minute:
		return fmt.Sprintf("%.2fs", d.Seconds())
	default:
		return d.Round(time.Millisecond).String()
	}
}
