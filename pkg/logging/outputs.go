package logging

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ConsoleOutput formats logs for human readability.
type ConsoleOutput struct {
	mu     sync.Mutex
	writer io.Writer
	color  bool // Whether to use ANSI color codes
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
		for i, part := range parts {
			if strings.HasPrefix(part, "gpt") && i+1 < len(parts) {
				return fmt.Sprintf("%s-%s", part, parts[i+1])
			}
		}
		return parts[0]

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
		if len(v) > 50 {
			return fmt.Sprintf("%s='%s...'", key, v[:47])
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
