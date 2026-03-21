package agentutil

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// StringValue extracts a string from loosely typed map inputs.
func StringValue(value interface{}) string {
	if str, ok := value.(string); ok {
		return str
	}
	return ""
}

// IntValue extracts an integer from loosely typed map inputs.
func IntValue(value interface{}) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case json.Number:
		parsed, err := typed.Int64()
		if err == nil {
			return int(parsed)
		}
	}
	return 0
}

// DurationValue interprets generic input values as a duration.
//
// Semantics:
// - time.Duration and int64 are treated as already-normalized durations.
// - int/int32/float/json.Number/string numeric values are treated as seconds.
// - duration strings like "30s" use time.ParseDuration.
func DurationValue(value interface{}) time.Duration {
	switch typed := value.(type) {
	case time.Duration:
		return typed
	case int:
		return time.Duration(typed) * time.Second
	case int32:
		return time.Duration(typed) * time.Second
	case int64:
		return time.Duration(typed)
	case float64:
		return time.Duration(typed * float64(time.Second))
	case json.Number:
		if parsed, err := typed.Int64(); err == nil {
			return time.Duration(parsed) * time.Second
		}
		if parsed, err := typed.Float64(); err == nil {
			return time.Duration(parsed * float64(time.Second))
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return 0
		}
		if parsed, err := time.ParseDuration(trimmed); err == nil {
			return parsed
		}
		if parsed, err := strconv.ParseFloat(trimmed, 64); err == nil {
			return time.Duration(parsed * float64(time.Second))
		}
	}
	return 0
}

// TruncateString shortens text while preserving a visible ellipsis.
func TruncateString(value string, limit int) string {
	if limit <= 0 || utf8.RuneCountInString(value) <= limit {
		return value
	}
	runes := []rune(value)
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

// StringifyToolResult formats tool output for trace/transcript usage.
func StringifyToolResult(result core.ToolResult) string {
	text := strings.TrimSpace(fmt.Sprint(result.Data))
	if text == "" {
		text = "(no output)"
	}
	if isError, _ := result.Metadata["isError"].(bool); isError {
		return "tool reported error: " + text
	}
	return text
}
