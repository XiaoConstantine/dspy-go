package logging

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// contextKey is a custom type for context keys to avoid collisions.
type contextKey string

const (
	// ModelIDKey is used to store/retrieve ModelID from context.
	ModelIDKey contextKey = "model_id"

	// TokenInfoKey is used to store/retrieve token usage information.
	TokenInfoKey contextKey = "token_info"
)

// LogEntry represents a structured log record with fields particularly relevant to LLM operations.
type LogEntry struct {
	// Standard fields
	Time     int64
	Severity Severity
	Message  string
	File     string
	Line     int
	Function string
	TraceID  string // Added trace ID field

	// LLM-specific fields
	ModelID   string          // The LLM model being used
	TokenInfo *core.TokenInfo // Token usage information
	Latency   int64           // Operation duration in milliseconds
	Cost      float64         // Operation cost in dollars

	// General structured data
	Fields map[string]interface{}
}

// WithModelID adds a ModelID to the context.
func WithModelID(ctx context.Context, modelID core.ModelID) context.Context {
	return context.WithValue(ctx, ModelIDKey, modelID)
}

// GetModelID retrieves ModelID from context.
func GetModelID(ctx context.Context) (core.ModelID, bool) {
	if v := ctx.Value(ModelIDKey); v != nil {
		if mid, ok := v.(core.ModelID); ok {
			return mid, true
		}
	}
	return "", false
}

// WithTokenInfo adds TokenInfo to the context.
func WithTokenInfo(ctx context.Context, info *core.TokenInfo) context.Context {
	return context.WithValue(ctx, TokenInfoKey, info)
}

// GetTokenInfo retrieves TokenInfo from context.
func GetTokenInfo(ctx context.Context) (*core.TokenInfo, bool) {
	if v := ctx.Value(TokenInfoKey); v != nil {
		if ti, ok := v.(*core.TokenInfo); ok {
			return ti, true
		}
	}
	return nil, false
}
