package logging

// LogEntry represents a structured log record with fields particularly relevant to LLM operations
type LogEntry struct {
	// Standard fields
	Time     int64
	Severity Severity
	Message  string
	File     string
	Line     int
	Function string

	// LLM-specific fields
	ModelID   string     // The LLM model being used
	TokenInfo *TokenInfo // Token usage information
	Latency   int64      // Operation duration in milliseconds
	Cost      float64    // Operation cost in dollars

	// General structured data
	Fields map[string]interface{}
}

// TokenInfo tracks token usage for cost and performance monitoring
type TokenInfo struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}
