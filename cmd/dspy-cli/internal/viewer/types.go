// Package viewer provides an enhanced CLI viewer for dspy-go JSONL log files.
package viewer

import "time"

// ANSI color codes (variables so they can be disabled).
var (
	Reset      = "\033[0m"
	Bold       = "\033[1m"
	Dim        = "\033[2m"
	Italic     = "\033[3m"
	Cyan       = "\033[36m"
	Green      = "\033[32m"
	Yellow     = "\033[33m"
	Blue       = "\033[34m"
	Magenta    = "\033[35m"
	Red        = "\033[31m"
	BoldCyan   = "\033[1;36m"
	BoldGreen  = "\033[1;32m"
	BoldYellow = "\033[1;33m"
	BoldBlue   = "\033[1;34m"
	BoldRed    = "\033[1;31m"
	BgBlue     = "\033[44m"
	BgGreen    = "\033[42m"
	White      = "\033[37m"
)

// LogFormat represents the type of log format detected.
type LogFormat int

const (
	FormatUnknown LogFormat = iota
	FormatRLM               // RLM-compatible format (metadata + iterations)
	FormatDSPy              // Native dspy-go format (session + events)
)

// String returns the string representation of the log format.
func (f LogFormat) String() string {
	switch f {
	case FormatRLM:
		return "RLM"
	case FormatDSPy:
		return "DSPy"
	default:
		return "Unknown"
	}
}

// ============================================================================
// RLM Format Types (iteration-based)
// ============================================================================

// Metadata represents the metadata entry in RLM JSONL format.
type Metadata struct {
	Type          string         `json:"type"`
	Timestamp     string         `json:"timestamp"`
	RootModel     string         `json:"root_model"`
	MaxIterations int            `json:"max_iterations"`
	Backend       string         `json:"backend"`
	Context       string         `json:"context"`
	Query         string         `json:"query"`
	BackendKwargs map[string]any `json:"backend_kwargs"`
}

// Iteration represents an iteration entry in RLM JSONL format.
type Iteration struct {
	Type          string      `json:"type"`
	Iteration     int         `json:"iteration"`
	Timestamp     string      `json:"timestamp"`
	Prompt        []Message   `json:"prompt"`
	Response      string      `json:"response"`
	CodeBlocks    []CodeBlock `json:"code_blocks"`
	FinalAnswer   any         `json:"final_answer"`
	IterationTime float64     `json:"iteration_time"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CodeBlock represents an executed code block.
type CodeBlock struct {
	Code   string     `json:"code"`
	Result CodeResult `json:"result"`
}

// CodeResult represents code execution results.
type CodeResult struct {
	Stdout        string         `json:"stdout"`
	Stderr        string         `json:"stderr"`
	Locals        map[string]any `json:"locals"`
	ExecutionTime float64        `json:"execution_time"`
	RLMCalls      []RLMCall      `json:"rlm_calls"`
}

// RLMCall represents a sub-LLM call.
type RLMCall struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	ExecutionTime    float64 `json:"execution_time"`
}

// ============================================================================
// Native DSPy Format Types (event-based)
// ============================================================================

// TraceEventType represents the type of trace event.
type TraceEventType string

const (
	TraceEventSession  TraceEventType = "session"
	TraceEventSpan     TraceEventType = "span"
	TraceEventLLMCall  TraceEventType = "llm_call"
	TraceEventModule   TraceEventType = "module"
	TraceEventCodeExec TraceEventType = "code_exec"
	TraceEventToolCall TraceEventType = "tool_call"
	TraceEventError    TraceEventType = "error"
)

// TraceEvent represents a single event in the native dspy-go format.
type TraceEvent struct {
	Type      TraceEventType         `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	TraceID   string                 `json:"trace_id"`
	SpanID    string                 `json:"span_id,omitempty"`
	ParentID  string                 `json:"parent_id,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// SessionData holds parsed session metadata from native format.
type SessionData struct {
	TraceID   string
	StartTime time.Time
	Metadata  map[string]interface{}
}

// SpanData holds parsed span information.
type SpanData struct {
	SpanID     string
	ParentID   string
	Operation  string
	StartTime  time.Time
	EndTime    time.Time
	DurationMs int64
	Inputs     map[string]interface{}
	Outputs    map[string]interface{}
	Error      string
	Events     []TraceEvent // Child events within this span
}

// LLMCallData holds parsed LLM call information.
type LLMCallData struct {
	SpanID           string
	Timestamp        time.Time
	Provider         string
	Model            string
	Prompt           string
	Response         string
	LatencyMs        int64
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Cost             float64
}

// ModuleData holds parsed module execution information.
type ModuleData struct {
	SpanID      string
	Timestamp   time.Time
	ModuleType  string
	ModuleName  string
	Signature   string
	DurationMs  int64
	LLMCalls    int
	TotalTokens int
	Success     bool
	Inputs      map[string]interface{}
	Outputs     map[string]interface{}
}

// CodeExecData holds parsed code execution information.
type CodeExecData struct {
	SpanID      string
	Timestamp   time.Time
	Iteration   int
	Code        string
	Stdout      string
	Stderr      string
	Locals      map[string]interface{}
	DurationMs  int64
	SubLLMCalls []map[string]interface{}
}

// ToolCallData holds parsed tool call information.
type ToolCallData struct {
	SpanID     string
	Timestamp  time.Time
	ToolName   string
	Input      interface{}
	Output     interface{}
	DurationMs int64
	Success    bool
	Error      string
}

// ErrorData holds parsed error information.
type ErrorData struct {
	SpanID      string
	Timestamp   time.Time
	ErrorType   string
	Message     string
	Recoverable bool
}

// ============================================================================
// Unified Log Data
// ============================================================================

// LogData holds all parsed log data from either format.
type LogData struct {
	Filename string
	Format   LogFormat

	// RLM format data
	Metadata   *Metadata
	Iterations []Iteration

	// Native DSPy format data
	Session   *SessionData
	Events    []TraceEvent
	LLMCalls  []LLMCallData
	Modules   []ModuleData
	CodeExecs []CodeExecData
	ToolCalls []ToolCallData
	Errors    []ErrorData
	Spans     map[string]*SpanData // SpanID -> SpanData
}

// Config holds viewer configuration.
type Config struct {
	Compact     bool
	NoColor     bool
	Interactive bool
	Watch       bool
	Iteration   int // -1 means all
	ErrorsOnly  bool
	FinalOnly   bool
	Search      string
	Stats       bool
	Export      string
}

// TokenCount holds token counts for prompt and completion.
type TokenCount struct {
	Prompt     int
	Completion int
}

// DisableColors sets all color codes to empty strings.
func DisableColors() {
	Reset = ""
	Bold = ""
	Dim = ""
	Italic = ""
	Cyan = ""
	Green = ""
	Yellow = ""
	Blue = ""
	Magenta = ""
	Red = ""
	BoldCyan = ""
	BoldGreen = ""
	BoldYellow = ""
	BoldBlue = ""
	BoldRed = ""
	BgBlue = ""
	BgGreen = ""
	White = ""
}

// HasErrors returns true if the log contains any errors.
func (d *LogData) HasErrors() bool {
	if d.Format == FormatRLM {
		for _, iter := range d.Iterations {
			for _, block := range iter.CodeBlocks {
				if block.Result.Stderr != "" {
					return true
				}
			}
		}
		return false
	}
	return len(d.Errors) > 0
}

// GetTotalLLMCalls returns the total number of LLM calls.
func (d *LogData) GetTotalLLMCalls() int {
	if d.Format == FormatRLM {
		count := 0
		for _, iter := range d.Iterations {
			for _, block := range iter.CodeBlocks {
				count += len(block.Result.RLMCalls)
			}
		}
		return count
	}
	return len(d.LLMCalls)
}

// GetTotalTokens returns total prompt and completion tokens.
func (d *LogData) GetTotalTokens() (prompt, completion int) {
	if d.Format == FormatRLM {
		for _, iter := range d.Iterations {
			for _, block := range iter.CodeBlocks {
				for _, call := range block.Result.RLMCalls {
					prompt += call.PromptTokens
					completion += call.CompletionTokens
				}
			}
		}
		return
	}
	for _, call := range d.LLMCalls {
		prompt += call.PromptTokens
		completion += call.CompletionTokens
	}
	return
}
