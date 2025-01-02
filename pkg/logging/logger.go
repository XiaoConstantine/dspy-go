package logging

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

var (
	defaultLogger *Logger
	mu            sync.RWmutex
)

// Logger provides the core logging functionality
type Logger struct {
	mu         sync.Mutex
	severity   Severity
	outputs    []Output
	sampleRate uint32                 // For high-frequency event sampling
	fields     map[string]interface{} // Default fields for all logs
}

// Output interface allows for different logging destinations
type Output interface {
	Write(LogEntry) error
	Sync() error
	Close() error
}

// Config allows flexible logger configuration
type Config struct {
	Severity      Severity
	Outputs       []Output
	SampleRate    uint32
	DefaultFields map[string]interface{}
}

// NewLogger creates a new logger with the given configuration
func NewLogger(cfg Config) *Logger {
	return &Logger{
		severity:   cfg.Severity,
		outputs:    cfg.Outputs,
		sampleRate: cfg.SampleRate,
		fields:     cfg.DefaultFields,
	}
}

// logf is the core logging function that handles all severity levels
func (l *Logger) logf(ctx context.Context, s Severity, format string, args ...interface{}) {
	// Early severity check for performance
	if s < l.severity {
		return
	}

	// Get caller information
	pc, file, line, _ := runtime.Caller(2)
	fn := runtime.FuncForPC(pc).Name()

	// Create base entry
	entry := LogEntry{
		Time:     time.Now().UnixNano(),
		Severity: s,
		Message:  fmt.Sprintf(format, args...),
		File:     filepath.Base(file),
		Line:     line,
		Function: filepath.Base(fn),
		Fields:   make(map[string]interface{}),
	}

	// Add context values if present
	if ctx != nil {
		if modelID := ctx.Value(ModelIDKey); modelID != nil {
			entry.ModelID = modelID.(string)
		}
		if tokenInfo := ctx.Value(TokenInfoKey); tokenInfo != nil {
			entry.TokenInfo = tokenInfo.(*TokenInfo)
		}
	}

	// Add default fields
	for k, v := range l.fields {
		if _, exists := entry.Fields[k]; !exists {
			entry.Fields[k] = v
		}
	}

	// Write to all outputs
	l.mu.Lock()
	defer l.mu.Unlock()

	for _, out := range l.outputs {
		if err := out.Write(entry); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write log entry: %v\n", err)
		}
	}
}

// LLM-specific logging methods
func (l *Logger) PromptCompletion(ctx context.Context, prompt, completion string, tokenInfo *TokenInfo) {
	if l.severity > DEBUG {
		return
	}

	l.Debug(ctx, "LLM Interaction",
		"prompt", prompt,
		"completion", completion,
		"token_info", tokenInfo,
	)
}

// Regular severity-based logging methods
func (l *Logger) Debug(ctx context.Context, format string, args ...interface{}) {
	l.logf(ctx, DEBUG, format, args...)
}

func (l *Logger) Info(ctx context.Context, format string, args ...interface{}) {
	l.logf(ctx, INFO, format, args...)
}

func (l *Logger) Warn(ctx context.Context, format string, args ...interface{}) {
	l.logf(ctx, WARN, format, args...)
}

func (l *Logger) Error(ctx context.Context, format string, args ...interface{}) {
	l.logf(ctx, ERROR, format, args...)
}
