package logging

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

var (
	defaultLogger *Logger
	mu            sync.RWMutex
)

// Logger provides the core logging functionality.
type Logger struct {
	mu         sync.Mutex
	severity   Severity
	outputs    []Output
	sampleRate uint32                 // For high-frequency event sampling
	fields     map[string]interface{} // Default fields for all logs
}

// Output interface allows for different logging destinations.
type Output interface {
	Write(LogEntry) error
	Sync() error
	Close() error
}

// Config allows flexible logger configuration.
type Config struct {
	Severity      Severity
	Outputs       []Output
	SampleRate    uint32
	DefaultFields map[string]interface{}
}

// NewLogger creates a new logger with the given configuration.
func NewLogger(cfg Config) *Logger {
	return &Logger{
		severity:   cfg.Severity,
		outputs:    cfg.Outputs,
		sampleRate: cfg.SampleRate,
		fields:     cfg.DefaultFields,
	}
}

// logf is the core logging function that handles all severity levels.
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
		if modelID, ok := GetModelID(ctx); ok {
			entry.ModelID = string(modelID)
		}

		if tokenInfo, ok := GetTokenInfo(ctx); ok {
			entry.TokenInfo = tokenInfo
		}
	}
	if state := core.GetExecutionState(ctx); state != nil {
		entry.TraceID = state.GetTraceID()
	}

	// Add default fields
	for k, v := range l.fields {
		if _, exists := entry.Fields[k]; !exists {
			entry.Fields[k] = v
		}
	}
	// Add execution context information if available
	if state := core.GetExecutionState(ctx); state != nil {
		entry.Fields["model_id"] = state.GetModelID()
		if usage := state.GetTokenUsage(); usage != nil {
			entry.Fields["token_usage"] = usage
		}
		entry.Fields["spans"] = core.CollectSpans(ctx)
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

// LLM-specific logging methods.
func (l *Logger) PromptCompletion(ctx context.Context, prompt, completion string, tokenInfo *core.TokenInfo) {
	if l.severity > DEBUG {
		return
	}

	l.Debug(ctx, "LLM Interaction: prompt: %s, completion: %v, token_info: %v",
		prompt,
		completion,
		tokenInfo,
	)
}

// Regular severity-based logging methods.
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

// GetLogger returns the global logger instance.
func GetLogger() *Logger {
	// First try reading without a write lock
	mu.RLock()
	if l := defaultLogger; l != nil {
		mu.RUnlock()
		return l
	}
	mu.RUnlock()

	// If no logger exists, create one with write lock
	mu.Lock()
	defer mu.Unlock()

	if defaultLogger == nil {
		// Create default logger with reasonable defaults
		defaultLogger = NewLogger(Config{
			Severity: INFO,
			Outputs: []Output{
				NewConsoleOutput(false),
			},
		})
	}

	return defaultLogger
}

// SetLogger allows setting a custom configured logger as the global instance.
func SetLogger(l *Logger) {
	mu.Lock()
	defaultLogger = l
	mu.Unlock()
}
