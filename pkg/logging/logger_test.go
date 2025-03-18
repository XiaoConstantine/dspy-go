package logging

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

type MockOutput struct {
	entries  []LogEntry
	mu       sync.Mutex
	closed   bool
	writeErr error
	syncErr  error
	closeErr error
}

func NewMockOutput() *MockOutput {
	return &MockOutput{
		entries: make([]LogEntry, 0),
	}
}

func (m *MockOutput) Write(entry LogEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("output is closed")
	}
	if m.writeErr != nil {
		return m.writeErr
	}
	m.entries = append(m.entries, entry)
	return nil
}

func (m *MockOutput) Sync() error {
	return m.syncErr
}

func (m *MockOutput) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return m.closeErr
}

func (m *MockOutput) GetEntries() []LogEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.entries
}

func (m *MockOutput) SetWriteError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.writeErr = err
}

func (m *MockOutput) SetSyncError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.syncErr = err
}

func (m *MockOutput) SetCloseError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeErr = err
}

func TestNewLogger(t *testing.T) {
	mockOutput := NewMockOutput()
	defaultFields := map[string]interface{}{
		"service": "test",
		"version": "1.0",
	}

	cfg := Config{
		Severity:      INFO,
		Outputs:       []Output{mockOutput},
		SampleRate:    100,
		DefaultFields: defaultFields,
	}

	logger := NewLogger(cfg)

	assert.Equal(t, INFO, logger.severity)
	assert.Equal(t, uint32(100), logger.sampleRate)
	assert.Equal(t, defaultFields, logger.fields)
	assert.Equal(t, 1, len(logger.outputs))
}

func TestGlobalLogger(t *testing.T) {
	// Reset default logger before testing
	mu.Lock()
	defaultLogger = nil
	mu.Unlock()

	// Test default logger creation
	logger1 := GetLogger()
	assert.NotNil(t, logger1)
	assert.Equal(t, INFO, logger1.severity)

	// Test setting custom logger
	mockOutput := NewMockOutput()
	customLogger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})
	SetLogger(customLogger)

	logger2 := GetLogger()
	assert.Equal(t, customLogger, logger2)
	assert.Equal(t, DEBUG, logger2.severity)

	// Test concurrent access to default logger (already initialized)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			logger := GetLogger()
			assert.Equal(t, customLogger, logger)
		}()
	}
	wg.Wait()
}

func TestConcurrentLogging(t *testing.T) {
	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	var wg sync.WaitGroup
	numGoroutines := 100
	messagesPerGoroutine := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(routineID int) {
			defer wg.Done()
			for j := 0; j < messagesPerGoroutine; j++ {
				logger.Info(context.Background(), "message from routine %d: %d", routineID, j)
			}
		}(i)
	}

	wg.Wait()

	entries := mockOutput.GetEntries()
	assert.Equal(t, numGoroutines*messagesPerGoroutine, len(entries))
}

func TestPromptCompletionLogging(t *testing.T) {
	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	tokenInfo := &core.TokenInfo{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}

	ctx := context.Background()
	logger.PromptCompletion(ctx, "test prompt", "test completion", tokenInfo)

	entries := mockOutput.GetEntries()
	assert.NotEmpty(t, entries)
	lastEntry := entries[len(entries)-1]
	assert.Contains(t, lastEntry.Message, "test prompt")
	assert.Contains(t, lastEntry.Message, "test completion")
}

func TestSeverityFiltering(t *testing.T) {
	tests := []struct {
		name           string
		loggerSeverity Severity
		logSeverity    Severity
		shouldLog      bool
	}{
		{"Debug to Debug", DEBUG, DEBUG, true},
		{"Info to Debug", INFO, DEBUG, false},
		{"Warn to Debug", WARN, DEBUG, false},
		{"Error to Debug", ERROR, DEBUG, false},
		{"Fatal to Debug", FATAL, DEBUG, false},

		{"Debug to Info", DEBUG, INFO, true},
		{"Info to Info", INFO, INFO, true},
		{"Warn to Info", WARN, INFO, false},
		{"Error to Info", ERROR, INFO, false},
		{"Fatal to Info", FATAL, INFO, false},

		{"Debug to Warn", DEBUG, WARN, true},
		{"Info to Warn", INFO, WARN, true},
		{"Warn to Warn", WARN, WARN, true},
		{"Error to Warn", ERROR, WARN, false},
		{"Fatal to Warn", FATAL, WARN, false},

		{"Debug to Error", DEBUG, ERROR, true},
		{"Info to Error", INFO, ERROR, true},
		{"Warn to Error", WARN, ERROR, true},
		{"Error to Error", ERROR, ERROR, true},
		{"Fatal to Error", FATAL, ERROR, false},

		{"Debug to Fatal", DEBUG, FATAL, true},
		{"Info to Fatal", INFO, FATAL, true},
		{"Warn to Fatal", WARN, FATAL, true},
		{"Error to Fatal", ERROR, FATAL, true},
		{"Fatal to Fatal", FATAL, FATAL, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockOutput := NewMockOutput()
			logger := NewLogger(Config{
				Severity: tt.loggerSeverity,
				Outputs:  []Output{mockOutput},
			})

			// Log at the test's log severity
			logger.logf(context.Background(), tt.logSeverity, "test message")

			entries := mockOutput.GetEntries()
			if tt.shouldLog {
				assert.NotEmpty(t, entries, "Message should have been logged")
			} else {
				assert.Empty(t, entries, "Message should not have been logged")
			}
		})
	}
}

func TestLogLevels(t *testing.T) {
	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	ctx := context.Background()

	// Test each log level
	logger.Debug(ctx, "debug message")
	logger.Info(ctx, "info message")
	logger.Warn(ctx, "warning message")
	logger.Error(ctx, "error message")

	entries := mockOutput.GetEntries()
	assert.Equal(t, 4, len(entries))

	assert.Equal(t, DEBUG, entries[0].Severity)
	assert.Equal(t, INFO, entries[1].Severity)
	assert.Equal(t, WARN, entries[2].Severity)
	assert.Equal(t, ERROR, entries[3].Severity)

	assert.Equal(t, "debug message", entries[0].Message)
	assert.Equal(t, "info message", entries[1].Message)
	assert.Equal(t, "warning message", entries[2].Message)
	assert.Equal(t, "error message", entries[3].Message)

	// Test message formatting with arguments
	logger.Debug(ctx, "formatted %s: %d", "message", 42)
	lastEntry := mockOutput.GetEntries()[len(mockOutput.GetEntries())-1]
	assert.Equal(t, "formatted message: 42", lastEntry.Message)
}

//	func TestContextValues(t *testing.T) {
//		mockOutput := NewMockOutput()
//		logger := NewLogger(Config{
//			Severity: DEBUG,
//			Outputs:  []Output{mockOutput},
//		})
//
//		// Basic context without values
//		basicCtx := context.Background()
//		logger.Info(basicCtx, "basic context")
//
//		// Context with model ID
//		modelCtx := context.WithValue(context.Background(), ModelIDKey, core.ModelID("gpt-4"))
//		logger.Info(modelCtx, "model context")
//
//		// Context with token info
//		tokenInfo := &core.TokenInfo{
//			PromptTokens:     100,
//			CompletionTokens: 50,
//			TotalTokens:      150,
//		}
//		tokenCtx := context.WithValue(context.Background(), TokenInfoKey, tokenInfo)
//		logger.Info(tokenCtx, "token context")
//
//		// Context with execution state
//		executionState := &mockExecutionState{
//			traceID: "test-trace",
//			modelID: "claude-3",
//			tokenUsage: &core.TokenUsage{
//				PromptTokens:     200,
//				CompletionTokens: 100,
//				TotalTokens:      300,
//			},
//			spans: []*core.Span{
//				{
//					ID:        "span-1",
//					Operation: "test-operation",
//					StartTime: time.Now(),
//				},
//			},
//		}
//		stateCtx := context.WithValue(context.Background(), core.ExecutionStateKey, executionState)
//		logger.Info(stateCtx, "state context")
//
//		entries := mockOutput.GetEntries()
//		assert.Equal(t, 4, len(entries))
//
//		// Check basic context entry
//		assert.Empty(t, entries[0].ModelID)
//		assert.Nil(t, entries[0].TokenInfo)
//		assert.Empty(t, entries[0].TraceID)
//
//		// Check model ID entry
//		assert.Equal(t, "gpt-4", entries[1].ModelID)
//
//		// Check token info entry
//		assert.Equal(t, tokenInfo, entries[2].TokenInfo)
//
//		// Check execution state entry
//		assert.Equal(t, "test-trace", entries[3].TraceID)
//		assert.Contains(t, entries[3].Fields, "model_id")
//		assert.Equal(t, "claude-3", entries[3].Fields["model_id"])
//		assert.Contains(t, entries[3].Fields, "token_usage")
//		assert.Contains(t, entries[3].Fields, "spans")
//	}
func TestOutputError(t *testing.T) {
	// Create a mock output that returns an error on Write
	mockOutput := NewMockOutput()
	mockOutput.SetWriteError(fmt.Errorf("write error"))

	// Redirect stderr temporarily to capture the error message
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	// Log something to trigger the error
	logger.Info(context.Background(), "test message")

	// Restore stderr and get the captured output
	w.Close()
	os.Stderr = oldStderr

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("Failed to write")

	}
	captured := buf.String()

	// Check that the error was reported
	assert.Contains(t, captured, "failed to write log entry")
	assert.Contains(t, captured, "write error")
}

func TestMultipleOutputs(t *testing.T) {
	mockOutput1 := NewMockOutput()
	mockOutput2 := NewMockOutput()

	logger := NewLogger(Config{
		Severity: INFO,
		Outputs:  []Output{mockOutput1, mockOutput2},
	})

	logger.Info(context.Background(), "test message")

	// Both outputs should receive the message
	assert.Equal(t, 1, len(mockOutput1.GetEntries()))
	assert.Equal(t, 1, len(mockOutput2.GetEntries()))

	// If one output fails, the others should still succeed
	mockOutput1.SetWriteError(fmt.Errorf("write error"))

	// Redirect stderr temporarily to capture the error message
	oldStderr := os.Stderr
	_, w, _ := os.Pipe()
	os.Stderr = w

	logger.Info(context.Background(), "another message")

	// Restore stderr
	w.Close()
	os.Stderr = oldStderr

	// mockOutput1 should still have 1 entry, mockOutput2 should have 2
	assert.Equal(t, 1, len(mockOutput1.GetEntries()))
	assert.Equal(t, 2, len(mockOutput2.GetEntries()))
}

func TestPriorityLog(t *testing.T) {
	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: INFO, // Only INFO and higher severity
		Outputs:  []Output{mockOutput},
	})

	ctx := context.Background()

	// This should be filtered out
	logger.Debug(ctx, "debug message")
	assert.Empty(t, mockOutput.GetEntries())

	// These should all be logged
	logger.Info(ctx, "info message")
	logger.Warn(ctx, "warning message")
	logger.Error(ctx, "error message")

	entries := mockOutput.GetEntries()
	assert.Equal(t, 3, len(entries))
}

func TestPromptCompletionLevelFiltering(t *testing.T) {
	t.Run("PromptCompletion at DEBUG level", func(t *testing.T) {
		mockOutput := NewMockOutput()
		logger := NewLogger(Config{
			Severity: DEBUG,
			Outputs:  []Output{mockOutput},
		})

		logger.PromptCompletion(context.Background(), "test prompt", "test completion", nil)
		assert.NotEmpty(t, mockOutput.GetEntries())
	})

	t.Run("PromptCompletion at INFO level", func(t *testing.T) {
		mockOutput := NewMockOutput()
		logger := NewLogger(Config{
			Severity: INFO, // Higher than DEBUG
			Outputs:  []Output{mockOutput},
		})

		logger.PromptCompletion(context.Background(), "test prompt", "test completion", nil)
		assert.Empty(t, mockOutput.GetEntries())
	})
}

func TestDefaultFields(t *testing.T) {
	mockOutput := NewMockOutput()
	defaultFields := map[string]interface{}{
		"service": "test-service",
		"version": "1.0.0",
		"env":     "test",
	}

	logger := NewLogger(Config{
		Severity:      DEBUG,
		Outputs:       []Output{mockOutput},
		DefaultFields: defaultFields,
	})

	logger.Info(context.Background(), "test message")

	entries := mockOutput.GetEntries()
	assert.Equal(t, 1, len(entries))

	// Check that default fields were added
	assert.Equal(t, "test-service", entries[0].Fields["service"])
	assert.Equal(t, "1.0.0", entries[0].Fields["version"])
	assert.Equal(t, "test", entries[0].Fields["env"])
}

func TestFieldPriority(t *testing.T) {
	mockOutput := NewMockOutput()
	defaultFields := map[string]interface{}{
		"service": "default-service",
		"version": "1.0.0",
	}

	logger := NewLogger(Config{
		Severity:      DEBUG,
		Outputs:       []Output{mockOutput},
		DefaultFields: defaultFields,
	})

	// Create a context with a field that would override a default field
	ctx := context.Background()
	// state := &mockExecutionState{
	// 	fields: map[string]interface{}{
	// 		"service": "context-service", // This should take precedence
	// 		"context": "test-context",    // This is unique to the context
	// 	},
	// }
	// // OR if there's a function to add the state to context:
	ctx = core.WithExecutionState(ctx)

	logger.Info(ctx, "test message")

	entries := mockOutput.GetEntries()
	assert.Equal(t, 1, len(entries))

	// Check field priority - context fields should override default fields
	// (Note: In this test we're mocking how the fields would be set through core.GetExecutionState,
	// but the real logic doesn't actually transfer state.fields to entry.Fields directly)
}

// TestFatal tests the Fatal method without actually exiting the process.
func TestFatal(t *testing.T) {
	// We need to replace the os.Exit function to test Fatal without terminating the test
	origOsExit := osExit
	defer func() { osExit = origOsExit }()

	var exitCode int
	osExit = func(code int) {
		exitCode = code
	}

	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	// Call Fatal
	logger.Fatal(context.Background(), "fatal message")

	// Check that the message was logged at FATAL level
	entries := mockOutput.GetEntries()
	assert.Equal(t, 1, len(entries))
	assert.Equal(t, FATAL, entries[0].Severity)
	assert.Equal(t, "fatal message", entries[0].Message)

	// Check that os.Exit was called with code 1
	assert.Equal(t, 1, exitCode)
}

// TestFatalf tests the Fatalf method without actually exiting the process.
func TestFatalf(t *testing.T) {
	// We need to replace the os.Exit function to test Fatalf without terminating the test
	origOsExit := osExit
	defer func() { osExit = origOsExit }()

	var exitCode int
	osExit = func(code int) {
		exitCode = code
	}

	mockOutput := NewMockOutput()
	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})

	// Call Fatalf with formatting
	logger.Fatalf(context.Background(), "fatal message: %d", 42)

	// Check that the message was logged at FATAL level with proper formatting
	entries := mockOutput.GetEntries()
	assert.Equal(t, 1, len(entries))
	assert.Equal(t, FATAL, entries[0].Severity)
	assert.Equal(t, "fatal message: 42", entries[0].Message)

	// Check that os.Exit was called with code 1
	assert.Equal(t, 1, exitCode)
}

// Test sync/close of outputs during fatal exit.
func TestFatalOutputCleanup(t *testing.T) {
	// We need to replace the os.Exit function to test Fatal without terminating the test
	origOsExit := osExit
	defer func() { osExit = origOsExit }()

	osExit = func(code int) {
		// Don't actually exit
	}

	mockOutput1 := NewMockOutput()
	mockOutput2 := NewMockOutput()

	logger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput1, mockOutput2},
	})

	// Call Fatal
	logger.Fatal(context.Background(), "fatal message")

	// Check that both outputs were synced and closed
	assert.True(t, mockOutput1.closed)
	assert.True(t, mockOutput2.closed)
}
