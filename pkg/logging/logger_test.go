package logging

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"

	"github.com/stretchr/testify/assert"
)

type MockOutput struct {
	entries []LogEntry
	mu      sync.Mutex
	closed  bool
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
	m.entries = append(m.entries, entry)
	return nil
}

func (m *MockOutput) Sync() error {
	return nil
}

func (m *MockOutput) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func (m *MockOutput) GetEntries() []LogEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.entries
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
}

func TestGlobalLogger(t *testing.T) {
	// Test default logger creation
	logger1 := GetLogger()
	assert.NotNil(t, logger1)

	// Test setting custom logger
	mockOutput := NewMockOutput()
	customLogger := NewLogger(Config{
		Severity: DEBUG,
		Outputs:  []Output{mockOutput},
	})
	SetLogger(customLogger)

	logger2 := GetLogger()
	assert.Equal(t, customLogger, logger2)
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

func TestFieldTruncation(t *testing.T) {
	longText := strings.Repeat("a", 200)
	fields := map[string]interface{}{
		"prompt":     longText,
		"completion": longText,
	}

	formatted := formatFields(fields)
	assert.True(t, len(formatted) < len(longText)*2)
	assert.Contains(t, formatted, "...")
}
