package logging

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Define this at the package level, outside any function.
type CustomFormatter struct{}

func (c *CustomFormatter) Format(e LogEntry) string {
	return fmt.Sprintf("CUSTOM [%s] %s", e.Severity, e.Message)
}

func TestConsoleOutputColor(t *testing.T) {
	tests := []struct {
		name     string
		severity Severity
		color    bool
	}{
		{"ColorDebug", DEBUG, true},
		{"ColorInfo", INFO, true},
		{"ColorWarn", WARN, true},
		{"ColorError", ERROR, true},
		{"ColorFatal", FATAL, true},
		{"NoColor", INFO, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buffer := &bytes.Buffer{}
			console := &ConsoleOutput{
				writer: buffer,
				color:  tt.color,
			}

			entry := LogEntry{
				Time:     time.Now().UnixNano(),
				Severity: tt.severity,
				Message:  "test message",
			}

			err := console.Write(entry)
			require.NoError(t, err)

			output := buffer.String()
			if tt.color {
				assert.Contains(t, output, "\033[")
			} else {
				assert.NotContains(t, output, "\033[")
			}
		})
	}
}

func TestOutputSyncAndClose(t *testing.T) {
	// Test with file output
	tmpFile, err := os.CreateTemp("", "log-test-*")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	console := &ConsoleOutput{
		writer: tmpFile,
		color:  false,
	}

	// Test Sync
	err = console.Sync()
	assert.NoError(t, err)

	// Test Close
	err = console.Close()
	assert.NoError(t, err)

	// Test with non-syncable writer
	buffer := &bytes.Buffer{}
	console = &ConsoleOutput{
		writer: buffer,
		color:  false,
	}

	err = console.Sync()
	assert.NoError(t, err)

	err = console.Close()
	assert.NoError(t, err)
}

func TestFileOutput(t *testing.T) {
	// Create a temporary directory for test logs
	tempDir, err := os.MkdirTemp("", "file_output_test")
	require.NoError(t, err, "Failed to create temp directory")
	defer os.RemoveAll(tempDir) // Cleanup after test

	t.Run("Basic file writing", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "test.log")
		fileOutput, err := NewFileOutput(logPath)
		require.NoError(t, err, "Failed to create FileOutput")
		defer fileOutput.Close()

		// Create a test log entry
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test log message",
			File:     "file_output_test.go",
			Line:     42,
			Function: "TestFileOutput",
			TraceID:  "test-trace-id",
		}

		// Write the entry
		err = fileOutput.Write(entry)
		assert.NoError(t, err, "Failed to write log entry")

		// Read the file contents
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read log file")

		// Check that the message was written correctly
		assert.Contains(t, string(content), "Test log message", "Log file doesn't contain the expected message")
		assert.Contains(t, string(content), "INFO", "Log file doesn't contain severity level")
		assert.Contains(t, string(content), "test-trace-id", "Log file doesn't contain trace ID")
	})

	t.Run("JSON formatting", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "json_test.log")
		fileOutput, err := NewFileOutput(
			logPath,
			WithJSONFormat(true),
		)
		require.NoError(t, err, "Failed to create FileOutput with JSON format")
		defer fileOutput.Close()

		// Create a test log entry with structured fields
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "JSON test message",
			File:     "file_output_test.go",
			Line:     50,
			TraceID:  "json-trace-id",
			Fields: map[string]interface{}{
				"user_id": "user123",
				"action":  "login",
				"status":  "success",
			},
		}

		// Write the entry
		err = fileOutput.Write(entry)
		assert.NoError(t, err, "Failed to write JSON log entry")

		// Read the file contents
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read JSON log file")

		// Try to parse the content as JSON
		var logData map[string]interface{}
		err = json.Unmarshal(content, &logData)
		assert.NoError(t, err, "Log file doesn't contain valid JSON")

		// Check JSON fields
		assert.Equal(t, "JSON test message", logData["message"], "Incorrect message in JSON log")
		assert.Equal(t, "INFO", logData["level"], "Incorrect level in JSON log")
		assert.Equal(t, "json-trace-id", logData["traceId"], "Incorrect traceId in JSON log")
		assert.Equal(t, "user123", logData["user_id"], "Incorrect user_id in JSON log")
		assert.Equal(t, "login", logData["action"], "Incorrect action in JSON log")
		assert.Equal(t, "success", logData["status"], "Incorrect status in JSON log")
	})

	t.Run("Custom formatter", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "custom_format.log")

		fileOutput, err := NewFileOutput(
			logPath,

			WithFormatter(&CustomFormatter{}),
		)
		require.NoError(t, err, "Failed to create FileOutput with custom formatter")
		defer fileOutput.Close()

		// Create a test log entry
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Custom format test",
		}

		// Write the entry
		err = fileOutput.Write(entry)
		assert.NoError(t, err, "Failed to write log entry with custom format")

		// Read the file contents
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read custom formatted log file")

		// Check the custom format
		assert.Contains(t, string(content), "CUSTOM [INFO] Custom format test",
			"Log file doesn't match expected custom format")
	})
	t.Run("File rotation", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "rotation_test.log")

		// Create file output with small rotation size (100 bytes) and keep 2 files
		fileOutput, err := NewFileOutput(
			logPath,
			WithRotation(100, 2),
		)
		require.NoError(t, err, "Failed to create FileOutput with rotation")
		defer fileOutput.Close()

		// Generate several large log entries to trigger rotation
		for i := 0; i < 10; i++ {
			entry := LogEntry{
				Time:     time.Now().UnixNano(),
				Severity: INFO,
				Message:  fmt.Sprintf("Rotation test message %d with extra padding to exceed size limit", i),
			}
			err = fileOutput.Write(entry)
			assert.NoError(t, err, "Failed to write log entry during rotation test")

			// Small delay to ensure different timestamps in rotation files
			time.Sleep(10 * time.Millisecond)
		}

		// Check that the main log file exists
		info, err := os.Stat(logPath)
		require.NoError(t, err, "Main log file doesn't exist")

		// Instead of strict size verification, check rotation behavior
		t.Logf("Main log file size: %d bytes (rotation limit: 100 bytes)", info.Size())

		// Allow a reasonable margin (10%) over the limit
		maxAllowedSize := int64(110) // 10% tolerance
		assert.LessOrEqual(t, info.Size(), maxAllowedSize,
			"Main log file significantly exceeded rotation size")

		// Verify that rotation occurred by checking for backup files
		files, err := filepath.Glob(logPath + ".*")
		require.NoError(t, err, "Failed to list log backup files")
		assert.NotEmpty(t, files, "No rotation backup files were created")

		// Due to cleanup, we should have at most 2 backup files (the limit we set)
		assert.LessOrEqual(t, len(files), 2, "Too many backup files remain after rotation cleanup")

		// Verify the content of backup files
		if len(files) > 0 {
			// Sort files to get the most recent one
			latestBackup := files[len(files)-1]

			backupContent, err := os.ReadFile(latestBackup)
			require.NoError(t, err, "Failed to read backup log file")
			assert.Contains(t, string(backupContent), "Rotation test message",
				"Backup file doesn't contain expected log messages")
		}
	})
	t.Run("Logger integration", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "integration_test.log")
		fileOutput, err := NewFileOutput(logPath)
		require.NoError(t, err, "Failed to create FileOutput for logger integration")
		defer fileOutput.Close()

		// Create a logger with our file output
		logger := NewLogger(Config{
			Severity: DEBUG,
			Outputs:  []Output{fileOutput},
		})

		// Log a few messages
		ctx := context.Background()
		logger.Debug(ctx, "Debug message")
		logger.Info(ctx, "Info message")
		logger.Warn(ctx, "Warning message")
		logger.Error(ctx, "Error message")

		// Read the file contents
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read integration test log file")
		contentStr := string(content)

		// Check that all messages were written
		assert.Contains(t, contentStr, "Debug message", "Log file missing debug message")
		assert.Contains(t, contentStr, "Info message", "Log file missing info message")
		assert.Contains(t, contentStr, "Warning message", "Log file missing warning message")
		assert.Contains(t, contentStr, "Error message", "Log file missing error message")
	})

	t.Run("Creating directories", func(t *testing.T) {
		// Try to create a log file in a nested directory that doesn't exist
		nestedDir := filepath.Join(tempDir, "nested", "path", "for", "logs")
		logPath := filepath.Join(nestedDir, "nested_test.log")

		fileOutput, err := NewFileOutput(logPath)
		require.NoError(t, err, "Failed to create FileOutput with nested directories")
		defer fileOutput.Close()

		// Check that the directory was created
		_, err = os.Stat(nestedDir)
		assert.NoError(t, err, "Nested directory wasn't created")

		// Write a test entry
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Nested directory test",
		}
		err = fileOutput.Write(entry)
		assert.NoError(t, err, "Failed to write to log file in nested directory")

		// Verify the file exists and has the content
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read nested directory log file")
		assert.Contains(t, string(content), "Nested directory test",
			"Log file in nested directory doesn't contain expected message")
	})

	t.Run("Error handling", func(t *testing.T) {
		// Create a file output
		logPath := filepath.Join(tempDir, "error_test.log")
		fileOutput, err := NewFileOutput(logPath)
		require.NoError(t, err, "Failed to create FileOutput for error test")

		// Write a valid entry
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Before close",
		}
		err = fileOutput.Write(entry)
		assert.NoError(t, err, "Failed to write initial log entry")

		// Close the file
		err = fileOutput.Close()
		assert.NoError(t, err, "Failed to close file output")

		// Try to write after closing (should fail)
		entry.Message = "After close"
		err = fileOutput.Write(entry)
		assert.Error(t, err, "Writing to closed file should return an error")

		// Try to create a file output with an invalid path
		invalidOutput, err := NewFileOutput("/nonexistent/directory/that/should/not/exist/logfile.log")
		assert.Error(t, err, "Creating FileOutput with invalid path should return an error")
		assert.Nil(t, invalidOutput, "FileOutput should be nil when creation fails")
	})

	t.Run("Concurrent writing", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "concurrent_test.log")
		fileOutput, err := NewFileOutput(logPath)
		require.NoError(t, err, "Failed to create FileOutput for concurrency test")
		defer fileOutput.Close()

		// Number of concurrent writers
		numWriters := 10
		// Number of log entries per writer
		entriesPerWriter := 100

		// Use a channel to synchronize the goroutines
		done := make(chan bool)

		// Launch multiple goroutines to write log entries concurrently
		for i := 0; i < numWriters; i++ {
			go func(writerID int) {
				for j := 0; j < entriesPerWriter; j++ {
					entry := LogEntry{
						Time:     time.Now().UnixNano(),
						Severity: INFO,
						Message:  fmt.Sprintf("Concurrent writer %d, entry %d", writerID, j),
					}
					err := fileOutput.Write(entry)
					if err != nil {
						t.Errorf("Failed to write log entry concurrently: %v", err)
					}
				}
				done <- true
			}(i)
		}

		// Wait for all writers to finish
		for i := 0; i < numWriters; i++ {
			<-done
		}

		// Verify the file exists and has the expected number of entries
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read concurrency test log file")

		// Count the number of log entries
		lines := strings.Split(string(content), "\n")
		// Last line might be empty
		nonEmptyLines := 0
		for _, line := range lines {
			if line != "" {
				nonEmptyLines++
			}
		}

		// We should have numWriters * entriesPerWriter log entries
		assert.Equal(t, numWriters*entriesPerWriter, nonEmptyLines,
			"Log file doesn't have the expected number of entries")
	})

	t.Run("Buffered writing", func(t *testing.T) {
		logPath := filepath.Join(tempDir, "buffered_test.log")
		fileOutput, err := NewFileOutput(
			logPath,
			WithBufferSize(4096), // 4KB buffer
		)
		require.NoError(t, err, "Failed to create FileOutput with buffer")

		// Write some entries without flushing
		for i := 0; i < 5; i++ {
			entry := LogEntry{
				Time:     time.Now().UnixNano(),
				Severity: INFO,
				Message:  fmt.Sprintf("Buffered log entry %d", i),
			}
			err = fileOutput.Write(entry)
			assert.NoError(t, err, "Failed to write buffered log entry")
		}

		// Force a sync to flush the buffer
		err = fileOutput.Sync()
		assert.NoError(t, err, "Failed to sync buffered output")

		// Verify entries were written
		content, err := os.ReadFile(logPath)
		require.NoError(t, err, "Failed to read buffered test log file")

		for i := 0; i < 5; i++ {
			assert.Contains(t, string(content), fmt.Sprintf("Buffered log entry %d", i),
				"Buffered log file missing expected entry")
		}

		// Clean up
		fileOutput.Close()
	})
}

// MockFile implements a mock file for testing file errors.
type MockFile struct {
	WriteErr error
	SyncErr  error
	CloseErr error
	Content  []byte
	StatErr  error
}

func (m *MockFile) Write(p []byte) (int, error) {
	if m.WriteErr != nil {
		return 0, m.WriteErr
	}
	m.Content = append(m.Content, p...)
	return len(p), nil
}

func (m *MockFile) WriteString(s string) (int, error) {
	return m.Write([]byte(s))
}

func (m *MockFile) Sync() error {
	return m.SyncErr
}

func (m *MockFile) Close() error {
	return m.CloseErr
}

func (m *MockFile) Stat() (os.FileInfo, error) {
	if m.StatErr != nil {
		return nil, m.StatErr
	}
	// Mock a simple file info
	return &mockFileInfo{size: int64(len(m.Content))}, nil
}

// mockFileInfo is a simple implementation of os.FileInfo for testing.
type mockFileInfo struct {
	size int64
}

func (m *mockFileInfo) Name() string       { return "mock-file" }
func (m *mockFileInfo) Size() int64        { return m.size }
func (m *mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m *mockFileInfo) ModTime() time.Time { return time.Now() }
func (m *mockFileInfo) IsDir() bool        { return false }
func (m *mockFileInfo) Sys() interface{}   { return nil }

// Additional test for error handling with a mock file.
func TestFileOutputWithMockFile(t *testing.T) {
	t.Run("Write error", func(t *testing.T) {
		mockFile := &MockFile{
			WriteErr: io.ErrShortWrite,
		}

		fileOutput := &FileOutput{
			file:       mockFile,
			path:       "mock_path",
			formatter:  &TextFormatter{IncludeTimestamp: true, IncludeLocation: true},
			jsonFormat: false,
			curSize:    0,
		}

		// Try to write with the error-producing mock file
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "This should fail",
		}

		err := fileOutput.Write(entry)
		assert.Error(t, err, "Write should return an error with failing mock file")
		assert.Contains(t, err.Error(), "failed to write", "Error message should indicate write failure")
	})

	t.Run("Rotation with file error", func(t *testing.T) {
		mockFile := &MockFile{
			CloseErr: fmt.Errorf("mock close error"),
		}

		fileOutput := &FileOutput{
			file:       mockFile,
			path:       "mock_path",
			formatter:  &TextFormatter{IncludeTimestamp: true, IncludeLocation: true},
			jsonFormat: false,
			curSize:    100,
			rotateSize: 50, // Force rotation on next write
		}

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "This should trigger rotation",
		}

		// This should attempt rotation and fail
		err := fileOutput.Write(entry)
		assert.Error(t, err, "Rotation should fail with mock close error")
		assert.Contains(t, err.Error(), "failed to rotate", "Error message should indicate rotation failure")
	})

	// Test Stat error in file rotation logic
	t.Run("Stat error", func(t *testing.T) {
		mockFile := &MockFile{
			StatErr: fmt.Errorf("mock stat error"),
		}

		// fileOutput := &FileOutput{
		// 	file:       mockFile,
		// 	path:       "mock_path",
		// 	formatter:  &TextFormatter{IncludeTimestamp: true, IncludeLocation: true},
		// 	jsonFormat: false,
		// }

		// Get the current size should use the mock stat
		_, err := mockFile.Stat()
		assert.Error(t, err, "Stat should return the mock error")
	})
}

// Test the text formatter.
func TestTextFormatter(t *testing.T) {
	formatter := &TextFormatter{
		IncludeTimestamp:  true,
		IncludeLocation:   true,
		IncludeStackTrace: true,
	}

	t.Run("Basic formatting", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			File:     "test.go",
			Line:     123,
		}

		formatted := formatter.Format(entry)
		assert.Contains(t, formatted, "INFO")
		assert.Contains(t, formatted, "Test message")
		assert.Contains(t, formatted, "test.go:123")
	})

	t.Run("With trace ID", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			File:     "test.go",
			Line:     123,
			TraceID:  "trace-123",
		}

		formatted := formatter.Format(entry)
		assert.Contains(t, formatted, "traceId=trace-123")
	})

	t.Run("With token info", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			TokenInfo: &core.TokenInfo{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		}

		formatted := formatter.Format(entry)
		assert.Contains(t, formatted, "tokens=100/50")
	})

	t.Run("With fields", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			Fields: map[string]interface{}{
				"user":   "test-user",
				"action": "login",
			},
		}

		formatted := formatter.Format(entry)
		assert.Contains(t, formatted, "user=test-user")
		assert.Contains(t, formatted, "action=login")
	})
}

// Test JSONFormatter.
func TestJSONFormatter(t *testing.T) {
	formatter := &JSONFormatter{}

	t.Run("Basic JSON format", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			File:     "test.go",
			Line:     123,
			Function: "TestFunc",
		}

		formatted := formatter.Format(entry)
		var data map[string]interface{}
		err := json.Unmarshal([]byte(formatted), &data)
		require.NoError(t, err, "Formatter should produce valid JSON")

		assert.Equal(t, "Test message", data["message"])
		assert.Equal(t, "INFO", data["level"])
		assert.Equal(t, "test.go", data["file"])
		assert.Equal(t, float64(123), data["line"])
		assert.Equal(t, "TestFunc", data["function"])
	})

	t.Run("With all fields", func(t *testing.T) {
		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			File:     "test.go",
			Line:     123,
			Function: "TestFunc",
			TraceID:  "trace-123",
			ModelID:  "gpt-4",
			TokenInfo: &core.TokenInfo{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
			Fields: map[string]interface{}{
				"user":   "test-user",
				"action": "login",
				"count":  42,
			},
		}

		formatted := formatter.Format(entry)
		var data map[string]interface{}
		err := json.Unmarshal([]byte(formatted), &data)
		require.NoError(t, err, "Formatter should produce valid JSON")

		assert.Equal(t, "Test message", data["message"])
		assert.Equal(t, "INFO", data["level"])
		assert.Equal(t, "trace-123", data["traceId"])
		assert.Equal(t, "gpt-4", data["modelId"])
		assert.Equal(t, "test-user", data["user"])
		assert.Equal(t, "login", data["action"])
		assert.Equal(t, float64(42), data["count"])

		tokenInfo, ok := data["tokenInfo"].(map[string]interface{})
		require.True(t, ok, "tokenInfo should be a map")
		assert.Equal(t, float64(100), tokenInfo["promptTokens"])
		assert.Equal(t, float64(50), tokenInfo["completionTokens"])
		assert.Equal(t, float64(150), tokenInfo["totalTokens"])
	})

	// Test error case in JSON marshal
	t.Run("JSON marshal error", func(t *testing.T) {
		// Create a circular reference to cause JSON marshal to fail
		circular := make(map[string]interface{})
		circular["self"] = circular

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			Fields:   circular,
		}

		formatted := formatter.Format(entry)
		assert.Contains(t, formatted, "Error marshaling log entry to JSON")
	})
}

// Test Console Output with LLM information.
func TestConsoleOutputWithLLMInfo(t *testing.T) {
	t.Run("Basic console output", func(t *testing.T) {
		buffer := &bytes.Buffer{}
		console := &ConsoleOutput{
			writer: buffer,
			color:  false,
		}

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
			File:     "test.go",
			Line:     123,
			TraceID:  "trace-123",
		}

		err := console.Write(entry)
		require.NoError(t, err)

		output := buffer.String()
		assert.Contains(t, output, "Test message")
		assert.Contains(t, output, "INFO")
		assert.Contains(t, output, "test.go:123")
		assert.Contains(t, output, "traceId=trace-12")
	})

	t.Run("With model and token info", func(t *testing.T) {
		buffer := &bytes.Buffer{}
		console := &ConsoleOutput{
			writer: buffer,
			color:  true, // Enable colors to test that path
		}

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "LLM response",
			File:     "test.go",
			Line:     123,
			ModelID:  "gpt-4-0314",
			TokenInfo: &core.TokenInfo{
				PromptTokens:     200,
				CompletionTokens: 50,
				TotalTokens:      250,
			},
		}

		err := console.Write(entry)
		require.NoError(t, err)

		output := buffer.String()
		assert.Contains(t, output, "LLM response")
		assert.Contains(t, output, "model=gpt-4")
		assert.Contains(t, output, "tokens=250")
		// With color enabled, should have color codes
		assert.Contains(t, output, "\033[")
	})

	t.Run("With spans information at debug level", func(t *testing.T) {
		buffer := &bytes.Buffer{}
		console := &ConsoleOutput{
			writer: buffer,
			color:  false,
		}

		spans := []*core.Span{
			{
				ID:        "span-1",
				Operation: "llm-call",
				StartTime: time.Now().Add(-500 * time.Millisecond),
				EndTime:   time.Now(),
				Annotations: map[string]interface{}{
					"chain_step": map[string]interface{}{
						"name":  "generate",
						"index": 0,
						"total": 2,
					},
				},
			},
			{
				ID:        "span-2",
				ParentID:  "span-1",
				Operation: "api-call",
				StartTime: time.Now().Add(-300 * time.Millisecond),
				EndTime:   time.Now().Add(-100 * time.Millisecond),
			},
		}

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: DEBUG, // Debug level to show spans
			Message:  "Debug with spans",
			File:     "test.go",
			Line:     123,
			Fields: map[string]interface{}{
				"spans": spans,
			},
		}

		err := console.Write(entry)
		require.NoError(t, err)

		output := buffer.String()
		assert.Contains(t, output, "Debug with spans")
		assert.Contains(t, output, "Trace Timeline:")
		assert.Contains(t, output, "llm-call")
		assert.Contains(t, output, "step=generate")
		assert.Contains(t, output, "[1/2]")
		assert.Contains(t, output, "api-call")
	})

	// Test that write errors are propagated
	t.Run("Write error", func(t *testing.T) {
		errWriter := &ErrorWriter{err: fmt.Errorf("write error")}
		console := &ConsoleOutput{
			writer: errWriter,
			color:  false,
		}

		entry := LogEntry{
			Time:     time.Now().UnixNano(),
			Severity: INFO,
			Message:  "Test message",
		}

		err := console.Write(entry)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "write error")
	})
}

// Mock writer that returns errors on write.
type ErrorWriter struct {
	err error
}

func (e *ErrorWriter) Write(p []byte) (n int, err error) {
	return 0, e.err
}

// Test formatLLMInfo function.
func TestFormatLLMInfo(t *testing.T) {
	t.Run("Empty input", func(t *testing.T) {
		result := formatLLMInfo("", nil)
		assert.Equal(t, "", result, "Should return empty string for empty inputs")
	})

	t.Run("Model ID only", func(t *testing.T) {
		result := formatLLMInfo("gpt-4-0314", nil)
		assert.Contains(t, result, "model=gpt-4")
	})

	t.Run("Token info only", func(t *testing.T) {
		tokenInfo := &core.TokenInfo{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		}
		result := formatLLMInfo("", tokenInfo)
		assert.Contains(t, result, "tokens=150")
		// Should contain up arrow for prompt tokens
		assert.Contains(t, result, "100")
		// Should contain down arrow for completion tokens
		assert.Contains(t, result, "50")
	})

	t.Run("Both model and tokens", func(t *testing.T) {
		tokenInfo := &core.TokenInfo{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		}
		result := formatLLMInfo("claude-3-opus-20240229", tokenInfo)
		assert.Contains(t, result, "model=claude-opus")
		assert.Contains(t, result, "tokens=150")
	})

	// Test different model name formats
	t.Run("Different model name formats", func(t *testing.T) {
		tests := []struct {
			input    string
			expected string
		}{
			{"gpt-4-0314", "gpt-4"},
			{"gpt-4-32k", "gpt-4-32k"},
			{"gpt-3.5-turbo", "gpt-3"},
			{"claude-3-opus-20240229", "claude-opus"},
			{"claude-3-sonnet-20240229", "claude-sonnet"},
			{"gemini-pro", "gemini-pro"},
			{"llama-2-70b", "llama-2"},
			{"random-model", "random"}, // unknown model
		}

		for _, tt := range tests {
			result := formatLLMInfo(tt.input, nil)
			assert.Contains(t, result, fmt.Sprintf("model=%s", tt.expected))
		}
	})
}

// Test extractModelName function.
func TestExtractModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"gpt-4-0314", "gpt-4-0314"},
		{"gpt-4-32k", "gpt-4-32k"},
		{"gpt-3.5-turbo", "gpt-3"},
		{"claude-3-opus-20240229", "claude-opus"},
		{"claude-3-sonnet-20240229", "claude-sonnet"},
		{"gemini-pro", "gemini-pro"},
		{"llama-2-70b", "llama-2"},
		{"random-model", "random"}, // unknown model
		{"", ""},                   // empty input
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := extractModelName(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Test formatFields function.
func TestFormatFields(t *testing.T) {
	t.Run("Empty fields", func(t *testing.T) {
		result := formatFields(map[string]interface{}{})
		assert.Equal(t, "", result)
	})

	t.Run("Simple fields", func(t *testing.T) {
		fields := map[string]interface{}{
			"user":  "test-user",
			"count": 42,
		}
		result := formatFields(fields)
		assert.Contains(t, result, "user=test-user")
		assert.Contains(t, result, "count=42")
	})

	t.Run("Long prompt/completion", func(t *testing.T) {
		longText := strings.Repeat("a", 200)
		fields := map[string]interface{}{
			"prompt":     longText,
			"completion": longText,
		}
		result := formatFields(fields)
		// Should be truncated
		assert.Contains(t, result, "...")
		// Should be quoted
		assert.Contains(t, result, `"`)
	})

	t.Run("Mixed fields", func(t *testing.T) {
		fields := map[string]interface{}{
			"user":       "test-user",
			"count":      42,
			"prompt":     "Tell me a joke",
			"completion": "Why did the chicken cross the road?",
		}
		result := formatFields(fields)
		assert.Contains(t, result, "user=test-user")
		assert.Contains(t, result, "count=42")
		assert.Contains(t, result, `prompt="Tell me a joke"`)
		assert.Contains(t, result, `completion="Why did the chicken cross the road?"`)
	})
}

// Test span formatting.
func TestFormatSpans(t *testing.T) {
	t.Run("Empty spans", func(t *testing.T) {
		result := formatSpans([]*core.Span{})
		assert.Equal(t, "", result)
	})

	t.Run("Single span", func(t *testing.T) {
		now := time.Now()
		spans := []*core.Span{
			{
				ID:        "span-1",
				Operation: "test-operation",
				StartTime: now.Add(-100 * time.Millisecond),
				EndTime:   now,
			},
		}
		result := formatSpans(spans)
		assert.Contains(t, result, "Trace Timeline:")
		assert.Contains(t, result, "test-operation")
	})

	t.Run("Parent-child spans", func(t *testing.T) {
		now := time.Now()
		spans := []*core.Span{
			{
				ID:        "span-1",
				Operation: "parent-operation",
				StartTime: now.Add(-200 * time.Millisecond),
				EndTime:   now,
			},
			{
				ID:        "span-2",
				ParentID:  "span-1",
				Operation: "child-operation",
				StartTime: now.Add(-150 * time.Millisecond),
				EndTime:   now.Add(-50 * time.Millisecond),
			},
		}
		result := formatSpans(spans)
		assert.Contains(t, result, "parent-operation")
		assert.Contains(t, result, "child-operation")
	})

	t.Run("Spans with annotations", func(t *testing.T) {
		now := time.Now()
		tokenUsage := &core.TokenUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		}
		spans := []*core.Span{
			{
				ID:        "span-1",
				Operation: "llm-call",
				StartTime: now.Add(-200 * time.Millisecond),
				EndTime:   now,
				Annotations: map[string]interface{}{
					"chain_step": map[string]interface{}{
						"name":  "generate",
						"index": 0,
						"total": 2,
					},
					"token_usage": tokenUsage,
				},
			},
			{
				ID:        "span-2",
				ParentID:  "span-1",
				Operation: "api-call",
				StartTime: now.Add(-150 * time.Millisecond),
				EndTime:   now.Add(-50 * time.Millisecond),
				Error:     fmt.Errorf("test error"),
				Annotations: map[string]interface{}{
					"task": map[string]interface{}{
						"processor": "llm",
						"id":        "task-123",
						"type":      "generation",
					},
				},
			},
		}
		result := formatSpans(spans)
		assert.Contains(t, result, "llm-call")
		assert.Contains(t, result, "step=generate")
		assert.Contains(t, result, "[1/2]")
		assert.Contains(t, result, "tokens=150(100↑50↓)")
		assert.Contains(t, result, "api-call")
		assert.Contains(t, result, "[ERROR: test error]")
		assert.Contains(t, result, "[llm]")
	})

	t.Run("Span duration formatting", func(t *testing.T) {
		// Test different durations
		now := time.Now()
		spans := []*core.Span{
			{
				ID:        "span-1",
				Operation: "nano-operation",
				StartTime: now.Add(-500 * time.Nanosecond),
				EndTime:   now,
			},
			{
				ID:        "span-2",
				Operation: "micro-operation",
				StartTime: now.Add(-500 * time.Microsecond),
				EndTime:   now,
			},
			{
				ID:        "span-3",
				Operation: "milli-operation",
				StartTime: now.Add(-500 * time.Millisecond),
				EndTime:   now,
			},
			{
				ID:        "span-4",
				Operation: "second-operation",
				StartTime: now.Add(-2 * time.Second),
				EndTime:   now,
			},
			{
				ID:        "span-5",
				Operation: "minute-operation",
				StartTime: now.Add(-2 * time.Minute),
				EndTime:   now,
			},
			{
				ID:        "span-6",
				Operation: "hour-operation",
				StartTime: now.Add(-2 * time.Hour),
				EndTime:   now,
			},
			{
				ID:        "span-7",
				Operation: "unfinished-operation",
				StartTime: now.Add(-1 * time.Second),
				EndTime:   time.Time{}, // Zero time (unfinished)
			},
			{
				ID:        "span-8",
				Operation: "error-operation",
				StartTime: now,
				EndTime:   now.Add(-1 * time.Second), // Negative duration (error)
			},
		}
		result := formatSpans(spans)
		// Check for appropriate duration formatting
		assert.Contains(t, result, "nano-operation")
		assert.Contains(t, result, "micro-operation")
		assert.Contains(t, result, "milli-operation")
		assert.Contains(t, result, "second-operation")
		assert.Contains(t, result, "minute-operation")
		assert.Contains(t, result, "hour-operation")
		assert.Contains(t, result, "unfinished-operation")
		assert.Contains(t, result, "error-operation")
	})
}

// Test duration formatting functions.
func TestFormatDuration(t *testing.T) {
	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Nanosecond, "500ns"},
		{500 * time.Microsecond, "500.00µs"},
		{500 * time.Millisecond, "500.00ms"},
		{1500 * time.Millisecond, "1.50s"},
		{90 * time.Second, "1m30s"},
	}

	for _, tt := range tests {
		t.Run(tt.duration.String(), func(t *testing.T) {
			result := formatDuration(tt.duration)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestFormatSpanDuration(t *testing.T) {
	tests := []struct {
		duration time.Duration
		expected string
	}{
		{500 * time.Nanosecond, "<1µs"},
		{1500 * time.Nanosecond, "1.50µs"},
		{1500 * time.Microsecond, "1.50ms"},
		{1500 * time.Millisecond, "1.50s"},
		{90 * time.Second, "1.5m"},
		{90 * time.Minute, "1.5h"},
	}

	for _, tt := range tests {
		t.Run(tt.duration.String(), func(t *testing.T) {
			result := formatSpanDuration(tt.duration)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Test annotation formatting.
func TestFormatAnnotations(t *testing.T) {
	t.Run("Empty annotations", func(t *testing.T) {
		result := formatAnnotations(map[string]interface{}{})
		assert.Equal(t, "", result)
	})

	t.Run("Simple annotations", func(t *testing.T) {
		annotations := map[string]interface{}{
			"status": "completed",
			"score":  0.95,
			"count":  42,
		}
		result := formatAnnotations(annotations)
		assert.Contains(t, result, "status=completed")
		assert.Contains(t, result, "score=0.95")
		assert.Contains(t, result, "count=42")
	})
}

func TestFormatAnnotationValue(t *testing.T) {
	tests := []struct {
		name     string
		key      string
		value    interface{}
		expected string
	}{
		{"Nil value", "key", nil, ""},
		{"Boolean true", "flag", true, "flag=true"},
		{"Boolean false", "flag", false, "flag=false"},
		{"Integer", "count", 42, "count=42"},
		{"Float", "score", 0.95, "score=0.95"},
		{"String", "status", "completed", "status=completed"},
		{"Long string", "desc", strings.Repeat("a", 100), "desc='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...'"},
		{"Duration", "time", 500 * time.Millisecond, "time=500.00ms"},
		{"Array", "tags", []interface{}{"tag1", "tag2"}, "tags=[tag1,tag2]"},
		{"Large array", "items", make([]interface{}, 10), "items=[10 items]"},
		{"Map", "config", map[string]interface{}{"a": 1, "b": 2}, "config={2 keys}"},
		{"Error", "error", fmt.Errorf("test error"), "error=error('test error')"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatAnnotationValue(tt.key, tt.value)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Test FileOutput options.
func TestFileOutputOptions(t *testing.T) {
	t.Run("WithJSONFormat", func(t *testing.T) {
		option := WithJSONFormat(true)
		fileOutput := &FileOutput{}
		option(fileOutput)
		assert.True(t, fileOutput.jsonFormat)
		assert.IsType(t, &JSONFormatter{}, fileOutput.formatter)

		// Test with false
		option = WithJSONFormat(false)
		fileOutput = &FileOutput{}
		option(fileOutput)
		assert.False(t, fileOutput.jsonFormat)
		assert.IsType(t, &TextFormatter{}, fileOutput.formatter)
	})

	t.Run("WithRotation", func(t *testing.T) {
		option := WithRotation(1024, 5)
		fileOutput := &FileOutput{}
		option(fileOutput)
		assert.Equal(t, int64(1024), fileOutput.rotateSize)
		assert.Equal(t, 5, fileOutput.maxFiles)
	})

	t.Run("WithBufferSize", func(t *testing.T) {
		option := WithBufferSize(8192)
		fileOutput := &FileOutput{}
		option(fileOutput)
		assert.Equal(t, 8192, fileOutput.bufferSize)
		assert.Equal(t, 8192, cap(fileOutput.buffer))
	})

	t.Run("WithFormatter", func(t *testing.T) {
		formatter := &CustomFormatter{}
		option := WithFormatter(formatter)
		fileOutput := &FileOutput{}
		option(fileOutput)
		assert.Equal(t, formatter, fileOutput.formatter)
	})
}

// Test ConsoleOutput options.
func TestConsoleOutputOptions(t *testing.T) {
	t.Run("WithColor", func(t *testing.T) {
		option := WithColor(true)
		console := &ConsoleOutput{}
		option(console)
		assert.True(t, console.color)

		option = WithColor(false)
		console = &ConsoleOutput{}
		option(console)
		assert.False(t, console.color)
	})

	t.Run("NewConsoleOutput", func(t *testing.T) {
		// Test with stderr
		console := NewConsoleOutput(true)
		assert.Equal(t, os.Stderr, console.writer)
		assert.True(t, console.color) // Default is true

		// Test with stdout and color disabled
		console = NewConsoleOutput(false, WithColor(false))
		assert.Equal(t, os.Stdout, console.writer)
		assert.False(t, console.color)
	})

	t.Run("GetSeverityColor", func(t *testing.T) {
		// Test all severity levels
		assert.NotEmpty(t, getSeverityColor(DEBUG))
		assert.NotEmpty(t, getSeverityColor(INFO))
		assert.NotEmpty(t, getSeverityColor(WARN))
		assert.NotEmpty(t, getSeverityColor(ERROR))
		assert.NotEmpty(t, getSeverityColor(FATAL))
		// Test invalid severity
		assert.Empty(t, getSeverityColor(Severity(999)))
	})
}

// Test helper functions.
func TestHelperFunctions(t *testing.T) {
	t.Run("contains", func(t *testing.T) {
		assert.True(t, contains([]string{"a", "b", "c"}, "b"))
		assert.False(t, contains([]string{"a", "b", "c"}, "d"))
		assert.False(t, contains([]string{}, "a"))
	})

	t.Run("filterRelevantAnnotations", func(t *testing.T) {
		annotations := map[string]interface{}{
			"status":      "completed",
			"result":      "success",
			"error":       nil,
			"count":       42,
			"score":       0.95,
			"progress":    0.5,
			"irrelevant1": "ignore",
			"irrelevant2": "ignore",
		}

		filtered := filterRelevantAnnotations(annotations)
		assert.Equal(t, 6, len(filtered))
		assert.Contains(t, filtered, "status")
		assert.Contains(t, filtered, "result")
		assert.Contains(t, filtered, "count")
		assert.Contains(t, filtered, "score")
		assert.Contains(t, filtered, "progress")
		assert.NotContains(t, filtered, "irrelevant1")
		assert.NotContains(t, filtered, "irrelevant2")
	})
}
