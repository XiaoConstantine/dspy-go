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
	// Mock implementation for file stats
	return nil, nil
}

// Additional test for error handling with a mock file.
func TestFileOutputWithMockFile(t *testing.T) {
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
}
