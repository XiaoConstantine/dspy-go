package logging

import (
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
