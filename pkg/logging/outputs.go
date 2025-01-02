package logging

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// ConsoleOutput formats logs for human readability.
type ConsoleOutput struct {
	mu     sync.Mutex
	writer io.Writer
	color  bool // Whether to use ANSI color codes
}

type ConsoleOutputOption func(*ConsoleOutput)

func WithColor(enabled bool) ConsoleOutputOption {
	return func(c *ConsoleOutput) {
		c.color = enabled
	}
}

func NewConsoleOutput(useStderr bool, opts ...ConsoleOutputOption) *ConsoleOutput {
	// Choose the appropriate writer based on useStderr flag
	writer := os.Stdout
	if useStderr {
		writer = os.Stderr
	}

	// Create the base console output
	c := &ConsoleOutput{
		writer: writer,
		color:  true, // Enable colors by default
	}

	// Apply any provided options
	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Helper function to get ANSI color codes for different severity levels.
func getSeverityColor(s Severity) string {
	switch s {
	case DEBUG:
		return "\033[37m" // Gray
	case INFO:
		return "\033[32m" // Green
	case WARN:
		return "\033[33m" // Yellow
	case ERROR:
		return "\033[31m" // Red
	case FATAL:
		return "\033[35m" // Magenta
	default:
		return ""
	}
}

func formatFields(fields map[string]interface{}) string {
	if len(fields) == 0 {
		return ""
	}

	var result string
	for k, v := range fields {
		// Handle special fields like prompts and completions
		if k == "prompt" || k == "completion" {
			// Truncate long text for console display
			str := fmt.Sprintf("%v", v)
			if len(str) > 100 {
				str = str[:97] + "..."
			}
			result += fmt.Sprintf("%s=%q ", k, str)
		} else {
			result += fmt.Sprintf("%s=%v ", k, v)
		}
	}

	return result
}

func (o *ConsoleOutput) Write(e LogEntry) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	timestamp := time.Unix(0, e.Time).Format("2006-01-02 15:04:05.000")

	var levelColor, resetColor string
	if o.color {
		levelColor = getSeverityColor(e.Severity)
		resetColor = "\033[0m"
	}

	// Format for easy reading
	basic := fmt.Sprintf("%s %s%-5s%s [%s:%d] %s",
		timestamp,
		levelColor,
		e.Severity,
		resetColor,
		e.File,
		e.Line,
		e.Message,
	)

	// Add LLM-specific information if present
	if e.ModelID != "" {
		basic += fmt.Sprintf(" [model=%s]", e.ModelID)
	}

	if e.TokenInfo != nil {
		basic += fmt.Sprintf(" [tokens=%d]", e.TokenInfo.TotalTokens)
	}
	// Add structured fields if any exist
	if len(e.Fields) > 0 {
		fields := formatFields(e.Fields)
		basic += " " + fields
	}

	_, err := fmt.Fprintln(o.writer, basic)

	return err
}

func (c *ConsoleOutput) Sync() error {
	if syncer, ok := c.writer.(interface{ Sync() error }); ok {
		return syncer.Sync()
	}
	return nil
}

// Close cleans up any resources.
func (c *ConsoleOutput) Close() error {
	if closer, ok := c.writer.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
