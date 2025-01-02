package logging

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// ConsoleOutput formats logs for human readability
type ConsoleOutput struct {
	mu     sync.Mutex
	writer *os.File
}

func (o *ConsoleOutput) Write(e LogEntry) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Format for easy reading
	fmt.Fprintf(o.writer, "[%s] %-5s %s:%d %s",
		time.Unix(0, e.Time).Format("15:04:05.000"),
		e.Severity,
		e.File,
		e.Line,
		e.Message,
	)

	// Add LLM-specific information if present
	if e.ModelID != "" {
		fmt.Fprintf(o.writer, " [model=%s]", e.ModelID)
	}
	if e.TokenInfo != nil {
		fmt.Fprintf(o.writer, " [tokens=%d]", e.TokenInfo.TotalTokens)
	}

	fmt.Fprintln(o.writer)
	return nil
}

// JSONOutput writes structured logs for machine processing
type JSONOutput struct {
	mu     sync.Mutex
	writer *os.File
}

func (o *JSONOutput) Write(e LogEntry) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	return json.NewEncoder(o.writer).Encode(e)
}
