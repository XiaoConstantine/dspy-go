// Package rlm provides a native Recursive Language Model implementation for dspy-go.
// RLM enables LLMs to explore large contexts programmatically through a Go REPL,
// making iterative queries to sub-LLMs until a final answer is reached.
package rlm

import "time"

// Config holds RLM configuration.
type Config struct {
	// MaxIterations is the maximum number of iteration loops (default: 30).
	MaxIterations int

	// Verbose enables verbose logging.
	Verbose bool

	// Timeout is the maximum duration for the entire RLM completion.
	// Zero means no timeout (default).
	Timeout time.Duration

	// TraceDir is the directory for RLM trace logs (JSONL format compatible with rlm-viewer).
	// Empty string disables tracing.
	TraceDir string

	// HistoryCompression configures incremental history compression.
	// When enabled, older iterations are summarized to reduce context size.
	HistoryCompression *HistoryCompressionConfig

	// AdaptiveIteration configures adaptive iteration strategy.
	// When enabled, max iterations are dynamically calculated based on context size.
	AdaptiveIteration *AdaptiveIterationConfig

	// OnProgress is called at the start of each iteration with progress info.
	// Can be used to display progress to users or implement custom termination logic.
	OnProgress func(progress IterationProgress)
}

// HistoryCompressionConfig configures how message history is compressed.
type HistoryCompressionConfig struct {
	// Enabled turns on history compression (default: false).
	Enabled bool

	// VerbatimIterations is the number of recent iterations to keep verbatim.
	// Older iterations will be summarized. Default: 3.
	VerbatimIterations int

	// MaxSummaryTokens is the approximate maximum tokens for summarized history.
	// Default: 500.
	MaxSummaryTokens int
}

// AdaptiveIterationConfig configures adaptive iteration behavior.
type AdaptiveIterationConfig struct {
	// Enabled turns on adaptive iteration (default: false).
	Enabled bool

	// BaseIterations is the base number of iterations before context scaling.
	// Default: 10.
	BaseIterations int

	// MaxIterations caps the total iterations regardless of context size.
	// Default: 50.
	MaxIterations int

	// ContextScaleFactor determines how much context size increases iterations.
	// iterations = BaseIterations + (contextSize / ContextScaleFactor)
	// Default: 100000 (100KB per additional iteration).
	ContextScaleFactor int

	// EnableEarlyTermination allows early exit when model signals confidence.
	// Default: true.
	EnableEarlyTermination bool

	// ConfidenceThreshold is the number of confidence signals needed for early termination.
	// Default: 1.
	ConfidenceThreshold int

	// ConfidenceDetector is a custom function to detect confidence signals in responses.
	// If nil, a default heuristic based on FINAL markers is used.
	// The function returns true if the response indicates high confidence.
	ConfidenceDetector func(response string) bool
}

// IterationProgress tracks progress and confidence during iteration.
type IterationProgress struct {
	// CurrentIteration is the current iteration number (1-indexed).
	CurrentIteration int

	// MaxIterations is the computed maximum iterations for this request.
	MaxIterations int

	// ConfidenceSignals counts how many times the model has signaled confidence.
	ConfidenceSignals int

	// HasFinalAttempt indicates the model tried to give a final answer.
	HasFinalAttempt bool

	// ContextSize is the size of the input context in bytes.
	ContextSize int
}

// DefaultConfig returns the default RLM configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations: 30,
		Verbose:       false,
		Timeout:       0,
		TraceDir:      "",
	}
}

// Option configures the RLM.
type Option func(*Config)

// WithMaxIterations sets the maximum number of iterations.
// Values <= 0 are ignored and the default is used.
func WithMaxIterations(n int) Option {
	return func(c *Config) {
		if n > 0 {
			c.MaxIterations = n
		}
	}
}

// WithVerbose enables verbose logging.
func WithVerbose(v bool) Option {
	return func(c *Config) {
		c.Verbose = v
	}
}

// WithTimeout sets the maximum duration for the completion.
func WithTimeout(d time.Duration) Option {
	return func(c *Config) {
		c.Timeout = d
	}
}

// WithTraceDir enables JSONL tracing to the specified directory.
// The trace files are compatible with rlm-go's rlm-viewer command.
func WithTraceDir(dir string) Option {
	return func(c *Config) {
		c.TraceDir = dir
	}
}

// WithHistoryCompression enables incremental history compression.
// verbatimIterations is how many recent iterations to keep in full (default: 3).
// maxSummaryTokens is the approximate max tokens for the summary (default: 500).
func WithHistoryCompression(verbatimIterations, maxSummaryTokens int) Option {
	return func(c *Config) {
		if verbatimIterations <= 0 {
			verbatimIterations = 3
		}
		if maxSummaryTokens <= 0 {
			maxSummaryTokens = 500
		}
		c.HistoryCompression = &HistoryCompressionConfig{
			Enabled:            true,
			VerbatimIterations: verbatimIterations,
			MaxSummaryTokens:   maxSummaryTokens,
		}
	}
}

// WithAdaptiveIteration enables adaptive iteration strategy with default configuration.
// This dynamically adjusts max iterations based on context size and enables
// early termination when the model signals confidence.
func WithAdaptiveIteration() Option {
	return func(c *Config) {
		c.AdaptiveIteration = &AdaptiveIterationConfig{
			Enabled:                true,
			BaseIterations:         10,
			MaxIterations:          50,
			ContextScaleFactor:     100000, // 100KB per additional iteration
			EnableEarlyTermination: true,
			ConfidenceThreshold:    1,
		}
	}
}

// WithAdaptiveIterationConfig enables adaptive iteration with custom configuration.
func WithAdaptiveIterationConfig(cfg AdaptiveIterationConfig) Option {
	return func(c *Config) {
		cfg.Enabled = true
		if cfg.BaseIterations <= 0 {
			cfg.BaseIterations = 10
		}
		if cfg.MaxIterations <= 0 {
			cfg.MaxIterations = 50
		}
		if cfg.ContextScaleFactor <= 0 {
			cfg.ContextScaleFactor = 100000
		}
		if cfg.ConfidenceThreshold <= 0 {
			cfg.ConfidenceThreshold = 1
		}
		c.AdaptiveIteration = &cfg
	}
}

// WithProgressHandler sets a callback for iteration progress updates.
func WithProgressHandler(handler func(IterationProgress)) Option {
	return func(c *Config) {
		c.OnProgress = handler
	}
}
