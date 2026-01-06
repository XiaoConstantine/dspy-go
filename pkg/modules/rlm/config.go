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
}

// DefaultConfig returns the default RLM configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations: 30,
		Verbose:       false,
		Timeout:       0,
	}
}

// Option configures the RLM.
type Option func(*Config)

// WithMaxIterations sets the maximum number of iterations.
func WithMaxIterations(n int) Option {
	return func(c *Config) {
		c.MaxIterations = n
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
