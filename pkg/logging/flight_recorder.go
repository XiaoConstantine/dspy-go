// Package logging provides FlightRecorder integration for Go 1.25+.
// FlightRecorder enables lightweight, always-on tracing suitable for production.
package logging

import (
	"context"
	"os"
	"runtime/trace"
	"sync"
	"time"
)

// FlightRecorder wraps Go 1.25's runtime/trace.FlightRecorder for production diagnostics.
// It maintains a ring buffer of recent trace data that can be dumped on-demand
// (e.g., when an error occurs or performance degrades).
//
// Usage:
//
//	fr := NewFlightRecorder(WithMinAge(10 * time.Second))
//	fr.Start()
//	defer fr.Stop()
//
//	// When an interesting event occurs:
//	fr.Snapshot("error_occurred.trace")
type FlightRecorder struct {
	recorder *trace.FlightRecorder
	mu       sync.Mutex
	running  bool
	config   trace.FlightRecorderConfig
}

// FlightRecorderOption configures a FlightRecorder.
type FlightRecorderOption func(*FlightRecorder)

// WithMinAge sets the minimum age of events to keep in the trace buffer.
// Default is 10 seconds. Longer durations capture more history but use more memory.
func WithMinAge(d time.Duration) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.config.MinAge = d
	}
}

// WithMaxBytes sets the maximum size of the trace buffer in bytes.
// This takes precedence over MinAge. If 0, the maximum is implementation defined.
func WithMaxBytes(n uint64) FlightRecorderOption {
	return func(fr *FlightRecorder) {
		fr.config.MaxBytes = n
	}
}

// NewFlightRecorder creates a new FlightRecorder with the given options.
// The FlightRecorder uses Go 1.25's runtime/trace.FlightRecorder which maintains
// a ring buffer of trace data with minimal overhead (~1% CPU).
func NewFlightRecorder(opts ...FlightRecorderOption) *FlightRecorder {
	fr := &FlightRecorder{
		config: trace.FlightRecorderConfig{
			MinAge: 10 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(fr)
	}

	fr.recorder = trace.NewFlightRecorder(fr.config)

	return fr
}

// Start begins recording trace data into the ring buffer.
// This is safe to call in production with minimal overhead.
func (fr *FlightRecorder) Start() error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if fr.running {
		return nil
	}

	if err := fr.recorder.Start(); err != nil {
		return err
	}

	fr.running = true
	return nil
}

// Stop stops recording trace data.
func (fr *FlightRecorder) Stop() {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.running {
		return
	}

	fr.recorder.Stop()
	fr.running = false
}

// Enabled returns true if the flight recorder is currently running.
func (fr *FlightRecorder) Enabled() bool {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	return fr.running && fr.recorder.Enabled()
}

// Snapshot writes the current trace buffer to a file.
// Call this when an interesting event occurs (error, slow request, etc.)
// to capture what happened leading up to that moment.
func (fr *FlightRecorder) Snapshot(filename string) error {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if !fr.running {
		return nil
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = fr.recorder.WriteTo(f)
	return err
}

// SnapshotOnError is a helper that takes a snapshot when an error occurs.
// Returns the original error for chaining.
//
// Example:
//
//	result, err := module.Process(ctx, inputs)
//	if err != nil {
//	    return fr.SnapshotOnError(err, "module_error.trace")
//	}
func (fr *FlightRecorder) SnapshotOnError(err error, filename string) error {
	if err != nil {
		fr.Snapshot(filename)
	}
	return err
}

// Global FlightRecorder instance for convenience.
var globalRecorder *FlightRecorder
var globalRecorderOnce sync.Once

// GlobalFlightRecorder returns a shared FlightRecorder instance.
// Initialize with InitGlobalFlightRecorder before use.
func GlobalFlightRecorder() *FlightRecorder {
	return globalRecorder
}

// InitGlobalFlightRecorder initializes and starts the global FlightRecorder.
// Safe to call multiple times; only the first call has effect.
func InitGlobalFlightRecorder(opts ...FlightRecorderOption) {
	globalRecorderOnce.Do(func() {
		globalRecorder = NewFlightRecorder(opts...)
		globalRecorder.Start()
	})
}

// TraceRegion wraps trace.WithRegion for convenient span creation.
// Use this to annotate important code sections in traces.
//
// Example:
//
//	defer TraceRegion(ctx, "ProcessModule")()
func TraceRegion(ctx context.Context, name string) func() {
	region := trace.StartRegion(ctx, name)
	return region.End
}

// TraceTask creates a trace task for tracking high-level operations.
// Returns a new context and an end function.
//
// Example:
//
//	ctx, endTask := TraceTask(ctx, "Optimization")
//	defer endTask()
func TraceTask(ctx context.Context, name string) (context.Context, func()) {
	ctx, task := trace.NewTask(ctx, name)
	return ctx, task.End
}

// TraceLog logs a message to the trace at the current point.
// Useful for marking significant events in the trace timeline.
func TraceLog(ctx context.Context, category, message string) {
	trace.Log(ctx, category, message)
}
