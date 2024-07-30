package core

import (
	"context"
	"log"
	"sync"
	"time"
)

// Trace represents the execution trace of a single module or the entire program.
type Trace struct {
	ModuleName string
	ModuleType string
	Inputs     map[string]interface{}
	Outputs    map[string]interface{}
	StartTime  time.Time
	Duration   time.Duration
	Subtraces  []*Trace
	Parent     *Trace
	mu         sync.Mutex

	Error error
}

// NewTrace creates a new Trace instance.
func NewTrace(moduleName, moduleType string) *Trace {
	return &Trace{
		ModuleName: moduleName,
		ModuleType: moduleType,
		Inputs:     make(map[string]interface{}),
		Outputs:    make(map[string]interface{}),
		StartTime:  time.Now(),
		Subtraces:  []*Trace{},
	}
}

// SetInputs sets the inputs for the trace.
func (t *Trace) SetInputs(inputs map[string]interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Inputs = inputs
}

// SetOutputs sets the outputs for the trace and calculates the duration.
func (t *Trace) SetOutputs(outputs map[string]interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Outputs = outputs
	t.Duration = time.Since(t.StartTime)
}

// AddSubtrace adds a subtrace to the current trace.
func (t *Trace) AddSubtrace(subtrace *Trace) {
	t.mu.Lock()
	defer t.mu.Unlock()
	subtrace.Parent = t
	t.Subtraces = append(t.Subtraces, subtrace)
}

// SetError sets the error for the trace if an error occurred during execution.
func (t *Trace) SetError(err error) {
	t.Error = err
}

// TraceManager manages traces for the entire program execution.
type TraceManager struct {
	mu           sync.Mutex
	CurrentTrace *Trace
	RootTrace    *Trace
}

// NewTraceManager creates a new TraceManager instance.
func NewTraceManager() *TraceManager {
	return &TraceManager{}
}

// StartTrace starts a new trace for a module.
func (tm *TraceManager) StartTrace(moduleName, moduleType string) *Trace {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	trace := NewTrace(moduleName, moduleType)

	if tm.RootTrace == nil {
		tm.RootTrace = trace
		tm.CurrentTrace = trace
	} else {
		tm.CurrentTrace.AddSubtrace(trace)
		tm.CurrentTrace = trace
	}
	return trace
}

// EndTrace ends the current trace and moves back to the parent trace.
func (tm *TraceManager) EndTrace() {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if tm.CurrentTrace != nil {
		tm.CurrentTrace.Duration = time.Since(tm.CurrentTrace.StartTime)
		if tm.CurrentTrace.Parent != nil {
			tm.CurrentTrace = tm.CurrentTrace.Parent
		}
	}
}

func (tm *TraceManager) GetRootTrace() *Trace {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	return tm.RootTrace
}

// TraceKey is the key used to store the TraceManager in the context.
type traceKey struct{}

// WithTraceManager adds a TraceManager to the context.
func WithTraceManager(ctx context.Context) context.Context {
	return context.WithValue(ctx, traceKey{}, NewTraceManager())
}

// GetTraceManager retrieves the TraceManager from the context.
func GetTraceManager(ctx context.Context) *TraceManager {
	// tm, _ := ctx.Value(traceKey{}).(*TraceManager)
	tm, ok := ctx.Value(traceKey{}).(*TraceManager)
	if !ok || tm == nil {
		log.Println("Warning: Failed to get TraceManager from context")
		return NewTraceManager()
	}
	return tm
}
