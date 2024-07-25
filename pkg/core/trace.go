package core

import (
	"time"
)

// Trace represents the execution trace of a single module in the program.
type Trace struct {
    ModuleName   string
    ModuleType   string
    PredictorID  string // Unique identifier for the predictor instance
    Inputs       map[string]interface{}
    Outputs      map[string]interface{}
    StartTime    time.Time
    Duration     time.Duration
    Subtraces    []Trace // For nested module calls
}

// NewTrace creates a new Trace instance.
func NewTrace(moduleName, moduleType, predictorID string) *Trace {
    return &Trace{
        ModuleName:  moduleName,
        ModuleType:  moduleType,
        PredictorID: predictorID,
        Inputs:      make(map[string]interface{}),
        Outputs:     make(map[string]interface{}),
        StartTime:   time.Now(),
        Subtraces:   []Trace{},
    }
}

// SetInputs sets the inputs for the trace.
func (t *Trace) SetInputs(inputs map[string]interface{}) {
    t.Inputs = inputs
}

// SetOutputs sets the outputs for the trace and calculates the duration.
func (t *Trace) SetOutputs(outputs map[string]interface{}) {
    t.Outputs = outputs
    t.Duration = time.Since(t.StartTime)
}

// AddSubtrace adds a subtrace to the current trace.
func (t *Trace) AddSubtrace(subtrace Trace) {
    t.Subtraces = append(t.Subtraces, subtrace)
}
