package core

import (
	"context"
	"errors"
)

// Optimizer represents an interface for optimizing DSPy programs.
type Optimizer interface {
	// Compile optimizes a given program using the provided dataset and metric
	Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error)
}

// Metric is a function type that evaluates the performance of a program.
type Metric func(expected, actual map[string]interface{}) float64

// Dataset represents a collection of examples for training/evaluation.
type Dataset interface {
	// Next returns the next example in the dataset
	Next() (Example, bool)
	// Reset resets the dataset iterator
	Reset()
}

// Example represents a single training/evaluation example.
type Example struct {
	Inputs  map[string]interface{}
	Outputs map[string]interface{}
}

// BaseOptimizer provides a basic implementation of the Optimizer interface.
type BaseOptimizer struct {
	Name string
}

// Compile is a placeholder implementation and should be overridden by specific optimizer implementations.
func (bo *BaseOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return Program{}, errors.New("Compile method not implemented")
}

// OptimizerFactory is a function type for creating Optimizer instances.
type OptimizerFactory func() (Optimizer, error)

// OptimizerRegistry maintains a registry of available Optimizer implementations.
type OptimizerRegistry struct {
	factories map[string]OptimizerFactory
}

// NewOptimizerRegistry creates a new OptimizerRegistry.
func NewOptimizerRegistry() *OptimizerRegistry {
	return &OptimizerRegistry{
		factories: make(map[string]OptimizerFactory),
	}
}

// Register adds a new Optimizer factory to the registry.
func (r *OptimizerRegistry) Register(name string, factory OptimizerFactory) {
	r.factories[name] = factory
}

// Create instantiates a new Optimizer based on the given name.
func (r *OptimizerRegistry) Create(name string) (Optimizer, error) {
	factory, exists := r.factories[name]
	if !exists {
		return nil, errors.New("unknown Optimizer type: " + name)
	}
	return factory()
}

// CompileOptions represents options for the compilation process.
type CompileOptions struct {
	MaxTrials int
	Teacher   *Program
}

// WithMaxTrials sets the maximum number of trials for optimization.
func WithMaxTrials(n int) func(*CompileOptions) {
	return func(o *CompileOptions) {
		o.MaxTrials = n
	}
}

// WithTeacher sets a teacher program for optimization.
func WithTeacher(teacher *Program) func(*CompileOptions) {
	return func(o *CompileOptions) {
		o.Teacher = teacher
	}
}

// BootstrapFewShot implements a basic few-shot learning optimizer.
type BootstrapFewShot struct {
	BaseOptimizer
	MaxExamples int
}

// NewBootstrapFewShot creates a new BootstrapFewShot optimizer.
func NewBootstrapFewShot(maxExamples int) *BootstrapFewShot {
	return &BootstrapFewShot{
		BaseOptimizer: BaseOptimizer{Name: "BootstrapFewShot"},
		MaxExamples:   maxExamples,
	}
}

// Compile implements the optimization logic for BootstrapFewShot.
func (bfs *BootstrapFewShot) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	// Implementation of bootstrap few-shot learning
	// This is a placeholder and should be implemented based on the DSPy paper's description
	return program, nil
}

type ProgressReporter interface {
	Report(stage string, processed, total int)
}
