package core

import (
	"context"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// Optimizer represents an interface for optimizing DSPy programs.
type Optimizer interface {
	// Compile optimizes a given program using the provided dataset and metric.
	Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error)
}

// ExamplesCompiler optimizes a program from a materialized collection of
// examples. Implementations must treat the slice and its examples as read-only.
//
// Optimizers should implement this interface alongside Optimizer while callers
// migrate away from Dataset's shared mutable cursor.
type ExamplesCompiler interface {
	CompileExamples(ctx context.Context, program Program, examples []Example, metric Metric) (Program, error)
}

// Metric is a function type that evaluates the performance of a program.
//
// A Metric cannot observe cancellation or report a failure. New code that needs
// either should implement Scorer and call a ScorerCompiler. MetricScorer adapts
// an existing Metric.
type Metric func(expected, actual map[string]any) float64

// MetricScorer adapts metric to a Scorer that calls it with the example's
// expected outputs and never returns an error. It returns nil for a nil metric.
//
// The conversion asserts that metric is safe for concurrent use, as Scorer
// requires.
func MetricScorer(metric Metric) Scorer {
	if metric == nil {
		return nil
	}
	return ScorerFunc(func(_ context.Context, example Example, prediction map[string]any) (float64, error) {
		return metric(example.Outputs, prediction), nil
	})
}

// Scorer scores one prediction against the example that produced it.
//
// Implementations must be safe for concurrent use: optimizers call Score from
// several goroutines at once. A non-nil error means the prediction could not be
// scored, such as a failed judge call. An unscorable example never raises a
// candidate's score; each optimizer's CompileWithScorer documents whether it
// counts as zero, rejects the candidate, or fails the run. Scores should not be
// negative: an optimizer that counts a failed example as zero treats zero as the
// lowest score, and GEPA and COPRO count a negative score as zero. Score should
// return promptly once ctx is done.
type Scorer interface {
	Score(ctx context.Context, example Example, prediction map[string]any) (float64, error)
}

// ScorerFunc adapts a function to a Scorer.
type ScorerFunc func(ctx context.Context, example Example, prediction map[string]any) (float64, error)

// Score calls f. A nil ScorerFunc reports every prediction as unscorable.
func (f ScorerFunc) Score(ctx context.Context, example Example, prediction map[string]any) (float64, error) {
	if f == nil {
		return 0, errors.New(errors.InvalidInput, "nil ScorerFunc")
	}
	return f(ctx, example, prediction)
}

// ScorerCompiler optimizes a program from materialized examples against a
// Scorer. Implementations must treat the slice and its examples as read-only.
//
// Optimizers implement this interface alongside Optimizer. Unlike Compile, it
// may call the scorer concurrently and it reports scorer errors.
type ScorerCompiler interface {
	CompileWithScorer(ctx context.Context, program Program, examples []Example, scorer Scorer) (Program, error)
}

// Dataset represents a collection of examples for training/evaluation.
type Dataset interface {
	// Next returns the next example in the dataset
	Next() (Example, bool)
	// Reset resets the dataset iterator
	Reset()
}

// Example represents a single training/evaluation example.
type Example struct {
	Inputs  map[string]any
	Outputs map[string]any
}

// MaterializeDataset consumes a legacy cursor-based Dataset into a slice.
// Callers must have exclusive access to dataset while it is being consumed.
func MaterializeDataset(dataset Dataset) []Example {
	examples, _ := MaterializeDatasetContext(context.Background(), dataset)
	return examples
}

// MaterializeDatasetContext consumes a legacy cursor-based Dataset into a
// slice, stopping if ctx is canceled between cursor operations. Callers must
// have exclusive access to dataset while it is being consumed.
func MaterializeDatasetContext(ctx context.Context, dataset Dataset) ([]Example, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var examples []Example
	dataset.Reset()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		example, hasNext := dataset.Next()
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !hasNext {
			return examples, nil
		}
		examples = append(examples, example)
	}
}

// DatasetToSlice converts a Dataset to a slice of Examples.
//
// Deprecated: use MaterializeDataset while migrating legacy Dataset callers,
// or pass []Example directly to an ExamplesCompiler.
func DatasetToSlice(dataset Dataset) []Example {
	return MaterializeDataset(dataset)
}

// BaseOptimizer provides a basic implementation of the Optimizer interface.
type BaseOptimizer struct {
	Name string
}

// Compile is a placeholder implementation and should be overridden by specific optimizer implementations.
func (bo *BaseOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return Program{}, errors.New(errors.UnsupportedOperation, "Compile method not implemented")
}

// OptimizerFactory is a function type for creating Optimizer instances.
type OptimizerFactory func() (Optimizer, error)

// OptimizerRegistry maintains a registry of available Optimizer implementations.
type OptimizerRegistry struct {
	mu        sync.RWMutex
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
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[name] = factory
}

// Create instantiates a new Optimizer based on the given name.
func (r *OptimizerRegistry) Create(name string) (Optimizer, error) {
	r.mu.RLock()
	factory, exists := r.factories[name]
	r.mu.RUnlock()
	if !exists {
		return nil, errors.New(errors.InvalidInput, "unknown Optimizer type: "+name)
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
		Name:        "BootstrapFewShot",
		MaxExamples: maxExamples,
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
