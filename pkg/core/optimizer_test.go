package core

import (
	"context"
	"errors"
	"testing"

	pkgerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// TestOptimizerRegistry tests the OptimizerRegistry.
func TestOptimizerRegistry(t *testing.T) {
	registry := NewOptimizerRegistry()

	// Test registering an Optimizer
	registry.Register("test", func() (Optimizer, error) {
		return &MockOptimizer{}, nil
	})

	// Test creating a registered Optimizer
	optimizer, err := registry.Create("test")
	if err != nil {
		t.Errorf("Unexpected error creating Optimizer: %v", err)
	}
	if _, ok := optimizer.(*MockOptimizer); !ok {
		t.Error("Created Optimizer is not of expected type")
	}

	// Test creating an unregistered Optimizer
	_, err = registry.Create("nonexistent")
	if err == nil {
		t.Error("Expected error when creating unregistered Optimizer, got nil")
	}
	var typedErr *pkgerrors.Error
	if !errors.As(err, &typedErr) || typedErr.Code() != pkgerrors.InvalidInput {
		t.Fatalf("expected InvalidInput error, got %v", err)
	}
}

// TestCompileOptions tests the CompileOptions and related functions.
func TestCompileOptions(t *testing.T) {
	opts := &CompileOptions{}

	WithMaxTrials(10)(opts)
	if opts.MaxTrials != 10 {
		t.Errorf("Expected MaxTrials 10, got %d", opts.MaxTrials)
	}

	teacherProgram := &Program{
		Modules: map[string]Module{
			"test": NewModule(NewSignature(
				[]InputField{{Name: "input"}},
				[]OutputField{{Name: "output"}},
			)),
		},
		Forward: func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			return inputs, nil
		},
	}

	WithTeacher(teacherProgram)(opts)
	if opts.Teacher == nil {
		t.Error("Expected Teacher program to be set")
	} else {
		if len(opts.Teacher.Modules) != 1 {
			t.Errorf("Expected 1 module in Teacher program, got %d", len(opts.Teacher.Modules))
		}
		if opts.Teacher.Forward == nil {
			t.Error("Expected Forward function to be set in Teacher program")
		}
	}
}

// TestBootstrapFewShot tests the BootstrapFewShot optimizer.
func TestBootstrapFewShot(t *testing.T) {
	optimizer := NewBootstrapFewShot(5)

	if optimizer.MaxExamples != 5 {
		t.Errorf("Expected MaxExamples 5, got %d", optimizer.MaxExamples)
	}

	// Create a simple program for testing
	program := NewProgram(map[string]Module{
		"test": NewModule(NewSignature(
			[]InputField{{Name: "input"}},
			[]OutputField{{Name: "output"}},
		)),
	}, nil)

	// Create a simple dataset for testing
	dataset := &MockDataset{}

	// Create a simple metric for testing
	metric := func(expected, actual map[string]any) float64 {
		return 1.0 // Always return 1.0 for this test
	}

	optimizedProgram, err := optimizer.Compile(context.Background(), program, dataset, metric)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(optimizedProgram.Modules) != 1 {
		t.Errorf("Expected 1 module in optimized program, got %d", len(optimizedProgram.Modules))
	}
}

func TestMaterializeDataset(t *testing.T) {
	dataset := &MockDataset{
		data: []Example{
			{Inputs: map[string]any{"id": 1}},
			{Inputs: map[string]any{"id": 2}},
		},
		index: 1,
	}

	examples := MaterializeDataset(dataset)
	if len(examples) != 2 {
		t.Fatalf("MaterializeDataset() returned %d examples, want 2", len(examples))
	}
	if got := examples[0].Inputs["id"]; got != 1 {
		t.Fatalf("first example id = %v, want 1", got)
	}
	if got := examples[1].Inputs["id"]; got != 2 {
		t.Fatalf("second example id = %v, want 2", got)
	}
}

func TestMaterializeDatasetContextDoesNotConsumeCanceledDataset(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	dataset := &MockDataset{data: []Example{{Inputs: map[string]any{"id": 1}}}}

	_, err := MaterializeDatasetContext(ctx, dataset)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("MaterializeDatasetContext() error = %v, want context.Canceled", err)
	}
	if dataset.resetCalls != 0 || dataset.nextCalls != 0 {
		t.Fatalf("canceled materialization consumed dataset: Reset=%d Next=%d", dataset.resetCalls, dataset.nextCalls)
	}
}

func TestMaterializeDatasetContextStopsDuringTraversal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	dataset := &MockDataset{
		data:   []Example{{Inputs: map[string]any{"id": 1}}, {Inputs: map[string]any{"id": 2}}},
		onNext: cancel,
	}

	_, err := MaterializeDatasetContext(ctx, dataset)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("MaterializeDatasetContext() error = %v, want context.Canceled", err)
	}
	if dataset.resetCalls != 1 || dataset.nextCalls != 1 {
		t.Fatalf("materialization calls: Reset=%d Next=%d, want Reset=1 Next=1", dataset.resetCalls, dataset.nextCalls)
	}
}

func TestBaseOptimizerCompileReturnsUnsupportedOperation(t *testing.T) {
	optimizer := &BaseOptimizer{Name: "base"}

	_, err := optimizer.Compile(context.Background(), Program{}, &MockDataset{}, func(expected, actual map[string]any) float64 {
		return 0
	})
	if err == nil {
		t.Fatal("expected error from BaseOptimizer.Compile")
	}

	var typedErr *pkgerrors.Error
	if !errors.As(err, &typedErr) || typedErr.Code() != pkgerrors.UnsupportedOperation {
		t.Fatalf("expected UnsupportedOperation error, got %v", err)
	}
}

// MockOptimizer is a mock implementation of the Optimizer interface for testing.
type MockOptimizer struct{}

func (m *MockOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return program, nil
}

// MockDataset is a mock implementation of the Dataset interface for testing.
type MockDataset struct {
	data       []Example
	index      int
	resetCalls int
	nextCalls  int
	onNext     func()
}

func (m *MockDataset) Next() (Example, bool) {
	m.nextCalls++
	if m.onNext != nil {
		m.onNext()
	}
	if m.index >= len(m.data) {
		return Example{}, false
	}
	example := m.data[m.index]
	m.index++
	return example, true
}

func (m *MockDataset) Reset() {
	m.resetCalls++
	m.index = 0
}
