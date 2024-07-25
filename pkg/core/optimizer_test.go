package core

import (
	"context"
	"testing"
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
				[]InputField{{Field: Field{Name: "input"}}},
				[]OutputField{{Field: Field{Name: "output"}}},
			)),
		},
		Forward: func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
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
			[]InputField{{Field: Field{Name: "input"}}},
			[]OutputField{{Field: Field{Name: "output"}}},
		)),
	}, nil)

	// Create a simple dataset for testing
	dataset := &MockDataset{}

	// Create a simple metric for testing
	metric := func(expected, actual map[string]interface{}) float64 {
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

// MockOptimizer is a mock implementation of the Optimizer interface for testing.
type MockOptimizer struct{}

func (m *MockOptimizer) Compile(ctx context.Context, program Program, dataset Dataset, metric Metric) (Program, error) {
	return program, nil
}

// MockDataset is a mock implementation of the Dataset interface for testing.
type MockDataset struct {
	data  []Example
	index int
}

func (m *MockDataset) Next() (Example, bool) {
	if m.index >= len(m.data) {
		return Example{}, false
	}
	example := m.data[m.index]
	m.index++
	return example, true
}

func (m *MockDataset) Reset() {
	m.index = 0
}
