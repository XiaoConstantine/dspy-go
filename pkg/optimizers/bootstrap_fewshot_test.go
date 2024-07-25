package optimizers

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
)

func TestBootstrapFewShot(t *testing.T) {
	// Create a mock LLM for testing
	mockLLM := &MockLLM{
		generateFunc: func(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
			return "answer: Paris", nil
		},
	}

	// Create student and teacher programs
	studentPredict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	))
	studentPredict.LLM = mockLLM
	student := core.NewProgram(map[string]core.Module{"predict": studentPredict}, nil)

	teacherPredict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	))
	teacherPredict.LLM = mockLLM
	teacher := core.NewProgram(map[string]core.Module{"predict": teacherPredict}, nil)

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		module := teacher.Modules["predict"].(*modules.Predict)
		outputs, err := module.Process(ctx, inputs)
		if err != nil {
			return nil, err
		}

		// Simulate adding a trace
		if tracing, ok := ctx.Value("tracing").(bool); ok && tracing {
			trace := core.NewTrace("Predict", "Predict", "")
			trace.SetInputs(inputs)
			trace.SetOutputs(outputs)
			traces := ctx.Value("traces").(*[]core.Trace)
			*traces = append(*traces, *trace)
		}

		return outputs, nil
	}
	student.Forward = forwardFunc
	teacher.Forward = forwardFunc

	// Create training set
	trainset := []map[string]interface{}{
		{"question": "What is the capital of France?"},
		{"question": "What is the capital of Germany?"},
		{"question": "What is the capital of Italy?"},
	}

	// Define metric function
	// Define metric function
	metric := func(example, prediction map[string]interface{}, traces *[]core.Trace) bool {
		return true // Always return true for this test
	}

	// Create BootstrapFewShot optimizer
	maxBootstrapped := 2
	optimizer := NewBootstrapFewShot(metric, maxBootstrapped)

	// Compile the program
	optimizedProgram, _ := optimizer.Compile(student, teacher, trainset)

	// Check if the optimized program has the correct number of demonstrations
	optimizedPredict, ok := optimizedProgram.Modules["predict"].(*modules.Predict)
	assert.True(t, ok)
	assert.Equal(t, maxBootstrapped, len(optimizedPredict.Demos))

	// Check if the demonstrations are correct
	for _, demo := range optimizedPredict.Demos {
		assert.Contains(t, demo.Inputs, "question")
		assert.Contains(t, demo.Outputs, "answer")
		assert.Equal(t, "Paris", demo.Outputs["answer"])
	}
}

// MockLLM is a mock implementation of the LLM interface for testing.
type MockLLM struct {
	generateFunc func(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error)
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
	return m.generateFunc(ctx, prompt, options...)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	// This method is not used in this test, so we'll leave it empty
	return nil, nil
}

func TestBootstrapFewShotEdgeCases(t *testing.T) {
	mockLLM := &MockLLM{
		generateFunc: func(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
			return "answer: Test", nil
		},
	}

	createProgram := func() core.Program {
		predict := modules.NewPredict(core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "question"}}},
			[]core.OutputField{{Field: core.Field{Name: "answer"}}},
		))
		predict.LLM = mockLLM
		return core.NewProgram(map[string]core.Module{"predict": predict}, nil)
	}

	trainset := []map[string]interface{}{
		{"question": "Q1"},
		{"question": "Q2"},
		{"question": "Q3"},
	}

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool { return true }, 0)
		optimized, _ := optimizer.Compile(createProgram(), createProgram(), trainset)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	// t.Run("MaxBootstrapped Large", func(t *testing.T) {
	// 	optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool { return true }, 100)
	// 	optimized, _ := optimizer.Compile(createProgram(), createProgram(), trainset)
	// 	assert.Equal(t, len(trainset), len(optimized.Modules["predict"].(*modules.Predict).Demos))
	// })

	t.Run("Metric Rejects All", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool { return false }, 2)
		optimized, _ := optimizer.Compile(createProgram(), createProgram(), trainset)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})
}
