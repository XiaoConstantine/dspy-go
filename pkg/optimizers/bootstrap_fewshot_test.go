package optimizers

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
)

var mockLLM = &MockLLM{
	generateFunc: func(ctx context.Context, prompt string, options ...core.GenerateOption) (string, error) {
		return "answer: Paris", nil
	},
}

func createProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	))
	predict.LLM = mockLLM

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		outputs, err := predict.Process(ctx, inputs)
		if err != nil {
			return nil, err
		}
		traces, ok := ctx.Value("traces").(*[]core.Trace)
		if ok && traces != nil {
			trace := core.NewTrace("Predict", "Predict", "")
			trace.SetInputs(inputs)
			trace.SetOutputs(outputs)
			*traces = append(*traces, *trace)
		}
		return outputs, nil
	}

	return core.NewProgram(map[string]core.Module{"predict": predict}, forwardFunc)
}

func TestBootstrapFewShot(t *testing.T) {
	student := createProgram()
	teacher := createProgram()
	// Create training set
	trainset := []map[string]interface{}{
		{"question": "What is the capital of France?"},
		{"question": "What is the capital of Germany?"},
		{"question": "What is the capital of Italy?"},
	}

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

	trainset := []map[string]interface{}{
		{"question": "Q1"},
		{"question": "Q2"},
		{"question": "Q3"},
	}

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool { return true }, 0)
		optimized, err := optimizer.Compile(createProgram(), createProgram(), trainset)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("MaxBootstrapped Large", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool {
			return true
		}, 100)
		optimized, err := optimizer.Compile(createProgram(), createProgram(), trainset)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}
		demoCount := len(optimized.Modules["predict"].(*modules.Predict).Demos)
		assert.Equal(t, len(trainset), demoCount)
	})

	t.Run("Metric Rejects All", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *[]core.Trace) bool { return false }, 2)
		optimized, _ := optimizer.Compile(createProgram(), createProgram(), trainset)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})
}
