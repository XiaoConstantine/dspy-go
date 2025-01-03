package optimizers

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func init() {
	mockLLM := new(testutil.MockLLM)

	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: `answer: 
	Paris`}, nil)
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(map[string]interface{}{"answer": "Paris"}, nil)

	config.GlobalConfig.DefaultLLM = mockLLM
	config.GlobalConfig.TeacherLLM = mockLLM
	config.GlobalConfig.ConcurrencyLevel = 1
}

func createProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	))

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {

		ctx, span := core.StartSpan(ctx, "Forward")
		defer core.EndSpan(ctx)
		span.WithAnnotation("inputs", inputs)
		outputs, err := predict.Process(ctx, inputs)
		if err != nil {
			span.WithError(err)
			return nil, err
		}
		span.WithAnnotation("outputs", outputs)
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
	metric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
		return true // Always return true for this test
	}

	// Create BootstrapFewShot optimizer
	maxBootstrapped := 2
	optimizer := NewBootstrapFewShot(metric, maxBootstrapped)

	ctx := core.WithExecutionState(context.Background())

	// Compile the program
	optimizedProgram, _ := optimizer.Compile(ctx, student, teacher, trainset)

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
	// Verify the trace structure
	spans := core.CollectSpans(ctx)
	require.NotEmpty(t, spans, "Expected spans to be recorded")
	rootSpan := spans[0]
	assert.Equal(t, "Compilation", rootSpan.Operation, "Expected Compilation as root span")

	// Find Example spans (should be direct children of Compilation)
	var exampleSpans []*core.Span
	for _, span := range spans {
		if span.Operation == "Example" {
			exampleSpans = append(exampleSpans, span)
		}
	}
	assert.Equal(t, maxBootstrapped, len(exampleSpans),
		"Expected number of Example spans to match maxBootstrapped")

	// Verify span structure and content
	var compilationSpan *core.Span
	for _, span := range spans {
		if span.Operation == "Compilation" {
			compilationSpan = span
		}
	}

	// Verify compilation span
	require.NotNil(t, compilationSpan, "Expected to find compilation span")
	assert.NotZero(t, compilationSpan.StartTime)
	assert.Nil(t, compilationSpan.Error)

}

func TestBootstrapFewShotEdgeCases(t *testing.T) {

	trainset := []map[string]interface{}{
		{"question": "Q1"},
		{"question": "Q2"},
		{"question": "Q3"},
	}

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool { return true }, 0)
		ctx := context.Background()

		optimized, err := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("MaxBootstrapped Large", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool {
			return true
		}, 100)
		ctx := context.Background()

		optimized, err := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}
		demoCount := len(optimized.Modules["predict"].(*modules.Predict).Demos)
		assert.Equal(t, len(trainset), demoCount)
	})

	t.Run("Metric Rejects All", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ context.Context) bool { return false }, 2)
		ctx := context.Background()
		optimized, _ := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})
}
