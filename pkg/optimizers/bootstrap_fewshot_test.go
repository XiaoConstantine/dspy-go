package optimizers

import (
	"context"
	"log"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func init() {
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("answer: Paris", nil)
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(map[string]interface{}{"answer": "Paris"}, nil)

	config.GlobalConfig.DefaultLLM = mockLLM
	config.GlobalConfig.TeacherLLM = mockLLM
	config.GlobalConfig.ConcurrencyLevel = 1
}

func createProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer"}}},
	))

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		tm := core.GetTraceManager(ctx)
		trace := tm.StartTrace("Program", "Program")
		defer tm.EndTrace()

		trace.SetInputs(inputs)
		outputs, err := predict.Process(ctx, inputs)
		if err != nil {
			trace.SetError(err)
			return nil, err
		}
		trace.SetOutputs(outputs)
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
	metric := func(example, prediction map[string]interface{}, trace *core.Trace) bool {
		return true // Always return true for this test
	}

	// Create BootstrapFewShot optimizer
	maxBootstrapped := 2
	optimizer := NewBootstrapFewShot(metric, maxBootstrapped)
	ctx := core.WithTraceManager(context.Background())

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
	tm := core.GetTraceManager(ctx)
	rootTrace := tm.GetRootTrace()
	if assert.NotNil(t, rootTrace) {
		assert.Equal(t, "Compilation", rootTrace.ModuleName)
		assert.Equal(t, "Compilation", rootTrace.ModuleType)
		assert.Len(t, rootTrace.Subtraces, maxBootstrapped+1) // Should have 2 example traces + 1 compliation trace

		for i, subtrace := range rootTrace.Subtraces {
			assert.Equal(t, "Example", subtrace.ModuleName)
			assert.Equal(t, "Example", subtrace.ModuleType)
			assert.Contains(t, subtrace.Inputs, "question")
			assert.Contains(t, subtrace.Outputs, "answer")
			assert.Equal(t, "Paris", subtrace.Outputs["answer"])
			assert.Len(t, subtrace.Subtraces, 1) // Should have 1 TeacherPrediction subtrace

			predictionTrace := subtrace.Subtraces[0]
			assert.Equal(t, "TeacherPrediction", predictionTrace.ModuleName)
			assert.Equal(t, "Prediction", predictionTrace.ModuleType)
			assert.Contains(t, predictionTrace.Inputs, "question")
			assert.Contains(t, predictionTrace.Outputs, "answer")
			assert.Equal(t, "Paris", predictionTrace.Outputs["answer"])

			log.Printf("Example %d - Inputs: %v, Outputs: %v", i+1, subtrace.Inputs, subtrace.Outputs)
		}

		// Verify the final outputs of the compilation trace
		assert.Contains(t, rootTrace.Outputs, "compiledStudent")
	}
}

func TestBootstrapFewShotEdgeCases(t *testing.T) {

	trainset := []map[string]interface{}{
		{"question": "Q1"},
		{"question": "Q2"},
		{"question": "Q3"},
	}

	t.Run("MaxBootstrapped Zero", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *core.Trace) bool { return true }, 0)
		ctx := core.WithTraceManager(context.Background())

		optimized, err := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		assert.NoError(t, err)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})

	t.Run("MaxBootstrapped Large", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *core.Trace) bool {
			return true
		}, 100)
		ctx := core.WithTraceManager(context.Background())

		optimized, err := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}
		demoCount := len(optimized.Modules["predict"].(*modules.Predict).Demos)
		assert.Equal(t, len(trainset), demoCount)
	})

	t.Run("Metric Rejects All", func(t *testing.T) {
		optimizer := NewBootstrapFewShot(func(_, _ map[string]interface{}, _ *core.Trace) bool { return false }, 2)
		ctx := core.WithTraceManager(context.Background())
		optimized, _ := optimizer.Compile(ctx, createProgram(), createProgram(), trainset)
		assert.Equal(t, 0, len(optimized.Modules["predict"].(*modules.Predict).Demos))
	})
}
