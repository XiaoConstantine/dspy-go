package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestPredict(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `answer:
	42
	`,
	}, nil)

	// Create a Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Test the Process method
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := predict.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
	// Verify traces
	spans := core.CollectSpans(ctx)
	require.Len(t, spans, 1)
	span := spans[0]

	inputsMap, _ := span.Annotations["inputs"].(map[string]interface{})
	question, _ := inputsMap["question"].(string)

	outputsMap, _ := span.Annotations["outputs"].(map[string]interface{})
	answer, _ := outputsMap["answer"].(string)

	assert.Contains(t, question, "What is the meaning of life?")
	assert.Contains(t, answer, "4")
}

func TestPredict_WithLLMError(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior with an error
	expectedErr := errors.New(errors.LLMGenerationFailed, "LLM service unavailable")
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return((*core.LLMResponse)(nil), expectedErr)

	// Create a Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Test the Process method with an error
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := predict.Process(ctx, inputs)

	// Assert the results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "failed to generate prediction")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestPredict_WithMissingInput(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create a Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Test the Process method with missing input
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	// Empty inputs map will cause validation to fail
	inputs := map[string]any{}
	outputs, err := predict.Process(ctx, inputs)

	// Assert the results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "input validation failed")
}

func TestPredict_WithGenerateOptions(t *testing.T) {
	// Create a mock LLM that can capture the generate options
	mockLLM := new(testutil.MockLLM)

	var capturedOpts []core.GenerateOption

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.MatchedBy(func(opts []core.GenerateOption) bool {
		capturedOpts = opts
		return true
	})).Return(&core.LLMResponse{
		Content: "answer: Test response",
	}, nil)

	// Create a Predict module with default options
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Add default options
	predict.WithDefaultOptions(
		core.WithGenerateOptions(
			core.WithTemperature(0.8),
			core.WithMaxTokens(1000),
		),
	)

	// Call with additional process-specific options
	ctx := context.Background()
	inputs := map[string]any{"question": "Test question"}
	_, err := predict.Process(ctx, inputs,
		core.WithGenerateOptions(
			core.WithTemperature(0.5), // Override temperature
		),
	)

	// Verify results
	assert.NoError(t, err)
	assert.NotEmpty(t, capturedOpts)

	// We can't directly test the options since they're opaque functions,
	// but we can verify the mock was called with some options
	mockLLM.AssertExpectations(t)
}

func TestPredict_WithStreamHandler(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Setup streaming
	streamConfig := &testutil.MockStreamConfig{
		Content:   "answer: Streaming response",
		ChunkSize: 5,
		TokenCounts: &core.TokenInfo{
			PromptTokens: 10,
		},
	}

	// Set up the mock behavior for streaming
	mockLLM.On("StreamGenerate", mock.Anything, mock.Anything, mock.Anything).Return(streamConfig, nil)

	// Create a Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Create a handler to collect chunks
	var chunks []string
	handler := func(chunk core.StreamChunk) error {
		if !chunk.Done && chunk.Error == nil {
			chunks = append(chunks, chunk.Content)
		}
		return nil
	}

	// Process with streaming
	ctx := context.Background()
	inputs := map[string]any{"question": "Stream test"}
	outputs, err := predict.Process(ctx, inputs, core.WithStreamHandler(handler))

	// Verify results
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	assert.Greater(t, len(chunks), 0, "Should have received some chunks")

	// Verify the mock was called with streaming
	mockLLM.AssertExpectations(t)
}
