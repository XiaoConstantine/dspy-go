package modules

import (
	"context"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestChainOfThought(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `<response><rationale>Step 1, Step 2, Step 3</rationale><answer>42</answer></response>`,
	}, nil)

	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Create a context with TraceManager
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)
	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])
	assert.Equal(t, "Step 1, Step 2, Step 3", outputs["rationale"])

	// Verify traces
	spans := core.CollectSpans(ctx)
	require.Len(t, spans, 3, "Should have three spans with XML parsing")

	assert.Equal(t, "ChainOfThought (ChainOfThought)", spans[0].Operation)
	assert.Equal(t, "Predict (Predict)", spans[1].Operation)
	assert.Equal(t, spans[0].ID, spans[1].ParentID)
	rootSpan := spans[0]

	// Verify the span structure
	assert.Contains(t, rootSpan.Annotations, "inputs")
	assert.Contains(t, rootSpan.Annotations, "outputs")
	assert.NotZero(t, rootSpan.StartTime)
	assert.Nil(t, rootSpan.Error)

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestChainOfThought_WithLLMError(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior with an error
	expectedErr := errors.New(errors.LLMGenerationFailed, "LLM service unavailable")
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return((*core.LLMResponse)(nil), expectedErr)

	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Test the Process method with an error
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "failed to generate prediction")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestChainOfThought_WithMissingInput(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Test the Process method with missing input
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	// Empty inputs map will cause validation to fail
	inputs := map[string]any{}
	outputs, err := cot.Process(ctx, inputs)

	// Assert the results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "input validation failed")
}

func TestChainOfThought_RationaleAdded(t *testing.T) {
	// Test that the rationale field is automatically added to the signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	cot := NewChainOfThought(signature)

	// Check that rationale is the first output field
	cotSignature := cot.GetSignature()
	assert.Equal(t, 2, len(cotSignature.Outputs))
	assert.Equal(t, "rationale", cotSignature.Outputs[0].Name)
	assert.Equal(t, "answer", cotSignature.Outputs[1].Name)
}

func TestChainOfThought_WithDefaultOptions(t *testing.T) {
	// Create a mock LLM that can capture the generate options
	mockLLM := new(testutil.MockLLM)

	var capturedOpts []core.GenerateOption

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.MatchedBy(func(opts []core.GenerateOption) bool {
		capturedOpts = opts
		return true
	})).Return(&core.LLMResponse{
		Content: `
rationale:
Test reasoning

answer:
Test response
`,
	}, nil)

	// Create a ChainOfThought module with default options
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

	// Add default options
	cot.WithDefaultOptions(
		core.WithGenerateOptions(
			core.WithTemperature(0.8),
			core.WithMaxTokens(1000),
		),
	)

	// Call with additional process-specific options
	ctx := context.Background()
	inputs := map[string]any{"question": "Test question"}
	_, err := cot.Process(ctx, inputs,
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

func TestChainOfThought_WithStreamHandler(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Setup streaming
	streamConfig := &testutil.MockStreamConfig{
		Content:    "rationale: Streaming rationale\n\nanswer: Streaming response",
		ChunkSize:  5,
		TokenCounts: &core.TokenInfo{
			PromptTokens: 10,
		},
	}

	// Set up the mock behavior for streaming
	mockLLM.On("StreamGenerate", mock.Anything, mock.Anything, mock.Anything).Return(streamConfig, nil)

	// Create a ChainOfThought module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	cot := NewChainOfThought(signature)
	cot.SetLLM(mockLLM)

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
	outputs, err := cot.Process(ctx, inputs, core.WithStreamHandler(handler))

	// Verify results
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	assert.Greater(t, len(chunks), 0, "Should have received some chunks")

	// Verify the mock was called with streaming
	mockLLM.AssertExpectations(t)
}

func TestChainOfThought_Clone(t *testing.T) {
    // Create original instance
    signature := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.NewField("answer")}},
    )
    original := NewChainOfThought(signature)

    // Test Clone method
    cloned := original.Clone()

    // Verify it's the correct type
    clonedCOT, ok := cloned.(*ChainOfThought)
    assert.True(t, ok, "Clone should return a ChainOfThought instance")

    // Verify the signature was cloned correctly
    assert.Equal(t, original.GetSignature(), clonedCOT.GetSignature())

    // Verify the Predict module was cloned (not just referenced)
    assert.NotSame(t, original.Predict, clonedCOT.Predict,
        "The Predict module should be cloned, not just referenced")
}

func TestChainOfThought_Compose(t *testing.T) {
    // Create two modules
    signature1 := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.NewField("answer")}},
    )
    cot := NewChainOfThought(signature1)

    signature2 := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "answer"}}},
        []core.OutputField{{Field: core.NewField("summary")}},
    )
    predict := NewPredict(signature2)

    // Compose the modules
    composed := cot.Compose(predict)

    // Verify the type and structure
    chain, ok := composed.(*core.ModuleChain)
    assert.True(t, ok, "Compose should return a ModuleChain")
    assert.Equal(t, 2, len(chain.Modules), "Chain should have 2 modules")

    // Verify the modules in the chain
    assert.Same(t, cot, chain.Modules[0], "First module should be the ChainOfThought")
    assert.Same(t, predict, chain.Modules[1], "Second module should be the Predict")
}

func TestChainOfThought_GetSetSubModules(t *testing.T) {
    // Create original module
    signature := core.NewSignature(
        []core.InputField{{Field: core.Field{Name: "question"}}},
        []core.OutputField{{Field: core.NewField("answer")}},
    )
    cot := NewChainOfThought(signature)

    // Get submodules
    subModules := cot.GetSubModules()
    assert.Equal(t, 1, len(subModules), "Should have 1 submodule")
    assert.Same(t, cot.Predict, subModules[0], "Submodule should be the Predict module")

    // Create a new Predict module
    newPredict := NewPredict(signature)

    // Set submodules
    cot.SetSubModules([]core.Module{newPredict})

    // Verify the submodule was set
    assert.Same(t, newPredict, cot.Predict, "Predict module should be updated")
}
