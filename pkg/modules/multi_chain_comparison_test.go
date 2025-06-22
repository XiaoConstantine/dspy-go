package modules

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestNewMultiChainComparison(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	M := 3
	temperature := 0.7
	multiChain := NewMultiChainComparison(signature, M, temperature)

	// Verify the module was created correctly
	assert.NotNil(t, multiChain)
	assert.Equal(t, M, multiChain.M)
	assert.Equal(t, "answer", multiChain.lastKey)

	// Verify the signature was modified correctly
	modifiedSignature := multiChain.GetSignature()
	
	// Should have original inputs plus M reasoning attempt fields
	assert.Len(t, modifiedSignature.Inputs, 1+M)
	assert.Equal(t, "question", modifiedSignature.Inputs[0].Name)
	assert.Equal(t, "reasoning_attempt_1", modifiedSignature.Inputs[1].Name)
	assert.Equal(t, "reasoning_attempt_2", modifiedSignature.Inputs[2].Name)
	assert.Equal(t, "reasoning_attempt_3", modifiedSignature.Inputs[3].Name)

	// Should have rationale prepended to original outputs
	assert.Len(t, modifiedSignature.Outputs, 2)
	assert.Equal(t, "rationale", modifiedSignature.Outputs[0].Name)
	assert.Equal(t, "answer", modifiedSignature.Outputs[1].Name)
}

func TestMultiChainComparison_Process_Success(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `Accurate Reasoning: Thank you everyone. Let's now holistically
After analyzing all attempts, the correct answer is 42

answer:
42`,
	}, nil)

	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 3, 0.7)
	multiChain.SetLLM(mockLLM)

	// Create test completions
	completions := []map[string]interface{}{
		{
			"rationale": "solve the math problem",
			"answer":    "40",
		},
		{
			"reasoning": "calculate step by step",
			"answer":    "42",
		},
		{
			"rationale": "use mathematical principles",
			"answer":    "38",
		},
	}

	// Test the Process method
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]interface{}{
		"question":    "What is the meaning of life?",
		"completions": completions,
	}

	outputs, err := multiChain.Process(ctx, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	assert.Contains(t, outputs, "rationale")
	assert.Contains(t, outputs, "answer")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestMultiChainComparison_Process_InvalidCompletions(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 3, 0.7)

	ctx := context.Background()

	// Test with missing completions
	inputs := map[string]interface{}{
		"question": "What is the meaning of life?",
	}

	outputs, err := multiChain.Process(ctx, inputs)
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "completions not found")

	// Test with invalid completions type
	inputs = map[string]interface{}{
		"question":    "What is the meaning of life?",
		"completions": "invalid",
	}

	outputs, err = multiChain.Process(ctx, inputs)
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "completions must be a slice of maps")
}

func TestMultiChainComparison_Process_WrongNumberOfCompletions(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module expecting 3 completions
	multiChain := NewMultiChainComparison(signature, 3, 0.7)

	ctx := context.Background()

	// Test with wrong number of completions (only 2 instead of 3)
	completions := []map[string]interface{}{
		{
			"rationale": "solve the math problem",
			"answer":    "40",
		},
		{
			"reasoning": "calculate step by step",
			"answer":    "42",
		},
	}

	inputs := map[string]interface{}{
		"question":    "What is the meaning of life?",
		"completions": completions,
	}

	outputs, err := multiChain.Process(ctx, inputs)
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "number of attempts doesn't match expected M")
}

func TestMultiChainComparison_ProcessCompletions(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 3, 0.7)

	// Test completions with rationale
	completions := []map[string]interface{}{
		{
			"rationale": "solve the math problem step by step",
			"answer":    "40",
		},
		{
			"reasoning": "calculate using mathematical principles",
			"answer":    "42",
		},
		{
			"rationale": "use logical reasoning",
			"answer":    "38",
		},
	}

	attempts, err := multiChain.processCompletions(completions)

	assert.NoError(t, err)
	assert.Len(t, attempts, 3)
	assert.Equal(t, "«I'm trying to solve the math problem step by step I'm not sure but my prediction is 40»", attempts[0])
	assert.Equal(t, "«I'm trying to calculate using mathematical principles I'm not sure but my prediction is 42»", attempts[1])
	assert.Equal(t, "«I'm trying to use logical reasoning I'm not sure but my prediction is 38»", attempts[2])
}

func TestMultiChainComparison_ProcessCompletions_EmptyFields(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 2, 0.7)

	// Test completions with empty/missing fields
	completions := []map[string]interface{}{
		{
			"answer": "40",
		},
		{
			"rationale": "",
			"answer":    "42",
		},
	}

	attempts, err := multiChain.processCompletions(completions)

	assert.NoError(t, err)
	assert.Len(t, attempts, 2)
	assert.Equal(t, "«I'm trying to  I'm not sure but my prediction is 40»", attempts[0])
	assert.Equal(t, "«I'm trying to  I'm not sure but my prediction is 42»", attempts[1])
}

func TestMultiChainComparison_Clone(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	original := NewMultiChainComparison(signature, 3, 0.7)

	// Clone the module
	cloned := original.Clone().(*MultiChainComparison)

	// Verify the clone
	assert.NotNil(t, cloned)
	assert.Equal(t, original.M, cloned.M)
	assert.Equal(t, original.lastKey, cloned.lastKey)
	assert.NotSame(t, original, cloned)
	assert.NotSame(t, original.predict, cloned.predict)
}

func TestMultiChainComparison_SetLLM(t *testing.T) {
	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 3, 0.7)

	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set the LLM
	multiChain.SetLLM(mockLLM)

	// Verify the LLM was set on the internal predict module
	assert.Equal(t, mockLLM, multiChain.predict.LLM)
}

func TestMultiChainComparison_WithLLMError(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior with an error
	expectedErr := errors.New(errors.LLMGenerationFailed, "LLM service unavailable")
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return((*core.LLMResponse)(nil), expectedErr)

	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 3, 0.7)
	multiChain.SetLLM(mockLLM)

	// Create test completions
	completions := []map[string]interface{}{
		{"rationale": "solve", "answer": "40"},
		{"reasoning": "calculate", "answer": "42"},
		{"rationale": "reason", "answer": "38"},
	}

	// Test the Process method with an error
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]interface{}{
		"question":    "What is the meaning of life?",
		"completions": completions,
	}

	outputs, err := multiChain.Process(ctx, inputs)

	// Assert the results
	assert.Error(t, err)
	assert.Nil(t, outputs)
	assert.Contains(t, err.Error(), "failed to generate prediction")

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestMultiChainComparison_WithStreamHandler(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Setup streaming
	streamConfig := &testutil.MockStreamConfig{
		Content:   "rationale: Comprehensive analysis\nanswer: 42",
		ChunkSize: 10,
		TokenCounts: &core.TokenInfo{
			PromptTokens: 15,
		},
	}

	// Set up the mock behavior for streaming
	mockLLM.On("StreamGenerate", mock.Anything, mock.Anything, mock.Anything).Return(streamConfig, nil)

	// Create a basic signature
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create a MultiChainComparison module
	multiChain := NewMultiChainComparison(signature, 2, 0.7)
	multiChain.SetLLM(mockLLM)

	// Create test completions
	completions := []map[string]interface{}{
		{"rationale": "solve", "answer": "40"},
		{"reasoning": "calculate", "answer": "42"},
	}

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
	inputs := map[string]interface{}{
		"question":    "What is the meaning of life?",
		"completions": completions,
	}

	outputs, err := multiChain.Process(ctx, inputs, core.WithStreamHandler(handler))

	// Verify results
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	assert.Greater(t, len(chunks), 0, "Should have received some chunks")

	// Verify the mock was called with streaming
	mockLLM.AssertExpectations(t)
}

func TestMultiChainComparison_SignatureModification(t *testing.T) {
	// Test signature modification with different M values
	testCases := []struct {
		name        string
		M           int
		inputFields int
		expectedInputs int
		expectedOutputs int
	}{
		{"M=1", 1, 2, 3, 3}, // 2 original + 1 reasoning attempt = 3 inputs, 2 original + 1 rationale = 3 outputs
		{"M=3", 3, 1, 4, 2}, // 1 original + 3 reasoning attempts = 4 inputs, 1 original + 1 rationale = 2 outputs
		{"M=5", 5, 3, 8, 4}, // 3 original + 5 reasoning attempts = 8 inputs, 3 original + 1 rationale = 4 outputs
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create signature with specified number of input fields
			inputs := make([]core.InputField, tc.inputFields)
			for i := 0; i < tc.inputFields; i++ {
				inputs[i] = core.InputField{Field: core.Field{Name: fmt.Sprintf("input_%d", i)}}
			}

			outputs := make([]core.OutputField, tc.inputFields)
			for i := 0; i < tc.inputFields; i++ {
				outputs[i] = core.OutputField{Field: core.Field{Name: fmt.Sprintf("output_%d", i)}}
			}

			signature := core.NewSignature(inputs, outputs)

			// Create MultiChainComparison module
			multiChain := NewMultiChainComparison(signature, tc.M, 0.7)

			// Verify signature modification
			modifiedSignature := multiChain.GetSignature()
			assert.Len(t, modifiedSignature.Inputs, tc.expectedInputs)
			assert.Len(t, modifiedSignature.Outputs, tc.expectedOutputs)

			// Verify rationale is first output
			assert.Equal(t, "rationale", modifiedSignature.Outputs[0].Name)

			// Verify reasoning attempts are added to inputs
			for i := 0; i < tc.M; i++ {
				expectedName := fmt.Sprintf("reasoning_attempt_%d", i+1)
				found := false
				for _, input := range modifiedSignature.Inputs {
					if input.Name == expectedName {
						found = true
						break
					}
				}
				assert.True(t, found, "Expected reasoning attempt field %s not found", expectedName)
			}
		})
	}
}