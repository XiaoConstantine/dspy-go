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

func TestParseJSONResponse(t *testing.T) {
	// Create a test signature with multiple output fields
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{
			{Field: core.NewField("rationale", core.WithCustomPrefix("Rationale:"))},
			{Field: core.NewField("answer", core.WithCustomPrefix("Answer:"))},
			{Field: core.NewField("confidence", core.WithCustomPrefix("Confidence:"))},
		},
	)

	tests := []struct {
		name     string
		content  string
		expected string
	}{
		{
			name: "Valid JSON with all fields",
			content: "Here's my response:\n\n```json\n{\n  \"rationale\": \"This is the reasoning behind the answer\",\n  \"answer\": \"42\",\n  \"confidence\": \"high\"\n}\n```\n\nThat's the response.",
			expected: "Rationale:\nThis is the reasoning behind the answer\n\nAnswer:\n42\n\nConfidence:\nhigh",
		},
		{
			name: "Valid JSON with subset of fields",
			content: "```json\n{\n  \"rationale\": \"Only rationale provided\",\n  \"answer\": \"incomplete\"\n}\n```",
			expected: "Rationale:\nOnly rationale provided\n\nAnswer:\nincomplete",
		},
		{
			name: "JSON with extra fields (should ignore)",
			content: "```json\n{\n  \"rationale\": \"Test rationale\",\n  \"answer\": \"test answer\",\n  \"confidence\": \"medium\",\n  \"extra_field\": \"should be ignored\"\n}\n```",
			expected: "Rationale:\nTest rationale\n\nAnswer:\ntest answer\n\nConfidence:\nmedium",
		},
		{
			name: "JSON with non-string values",
			content: "```json\n{\n  \"rationale\": \"Number test\",\n  \"answer\": 123,\n  \"confidence\": true\n}\n```",
			expected: "Rationale:\nNumber test\n\nAnswer:\n123\n\nConfidence:\ntrue",
		},
		{
			name: "Malformed JSON (should fall back)",
			content: "```json\n{\n  \"rationale\": \"incomplete json\"\n  \"answer\": \"missing comma\"\n}\n```",
			expected: "```json\n{\n  \"rationale\": \"incomplete json\"\n  \"answer\": \"missing comma\"\n}\n```",
		},
		{
			name: "No JSON markers (should fall back)",
			content: "This is just plain text\nrationale: Some reasoning\nanswer: Some answer",
			expected: "This is just plain text\nrationale: Some reasoning\nanswer: Some answer",
		},
		{
			name: "Empty JSON object",
			content: "```json\n{}\n```",
			expected: "",
		},
		{
			name: "Missing closing marker (should fall back)",
			content: "```json\n{\n  \"rationale\": \"test\",\n  \"answer\": \"test\"\n}",
			expected: "```json\n{\n  \"rationale\": \"test\",\n  \"answer\": \"test\"\n}",
		},
		{
			name: "Multiple JSON blocks (should use first)",
			content: "First block:\n```json\n{\n  \"rationale\": \"first\",\n  \"answer\": \"first answer\"\n}\n```\n\nSecond block:\n```json\n{\n  \"rationale\": \"second\",\n  \"answer\": \"second answer\"\n}\n```",
			expected: "Rationale:\nfirst\n\nAnswer:\nfirst answer",
		},
		{
			name: "JSON with newlines in values",
			content: "```json\n{\n  \"rationale\": \"This is a\\nmulti-line\\nrationale\",\n  \"answer\": \"Simple answer\"\n}\n```",
			expected: "Rationale:\nThis is a\nmulti-line\nrationale\n\nAnswer:\nSimple answer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseJSONResponse(tt.content, signature)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseJSONResponse_EmptySignature(t *testing.T) {
	// Test with signature that has no output fields
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{},
	)

	content := "```json\n{\n  \"rationale\": \"This won't be used\",\n  \"answer\": \"This won't be used either\"\n}\n```"

	result := parseJSONResponse(content, signature)
	assert.Equal(t, "", result)
}

func TestParseJSONResponse_FieldOrderPreservation(t *testing.T) {
	// Test that fields are processed in signature order, not JSON order
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{
			{Field: core.NewField("third", core.WithCustomPrefix("Third:"))},
			{Field: core.NewField("first", core.WithCustomPrefix("First:"))},
			{Field: core.NewField("second", core.WithCustomPrefix("Second:"))},
		},
	)

	content := "```json\n{\n  \"first\": \"1st value\",\n  \"second\": \"2nd value\",\n  \"third\": \"3rd value\"\n}\n```"

	expected := "Third:\n3rd value\n\nFirst:\n1st value\n\nSecond:\n2nd value"

	result := parseJSONResponse(content, signature)
	assert.Equal(t, expected, result)
}
