package modules

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
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
			name:     "Valid JSON with all fields",
			content:  "Here's my response:\n\n```json\n{\n  \"rationale\": \"This is the reasoning behind the answer\",\n  \"answer\": \"42\",\n  \"confidence\": \"high\"\n}\n```\n\nThat's the response.",
			expected: "Rationale:\nThis is the reasoning behind the answer\n\nAnswer:\n42\n\nConfidence:\nhigh",
		},
		{
			name:     "Valid JSON with subset of fields",
			content:  "```json\n{\n  \"rationale\": \"Only rationale provided\",\n  \"answer\": \"incomplete\"\n}\n```",
			expected: "Rationale:\nOnly rationale provided\n\nAnswer:\nincomplete",
		},
		{
			name:     "JSON with extra fields (should ignore)",
			content:  "```json\n{\n  \"rationale\": \"Test rationale\",\n  \"answer\": \"test answer\",\n  \"confidence\": \"medium\",\n  \"extra_field\": \"should be ignored\"\n}\n```",
			expected: "Rationale:\nTest rationale\n\nAnswer:\ntest answer\n\nConfidence:\nmedium",
		},
		{
			name:     "JSON with non-string values",
			content:  "```json\n{\n  \"rationale\": \"Number test\",\n  \"answer\": 123,\n  \"confidence\": true\n}\n```",
			expected: "Rationale:\nNumber test\n\nAnswer:\n123\n\nConfidence:\ntrue",
		},
		{
			name:     "Malformed JSON (should fall back)",
			content:  "```json\n{\n  \"rationale\": \"incomplete json\"\n  \"answer\": \"missing comma\"\n}\n```",
			expected: "```json\n{\n  \"rationale\": \"incomplete json\"\n  \"answer\": \"missing comma\"\n}\n```",
		},
		{
			name:     "No JSON markers (should fall back)",
			content:  "This is just plain text\nrationale: Some reasoning\nanswer: Some answer",
			expected: "This is just plain text\nrationale: Some reasoning\nanswer: Some answer",
		},
		{
			name:     "Empty JSON object",
			content:  "```json\n{}\n```",
			expected: "",
		},
		{
			name:     "Missing closing marker (should fall back)",
			content:  "```json\n{\n  \"rationale\": \"test\",\n  \"answer\": \"test\"\n}",
			expected: "```json\n{\n  \"rationale\": \"test\",\n  \"answer\": \"test\"\n}",
		},
		{
			name:     "Multiple JSON blocks (should use first)",
			content:  "First block:\n```json\n{\n  \"rationale\": \"first\",\n  \"answer\": \"first answer\"\n}\n```\n\nSecond block:\n```json\n{\n  \"rationale\": \"second\",\n  \"answer\": \"second answer\"\n}\n```",
			expected: "Rationale:\nfirst\n\nAnswer:\nfirst answer",
		},
		{
			name:     "JSON with newlines in values",
			content:  "```json\n{\n  \"rationale\": \"This is a\\nmulti-line\\nrationale\",\n  \"answer\": \"Simple answer\"\n}\n```",
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

// Type-safe Predict module tests

type TestQAInputs struct {
	Question string `dspy:"question,required" description:"The question to answer"`
	Context  string `dspy:"context,required" description:"Context for answering"`
}

type TestQAOutputs struct {
	Answer     string `dspy:"answer" description:"The generated answer" prefix:"Answer:"`
	Confidence int    `dspy:"confidence" description:"Confidence score" prefix:"Confidence:"`
}

func TestPredictTyped(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `Answer:
		Machine learning is a subset of AI

		Confidence:
		85`,
	}, nil)

	// Create a typed Predict module
	predict := NewTypedPredict[TestQAInputs, TestQAOutputs]()
	predict.SetLLM(mockLLM)

	// Test the ProcessTyped method
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := TestQAInputs{
		Question: "What is machine learning?",
		Context:  "ML is a type of artificial intelligence",
	}

	outputs, err := ProcessTyped[TestQAInputs, TestQAOutputs](ctx, predict, inputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Contains(t, outputs.Answer, "Machine learning")
	assert.Equal(t, 85, outputs.Confidence)

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestPredictTypedWithValidation(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `Answer:
		Deep learning uses neural networks

		Confidence:
		90`,
	}, nil)

	// Create a typed Predict module
	predict := NewTypedPredict[TestQAInputs, TestQAOutputs]()
	predict.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	// Test with valid inputs
	validInputs := TestQAInputs{
		Question: "What is deep learning?",
		Context:  "Deep learning is a subset of machine learning",
	}

	outputs, err := ProcessTypedWithValidation[TestQAInputs, TestQAOutputs](ctx, predict, validInputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Contains(t, outputs.Answer, "Deep learning")
	assert.Equal(t, 90, outputs.Confidence)

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestPredictTypedWithValidation_InvalidInput(t *testing.T) {
	// Create a typed Predict module (no need for mock LLM since validation should fail first)
	predict := NewTypedPredict[TestQAInputs, TestQAOutputs]()

	ctx := context.Background()

	// Test with invalid inputs (missing required field)
	invalidInputs := TestQAInputs{
		Question: "What is AI?",
		// Context missing - should fail validation
	}

	outputs, err := ProcessTypedWithValidation[TestQAInputs, TestQAOutputs](ctx, predict, invalidInputs)

	// Assert that validation failed
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "typed input validation failed")
	assert.Contains(t, err.Error(), "required input field 'context' cannot be empty")
	assert.Empty(t, outputs.Answer)
	assert.Equal(t, 0, outputs.Confidence)
}

func TestPredictTypedWithMapInputs(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `Answer:
		Artificial intelligence systems

		Confidence:
		95`,
	}, nil)

	// Create a regular Predict module
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewTextField("question", core.WithDescription("The question"))},
			{Field: core.NewTextField("context", core.WithDescription("Context"))},
		},
		[]core.OutputField{
			{Field: core.NewTextField("answer", core.WithCustomPrefix("Answer:"))},
			{Field: core.NewTextField("confidence", core.WithCustomPrefix("Confidence:"))},
		},
	)
	predict := NewPredict(signature).WithTextOutput() // These tests expect prefix-based parsing
	predict.SetLLM(mockLLM)

	ctx := context.Background()

	// Test that ProcessTyped works with map inputs and typed outputs
	legacyInputs := map[string]any{
		"question": "What is AI?",
		"context":  "AI stands for artificial intelligence",
	}

	outputs, err := ProcessTyped[map[string]any, TestQAOutputs](ctx, predict, legacyInputs)

	// Assert the results
	assert.NoError(t, err)
	assert.Contains(t, outputs.Answer, "Artificial intelligence")
	assert.Equal(t, 95, outputs.Confidence)

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestPredictTypedWithMapOutputs(t *testing.T) {
	// Create a mock LLM
	mockLLM := new(testutil.MockLLM)

	// Set up the expected behavior
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `Answer:
		Machine learning algorithms

		Confidence:
		80`,
	}, nil)

	// Create a regular Predict module
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewTextField("question", core.WithDescription("The question"))},
			{Field: core.NewTextField("context", core.WithDescription("Context"))},
		},
		[]core.OutputField{
			{Field: core.NewTextField("answer", core.WithCustomPrefix("Answer:"))},
			{Field: core.NewTextField("confidence", core.WithCustomPrefix("Confidence:"))},
		},
	)
	predict := NewPredict(signature).WithTextOutput() // These tests expect prefix-based parsing
	predict.SetLLM(mockLLM)

	ctx := context.Background()

	// Test that ProcessTyped works with typed inputs and map outputs
	inputs := TestQAInputs{
		Question: "What is ML?",
		Context:  "ML means machine learning",
	}

	outputs, err := ProcessTyped[TestQAInputs, map[string]any](ctx, predict, inputs)

	// Assert the results
	assert.NoError(t, err)

	answer, exists := outputs["answer"]
	assert.True(t, exists)
	assert.Contains(t, answer.(string), "Machine learning")

	confidence, exists := outputs["confidence"]
	assert.True(t, exists)
	assert.Equal(t, "80", confidence.(string))

	// Verify that the mock was called as expected
	mockLLM.AssertExpectations(t)
}

func TestNewTypedPredict(t *testing.T) {
	// Test that NewTypedPredict creates a properly configured module
	predict := NewTypedPredict[TestQAInputs, TestQAOutputs]()

	// Verify the module is properly initialized
	assert.NotNil(t, predict)
	assert.Contains(t, predict.GetDisplayName(), "TypedPredict")
	assert.Equal(t, "Predict", predict.GetModuleType())

	// Verify the signature was properly converted
	signature := predict.GetSignature()
	assert.Len(t, signature.Inputs, 2)
	assert.Len(t, signature.Outputs, 2)

	// Check input fields
	questionField := signature.Inputs[0].Field
	contextField := signature.Inputs[1].Field
	assert.Equal(t, "question", questionField.Name)
	assert.Equal(t, "context", contextField.Name)

	// Check output fields
	answerField := signature.Outputs[0].Field
	confidenceField := signature.Outputs[1].Field
	assert.Equal(t, "answer", answerField.Name)
	assert.Equal(t, "confidence", confidenceField.Name)
	assert.Equal(t, "Answer:", answerField.Prefix)
	assert.Equal(t, "Confidence:", confidenceField.Prefix)
}

// Benchmark type-safe vs legacy processing.
func BenchmarkPredictTyped(b *testing.B) {
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: "Answer:\nTest answer\n\nConfidence:\n85",
	}, nil)

	predict := NewTypedPredict[TestQAInputs, TestQAOutputs]()
	predict.SetLLM(mockLLM)

	ctx := context.Background()
	inputs := TestQAInputs{
		Question: "Test question",
		Context:  "Test context",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ProcessTyped[TestQAInputs, TestQAOutputs](ctx, predict, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredictLegacy(b *testing.B) {
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: "Answer:\nTest answer\n\nConfidence:\n85",
	}, nil)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewTextField("question", core.WithDescription("The question"))},
			{Field: core.NewTextField("context", core.WithDescription("Context"))},
		},
		[]core.OutputField{
			{Field: core.NewTextField("answer", core.WithCustomPrefix("Answer:"))},
			{Field: core.NewTextField("confidence", core.WithCustomPrefix("Confidence:"))},
		},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	ctx := context.Background()
	inputs := map[string]any{
		"question": "Test question",
		"context":  "Test context",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := predict.Process(ctx, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestPredict_WithXMLOutput(t *testing.T) {
	// Test XML output functionality in Predict module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)

	// Mock LLM
	mockLLM := new(testutil.MockLLM)
	predict.SetLLM(mockLLM)

	// Test 1: Predict without XML output (default behavior)
	t.Run("without_xml_output", func(t *testing.T) {
		assert.False(t, predict.IsXMLModeEnabled())
		assert.Nil(t, predict.GetXMLConfig())
	})

	// Test 2: Enable XML output
	t.Run("enable_xml_output", func(t *testing.T) {
		xmlConfig := interceptors.DefaultXMLConfig().
			WithMaxSize(1024).
			WithStrictParsing(true)

		predict.WithXMLOutput(xmlConfig)

		assert.True(t, predict.IsXMLModeEnabled())
		assert.NotNil(t, predict.GetXMLConfig())
		assert.Equal(t, int64(1024), predict.GetXMLConfig().MaxSize)
		assert.True(t, predict.GetXMLConfig().StrictParsing)
	})

	// Test 3: XML config with flexible parsing
	t.Run("flexible_xml_config", func(t *testing.T) {
		xmlConfig := interceptors.FlexibleXMLConfig().
			WithMaxSize(2048).
			WithValidation(true)

		flexiblePredict := NewPredict(signature).WithXMLOutput(xmlConfig)

		assert.True(t, flexiblePredict.IsXMLModeEnabled())
		assert.NotNil(t, flexiblePredict.GetXMLConfig())
		assert.Equal(t, int64(2048), flexiblePredict.GetXMLConfig().MaxSize)
		assert.True(t, flexiblePredict.GetXMLConfig().ValidateXML)
		assert.False(t, flexiblePredict.GetXMLConfig().StrictParsing)
	})

	// Test 4: Chain multiple configuration methods
	t.Run("chained_configuration", func(t *testing.T) {
		xmlConfig := interceptors.SecureXMLConfig()

		chainedPredict := NewPredict(signature).
			WithName("XMLPredict").
			WithXMLOutput(xmlConfig)

		assert.Equal(t, "XMLPredict", chainedPredict.GetDisplayName())
		assert.True(t, chainedPredict.IsXMLModeEnabled())
		assert.NotNil(t, chainedPredict.GetXMLConfig())
	})
}

func TestPredict_XMLOutputIntegration(t *testing.T) {
	// Integration test to ensure XML output works with the underlying module processing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "task"}}},
		[]core.OutputField{{Field: core.NewField("result")}},
	)

	// Mock LLM with XML responses
	mockLLM := new(testutil.MockLLM)

	// Mock LLM response that would be formatted by XML interceptors
	resp := &core.LLMResponse{Content: `<response><result>Integration test successful</result></response>`}
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(resp, nil)

	// Create Predict with XML output enabled
	xmlConfig := interceptors.DefaultXMLConfig().
		WithStrictParsing(true).
		WithValidation(true)

	predict := NewPredict(signature).WithXMLOutput(xmlConfig)
	predict.SetLLM(mockLLM)

	// Execute
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	input := map[string]interface{}{"task": "Test XML output integration"}
	result, err := predict.Process(ctx, input)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Contains(t, result["result"], "Integration test successful")

	// Verify mocks
	mockLLM.AssertExpectations(t)
}

func TestPredict_XMLOutputClone(t *testing.T) {
	// Test that XML configuration is properly cloned
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)

	xmlConfig := interceptors.DefaultXMLConfig().WithMaxSize(512)
	original := NewPredict(signature).WithXMLOutput(xmlConfig)

	// Clone the module
	cloned := original.Clone().(*Predict)

	// Verify XML configuration is preserved
	assert.True(t, cloned.IsXMLModeEnabled())
	assert.NotNil(t, cloned.GetXMLConfig())
	assert.Equal(t, int64(512), cloned.GetXMLConfig().MaxSize)

	// Verify they are separate instances (deep copy)
	assert.NotSame(t, original.GetXMLConfig(), cloned.GetXMLConfig())

	// Modify original config to verify independence
	original.xmlConfig.MaxSize = 1024
	assert.Equal(t, int64(1024), original.GetXMLConfig().MaxSize)
	assert.Equal(t, int64(512), cloned.GetXMLConfig().MaxSize) // Should remain unchanged
}

func TestPredict_XMLOutputBackwardCompatibility(t *testing.T) {
	// Test that existing Predict usage continues to work unchanged
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: `answer: 42`,
	}, nil)

	// Create a standard Predict module (no XML)
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predict := NewPredict(signature)
	predict.SetLLM(mockLLM)

	// Test that it works exactly as before
	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"question": "What is the meaning of life?"}
	outputs, err := predict.Process(ctx, inputs)

	assert.NoError(t, err)
	assert.Equal(t, "42", outputs["answer"])
	assert.False(t, predict.IsXMLModeEnabled())
	assert.Nil(t, predict.GetXMLConfig())

	mockLLM.AssertExpectations(t)
}

func TestPredict_XMLModeBypassesParsing(t *testing.T) {
	// Test that XML mode bypasses traditional parsing logic for better performance
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{{Field: core.NewField("output")}},
	)

	mockLLM := new(testutil.MockLLM)
	// XML interceptors will handle this response
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(&core.LLMResponse{
		Content: `output: XML interceptor processed result`,
	}, nil)

	// Create Predict with XML output enabled - this uses interceptors instead of traditional parsing
	xmlConfig := interceptors.DefaultXMLConfig()
	predict := NewPredict(signature).WithXMLOutput(xmlConfig)
	predict.SetLLM(mockLLM)

	ctx := context.Background()
	ctx = core.WithExecutionState(ctx)

	inputs := map[string]any{"input": "test input"}
	outputs, err := predict.Process(ctx, inputs)

	assert.NoError(t, err)
	assert.NotNil(t, outputs)

	// The key point is that XML mode enabled interceptors, which bypass traditional parsing
	// This test verifies the flow works (the exact output format depends on interceptor implementation)
	assert.True(t, predict.IsXMLModeEnabled(), "XML mode should be enabled")
	assert.NotNil(t, predict.GetXMLConfig(), "XML config should be present")
	assert.Greater(t, len(predict.GetInterceptors()), 0, "Interceptors should be configured")

	mockLLM.AssertExpectations(t)
}
