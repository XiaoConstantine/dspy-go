package core

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test struct definitions.
type TestInputs struct {
	Question string `dspy:"question,required" description:"The question to answer"`
	Context  string `dspy:"context,required" description:"Context for answering"`
	Optional string `dspy:"optional" description:"Optional parameter"`
}

type TestOutputs struct {
	Answer     string `dspy:"answer" description:"The generated answer" prefix:"Answer:"`
	Confidence int    `dspy:"confidence" description:"Confidence score"`
}

type TestComplexInputs struct {
	Text      string `dspy:"text,required"`
	ImageData []byte `dspy:"image,required"`
	Count     *int   `dspy:"count"`
}

func TestNewTypedSignature(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()

	assert.NotNil(t, sig)
	assert.Equal(t, reflect.TypeOf(TestInputs{}), sig.GetInputType())
	assert.Equal(t, reflect.TypeOf(TestOutputs{}), sig.GetOutputType())
}

func TestStructTagParsing(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()
	metadata := sig.GetFieldMetadata()

	// Test input fields
	require.Len(t, metadata.Inputs, 3)

	questionField := metadata.Inputs[0]
	assert.Equal(t, "question", questionField.Name)
	assert.True(t, questionField.Required)
	assert.Equal(t, "The question to answer", questionField.Description)
	assert.Equal(t, FieldTypeText, questionField.Type)

	contextField := metadata.Inputs[1]
	assert.Equal(t, "context", contextField.Name)
	assert.True(t, contextField.Required)

	optionalField := metadata.Inputs[2]
	assert.Equal(t, "optional", optionalField.Name)
	assert.False(t, optionalField.Required)

	// Test output fields
	require.Len(t, metadata.Outputs, 2)

	answerField := metadata.Outputs[0]
	assert.Equal(t, "answer", answerField.Name)
	assert.Equal(t, "Answer:", answerField.Prefix)
	assert.Equal(t, "The generated answer", answerField.Description)

	confidenceField := metadata.Outputs[1]
	assert.Equal(t, "confidence", confidenceField.Name)
	assert.Equal(t, "confidence:", confidenceField.Prefix) // Default prefix
}

func TestFieldTypeInference(t *testing.T) {
	sig := NewTypedSignature[TestComplexInputs, TestOutputs]()
	metadata := sig.GetFieldMetadata()

	require.Len(t, metadata.Inputs, 3)

	// Text field should be FieldTypeText
	textField := metadata.Inputs[0]
	assert.Equal(t, FieldTypeText, textField.Type)

	// Byte slice should be inferred as image
	imageField := metadata.Inputs[1]
	assert.Equal(t, FieldTypeImage, imageField.Type)

	// Pointer field should work
	countField := metadata.Inputs[2]
	assert.Equal(t, FieldTypeText, countField.Type) // int inferred as text
}

func TestInputValidation(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()

	tests := []struct {
		name    string
		input   TestInputs
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid input",
			input: TestInputs{
				Question: "What is AI?",
				Context:  "AI is artificial intelligence",
				Optional: "some value",
			},
			wantErr: false,
		},
		{
			name: "missing required field",
			input: TestInputs{
				Question: "What is AI?",
				// Context missing
				Optional: "some value",
			},
			wantErr: true,
			errMsg:  "required input field 'context' cannot be empty",
		},
		{
			name: "empty required field",
			input: TestInputs{
				Question: "", // Empty required field
				Context:  "AI is artificial intelligence",
			},
			wantErr: true,
			errMsg:  "required input field 'question' cannot be empty",
		},
		{
			name: "missing optional field is ok",
			input: TestInputs{
				Question: "What is AI?",
				Context:  "AI is artificial intelligence",
				// Optional field missing - should be OK
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := sig.ValidateInput(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestOutputValidation(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()

	// Test valid output
	output := TestOutputs{
		Answer:     "AI is artificial intelligence",
		Confidence: 85,
	}

	err := sig.ValidateOutput(output)
	assert.NoError(t, err)
}

func TestPointerTypes(t *testing.T) {
	sig := NewTypedSignature[*TestInputs, *TestOutputs]()

	assert.Equal(t, reflect.TypeOf(TestInputs{}), sig.GetInputType())
	assert.Equal(t, reflect.TypeOf(TestOutputs{}), sig.GetOutputType())

	// Test validation with pointer
	input := &TestInputs{
		Question: "What is AI?",
		Context:  "AI is artificial intelligence",
	}

	err := sig.ValidateInput(input)
	assert.NoError(t, err)
}

func TestNilPointerValidation(t *testing.T) {
	sig := NewTypedSignature[*TestInputs, *TestOutputs]()

	// Test nil pointer
	err := sig.ValidateInput((*TestInputs)(nil))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input cannot be nil")
}

func TestToLegacySignature(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()
	legacy := sig.ToLegacySignature()

	// Test input fields conversion
	require.Len(t, legacy.Inputs, 3)

	assert.Equal(t, "question", legacy.Inputs[0].Name)
	assert.Equal(t, "The question to answer", legacy.Inputs[0].Description)
	assert.Equal(t, FieldTypeText, legacy.Inputs[0].Type)

	// Test output fields conversion
	require.Len(t, legacy.Outputs, 2)

	assert.Equal(t, "answer", legacy.Outputs[0].Name)
	assert.Equal(t, "Answer:", legacy.Outputs[0].Prefix)
	assert.Equal(t, "The generated answer", legacy.Outputs[0].Description)
}

func TestFromLegacySignature(t *testing.T) {
	// Create a legacy signature
	legacy := NewSignature(
		[]InputField{
			{Field: NewTextField("question", WithDescription("The question"))},
			{Field: NewTextField("context", WithDescription("Context"))},
		},
		[]OutputField{
			{Field: NewTextField("answer", WithDescription("The answer"))},
		},
	).WithInstruction("Answer the question using the context")

	// Convert to typed signature
	typed := FromLegacySignature(legacy)

	metadata := typed.GetFieldMetadata()
	assert.Equal(t, "Answer the question using the context", metadata.Instruction)

	require.Len(t, metadata.Inputs, 2)
	assert.Equal(t, "question", metadata.Inputs[0].Name)
	assert.True(t, metadata.Inputs[0].Required) // Legacy fields assumed required

	require.Len(t, metadata.Outputs, 1)
	assert.Equal(t, "answer", metadata.Outputs[0].Name)
	assert.False(t, metadata.Outputs[0].Required) // Outputs not required
}

func TestWithInstruction(t *testing.T) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()
	instruction := "Answer the question using the provided context"

	// Note: This test assumes we'll implement a fluent API for WithInstruction
	// For now, we'll test the current implementation
	if impl, ok := sig.(*typedSignatureImpl[TestInputs, TestOutputs]); ok {
		impl.instruction = instruction
		impl.metadata.Instruction = instruction

		metadata := sig.GetFieldMetadata()
		assert.Equal(t, instruction, metadata.Instruction)

		legacy := sig.ToLegacySignature()
		assert.Equal(t, instruction, legacy.Instruction)
	}
}

// Benchmark tests to ensure performance is acceptable.
func BenchmarkNewTypedSignature(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = NewTypedSignature[TestInputs, TestOutputs]()
	}
}

func BenchmarkStructTagParsing(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sig := NewTypedSignature[TestInputs, TestOutputs]()
		_ = sig.GetFieldMetadata()
	}
}

func BenchmarkInputValidation(b *testing.B) {
	sig := NewTypedSignature[TestInputs, TestOutputs]()
	input := TestInputs{
		Question: "What is AI?",
		Context:  "AI is artificial intelligence",
		Optional: "test",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sig.ValidateInput(input)
	}
}
