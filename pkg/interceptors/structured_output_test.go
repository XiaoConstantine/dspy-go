package interceptors

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockLLMWithJSON implements core.LLM with JSON output support.
type mockLLMWithJSON struct {
	core.BaseLLM
	generateWithJSONResult map[string]interface{}
	generateWithJSONError  error
	lastPrompt             string
}

func newMockLLMWithJSON() *mockLLMWithJSON {
	return &mockLLMWithJSON{
		BaseLLM: *core.NewBaseLLM("mock", "mock-model", []core.Capability{
			core.CapabilityJSON,
			core.CapabilityChat,
		}, nil),
	}
}

func (m *mockLLMWithJSON) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLMWithJSON) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.lastPrompt = prompt
	if m.generateWithJSONError != nil {
		return nil, m.generateWithJSONError
	}
	return m.generateWithJSONResult, nil
}

func (m *mockLLMWithJSON) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *mockLLMWithJSON) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithJSON) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithJSON) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

// mockLLMWithoutJSON implements core.LLM without JSON support.
type mockLLMWithoutJSON struct {
	core.BaseLLM
}

func newMockLLMWithoutJSON() *mockLLMWithoutJSON {
	return &mockLLMWithoutJSON{
		BaseLLM: *core.NewBaseLLM("mock", "mock-model", []core.Capability{
			core.CapabilityChat,
		}, nil),
	}
}

func (m *mockLLMWithoutJSON) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLMWithoutJSON) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *mockLLMWithoutJSON) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *mockLLMWithoutJSON) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithoutJSON) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithoutJSON) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func TestDefaultStructuredOutputConfig(t *testing.T) {
	config := DefaultStructuredOutputConfig()

	assert.False(t, config.StrictSchema, "StrictSchema should default to false")
	assert.True(t, config.IncludeDescriptions, "IncludeDescriptions should default to true")
	assert.Empty(t, config.CustomInstructions, "CustomInstructions should be empty by default")
}

func TestSupportsJSONOutput(t *testing.T) {
	tests := []struct {
		name     string
		llm      core.LLM
		expected bool
	}{
		{
			name:     "LLM with JSON capability",
			llm:      newMockLLMWithJSON(),
			expected: true,
		},
		{
			name:     "LLM without JSON capability",
			llm:      newMockLLMWithoutJSON(),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := supportsJSONOutput(tt.llm)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetJSONType(t *testing.T) {
	tests := []struct {
		fieldType core.FieldType
		expected  string
	}{
		{core.FieldTypeText, "string"},
		{core.FieldTypeImage, "string (base64 or URL)"},
		{core.FieldTypeAudio, "string (base64 or URL)"},
		{"unknown", "string"},
	}

	for _, tt := range tests {
		t.Run(string(tt.fieldType), func(t *testing.T) {
			result := getJSONType(tt.fieldType)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBuildStructuredPrompt(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question", core.WithDescription("The question to answer"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The answer to the question"))},
			{Field: core.NewField("confidence", core.WithDescription("Confidence score 0-1"))},
		},
	).WithInstruction("Answer questions accurately")

	inputs := map[string]any{
		"question": "What is the capital of France?",
	}

	config := DefaultStructuredOutputConfig()
	prompt := buildStructuredPrompt(inputs, signature, config)

	// Verify prompt contains key elements
	assert.Contains(t, prompt, "Answer questions accurately")
	assert.Contains(t, prompt, "What is the capital of France?")
	assert.Contains(t, prompt, "\"answer\"")
	assert.Contains(t, prompt, "\"confidence\"")
	assert.Contains(t, prompt, "The answer to the question")
	assert.Contains(t, prompt, "JSON")
}

func TestBuildStructuredPromptWithoutDescriptions(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question")},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The answer"))},
		},
	)

	inputs := map[string]any{"question": "test"}

	config := StructuredOutputConfig{
		IncludeDescriptions: false,
	}
	prompt := buildStructuredPrompt(inputs, signature, config)

	// Should not contain the description text since we disabled it
	assert.NotContains(t, prompt, "### Field Descriptions")
}

func TestTransformJSONResult(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{},
		[]core.OutputField{
			{Field: core.NewField("answer")},
			{Field: core.NewField("confidence")},
		},
	)

	tests := []struct {
		name          string
		result        map[string]interface{}
		config        StructuredOutputConfig
		expectedKeys  []string
		expectError   bool
	}{
		{
			name: "all fields present",
			result: map[string]interface{}{
				"answer":     "Paris",
				"confidence": "0.95",
			},
			config:       DefaultStructuredOutputConfig(),
			expectedKeys: []string{"answer", "confidence"},
			expectError:  false,
		},
		{
			name: "missing field with non-strict mode",
			result: map[string]interface{}{
				"answer": "Paris",
			},
			config:       DefaultStructuredOutputConfig(),
			expectedKeys: []string{"answer", "confidence"},
			expectError:  false,
		},
		{
			name: "missing field with strict mode",
			result: map[string]interface{}{
				"answer": "Paris",
			},
			config: StructuredOutputConfig{
				StrictSchema: true,
			},
			expectError: true,
		},
		{
			name: "extra fields preserved",
			result: map[string]interface{}{
				"answer":     "Paris",
				"confidence": "0.9",
				"extra":      "some extra data",
			},
			config:       DefaultStructuredOutputConfig(),
			expectedKeys: []string{"answer", "confidence", "extra"},
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := transformJSONResult(tt.result, signature, tt.config)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			for _, key := range tt.expectedKeys {
				assert.Contains(t, output, key)
			}
		})
	}
}

func TestStructuredOutputInterceptor_WithJSONCapableLLM(t *testing.T) {
	// Setup mock LLM with JSON capability
	mockLLM := newMockLLMWithJSON()
	mockLLM.generateWithJSONResult = map[string]interface{}{
		"answer":     "Paris",
		"confidence": "0.95",
	}

	// Set as default LLM
	originalLLM := core.GlobalConfig.DefaultLLM
	core.GlobalConfig.DefaultLLM = mockLLM
	defer func() { core.GlobalConfig.DefaultLLM = originalLLM }()

	config := DefaultStructuredOutputConfig()
	interceptor := StructuredOutputInterceptor(config)

	// Create module info with signature
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question")},
		},
		[]core.OutputField{
			{Field: core.NewField("answer")},
			{Field: core.NewField("confidence")},
		},
	)
	info := core.NewModuleInfo("TestModule", "Predict", signature)

	// Create mock handler that should NOT be called
	handlerCalled := false
	mockHandler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return map[string]any{"fallback": true}, nil
	}

	inputs := map[string]any{
		"question": "What is the capital of France?",
	}

	result, err := interceptor(context.Background(), inputs, info, mockHandler)
	require.NoError(t, err)

	// Handler should NOT be called since LLM supports JSON
	assert.False(t, handlerCalled, "Handler should not be called when LLM supports JSON output")

	// Verify result
	assert.Equal(t, "Paris", result["answer"])
	assert.Equal(t, "0.95", result["confidence"])

	// Verify prompt was built correctly
	assert.Contains(t, mockLLM.lastPrompt, "What is the capital of France?")
}

func TestStructuredOutputInterceptor_FallbackWithoutJSONCapability(t *testing.T) {
	// Setup mock LLM WITHOUT JSON capability
	mockLLM := newMockLLMWithoutJSON()

	// Set as default LLM
	originalLLM := core.GlobalConfig.DefaultLLM
	core.GlobalConfig.DefaultLLM = mockLLM
	defer func() { core.GlobalConfig.DefaultLLM = originalLLM }()

	config := DefaultStructuredOutputConfig()
	interceptor := StructuredOutputInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	info := core.NewModuleInfo("TestModule", "Predict", signature)

	// Create mock handler that SHOULD be called as fallback
	handlerCalled := false
	mockHandler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return map[string]any{"fallback": true}, nil
	}

	inputs := map[string]any{"question": "test"}

	result, err := interceptor(context.Background(), inputs, info, mockHandler)
	require.NoError(t, err)

	// Handler SHOULD be called since LLM doesn't support JSON
	assert.True(t, handlerCalled, "Handler should be called when LLM doesn't support JSON output")
	assert.Equal(t, true, result["fallback"])
}

func TestDefaultChainOfThoughtStructuredConfig(t *testing.T) {
	config := DefaultChainOfThoughtStructuredConfig()

	assert.Equal(t, "reasoning", config.ReasoningFieldName)
	assert.True(t, config.IncludeReasoningInOutput)
	assert.False(t, config.StrictSchema)
	assert.True(t, config.IncludeDescriptions)
}

func TestBuildCoTStructuredPrompt(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question")},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The final answer"))},
		},
	).WithInstruction("Think step by step")

	inputs := map[string]any{
		"question": "What is 2+2?",
	}

	config := DefaultChainOfThoughtStructuredConfig()
	prompt := buildCoTStructuredPrompt(inputs, signature, "reasoning", config)

	// Verify prompt contains CoT-specific elements
	assert.Contains(t, prompt, "Think step by step")
	assert.Contains(t, prompt, "What is 2+2?")
	assert.Contains(t, prompt, "\"reasoning\"")
	assert.Contains(t, prompt, "step-by-step reasoning")
	assert.Contains(t, prompt, "\"answer\"")
}

func TestChainOfThoughtStructuredInterceptor(t *testing.T) {
	// Setup mock LLM with JSON capability
	mockLLM := newMockLLMWithJSON()
	mockLLM.generateWithJSONResult = map[string]interface{}{
		"reasoning": "First, I need to add 2 and 2. 2+2=4.",
		"answer":    "4",
	}

	// Set as default LLM
	originalLLM := core.GlobalConfig.DefaultLLM
	core.GlobalConfig.DefaultLLM = mockLLM
	defer func() { core.GlobalConfig.DefaultLLM = originalLLM }()

	config := DefaultChainOfThoughtStructuredConfig()
	interceptor := ChainOfThoughtStructuredInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	info := core.NewModuleInfo("TestCoT", "ChainOfThought", signature)

	handlerCalled := false
	mockHandler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return nil, nil
	}

	inputs := map[string]any{"question": "What is 2+2?"}

	result, err := interceptor(context.Background(), inputs, info, mockHandler)
	require.NoError(t, err)

	assert.False(t, handlerCalled)
	assert.Equal(t, "4", result["answer"])
	assert.Equal(t, "First, I need to add 2 and 2. 2+2=4.", result["reasoning"])
}

func TestStructuredOutputAdapter(t *testing.T) {
	adapter := NewStructuredOutputAdapter()
	assert.NotNil(t, adapter)
	assert.NotNil(t, adapter.GetInterceptor())

	// Test fluent configuration
	adapter.WithStrictSchema().WithCustomInstructions("Be precise").WithoutDescriptions()
	assert.True(t, adapter.config.StrictSchema)
	assert.Equal(t, "Be precise", adapter.config.CustomInstructions)
	assert.False(t, adapter.config.IncludeDescriptions)
}
