package interceptors

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockTool implements the core.Tool interface for testing.
type mockTool struct {
	name        string
	description string
	schema      models.InputSchema
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return m.description }
func (m *mockTool) InputSchema() models.InputSchema {
	return m.schema
}
func (m *mockTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:        m.name,
		Description: m.description,
		InputSchema: m.schema,
	}
}
func (m *mockTool) CanHandle(ctx context.Context, intent string) bool {
	return true
}
func (m *mockTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{
		Data: "mock result",
	}, nil
}
func (m *mockTool) Validate(params map[string]interface{}) error {
	return nil
}

// mockLLMWithFunctionCalling implements core.LLM with function calling support.
type mockLLMWithFunctionCalling struct {
	core.BaseLLM
	generateWithFunctionsResult map[string]interface{}
	generateWithFunctionsError  error
	lastFunctions               []map[string]interface{}
	lastPrompt                  string
}

func newMockLLMWithFunctionCalling() *mockLLMWithFunctionCalling {
	return &mockLLMWithFunctionCalling{
		BaseLLM: *core.NewBaseLLM("mock", "mock-model", []core.Capability{
			core.CapabilityToolCalling,
			core.CapabilityChat,
		}, nil),
	}
}

func (m *mockLLMWithFunctionCalling) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLMWithFunctionCalling) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"result": "mock"}, nil
}

func (m *mockLLMWithFunctionCalling) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.lastPrompt = prompt
	m.lastFunctions = functions
	if m.generateWithFunctionsError != nil {
		return nil, m.generateWithFunctionsError
	}
	return m.generateWithFunctionsResult, nil
}

func (m *mockLLMWithFunctionCalling) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithFunctionCalling) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithFunctionCalling) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

// mockLLMWithoutFunctionCalling implements core.LLM without function calling support.
type mockLLMWithoutFunctionCalling struct {
	core.BaseLLM
}

func newMockLLMWithoutFunctionCalling() *mockLLMWithoutFunctionCalling {
	return &mockLLMWithoutFunctionCalling{
		BaseLLM: *core.NewBaseLLM("mock", "mock-model", []core.Capability{
			core.CapabilityChat,
		}, nil),
	}
}

func (m *mockLLMWithoutFunctionCalling) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *mockLLMWithoutFunctionCalling) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"result": "mock"}, nil
}

func (m *mockLLMWithoutFunctionCalling) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *mockLLMWithoutFunctionCalling) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithoutFunctionCalling) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}

func (m *mockLLMWithoutFunctionCalling) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func TestDefaultFunctionCallingConfig(t *testing.T) {
	config := DefaultFunctionCallingConfig()

	assert.False(t, config.StrictMode, "StrictMode should default to false")
	assert.True(t, config.IncludeFinishTool, "IncludeFinishTool should default to true")
	assert.NotEmpty(t, config.FinishToolDescription, "FinishToolDescription should have a default")
}

func TestBuildFunctionSchemas(t *testing.T) {
	registry := tools.NewInMemoryToolRegistry()

	// Register a mock tool
	mockTool := &mockTool{
		name:        "search",
		description: "Search for information",
		schema: models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"query": {
					Type:        "string",
					Description: "The search query",
					Required:    true,
				},
				"limit": {
					Type:        "integer",
					Description: "Maximum results",
					Required:    false,
				},
			},
		},
	}
	err := registry.Register(mockTool)
	require.NoError(t, err)

	config := FunctionCallingConfig{
		ToolRegistry:      registry,
		IncludeFinishTool: true,
	}

	functions, err := buildFunctionSchemas(config)
	require.NoError(t, err)

	// Should have search tool + Finish tool
	assert.Len(t, functions, 2)

	// Find the search tool
	var searchFunc map[string]interface{}
	var finishFunc map[string]interface{}
	for _, f := range functions {
		if f["name"] == "search" {
			searchFunc = f
		}
		if f["name"] == "Finish" {
			finishFunc = f
		}
	}

	require.NotNil(t, searchFunc, "search function should be present")
	require.NotNil(t, finishFunc, "Finish function should be present")

	// Verify search function structure
	assert.Equal(t, "search", searchFunc["name"])
	assert.Equal(t, "Search for information", searchFunc["description"])

	params := searchFunc["parameters"].(map[string]interface{})
	assert.Equal(t, "object", params["type"])

	required := params["required"].([]string)
	assert.Contains(t, required, "query")

	// Verify Finish function
	assert.Equal(t, "Finish", finishFunc["name"])
	finishParams := finishFunc["parameters"].(map[string]interface{})
	finishRequired := finishParams["required"].([]string)
	assert.Contains(t, finishRequired, "answer")
}

func TestBuildFunctionSchemasWithoutFinish(t *testing.T) {
	registry := tools.NewInMemoryToolRegistry()

	mockTool := &mockTool{
		name:        "search",
		description: "Search for information",
		schema: models.InputSchema{
			Type:       "object",
			Properties: map[string]models.ParameterSchema{},
		},
	}
	err := registry.Register(mockTool)
	require.NoError(t, err)

	config := FunctionCallingConfig{
		ToolRegistry:      registry,
		IncludeFinishTool: false, // Disable Finish tool
	}

	functions, err := buildFunctionSchemas(config)
	require.NoError(t, err)

	// Should only have the search tool
	assert.Len(t, functions, 1)
	assert.Equal(t, "search", functions[0]["name"])
}

func TestBuildPromptFromInputs(t *testing.T) {
	tests := []struct {
		name     string
		inputs   map[string]any
		info     *core.ModuleInfo
		contains []string
	}{
		{
			name: "with task",
			inputs: map[string]any{
				"task": "Find the weather",
			},
			info:     nil,
			contains: []string{"Task: Find the weather"},
		},
		{
			name: "with question",
			inputs: map[string]any{
				"question": "What is the capital of France?",
			},
			info:     nil,
			contains: []string{"Question: What is the capital of France?"},
		},
		{
			name: "with observation",
			inputs: map[string]any{
				"task":        "Find info",
				"observation": "Previous search returned: Paris",
			},
			info:     nil,
			contains: []string{"Task: Find info", "Observation from previous action: Previous search returned: Paris"},
		},
		{
			name: "with conversation context",
			inputs: map[string]any{
				"task":                 "Continue task",
				"conversation_context": "Iteration 1: Searched for Paris",
			},
			info:     nil,
			contains: []string{"Previous conversation:", "Iteration 1: Searched for Paris"},
		},
		{
			name: "with signature instruction",
			inputs: map[string]any{
				"task": "Do something",
			},
			info: &core.ModuleInfo{
				ModuleName: "test",
				Signature: core.Signature{
					Instruction: "You are a helpful assistant",
				},
			},
			contains: []string{"You are a helpful assistant", "Task: Do something"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prompt := buildPromptFromInputs(tt.inputs, tt.info)

			for _, expected := range tt.contains {
				assert.Contains(t, prompt, expected)
			}
		})
	}
}

func TestTransformFunctionCallResult(t *testing.T) {
	tests := []struct {
		name           string
		result         map[string]interface{}
		originalInputs map[string]any
		expectedAction map[string]interface{}
		expectedAnswer string
	}{
		{
			name: "function call with arguments",
			result: map[string]interface{}{
				"function_call": map[string]interface{}{
					"name": "search",
					"arguments": map[string]interface{}{
						"query": "weather in Paris",
					},
				},
			},
			originalInputs: map[string]any{"task": "find weather"},
			expectedAction: map[string]interface{}{
				"tool_name": "search",
				"arguments": map[string]interface{}{
					"query": "weather in Paris",
				},
			},
		},
		{
			name: "Finish function call",
			result: map[string]interface{}{
				"function_call": map[string]interface{}{
					"name": "Finish",
					"arguments": map[string]interface{}{
						"answer":    "Paris is the capital of France",
						"reasoning": "Based on the search results",
					},
				},
			},
			originalInputs: map[string]any{},
			expectedAnswer: "Paris is the capital of France",
		},
		{
			name: "text response (no function call)",
			result: map[string]interface{}{
				"content": "I already know the answer is 42",
			},
			originalInputs: map[string]any{},
			// When LLM responds with text (no function call), we wrap it as a Finish action
			expectedAction: map[string]interface{}{
				"tool_name": "Finish",
				"arguments": map[string]interface{}{
					"answer": "I already know the answer is 42",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := transformFunctionCallResult(tt.result, tt.originalInputs)
			require.NoError(t, err)

			if tt.expectedAction != nil {
				action := output["action"].(map[string]interface{})
				assert.Equal(t, tt.expectedAction["tool_name"], action["tool_name"])

				// Also verify arguments if expected
				if expectedArgs, ok := tt.expectedAction["arguments"].(map[string]interface{}); ok {
					actualArgs := action["arguments"].(map[string]interface{})
					for key, expectedVal := range expectedArgs {
						assert.Equal(t, expectedVal, actualArgs[key], "argument %s mismatch", key)
					}
				}
			}

			if tt.expectedAnswer != "" {
				assert.Equal(t, tt.expectedAnswer, output["answer"])
			}
		})
	}
}

func TestSupportsToolCalling(t *testing.T) {
	tests := []struct {
		name     string
		llm      core.LLM
		expected bool
	}{
		{
			name:     "LLM with tool calling capability",
			llm:      newMockLLMWithFunctionCalling(),
			expected: true,
		},
		{
			name:     "LLM without tool calling capability",
			llm:      newMockLLMWithoutFunctionCalling(),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := supportsToolCalling(tt.llm)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestNativeFunctionCallingInterceptor_WithToolCallingLLM(t *testing.T) {
	// Setup mock LLM with function calling
	mockLLM := newMockLLMWithFunctionCalling()
	mockLLM.generateWithFunctionsResult = map[string]interface{}{
		"function_call": map[string]interface{}{
			"name": "search",
			"arguments": map[string]interface{}{
				"query": "test query",
			},
		},
	}

	// Set as default LLM
	originalLLM := core.GlobalConfig.DefaultLLM
	core.GlobalConfig.DefaultLLM = mockLLM
	defer func() { core.GlobalConfig.DefaultLLM = originalLLM }()

	// Setup registry
	registry := tools.NewInMemoryToolRegistry()
	mockTool := &mockTool{
		name:        "search",
		description: "Search for information",
		schema: models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"query": {Type: "string", Description: "The query", Required: true},
			},
		},
	}
	err := registry.Register(mockTool)
	require.NoError(t, err)

	config := FunctionCallingConfig{
		ToolRegistry:      registry,
		IncludeFinishTool: true,
	}

	interceptor := NativeFunctionCallingInterceptor(config)

	// Create a mock handler that should NOT be called
	handlerCalled := false
	mockHandler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return map[string]any{"fallback": true}, nil
	}

	inputs := map[string]any{
		"task": "Find information about Go",
	}

	result, err := interceptor(context.Background(), inputs, nil, mockHandler)
	require.NoError(t, err)

	// Handler should NOT be called since LLM supports function calling
	assert.False(t, handlerCalled, "Handler should not be called when LLM supports function calling")

	// Result should contain the action from the function call
	action, ok := result["action"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "search", action["tool_name"])

	// Verify the functions were passed to the LLM
	assert.Len(t, mockLLM.lastFunctions, 2) // search + Finish
}

func TestNativeFunctionCallingInterceptor_FallbackWithoutToolCalling(t *testing.T) {
	// Setup mock LLM WITHOUT function calling
	mockLLM := newMockLLMWithoutFunctionCalling()

	// Set as default LLM
	originalLLM := core.GlobalConfig.DefaultLLM
	core.GlobalConfig.DefaultLLM = mockLLM
	defer func() { core.GlobalConfig.DefaultLLM = originalLLM }()

	registry := tools.NewInMemoryToolRegistry()
	config := FunctionCallingConfig{
		ToolRegistry:      registry,
		IncludeFinishTool: true,
	}

	interceptor := NativeFunctionCallingInterceptor(config)

	// Create a mock handler that SHOULD be called as fallback
	handlerCalled := false
	mockHandler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		handlerCalled = true
		return map[string]any{"fallback": true}, nil
	}

	inputs := map[string]any{
		"task": "Find information",
	}

	result, err := interceptor(context.Background(), inputs, nil, mockHandler)
	require.NoError(t, err)

	// Handler SHOULD be called since LLM doesn't support function calling
	assert.True(t, handlerCalled, "Handler should be called when LLM doesn't support function calling")
	assert.Equal(t, true, result["fallback"])
}

func TestFunctionCallingReActAdapter(t *testing.T) {
	registry := tools.NewInMemoryToolRegistry()

	adapter := NewFunctionCallingReActAdapter(registry)
	assert.NotNil(t, adapter)
	assert.NotNil(t, adapter.GetInterceptor())

	// Test fluent configuration
	adapter.WithStrictMode().WithCustomFinishDescription("Custom finish description")
	assert.True(t, adapter.config.StrictMode)
	assert.Equal(t, "Custom finish description", adapter.config.FinishToolDescription)
}
