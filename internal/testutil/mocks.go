package testutil

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/mock"
)

// MockDataset is a mock implementation of core.Dataset.
type MockDataset struct {
	mock.Mock
	Examples []core.Example
	Index    int
}

func (m *MockDataset) Next() (core.Example, bool) {
	m.Called()
	if m.Index < len(m.Examples) {
		example := m.Examples[m.Index]
		m.Index++
		return example, true
	}
	return core.Example{}, false
}

// Reset resets the dataset iterator.
func (m *MockDataset) Reset() {
	m.Called()
	m.Index = 0
}

// NewMockDataset creates a new MockDataset with the given examples.
func NewMockDataset(examples []core.Example) *MockDataset {
	return &MockDataset{
		Examples: examples,
	}
}

// MockLLM is a mock implementation of core.LLM.
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, prompt, opts)
	// Handle both string and struct returns
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	if response, ok := args.Get(0).(*core.LLMResponse); ok {
		return response, args.Error(1)
	}
	// Fall back to string conversion for simple cases
	return &core.LLMResponse{Content: args.String(0)}, args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, opts)
	result := args.Get(0)
	if result == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

// CreateEmbedding mocks the single embedding creation following the same pattern as Generate.
func (m *MockLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	// Record the method call and get the mock results
	args := m.Called(ctx, input, options)

	// Handle nil case first - if first argument is nil, return error
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	// Check if we got a properly structured EmbeddingResult
	if result, ok := args.Get(0).(*core.EmbeddingResult); ok {
		return result, args.Error(1)
	}

	// Fallback case: create a simple embedding result with basic values
	// This is similar to how Generate falls back to string conversion
	return &core.EmbeddingResult{
		Vector:     []float32{0.1, 0.2, 0.3}, // Default vector
		TokenCount: len(input),
		Metadata: map[string]interface{}{
			"fallback": true,
		},
	}, args.Error(1)
}

// CreateEmbeddings mocks the batch embedding creation.
func (m *MockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	// Record the method call and get the mock results
	args := m.Called(ctx, inputs, options)

	// Handle nil case
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	// Check if we got a properly structured BatchEmbeddingResult
	if result, ok := args.Get(0).(*core.BatchEmbeddingResult); ok {
		return result, args.Error(1)
	}

	// Similar to the single embedding case, provide a fallback
	embeddings := make([]core.EmbeddingResult, len(inputs))
	for i := range inputs {
		embeddings[i] = core.EmbeddingResult{
			Vector:     []float32{0.1, 0.2, 0.3},
			TokenCount: len(inputs[i]),
			Metadata: map[string]interface{}{
				"fallback": true,
				"index":    i,
			},
		}
	}

	return &core.BatchEmbeddingResult{
		Embeddings: embeddings,
		Error:      nil,
		ErrorIndex: -1,
	}, args.Error(1)
}

// ModelID mocks the GetModelID method from the LLM interface.
func (m *MockLLM) ModelID() string {
	args := m.Called()

	ret0, _ := args.Get(0).(string)

	return ret0
}

// GetProviderName mocks the GetProviderName method from the LLM interface.
func (m *MockLLM) ProviderName() string {
	args := m.Called()

	ret0, _ := args.Get(0).(string)

	return ret0
}

func (m *MockLLM) Capabilities() []core.Capability {
	return []core.Capability{}
}

type MockTool struct {
	mock.Mock
	metadata *core.ToolMetadata // Holds static metadata for the mock tool
}

// NewMockTool creates a new mock tool with default metadata.
func NewMockTool(name string) *MockTool {
	return &MockTool{
		metadata: &core.ToolMetadata{
			Name:        name,
			Description: "Mock tool for testing",
			InputSchema: map[string]string{
				"action": "string",
			},
			OutputSchema: map[string]string{
				"result": "string",
			},
			Capabilities:  []string{"test"},
			ContextNeeded: []string{"trace_id"},
		},
	}
}

// Metadata implements the Tool interface.
func (m *MockTool) Metadata() *core.ToolMetadata {
	args := m.Called()
	if ret := args.Get(0); ret != nil {
		return ret.(*core.ToolMetadata)
	}
	return m.metadata
}

// CanHandle implements the Tool interface.
func (m *MockTool) CanHandle(ctx context.Context, intent string) bool {
	args := m.Called(ctx, intent)
	return args.Bool(0)
}

// Execute implements the Tool interface.
func (m *MockTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	args := m.Called(ctx, params)

	// Handle the case where the mock is configured to return an error
	if err := args.Error(1); err != nil {
		return core.ToolResult{}, err
	}

	// Convert the result based on type
	switch result := args.Get(0).(type) {
	case core.ToolResult:
		return result, nil
	case string:
		// Convert string results to ToolResult for backward compatibility
		return core.ToolResult{
			Data: result,
			Metadata: map[string]interface{}{
				"source": "mock_tool",
			},
		}, nil
	default:
		return core.ToolResult{
			Data: result,
			Metadata: map[string]interface{}{
				"source": "mock_tool",
			},
		}, nil
	}
}

// Validate implements the Tool interface.
func (m *MockTool) Validate(params map[string]interface{}) error {
	args := m.Called(params)
	return args.Error(0)
}
