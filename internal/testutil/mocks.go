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
