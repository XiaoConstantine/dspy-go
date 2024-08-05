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

// Next returns the next example in the dataset.
func (m *MockDataset) Next() (core.Example, bool) {
	args := m.Called()
	if args.Get(0) != nil {
		return args.Get(0).(core.Example), args.Bool(1)
	}

	// If no explicit return value is set, use the Examples slice
	if m.Index >= len(m.Examples) {
		return core.Example{}, false
	}
	example := m.Examples[m.Index]
	m.Index++
	return example, true
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

func (m *MockLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (string, error) {
    args := m.Called(ctx, prompt, opts)
    return args.String(0), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
    args := m.Called(ctx, prompt, opts)
    return args.Get(0).(map[string]interface{}), args.Error(1)
}
