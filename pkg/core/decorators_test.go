package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockBaseLLM implements the LLM interface for testing.
type MockBaseLLM struct {
	mock.Mock
}

func (m *MockBaseLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	if resp, ok := args.Get(0).(*LLMResponse); ok {
		return resp, args.Error(1)
	}
	return nil, args.Error(1)
}

func (m *MockBaseLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

func (m *MockBaseLLM) ModelID() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBaseLLM) ProviderName() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBaseLLM) Capabilities() []Capability {
	args := m.Called()
	return args.Get(0).([]Capability)
}

func TestBaseDecorator_Unwrap(t *testing.T) {
	// Test that Unwrap correctly returns the wrapped LLM
	baseLLM := new(MockBaseLLM)
	decorator := BaseDecorator{LLM: baseLLM}

	unwrapped := decorator.Unwrap()
	assert.Equal(t, baseLLM, unwrapped, "Unwrap should return the original LLM")
}

func TestModelContextDecorator_Generate(t *testing.T) {
	t.Run("with execution state", func(t *testing.T) {
		// Set up the mock base LLM
		baseLLM := new(MockBaseLLM)
		expectedModelID := "test-model"
		expectedResponse := &LLMResponse{Content: "test response"}

		// Set up mock expectations
		baseLLM.On("ModelID").Return(expectedModelID)
		baseLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).Return(expectedResponse, nil)

		// Create the decorator
		decorator := NewModelContextDecorator(baseLLM)

		// Create context with execution state
		ctx := WithExecutionState(context.Background())

		// Execute the decorated Generate method
		response, err := decorator.Generate(ctx, "test prompt")

		// Verify results
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, response)

		// Verify the model ID was set in the execution state
		state := GetExecutionState(ctx)
		assert.Equal(t, expectedModelID, state.GetModelID())

		// Verify mock expectations were met
		baseLLM.AssertExpectations(t)
	})

	t.Run("without execution state", func(t *testing.T) {
		// Set up the mock base LLM
		baseLLM := new(MockBaseLLM)
		expectedResponse := &LLMResponse{Content: "test response"}

		// Set up mock expectations
		baseLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).Return(expectedResponse, nil)

		// Create the decorator
		decorator := NewModelContextDecorator(baseLLM)

		// Execute with context that has no execution state
		ctx := context.Background()
		response, err := decorator.Generate(ctx, "test prompt")

		// Verify results
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, response)

		// Verify mock expectations were met
		baseLLM.AssertExpectations(t)
	})
}

func TestChain(t *testing.T) {
	// Create mock LLM
	baseLLM := new(MockBaseLLM)

	// Create test decorators
	decorator1 := func(base LLM) LLM {
		return &ModelContextDecorator{BaseDecorator{LLM: base}}
	}

	decorator2 := func(base LLM) LLM {
		return &ModelContextDecorator{BaseDecorator{LLM: base}}
	}

	t.Run("single decorator", func(t *testing.T) {
		result := Chain(baseLLM, decorator1)

		// Verify the type and structure
		decorated, ok := result.(*ModelContextDecorator)
		assert.True(t, ok)
		assert.Equal(t, baseLLM, decorated.Unwrap())
	})

	t.Run("multiple decorators", func(t *testing.T) {
		result := Chain(baseLLM, decorator1, decorator2)

		// Verify the decorator chain structure
		decorated1, ok := result.(*ModelContextDecorator)
		assert.True(t, ok)

		decorated2, ok := decorated1.Unwrap().(*ModelContextDecorator)
		assert.True(t, ok)

		assert.Equal(t, baseLLM, decorated2.Unwrap())
	})

	t.Run("no decorators", func(t *testing.T) {
		result := Chain(baseLLM)
		assert.Equal(t, baseLLM, result)
	})
}
