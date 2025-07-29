package agents

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// TestInterceptorAgentAdapter tests the InterceptorAgentAdapter functionality.
func TestInterceptorAgentAdapter(t *testing.T) {
	originalAgent := &MockTestAgent{
		id:           "test-agent",
		capabilities: []core.Tool{},
	}

	adapter := NewInterceptorAgentAdapter(originalAgent, "adapter-agent-1", "TestAgent")

	t.Run("basic functionality", func(t *testing.T) {
		// Test that adapter implements InterceptableAgent
		var _ InterceptableAgent = adapter

		// Test basic Agent interface methods
		assert.Equal(t, "adapter-agent-1", adapter.GetAgentID())
		assert.Equal(t, "TestAgent", adapter.GetAgentType())
		assert.Equal(t, originalAgent.capabilities, adapter.GetCapabilities())
		assert.Equal(t, originalAgent.GetMemory(), adapter.GetMemory())
	})

	t.Run("execute without interceptors", func(t *testing.T) {
		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}

		result, err := adapter.Execute(ctx, input)
		require.NoError(t, err)
		assert.Equal(t, "executed: test", result["result"])
	})

	t.Run("execute with interceptors", func(t *testing.T) {
		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}

		interceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			// Modify input
			input["intercepted"] = true
			result, err := handler(ctx, input)
			if err != nil {
				return nil, err
			}
			// Modify result
			result["interceptor_applied"] = true
			return result, nil
		}

		result, err := adapter.ExecuteWithInterceptors(ctx, input, []core.AgentInterceptor{interceptor})
		require.NoError(t, err)
		assert.Equal(t, "executed: test", result["result"])
		assert.True(t, result["interceptor_applied"].(bool))
	})

	t.Run("multiple interceptors execution order", func(t *testing.T) {
		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}
		var executionOrder []string

		interceptor1 := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			executionOrder = append(executionOrder, "interceptor1_start")
			result, err := handler(ctx, input)
			executionOrder = append(executionOrder, "interceptor1_end")
			return result, err
		}

		interceptor2 := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			executionOrder = append(executionOrder, "interceptor2_start")
			result, err := handler(ctx, input)
			executionOrder = append(executionOrder, "interceptor2_end")
			return result, err
		}

		_, err := adapter.ExecuteWithInterceptors(ctx, input, []core.AgentInterceptor{interceptor1, interceptor2})
		require.NoError(t, err)

		expected := []string{"interceptor1_start", "interceptor2_start", "interceptor2_end", "interceptor1_end"}
		assert.Equal(t, expected, executionOrder)
	})

	t.Run("interceptor receives correct agent info", func(t *testing.T) {
		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}
		var receivedInfo *core.AgentInfo

		interceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			receivedInfo = info
			return handler(ctx, input)
		}

		_, err := adapter.ExecuteWithInterceptors(ctx, input, []core.AgentInterceptor{interceptor})
		require.NoError(t, err)
		require.NotNil(t, receivedInfo)
		assert.Equal(t, "adapter-agent-1", receivedInfo.AgentID)
		assert.Equal(t, "TestAgent", receivedInfo.AgentType)
		assert.Equal(t, originalAgent.capabilities, receivedInfo.Capabilities)
	})

	t.Run("interceptor management", func(t *testing.T) {
		// Test initial state
		assert.Empty(t, adapter.GetInterceptors())

		// Test setting interceptors
		interceptor1 := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			return handler(ctx, input)
		}
		interceptor2 := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			return handler(ctx, input)
		}

		adapter.SetInterceptors([]core.AgentInterceptor{interceptor1, interceptor2})
		assert.Len(t, adapter.GetInterceptors(), 2)

		// Test clearing interceptors
		adapter.ClearInterceptors()
		assert.Empty(t, adapter.GetInterceptors())
	})

	t.Run("use default interceptors", func(t *testing.T) {
		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}

		defaultInterceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			result, err := handler(ctx, input)
			if err != nil {
				return nil, err
			}
			result["default_interceptor"] = true
			return result, err
		}

		adapter.SetInterceptors([]core.AgentInterceptor{defaultInterceptor})
		result, err := adapter.ExecuteWithInterceptors(ctx, input, nil) // Use default interceptors
		require.NoError(t, err)
		assert.True(t, result["default_interceptor"].(bool))
	})

	t.Run("interceptor error handling", func(t *testing.T) {
		errorAgent := &ErrorTestAgent{}
		errorAdapter := NewInterceptorAgentAdapter(errorAgent, "error-agent", "ErrorAgent")

		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}

		var interceptorCalled bool
		interceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
			interceptorCalled = true
			result, err := handler(ctx, input)
			if err != nil {
				return nil, fmt.Errorf("interceptor wrapped: %w", err)
			}
			return result, nil
		}

		_, err := errorAdapter.ExecuteWithInterceptors(ctx, input, []core.AgentInterceptor{interceptor})
		require.Error(t, err)
		assert.True(t, interceptorCalled)
		assert.Contains(t, err.Error(), "interceptor wrapped")
		assert.Contains(t, err.Error(), "agent error")
	})
}

// TestWrapAgentWithInterceptors tests the convenience function.
func TestWrapAgentWithInterceptors(t *testing.T) {
	originalAgent := &MockTestAgent{
		id:           "test-agent",
		capabilities: []core.Tool{},
	}

	interceptor := func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		result, err := handler(ctx, input)
		if err != nil {
			return nil, err
		}
		result["wrapped"] = true
		return result, nil
	}

	// Test wrapping with interceptors
	wrapped := WrapAgentWithInterceptors(originalAgent, "wrapped-agent", "WrappedAgent", interceptor)

	ctx := context.Background()
	input := map[string]interface{}{"task": "test"}

	t.Run("wrapper function creates proper adapter", func(t *testing.T) {
		assert.Equal(t, "wrapped-agent", wrapped.GetAgentID())
		assert.Equal(t, "WrappedAgent", wrapped.GetAgentType())
		assert.Len(t, wrapped.GetInterceptors(), 1)
	})

	t.Run("wrapped agent executes with interceptors", func(t *testing.T) {
		result, err := wrapped.ExecuteWithInterceptors(ctx, input, nil) // Use default interceptors
		require.NoError(t, err)
		assert.True(t, result["wrapped"].(bool))
	})

	t.Run("wrapped agent direct execution", func(t *testing.T) {
		result, err := wrapped.Execute(ctx, input)
		require.NoError(t, err)
		assert.Equal(t, "executed: test", result["result"])
		assert.Nil(t, result["wrapped"]) // No interceptor applied
	})
}

// MockTestAgent is a test implementation of Agent.
type MockTestAgent struct {
	id           string
	capabilities []core.Tool
}

func (mta *MockTestAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	task := input["task"].(string)
	return map[string]interface{}{"result": "executed: " + task}, nil
}

func (mta *MockTestAgent) GetCapabilities() []core.Tool {
	return mta.capabilities
}

func (mta *MockTestAgent) GetMemory() Memory {
	return nil
}

// ErrorTestAgent is a test agent that always returns an error.
type ErrorTestAgent struct{}

func (eta *ErrorTestAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return nil, fmt.Errorf("agent error")
}

func (eta *ErrorTestAgent) GetCapabilities() []core.Tool {
	return []core.Tool{}
}

func (eta *ErrorTestAgent) GetMemory() Memory {
	return nil
}
