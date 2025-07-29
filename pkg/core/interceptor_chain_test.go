package core

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// TestChainModuleInterceptorsAdvanced tests the module interceptor chaining functionality.
func TestChainModuleInterceptorsAdvanced(t *testing.T) {
	t.Run("empty chain", func(t *testing.T) {
		chainedInterceptor := ChainModuleInterceptors()

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := &ModuleInfo{ModuleName: "test", ModuleType: "test"}

		var handlerCalled bool
		handler := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
			handlerCalled = true
			return map[string]any{"result": "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, inputs, info, handler)
		require.NoError(t, err)
		assert.True(t, handlerCalled)
		assert.Equal(t, "handler", result["result"])
	})

	t.Run("single interceptor", func(t *testing.T) {
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			result, err := handler(ctx, inputs, opts...)
			if err != nil {
				return nil, err
			}
			result["intercepted"] = true
			return result, nil
		}

		chainedInterceptor := ChainModuleInterceptors(interceptor)

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := &ModuleInfo{ModuleName: "test", ModuleType: "test"}

		handler := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
			return map[string]any{"result": "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, inputs, info, handler)
		require.NoError(t, err)
		assert.Equal(t, "handler", result["result"])
		assert.True(t, result["intercepted"].(bool))
	})

	t.Run("multiple interceptors execution order", func(t *testing.T) {
		var executionOrder []string

		interceptor1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			executionOrder = append(executionOrder, "interceptor1_start")
			result, err := handler(ctx, inputs, opts...)
			executionOrder = append(executionOrder, "interceptor1_end")
			return result, err
		}

		interceptor2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			executionOrder = append(executionOrder, "interceptor2_start")
			result, err := handler(ctx, inputs, opts...)
			executionOrder = append(executionOrder, "interceptor2_end")
			return result, err
		}

		interceptor3 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			executionOrder = append(executionOrder, "interceptor3_start")
			result, err := handler(ctx, inputs, opts...)
			executionOrder = append(executionOrder, "interceptor3_end")
			return result, err
		}

		chainedInterceptor := ChainModuleInterceptors(interceptor1, interceptor2, interceptor3)

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := &ModuleInfo{ModuleName: "test", ModuleType: "test"}

		handler := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
			executionOrder = append(executionOrder, "handler")
			return map[string]any{"result": "handler"}, nil
		}

		_, err := chainedInterceptor(ctx, inputs, info, handler)
		require.NoError(t, err)

		expected := []string{
			"interceptor1_start",
			"interceptor2_start",
			"interceptor3_start",
			"handler",
			"interceptor3_end",
			"interceptor2_end",
			"interceptor1_end",
		}
		assert.Equal(t, expected, executionOrder)
	})

	t.Run("interceptor error propagation", func(t *testing.T) {
		errorInterceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return nil, fmt.Errorf("interceptor error")
		}

		chainedInterceptor := ChainModuleInterceptors(errorInterceptor)

		ctx := context.Background()
		inputs := map[string]any{"test": "value"}
		info := &ModuleInfo{ModuleName: "test", ModuleType: "test"}

		var handlerCalled bool
		handler := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
			handlerCalled = true
			return map[string]any{"result": "handler"}, nil
		}

		_, err := chainedInterceptor(ctx, inputs, info, handler)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "interceptor error")
		assert.False(t, handlerCalled) // Handler should not be called if interceptor errors
	})
}

// TestChainAgentInterceptorsAdvanced tests the agent interceptor chaining functionality.
func TestChainAgentInterceptorsAdvanced(t *testing.T) {
	t.Run("empty chain", func(t *testing.T) {
		chainedInterceptor := ChainAgentInterceptors()

		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}
		info := &AgentInfo{AgentID: "test", AgentType: "test"}

		var handlerCalled bool
		handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
			handlerCalled = true
			return map[string]interface{}{"result": "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, input, info, handler)
		require.NoError(t, err)
		assert.True(t, handlerCalled)
		assert.Equal(t, "handler", result["result"])
	})

	t.Run("single interceptor", func(t *testing.T) {
		interceptor := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			result, err := handler(ctx, input)
			if err != nil {
				return nil, err
			}
			result["intercepted"] = true
			return result, nil
		}

		chainedInterceptor := ChainAgentInterceptors(interceptor)

		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}
		info := &AgentInfo{AgentID: "test", AgentType: "test"}

		handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"result": "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, input, info, handler)
		require.NoError(t, err)
		assert.Equal(t, "handler", result["result"])
		assert.True(t, result["intercepted"].(bool))
	})

	t.Run("multiple interceptors execution order", func(t *testing.T) {
		var executionOrder []string

		interceptor1 := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			executionOrder = append(executionOrder, "interceptor1_start")
			result, err := handler(ctx, input)
			executionOrder = append(executionOrder, "interceptor1_end")
			return result, err
		}

		interceptor2 := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			executionOrder = append(executionOrder, "interceptor2_start")
			result, err := handler(ctx, input)
			executionOrder = append(executionOrder, "interceptor2_end")
			return result, err
		}

		chainedInterceptor := ChainAgentInterceptors(interceptor1, interceptor2)

		ctx := context.Background()
		input := map[string]interface{}{"task": "test"}
		info := &AgentInfo{AgentID: "test", AgentType: "test"}

		handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
			executionOrder = append(executionOrder, "handler")
			return map[string]interface{}{"result": "handler"}, nil
		}

		_, err := chainedInterceptor(ctx, input, info, handler)
		require.NoError(t, err)

		expected := []string{
			"interceptor1_start",
			"interceptor2_start",
			"handler",
			"interceptor2_end",
			"interceptor1_end",
		}
		assert.Equal(t, expected, executionOrder)
	})
}

// TestChainToolInterceptorsAdvanced tests the tool interceptor chaining functionality.
func TestChainToolInterceptorsAdvanced(t *testing.T) {
	t.Run("empty chain", func(t *testing.T) {
		chainedInterceptor := ChainToolInterceptors()

		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}
		info := &ToolInfo{Name: "test", Description: "test"}

		var handlerCalled bool
		handler := func(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
			handlerCalled = true
			return ToolResult{Data: "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, args, info, handler)
		require.NoError(t, err)
		assert.True(t, handlerCalled)
		assert.Equal(t, "handler", result.Data)
	})

	t.Run("single interceptor", func(t *testing.T) {
		interceptor := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			result, err := handler(ctx, args)
			if err != nil {
				return ToolResult{}, err
			}
			if result.Metadata == nil {
				result.Metadata = make(map[string]interface{})
			}
			result.Metadata["intercepted"] = true
			return result, nil
		}

		chainedInterceptor := ChainToolInterceptors(interceptor)

		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}
		info := &ToolInfo{Name: "test", Description: "test"}

		handler := func(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
			return ToolResult{Data: "handler"}, nil
		}

		result, err := chainedInterceptor(ctx, args, info, handler)
		require.NoError(t, err)
		assert.Equal(t, "handler", result.Data)
		assert.True(t, result.Metadata["intercepted"].(bool))
	})

	t.Run("multiple interceptors execution order", func(t *testing.T) {
		var executionOrder []string

		interceptor1 := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			executionOrder = append(executionOrder, "interceptor1_start")
			result, err := handler(ctx, args)
			executionOrder = append(executionOrder, "interceptor1_end")
			return result, err
		}

		interceptor2 := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			executionOrder = append(executionOrder, "interceptor2_start")
			result, err := handler(ctx, args)
			executionOrder = append(executionOrder, "interceptor2_end")
			return result, err
		}

		chainedInterceptor := ChainToolInterceptors(interceptor1, interceptor2)

		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}
		info := &ToolInfo{Name: "test", Description: "test"}

		handler := func(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
			executionOrder = append(executionOrder, "handler")
			return ToolResult{Data: "handler"}, nil
		}

		_, err := chainedInterceptor(ctx, args, info, handler)
		require.NoError(t, err)

		expected := []string{
			"interceptor1_start",
			"interceptor2_start",
			"handler",
			"interceptor2_end",
			"interceptor1_end",
		}
		assert.Equal(t, expected, executionOrder)
	})
}

// TestInterceptorChainAdvanced tests the InterceptorChain functionality.
func TestInterceptorChainAdvanced(t *testing.T) {
	t.Run("new chain is empty", func(t *testing.T) {
		chain := NewInterceptorChain()
		assert.True(t, chain.IsEmpty())
		assert.Equal(t, 0, chain.Count())

		modules, agents, tools := chain.CountByType()
		assert.Equal(t, 0, modules)
		assert.Equal(t, 0, agents)
		assert.Equal(t, 0, tools)
	})

	t.Run("add interceptors", func(t *testing.T) {
		chain := NewInterceptorChain()

		moduleInterceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}

		agentInterceptor := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			return handler(ctx, input)
		}

		toolInterceptor := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			return handler(ctx, args)
		}

		chain.AddModuleInterceptor(moduleInterceptor)
		chain.AddAgentInterceptor(agentInterceptor)
		chain.AddToolInterceptor(toolInterceptor)

		assert.False(t, chain.IsEmpty())
		assert.Equal(t, 3, chain.Count())

		modules, agents, tools := chain.CountByType()
		assert.Equal(t, 1, modules)
		assert.Equal(t, 1, agents)
		assert.Equal(t, 1, tools)

		// Test getting interceptors
		assert.Len(t, chain.GetModuleInterceptors(), 1)
		assert.Len(t, chain.GetAgentInterceptors(), 1)
		assert.Len(t, chain.GetToolInterceptors(), 1)
	})

	t.Run("chain composition", func(t *testing.T) {
		chain1 := NewInterceptorChain()
		chain2 := NewInterceptorChain()

		moduleInterceptor1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}
		moduleInterceptor2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}

		chain1.AddModuleInterceptor(moduleInterceptor1)
		chain2.AddModuleInterceptor(moduleInterceptor2)

		composedChain := chain1.Compose(chain2)
		assert.Len(t, composedChain.GetModuleInterceptors(), 2)

		// Original chains should be unchanged
		assert.Len(t, chain1.GetModuleInterceptors(), 1)
		assert.Len(t, chain2.GetModuleInterceptors(), 1)
	})

	t.Run("compose with nil chain", func(t *testing.T) {
		chain := NewInterceptorChain()
		moduleInterceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}
		chain.AddModuleInterceptor(moduleInterceptor)

		result := chain.Compose(nil)
		assert.Len(t, result.GetModuleInterceptors(), 1)
	})

	t.Run("clear chain", func(t *testing.T) {
		chain := NewInterceptorChain()

		moduleInterceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			return handler(ctx, inputs, opts...)
		}

		chain.AddModuleInterceptor(moduleInterceptor)
		assert.False(t, chain.IsEmpty())

		chain.Clear()
		assert.True(t, chain.IsEmpty())
	})
}

// TestNewInfoFunctionsAdvanced tests the convenience functions for creating info structs.
func TestNewInfoFunctionsAdvanced(t *testing.T) {
	t.Run("NewModuleInfo", func(t *testing.T) {
		signature := Signature{}
		info := NewModuleInfo("TestModule", "Test", signature)

		assert.Equal(t, "TestModule", info.ModuleName)
		assert.Equal(t, "Test", info.ModuleType)
		assert.Equal(t, signature, info.Signature)
		assert.Equal(t, "1.0.0", info.Version)
		assert.NotNil(t, info.Metadata)

		// Test fluent methods
		info.WithVersion("2.0.0").WithMetadata("key", "value")
		assert.Equal(t, "2.0.0", info.Version)
		assert.Equal(t, "value", info.Metadata["key"])
	})

	t.Run("NewAgentInfo", func(t *testing.T) {
		tools := []Tool{}
		info := NewAgentInfo("agent-1", "TestAgent", tools)

		assert.Equal(t, "agent-1", info.AgentID)
		assert.Equal(t, "TestAgent", info.AgentType)
		assert.Equal(t, tools, info.Capabilities)
		assert.Equal(t, "1.0.0", info.Version)
		assert.NotNil(t, info.Metadata)

		// Test fluent methods
		info.WithVersion("2.0.0").WithMetadata("key", "value")
		assert.Equal(t, "2.0.0", info.Version)
		assert.Equal(t, "value", info.Metadata["key"])
	})

	t.Run("NewToolInfo", func(t *testing.T) {
		schema := models.InputSchema{Type: "object"}
		info := NewToolInfo("test_tool", "A test tool", "function", schema)

		assert.Equal(t, "test_tool", info.Name)
		assert.Equal(t, "A test tool", info.Description)
		assert.Equal(t, "function", info.ToolType)
		assert.Equal(t, schema, info.InputSchema)
		assert.Equal(t, "1.0.0", info.Version)
		assert.NotNil(t, info.Capabilities)
		assert.NotNil(t, info.Metadata)

		// Test fluent methods
		info.WithVersion("2.0.0").WithMetadata("key", "value").WithCapabilities("read", "write")
		assert.Equal(t, "2.0.0", info.Version)
		assert.Equal(t, "value", info.Metadata["key"])
		assert.Contains(t, info.Capabilities, "read")
		assert.Contains(t, info.Capabilities, "write")
	})
}
