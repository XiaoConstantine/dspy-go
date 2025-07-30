package tools

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// TestInterceptorToolWrapper tests the InterceptorToolWrapper functionality.
func TestInterceptorToolWrapper(t *testing.T) {
	originalTool := &MockTestTool{
		name:        "test_tool",
		description: "A test tool",
	}

	wrapper := NewInterceptorToolWrapper(originalTool, "function", "1.0.0")

	t.Run("basic functionality", func(t *testing.T) {
		// Test that wrapper implements InterceptableTool
		var _ InterceptableTool = wrapper

		// Test basic Tool interface methods
		assert.Equal(t, "test_tool", wrapper.Name())
		assert.Equal(t, "A test tool", wrapper.Description())
		assert.Equal(t, "function", wrapper.GetToolType())
		assert.Equal(t, "1.0.0", wrapper.GetVersion())

		schema := wrapper.InputSchema()
		assert.Equal(t, "object", schema.Type)
		assert.Contains(t, schema.Properties, "input")
	})

	t.Run("call without interceptors", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		result, err := wrapper.Call(ctx, args)
		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "text", textContent.Type)
		assert.Equal(t, "Tool result: test", textContent.Text)
		assert.False(t, result.IsError)
	})

	t.Run("call with interceptors", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			// Modify args
			args["intercepted"] = true
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, err
			}
			// Modify result metadata
			if result.Metadata == nil {
				result.Metadata = make(map[string]interface{})
			}
			result.Metadata["interceptor_applied"] = true
			return result, nil
		}

		result, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor})
		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "Tool result: test", textContent.Text)
	})

	t.Run("multiple interceptors execution order", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}
		var executionOrder []string

		interceptor1 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			executionOrder = append(executionOrder, "interceptor1_start")
			result, err := handler(ctx, args)
			executionOrder = append(executionOrder, "interceptor1_end")
			return result, err
		}

		interceptor2 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			executionOrder = append(executionOrder, "interceptor2_start")
			result, err := handler(ctx, args)
			executionOrder = append(executionOrder, "interceptor2_end")
			return result, err
		}

		_, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor1, interceptor2})
		require.NoError(t, err)

		expected := []string{"interceptor1_start", "interceptor2_start", "interceptor2_end", "interceptor1_end"}
		assert.Equal(t, expected, executionOrder)
	})

	t.Run("interceptor receives correct tool info", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}
		var receivedInfo *core.ToolInfo

		interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			receivedInfo = info
			return handler(ctx, args)
		}

		_, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor})
		require.NoError(t, err)
		require.NotNil(t, receivedInfo)
		assert.Equal(t, "test_tool", receivedInfo.Name)
		assert.Equal(t, "A test tool", receivedInfo.Description)
		assert.Equal(t, "function", receivedInfo.ToolType)
		assert.Equal(t, "1.0.0", receivedInfo.Version)
	})

	t.Run("interceptor management", func(t *testing.T) {
		// Test initial state
		assert.Empty(t, wrapper.GetInterceptors())

		// Test setting interceptors
		interceptor1 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			return handler(ctx, args)
		}
		interceptor2 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			return handler(ctx, args)
		}

		wrapper.SetInterceptors([]core.ToolInterceptor{interceptor1, interceptor2})
		assert.Len(t, wrapper.GetInterceptors(), 2)

		// Test clearing interceptors
		wrapper.ClearInterceptors()
		assert.Empty(t, wrapper.GetInterceptors())
	})

	t.Run("use default interceptors", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		defaultInterceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, err
			}
			if result.Metadata == nil {
				result.Metadata = make(map[string]interface{})
			}
			result.Metadata["default_interceptor"] = true
			return result, err
		}

		wrapper.SetInterceptors([]core.ToolInterceptor{defaultInterceptor})
		result, err := wrapper.CallWithInterceptors(ctx, args, nil) // Use default interceptors
		require.NoError(t, err)

		// Since the result is converted back to CallToolResult, we check the original result format
		require.Len(t, result.Content, 1)
		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "Tool result: test", textContent.Text)
	})

	t.Run("interceptor error handling", func(t *testing.T) {
		errorTool := &ErrorTestTool{}
		errorWrapper := NewInterceptorToolWrapper(errorTool, "error", "1.0.0")

		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		var interceptorCalled bool
		interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			interceptorCalled = true
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, fmt.Errorf("interceptor wrapped: %w", err)
			}
			return result, nil
		}

		_, err := errorWrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor})
		require.Error(t, err)
		assert.True(t, interceptorCalled)
		assert.Contains(t, err.Error(), "interceptor wrapped")
		assert.Contains(t, err.Error(), "tool error")
	})

	t.Run("result conversion", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		// Interceptor that returns a different core.ToolResult
		interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			return core.ToolResult{
				Data: "custom data",
				Metadata: map[string]interface{}{
					"custom": true,
				},
				Annotations: map[string]interface{}{
					"note": "custom result",
				},
			}, nil
		}

		result, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor})
		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "text", textContent.Type)
		assert.Equal(t, "custom data", textContent.Text)
		assert.False(t, result.IsError)
	})

	t.Run("metadata preservation", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		// Interceptor that adds metadata and annotations
		interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, err
			}
			
			// Add rich metadata
			if result.Metadata == nil {
				result.Metadata = make(map[string]interface{})
			}
			result.Metadata["execution_time_ms"] = 150
			result.Metadata["interceptor_id"] = "test-interceptor"
			result.Metadata["success"] = true
			
			// Add annotations
			if result.Annotations == nil {
				result.Annotations = make(map[string]interface{})
			}
			result.Annotations["trace_id"] = "trace-12345"
			result.Annotations["debug_info"] = "interceptor executed successfully"
			
			return result, nil
		}

		// Execute with interceptor
		result, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor})
		require.NoError(t, err)
		
		// Check that the MCP result is still correct
		require.Len(t, result.Content, 1)
		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "Tool result: test", textContent.Text)
		
		// Check that metadata was preserved
		metadata := wrapper.GetLastExecutionMetadata()
		assert.Equal(t, 150, metadata["execution_time_ms"])
		assert.Equal(t, "test-interceptor", metadata["interceptor_id"])
		assert.True(t, metadata["success"].(bool))
		
		// Check that annotations were preserved
		annotations := wrapper.GetLastExecutionAnnotations()
		assert.Equal(t, "trace-12345", annotations["trace_id"])
		assert.Equal(t, "interceptor executed successfully", annotations["debug_info"])
	})

	t.Run("metadata isolation between calls", func(t *testing.T) {
		ctx := context.Background()
		args := map[string]interface{}{"input": "test"}

		// First interceptor
		interceptor1 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, err
			}
			result.Metadata = map[string]interface{}{"call": "first"}
			return result, nil
		}

		// Second interceptor  
		interceptor2 := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
			result, err := handler(ctx, args)
			if err != nil {
				return core.ToolResult{}, err
			}
			result.Metadata = map[string]interface{}{"call": "second"}
			return result, nil
		}

		// First call
		_, err := wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor1})
		require.NoError(t, err)
		assert.Equal(t, "first", wrapper.GetLastExecutionMetadata()["call"])

		// Second call - should replace first call's metadata
		_, err = wrapper.CallWithInterceptors(ctx, args, []core.ToolInterceptor{interceptor2})
		require.NoError(t, err)
		assert.Equal(t, "second", wrapper.GetLastExecutionMetadata()["call"])
		
		// Should not contain first call's metadata
		_, exists := wrapper.GetLastExecutionMetadata()["first"]
		assert.False(t, exists)
	})
}

// TestWrapToolWithInterceptors tests the convenience function.
func TestWrapToolWithInterceptors(t *testing.T) {
	originalTool := &MockTestTool{
		name:        "test_tool",
		description: "A test tool",
	}

	interceptor := func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		result, err := handler(ctx, args)
		if err != nil {
			return core.ToolResult{}, err
		}
		if result.Metadata == nil {
			result.Metadata = make(map[string]interface{})
		}
		result.Metadata["wrapped"] = true
		return result, nil
	}

	// Test wrapping with interceptors
	wrapped := WrapToolWithInterceptors(originalTool, "function", "1.0.0", interceptor)

	ctx := context.Background()
	args := map[string]interface{}{"input": "test"}

	t.Run("wrapper function creates proper wrapper", func(t *testing.T) {
		assert.Equal(t, "test_tool", wrapped.Name())
		assert.Equal(t, "function", wrapped.GetToolType())
		assert.Equal(t, "1.0.0", wrapped.GetVersion())
		assert.Len(t, wrapped.GetInterceptors(), 1)
	})

	t.Run("wrapped tool executes with interceptors", func(t *testing.T) {
		result, err := wrapped.CallWithInterceptors(ctx, args, nil) // Use default interceptors
		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "Tool result: test", textContent.Text)
	})

	t.Run("wrapped tool direct execution", func(t *testing.T) {
		result, err := wrapped.Call(ctx, args)
		require.NoError(t, err)
		require.Len(t, result.Content, 1)

		textContent := result.Content[0].(models.TextContent)
		assert.Equal(t, "Tool result: test", textContent.Text)
	})
}

// MockTestTool is a test implementation of Tool.
type MockTestTool struct {
	name        string
	description string
}

func (mtt *MockTestTool) Name() string {
	return mtt.name
}

func (mtt *MockTestTool) Description() string {
	return mtt.description
}

func (mtt *MockTestTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"input": {
				Type:        "string",
				Description: "Input parameter",
				Required:    true,
			},
		},
	}
}

func (mtt *MockTestTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	input := args["input"].(string)
	return &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: "Tool result: " + input,
			},
		},
		IsError: false,
	}, nil
}

// ErrorTestTool is a test tool that always returns an error.
type ErrorTestTool struct{}

func (ett *ErrorTestTool) Name() string {
	return "error_tool"
}

func (ett *ErrorTestTool) Description() string {
	return "A tool that always errors"
}

func (ett *ErrorTestTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type:       "object",
		Properties: map[string]models.ParameterSchema{},
	}
}

func (ett *ErrorTestTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	return nil, fmt.Errorf("tool error")
}
