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
