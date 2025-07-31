package core

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBaseModuleInterceptorMethods tests the interceptor management methods.
func TestBaseModuleInterceptorMethods(t *testing.T) {
	module := &TestModule{
		BaseModule: BaseModule{
			Signature:   Signature{},
			DisplayName: "TestModule",
			ModuleType:  "Test",
		},
	}

	// Test initial state
	assert.Empty(t, module.GetInterceptors())

	// Test setting interceptors
	interceptor1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
		return handler(ctx, inputs, opts...)
	}
	interceptor2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
		return handler(ctx, inputs, opts...)
	}

	module.SetInterceptors([]ModuleInterceptor{interceptor1, interceptor2})
	assert.Len(t, module.GetInterceptors(), 2)

	// Test clearing interceptors
	module.ClearInterceptors()
	assert.Empty(t, module.GetInterceptors())
}

// TestBaseModuleProcessWithInterceptors tests the ProcessWithInterceptors method.
func TestBaseModuleProcessWithInterceptors(t *testing.T) {
	module := &TestModule{
		BaseModule: BaseModule{
			Signature:   Signature{},
			DisplayName: "TestModule",
			ModuleType:  "Test",
		},
	}

	ctx := context.Background()
	inputs := map[string]any{"test": "value"}

	t.Run("without interceptors", func(t *testing.T) {
		result, err := module.ProcessWithInterceptors(ctx, inputs, nil)
		require.NoError(t, err)
		assert.Equal(t, "processed: value", result["result"])
	})

	t.Run("with single interceptor", func(t *testing.T) {
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			// Modify inputs before calling handler
			inputs["intercepted"] = true
			result, err := handler(ctx, inputs, opts...)
			if err != nil {
				return nil, err
			}
			// Modify result after handler
			result["interceptor_applied"] = true
			return result, nil
		}

		result, err := module.ProcessWithInterceptors(ctx, inputs, []ModuleInterceptor{interceptor})
		require.NoError(t, err)
		assert.Equal(t, "processed: value", result["result"])
		assert.True(t, result["interceptor_applied"].(bool))
	})

	t.Run("with multiple interceptors", func(t *testing.T) {
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

		result, err := module.ProcessWithInterceptors(ctx, inputs, []ModuleInterceptor{interceptor1, interceptor2})
		require.NoError(t, err)
		assert.Equal(t, "processed: value", result["result"])

		// Check execution order: outer interceptor executes first
		expected := []string{"interceptor1_start", "interceptor2_start", "interceptor2_end", "interceptor1_end"}
		assert.Equal(t, expected, executionOrder)
	})

	t.Run("with default interceptors", func(t *testing.T) {
		defaultInterceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			result, err := handler(ctx, inputs, opts...)
			if err != nil {
				return nil, err
			}
			result["default_interceptor"] = true
			return result, err
		}

		module.SetInterceptors([]ModuleInterceptor{defaultInterceptor})
		result, err := module.ProcessWithInterceptors(ctx, inputs, nil) // Use default interceptors
		require.NoError(t, err)
		assert.True(t, result["default_interceptor"].(bool))
	})

	t.Run("interceptor receives correct info", func(t *testing.T) {
		var receivedInfo *ModuleInfo
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			receivedInfo = info
			return handler(ctx, inputs, opts...)
		}

		_, err := module.ProcessWithInterceptors(ctx, inputs, []ModuleInterceptor{interceptor})
		require.NoError(t, err)
		require.NotNil(t, receivedInfo)
		assert.Equal(t, "TestModule", receivedInfo.ModuleName)
		assert.Equal(t, "Test", receivedInfo.ModuleType)
	})

	t.Run("interceptor can handle errors", func(t *testing.T) {
		errorModule := &ErrorModule{
			BaseModule: BaseModule{
				Signature:   Signature{},
				DisplayName: "ErrorModule",
				ModuleType:  "Error",
			},
		}

		var interceptorCalled bool
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
			interceptorCalled = true
			result, err := handler(ctx, inputs, opts...)
			if err != nil {
				// Interceptor can transform or wrap errors
				return nil, fmt.Errorf("interceptor wrapped: %w", err)
			}
			return result, nil
		}

		_, err := errorModule.ProcessWithInterceptors(ctx, inputs, []ModuleInterceptor{interceptor})
		require.Error(t, err)
		assert.True(t, interceptorCalled)
		assert.Contains(t, err.Error(), "interceptor wrapped")
		assert.Contains(t, err.Error(), "test error")
	})
}

// TestModuleInterceptorClone tests that interceptors are properly cloned.
func TestModuleInterceptorClone(t *testing.T) {
	original := &TestModule{
		BaseModule: BaseModule{
			Signature:   Signature{},
			DisplayName: "Original",
			ModuleType:  "Test",
		},
	}

	interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
		return handler(ctx, inputs, opts...)
	}

	original.SetInterceptors([]ModuleInterceptor{interceptor})
	cloned := original.Clone().(*TestModule)

	// Verify interceptors were copied
	assert.Len(t, cloned.GetInterceptors(), 1)

	// Verify independence - modifying one doesn't affect the other
	original.ClearInterceptors()
	assert.Empty(t, original.GetInterceptors())
	assert.Len(t, cloned.GetInterceptors(), 1)
}

// TestModule is a test implementation for testing interceptors.
type TestModule struct {
	BaseModule
}

func (tm *TestModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	testValue := inputs["test"].(string)
	return map[string]any{"result": "processed: " + testValue}, nil
}

func (tm *TestModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error) {
	return tm.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, tm.Process, opts...)
}

func (tm *TestModule) Clone() Module {
	return &TestModule{
		BaseModule: *tm.BaseModule.Clone().(*BaseModule),
	}
}

// ErrorModule is a test module that always returns an error.
type ErrorModule struct {
	BaseModule
}

func (em *ErrorModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return nil, fmt.Errorf("test error")
}

func (em *ErrorModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error) {
	return em.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, em.Process, opts...)
}

func (em *ErrorModule) Clone() Module {
	return &ErrorModule{
		BaseModule: *em.BaseModule.Clone().(*BaseModule),
	}
}
