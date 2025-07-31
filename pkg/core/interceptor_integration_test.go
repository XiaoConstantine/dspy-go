package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestModuleInterceptorIntegration tests the integration of module interceptors with BaseModule.
func TestModuleInterceptorIntegration(t *testing.T) {
	// Create a test module that implements Process
	testModule := &TestModuleWithInterceptors{
		BaseModule: BaseModule{
			Signature:   Signature{},
			DisplayName: "TestModule",
			ModuleType:  "Test",
		},
	}

	// Test interceptor that adds metadata
	interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler, opts ...Option) (map[string]any, error) {
		// Call the handler
		result, err := handler(ctx, inputs, opts...)
		if err != nil {
			return nil, err
		}

		// Add interceptor metadata
		if result == nil {
			result = make(map[string]any)
		}
		result["intercepted"] = true
		result["module_name"] = info.ModuleName
		return result, nil
	}

	// Test without interceptors (direct call to the test module's Process method)
	ctx := context.Background()
	inputs := map[string]any{"test": "value"}

	result, err := testModule.Process(ctx, inputs)
	require.NoError(t, err)
	assert.Equal(t, "processed", result["result"])
	assert.Nil(t, result["intercepted"])

	// Test with interceptors
	result, err = testModule.ProcessWithInterceptors(ctx, inputs, []ModuleInterceptor{interceptor})
	require.NoError(t, err)
	assert.Equal(t, "processed", result["result"])
	assert.True(t, result["intercepted"].(bool))
	assert.Equal(t, "TestModule", result["module_name"])

	// Test setting default interceptors
	testModule.SetInterceptors([]ModuleInterceptor{interceptor})
	result, err = testModule.ProcessWithInterceptors(ctx, inputs, nil) // Use default interceptors
	require.NoError(t, err)
	assert.True(t, result["intercepted"].(bool))
}

// TestModuleWithInterceptors is a test implementation of a module for testing interceptors.
type TestModuleWithInterceptors struct {
	BaseModule
}

// Process implements the Module interface for testing.
func (tm *TestModuleWithInterceptors) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return map[string]any{"result": "processed"}, nil
}

// ProcessWithInterceptors overrides the BaseModule implementation to use the correct Process method.
func (tm *TestModuleWithInterceptors) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error) {
	return tm.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, tm.Process, opts...)
}

// Clone creates a copy of the test module.
func (tm *TestModuleWithInterceptors) Clone() Module {
	return &TestModuleWithInterceptors{
		BaseModule: *tm.BaseModule.Clone().(*BaseModule),
	}
}
