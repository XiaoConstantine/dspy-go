package core

import (
	"context"
	"errors"
	"reflect"
	"testing"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Test helper functions and mock implementations

func createTestModuleInfo() *ModuleInfo {
	return NewModuleInfo("TestModule", "ChainOfThought", Signature{
		Inputs:  []InputField{{Field: NewField("question")}},
		Outputs: []OutputField{{Field: NewField("answer")}},
	})
}

func createTestAgentInfo() *AgentInfo {
	return NewAgentInfo("test-agent-1", "ReactiveAgent", []Tool{})
}

func createTestToolInfo() *ToolInfo {
	return NewToolInfo("test-tool", "A test tool", "FunctionTool", models.InputSchema{})
}

// Mock handlers for testing.
func mockModuleHandler(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return map[string]any{"result": "module-processed"}, nil
}

func mockAgentHandler(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{"result": "agent-processed"}, nil
}

func mockToolHandler(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
	return ToolResult{Data: "tool-processed"}, nil
}

func errorModuleHandler(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return nil, errors.New("module error")
}

func errorAgentHandler(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return nil, errors.New("agent error")
}

func errorToolHandler(ctx context.Context, args map[string]interface{}) (ToolResult, error) {
	return ToolResult{}, errors.New("tool error")
}

// Test NewModuleInfo.
func TestNewModuleInfo(t *testing.T) {
	signature := Signature{
		Inputs:  []InputField{{Field: NewField("input")}},
		Outputs: []OutputField{{Field: NewField("output")}},
	}

	info := NewModuleInfo("TestModule", "ChainOfThought", signature)

	if info.ModuleName != "TestModule" {
		t.Errorf("Expected ModuleName 'TestModule', got '%s'", info.ModuleName)
	}
	if info.ModuleType != "ChainOfThought" {
		t.Errorf("Expected ModuleType 'ChainOfThought', got '%s'", info.ModuleType)
	}
	if !reflect.DeepEqual(info.Signature, signature) {
		t.Errorf("Expected signature to match")
	}
	if info.Version != "1.0.0" {
		t.Errorf("Expected default version '1.0.0', got '%s'", info.Version)
	}
	if info.Metadata == nil {
		t.Errorf("Expected metadata to be initialized")
	}
}

// Test NewAgentInfo.
func TestNewAgentInfo(t *testing.T) {
	capabilities := []Tool{}
	info := NewAgentInfo("agent-1", "ReactiveAgent", capabilities)

	if info.AgentID != "agent-1" {
		t.Errorf("Expected AgentID 'agent-1', got '%s'", info.AgentID)
	}
	if info.AgentType != "ReactiveAgent" {
		t.Errorf("Expected AgentType 'ReactiveAgent', got '%s'", info.AgentType)
	}
	if info.Version != "1.0.0" {
		t.Errorf("Expected default version '1.0.0', got '%s'", info.Version)
	}
	if info.Metadata == nil {
		t.Errorf("Expected metadata to be initialized")
	}
}

// Test NewToolInfo.
func TestNewToolInfo(t *testing.T) {
	schema := models.InputSchema{}
	info := NewToolInfo("tool-1", "Test Tool", "FunctionTool", schema)

	if info.Name != "tool-1" {
		t.Errorf("Expected Name 'tool-1', got '%s'", info.Name)
	}
	if info.Description != "Test Tool" {
		t.Errorf("Expected Description 'Test Tool', got '%s'", info.Description)
	}
	if info.ToolType != "FunctionTool" {
		t.Errorf("Expected ToolType 'FunctionTool', got '%s'", info.ToolType)
	}
	if info.Version != "1.0.0" {
		t.Errorf("Expected default version '1.0.0', got '%s'", info.Version)
	}
	if info.Capabilities == nil {
		t.Errorf("Expected capabilities to be initialized")
	}
	if info.Metadata == nil {
		t.Errorf("Expected metadata to be initialized")
	}
}

// Test metadata and version methods.
func TestModuleInfoMethods(t *testing.T) {
	info := createTestModuleInfo()

	// Test WithMetadata
	info.WithMetadata("key1", "value1")
	if info.Metadata["key1"] != "value1" {
		t.Errorf("Expected metadata key1 to be 'value1', got '%v'", info.Metadata["key1"])
	}

	// Test WithVersion
	info.WithVersion("2.0.0")
	if info.Version != "2.0.0" {
		t.Errorf("Expected version '2.0.0', got '%s'", info.Version)
	}
}

func TestAgentInfoMethods(t *testing.T) {
	info := createTestAgentInfo()

	// Test WithMetadata
	info.WithMetadata("key1", "value1")
	if info.Metadata["key1"] != "value1" {
		t.Errorf("Expected metadata key1 to be 'value1', got '%v'", info.Metadata["key1"])
	}

	// Test WithVersion
	info.WithVersion("2.0.0")
	if info.Version != "2.0.0" {
		t.Errorf("Expected version '2.0.0', got '%s'", info.Version)
	}
}

func TestToolInfoMethods(t *testing.T) {
	info := createTestToolInfo()

	// Test WithMetadata
	info.WithMetadata("key1", "value1")
	if info.Metadata["key1"] != "value1" {
		t.Errorf("Expected metadata key1 to be 'value1', got '%v'", info.Metadata["key1"])
	}

	// Test WithVersion
	info.WithVersion("2.0.0")
	if info.Version != "2.0.0" {
		t.Errorf("Expected version '2.0.0', got '%s'", info.Version)
	}

	// Test WithCapabilities
	info.WithCapabilities("read", "write")
	if len(info.Capabilities) != 2 {
		t.Errorf("Expected 2 capabilities, got %d", len(info.Capabilities))
	}
	if info.Capabilities[0] != "read" || info.Capabilities[1] != "write" {
		t.Errorf("Expected capabilities ['read', 'write'], got %v", info.Capabilities)
	}
}

// Test NewInterceptorChain.
func TestNewInterceptorChain(t *testing.T) {
	chain := NewInterceptorChain()

	if chain == nil {
		t.Error("Expected non-nil interceptor chain")
	}
	if !chain.IsEmpty() {
		t.Error("Expected new chain to be empty")
	}
	if chain.Count() != 0 {
		t.Errorf("Expected count 0, got %d", chain.Count())
	}
}

// Test InterceptorChain methods.
func TestInterceptorChainMethods(t *testing.T) {
	chain := NewInterceptorChain()

	// Test adding interceptors
	moduleInt := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	agentInt := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
		return handler(ctx, input)
	}
	toolInt := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
		return handler(ctx, args)
	}

	chain.AddModuleInterceptor(moduleInt)
	chain.AddAgentInterceptor(agentInt)
	chain.AddToolInterceptor(toolInt)

	// Test counts
	modules, agents, tools := chain.CountByType()
	if modules != 1 {
		t.Errorf("Expected 1 module interceptor, got %d", modules)
	}
	if agents != 1 {
		t.Errorf("Expected 1 agent interceptor, got %d", agents)
	}
	if tools != 1 {
		t.Errorf("Expected 1 tool interceptor, got %d", tools)
	}

	if chain.Count() != 3 {
		t.Errorf("Expected total count 3, got %d", chain.Count())
	}

	if chain.IsEmpty() {
		t.Error("Expected chain to not be empty")
	}

	// Test getters
	moduleInterceptors := chain.GetModuleInterceptors()
	if len(moduleInterceptors) != 1 {
		t.Errorf("Expected 1 module interceptor in getter, got %d", len(moduleInterceptors))
	}

	agentInterceptors := chain.GetAgentInterceptors()
	if len(agentInterceptors) != 1 {
		t.Errorf("Expected 1 agent interceptor in getter, got %d", len(agentInterceptors))
	}

	toolInterceptors := chain.GetToolInterceptors()
	if len(toolInterceptors) != 1 {
		t.Errorf("Expected 1 tool interceptor in getter, got %d", len(toolInterceptors))
	}

	// Test clear
	chain.Clear()
	if !chain.IsEmpty() {
		t.Error("Expected chain to be empty after clear")
	}
	if chain.Count() != 0 {
		t.Errorf("Expected count 0 after clear, got %d", chain.Count())
	}
}

// Test chain composition.
func TestInterceptorChainCompose(t *testing.T) {
	chain1 := NewInterceptorChain()
	chain2 := NewInterceptorChain()

	moduleInt1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	moduleInt2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}

	chain1.AddModuleInterceptor(moduleInt1)
	chain2.AddModuleInterceptor(moduleInt2)

	// Test compose
	composed := chain1.Compose(chain2)
	if composed.Count() != 2 {
		t.Errorf("Expected composed chain to have 2 interceptors, got %d", composed.Count())
	}

	modules, _, _ := composed.CountByType()
	if modules != 2 {
		t.Errorf("Expected 2 module interceptors in composed chain, got %d", modules)
	}

	// Test compose with nil
	composedWithNil := chain1.Compose(nil)
	if composedWithNil.Count() != chain1.Count() {
		t.Error("Expected compose with nil to return copy of original chain")
	}
}

// Test ChainModuleInterceptors.
func TestChainModuleInterceptors(t *testing.T) {
	ctx := context.Background()
	inputs := map[string]any{"input": "test"}
	info := createTestModuleInfo()

	t.Run("no interceptors", func(t *testing.T) {
		chained := ChainModuleInterceptors()
		result, err := chained(ctx, inputs, info, mockModuleHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "module-processed" {
			t.Errorf("Expected 'module-processed', got %v", result["result"])
		}
	})

	t.Run("single interceptor", func(t *testing.T) {
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
			result, err := handler(ctx, inputs)
			if err != nil {
				return nil, err
			}
			result["intercepted"] = true
			return result, nil
		}

		chained := ChainModuleInterceptors(interceptor)
		result, err := chained(ctx, inputs, info, mockModuleHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "module-processed" {
			t.Errorf("Expected 'module-processed', got %v", result["result"])
		}
		if result["intercepted"] != true {
			t.Errorf("Expected intercepted to be true, got %v", result["intercepted"])
		}
	})

	t.Run("multiple interceptors", func(t *testing.T) {
		var order []string

		interceptor1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
			order = append(order, "interceptor1-before")
			result, err := handler(ctx, inputs)
			order = append(order, "interceptor1-after")
			return result, err
		}

		interceptor2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
			order = append(order, "interceptor2-before")
			result, err := handler(ctx, inputs)
			order = append(order, "interceptor2-after")
			return result, err
		}

		handlerWithOrder := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
			order = append(order, "handler")
			return mockModuleHandler(ctx, inputs, opts...)
		}

		chained := ChainModuleInterceptors(interceptor1, interceptor2)
		_, err := chained(ctx, inputs, info, handlerWithOrder)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		expected := []string{"interceptor1-before", "interceptor2-before", "handler", "interceptor2-after", "interceptor1-after"}
		if !reflect.DeepEqual(order, expected) {
			t.Errorf("Expected order %v, got %v", expected, order)
		}
	})

	t.Run("error handling", func(t *testing.T) {
		interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
			return handler(ctx, inputs)
		}

		chained := ChainModuleInterceptors(interceptor)
		_, err := chained(ctx, inputs, info, errorModuleHandler)
		if err == nil {
			t.Error("Expected error to be propagated")
		}
		if err.Error() != "module error" {
			t.Errorf("Expected 'module error', got '%s'", err.Error())
		}
	})
}

// Test ChainAgentInterceptors.
func TestChainAgentInterceptors(t *testing.T) {
	ctx := context.Background()
	input := map[string]interface{}{"input": "test"}
	info := createTestAgentInfo()

	t.Run("no interceptors", func(t *testing.T) {
		chained := ChainAgentInterceptors()
		result, err := chained(ctx, input, info, mockAgentHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "agent-processed" {
			t.Errorf("Expected 'agent-processed', got %v", result["result"])
		}
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

		chained := ChainAgentInterceptors(interceptor)
		result, err := chained(ctx, input, info, mockAgentHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "agent-processed" {
			t.Errorf("Expected 'agent-processed', got %v", result["result"])
		}
		if result["intercepted"] != true {
			t.Errorf("Expected intercepted to be true, got %v", result["intercepted"])
		}
	})

	t.Run("multiple interceptors", func(t *testing.T) {
		var callOrder []string

		interceptor1 := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			callOrder = append(callOrder, "agent_interceptor1_before")
			result, err := handler(ctx, input)
			callOrder = append(callOrder, "agent_interceptor1_after")
			return result, err
		}

		interceptor2 := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			callOrder = append(callOrder, "agent_interceptor2_before")
			result, err := handler(ctx, input)
			callOrder = append(callOrder, "agent_interceptor2_after")
			return result, err
		}

		chained := ChainAgentInterceptors(interceptor1, interceptor2)
		result, err := chained(ctx, input, info, mockAgentHandler)

		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result["result"] != "agent-processed" {
			t.Errorf("Expected 'agent-processed', got %v", result["result"])
		}

		// Check call order
		expectedOrder := []string{"agent_interceptor1_before", "agent_interceptor2_before", "agent_interceptor2_after", "agent_interceptor1_after"}
		if len(callOrder) != len(expectedOrder) {
			t.Errorf("Expected %d calls, got %d", len(expectedOrder), len(callOrder))
		}
		for i, expected := range expectedOrder {
			if i >= len(callOrder) || callOrder[i] != expected {
				t.Errorf("Expected call order %v, got %v", expectedOrder, callOrder)
				break
			}
		}
	})

	t.Run("error handling", func(t *testing.T) {
		interceptor := func(ctx context.Context, input map[string]interface{}, info *AgentInfo, handler AgentHandler) (map[string]interface{}, error) {
			return handler(ctx, input)
		}

		chained := ChainAgentInterceptors(interceptor)
		_, err := chained(ctx, input, info, errorAgentHandler)
		if err == nil {
			t.Error("Expected error to be propagated")
		}
		if err.Error() != "agent error" {
			t.Errorf("Expected 'agent error', got '%s'", err.Error())
		}
	})
}

// Test ChainToolInterceptors.
func TestChainToolInterceptors(t *testing.T) {
	ctx := context.Background()
	args := map[string]interface{}{"arg": "test"}
	info := createTestToolInfo()

	t.Run("no interceptors", func(t *testing.T) {
		chained := ChainToolInterceptors()
		result, err := chained(ctx, args, info, mockToolHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result.Data != "tool-processed" {
			t.Errorf("Expected 'tool-processed', got %v", result.Data)
		}
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

		chained := ChainToolInterceptors(interceptor)
		result, err := chained(ctx, args, info, mockToolHandler)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result.Data != "tool-processed" {
			t.Errorf("Expected 'tool-processed', got %v", result.Data)
		}
		if result.Metadata["intercepted"] != true {
			t.Errorf("Expected intercepted to be true, got %v", result.Metadata["intercepted"])
		}
	})

	t.Run("multiple interceptors", func(t *testing.T) {
		var callOrder []string

		interceptor1 := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			callOrder = append(callOrder, "tool_interceptor1_before")
			result, err := handler(ctx, args)
			callOrder = append(callOrder, "tool_interceptor1_after")
			return result, err
		}

		interceptor2 := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			callOrder = append(callOrder, "tool_interceptor2_before")
			result, err := handler(ctx, args)
			callOrder = append(callOrder, "tool_interceptor2_after")
			return result, err
		}

		chained := ChainToolInterceptors(interceptor1, interceptor2)
		result, err := chained(ctx, args, info, mockToolHandler)

		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result.Data != "tool-processed" {
			t.Errorf("Expected 'tool-processed', got %v", result.Data)
		}

		// Check call order
		expectedOrder := []string{"tool_interceptor1_before", "tool_interceptor2_before", "tool_interceptor2_after", "tool_interceptor1_after"}
		if len(callOrder) != len(expectedOrder) {
			t.Errorf("Expected %d calls, got %d", len(expectedOrder), len(callOrder))
		}
		for i, expected := range expectedOrder {
			if i >= len(callOrder) || callOrder[i] != expected {
				t.Errorf("Expected call order %v, got %v", expectedOrder, callOrder)
				break
			}
		}
	})

	t.Run("error handling", func(t *testing.T) {
		interceptor := func(ctx context.Context, args map[string]interface{}, info *ToolInfo, handler ToolHandler) (ToolResult, error) {
			return handler(ctx, args)
		}

		chained := ChainToolInterceptors(interceptor)
		_, err := chained(ctx, args, info, errorToolHandler)
		if err == nil {
			t.Error("Expected error to be propagated")
		}
		if err.Error() != "tool error" {
			t.Errorf("Expected 'tool error', got '%s'", err.Error())
		}
	})
}

// Benchmark tests.
func BenchmarkChainModuleInterceptors_NoInterceptors(b *testing.B) {
	ctx := context.Background()
	inputs := map[string]any{"input": "test"}
	info := createTestModuleInfo()
	chained := ChainModuleInterceptors()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = chained(ctx, inputs, info, mockModuleHandler)
	}
}

func BenchmarkChainModuleInterceptors_OneInterceptor(b *testing.B) {
	ctx := context.Background()
	inputs := map[string]any{"input": "test"}
	info := createTestModuleInfo()

	interceptor := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	chained := ChainModuleInterceptors(interceptor)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = chained(ctx, inputs, info, mockModuleHandler)
	}
}

func BenchmarkChainModuleInterceptors_ThreeInterceptors(b *testing.B) {
	ctx := context.Background()
	inputs := map[string]any{"input": "test"}
	info := createTestModuleInfo()

	interceptor1 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	interceptor2 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	interceptor3 := func(ctx context.Context, inputs map[string]any, info *ModuleInfo, handler ModuleHandler) (map[string]any, error) {
		return handler(ctx, inputs)
	}
	chained := ChainModuleInterceptors(interceptor1, interceptor2, interceptor3)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = chained(ctx, inputs, info, mockModuleHandler)
	}
}
