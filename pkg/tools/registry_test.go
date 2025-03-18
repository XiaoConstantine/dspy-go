package tools

import (
	"context"
	"errors"
	"testing"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// MockTool implements the Tool interface for testing purposes.
type MockTool struct {
	name        string
	description string
	schema      models.InputSchema
	callFunc    func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error)
}

func (m *MockTool) Name() string {
	return m.name
}

func (m *MockTool) Description() string {
	return m.description
}

func (m *MockTool) InputSchema() models.InputSchema {
	return m.schema
}

func (m *MockTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	return m.callFunc(ctx, args)
}

func TestNewRegistry(t *testing.T) {
	registry := NewRegistry()
	if registry == nil {
		t.Fatal("Expected non-nil registry")
	}
	if registry.tools == nil {
		t.Fatal("Expected tools map to be initialized")
	}
	if len(registry.tools) != 0 {
		t.Errorf("Expected empty tools map, got %d entries", len(registry.tools))
	}
}

func TestRegister(t *testing.T) {
	registry := NewRegistry()
	
	// Create a mock tool
	mockTool := &MockTool{
		name:        "test-tool",
		description: "A test tool",
		schema: models.InputSchema{
			Type:       "object",
			Properties: map[string]models.ParameterSchema{},
		},
		callFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return &models.CallToolResult{}, nil
		},
	}
	
	// Test successful registration
	err := registry.Register(mockTool)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	
	// Test duplicate registration
	err = registry.Register(mockTool)
	if err == nil {
		t.Error("Expected error for duplicate registration, got nil")
	}
}

func TestGet(t *testing.T) {
	registry := NewRegistry()
	
	// Create a mock tool
	mockTool := &MockTool{
		name:        "test-tool",
		description: "A test tool",
		schema: models.InputSchema{
			Type:       "object",
			Properties: map[string]models.ParameterSchema{},
		},
		callFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return &models.CallToolResult{}, nil
		},
	}
	
	// Register the tool
	_ = registry.Register(mockTool)
	
	// Test successful retrieval
	tool, err := registry.Get("test-tool")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if tool == nil {
		t.Fatal("Expected non-nil tool")
	}
	if tool.Name() != "test-tool" {
		t.Errorf("Expected name 'test-tool', got '%s'", tool.Name())
	}
	
	// Test non-existent tool
	tool, err = registry.Get("non-existent-tool")
	if err == nil {
		t.Error("Expected error for non-existent tool, got nil")
	}
	if tool != nil {
		t.Errorf("Expected nil tool, got %v", tool)
	}
}

func TestList(t *testing.T) {
	registry := NewRegistry()
	
	// Check empty list
	tools := registry.List()
	if len(tools) != 0 {
		t.Errorf("Expected empty list, got %d tools", len(tools))
	}
	
	// Add some tools
	mockTool1 := &MockTool{name: "tool1", description: "Tool 1"}
	mockTool2 := &MockTool{name: "tool2", description: "Tool 2"}
	
	_ = registry.Register(mockTool1)
	_ = registry.Register(mockTool2)
	
	// Check list with tools
	tools = registry.List()
	if len(tools) != 2 {
		t.Errorf("Expected 2 tools, got %d", len(tools))
	}
	
	// Check that all tools are in the list
	foundTool1 := false
	foundTool2 := false
	
	for _, tool := range tools {
		switch tool.Name() {
		case "tool1":
			foundTool1 = true
		case "tool2":
			foundTool2 = true
		}
	}
	
	if !foundTool1 {
		t.Error("Expected tool1 in the list, but not found")
	}
	if !foundTool2 {
		t.Error("Expected tool2 in the list, but not found")
	}
}

func TestUnregister(t *testing.T) {
	registry := NewRegistry()
	
	// Create a mock tool
	mockTool := &MockTool{
		name:        "test-tool",
		description: "A test tool",
	}
	
	// Register the tool
	_ = registry.Register(mockTool)
	
	// Test successful unregistration
	err := registry.Unregister("test-tool")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	
	// Verify the tool is no longer in the registry
	_, err = registry.Get("test-tool")
	if err == nil {
		t.Error("Expected error after unregistration, got nil")
	}
	
	// Test unregistering non-existent tool
	err = registry.Unregister("non-existent-tool")
	if err == nil {
		t.Error("Expected error for non-existent tool, got nil")
	}
}

func TestCall(t *testing.T) {
	registry := NewRegistry()
	
	// Mock return values
	mockResult := &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: "Test result",
			},
		},
	}
	mockError := errors.New("test error")
	
	// Create mock tools
	successTool := &MockTool{
		name:        "success-tool",
		description: "A successful tool",
		callFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return mockResult, nil
		},
	}
	
	errorTool := &MockTool{
		name:        "error-tool",
		description: "A failing tool",
		callFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return nil, mockError
		},
	}
	
	// Register the tools
	_ = registry.Register(successTool)
	_ = registry.Register(errorTool)
	
	// Test successful call
	result, err := registry.Call(context.Background(), "success-tool", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if result != mockResult {
		t.Error("Expected mock result, got different result")
	}
	
	// Test call with error
	result, err = registry.Call(context.Background(), "error-tool", nil)
	if err != mockError {
		t.Errorf("Expected mock error, got: %v", err)
	}
	if result != nil {
		t.Errorf("Expected nil result, got: %v", result)
	}
	
	// Test call with non-existent tool
	result, err = registry.Call(context.Background(), "non-existent-tool", nil)
	if err == nil {
		t.Error("Expected error for non-existent tool, got nil")
	}
	if result != nil {
		t.Errorf("Expected nil result, got: %v", result)
	}
}
