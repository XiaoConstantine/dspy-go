package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// TestFuncTool is a variation of FuncTool that allows us to mock the Call function.
type TestFuncTool struct {
	*FuncTool
	mockCallFunc func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error)
}

// Call overrides the FuncTool.Call method for testing.
func (t *TestFuncTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	if t.mockCallFunc != nil {
		return t.mockCallFunc(ctx, args)
	}
	return t.FuncTool.Call(ctx, args)
}

// Forward other Tool interface methods to the embedded FuncTool.
func (t *TestFuncTool) Name() string {
	return t.FuncTool.Name()
}

func (t *TestFuncTool) Description() string {
	return t.FuncTool.Description()
}

func (t *TestFuncTool) InputSchema() models.InputSchema {
	return t.FuncTool.InputSchema()
}

func (t *TestFuncTool) Metadata() *core.ToolMetadata {
	return t.FuncTool.Metadata()
}

func (t *TestFuncTool) CanHandle(ctx context.Context, intent string) bool {
	return t.FuncTool.CanHandle(ctx, intent)
}

// Override the Execute method for testing.
func (t *TestFuncTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	if t.mockCallFunc != nil {
		// Use the mock call function to get the result
		result, err := t.mockCallFunc(ctx, params)
		if err != nil {
			return core.ToolResult{}, err
		}
		
		// Convert to core.ToolResult
		toolResult := core.ToolResult{
			Data:        extractContentText(result.Content),
			Metadata:    map[string]interface{}{"isError": result.IsError},
			Annotations: map[string]interface{}{},
		}
		return toolResult, nil
	}
	return t.FuncTool.Execute(ctx, params)
}

func (t *TestFuncTool) Validate(params map[string]interface{}) error {
	return t.FuncTool.Validate(params)
}

func (t *TestFuncTool) Type() ToolType {
	return t.FuncTool.Type()
}

func TestNewFuncTool(t *testing.T) {
	// Create a test schema
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"param1": {
				Type:        "string",
				Description: "A string parameter",
				Required:    true,
			},
		},
	}

	// Create a test function
	fn := func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
		return &models.CallToolResult{}, nil
	}

	// Create a new FuncTool
	tool := NewFuncTool("test-tool", "A test tool for searching", schema, fn)

	// Check that the tool was created correctly
	if tool == nil {
		t.Fatal("Expected non-nil tool")
	}
	if tool.name != "test-tool" {
		t.Errorf("Expected name 'test-tool', got '%s'", tool.name)
	}
	if tool.description != "A test tool for searching" {
		t.Errorf("Expected description 'A test tool for searching', got '%s'", tool.description)
	}
	if tool.schema.Type != "object" {
		t.Errorf("Expected schema type 'object', got '%s'", tool.schema.Type)
	}
	if len(tool.schema.Properties) != 1 {
		t.Errorf("Expected 1 property, got %d", len(tool.schema.Properties))
	}
	if tool.fn == nil {
		t.Fatal("Expected non-nil function")
	}
	if tool.metadata == nil {
		t.Fatal("Expected non-nil metadata")
	}
	
	// Check capabilities were extracted from the description
	if len(tool.metadata.Capabilities) == 0 {
		t.Error("Expected capabilities to be extracted from description")
	}
	foundSearch := false
	for _, cap := range tool.metadata.Capabilities {
		if cap == "search" {
			foundSearch = true
			break
		}
	}
	if !foundSearch {
		t.Error("Expected 'search' capability to be extracted from description")
	}
}

func TestFuncToolName(t *testing.T) {
	tool := &FuncTool{name: "test-tool"}
	if name := tool.Name(); name != "test-tool" {
		t.Errorf("Expected name 'test-tool', got '%s'", name)
	}
}

func TestFuncToolDescription(t *testing.T) {
	tool := &FuncTool{description: "Test description"}
	if desc := tool.Description(); desc != "Test description" {
		t.Errorf("Expected description 'Test description', got '%s'", desc)
	}
}

func TestFuncToolInputSchema(t *testing.T) {
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"param1": {
				Type:        "string",
				Description: "A string parameter",
				Required:    true,
			},
		},
	}
	tool := &FuncTool{schema: schema}
	
	returnedSchema := tool.InputSchema()
	if returnedSchema.Type != "object" {
		t.Errorf("Expected schema type 'object', got '%s'", returnedSchema.Type)
	}
	if len(returnedSchema.Properties) != 1 {
		t.Errorf("Expected 1 property, got %d", len(returnedSchema.Properties))
	}
	if _, ok := returnedSchema.Properties["param1"]; !ok {
		t.Error("Expected property 'param1'")
	}
}

func TestFuncToolMetadata(t *testing.T) {
	metadata := &core.ToolMetadata{
		Name:        "test-tool",
		Description: "Test description",
	}
	tool := &FuncTool{metadata: metadata}
	
	returnedMetadata := tool.Metadata()
	if returnedMetadata != metadata {
		t.Error("Expected metadata to be the same object")
	}
}

func TestFuncToolCanHandle(t *testing.T) {
	// Create a tool with capabilities
	metadata := &core.ToolMetadata{
		Name:         "search-tool",
		Description:  "A search tool",
		Capabilities: []string{"search", "find"},
	}
	tool := &FuncTool{
		metadata:    metadata,
		matchCutoff: 0.3,
	}
	
	// Test various intents
	tests := []struct {
		intent   string
		expected bool
	}{
		{"I need to search for something", true},
		{"Can you find information about X", true},
		{"Search for documents about Go", true},
		{"I want to calculate something", false},
		{"search-tool please help me", true}, // Contains tool name
	}
	
	for _, test := range tests {
		result := tool.CanHandle(context.Background(), test.intent)
		if result != test.expected {
			t.Errorf("CanHandle('%s') = %v, expected %v", test.intent, result, test.expected)
		}
	}
}

func TestFuncToolCall(t *testing.T) {
	// Create test results
	mockResult := &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: "Test result",
			},
		},
	}
	mockError := errors.New("test error")
	
	// Test successful call
	successFn := func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
		return mockResult, nil
	}
	successTool := &FuncTool{fn: successFn}
	
	result, err := successTool.Call(context.Background(), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if result != mockResult {
		t.Error("Expected mock result, got different result")
	}
	
	// Test failing call
	errorFn := func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
		return nil, mockError
	}
	errorTool := &FuncTool{fn: errorFn}
	
	result, err = errorTool.Call(context.Background(), nil)
	if err != mockError {
		t.Errorf("Expected mock error, got: %v", err)
	}
	if result != nil {
		t.Errorf("Expected nil result, got: %v", result)
	}
}

func TestFuncToolExecute(t *testing.T) {
	// Create test results
	mockResult := &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: "Test result",
			},
		},
		IsError: false,
	}
	mockError := errors.New("test error")
	
	// Test successful execution
	successFn := func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
		return mockResult, nil
	}
	successTool := &FuncTool{fn: successFn}
	
	coreResult, err := successTool.Execute(context.Background(), nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if coreResult.Data != "Test result" {
		t.Errorf("Expected data 'Test result', got '%v'", coreResult.Data)
	}
	if isError, _ := coreResult.Metadata["isError"].(bool); isError {
		t.Error("Expected isError to be false")
	}
	
	// Test execution with error
	errorFn := func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
		return nil, mockError
	}
	errorTool := &FuncTool{fn: errorFn}
	
	coreResult, err = errorTool.Execute(context.Background(), nil)
	if err != mockError {
		t.Errorf("Expected mock error, got: %v", err)
	}
}

func TestFuncToolValidate(t *testing.T) {
	// Create a schema with required and optional parameters
	schema := models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"required_param": {
				Type:        "string",
				Description: "A required parameter",
				Required:    true,
			},
			"optional_param": {
				Type:        "number",
				Description: "An optional parameter",
				Required:    false,
			},
		},
	}
	tool := &FuncTool{schema: schema}
	
	// Test valid parameters
	validParams := map[string]interface{}{
		"required_param": "value",
		"optional_param": 42,
	}
	err := tool.Validate(validParams)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	
	// Test missing required parameter
	invalidParams := map[string]interface{}{
		"optional_param": 42,
	}
	err = tool.Validate(invalidParams)
	if err == nil {
		t.Error("Expected error for missing required parameter, got nil")
	}
	
	// Test with only required parameter
	requiredOnlyParams := map[string]interface{}{
		"required_param": "value",
	}
	err = tool.Validate(requiredOnlyParams)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestFuncToolType(t *testing.T) {
	tool := &FuncTool{}
	if tool.Type() != ToolTypeFunc {
		t.Errorf("Expected type '%s', got '%s'", ToolTypeFunc, tool.Type())
	}
}
