package tools

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// TestMCPTool is a variation of MCPTool that allows us to mock the Call function.
type TestMCPTool struct {
	*MCPTool
	mockCallFunc func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error)
}

// Call overrides the MCPTool.Call method for testing.
func (t *TestMCPTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	if t.mockCallFunc != nil {
		return t.mockCallFunc(ctx, args)
	}
	return t.MCPTool.Call(ctx, args)
}

// Forward other Tool interface methods to the embedded MCPTool.
func (t *TestMCPTool) Name() string {
	return t.MCPTool.Name()
}

func (t *TestMCPTool) Description() string {
	return t.MCPTool.Description()
}

func (t *TestMCPTool) InputSchema() models.InputSchema {
	return t.MCPTool.InputSchema()
}

func (t *TestMCPTool) Metadata() *core.ToolMetadata {
	return t.MCPTool.Metadata()
}

func (t *TestMCPTool) CanHandle(ctx context.Context, intent string) bool {
	return t.MCPTool.CanHandle(ctx, intent)
}

// Override the Execute method for testing.
func (t *TestMCPTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
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
	return t.MCPTool.Execute(ctx, params)
}

func (t *TestMCPTool) Validate(params map[string]interface{}) error {
	return t.MCPTool.Validate(params)
}

func (t *TestMCPTool) Type() ToolType {
	return t.MCPTool.Type()
}

func TestNewMCPTool(t *testing.T) {
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

	// Create a mock client
	mockClient := &client.Client{}

	// Create a new MCPTool
	tool := NewMCPTool("test-tool", "A test tool for searching", schema, mockClient, "remote_search")

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
	if tool.client != mockClient {
		t.Error("Expected client to be the same object")
	}
	if tool.toolName != "remote_search" {
		t.Errorf("Expected toolName 'remote_search', got '%s'", tool.toolName)
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

func TestMCPToolName(t *testing.T) {
	tool := &MCPTool{name: "test-tool"}
	if name := tool.Name(); name != "test-tool" {
		t.Errorf("Expected name 'test-tool', got '%s'", name)
	}
}

func TestMCPToolDescription(t *testing.T) {
	tool := &MCPTool{description: "Test description"}
	if desc := tool.Description(); desc != "Test description" {
		t.Errorf("Expected description 'Test description', got '%s'", desc)
	}
}

func TestMCPToolInputSchema(t *testing.T) {
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
	tool := &MCPTool{schema: schema}

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

func TestMCPToolMetadata(t *testing.T) {
	metadata := &core.ToolMetadata{
		Name:        "test-tool",
		Description: "Test description",
	}
	tool := &MCPTool{metadata: metadata}

	returnedMetadata := tool.Metadata()
	if returnedMetadata != metadata {
		t.Error("Expected metadata to be the same object")
	}
}

func TestMCPToolCanHandle(t *testing.T) {
	// Create a tool with capabilities
	metadata := &core.ToolMetadata{
		Name:         "search-tool",
		Description:  "A search tool",
		Capabilities: []string{"search", "find"},
	}
	tool := &MCPTool{
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

func TestMCPToolCall(t *testing.T) {
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

	// For this test, we'll use our testable mock implementation

	// Test successful call
	t.Run("successful call", func(t *testing.T) {
		// Create a base tool
		baseTool := &MCPTool{
			name:        "test-tool",
			toolName:    "remote-test-tool",
			description: "Test tool",
		}

		// Create a mock wrapper
		tool := &TestMCPTool{
			MCPTool: baseTool,
			mockCallFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
				return mockResult, nil
			},
		}

		// Call the tool
		result, err := tool.Call(context.Background(), nil)

		// Check results
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if result != mockResult {
			t.Errorf("Expected result %v, got %v", mockResult, result)
		}
	})

	// Test error call
	t.Run("error call", func(t *testing.T) {
		// Create a base tool
		baseTool := &MCPTool{
			name:        "test-tool",
			toolName:    "remote-test-tool",
			description: "Test tool",
		}

		// Create a mock wrapper
		tool := &TestMCPTool{
			MCPTool: baseTool,
			mockCallFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
				return nil, mockError
			},
		}

		// Call the tool
		result, err := tool.Call(context.Background(), nil)

		// Check results
		if err == nil {
			t.Error("Expected error, got nil")
		}
		if result != nil {
			t.Errorf("Expected nil result, got %v", result)
		}
	})
}

func TestMCPToolExecute(t *testing.T) {
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
	t.Run("successful execution", func(t *testing.T) {
		// Create a test tool with our mock function
		tool := &TestMCPTool{
			MCPTool: &MCPTool{
				name:        "test-tool",
				description: "Test tool",
				toolName:    "remote-test-tool",
				metadata: &core.ToolMetadata{
					Name:        "test-tool",
					Description: "Test tool",
				},
			},
			mockCallFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
				return mockResult, nil
			},
		}

		// Execute the tool
		coreResult, err := tool.Execute(context.Background(), nil)

		// Check results
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expectedData := "Test result"
		if coreResult.Data != expectedData {
			t.Errorf("Expected data '%s', got '%v'", expectedData, coreResult.Data)
		}

		isError, ok := coreResult.Metadata["isError"].(bool)
		if !ok {
			t.Error("Expected isError metadata to be a bool")
		} else if isError != false {
			t.Errorf("Expected isError to be false, got %v", isError)
		}
	})

	// Test error execution
	t.Run("error execution", func(t *testing.T) {
		// Create a test tool with our mock function
		tool := &TestMCPTool{
			MCPTool: &MCPTool{
				name:        "test-tool",
				description: "Test tool",
				toolName:    "remote-test-tool",
				metadata: &core.ToolMetadata{
					Name:        "test-tool",
					Description: "Test tool",
				},
			},
			mockCallFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
				return nil, mockError
			},
		}

		// Execute the tool
		_, err := tool.Execute(context.Background(), nil)

		// Check results
		if err == nil {
			t.Error("Expected error, got nil")
		}
	})
}

func TestMCPToolValidate(t *testing.T) {
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
	tool := &MCPTool{schema: schema}

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

func TestMCPToolType(t *testing.T) {
	tool := &MCPTool{}
	if tool.Type() != ToolTypeMCP {
		t.Errorf("Expected type '%s', got '%s'", ToolTypeMCP, tool.Type())
	}
}

func TestConvertMCPParams(t *testing.T) {
	// Define common schema parts for reuse
	schemaInt := models.ParameterSchema{Type: "integer"} // Changed type name
	schemaNum := models.ParameterSchema{Type: "number"}  // Changed type name
	schemaStr := models.ParameterSchema{Type: "string"}  // Changed type name

	// Test cases
	testCases := []struct {
		name           string
		schema         models.InputSchema
		inputParams    map[string]interface{}
		expectedParams map[string]interface{}
	}{
		{
			name: "String to Integer Success",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"count": schemaInt}, // Changed type name
			},
			inputParams:    map[string]interface{}{"count": "123"},
			expectedParams: map[string]interface{}{"count": 123},
		},
		{
			name: "String to Integer Failure (Float String)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"id": schemaInt}, // Changed type name
			},
			inputParams:    map[string]interface{}{"id": "123.45"},
			expectedParams: map[string]interface{}{"id": "123.45"}, // Stays string
		},
		{
			name: "String to Integer Failure (Non-numeric)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"id": schemaInt}, // Changed type name
			},
			inputParams:    map[string]interface{}{"id": "abc"},
			expectedParams: map[string]interface{}{"id": "abc"}, // Stays string
		},
		{
			name: "String to Number/Float Success (Integer String)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"value": schemaNum}, // Changed type name
			},
			inputParams:    map[string]interface{}{"value": "42"},
			expectedParams: map[string]interface{}{"value": 42.0},
		},
		{
			name: "String to Number/Float Success (Float String)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"price": schemaNum}, // Changed type name
			},
			inputParams:    map[string]interface{}{"price": "19.99"},
			expectedParams: map[string]interface{}{"price": 19.99},
		},
		{
			name: "String to Number/Float Failure (Non-numeric)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"rating": schemaNum}, // Changed type name
			},
			inputParams:    map[string]interface{}{"rating": "high"},
			expectedParams: map[string]interface{}{"rating": "high"}, // Stays string
		},
		{
			name: "No Conversion Needed (String)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"text": schemaStr}, // Changed type name
			},
			inputParams:    map[string]interface{}{"text": "hello"},
			expectedParams: map[string]interface{}{"text": "hello"},
		},
		{
			name: "No Conversion Needed (Int)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"count": schemaInt}, // Changed type name
			},
			inputParams:    map[string]interface{}{"count": 100},
			expectedParams: map[string]interface{}{"count": 100},
		},
		{
			name: "No Conversion Needed (Float)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"amount": schemaNum}, // Changed type name
			},
			inputParams:    map[string]interface{}{"amount": 99.9},
			expectedParams: map[string]interface{}{"amount": 99.9},
		},
		{
			name: "Mixed Types Conversion",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{ // Changed type name
					"count": schemaInt,
					"price": schemaNum,
					"label": schemaStr,
				},
			},
			inputParams: map[string]interface{}{
				"count": "5",
				"price": "25.50",
				"label": "widget",
			},
			expectedParams: map[string]interface{}{
				"count": 5,
				"price": 25.50,
				"label": "widget",
			},
		},
		{
			name: "Parameter Not In Schema",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"known": schemaStr}, // Changed type name
			},
			inputParams: map[string]interface{}{
				"known":   "yes",
				"unknown": 123,
			},
			expectedParams: map[string]interface{}{
				"known":   "yes",
				"unknown": 123, // Should pass through unchanged
			},
		},
		{
			name:           "Empty Input Params",
			schema:         models.InputSchema{Properties: map[string]models.ParameterSchema{"a": schemaStr}}, // Changed type name
			inputParams:    map[string]interface{}{},
			expectedParams: map[string]interface{}{},
		},
		{
			name:           "Empty Schema",
			schema:         models.InputSchema{Properties: map[string]models.ParameterSchema{}}, // Changed type name
			inputParams:    map[string]interface{}{"a": "1", "b": 2.0},
			expectedParams: map[string]interface{}{"a": "1", "b": 2.0}, // Should pass through
		},
		{
			name: "Case Sensitivity Check (Lowercase Schema)",
			schema: models.InputSchema{
				Properties: map[string]models.ParameterSchema{"numval": {Type: "number"}}, // Changed type name
			},
			inputParams:    map[string]interface{}{"numval": "98.6"},
			expectedParams: map[string]interface{}{"numval": 98.6},
		},
		// Add more tests as needed, e.g., for boolean conversion if implemented
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Use background context as it's not critical for conversion logic itself
			ctx := context.Background()
			actualParams := convertMCPParams(ctx, tc.schema, tc.inputParams)

			if !reflect.DeepEqual(actualParams, tc.expectedParams) {
				t.Errorf("convertMCPParams() failed:\nInput:    %v\nSchema:   %+v\nExpected: %v\nActual:   %v",
					tc.inputParams, tc.schema, tc.expectedParams, actualParams)
			}
		})
	}
}
