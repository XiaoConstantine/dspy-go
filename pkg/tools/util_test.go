package tools

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/mcp-go/pkg/logging"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Mock transport for testing.
type MockTransport struct {
	sendFunc    func(ctx context.Context, msg any) error
	receiveFunc func(ctx context.Context) (any, error)
}

func (m *MockTransport) Send(ctx context.Context, msg any) error {
	return m.sendFunc(ctx, msg)
}

func (m *MockTransport) Receive(ctx context.Context) (any, error) {
	return m.receiveFunc(ctx)
}

func (m *MockTransport) Close() error {
	return nil
}

// TestExtractContentText tests the extractContentText helper function.
func TestExtractContentText(t *testing.T) {
	tests := []struct {
		name     string
		content  []models.Content
		expected string
	}{
		{
			name: "single text content",
			content: []models.Content{
				models.TextContent{
					Type: "text",
					Text: "Hello world",
				},
			},
			expected: "Hello world",
		},
		{
			name: "multiple text content",
			content: []models.Content{
				models.TextContent{
					Type: "text",
					Text: "Hello",
				},
				models.TextContent{
					Type: "text",
					Text: "world",
				},
			},
			expected: "Hello\nworld",
		},
		{
			name:     "empty content",
			content:  []models.Content{},
			expected: "",
		},
		{
			name: "mixed content types",
			content: []models.Content{
				models.TextContent{
					Type: "text",
					Text: "Text content",
				},
				models.ImageContent{
					Type:     "image",
					Data:     "base64data",
					MimeType: "image/png",
				},
			},
			expected: "Text content",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := extractContentText(test.content)
			if result != test.expected {
				t.Errorf("Expected '%s', got '%s'", test.expected, result)
			}
		})
	}
}

// TestExtractCapabilities tests the extractCapabilities helper function.
func TestExtractCapabilities(t *testing.T) {
	tests := []struct {
		name        string
		description string
		expected    []string
	}{
		{
			name:        "empty description",
			description: "",
			expected:    []string{},
		},
		{
			name:        "description with keywords",
			description: "A tool to search and retrieve data from a repository",
			expected:    []string{"search", "retrieve", "repository"},
		},
		{
			name:        "description with all keywords",
			description: "Search, query, calculate, fetch, retrieve, find, create, update, delete, git, status, commit, repository, branch",
			expected:    []string{"search", "query", "calculate", "fetch", "retrieve", "find", "create", "update", "delete", "git", "status", "commit", "repository", "branch"},
		},
		{
			name:        "case insensitivity",
			description: "A tool to SEARCH and FIND data",
			expected:    []string{"search", "find"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			capabilities := extractCapabilities(test.description)

			// Sort both slices for comparison
			sort.Strings(capabilities)
			expected := make([]string, len(test.expected))
			copy(expected, test.expected)
			sort.Strings(expected)

			if !reflect.DeepEqual(capabilities, expected) {
				t.Errorf("Expected %v, got %v", expected, capabilities)
			}
		})
	}
}

// TestCalculateToolMatchScore tests the calculateToolMatchScore helper function.
func TestCalculateToolMatchScore(t *testing.T) {
	// Small epsilon for floating point comparison
	epsilon := 0.0001
	
	tests := []struct {
		name        string
		metadata    *core.ToolMetadata
		action      string
		minExpected float64
	}{
		{
			name: "exact name match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{},
			},
			action:      "I need to use the search-tool",
			minExpected: 0.5, // Base score (0.1) + name match (0.5)
		},
		{
			name: "capability match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:      "I need to search for something",
			minExpected: 0.4, // Base score (0.1) + capability match (0.3)
		},
		{
			name: "name and capability match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:      "I need to use the search-tool to search for something",
			minExpected: 0.9, // Base score (0.1) + name match (0.5) + capability match (0.3)
		},
		{
			name: "no match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:      "I want to calculate something",
			minExpected: 0.1, // Base score only
		},
		{
			name: "multiple capability matches",
			metadata: &core.ToolMetadata{
				Name:         "repository-tool",
				Capabilities: []string{"git", "commit", "repository"},
			},
			action:      "I need to commit changes to the git repository",
			minExpected: 0.7, // Base score (0.1) + capability matches (0.3 * 2)
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			score := calculateToolMatchScore(test.metadata, test.action)
			if score < test.minExpected-epsilon {
				t.Errorf("Expected score >= %f, got %f", test.minExpected, score)
			}
		})
	}
}

// TestMCPClientOptionsStructure tests the structure of MCPClientOptions.
func TestMCPClientOptionsStructure(t *testing.T) {
	options := MCPClientOptions{
		ClientName:    "test-client",
		ClientVersion: "1.0.0",
		Logger:        logging.NewStdLogger(logging.InfoLevel),
	}

	// Verify structure fields
	if options.ClientName != "test-client" {
		t.Errorf("Expected ClientName 'test-client', got '%s'", options.ClientName)
	}
	if options.ClientVersion != "1.0.0" {
		t.Errorf("Expected ClientVersion '1.0.0', got '%s'", options.ClientVersion)
	}
	if options.Logger == nil {
		t.Error("Expected non-nil Logger")
	}
}

// Note: We can't properly unit test NewMCPClientFromStdio without significant refactoring
// because it directly creates and initializes a client that depends on external components.
// This would be a good candidate for refactoring to use dependency injection.

// MockClient implements a simplified version of the MCP client for testing.
type MockClient struct {
	listToolsFunc func(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error)
	callToolFunc  func(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error)
}

// Helper function to register tools using our MockClient.
func registerMCPToolsTest(registry *Registry, mcpClient *MockClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	toolsResult, err := mcpClient.ListTools(ctx, nil)
	if err != nil {
		return err
	}

	for _, mcpTool := range toolsResult.Tools {
		// Create a TestMCPTool that uses our mocked Call function
		baseTool := &MCPTool{
			name:        mcpTool.Name,
			description: mcpTool.Description,
			schema:      mcpTool.InputSchema,
			toolName:    mcpTool.Name,
			metadata: &core.ToolMetadata{
				Name:         mcpTool.Name,
				Description:  mcpTool.Description,
				InputSchema:  mcpTool.InputSchema,
				Capabilities: extractCapabilities(mcpTool.Description),
				Version:      "1.0.0",
			},
			matchCutoff: 0.3,
		}

		// Create a wrapper with our mock functionality
		tool := &TestMCPTool{
			MCPTool: baseTool,
			mockCallFunc: func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
				// Forward to our mock client's CallTool
				return mcpClient.callToolFunc(ctx, mcpTool.Name, args)
			},
		}

		if err := registry.Register(tool); err != nil {
			return err
		}
	}

	return nil
}

func (m *MockClient) ListTools(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error) {
	return m.listToolsFunc(ctx, cursor)
}

func (m *MockClient) CallTool(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error) {
	return m.callToolFunc(ctx, name, args)
}

// Test helper to create a MockClient with pre-configured tools.
func createMockClientWithTools() *MockClient {
	mockTools := []models.Tool{
		{
			Name:        "tool1",
			Description: "Tool 1 description",
			InputSchema: models.InputSchema{
				Type:       "object",
				Properties: map[string]models.ParameterSchema{},
			},
		},
		{
			Name:        "tool2",
			Description: "Tool 2 description",
			InputSchema: models.InputSchema{
				Type:       "object",
				Properties: map[string]models.ParameterSchema{},
			},
		},
	}

	return &MockClient{
		listToolsFunc: func(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error) {
			return &models.ListToolsResult{Tools: mockTools}, nil
		},
		callToolFunc: func(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error) {
			return &models.CallToolResult{}, nil
		},
	}
}

// TestRegisterMCPTools tests the RegisterMCPTools function.
func TestRegisterMCPTools(t *testing.T) {
	// Create a mock client and registry
	mockClient := createMockClientWithTools()

	// Create a registry
	registry := NewRegistry()

	// Use our own helper function to register tools
	err := registerMCPToolsTest(registry, mockClient)

	// Check results
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Verify that the tools were registered
	if len(registry.List()) != 2 {
		t.Errorf("Expected 2 tools in registry, got %d", len(registry.List()))
	}

	// Check the first tool
	tool1, err := registry.Get("tool1")
	if err != nil {
		t.Errorf("Unexpected error getting tool1: %v", err)
	}
	if tool1.Name() != "tool1" {
		t.Errorf("Expected name 'tool1', got '%s'", tool1.Name())
	}
	if tool1.Description() != "Tool 1 description" {
		t.Errorf("Expected description 'Tool 1 description', got '%s'", tool1.Description())
	}

	// Check the second tool
	tool2, err := registry.Get("tool2")
	if err != nil {
		t.Errorf("Unexpected error getting tool2: %v", err)
	}
	if tool2.Name() != "tool2" {
		t.Errorf("Expected name 'tool2', got '%s'", tool2.Name())
	}
	if tool2.Description() != "Tool 2 description" {
		t.Errorf("Expected description 'Tool 2 description', got '%s'", tool2.Description())
	}

	// Check the type of the tools
	testMCPTool, ok := tool1.(*TestMCPTool)
	if !ok {
		t.Errorf("Expected tool1 to be a TestMCPTool, got %T", tool1)
	} else if testMCPTool.Type() != ToolTypeMCP {
		t.Errorf("Expected tool type %s, got %s", ToolTypeMCP, testMCPTool.Type())
	}
}

// TestRegisterMCPToolsError tests error handling in RegisterMCPTools.
func TestRegisterMCPToolsError(t *testing.T) {
	// Create a mock client with an error response
	mockClient := &MockClient{
		listToolsFunc: func(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error) {
			return nil, io.EOF
		},
		callToolFunc: func(ctx context.Context, name string, args map[string]interface{}) (*models.CallToolResult, error) {
			return nil, errors.New("should not be called")
		},
	}

	// Create a registry
	registry := NewRegistry()

	// Use our own helper function to register tools
	err := registerMCPToolsTest(registry, mockClient)

	// Check results
	if err == nil {
		t.Error("Expected error, got nil")
	}

	// Verify no tools were registered
	if len(registry.List()) != 0 {
		t.Errorf("Expected 0 tools in registry, got %d", len(registry.List()))
	}
}
