package tools

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sort"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	pkgErrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/mcp-go/pkg/logging"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
			assert.Equal(t, test.expected, result)
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
			description: "Search, query, calculate, fetch, retrieve, find, create, update, delete, git, status, commit, repository, branch, read, write, list, run, edit",
			expected:    []string{"search", "query", "calculate", "fetch", "retrieve", "find", "create", "update", "delete", "git", "status", "commit", "repository", "branch", "read", "write", "list", "run", "edit"},
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
			expected := make([]string, len(test.expected))
			copy(expected, test.expected)
			sort.Strings(capabilities)
			sort.Strings(expected)
			assert.True(t, reflect.DeepEqual(capabilities, expected), "Expected %v, got %v", expected, capabilities)
		})
	}
}

// TestCalculateToolMatchScore tests the calculateToolMatchScore helper function.
func TestCalculateToolMatchScore(t *testing.T) {
	// Small epsilon for floating point comparison
	epsilon := 0.0001

	tests := []struct {
		name          string
		metadata      *core.ToolMetadata
		action        string
		expectedScore float64 // Changed from minExpected for clarity
	}{
		{
			name: "exact name match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{}, // Explicitly empty
			},
			action:        "I need to use the search-tool",
			expectedScore: 0.6, // Corrected: 0.1 (base) + 0.5 (name)
		},
		{
			name: "capability match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:        "I need to search for something",
			expectedScore: 0.4, // 0.1 (base) + 0.3 (capability)
		},
		{
			name: "name and capability match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:        "I need to use the search-tool to search for something",
			expectedScore: 0.9, // 0.1 (base) + 0.5 (name) + 0.3 (capability)
		},
		{
			name: "no match",
			metadata: &core.ToolMetadata{
				Name:         "search-tool",
				Capabilities: []string{"search", "find"},
			},
			action:        "I want to calculate something",
			expectedScore: 0.1, // Base score only
		},
		{
			name: "multiple capability matches",
			metadata: &core.ToolMetadata{
				Name:         "repository-tool",
				Capabilities: []string{"git", "commit", "repository"},
			},
			action:        "I need to commit changes to the git repository",
			expectedScore: 1.0, // Corrected: 0.1 (base) + 0.3 (commit) + 0.3 (git) + 0.3 (repository)
		},
		{
			name:          "nil capabilities",
			metadata:      &core.ToolMetadata{Name: "nil-cap-tool"},
			action:        "use nil-cap-tool",
			expectedScore: 0.6, // 0.1 (base) + 0.5 (name)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := calculateToolMatchScore(tt.metadata, tt.action)
			assert.InEpsilon(t, tt.expectedScore, score, epsilon, "Incorrect score calculated")
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
	assert.Equal(t, "test-client", options.ClientName)
	assert.Equal(t, "1.0.0", options.ClientVersion)
	assert.NotNil(t, options.Logger)
}

// TestRegisterMCPTools tests the RegisterMCPTools function successfully registers wrapped tools.
func TestRegisterMCPTools(t *testing.T) {
	// Create mocks
	mockClient := testutil.NewMockMCPClientWithTools()
	registry := NewInMemoryToolRegistry()

	// Call the function under test, passing the mock client which satisfies the interface
	err := RegisterMCPTools(registry, mockClient)

	// Assert results
	require.NoError(t, err, "RegisterMCPTools failed unexpectedly")
	assert.Len(t, registry.List(), 2, "Expected 2 tools to be registered")

	// Check tool 1
	tool1, err := registry.Get("mcp-tool1")
	require.NoError(t, err, "Failed to get tool1")
	require.NotNil(t, tool1)
	assert.Equal(t, "mcp-tool1", tool1.Name())
	assert.Equal(t, "MCP Tool 1 description", tool1.Description())
	_, ok := tool1.(*mcpCoreToolWrapper)
	assert.True(t, ok, "Expected tool1 to be type *mcpCoreToolWrapper, got %T", tool1)

	// Check tool 2
	tool2, err := registry.Get("mcp-tool2")
	require.NoError(t, err, "Failed to get tool2")
	require.NotNil(t, tool2)
	assert.Equal(t, "mcp-tool2", tool2.Name())
	assert.Equal(t, "MCP Tool 2 description", tool2.Description())
	_, ok = tool2.(*mcpCoreToolWrapper)
	assert.True(t, ok, "Expected tool2 to be type *mcpCoreToolWrapper, got %T", tool2)
}

// TestRegisterMCPToolsError tests error handling when ListTools fails.
func TestRegisterMCPToolsError(t *testing.T) {
	// Create a mock client that returns an error
	mockClient := &testutil.MockMCPClient{
		ListToolsFunc: func(ctx context.Context, cursor *models.Cursor) (*models.ListToolsResult, error) {
			return nil, io.EOF // Simulate an error during ListTools
		},
	}
	registry := NewInMemoryToolRegistry()

	// Call the function under test
	err := RegisterMCPTools(registry, mockClient)

	// Assert results
	require.Error(t, err, "Expected an error from RegisterMCPTools")
	var e *pkgErrors.Error
	require.True(t, errors.As(err, &e), "Error should be unwrappable to *pkgErrors.Error")
	assert.Equal(t, pkgErrors.ResourceNotFound, e.Code(), "Expected ResourceNotFound error code")
	assert.ErrorIs(t, err, io.EOF, "Expected original error to be io.EOF")

	assert.Empty(t, registry.List(), "Registry should be empty after error")
}

// TestRegisterMCPToolsDuplicateError tests error handling when a tool registration fails (e.g., duplicate).
func TestRegisterMCPToolsDuplicateError(t *testing.T) {
	mockClient := testutil.NewMockMCPClientWithTools()
	registry := NewInMemoryToolRegistry()

	// Manually register one tool first to create a conflict
	// Use the new constructor for MockCoreTool
	manualTool := testutil.NewMockCoreTool("mcp-tool1", "Manual tool 1", nil) // Pass nil for default execute func
	err := registry.Register(manualTool)
	require.NoError(t, err)

	// Call the function under test - should fail when trying to register mcp-tool1 again
	err = RegisterMCPTools(registry, mockClient)

	require.Error(t, err, "Expected an error due to duplicate registration")
	var e *pkgErrors.Error
	require.True(t, errors.As(err, &e), "Error should be unwrappable to *pkgErrors.Error")
	assert.Equal(t, pkgErrors.InvalidInput, e.Code(), "Expected InvalidInput error code for duplicate")
	assert.Contains(t, err.Error(), "mcp-tool1", "Error message should mention the duplicate tool name")

	assert.LessOrEqual(t, len(registry.List()), 2, "Registry should have at most 2 tools")
	tool1, getErr := registry.Get("mcp-tool1")
	require.NoError(t, getErr)
	assert.IsType(t, &testutil.MockCoreTool{}, tool1, "mcp-tool1 should still be the manually registered MockCoreTool")
}
