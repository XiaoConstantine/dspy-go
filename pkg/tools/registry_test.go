package tools

import (
	"fmt"
	"testing"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	pkgErrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewInMemoryRegistry(t *testing.T) {
	registry := NewInMemoryToolRegistry() // Use the new constructor
	require.NotNil(t, registry, "Expected non-nil registry")
	assert.NotNil(t, registry.tools, "Expected tools map to be initialized")
	assert.Empty(t, registry.tools, "Expected empty tools map")
}

func TestInMemoryRegister(t *testing.T) {
	registry := NewInMemoryToolRegistry()

	// Create a mock core tool using the constructor
	mockTool := testutil.NewMockCoreTool("test-tool", "A test tool", nil)

	// Test successful registration
	err := registry.Register(mockTool)
	assert.NoError(t, err, "Unexpected error during registration")

	// Verify registration
	retrievedTool, err := registry.Get("test-tool")
	assert.NoError(t, err, "Failed to get registered tool")
	assert.Equal(t, mockTool, retrievedTool, "Retrieved tool does not match registered tool")

	// Test duplicate registration
	err = registry.Register(mockTool)
	assert.Error(t, err, "Expected error for duplicate registration, got nil")
	if err != nil {
		// Check if it's the expected error type using type assertion
		e, ok := err.(*pkgErrors.Error)
		assert.True(t, ok, "Error should be of type *pkgErrors.Error")
		if ok {
			assert.Equal(t, pkgErrors.InvalidInput, e.Code(), "Expected InvalidInput error code")
		}
	}

	// Test registering nil tool
	err = registry.Register(nil)
	assert.Error(t, err, "Expected error for registering nil tool")
	if err != nil {
		e, ok := err.(*pkgErrors.Error)
		assert.True(t, ok, "Error should be of type *pkgErrors.Error")
		if ok {
			assert.Equal(t, pkgErrors.InvalidInput, e.Code(), "Expected InvalidInput error code")
		}
	}
}

func TestInMemoryGet(t *testing.T) {
	registry := NewInMemoryToolRegistry()

	// Create and register a mock core tool using the constructor
	mockTool := testutil.NewMockCoreTool("test-tool", "", nil)
	err := registry.Register(mockTool)
	require.NoError(t, err)

	// Test successful retrieval
	tool, err := registry.Get("test-tool")
	assert.NoError(t, err, "Unexpected error getting existing tool")
	require.NotNil(t, tool, "Expected non-nil tool")
	assert.Equal(t, "test-tool", tool.Name(), "Expected name 'test-tool'")

	// Test non-existent tool
	tool, err = registry.Get("non-existent-tool")
	assert.Error(t, err, "Expected error for non-existent tool, got nil")
	assert.Nil(t, tool, "Expected nil tool for non-existent name")
	if err != nil {
		e, ok := err.(*pkgErrors.Error)
		assert.True(t, ok, "Error should be of type *pkgErrors.Error")
		if ok {
			assert.Equal(t, pkgErrors.ResourceNotFound, e.Code(), "Expected ResourceNotFound error code")
		}
	}
}

func TestInMemoryList(t *testing.T) {
	registry := NewInMemoryToolRegistry()

	// Check empty list
	tools := registry.List()
	assert.Empty(t, tools, "Expected empty list for new registry")

	// Add some tools using the constructor
	mockTool1 := testutil.NewMockCoreTool("tool1", "Tool 1", nil)
	mockTool2 := testutil.NewMockCoreTool("tool2", "Tool 2", nil)

	err := registry.Register(mockTool1)
	require.NoError(t, err)
	err = registry.Register(mockTool2)
	require.NoError(t, err)

	// Check list with tools
	tools = registry.List()
	assert.Len(t, tools, 2, "Expected 2 tools in the list")

	// Check that all tools are in the list (order doesn't matter)
	foundNames := make(map[string]bool)
	for _, tool := range tools {
		foundNames[tool.Name()] = true
	}
	assert.True(t, foundNames["tool1"], "Expected tool1 in the list")
	assert.True(t, foundNames["tool2"], "Expected tool2 in the list")
}

func TestInMemoryMatch(t *testing.T) {
	registry := NewInMemoryToolRegistry()

	// Add some tools using the constructor
	mockTool1 := testutil.NewMockCoreTool("ReadFile", "Reads a file", nil)
	mockTool2 := testutil.NewMockCoreTool("WriteFile", "Writes a file", nil)
	mockTool3 := testutil.NewMockCoreTool("ListFiles", "Lists directory contents", nil)

	err := registry.Register(mockTool1)
	require.NoError(t, err)
	err = registry.Register(mockTool2)
	require.NoError(t, err)
	err = registry.Register(mockTool3)
	require.NoError(t, err)

	tests := []struct {
		intent       string
		expectedLen  int
		expectedName string // Only checks first match if len=1
	}{
		{"read the input file", 0, ""},             // "read the input file" does NOT contain "readfile"
		{"I want to WRITE a file", 0, ""},          // "i want to write a file" does NOT contain "writefile"
		{"list the files in the directory", 0, ""}, // does NOT contain "listfiles"
		{"something about files", 0, ""},           // does NOT contain "readfile", "writefile", or "listfiles"
		{"delete everything", 0, ""},               // No matching tool names
		{"READFILE", 1, "ReadFile"},                // "readfile" contains "readfile"
		{"use WriteFile command", 1, "WriteFile"},  // "use writefile command" contains "writefile"
		{"ListFilesPlease", 1, "ListFiles"},        // "listfilesplease" contains "listfiles"
		{"no match", 0, ""},
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("Intent_%s", tt.intent), func(t *testing.T) {
			matches := registry.Match(tt.intent)
			assert.Len(t, matches, tt.expectedLen, "Incorrect number of matches for intent: '%s'", tt.intent)
			if tt.expectedLen == 1 && len(matches) == 1 {
				assert.Equal(t, tt.expectedName, matches[0].Name(), "Incorrect tool matched for intent: '%s'", tt.intent)
			}
		})
	}
}
