package tools

import (
	"testing"
)

func TestToolTypeConstants(t *testing.T) {
	// Test that the tool type constants are defined correctly
	if ToolTypeFunc != "function" {
		t.Errorf("Expected ToolTypeFunc to be 'function', got %s", ToolTypeFunc)
	}

	if ToolTypeMCP != "mcp" {
		t.Errorf("Expected ToolTypeMCP to be 'mcp', got %s", ToolTypeMCP)
	}
}
