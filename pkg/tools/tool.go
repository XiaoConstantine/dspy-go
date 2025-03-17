package tools

import (
	"context"
	"github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Tool defines a callable tool interface that abstracts both local functions
// and remote MCP tools. This provides a unified way to interact with tools
// regardless of their implementation details.
type Tool interface {
	// Name returns the tool's identifier
	Name() string

	// Description returns human-readable explanation of the tool's purpose
	Description() string

	// InputSchema returns the expected parameter structure
	InputSchema() models.InputSchema

	// Call executes the tool with the provided arguments
	Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error)
}

// ToolType represents the source/type of a tool.
type ToolType string

const (
	// ToolTypeFunc represents a tool backed by a local Go function.
	ToolTypeFunc ToolType = "function"

	// ToolTypeMCP represents a tool backed by an MCP server.
	ToolTypeMCP ToolType = "mcp"
)
