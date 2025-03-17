package core

import (
	"context"

	"github.com/XiaoConstantine/mcp-go/pkg/model"
)

// ToolMetadata contains information about a tool's capabilities and requirements.
type ToolMetadata struct {
	Name          string             // Unique identifier for the tool
	Description   string             // Human-readable description
	InputSchema   models.InputSchema // Rich schema from MCP-Go - now consistent with Tool interface
	OutputSchema  map[string]string  // Keep this field for backward compatibility
	Capabilities  []string           // List of supported capabilities
	ContextNeeded []string           // Required context keys
	Version       string             // Tool version for compatibility
}

// Tool represents a capability that can be used by both agents and modules.
type Tool interface {
	// Name returns the tool's identifier
	Name() string

	// Description returns human-readable explanation of the tool's purpose
	Description() string

	// Metadata returns the tool's metadata
	Metadata() *ToolMetadata

	// CanHandle checks if the tool can handle a specific action/intent
	CanHandle(ctx context.Context, intent string) bool

	// Execute runs the tool with provided parameters
	Execute(ctx context.Context, params map[string]interface{}) (ToolResult, error)

	// Validate checks if the parameters match the expected schema
	Validate(params map[string]interface{}) error

	// InputSchema returns the expected parameter structure
	InputSchema() models.InputSchema
}

// ToolResult wraps tool execution results with metadata.
type ToolResult struct {
	Data        interface{}            // The actual result data
	Metadata    map[string]interface{} // Execution metadata (timing, resources used, etc)
	Annotations map[string]interface{} // Additional context for result interpretation
}

// ToolRegistry manages available tools.
type ToolRegistry interface {
	Register(tool Tool) error
	Get(name string) (Tool, error)
	List() []Tool
	Match(intent string) []Tool
}
