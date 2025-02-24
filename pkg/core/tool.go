package core

import "context"

// ToolMetadata contains information about a tool's capabilities and requirements.
type ToolMetadata struct {
	Name          string            // Unique identifier for the tool
	Description   string            // Human-readable description
	InputSchema   map[string]string // Expected input parameter types
	OutputSchema  map[string]string // Expected output types
	Capabilities  []string          // List of supported capabilities
	ContextNeeded []string          // Required context keys
	Version       string            // Tool version for compatibility
}

// Tool represents a capability that can be used by both agents and modules.
type Tool interface {
	// Metadata returns the tool's metadata
	Metadata() *ToolMetadata

	// CanHandle checks if the tool can handle a specific action/intent
	CanHandle(ctx context.Context, intent string) bool

	// Execute runs the tool with provided parameters
	Execute(ctx context.Context, params map[string]interface{}) (ToolResult, error)

	// Validate checks if the parameters match the expected schema
	Validate(params map[string]interface{}) error
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
