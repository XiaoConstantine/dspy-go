package tools

import (
	"context"
	"encoding/xml"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
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

type XMLArgument struct {
	XMLName xml.Name `xml:"arg"`
	Key     string   `xml:"key,attr"`
	Value   string   `xml:",chardata"` // Store raw value as string for now
}

type XMLAction struct {
	XMLName   xml.Name      `xml:"action"`
	ToolName  string        `xml:"tool_name,omitempty"`
	Arguments []XMLArgument `xml:"arguments>arg,omitempty"`

	Content string `xml:",chardata"`
}

// Helper to convert XML arguments to map[string]interface{}
// Note: This currently stores all values as strings. More sophisticated type
// inference or checking could be added later if needed based on tool schemas.
func (xa *XMLAction) GetArgumentsMap() map[string]interface{} {
	argsMap := make(map[string]interface{})
	if xa == nil {
		return argsMap
	}
	// If it's a finish action or other simple action, there may be no arguments
	if len(xa.Arguments) == 0 {
		return argsMap
	}
	for _, arg := range xa.Arguments {
		argsMap[arg.Key] = arg.Value
	}
	return argsMap
}
