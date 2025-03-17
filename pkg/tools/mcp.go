package tools

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/mcp-go/pkg/client"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// MCPTool represents a tool that delegates to an MCP server.
type MCPTool struct {
	name        string
	description string
	schema      models.InputSchema
	client      *client.Client
	toolName    string
	metadata    *core.ToolMetadata
	matchCutoff float64
}

// NewMCPTool creates a new MCP-based tool.
func NewMCPTool(name, description string, schema models.InputSchema,
	client *client.Client, toolName string) *MCPTool {

	// Extract capabilities from description
	capabilities := extractCapabilities(description)

	// Create the metadata with the full schema - no conversion needed!
	metadata := &core.ToolMetadata{
		Name:         name,
		Description:  description,
		InputSchema:  schema, // Use the schema directly - no conversion required
		Capabilities: capabilities,
		Version:      "1.0.0",
	}

	return &MCPTool{
		name:        name,
		description: description,
		schema:      schema,
		client:      client,
		toolName:    toolName,
		metadata:    metadata,
		matchCutoff: 0.3,
	}
}

// Name returns the tool's identifier.
func (t *MCPTool) Name() string {
	return t.name
}

// Description returns human-readable explanation of the tool.
func (t *MCPTool) Description() string {
	return t.description
}

// InputSchema returns the expected parameter structure.
func (t *MCPTool) InputSchema() models.InputSchema {
	return t.schema
}

// Metadata returns the tool's metadata for intent matching.
func (t *MCPTool) Metadata() *core.ToolMetadata {
	return t.metadata
}

// CanHandle checks if the tool can handle a specific action/intent.
func (t *MCPTool) CanHandle(ctx context.Context, intent string) bool {
	score := calculateToolMatchScore(t.metadata, intent)
	return score >= t.matchCutoff
}

// Call forwards the call to the MCP server and returns the result.
func (t *MCPTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	return t.client.CallTool(ctx, t.toolName, args)
}

// Execute runs the tool with provided parameters and adapts the result to the core interface.
func (t *MCPTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := t.Call(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}

	// Convert MCP call result to core.ToolResult
	toolResult := core.ToolResult{
		Data:        extractContentText(result.Content),
		Metadata:    map[string]interface{}{"isError": result.IsError},
		Annotations: map[string]interface{}{},
	}

	return toolResult, nil
}

// Validate checks if the parameters match the expected schema.
func (t *MCPTool) Validate(params map[string]interface{}) error {
	// Use the full InputSchema for validation
	for name, param := range t.schema.Properties {
		if param.Required {
			if _, exists := params[name]; !exists {
				return fmt.Errorf("missing required parameter: %s", name)
			}

			// We could add type checking here based on param.Type
			// For example, check if numbers are actually numbers, etc.
		}
	}

	return nil
}

// Type returns the tool type.
func (t *MCPTool) Type() ToolType {
	return ToolTypeMCP
}

