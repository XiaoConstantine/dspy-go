// pkg/tools/func.go
package tools

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// ToolFunc represents a function that can be called as a tool.
type ToolFunc func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error)

// FuncTool wraps a Go function as a Tool implementation.
type FuncTool struct {
	name        string
	description string
	schema      models.InputSchema
	fn          ToolFunc
	metadata    *core.ToolMetadata
	matchCutoff float64
}

// NewFuncTool creates a new function-based tool.
func NewFuncTool(name, description string, schema models.InputSchema, fn ToolFunc) *FuncTool {
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

	return &FuncTool{
		name:        name,
		description: description,
		schema:      schema,
		fn:          fn,
		metadata:    metadata,
		matchCutoff: 0.3,
	}
}

// Name returns the tool's identifier.
func (t *FuncTool) Name() string {
	return t.name
}

// Description returns human-readable explanation of the tool.
func (t *FuncTool) Description() string {
	return t.description
}

// InputSchema returns the expected parameter structure.
func (t *FuncTool) InputSchema() models.InputSchema {
	return t.schema
}

// Metadata returns the tool's metadata for intent matching.
func (t *FuncTool) Metadata() *core.ToolMetadata {
	return t.metadata
}

// CanHandle checks if the tool can handle a specific action/intent.
func (t *FuncTool) CanHandle(ctx context.Context, intent string) bool {
	score := calculateToolMatchScore(t.metadata, intent)
	return score >= t.matchCutoff
}

// Call executes the wrapped function with the provided arguments.
func (t *FuncTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	return t.fn(ctx, args)
}

// Execute runs the tool with provided parameters and adapts the result to the core interface.
func (t *FuncTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := t.Call(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}

	// Convert CallToolResult to core.ToolResult
	toolResult := core.ToolResult{
		Data:        extractContentText(result.Content),
		Metadata:    map[string]interface{}{"isError": result.IsError},
		Annotations: map[string]interface{}{},
	}

	return toolResult, nil
}

// Validate checks if the parameters match the expected schema.
func (t *FuncTool) Validate(params map[string]interface{}) error {
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
func (t *FuncTool) Type() ToolType {
	return ToolTypeFunc
}
