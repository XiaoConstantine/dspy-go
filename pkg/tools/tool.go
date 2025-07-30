package tools

import (
	"context"
	"encoding/xml"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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

// InterceptableTool extends Tool with interceptor support.
// This interface provides backward-compatible enhancement for tools that support interceptors.
type InterceptableTool interface {
	Tool

	// CallWithInterceptors executes the tool with interceptor support
	CallWithInterceptors(ctx context.Context, args map[string]interface{}, interceptors []core.ToolInterceptor) (*models.CallToolResult, error)

	// SetInterceptors sets the default interceptors for this tool instance
	SetInterceptors(interceptors []core.ToolInterceptor)

	// GetInterceptors returns the current interceptors for this tool
	GetInterceptors() []core.ToolInterceptor

	// ClearInterceptors removes all interceptors from this tool
	ClearInterceptors()

	// GetToolType returns the category/type of this tool
	GetToolType() string

	// GetVersion returns the tool version
	GetVersion() string
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

// InterceptorToolWrapper wraps an existing Tool to provide interceptor support.
// This allows any existing tool to be used with interceptors without modifying its implementation.
type InterceptorToolWrapper struct {
	tool                     Tool
	interceptors             []core.ToolInterceptor
	toolType                 string
	version                  string
	lastExecutionMetadata    map[string]interface{} // Preserves metadata from last interceptor execution
	lastExecutionAnnotations map[string]interface{} // Preserves annotations from last interceptor execution
}

// NewInterceptorToolWrapper creates a new wrapper that adds interceptor support to an existing tool.
func NewInterceptorToolWrapper(tool Tool, toolType, version string) *InterceptorToolWrapper {
	return &InterceptorToolWrapper{
		tool:                     tool,
		interceptors:             make([]core.ToolInterceptor, 0),
		toolType:                 toolType,
		version:                  version,
		lastExecutionMetadata:    make(map[string]interface{}),
		lastExecutionAnnotations: make(map[string]interface{}),
	}
}

// Name returns the tool's identifier from the wrapped tool.
func (itw *InterceptorToolWrapper) Name() string {
	return itw.tool.Name()
}

// Description returns human-readable explanation from the wrapped tool.
func (itw *InterceptorToolWrapper) Description() string {
	return itw.tool.Description()
}

// InputSchema returns the expected parameter structure from the wrapped tool.
func (itw *InterceptorToolWrapper) InputSchema() models.InputSchema {
	return itw.tool.InputSchema()
}

// Call executes the tool with the provided arguments using the wrapped tool.
func (itw *InterceptorToolWrapper) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	return itw.tool.Call(ctx, args)
}

// CallWithInterceptors executes the tool with interceptor support.
func (itw *InterceptorToolWrapper) CallWithInterceptors(ctx context.Context, args map[string]interface{}, interceptors []core.ToolInterceptor) (*models.CallToolResult, error) {
	// Use provided interceptors, or fall back to wrapper's default interceptors
	if interceptors == nil {
		interceptors = itw.interceptors
	}

	// Create tool info for interceptors
	info := core.NewToolInfo(itw.tool.Name(), itw.tool.Description(), itw.toolType, itw.tool.InputSchema())
	info.WithVersion(itw.version)

	// Create the base handler that calls the wrapped tool and converts the result
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		result, err := itw.tool.Call(ctx, args)
		if err != nil {
			return core.ToolResult{}, err
		}
		// Convert models.CallToolResult to core.ToolResult
		return core.ToolResult{
			Data:        result,
			Metadata:    make(map[string]interface{}),
			Annotations: make(map[string]interface{}),
		}, nil
	}

	// Chain the interceptors
	chainedInterceptor := core.ChainToolInterceptors(interceptors...)

	// Execute with interceptors
	coreResult, err := chainedInterceptor(ctx, args, info, handler)
	if err != nil {
		return nil, err
	}

	// Preserve interceptor metadata and annotations for later retrieval
	itw.preserveExecutionData(coreResult)

	// Convert core.ToolResult back to models.CallToolResult
	if mcpResult, ok := coreResult.Data.(*models.CallToolResult); ok {
		return mcpResult, nil
	}

	// If the result is not already a CallToolResult, create one
	return &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: fmt.Sprintf("%v", coreResult.Data),
			},
		},
		IsError: false,
	}, nil
}

// SetInterceptors sets the default interceptors for this wrapper.
func (itw *InterceptorToolWrapper) SetInterceptors(interceptors []core.ToolInterceptor) {
	itw.interceptors = make([]core.ToolInterceptor, len(interceptors))
	copy(itw.interceptors, interceptors)
}

// GetInterceptors returns the current interceptors for this wrapper.
func (itw *InterceptorToolWrapper) GetInterceptors() []core.ToolInterceptor {
	result := make([]core.ToolInterceptor, len(itw.interceptors))
	copy(result, itw.interceptors)
	return result
}

// ClearInterceptors removes all interceptors from this wrapper.
func (itw *InterceptorToolWrapper) ClearInterceptors() {
	itw.interceptors = nil
}

// GetToolType returns the category/type of this tool.
func (itw *InterceptorToolWrapper) GetToolType() string {
	return itw.toolType
}

// GetVersion returns the tool version.
func (itw *InterceptorToolWrapper) GetVersion() string {
	return itw.version
}

// preserveExecutionData stores the metadata and annotations from the interceptor execution.
func (itw *InterceptorToolWrapper) preserveExecutionData(coreResult core.ToolResult) {
	// Deep copy metadata to avoid reference issues
	if coreResult.Metadata != nil {
		itw.lastExecutionMetadata = make(map[string]interface{})
		for k, v := range coreResult.Metadata {
			itw.lastExecutionMetadata[k] = v
		}
	} else {
		itw.lastExecutionMetadata = make(map[string]interface{})
	}

	// Deep copy annotations to avoid reference issues
	if coreResult.Annotations != nil {
		itw.lastExecutionAnnotations = make(map[string]interface{})
		for k, v := range coreResult.Annotations {
			itw.lastExecutionAnnotations[k] = v
		}
	} else {
		itw.lastExecutionAnnotations = make(map[string]interface{})
	}
}

// GetLastExecutionMetadata returns the metadata from the most recent interceptor execution.
// This allows access to rich data added by interceptors (e.g., timing, logging, metrics).
func (itw *InterceptorToolWrapper) GetLastExecutionMetadata() map[string]interface{} {
	// Return a copy to prevent external modification
	result := make(map[string]interface{})
	for k, v := range itw.lastExecutionMetadata {
		result[k] = v
	}
	return result
}

// GetLastExecutionAnnotations returns the annotations from the most recent interceptor execution.
// This allows access to additional context added by interceptors (e.g., tracing, debugging info).
func (itw *InterceptorToolWrapper) GetLastExecutionAnnotations() map[string]interface{} {
	// Return a copy to prevent external modification
	result := make(map[string]interface{})
	for k, v := range itw.lastExecutionAnnotations {
		result[k] = v
	}
	return result
}

// WrapToolWithInterceptors is a convenience function to wrap any tool with interceptor support.
func WrapToolWithInterceptors(tool Tool, toolType, version string, interceptors ...core.ToolInterceptor) InterceptableTool {
	wrapper := NewInterceptorToolWrapper(tool, toolType, version)
	if len(interceptors) > 0 {
		wrapper.SetInterceptors(interceptors)
	}
	return wrapper
}
