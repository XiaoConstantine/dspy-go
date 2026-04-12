package core

import (
	"context"
	"errors"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
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
	Execute(ctx context.Context, params map[string]any) (ToolResult, error)

	// Validate checks if the parameters match the expected schema
	Validate(params map[string]any) error

	// InputSchema returns the expected parameter structure
	InputSchema() models.InputSchema
}

// CloneableTool is an optional interface for tools that can provide an
// isolated copy for concurrent agent or optimizer execution.
type CloneableTool interface {
	CloneTool() Tool
}

// ToolCall represents a tool invocation requested by an LLM.
// Used across packages: core.ChatMessage, agents.Message, tools.ParseToolCalls.
type ToolCall struct {
	ID        string         `json:"id"` // provider-assigned ID; required for tool-result correlation
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// ToolResult wraps tool execution results with metadata.
type ToolResult struct {
	Data        any            // The actual result data
	Metadata    map[string]any // Execution metadata (timing, resources used, etc)
	Annotations map[string]any // Additional context for result interpretation
}

const (
	ToolResultModelTextMeta     = "agent.model_text"
	ToolResultDisplayTextMeta   = "agent.display_text"
	ToolResultIsErrorMeta       = "agent.is_error"
	ToolResultSyntheticMeta     = "agent.synthetic"
	ToolResultRedactedMeta      = "agent.redacted"
	ToolResultTruncatedMeta     = "agent.truncated"
	ToolResultDetailsAnnotation = "agent.details"
)

var ErrToolBlocked = errors.New("tool blocked")

// ToolBlockedError indicates that a tool call was intentionally denied before execution.
type ToolBlockedError struct {
	Reason string
}

func (e *ToolBlockedError) Error() string {
	if e == nil || e.Reason == "" {
		return ErrToolBlocked.Error()
	}
	return ErrToolBlocked.Error() + ": " + e.Reason
}

func (e *ToolBlockedError) Unwrap() error {
	return ErrToolBlocked
}

// ToolApprovalDecision records the result of a policy or user approval check.
type ToolApprovalDecision struct {
	Allowed bool
	Reason  string
}

// ToolApprovalFunc decides whether a tool call should be allowed to proceed.
type ToolApprovalFunc func(ctx context.Context, info *ToolInfo, args map[string]any) (ToolApprovalDecision, error)

// ToolRegistry manages available tools.
type ToolRegistry interface {
	Register(tool Tool) error
	Get(name string) (Tool, error)
	List() []Tool
	Match(intent string) []Tool
}

// NewToolInfoFromTool creates ToolInfo from a core.Tool while tolerating nil metadata.
func NewToolInfoFromTool(tool Tool) *ToolInfo {
	info := NewToolInfo(tool.Name(), tool.Description(), "", tool.InputSchema())

	meta := tool.Metadata()
	if meta == nil {
		return info
	}

	if meta.Version != "" {
		info.WithVersion(meta.Version)
	}
	if len(meta.Capabilities) > 0 {
		info.WithCapabilities(meta.Capabilities...)
	}
	return info
}

// ToolResultMetadataString returns a string metadata value when present.
func ToolResultMetadataString(metadata map[string]any, key string) string {
	if metadata == nil {
		return ""
	}
	value, _ := metadata[key].(string)
	return value
}
