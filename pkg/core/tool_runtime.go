package core

import (
	"context"
	"errors"
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
func ToolResultMetadataString(metadata map[string]interface{}, key string) string {
	if metadata == nil {
		return ""
	}
	value, _ := metadata[key].(string)
	return value
}
