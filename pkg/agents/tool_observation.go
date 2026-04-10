package agents

import (
	"fmt"
	"maps"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
)

// ToolObservation separates model-visible output from operator-visible detail.
type ToolObservation struct {
	ModelText   string
	DisplayText string
	Details     map[string]any
	IsError     bool
	Synthetic   bool
	Redacted    bool
	Truncated   bool
}

// NormalizeToolResult converts a ToolResult into a dual-channel observation.
func NormalizeToolResult(result core.ToolResult) ToolObservation {
	modelText := core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta)
	displayText := core.ToolResultMetadataString(result.Metadata, core.ToolResultDisplayTextMeta)
	if displayText == "" {
		displayText = agentutil.StringifyToolResult(result)
	}
	if modelText == "" {
		modelText = displayText
	}

	return ToolObservation{
		ModelText:   strings.TrimSpace(modelText),
		DisplayText: strings.TrimSpace(displayText),
		Details:     detailsMap(result.Annotations),
		IsError:     boolMetadataValue(result.Metadata, core.ToolResultIsErrorMeta) || legacyIsError(result.Metadata),
		Synthetic:   boolMetadataValue(result.Metadata, core.ToolResultSyntheticMeta),
		Redacted:    boolMetadataValue(result.Metadata, core.ToolResultRedactedMeta),
		Truncated:   boolMetadataValue(result.Metadata, core.ToolResultTruncatedMeta),
	}
}

// BlockedToolObservation creates a synthetic error observation for blocked calls.
func BlockedToolObservation(toolName string, reason string) ToolObservation {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "blocked by tool policy"
	}
	text := fmt.Sprintf("Tool %q blocked: %s", toolName, reason)
	return ToolObservation{
		ModelText:   text,
		DisplayText: text,
		Details: map[string]any{
			"tool_name": toolName,
			"reason":    reason,
		},
		IsError:   true,
		Synthetic: true,
	}
}

func boolMetadataValue(metadata map[string]any, key string) bool {
	if metadata == nil {
		return false
	}
	value, _ := metadata[key].(bool)
	return value
}

func legacyIsError(metadata map[string]any) bool {
	if metadata == nil {
		return false
	}
	value, _ := metadata["isError"].(bool)
	return value
}

func detailsMap(annotations map[string]any) map[string]any {
	if annotations == nil {
		return nil
	}
	raw, ok := annotations[core.ToolResultDetailsAnnotation]
	if !ok || raw == nil {
		return nil
	}
	if typed, ok := raw.(map[string]any); ok {
		return maps.Clone(typed)
	}
	return map[string]any{"value": raw}
}
