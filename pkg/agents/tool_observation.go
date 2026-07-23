package agents

import (
	"fmt"
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
	modelText, hasModelText := toolResultTextOverride(result.Metadata, core.ToolResultModelTextMeta)
	displayText, hasDisplayText := toolResultTextOverride(result.Metadata, core.ToolResultDisplayTextMeta)
	if !hasDisplayText {
		displayText = agentutil.StringifyToolResult(result)
	}
	if !hasModelText {
		modelText = displayText
	}

	return ToolObservation{
		ModelText:   strings.TrimSpace(modelText),
		DisplayText: strings.TrimSpace(displayText),
		Details:     detailsMap(result.Annotations),
		IsError:     toolResultIsError(result),
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

func toolResultTextOverride(metadata map[string]any, key string) (string, bool) {
	if metadata == nil {
		return "", false
	}
	value, exists := metadata[key]
	if !exists {
		return "", false
	}
	text, _ := value.(string)
	return text, true
}

func boolMetadataValue(metadata map[string]any, key string) bool {
	if metadata == nil {
		return false
	}
	value, _ := metadata[key].(bool)
	return value
}

func toolResultIsError(result core.ToolResult) bool {
	return boolMetadataValue(result.Metadata, core.ToolResultIsErrorMeta) ||
		boolMetadataValue(result.Metadata, "isError") ||
		boolMetadataValue(result.Annotations, "is_error") ||
		boolMetadataValue(result.Annotations, "isError")
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
		return cloneAnyMap(typed)
	}
	return map[string]any{"value": raw}
}
