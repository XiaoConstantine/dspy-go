package interceptors

import (
	"context"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
)

// ApprovalToolInterceptor blocks tool execution when the approval callback denies it.
func ApprovalToolInterceptor(approve core.ToolApprovalFunc) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		if approve == nil {
			return handler(ctx, args)
		}

		decision, err := approve(ctx, info, core.ShallowCopyMap(args))
		if err != nil {
			return core.ToolResult{}, err
		}
		if !decision.Allowed {
			reason := strings.TrimSpace(decision.Reason)
			if reason == "" {
				reason = "tool call denied"
			}
			return core.ToolResult{}, &core.ToolBlockedError{Reason: reason}
		}

		return handler(ctx, args)
	}
}

// RedactionToolInterceptor rewrites model/display text and details using the provided redactor.
func RedactionToolInterceptor(redact func(string) string) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		result, err := handler(ctx, args)
		if err != nil {
			return result, err
		}
		if redact == nil {
			return result, nil
		}

		if result.Metadata == nil {
			result.Metadata = make(map[string]interface{})
		}
		if result.Annotations == nil {
			result.Annotations = make(map[string]interface{})
		}

		modelText := core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta)
		displayText := core.ToolResultMetadataString(result.Metadata, core.ToolResultDisplayTextMeta)
		if displayText == "" {
			displayText = agentutil.StringifyToolResult(result)
		}
		if modelText == "" {
			modelText = displayText
		}

		redacted := false
		redactedModel := redact(modelText)
		redactedDisplay := redact(displayText)
		if redactedModel != modelText || redactedDisplay != displayText {
			redacted = true
		}

		result.Metadata[core.ToolResultModelTextMeta] = redactedModel
		result.Metadata[core.ToolResultDisplayTextMeta] = redactedDisplay

		if details, ok := result.Annotations[core.ToolResultDetailsAnnotation].(map[string]any); ok {
			scrubbed := core.ShallowCopyMap(details)
			for key, value := range scrubbed {
				text, ok := value.(string)
				if !ok {
					continue
				}
				updated := redact(text)
				if updated != text {
					redacted = true
				}
				scrubbed[key] = updated
			}
			result.Annotations[core.ToolResultDetailsAnnotation] = scrubbed
		}

		if redacted {
			result.Metadata[core.ToolResultRedactedMeta] = true
		}
		return result, nil
	}
}

// TruncationToolInterceptor trims model/display text while preserving the full raw result.
func TruncationToolInterceptor(modelLimit int, displayLimit int) core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		result, err := handler(ctx, args)
		if err != nil {
			return result, err
		}

		if result.Metadata == nil {
			result.Metadata = make(map[string]interface{})
		}

		modelText := core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta)
		displayText := core.ToolResultMetadataString(result.Metadata, core.ToolResultDisplayTextMeta)
		if displayText == "" {
			displayText = agentutil.StringifyToolResult(result)
		}
		if modelText == "" {
			modelText = displayText
		}

		truncated := false
		if displayLimit > 0 && len(displayText) > displayLimit {
			displayText = agentutil.TruncateString(displayText, displayLimit)
			truncated = true
		}
		if modelLimit > 0 && len(modelText) > modelLimit {
			modelText = agentutil.TruncateString(modelText, modelLimit)
			truncated = true
		}

		result.Metadata[core.ToolResultModelTextMeta] = modelText
		result.Metadata[core.ToolResultDisplayTextMeta] = displayText
		if truncated {
			result.Metadata[core.ToolResultTruncatedMeta] = true
		}

		return result, nil
	}
}
