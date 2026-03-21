package interceptors

import (
	"context"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApprovalToolInterceptor_BlocksDeniedCalls(t *testing.T) {
	interceptor := ApprovalToolInterceptor(func(ctx context.Context, info *core.ToolInfo, args map[string]any) (core.ToolApprovalDecision, error) {
		return core.ToolApprovalDecision{Allowed: false, Reason: "approval denied"}, nil
	})

	handlerCalled := false
	_, err := interceptor(context.Background(), map[string]interface{}{"cmd": "rm"}, core.NewToolInfo("bash", "bash", "", models.InputSchema{}), func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		handlerCalled = true
		return core.ToolResult{}, nil
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, core.ErrToolBlocked)
	assert.False(t, handlerCalled)
}

func TestApprovalToolInterceptor_AllowsApprovedCalls(t *testing.T) {
	interceptor := ApprovalToolInterceptor(func(ctx context.Context, info *core.ToolInfo, args map[string]any) (core.ToolApprovalDecision, error) {
		return core.ToolApprovalDecision{Allowed: true}, nil
	})

	handlerCalled := false
	result, err := interceptor(context.Background(), map[string]interface{}{"path": "a.txt"}, core.NewToolInfo("write", "write", "", models.InputSchema{}), func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		handlerCalled = true
		return core.ToolResult{Data: "ok"}, nil
	})
	require.NoError(t, err)
	assert.True(t, handlerCalled)
	assert.Equal(t, "ok", result.Data)
}

func TestRedactionToolInterceptor_RedactsTextAndDetails(t *testing.T) {
	interceptor := RedactionToolInterceptor(func(input string) string {
		return strings.ReplaceAll(input, "secret", "[REDACTED]")
	})

	result, err := interceptor(context.Background(), nil, core.NewToolInfo("read", "read", "", models.InputSchema{}), func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return core.ToolResult{
			Data: "raw secret payload",
			Metadata: map[string]interface{}{
				core.ToolResultModelTextMeta:   "model secret text",
				core.ToolResultDisplayTextMeta: "display secret text",
			},
			Annotations: map[string]interface{}{
				core.ToolResultDetailsAnnotation: map[string]any{
					"stdout": "secret line",
				},
			},
		}, nil
	})
	require.NoError(t, err)
	assert.Equal(t, "model [REDACTED] text", result.Metadata[core.ToolResultModelTextMeta])
	assert.Equal(t, "display [REDACTED] text", result.Metadata[core.ToolResultDisplayTextMeta])
	assert.Equal(t, true, result.Metadata[core.ToolResultRedactedMeta])
	details, ok := result.Annotations[core.ToolResultDetailsAnnotation].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "[REDACTED] line", details["stdout"])
}

func TestTruncationToolInterceptor_TrimsModelAndDisplayText(t *testing.T) {
	interceptor := TruncationToolInterceptor(12, 18)

	result, err := interceptor(context.Background(), nil, core.NewToolInfo("read", "read", "", models.InputSchema{}), func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return core.ToolResult{
			Data: "raw payload that is much longer than the limits",
			Metadata: map[string]interface{}{
				core.ToolResultModelTextMeta:   "model payload that is much longer than the limit",
				core.ToolResultDisplayTextMeta: "display payload that is much longer than the limit",
			},
		}, nil
	})
	require.NoError(t, err)
	assert.Equal(t, "model pay...", result.Metadata[core.ToolResultModelTextMeta])
	assert.Equal(t, "display payload...", result.Metadata[core.ToolResultDisplayTextMeta])
	assert.Equal(t, true, result.Metadata[core.ToolResultTruncatedMeta])
}
