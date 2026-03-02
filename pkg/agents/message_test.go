package agents

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessage_ShouldSendToLLM(t *testing.T) {
	internal := Message{Role: RoleInternal}
	user := Message{Role: RoleUser}

	assert.False(t, internal.ShouldSendToLLM())
	assert.True(t, user.ShouldSendToLLM())
}

func TestMessage_ToChatMessage_InternalRoleIsSanitized(t *testing.T) {
	msg := Message{
		Role:    RoleInternal,
		Content: []core.ContentBlock{core.NewTextBlock("secret internal context")},
	}

	chat := msg.ToChatMessage()
	require.Equal(t, string(RoleSystem), chat.Role)
	require.Len(t, chat.Content, 1)
	assert.Equal(t, "", chat.Content[0].Text)
}

func TestNewToolResultMessage_PreservesRawPayload(t *testing.T) {
	result := core.ToolResult{
		Data: map[string]any{"answer": 42},
		Metadata: map[string]interface{}{
			"latency_ms": 123,
		},
		Annotations: map[string]interface{}{
			"is_error": true,
			"source":   "unit-test",
		},
	}

	msg := NewToolResultMessage("call-1", "calculator", result)
	require.NotNil(t, msg.ToolResult)
	assert.True(t, msg.ToolResult.IsError)

	raw := msg.ToolResult.Raw
	require.NotNil(t, raw)
	assert.Equal(t, result.Data, raw["data"])
	assert.Equal(t, result.Metadata, raw["metadata"])
	assert.Equal(t, result.Annotations, raw["annotations"])
}

