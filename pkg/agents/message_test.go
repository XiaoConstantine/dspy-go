package agents

import (
	"reflect"
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

func TestMessagesToChatMessages_FiltersInternalMessages(t *testing.T) {
	messages := []Message{
		NewTextMessage(RoleSystem, "system"),
		NewTextMessage(RoleInternal, "secret"),
		NewTextMessage(RoleUser, "hello"),
	}

	chat := MessagesToChatMessages(messages)
	require.Len(t, chat, 2)
	assert.Equal(t, "system", chat[0].Content[0].Text)
	assert.Equal(t, "hello", chat[1].Content[0].Text)
}

func TestNewToolResultMessage_PreservesTypedAndRawPayload(t *testing.T) {
	result := core.ToolResult{
		Data: map[string]any{"answer": 42},
		Metadata: map[string]any{
			"latency_ms":                   123,
			core.ToolResultModelTextMeta:   "model summary",
			core.ToolResultDisplayTextMeta: "operator detail",
			core.ToolResultSyntheticMeta:   true,
			core.ToolResultRedactedMeta:    true,
			core.ToolResultTruncatedMeta:   true,
		},
		Annotations: map[string]any{
			"is_error": true,
			"source":   "unit-test",
			core.ToolResultDetailsAnnotation: map[string]any{
				"command": "go test",
			},
		},
	}

	msg := NewToolResultMessage("call-1", "calculator", result)
	require.NotNil(t, msg.ToolResult)
	assert.Equal(t, "call-1", msg.ToolResult.ToolCallID)
	assert.Equal(t, "calculator", msg.ToolResult.Name)
	assert.Equal(t, "model summary", msg.ToolResult.Content[0].Text)
	assert.Equal(t, "operator detail", msg.ToolResult.DisplayContent[0].Text)
	assert.Equal(t, map[string]any{"command": "go test"}, msg.ToolResult.Details)
	assert.True(t, msg.ToolResult.IsError)
	assert.True(t, msg.ToolResult.Synthetic)
	assert.True(t, msg.ToolResult.Redacted)
	assert.True(t, msg.ToolResult.Truncated)

	raw := msg.ToolResult.Raw
	require.NotNil(t, raw)
	assert.Equal(t, result.Data, raw["data"])
	assert.Equal(t, result.Metadata, raw["metadata"])
	assert.Equal(t, result.Annotations, raw["annotations"])
}

func TestNewToolResultMessage_PreservesMultimodalModelContent(t *testing.T) {
	image := core.NewImageBlock([]byte{1, 2, 3}, "image/png")
	image.Metadata = map[string]any{"source": "tool"}
	result := core.ToolResult{Data: []core.ContentBlock{core.NewTextBlock("caption"), image}}

	msg := NewToolResultMessage("call-1", "screenshot", result)
	require.NotNil(t, msg.ToolResult)
	require.Len(t, msg.ToolResult.Content, 2)
	assert.Equal(t, "caption", msg.ToolResult.Content[0].Text)
	assert.Equal(t, []byte{1, 2, 3}, msg.ToolResult.Content[1].Data)
	assert.Equal(t, "image/png", msg.ToolResult.Content[1].MimeType)
	assert.Equal(t, msg.ToolResult.Content, msg.ToolResult.DisplayContent)
}

func TestNewToolResultMessage_DisplayOverrideHidesMultimodalContent(t *testing.T) {
	msg := NewToolResultMessage("call-1", "screenshot", core.ToolResult{
		Data: []core.ContentBlock{core.NewImageBlock([]byte{1, 2, 3}, "image/png")},
		Metadata: map[string]any{
			core.ToolResultDisplayTextMeta: "[redacted image]",
			core.ToolResultRedactedMeta:    true,
		},
	})

	require.NotNil(t, msg.ToolResult)
	require.Len(t, msg.ToolResult.DisplayContent, 1)
	assert.Equal(t, "[redacted image]", msg.ToolResult.DisplayContent[0].Text)
	assert.Nil(t, msg.ToolResult.DisplayContent[0].Data)
	assert.True(t, msg.ToolResult.Redacted)
}

func TestNewToolResultMessage_EmptyOverridesDoNotExposeRawData(t *testing.T) {
	msg := NewToolResultMessage("call-1", "secret", core.ToolResult{
		Data: "sensitive raw output",
		Metadata: map[string]any{
			core.ToolResultModelTextMeta:   "",
			core.ToolResultDisplayTextMeta: "",
			core.ToolResultRedactedMeta:    true,
		},
	})

	require.NotNil(t, msg.ToolResult)
	require.Len(t, msg.ToolResult.Content, 1)
	require.Len(t, msg.ToolResult.DisplayContent, 1)
	assert.Empty(t, msg.ToolResult.Content[0].Text)
	assert.Empty(t, msg.ToolResult.DisplayContent[0].Text)
	assert.NotContains(t, msg.ToolResult.Content[0].Text, "sensitive")
	assert.NotContains(t, msg.ToolResult.DisplayContent[0].Text, "sensitive")
	assert.True(t, msg.ToolResult.Redacted)
}

func TestNewToolResultMessage_UsesDisplayTextAsModelFallback(t *testing.T) {
	msg := NewToolResultMessage("call-1", "read", core.ToolResult{
		Data: "raw output",
		Metadata: map[string]any{
			core.ToolResultDisplayTextMeta: "safe display output",
		},
	})

	require.NotNil(t, msg.ToolResult)
	assert.Equal(t, "safe display output", msg.ToolResult.Content[0].Text)
	assert.Equal(t, "safe display output", msg.ToolResult.DisplayContent[0].Text)
}

func TestCloneMessages_PreservesNilAndClonesElements(t *testing.T) {
	assert.Nil(t, CloneMessages(nil))

	messages := []Message{NewTextMessage(RoleUser, "original")}
	cloned := CloneMessages(messages)
	cloned[0].Content[0].Text = "changed"

	assert.Equal(t, "original", messages[0].Content[0].Text)
}

func TestMessage_CloneDoesNotShareMutableState(t *testing.T) {
	message := Message{
		ID:   "message-1",
		Role: RoleAssistant,
		Content: []core.ContentBlock{{
			Type:     core.FieldTypeImage,
			Data:     []byte{1, 2, 3},
			Metadata: map[string]any{"nested": map[string]any{"value": "original"}},
		}},
		ToolCalls: []core.ToolCall{{
			ID:        "call-1",
			Name:      "read",
			Arguments: map[string]any{"paths": []any{"one", "two"}},
			Metadata:  map[string]any{"signature": []byte{4, 5}},
		}},
		ToolResult: &MessageToolResult{
			ToolCallID:     "call-0",
			Name:           "prior",
			Content:        []core.ContentBlock{core.NewTextBlock("model")},
			DisplayContent: []core.ContentBlock{core.NewTextBlock("display")},
			Details:        map[string]any{"nested": map[string]any{"value": "original"}},
			Raw:            map[string]any{"bytes": []byte{6, 7}},
		},
		Metadata: map[string]any{
			"labels": []any{"one", map[string]any{"two": 2}},
			"typed":  map[string][]string{"items": {"one", "two"}},
		},
	}

	cloned := message.Clone()
	cloned.Content[0].Data[0] = 9
	cloned.Content[0].Metadata["nested"].(map[string]any)["value"] = "changed"
	cloned.ToolCalls[0].Arguments["paths"].([]any)[0] = "changed"
	cloned.ToolCalls[0].Metadata["signature"].([]byte)[0] = 9
	cloned.ToolResult.Details["nested"].(map[string]any)["value"] = "changed"
	cloned.ToolResult.Raw["bytes"].([]byte)[0] = 9
	cloned.Metadata["labels"].([]any)[1].(map[string]any)["two"] = 3
	cloned.Metadata["typed"].(map[string][]string)["items"][0] = "changed"

	assert.Equal(t, byte(1), message.Content[0].Data[0])
	assert.Equal(t, "original", message.Content[0].Metadata["nested"].(map[string]any)["value"])
	assert.Equal(t, "one", message.ToolCalls[0].Arguments["paths"].([]any)[0])
	assert.Equal(t, byte(4), message.ToolCalls[0].Metadata["signature"].([]byte)[0])
	assert.Equal(t, "original", message.ToolResult.Details["nested"].(map[string]any)["value"])
	assert.Equal(t, byte(6), message.ToolResult.Raw["bytes"].([]byte)[0])
	assert.Equal(t, 2, message.Metadata["labels"].([]any)[1].(map[string]any)["two"])
	assert.Equal(t, "one", message.Metadata["typed"].(map[string][]string)["items"][0])
}

func TestMessage_ClonePreservesOverlappingSliceShapes(t *testing.T) {
	base := []any{"one", map[string]any{"value": "two"}, "three"}
	message := Message{Metadata: map[string]any{
		"short": base[:1],
		"long":  base[:2],
	}}

	cloned := message.Clone()
	short := cloned.Metadata["short"].([]any)
	long := cloned.Metadata["long"].([]any)
	require.Len(t, short, 1)
	require.Len(t, long, 2)
	assert.Equal(t, []any{"one"}, short)
	assert.Equal(t, "two", long[1].(map[string]any)["value"])

	shortWithTail := short[:cap(short)]
	require.Len(t, shortWithTail, 3)
	assert.Equal(t, "two", shortWithTail[1].(map[string]any)["value"])
	assert.Equal(t, "three", shortWithTail[2])
	shortWithTail[1].(map[string]any)["value"] = "changed"
	assert.Equal(t, "two", base[1].(map[string]any)["value"])
}

func TestMessage_CloneHandlesNilMessageValueClone(t *testing.T) {
	message := Message{Metadata: map[string]any{
		"value": nilMessageValueCloner{Mutable: []string{"original"}},
	}}

	cloned := message.Clone()
	assert.Nil(t, cloned.Metadata["value"])
	assert.Equal(t, []string{"original"}, message.Metadata["value"].(nilMessageValueCloner).Mutable)
}

func TestMessage_ClonePreservesCyclicMetadataWithoutSharing(t *testing.T) {
	cyclicMap := map[string]any{}
	cyclicMap["self"] = cyclicMap
	cyclicSlice := make([]any, 1)
	cyclicSlice[0] = cyclicSlice

	message := Message{Metadata: map[string]any{
		"map":   cyclicMap,
		"slice": cyclicSlice,
	}}
	cloned := message.Clone()

	clonedMap := cloned.Metadata["map"].(map[string]any)
	clonedSelf := clonedMap["self"].(map[string]any)
	assert.Equal(t, reflect.ValueOf(clonedMap).Pointer(), reflect.ValueOf(clonedSelf).Pointer())
	assert.NotEqual(t, reflect.ValueOf(cyclicMap).Pointer(), reflect.ValueOf(clonedMap).Pointer())
	clonedSelf["changed"] = true
	assert.NotContains(t, cyclicMap, "changed")

	clonedSlice := cloned.Metadata["slice"].([]any)
	clonedSliceSelf := clonedSlice[0].([]any)
	assert.Equal(t, reflect.ValueOf(clonedSlice).Pointer(), reflect.ValueOf(clonedSliceSelf).Pointer())
	assert.NotEqual(t, reflect.ValueOf(cyclicSlice).Pointer(), reflect.ValueOf(clonedSlice).Pointer())
	clonedSliceSelf[0] = "changed"
	_, originalStillCycles := cyclicSlice[0].([]any)
	assert.True(t, originalStillCycles)
}

func TestMessage_ClonePreservesNonNilEmptyBinaryContent(t *testing.T) {
	message := Message{Content: []core.ContentBlock{{
		Type: core.FieldTypeImage,
		Data: make([]byte, 0),
	}}}

	cloned := message.Clone()
	require.NotNil(t, cloned.Content[0].Data)
	assert.Empty(t, cloned.Content[0].Data)

	chat := message.ToChatMessage()
	require.NotNil(t, chat.Content[0].Data)
	assert.Empty(t, chat.Content[0].Data)
}

func TestMessage_CoreChatRoundTripPreservesProviderFields(t *testing.T) {
	message := Message{
		Role: RoleAssistant,
		Content: []core.ContentBlock{
			core.NewTextBlock("I will inspect the file."),
			{
				Type:     core.FieldTypeImage,
				Data:     []byte{1, 2},
				MimeType: "image/png",
				Metadata: map[string]any{"source": "assistant"},
			},
		},
		ToolCalls: []core.ToolCall{{
			ID:        "call-1",
			Name:      "read",
			Arguments: map[string]any{"path": "main.go", "range": map[string]any{"start": 1}},
			Metadata:  map[string]any{"signature": "sig"},
		}},
	}

	roundTrip := MessageFromChatMessage(message.ToChatMessage())
	assert.Equal(t, message.Role, roundTrip.Role)
	assert.Equal(t, message.Content, roundTrip.Content)
	assert.Equal(t, message.ToolCalls, roundTrip.ToolCalls)
}

type nilMessageValueCloner struct {
	Mutable []string
}

func (nilMessageValueCloner) CloneMessageValue() any {
	return nil
}

func TestMessage_CoreChatToolResultRoundTripPreservesProviderFields(t *testing.T) {
	message := Message{
		Role: RoleTool,
		ToolResult: &MessageToolResult{
			ToolCallID:     "call-1",
			Name:           "read",
			Content:        []core.ContentBlock{core.NewTextBlock("contents")},
			DisplayContent: []core.ContentBlock{core.NewTextBlock("full contents")},
			Details:        map[string]any{"path": "main.go"},
			IsError:        true,
		},
	}

	roundTrip := MessageFromChatMessage(message.ToChatMessage())
	require.NotNil(t, roundTrip.ToolResult)
	assert.Equal(t, message.ToolResult.ToolCallID, roundTrip.ToolResult.ToolCallID)
	assert.Equal(t, message.ToolResult.Name, roundTrip.ToolResult.Name)
	assert.Equal(t, message.ToolResult.Content, roundTrip.ToolResult.Content)
	assert.Equal(t, message.ToolResult.IsError, roundTrip.ToolResult.IsError)
	assert.Equal(t, message.ToolResult.Content, roundTrip.ToolResult.DisplayContent)
	assert.Nil(t, roundTrip.ToolResult.Details)
}
