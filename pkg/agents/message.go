package agents

import (
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// MessageRole defines the role of a message in the conversation.
type MessageRole string

const (
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleSystem    MessageRole = "system"
	RoleTool      MessageRole = "tool"
	RoleInternal  MessageRole = "internal" // never sent to LLM
)

// Message represents a single message in the agent's conversation.
// This is the agent-level message type; use ToChatMessage() to convert
// to core.ChatMessage at the LLM boundary.
type Message struct {
	ID         string              `json:"id"`
	Role       MessageRole         `json:"role"`
	Content    []core.ContentBlock `json:"content"`
	ToolCalls  []core.ToolCall     `json:"tool_calls,omitempty"`
	ToolResult *MessageToolResult  `json:"tool_result,omitempty"`
	Metadata   map[string]any      `json:"metadata,omitempty"`
}

// MessageToolResult carries the result of a tool execution at the agent message level.
// Richer than core.ChatToolResult — includes raw result data for trajectory capture.
type MessageToolResult struct {
	ToolCallID string              `json:"tool_call_id"`
	Name       string              `json:"name"`
	Content    []core.ContentBlock `json:"content"`
	IsError    bool                `json:"is_error,omitempty"`
	Raw        map[string]any      `json:"raw,omitempty"`
}

// NewTextMessage creates a message with a single text content block.
func NewTextMessage(role MessageRole, text string) Message {
	return Message{
		Role:    role,
		Content: []core.ContentBlock{core.NewTextBlock(text)},
	}
}

// NewToolResultMessage creates a tool result message from a core.ToolResult.
func NewToolResultMessage(callID, toolName string, result core.ToolResult) Message {
	isError := false
	if result.Annotations != nil {
		if v, ok := result.Annotations["is_error"]; ok {
			if b, ok := v.(bool); ok {
				isError = b
			}
		}
	}

	content := formatToolResultContent(result)

	return Message{
		Role: RoleTool,
		ToolResult: &MessageToolResult{
			ToolCallID: callID,
			Name:       toolName,
			Content:    content,
			IsError:    isError,
			Raw:        buildToolResultRaw(result),
		},
	}
}

func buildToolResultRaw(result core.ToolResult) map[string]any {
	return map[string]any{
		"data":        result.Data,
		"metadata":    result.Metadata,
		"annotations": result.Annotations,
	}
}

// formatToolResultContent converts a core.ToolResult into content blocks.
func formatToolResultContent(result core.ToolResult) []core.ContentBlock {
	switch v := result.Data.(type) {
	case string:
		return []core.ContentBlock{core.NewTextBlock(v)}
	case []core.ContentBlock:
		return v
	default:
		return []core.ContentBlock{core.NewTextBlock(fmt.Sprintf("%v", v))}
	}
}

// ToChatMessage converts an agent-level Message to a core.ChatMessage
// for the LLM boundary. Strips metadata and normalizes internal-only messages.
func (m Message) ToChatMessage() core.ChatMessage {
	if !m.ShouldSendToLLM() {
		// Keep a valid role/content shape to avoid provider-side validation errors
		// if callers accidentally pass internal messages through this conversion.
		return core.ChatMessage{
			Role:    string(RoleSystem),
			Content: []core.ContentBlock{core.NewTextBlock("")},
		}
	}

	cm := core.ChatMessage{
		Role:      string(m.Role),
		Content:   m.Content,
		ToolCalls: m.ToolCalls,
	}
	if m.ToolResult != nil {
		cm.ToolResult = &core.ChatToolResult{
			ToolCallID: m.ToolResult.ToolCallID,
			Name:       m.ToolResult.Name,
			Content:    m.ToolResult.Content,
			IsError:    m.ToolResult.IsError,
		}
	}
	return cm
}

// ShouldSendToLLM reports whether this message should be included in provider-facing chat history.
func (m Message) ShouldSendToLLM() bool {
	return m.Role != RoleInternal
}

// TextContent returns the concatenated text content of all text blocks in the message.
func (m Message) TextContent() string {
	var parts []string
	for _, block := range m.Content {
		if block.Type == core.FieldTypeText {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}
