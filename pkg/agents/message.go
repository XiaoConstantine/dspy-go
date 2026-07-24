package agents

import (
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
	ID           string              `json:"id"`
	Role         MessageRole         `json:"role"`
	Content      []core.ContentBlock `json:"content"`
	ToolCalls    []core.ToolCall     `json:"tool_calls,omitempty"`
	ToolResult   *MessageToolResult  `json:"tool_result,omitempty"`
	Metadata     map[string]any      `json:"metadata,omitempty"`
	ProviderData map[string]any      `json:"provider_data,omitempty"`
}

// MessageToolResult carries the result of a tool execution at the agent message level.
// Content is the provider-visible result, while DisplayContent and Details retain richer
// operator-facing information without sending it back to the model.
type MessageToolResult struct {
	ToolCallID     string              `json:"tool_call_id"`
	Name           string              `json:"name"`
	Content        []core.ContentBlock `json:"content"`
	DisplayContent []core.ContentBlock `json:"display_content,omitempty"`
	Details        map[string]any      `json:"details,omitempty"`
	IsError        bool                `json:"is_error,omitempty"`
	Synthetic      bool                `json:"synthetic,omitempty"`
	Redacted       bool                `json:"redacted,omitempty"`
	Truncated      bool                `json:"truncated,omitempty"`
	// Raw preserves the legacy ToolResult envelope for trace and compatibility
	// consumers. New code should prefer the typed fields above.
	Raw map[string]any `json:"raw,omitempty"`
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
	observation := NormalizeToolResult(result)
	content := toolResultModelContent(result, observation.ModelText)
	display := toolResultDisplayContent(result, observation.DisplayText)

	return Message{
		Role: RoleTool,
		ToolResult: &MessageToolResult{
			ToolCallID:     callID,
			Name:           toolName,
			Content:        content,
			DisplayContent: display,
			Details:        cloneAnyMap(observation.Details),
			IsError:        observation.IsError,
			Synthetic:      observation.Synthetic,
			Redacted:       observation.Redacted,
			Truncated:      observation.Truncated,
			Raw:            buildToolResultRaw(result),
		},
	}
}

func buildToolResultRaw(result core.ToolResult) map[string]any {
	return map[string]any{
		"data":        cloneAnyValue(result.Data),
		"metadata":    cloneAnyMap(result.Metadata),
		"annotations": cloneAnyMap(result.Annotations),
	}
}

func toolResultModelContent(result core.ToolResult, modelText string) []core.ContentBlock {
	_, hasModelText := toolResultTextOverride(result.Metadata, core.ToolResultModelTextMeta)
	_, hasDisplayText := toolResultTextOverride(result.Metadata, core.ToolResultDisplayTextMeta)
	hasTextOverride := hasModelText || hasDisplayText
	if !hasTextOverride {
		if blocks, ok := result.Data.([]core.ContentBlock); ok {
			return cloneContentBlocks(blocks)
		}
	}
	return []core.ContentBlock{core.NewTextBlock(modelText)}
}

func toolResultDisplayContent(result core.ToolResult, displayText string) []core.ContentBlock {
	_, hasDisplayText := toolResultTextOverride(result.Metadata, core.ToolResultDisplayTextMeta)
	if !hasDisplayText {
		if blocks, ok := result.Data.([]core.ContentBlock); ok {
			return cloneContentBlocks(blocks)
		}
	}
	return []core.ContentBlock{core.NewTextBlock(displayText)}
}

// ToChatMessage converts an agent-level Message to a core.ChatMessage for the
// LLM boundary. Agent-only metadata and operator-facing tool details are not
// included. Opaque provider continuation data is carried forward without
// interpretation. Use MessagesToChatMessages when converting a
// transcript so internal messages are filtered rather than represented by placeholders.
func (m Message) ToChatMessage() core.ChatMessage {
	if !m.ShouldSendToLLM() {
		// Keep the legacy direct-call behavior valid for callers that convert one
		// message without checking ShouldSendToLLM. Transcript conversion filters
		// internal messages through MessagesToChatMessages.
		return core.ChatMessage{
			Role:    string(RoleSystem),
			Content: []core.ContentBlock{core.NewTextBlock("")},
		}
	}

	cm := core.ChatMessage{
		Role:      string(m.Role),
		Content:   cloneContentBlocks(m.Content),
		ToolCalls: cloneToolCalls(m.ToolCalls),
	}
	cm.ProviderData = cloneAnyMap(m.ProviderData)
	if m.ToolResult != nil {
		cm.ToolResult = &core.ChatToolResult{
			ToolCallID: m.ToolResult.ToolCallID,
			Name:       m.ToolResult.Name,
			Content:    cloneContentBlocks(m.ToolResult.Content),
			IsError:    m.ToolResult.IsError,
		}
	}
	return cm
}

// MessagesToChatMessages converts provider-visible transcript messages to
// core chat messages. Internal messages are omitted entirely.
func MessagesToChatMessages(messages []Message) []core.ChatMessage {
	if messages == nil {
		return nil
	}
	converted := make([]core.ChatMessage, 0, len(messages))
	for _, message := range messages {
		if !message.ShouldSendToLLM() {
			continue
		}
		converted = append(converted, message.ToChatMessage())
	}
	return converted
}

// MessageFromChatMessage converts a provider-neutral core chat message into an
// agent message. Provider-facing fields round-trip losslessly; agent-only
// metadata and operator-facing tool details do not exist on core.ChatMessage.
func MessageFromChatMessage(message core.ChatMessage) Message {
	converted := Message{
		Role:      MessageRole(message.Role),
		Content:   cloneContentBlocks(message.Content),
		ToolCalls: cloneToolCalls(message.ToolCalls),
	}
	converted.ProviderData = cloneAnyMap(message.ProviderData)
	if message.ToolResult != nil {
		converted.ToolResult = &MessageToolResult{
			ToolCallID:     message.ToolResult.ToolCallID,
			Name:           message.ToolResult.Name,
			Content:        cloneContentBlocks(message.ToolResult.Content),
			DisplayContent: cloneContentBlocks(message.ToolResult.Content),
			IsError:        message.ToolResult.IsError,
		}
	}
	return converted
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
