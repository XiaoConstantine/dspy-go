package core

import "context"

// ChatMessage represents a single message in a multi-turn conversation.
// Used by ToolCallingChatLLM for true native tool-use loops.
type ChatMessage struct {
	Role       string          `json:"role"`                  // "system"|"user"|"assistant"|"tool"
	Content    []ContentBlock  `json:"content"`               // reuse existing multimodal blocks from llm.go
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`  // for assistant messages requesting tool calls
	ToolResult *ChatToolResult `json:"tool_result,omitempty"` // for tool role messages
}

// ChatToolResult carries the result of a single tool execution in a chat message.
type ChatToolResult struct {
	ToolCallID string         `json:"tool_call_id"`
	Name       string         `json:"name"`
	Content    []ContentBlock `json:"content"`
	IsError    bool           `json:"is_error,omitempty"`
}

// ToolCallingChatLLM is an OPTIONAL interface for providers that support
// multi-turn conversations with native tool calling (OpenAI, Anthropic, Gemini).
// NativeStrategy checks for this at runtime via type assertion.
//
// Providers that don't implement this fall back to GenerateWithFunctions
// with a flattened prompt string (single-turn, best-effort compatibility).
type ToolCallingChatLLM interface {
	GenerateWithTools(ctx context.Context, messages []ChatMessage, tools []map[string]any, opts ...GenerateOption) (map[string]any, error)
}
