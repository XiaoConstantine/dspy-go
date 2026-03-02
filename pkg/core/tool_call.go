package core

// ToolCall represents a tool invocation requested by an LLM.
// Used across packages: core.ChatMessage, agents.Message, tools.ParseToolCalls.
type ToolCall struct {
	ID        string         `json:"id"`        // provider-assigned ID; required for tool-result correlation
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}
