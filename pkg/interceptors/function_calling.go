// Package interceptors provides middleware components for dspy-go modules and agents.
package interceptors

import "github.com/XiaoConstantine/dspy-go/pkg/tools"

// FunctionCallingConfig configures shared native function-calling execution.
// ToolRegistry is retained for configuration compatibility with callers that
// construct tool registries before passing the config to a higher-level agent.
type FunctionCallingConfig struct {
	ToolRegistry *tools.InMemoryToolRegistry

	// StrictMode requires the model to call a function instead of completing
	// with free-form text.
	StrictMode bool

	// IncludeFinishTool adds a Finish completion tool.
	IncludeFinishTool bool

	// FinishToolDescription customizes the Finish tool description.
	FinishToolDescription string
}

// DefaultFunctionCallingConfig returns defaults for native ReAct execution.
func DefaultFunctionCallingConfig() FunctionCallingConfig {
	return FunctionCallingConfig{
		StrictMode:            false,
		IncludeFinishTool:     true,
		FinishToolDescription: "Call this tool when you have completed the task and have the final answer. Pass the final answer as the 'answer' argument.",
	}
}
