package agents

import "context"

type Agent interface {
	// Execute runs the agent's task with given input and returns output
	Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

	// GetCapabilities returns the tools/capabilities available to this agent
	GetCapabilities() []Tool

	// GetMemory returns the agent's memory store
	GetMemory() Memory
}

// Tool represents a capability available to the agent.
type Tool interface {
	// Name returns the tool's identifier
	Name() string

	// Description returns documentation for the tool
	Description() string

	// Execute runs the tool with given parameters
	Execute(ctx context.Context, params map[string]interface{}) (interface{}, error)

	// ValidateParams checks if parameters are valid for this tool
	ValidateParams(params map[string]interface{}) error
}
