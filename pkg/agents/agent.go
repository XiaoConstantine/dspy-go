package agents

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type Agent interface {
	// Execute runs the agent's task with given input and returns output
	Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

	// GetCapabilities returns the tools/capabilities available to this agent
	GetCapabilities() []core.Tool

	// GetMemory returns the agent's memory store
	GetMemory() Memory
}
