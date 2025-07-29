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

// InterceptableAgent extends Agent with interceptor support.
// This interface provides backward-compatible enhancement for agents that support interceptors.
type InterceptableAgent interface {
	Agent

	// ExecuteWithInterceptors runs the agent's task with interceptor support
	ExecuteWithInterceptors(ctx context.Context, input map[string]interface{}, interceptors []core.AgentInterceptor) (map[string]interface{}, error)

	// SetInterceptors sets the default interceptors for this agent instance
	SetInterceptors(interceptors []core.AgentInterceptor)

	// GetInterceptors returns the current interceptors for this agent
	GetInterceptors() []core.AgentInterceptor

	// ClearInterceptors removes all interceptors from this agent
	ClearInterceptors()

	// GetAgentID returns the unique identifier for this agent instance
	GetAgentID() string

	// GetAgentType returns the category/type of this agent
	GetAgentType() string
}

// InterceptorAgentAdapter wraps an existing Agent to provide interceptor support.
// This allows any existing agent to be used with interceptors without modifying its implementation.
type InterceptorAgentAdapter struct {
	agent        Agent
	interceptors []core.AgentInterceptor
	agentID      string
	agentType    string
}

// NewInterceptorAgentAdapter creates a new adapter that wraps an existing agent with interceptor support.
func NewInterceptorAgentAdapter(agent Agent, agentID, agentType string) *InterceptorAgentAdapter {
	return &InterceptorAgentAdapter{
		agent:        agent,
		interceptors: make([]core.AgentInterceptor, 0),
		agentID:      agentID,
		agentType:    agentType,
	}
}

// Execute implements the basic Agent interface by calling the wrapped agent.
func (iaa *InterceptorAgentAdapter) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return iaa.agent.Execute(ctx, input)
}

// GetCapabilities returns the tools/capabilities from the wrapped agent.
func (iaa *InterceptorAgentAdapter) GetCapabilities() []core.Tool {
	return iaa.agent.GetCapabilities()
}

// GetMemory returns the memory store from the wrapped agent.
func (iaa *InterceptorAgentAdapter) GetMemory() Memory {
	return iaa.agent.GetMemory()
}

// ExecuteWithInterceptors runs the agent's task with interceptor support.
func (iaa *InterceptorAgentAdapter) ExecuteWithInterceptors(ctx context.Context, input map[string]interface{}, interceptors []core.AgentInterceptor) (map[string]interface{}, error) {
	// Use provided interceptors, or fall back to adapter's default interceptors
	if interceptors == nil {
		interceptors = iaa.interceptors
	}

	// Create agent info for interceptors
	info := core.NewAgentInfo(iaa.agentID, iaa.agentType, iaa.agent.GetCapabilities())

	// Create the base handler that calls the wrapped agent's Execute method
	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		return iaa.agent.Execute(ctx, input)
	}

	// Chain the interceptors
	chainedInterceptor := core.ChainAgentInterceptors(interceptors...)

	// Execute with interceptors
	return chainedInterceptor(ctx, input, info, handler)
}

// SetInterceptors sets the default interceptors for this adapter.
func (iaa *InterceptorAgentAdapter) SetInterceptors(interceptors []core.AgentInterceptor) {
	iaa.interceptors = make([]core.AgentInterceptor, len(interceptors))
	copy(iaa.interceptors, interceptors)
}

// GetInterceptors returns the current interceptors for this adapter.
func (iaa *InterceptorAgentAdapter) GetInterceptors() []core.AgentInterceptor {
	result := make([]core.AgentInterceptor, len(iaa.interceptors))
	copy(result, iaa.interceptors)
	return result
}

// ClearInterceptors removes all interceptors from this adapter.
func (iaa *InterceptorAgentAdapter) ClearInterceptors() {
	iaa.interceptors = nil
}

// GetAgentID returns the unique identifier for this agent instance.
func (iaa *InterceptorAgentAdapter) GetAgentID() string {
	return iaa.agentID
}

// GetAgentType returns the category/type of this agent.
func (iaa *InterceptorAgentAdapter) GetAgentType() string {
	return iaa.agentType
}

// WrapAgentWithInterceptors is a convenience function to wrap any agent with interceptor support.
func WrapAgentWithInterceptors(agent Agent, agentID, agentType string, interceptors ...core.AgentInterceptor) InterceptableAgent {
	adapter := NewInterceptorAgentAdapter(agent, agentID, agentType)
	if len(interceptors) > 0 {
		adapter.SetInterceptors(interceptors)
	}
	return adapter
}
