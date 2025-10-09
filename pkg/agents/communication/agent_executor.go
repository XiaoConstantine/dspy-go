package communication

import (
	"context"
	"fmt"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ============================================================================
// A2AExecutor - In-process Agent Composition via a2a Protocol
// ============================================================================

// A2AExecutor wraps a dspy-go agent and enables a2a protocol communication.
// It supports sub-agents for hierarchical agent composition without HTTP.
//
// Similar to ADK Python's LLMAgent with sub_agents parameter:
//   parent = LLMAgent(model=llm, sub_agents=[agent1, agent2])
//
// Usage:
//   executor := a2a.NewExecutor(myAgent).
//       WithSubAgent("search", searchExecutor).
//       WithSubAgent("reasoning", reasoningExecutor)
//
//   result := executor.SendMessage(ctx, message)
type A2AExecutor struct {
	agent     agents.Agent
	subAgents map[string]*A2AExecutor
	name      string
	mu        sync.RWMutex
}

// ExecutorConfig holds configuration for the executor.
type ExecutorConfig struct {
	Name string // Agent name for identification
}

// ============================================================================
// Executor Creation
// ============================================================================

// NewExecutor creates a new a2a executor wrapping the given agent.
func NewExecutor(agent agents.Agent) *A2AExecutor {
	return NewExecutorWithConfig(agent, ExecutorConfig{
		Name: "agent",
	})
}

// NewExecutorWithConfig creates a new executor with custom configuration.
func NewExecutorWithConfig(agent agents.Agent, config ExecutorConfig) *A2AExecutor {
	if config.Name == "" {
		config.Name = "agent"
	}

	return &A2AExecutor{
		agent:     agent,
		subAgents: make(map[string]*A2AExecutor),
		name:      config.Name,
	}
}

// ============================================================================
// Sub-Agent Management
// ============================================================================

// WithSubAgent registers a sub-agent that can be called by this agent.
// The sub-agent becomes available as a capability.
func (e *A2AExecutor) WithSubAgent(name string, subAgent *A2AExecutor) *A2AExecutor {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.subAgents[name] = subAgent
	return e
}

// GetSubAgent retrieves a sub-agent by name.
func (e *A2AExecutor) GetSubAgent(name string) (*A2AExecutor, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	agent, ok := e.subAgents[name]
	return agent, ok
}

// ListSubAgents returns all registered sub-agent names.
func (e *A2AExecutor) ListSubAgents() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	names := make([]string, 0, len(e.subAgents))
	for name := range e.subAgents {
		names = append(names, name)
	}
	return names
}

// ============================================================================
// Message Execution (Main Entry Point)
// ============================================================================

// SendMessage executes the agent with an a2a message and returns a task.
// This is the primary way to interact with an a2a-wrapped agent.
//
// The message is converted to agent input, the agent executes, and the
// output is converted to an a2a artifact.
func (e *A2AExecutor) SendMessage(ctx context.Context, msg *Message) (*Task, error) {
	if msg == nil {
		return nil, fmt.Errorf("message cannot be nil")
	}

	// Create task to track execution
	task := NewTask()
	task.ContextID = msg.ContextID

	// Update to working state
	task.UpdateStatus(TaskStateWorking)

	// Convert a2a message to agent input
	input, err := MessageToAgentInput(msg)
	if err != nil {
		return e.failTask(task, err), nil
	}

	// Execute the wrapped agent
	output, err := e.agent.Execute(ctx, input)
	if err != nil {
		return e.failTask(task, err), nil
	}

	// Convert agent output to a2a artifact
	artifact, err := AgentOutputToArtifact(output)
	if err != nil {
		return e.failTask(task, err), nil
	}

	// Complete task with artifact
	task.AddArtifact(artifact)
	task.UpdateStatus(TaskStateCompleted)

	return task, nil
}

// Execute provides a simpler interface that returns just the artifact.
// This is useful when you don't need the full task tracking.
func (e *A2AExecutor) Execute(ctx context.Context, msg *Message) (Artifact, error) {
	task, err := e.SendMessage(ctx, msg)
	if err != nil {
		return Artifact{}, err
	}

	if task.Status.State == TaskStateFailed {
		if task.Status.Message != nil {
			return Artifact{}, fmt.Errorf("task failed: %s", ExtractTextFromMessage(task.Status.Message))
		}
		return Artifact{}, fmt.Errorf("task failed")
	}

	if len(task.Artifacts) == 0 {
		return Artifact{}, fmt.Errorf("no artifacts produced")
	}

	return task.Artifacts[0], nil
}

// ============================================================================
// Sub-Agent Calling (Internal Communication)
// ============================================================================

// CallSubAgent calls a registered sub-agent with an a2a message.
// This enables hierarchical agent composition:
//
//   parent.CallSubAgent("search", userMessage)
//   parent.CallSubAgent("reasoning", searchResults)
//
// The communication uses a2a protocol but happens in-process (no HTTP).
func (e *A2AExecutor) CallSubAgent(ctx context.Context, name string, msg *Message) (Artifact, error) {
	subAgent, ok := e.GetSubAgent(name)
	if !ok {
		return Artifact{}, fmt.Errorf("sub-agent '%s' not found", name)
	}

	// Execute sub-agent via a2a protocol
	return subAgent.Execute(ctx, msg)
}

// CallSubAgentSimple is a convenience method that creates a simple text message
// and calls the sub-agent.
func (e *A2AExecutor) CallSubAgentSimple(ctx context.Context, name string, text string) (string, error) {
	msg := NewUserMessage(text)

	artifact, err := e.CallSubAgent(ctx, name, msg)
	if err != nil {
		return "", err
	}

	return ExtractTextFromArtifact(artifact), nil
}

// ============================================================================
// Agent Interface Methods
// ============================================================================

// GetCapabilities returns the capabilities of this executor.
// This includes both the wrapped agent's tools and registered sub-agents.
func (e *A2AExecutor) GetCapabilities() []Capability {
	capabilities := []Capability{}

	// Add wrapped agent's tools as capabilities
	if tools := e.agent.GetCapabilities(); len(tools) > 0 {
		capabilities = append(capabilities, ToolsToCapabilities(tools)...)
	}

	// Add sub-agents as capabilities
	e.mu.RLock()
	defer e.mu.RUnlock()

	for name, subAgent := range e.subAgents {
		cap := Capability{
			Name:        name,
			Description: fmt.Sprintf("Sub-agent: %s", subAgent.name),
			Type:        "agent", // Special type for sub-agents
		}
		capabilities = append(capabilities, cap)
	}

	return capabilities
}

// GetAgentCard returns an AgentCard describing this executor.
// Useful for discovery and introspection.
func (e *A2AExecutor) GetAgentCard() AgentCard {
	return AgentCard{
		Name:         e.name,
		Description:  fmt.Sprintf("A2A-wrapped agent: %s", e.name),
		Version:      "1.0.0",
		Capabilities: e.GetCapabilities(),
	}
}

// ============================================================================
// Helper Methods
// ============================================================================

// failTask marks a task as failed with an error message.
func (e *A2AExecutor) failTask(task *Task, err error) *Task {
	task.UpdateStatus(TaskStateFailed)
	task.Status.Message = CreateErrorMessage(err)
	return task
}

// Name returns the executor's name.
func (e *A2AExecutor) Name() string {
	return e.name
}

// UnwrapAgent returns the underlying wrapped agent.
// Useful for accessing agent-specific functionality.
func (e *A2AExecutor) UnwrapAgent() agents.Agent {
	return e.agent
}

// ============================================================================
// Convenience Constructors
// ============================================================================

// NewExecutorWithSubAgents creates an executor with multiple sub-agents at once.
//
// Example:
//   executor := a2a.NewExecutorWithSubAgents(parent, map[string]*A2AExecutor{
//       "search": searchExecutor,
//       "reasoning": reasoningExecutor,
//   })
func NewExecutorWithSubAgents(agent agents.Agent, subAgents map[string]*A2AExecutor) *A2AExecutor {
	executor := NewExecutor(agent)
	for name, subAgent := range subAgents {
		executor.WithSubAgent(name, subAgent)
	}
	return executor
}
