package communication

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	nativeagent "github.com/XiaoConstantine/dspy-go/pkg/agents/native"
	reactagent "github.com/XiaoConstantine/dspy-go/pkg/agents/react"
	rlmagent "github.com/XiaoConstantine/dspy-go/pkg/agents/rlm"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// Mock Agents for Testing
// ============================================================================

// mockSimpleAgent is a simple agent implementation for testing.
type mockSimpleAgent struct {
	name     string
	response string
}

func (m *mockSimpleAgent) Execute(ctx context.Context, input map[string]any) (map[string]any, error) {
	question, _ := input["question"].(string)
	return map[string]any{
		"answer": m.response + " (responding to: " + question + ")",
	}, nil
}

func (m *mockSimpleAgent) GetCapabilities() []core.Tool {
	return nil
}

func (m *mockSimpleAgent) GetMemory() agents.Memory {
	return nil
}

type fixedOutputAgent struct {
	output map[string]any
	trace  *agents.ExecutionTrace
}

func (a *fixedOutputAgent) Execute(context.Context, map[string]any) (map[string]any, error) {
	return a.output, nil
}
func (*fixedOutputAgent) GetCapabilities() []core.Tool                 { return nil }
func (*fixedOutputAgent) GetMemory() agents.Memory                     { return nil }
func (a *fixedOutputAgent) LastExecutionTrace() *agents.ExecutionTrace { return a.trace }
func (a *fixedOutputAgent) ExecuteWithTrace(context.Context, map[string]any) (agents.AgentExecutionResult, error) {
	return agents.AgentExecutionResult{Output: a.output, Trace: a.trace}, nil
}

// ============================================================================
// Basic Executor Tests
// ============================================================================

func TestNewExecutor(t *testing.T) {
	agent := &mockSimpleAgent{name: "test", response: "hello"}
	executor := NewExecutor(agent)

	if executor == nil {
		t.Fatal("expected non-nil executor")
	}
	if executor.agent != agent {
		t.Error("executor should wrap the provided agent")
	}
	if len(executor.subAgents) != 0 {
		t.Error("new executor should have no sub-agents")
	}
}

func TestExecutorUsesBuiltInAgentInputContracts(t *testing.T) {
	t.Run("native task", func(t *testing.T) {
		var agent *nativeagent.Agent
		input, err := NewExecutor(agent).messageToAgentInput(NewUserMessage("inspect repository"))
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"task": "inspect repository"}, input)
		assert.NotContains(t, input, "question")
	})

	t.Run("ReAct task", func(t *testing.T) {
		agent := reactagent.NewReActAgent("react-1", "react")
		input, err := NewExecutor(agent).messageToAgentInput(NewUserMessage("plan work"))
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"task": "plan work"}, input)
		assert.NotContains(t, input, "question")
	})

	t.Run("RLM context and query", func(t *testing.T) {
		agent := rlmagent.NewAgent("rlm-1", nil)
		msg := NewMessage(RoleUser,
			NewDataPart(map[string]any{"context": map[string]any{"document": "facts"}}),
			NewTextPartWithMetadata("what matters?", map[string]any{"field": "query"}),
		)
		input, err := NewExecutor(agent).messageToAgentInput(msg)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"document": "facts"}, input["context"])
		assert.Equal(t, "what matters?", input["query"])
		assert.NotContains(t, input, "question")

		_, err = NewExecutor(agent).messageToAgentInput(NewMessage(RoleUser,
			NewDataPart(map[string]any{"context": "facts", "qurey": "typo"}),
			NewTextPartWithMetadata("what matters?", map[string]any{"field": "query"}),
		))
		require.ErrorContains(t, err, `data field "qurey" is not accepted`)
	})
}

func TestExecutorDoesNotCompleteExplicitlyIncompleteOutput(t *testing.T) {
	agent := &fixedOutputAgent{output: map[string]any{
		"completed": false,
		"error":     "maximum turns reached",
	}}
	task, err := NewExecutor(agent).SendMessage(context.Background(), NewUserMessage("work"))
	require.NoError(t, err)
	assert.Equal(t, TaskStateFailed, task.Status.State)
	require.NotNil(t, task.Status.Message)
	assert.Contains(t, ExtractTextFromMessage(task.Status.Message), "maximum turns reached")
}

func TestExecutorUsesExecutionScopedTraceStatus(t *testing.T) {
	agent := &fixedOutputAgent{
		output: map[string]any{"answer": "partial"},
		trace:  &agents.ExecutionTrace{Status: agents.TraceStatusPartial},
	}
	task, err := NewExecutor(agent).SendMessage(context.Background(), NewUserMessage("work"))
	require.NoError(t, err)
	assert.Equal(t, TaskStateFailed, task.Status.State)

	agent.trace = &agents.ExecutionTrace{Status: agents.TraceStatusSuccess}
	task, err = NewExecutor(agent).SendMessage(context.Background(), NewUserMessage("work"))
	require.NoError(t, err)
	assert.Equal(t, TaskStateCompleted, task.Status.State)
}

func TestExecutorSendMessage(t *testing.T) {
	agent := &mockSimpleAgent{name: "test", response: "Hello World"}
	executor := NewExecutor(agent)

	msg := NewUserMessage("What is 2+2?")
	task, err := executor.SendMessage(context.Background(), msg)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if task.Status.State != TaskStateCompleted {
		t.Errorf("expected completed state, got %s", task.Status.State)
	}
	if len(task.Artifacts) != 1 {
		t.Fatalf("expected 1 artifact, got %d", len(task.Artifacts))
	}

	// Verify artifact content
	text := ExtractTextFromArtifact(task.Artifacts[0])
	if text == "" {
		t.Error("expected non-empty artifact text")
	}
}

func TestExecutorExecute(t *testing.T) {
	agent := &mockSimpleAgent{name: "test", response: "42"}
	executor := NewExecutor(agent)

	msg := NewUserMessage("What is the meaning of life?")
	artifact, err := executor.Execute(context.Background(), msg)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text := ExtractTextFromArtifact(artifact)
	if !contains(text, "42") {
		t.Errorf("expected response to contain '42', got: %s", text)
	}
}

// ============================================================================
// Sub-Agent Composition Tests
// ============================================================================

func TestWithSubAgent(t *testing.T) {
	parent := &mockSimpleAgent{name: "parent", response: "I am parent"}
	child := &mockSimpleAgent{name: "child", response: "I am child"}

	parentExec := NewExecutor(parent)
	childExec := NewExecutor(child)

	// Register sub-agent
	parentExec.WithSubAgent("helper", childExec)

	// Verify sub-agent registered
	subAgent, ok := parentExec.GetSubAgent("helper")
	if !ok {
		t.Fatal("sub-agent 'helper' should be registered")
	}
	if subAgent != childExec {
		t.Error("retrieved sub-agent should match registered one")
	}

	// Verify listing
	names := parentExec.ListSubAgents()
	if len(names) != 1 {
		t.Errorf("expected 1 sub-agent, got %d", len(names))
	}
	if names[0] != "helper" {
		t.Errorf("expected sub-agent name 'helper', got '%s'", names[0])
	}
}

func TestCallSubAgent(t *testing.T) {
	// Create search agent
	searchAgent := &mockSimpleAgent{
		name:     "search",
		response: "Paris is the capital of France",
	}
	searchExec := NewExecutorWithConfig(searchAgent, ExecutorConfig{Name: "SearchAgent"})

	// Create parent agent that uses search
	parentAgent := &mockSimpleAgent{
		name:     "parent",
		response: "Based on search results",
	}
	parentExec := NewExecutorWithConfig(parentAgent, ExecutorConfig{Name: "ParentAgent"}).
		WithSubAgent("search", searchExec)

	// Parent calls search sub-agent
	msg := NewUserMessage("What is the capital of France?")
	artifact, err := parentExec.CallSubAgent(context.Background(), "search", msg)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text := ExtractTextFromArtifact(artifact)
	if !contains(text, "Paris") {
		t.Errorf("expected search result to contain 'Paris', got: %s", text)
	}
}

func TestCallSubAgentSimple(t *testing.T) {
	reasoning := &mockSimpleAgent{
		name:     "reasoning",
		response: "Step 1: Think. Step 2: Answer.",
	}
	reasoningExec := NewExecutor(reasoning)

	parent := &mockSimpleAgent{name: "parent", response: "ok"}
	parentExec := NewExecutor(parent).WithSubAgent("reasoning", reasoningExec)

	// Simple text call
	result, err := parentExec.CallSubAgentSimple(context.Background(), "reasoning", "Explain quantum physics")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !contains(result, "Step 1") {
		t.Errorf("expected reasoning steps, got: %s", result)
	}
}

func TestCallSubAgent_NotFound(t *testing.T) {
	parent := &mockSimpleAgent{name: "parent", response: "ok"}
	parentExec := NewExecutor(parent)

	msg := NewUserMessage("test")
	_, err := parentExec.CallSubAgent(context.Background(), "nonexistent", msg)

	if err == nil {
		t.Error("expected error for non-existent sub-agent")
	}
	if !contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' error, got: %v", err)
	}
}

// ============================================================================
// Multi-Level Agent Hierarchy Test
// ============================================================================

func TestMultiLevelAgentHierarchy(t *testing.T) {
	// Level 3: Leaf agents
	calculatorAgent := &mockSimpleAgent{
		name:     "calculator",
		response: "4",
	}
	calculatorExec := NewExecutorWithConfig(calculatorAgent, ExecutorConfig{Name: "Calculator"})

	searchAgent := &mockSimpleAgent{
		name:     "search",
		response: "Found on Wikipedia",
	}
	searchExec := NewExecutorWithConfig(searchAgent, ExecutorConfig{Name: "Search"})

	// Level 2: Tool agent that has calculator and search
	toolAgent := &mockSimpleAgent{
		name:     "tool_agent",
		response: "Using tools to answer",
	}
	toolExec := NewExecutorWithConfig(toolAgent, ExecutorConfig{Name: "ToolAgent"}).
		WithSubAgent("calculator", calculatorExec).
		WithSubAgent("search", searchExec)

	// Level 1: Orchestrator that uses tool agent
	orchestrator := &mockSimpleAgent{
		name:     "orchestrator",
		response: "Coordinating response",
	}
	orchestratorExec := NewExecutorWithConfig(orchestrator, ExecutorConfig{Name: "Orchestrator"}).
		WithSubAgent("tools", toolExec)

	// Test: Orchestrator → Tools → Calculator
	msg := NewUserMessage("What is 2+2?")

	// Orchestrator calls tools sub-agent
	toolsArtifact, err := orchestratorExec.CallSubAgent(context.Background(), "tools", msg)
	if err != nil {
		t.Fatalf("orchestrator→tools failed: %v", err)
	}
	if !contains(ExtractTextFromArtifact(toolsArtifact), "Using tools") {
		t.Error("tools agent should respond")
	}

	// Tools calls calculator sub-agent
	calcArtifact, err := toolExec.CallSubAgent(context.Background(), "calculator", msg)
	if err != nil {
		t.Fatalf("tools→calculator failed: %v", err)
	}
	if !contains(ExtractTextFromArtifact(calcArtifact), "4") {
		t.Error("calculator should return 4")
	}

	// Verify capabilities propagation
	caps := orchestratorExec.GetCapabilities()
	if len(caps) == 0 {
		t.Error("orchestrator should have capabilities")
	}
}

// ============================================================================
// Agent Card Tests
// ============================================================================

func TestGetAgentCard(t *testing.T) {
	agent := &mockSimpleAgent{name: "test", response: "ok"}
	executor := NewExecutorWithConfig(agent, ExecutorConfig{Name: "TestAgent"})

	card := executor.GetAgentCard()

	if card.Name != "TestAgent" {
		t.Errorf("expected name 'TestAgent', got '%s'", card.Name)
	}
	if card.Version == "" {
		t.Error("expected non-empty version")
	}
}

func TestGetCapabilities_WithSubAgents(t *testing.T) {
	sub1 := NewExecutor(&mockSimpleAgent{name: "sub1", response: "ok"})
	sub2 := NewExecutor(&mockSimpleAgent{name: "sub2", response: "ok"})

	parent := NewExecutor(&mockSimpleAgent{name: "parent", response: "ok"}).
		WithSubAgent("helper1", sub1).
		WithSubAgent("helper2", sub2)

	caps := parent.GetCapabilities()

	// Should have 2 sub-agent capabilities
	agentCaps := 0
	for _, cap := range caps {
		if cap.Type == "agent" {
			agentCaps++
		}
	}

	if agentCaps != 2 {
		t.Errorf("expected 2 agent capabilities, got %d", agentCaps)
	}
}

// ============================================================================
// Convenience Constructor Tests
// ============================================================================

func TestNewExecutorWithSubAgents(t *testing.T) {
	search := NewExecutor(&mockSimpleAgent{name: "search", response: "ok"})
	reasoning := NewExecutor(&mockSimpleAgent{name: "reasoning", response: "ok"})

	parent := NewExecutorWithSubAgents(
		&mockSimpleAgent{name: "parent", response: "ok"},
		map[string]*A2AExecutor{
			"search":    search,
			"reasoning": reasoning,
		},
	)

	if len(parent.ListSubAgents()) != 2 {
		t.Errorf("expected 2 sub-agents, got %d", len(parent.ListSubAgents()))
	}

	// Verify both sub-agents are accessible
	_, ok1 := parent.GetSubAgent("search")
	_, ok2 := parent.GetSubAgent("reasoning")

	if !ok1 || !ok2 {
		t.Error("both sub-agents should be registered")
	}
}

// ============================================================================
// Error Handling Tests
// ============================================================================

func TestExecutorNilMessage(t *testing.T) {
	executor := NewExecutor(&mockSimpleAgent{name: "test", response: "ok"})

	_, err := executor.SendMessage(context.Background(), nil)
	if err == nil {
		t.Error("expected error for nil message")
	}
}

// Note: Helper functions contains() and findSubstring() are defined in converters_test.go

// ============================================================================
// Helper Method Tests
// ============================================================================

func TestExecutorName(t *testing.T) {
	executor := NewExecutorWithConfig(
		&mockSimpleAgent{name: "test", response: "ok"},
		ExecutorConfig{Name: "MyAgent"},
	)

	if executor.Name() != "MyAgent" {
		t.Errorf("expected name 'MyAgent', got '%s'", executor.Name())
	}
}

func TestExecutorUnwrapAgent(t *testing.T) {
	agent := &mockSimpleAgent{name: "test", response: "ok"}
	executor := NewExecutor(agent)

	unwrapped := executor.UnwrapAgent()
	if unwrapped != agent {
		t.Error("unwrapped agent should match original")
	}
}
