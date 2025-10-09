package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	a2a "github.com/XiaoConstantine/dspy-go/pkg/agents/communication"
	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ============================================================================
// Example Agents - Simple implementations for demonstration
// ============================================================================

// CalculatorAgent performs basic arithmetic operations.
type CalculatorAgent struct {
	name string
}

func (c *CalculatorAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	question, ok := input["question"].(string)
	if !ok {
		return nil, fmt.Errorf("input 'question' is not a string or is missing")
	}

	// Simulate calculation
	var result string
	if containsAny(question, []string{"2+2", "add 2 and 2"}) {
		result = "The answer is 4"
	} else if containsAny(question, []string{"5*3", "multiply 5 by 3"}) {
		result = "The answer is 15"
	} else {
		result = "I can calculate basic arithmetic. Try asking '2+2' or '5*3'"
	}

	return map[string]interface{}{
		"answer": result,
	}, nil
}

func (c *CalculatorAgent) GetCapabilities() []core.Tool {
	return nil
}

func (c *CalculatorAgent) GetMemory() agents.Memory {
	return nil
}

// SearchAgent simulates web search capabilities.
type SearchAgent struct {
	name string
}

func (s *SearchAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	question, ok := input["question"].(string)
	if !ok {
		return nil, fmt.Errorf("input 'question' is not a string or is missing")
	}

	// Simulate search results
	var result string
	if containsAny(question, []string{"weather", "temperature"}) {
		result = "Current weather: Sunny, 72°F"
	} else if containsAny(question, []string{"capital", "France"}) {
		result = "The capital of France is Paris"
	} else {
		result = fmt.Sprintf("Search results for: %s - Found 3 relevant articles", question)
	}

	return map[string]interface{}{
		"answer": result,
	}, nil
}

func (s *SearchAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *SearchAgent) GetMemory() agents.Memory {
	return nil
}

// ReasoningAgent provides logical reasoning capabilities.
type ReasoningAgent struct {
	name string
}

func (r *ReasoningAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	question, ok := input["question"].(string)
	if !ok {
		return nil, fmt.Errorf("input 'question' is not a string or is missing")
	}

	// Simulate reasoning
	result := fmt.Sprintf("Let me think about '%s'...\n", question)
	result += "Step 1: Analyze the question\n"
	result += "Step 2: Consider relevant information\n"
	result += "Step 3: Draw conclusion\n"
	result += "Conclusion: Based on logical reasoning, the answer involves careful consideration of the context."

	return map[string]interface{}{
		"answer": result,
	}, nil
}

func (r *ReasoningAgent) GetCapabilities() []core.Tool {
	return nil
}

func (r *ReasoningAgent) GetMemory() agents.Memory {
	return nil
}

// OrchestratorAgent coordinates multiple sub-agents.
type OrchestratorAgent struct {
	name     string
	executor *a2a.A2AExecutor
}

// NewOrchestratorWithExecutor creates an OrchestratorAgent with its A2AExecutor properly initialized.
// This factory function ensures the agent and executor are always in a valid state,
// preventing nil pointer dereferences from the circular dependency.
func NewOrchestratorWithExecutor(name string) (*OrchestratorAgent, *a2a.A2AExecutor) {
	agent := &OrchestratorAgent{name: name}
	executor := a2a.NewExecutorWithConfig(agent, a2a.ExecutorConfig{
		Name: name + "Agent",
	})
	agent.executor = executor
	return agent, executor
}

func (o *OrchestratorAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	question, ok := input["question"].(string)
	if !ok {
		return nil, fmt.Errorf("input 'question' is not a string or is missing")
	}

	fmt.Printf("\n🎯 Orchestrator analyzing: %s\n", question)

	// Decide which sub-agent to call based on question
	var subAgent string
	if containsAny(question, []string{"calculate", "+", "*", "-", "/"}) {
		subAgent = "calculator"
		fmt.Println("   → Delegating to Calculator agent")
	} else if containsAny(question, []string{"search", "find", "weather", "capital"}) {
		subAgent = "search"
		fmt.Println("   → Delegating to Search agent")
	} else {
		subAgent = "reasoning"
		fmt.Println("   → Delegating to Reasoning agent")
	}

	// Call the appropriate sub-agent using a2a protocol
	result, err := o.executor.CallSubAgentSimple(ctx, subAgent, question)
	if err != nil {
		return nil, err
	}

	// Orchestrator adds its own summary
	finalAnswer := fmt.Sprintf("🤖 Orchestrator Response:\n\nI delegated to the %s agent:\n%s", subAgent, result)

	return map[string]interface{}{
		"answer": finalAnswer,
	}, nil
}

func (o *OrchestratorAgent) GetCapabilities() []core.Tool {
	return nil
}

func (o *OrchestratorAgent) GetMemory() agents.Memory {
	return nil
}

// ============================================================================
// Helper Functions
// ============================================================================

func containsAny(s string, substrs []string) bool {
	for _, substr := range substrs {
		if strings.Contains(s, substr) {
			return true
		}
	}
	return false
}

// ============================================================================
// Main Example
// ============================================================================

func main() {
	ctx := context.Background()

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          A2A Agent Composition Example - dspy-go               ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// ========================================================================
	// Example 1: Basic Agent-to-Agent Communication
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Example 1: Basic Agent-to-Agent Communication")
	fmt.Println(strings.Repeat("=", 64))

	// Create leaf agents
	calculator := &CalculatorAgent{name: "Calculator"}
	search := &SearchAgent{name: "Search"}

	// Wrap agents with A2AExecutor for composition
	calcExec := a2a.NewExecutorWithConfig(calculator, a2a.ExecutorConfig{
		Name: "CalculatorAgent",
	})

	searchExec := a2a.NewExecutorWithConfig(search, a2a.ExecutorConfig{
		Name: "SearchAgent",
	})

	// Test individual agents
	fmt.Println("\n📊 Testing Calculator Agent:")
	msg := a2a.NewUserMessage("What is 2+2?")
	artifact, err := calcExec.Execute(ctx, msg)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Result: %s\n", a2a.ExtractTextFromArtifact(artifact))

	fmt.Println("\n🔍 Testing Search Agent:")
	msg = a2a.NewUserMessage("What's the weather?")
	artifact, err = searchExec.Execute(ctx, msg)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Result: %s\n", a2a.ExtractTextFromArtifact(artifact))

	// ========================================================================
	// Example 2: Multi-Level Agent Hierarchy
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Example 2: Multi-Level Agent Hierarchy")
	fmt.Println(strings.Repeat("=", 64))

	// Create reasoning agent
	reasoning := &ReasoningAgent{name: "Reasoning"}
	reasoningExec := a2a.NewExecutorWithConfig(reasoning, a2a.ExecutorConfig{
		Name: "ReasoningAgent",
	})

	// Create orchestrator with sub-agents using factory function
	// This ensures the agent and executor are always in a valid state
	_, orchestratorExec := NewOrchestratorWithExecutor("Orchestrator")

	// Register sub-agents (this is the key a2a composition feature!)
	orchestratorExec.WithSubAgent("calculator", calcExec).
		WithSubAgent("search", searchExec).
		WithSubAgent("reasoning", reasoningExec)

	// ========================================================================
	// Example 3: Agent Orchestration in Action
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Example 3: Agent Orchestration in Action")
	fmt.Println(strings.Repeat("=", 64))

	questions := []string{
		"Can you calculate 2+2 for me?",
		"What's the weather like today?",
		"What is the meaning of life?",
	}

	for i, question := range questions {
		fmt.Printf("\n[Question %d] %s\n", i+1, question)
		fmt.Println(strings.Repeat("─", 64))

		msg := a2a.NewUserMessage(question)
		artifact, err := orchestratorExec.Execute(ctx, msg)
		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}

		result := a2a.ExtractTextFromArtifact(artifact)
		fmt.Printf("\n%s\n", result)
	}

	// ========================================================================
	// Example 4: Direct Sub-Agent Calls
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Example 4: Direct Sub-Agent Calls (via a2a protocol)")
	fmt.Println(strings.Repeat("=", 64))

	fmt.Println("\n📞 Calling calculator sub-agent directly:")
	result, err := orchestratorExec.CallSubAgentSimple(ctx, "calculator", "What is 5*3?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Result: %s\n", result)

	fmt.Println("\n📞 Calling search sub-agent directly:")
	result, err = orchestratorExec.CallSubAgentSimple(ctx, "search", "What is the capital of France?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Result: %s\n", result)

	// ========================================================================
	// Example 5: Agent Capabilities Discovery
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Example 5: Agent Capabilities Discovery")
	fmt.Println(strings.Repeat("=", 64))

	fmt.Println("\n🔧 Orchestrator's available sub-agents:")
	subAgents := orchestratorExec.ListSubAgents()
	for _, name := range subAgents {
		fmt.Printf("   • %s\n", name)
	}

	fmt.Println("\n📋 Agent Card:")
	card := orchestratorExec.GetAgentCard()
	fmt.Printf("   Name: %s\n", card.Name)
	fmt.Printf("   Description: %s\n", card.Description)
	fmt.Printf("   Version: %s\n", card.Version)
	fmt.Printf("   Capabilities: %d sub-agents\n", len(card.Capabilities))

	for _, cap := range card.Capabilities {
		fmt.Printf("     - %s (%s): %s\n", cap.Name, cap.Type, cap.Description)
	}

	// ========================================================================
	// Summary
	// ========================================================================

	fmt.Println("\n" + strings.Repeat("=", 64))
	fmt.Println("Summary: Key A2A Features Demonstrated")
	fmt.Println(strings.Repeat("=", 64))
	fmt.Print(`
✅ Agent Composition - Parent agents can have sub-agents
✅ In-Process Communication - Agents communicate via a2a protocol (no HTTP)
✅ Hierarchy Support - Multi-level agent hierarchies (orchestrator → tools)
✅ Message Protocol - Standardized a2a messages, tasks, and artifacts
✅ Capability Discovery - Agents can expose their sub-agents' capabilities
✅ Type-Safe API - Clean Go interfaces for agent interaction

🎯 This demonstrates Google's a2a protocol for agent interoperability!
   Agents from different sources can work together seamlessly.
`)
	fmt.Println()

	fmt.Println(strings.Repeat("=", 64))
}
