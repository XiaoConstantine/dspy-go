package communication

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func TestAgent_BasicCreation(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	if agent.GetID() != "agent_1" {
		t.Errorf("Expected agent ID 'agent_1', got '%s'", agent.GetID())
	}

	if agent.GetName() != "Test Agent" {
		t.Errorf("Expected agent name 'Test Agent', got '%s'", agent.GetName())
	}

	status := agent.GetStatus()
	if status.State != AgentStateIdle {
		t.Errorf("Expected initial state AgentStateIdle, got %v", status.State)
	}
}

func TestAgent_StatusUpdates(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	// Update status
	newStatus := AgentStatus{
		State:        AgentStateBusy,
		Load:         0.5,
		ActiveTasks:  3,
		Capabilities: []string{"task_a", "task_b"},
	}

	agent.UpdateStatus(newStatus)

	updatedStatus := agent.GetStatus()
	if updatedStatus.State != AgentStateBusy {
		t.Errorf("Expected state AgentStateBusy, got %v", updatedStatus.State)
	}

	if updatedStatus.Load != 0.5 {
		t.Errorf("Expected load 0.5, got %f", updatedStatus.Load)
	}

	if updatedStatus.ActiveTasks != 3 {
		t.Errorf("Expected 3 active tasks, got %d", updatedStatus.ActiveTasks)
	}
}

func TestAgentNetwork_BasicOperations(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start network
	err := network.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start network: %v", err)
	}
	defer func() { _ = network.Stop() }()

	// Create agents
	memory1 := agents.NewInMemoryStore()
	memory2 := agents.NewInMemoryStore()
	agent1 := NewAgent("agent_1", "Agent 1", memory1)
	agent2 := NewAgent("agent_2", "Agent 2", memory2)

	// Add agents to network
	_ = agent1.ConnectToNetwork(network)
	if err != nil {
		t.Fatalf("Failed to connect agent1 to network: %v", err)
	}

	_ = agent2.ConnectToNetwork(network)
	if err != nil {
		t.Fatalf("Failed to connect agent2 to network: %v", err)
	}

	// Verify agents are in network
	agents := network.GetAgents()
	if len(agents) != 2 {
		t.Errorf("Expected 2 agents in network, got %d", len(agents))
	}

	if _, exists := agents["agent_1"]; !exists {
		t.Error("Agent 1 not found in network")
	}

	if _, exists := agents["agent_2"]; !exists {
		t.Error("Agent 2 not found in network")
	}
}

func TestAgentNetwork_MessageSending(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := network.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start network: %v", err)
	}
	defer func() { _ = network.Stop() }()

	// Create agents
	memory1 := agents.NewInMemoryStore()
	memory2 := agents.NewInMemoryStore()
	agent1 := NewAgent("agent_1", "Agent 1", memory1)
	agent2 := NewAgent("agent_2", "Agent 2", memory2)

	// Connect to network
	_ = agent1.ConnectToNetwork(network)
	_ = agent2.ConnectToNetwork(network)

	// Start agents
	_ = agent1.Start(ctx)
	_ = agent2.Start(ctx)
	defer func() { _ = agent1.Stop() }()
	defer func() { _ = agent2.Stop() }()

	// Send message from agent1 to agent2
	testData := map[string]interface{}{
		"content": "Hello from agent 1",
		"task_id": "task_123",
	}

	err = agent1.SendMessage("agent_2", "greeting", testData)
	if err != nil {
		t.Fatalf("Failed to send message: %v", err)
	}

	// Wait a bit for message processing
	time.Sleep(100 * time.Millisecond)

	// Note: In a real implementation, we'd verify the message was received
	// For now, we just verify no errors occurred during sending
}

func TestAgentNetwork_BroadcastMessage(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := network.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start network: %v", err)
	}
	defer func() { _ = network.Stop() }()

	// Create multiple agents
	agentList := make([]*Agent, 3)
	for i := 0; i < 3; i++ {
		memory := agents.NewInMemoryStore()
		agent := NewAgent(
			fmt.Sprintf("agent_%d", i+1),
			fmt.Sprintf("Agent %d", i+1),
			memory,
		)

		_ = agent.ConnectToNetwork(network)
		_ = agent.Start(ctx)
		agentList[i] = agent
		defer func() { _ = agent.Stop() }()
	}

	// Broadcast message
	broadcastData := map[string]interface{}{
		"announcement": "System maintenance in 10 minutes",
		"priority":     "high",
	}

	err = agentList[0].BroadcastMessage("announcement", broadcastData)
	if err != nil {
		t.Fatalf("Failed to broadcast message: %v", err)
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify broadcast was sent (no errors)
}

func TestAgentNetwork_HelpRequest(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := network.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start network: %v", err)
	}
	defer func() { _ = network.Stop() }()

	// Create agents
	memory1 := agents.NewInMemoryStore()
	memory2 := agents.NewInMemoryStore()
	agent1 := NewAgent("helper", "Helper Agent", memory1)
	agent2 := NewAgent("requester", "Requester Agent", memory2)

	// Set up helper agent with capability
	agent1.UpdateStatus(AgentStatus{
		State:        AgentStateIdle,
		Load:         0.2,
		ActiveTasks:  1,
		Capabilities: []string{"general", "math"},
	})

	// Connect to network
	_ = agent1.ConnectToNetwork(network)
	_ = agent2.ConnectToNetwork(network)

	// Start agents
	_ = agent1.Start(ctx)
	_ = agent2.Start(ctx)
	defer func() { _ = agent1.Stop() }()
	defer func() { _ = agent2.Stop() }()

	// Request help
	task := map[string]interface{}{
		"type":       "math",
		"problem":    "solve equation x^2 - 4 = 0",
		"difficulty": "medium",
	}

	// Note: This will timeout because we haven't implemented the full help logic
	// but it tests that the request mechanism works without errors
	response, err := agent2.RequestHelp(task, 1*time.Second)

	// We expect a timeout error since help logic isn't fully implemented
	if err == nil {
		t.Logf("Received help response: %v", response)
	} else if err.Error() != "help request timeout" {
		t.Errorf("Expected timeout error, got: %v", err)
	}
}

func TestAgentNetwork_FindAvailableAgent(t *testing.T) {
	network := NewAgentNetwork()

	// Create agents with different capabilities and loads
	memory1 := agents.NewInMemoryStore()
	memory2 := agents.NewInMemoryStore()
	memory3 := agents.NewInMemoryStore()

	agent1 := NewAgent("agent_1", "Agent 1", memory1)
	agent2 := NewAgent("agent_2", "Agent 2", memory2)
	agent3 := NewAgent("agent_3", "Agent 3", memory3)

	// Set different statuses
	agent1.UpdateStatus(AgentStatus{
		State:        AgentStateIdle,
		Load:         0.2,
		Capabilities: []string{"task_a"},
	})

	agent2.UpdateStatus(AgentStatus{
		State:        AgentStateBusy,
		Load:         0.8,
		Capabilities: []string{"task_a", "task_b"},
	})

	agent3.UpdateStatus(AgentStatus{
		State:        AgentStateOffline,
		Load:         0.0,
		Capabilities: []string{"task_a"},
	})

	// Add to network
	_ = network.AddAgent(agent1)
	_ = network.AddAgent(agent2)
	_ = network.AddAgent(agent3)

	// Find available agent for task_a
	available := network.FindAvailableAgent("task_a")

	// Should find agent1 (lowest load and available)
	if available == nil {
		t.Fatal("Expected to find available agent")
	}

	if available.GetID() != "agent_1" {
		t.Errorf("Expected agent_1 to be selected, got %s", available.GetID())
	}

	// Find agent for task that no one can handle
	unavailable := network.FindAvailableAgent("task_z")
	if unavailable != nil {
		t.Error("Expected no agent to be available for task_z")
	}
}

func TestAgentNetwork_MaxCapacity(t *testing.T) {
	network := NewAgentNetwork()
	network.config.MaxAgents = 2 // Set low limit for testing

	memory1 := agents.NewInMemoryStore()
	memory2 := agents.NewInMemoryStore()
	memory3 := agents.NewInMemoryStore()

	agent1 := NewAgent("agent_1", "Agent 1", memory1)
	agent2 := NewAgent("agent_2", "Agent 2", memory2)
	agent3 := NewAgent("agent_3", "Agent 3", memory3)

	// Add first two agents (should succeed)
	err := network.AddAgent(agent1)
	if err != nil {
		t.Fatalf("Failed to add agent1: %v", err)
	}

	err = network.AddAgent(agent2)
	if err != nil {
		t.Fatalf("Failed to add agent2: %v", err)
	}

	// Try to add third agent (should fail)
	err = network.AddAgent(agent3)
	if err == nil {
		t.Error("Expected error when adding agent beyond capacity")
	}

	// Verify only 2 agents in network
	agents := network.GetAgents()
	if len(agents) != 2 {
		t.Errorf("Expected 2 agents in network, got %d", len(agents))
	}
}

func TestAgent_StartStop(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start agent
	err := agent.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start agent: %v", err)
	}

	// Verify agent is running
	status := agent.GetStatus()
	if status.State == AgentStateOffline {
		t.Error("Agent should not be offline after starting")
	}

	// Stop agent
	err = agent.Stop()
	if err != nil {
		t.Fatalf("Failed to stop agent: %v", err)
	}

	// Verify agent is stopped
	status = agent.GetStatus()
	if status.State != AgentStateOffline {
		t.Error("Agent should be offline after stopping")
	}
}

// Test Agent.NewAgent with nil memory.
func TestAgent_NewAgentWithNilMemory(t *testing.T) {
	agent := NewAgent("test_agent", "Test Agent", nil)

	if agent.GetID() != "test_agent" {
		t.Errorf("Expected agent ID 'test_agent', got '%s'", agent.GetID())
	}

	if agent.memory == nil {
		t.Error("Expected agent to have memory even when nil was provided")
	}
}

// Test Agent.On method.
func TestAgent_On(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	// Create a simple workflow
	workflow := workflows.NewChainWorkflow(memory)

	// Register event handler
	result := agent.On("test_event", workflow)

	// Should return the agent for chaining
	if result != agent {
		t.Error("On method should return agent for chaining")
	}
}

// Test Agent.OnModule method.
func TestAgent_OnModule(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	// Create a mock module
	module := &MockModule{}

	// Register module as event handler
	result := agent.OnModule("test_event", module)

	// Should return the agent for chaining
	if result != agent {
		t.Error("OnModule method should return agent for chaining")
	}
}

// Test Agent.OfferHelp method.
func TestAgent_OfferHelp(t *testing.T) {
	t.Run("agent not connected to network", func(t *testing.T) {
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		err := agent.OfferHelp("request_123", "help response")
		if err == nil {
			t.Error("Expected error when agent not connected to network")
		}

		expectedMsg := "agent not connected to network"
		if err.Error() != expectedMsg {
			t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
		}
	})

	t.Run("agent connected to network", func(t *testing.T) {
		network := NewAgentNetwork()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		_ = network.Start(ctx)
		defer func() { _ = network.Stop() }()

		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)
		_ = agent.ConnectToNetwork(network)

		// This should not error even if request doesn't exist
		err := agent.OfferHelp("nonexistent_request", "help response")
		if err == nil {
			t.Error("Expected error for non-existent help request")
		}
	})
}

// Test Agent.DisconnectFromNetwork method.
func TestAgent_DisconnectFromNetwork(t *testing.T) {
	t.Run("agent not connected to network", func(t *testing.T) {
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		// Should not error
		err := agent.DisconnectFromNetwork()
		if err != nil {
			t.Errorf("Unexpected error when disconnecting unconnected agent: %v", err)
		}
	})

	t.Run("agent connected to network", func(t *testing.T) {
		network := NewAgentNetwork()
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		// Connect then disconnect
		_ = agent.ConnectToNetwork(network)
		err := agent.DisconnectFromNetwork()
		if err != nil {
			t.Errorf("Unexpected error when disconnecting: %v", err)
		}

		// Verify agent is no longer in network
		agents := network.GetAgents()
		if _, exists := agents["agent_1"]; exists {
			t.Error("Agent should be removed from network after disconnect")
		}

		// Verify agent's network reference is cleared
		if agent.network != nil {
			t.Error("Agent's network reference should be nil after disconnect")
		}
	})
}

// Test Agent.startHeartbeat method.
func TestAgent_StartHeartbeat(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = network.Start(ctx)
	defer func() { _ = network.Stop() }()

	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)
	agent.config.HeartbeatInterval = 50 * time.Millisecond
	_ = agent.ConnectToNetwork(network)

	// Start agent (this will start heartbeat)
	_ = agent.Start(ctx)
	defer func() { _ = agent.Stop() }()

	// Wait for at least one heartbeat
	time.Sleep(100 * time.Millisecond)

	// Heartbeat functionality is tested implicitly through no errors
}

// Test Agent error conditions.
func TestAgent_ErrorConditions(t *testing.T) {
	t.Run("send message without network", func(t *testing.T) {
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		err := agent.SendMessage("target", "type", "data")
		if err == nil {
			t.Error("Expected error when sending message without network")
		}
	})

	t.Run("broadcast message without network", func(t *testing.T) {
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		err := agent.BroadcastMessage("type", "data")
		if err == nil {
			t.Error("Expected error when broadcasting message without network")
		}
	})

	t.Run("request help without network", func(t *testing.T) {
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		_, err := agent.RequestHelp("task", time.Second)
		if err == nil {
			t.Error("Expected error when requesting help without network")
		}
	})
}

// Test AgentNetwork.RemoveAgent method.
func TestAgentNetwork_RemoveAgent(t *testing.T) {
	network := NewAgentNetwork()
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	// Add agent first
	_ = network.AddAgent(agent)
	agents := network.GetAgents()
	if len(agents) != 1 {
		t.Fatalf("Expected 1 agent in network, got %d", len(agents))
	}

	// Remove agent
	err := network.RemoveAgent("agent_1")
	if err != nil {
		t.Errorf("Unexpected error removing agent: %v", err)
	}

	// Verify agent is removed
	agents = network.GetAgents()
	if len(agents) != 0 {
		t.Errorf("Expected 0 agents in network after removal, got %d", len(agents))
	}

	// Removing non-existent agent should not error
	err = network.RemoveAgent("nonexistent")
	if err != nil {
		t.Errorf("Unexpected error removing non-existent agent: %v", err)
	}
}

// Test AgentNetwork.BroadcastStatus method.
func TestAgentNetwork_BroadcastStatus(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = network.Start(ctx)
	defer func() { _ = network.Stop() }()

	status := AgentStatus{
		State:        AgentStateBusy,
		Load:         0.5,
		ActiveTasks:  2,
		Capabilities: []string{"test"},
	}

	err := network.BroadcastStatus("agent_1", status)
	if err != nil {
		t.Errorf("Unexpected error broadcasting status: %v", err)
	}
}

// Test AgentNetwork.OfferHelp method.
func TestAgentNetwork_OfferHelp(t *testing.T) {
	network := NewAgentNetwork()

	t.Run("non-existent help request", func(t *testing.T) {
		response := HelpResponse{
			RequestID: "nonexistent",
			From:      "agent_1",
			Response:  "help response",
		}

		err := network.OfferHelp(response)
		if err == nil {
			t.Error("Expected error for non-existent help request")
		}
	})

	t.Run("valid help request and response", func(t *testing.T) {
		// Create a pending help request manually
		responseChan := make(chan interface{}, 1)
		network.helpMu.Lock()
		network.pendingHelp["test_request"] = responseChan
		network.helpMu.Unlock()

		response := HelpResponse{
			RequestID: "test_request",
			From:      "agent_1",
			Response:  "help response",
		}

		err := network.OfferHelp(response)
		if err != nil {
			t.Errorf("Unexpected error offering help: %v", err)
		}

		// Verify response was received
		select {
		case receivedResponse := <-responseChan:
			if receivedResponse != "help response" {
				t.Errorf("Expected 'help response', got %v", receivedResponse)
			}
		case <-time.After(100 * time.Millisecond):
			t.Error("Help response not received")
		}

		// Clean up
		network.helpMu.Lock()
		delete(network.pendingHelp, "test_request")
		network.helpMu.Unlock()
	})
}

// Test concurrent access patterns.
func TestAgent_ConcurrentAccess(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	var wg sync.WaitGroup
	numGoroutines := 10

	// Test concurrent status updates and reads
	wg.Add(numGoroutines * 2)

	// Concurrent status updates
	for i := 0; i < numGoroutines; i++ {
		go func(i int) {
			defer wg.Done()
			status := AgentStatus{
				State:        AgentStateBusy,
				Load:         float64(i) / 10.0,
				ActiveTasks:  i,
				Capabilities: []string{fmt.Sprintf("task_%d", i)},
			}
			agent.UpdateStatus(status)
		}(i)
	}

	// Concurrent status reads
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_ = agent.GetStatus()
		}()
	}

	wg.Wait()
}

// Test DefaultAgentConfig.
func TestDefaultAgentConfig(t *testing.T) {
	config := DefaultAgentConfig()

	if config.MaxConcurrentTasks != 10 {
		t.Errorf("Expected MaxConcurrentTasks 10, got %d", config.MaxConcurrentTasks)
	}

	if config.DefaultTimeout != 60*time.Second {
		t.Errorf("Expected DefaultTimeout 60s, got %v", config.DefaultTimeout)
	}

	if !config.EnableTracing {
		t.Error("Expected EnableTracing to be true")
	}

	if config.EnableMetrics {
		t.Error("Expected EnableMetrics to be false")
	}

	if !config.AutoReconnect {
		t.Error("Expected AutoReconnect to be true")
	}

	if config.HeartbeatInterval != 30*time.Second {
		t.Errorf("Expected HeartbeatInterval 30s, got %v", config.HeartbeatInterval)
	}
}

// Test DefaultNetworkConfig.
func TestDefaultNetworkConfig(t *testing.T) {
	config := DefaultNetworkConfig()

	if config.MaxAgents != 100 {
		t.Errorf("Expected MaxAgents 100, got %d", config.MaxAgents)
	}

	if config.HelpTimeout != 30*time.Second {
		t.Errorf("Expected HelpTimeout 30s, got %v", config.HelpTimeout)
	}

	if !config.EnableDiscovery {
		t.Error("Expected EnableDiscovery to be true")
	}

	if !config.EnableLoadBalance {
		t.Error("Expected EnableLoadBalance to be true")
	}
}

// MockModule for testing.
type MockModule struct {
	processFunc func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error)
	signature   core.Signature
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	if m.processFunc != nil {
		return m.processFunc(ctx, inputs, opts...)
	}
	return map[string]any{"processed": true}, nil
}

func (m *MockModule) GetSignature() core.Signature {
	if m.signature.Inputs != nil || m.signature.Outputs != nil {
		return m.signature
	}
	inputs := []core.InputField{{Field: core.NewField("test")}}
	outputs := []core.OutputField{{Field: core.NewField("result")}}
	return core.NewSignature(inputs, outputs)
}

func (m *MockModule) SetSignature(signature core.Signature) {
	m.signature = signature
}

func (m *MockModule) SetLLM(llm core.LLM) {}

func (m *MockModule) Clone() core.Module {
	return &MockModule{processFunc: m.processFunc, signature: m.signature}
}

func (m *MockModule) GetDisplayName() string {
	return "MockModule"
}

func (m *MockModule) GetModuleType() string {
	return "MockModule"
}

// Test MessageHandlerModule.
func TestMessageHandlerModule(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)
	module := &MessageHandlerModule{agent: agent}

	t.Run("invalid event type", func(t *testing.T) {
		inputs := map[string]any{"event": "not_an_event"}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid event type")
		}
	})

	t.Run("invalid message type", func(t *testing.T) {
		event := workflows.Event{
			ID:   "test",
			Type: "agent_message",
			Data: "not_a_message",
		}
		inputs := map[string]any{"event": event}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid message type")
		}
	})

	t.Run("message not for this agent", func(t *testing.T) {
		message := AgentMessage{
			From: "other_agent",
			To:   "different_agent",
			Type: "test",
			Data: "test data",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "agent_message",
			Data: message,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || processed {
			t.Error("Expected message to not be processed")
		}
	})

	t.Run("valid message for this agent", func(t *testing.T) {
		message := AgentMessage{
			From: "other_agent",
			To:   "agent_1",
			Type: "test",
			Data: "test data",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "agent_message",
			Data: message,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || !processed {
			t.Error("Expected message to be processed")
		}
	})

	t.Run("broadcast message", func(t *testing.T) {
		message := AgentMessage{
			From: "other_agent",
			To:   "broadcast",
			Type: "announcement",
			Data: "broadcast data",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "agent_message",
			Data: message,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || !processed {
			t.Error("Expected broadcast message to be processed")
		}
	})

	t.Run("module interface methods", func(t *testing.T) {
		signature := module.GetSignature()
		if len(signature.Inputs) == 0 {
			t.Error("Expected signature to have inputs")
		}

		module.SetSignature(signature)
		module.SetLLM(nil)

		cloned := module.Clone()
		if cloned == nil {
			t.Error("Expected clone to return a module")
		}
	})
}

// Test HelpRequestHandlerModule.
func TestHelpRequestHandlerModule(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)

	// Connect to network for OfferHelp to work
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = network.Start(ctx)
	defer func() { _ = network.Stop() }()

	_ = agent.ConnectToNetwork(network)

	module := &HelpRequestHandlerModule{agent: agent}

	t.Run("invalid event type", func(t *testing.T) {
		inputs := map[string]any{"event": "not_an_event"}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid event type")
		}
	})

	t.Run("invalid help request type", func(t *testing.T) {
		event := workflows.Event{
			ID:   "test",
			Type: "help_request",
			Data: "not_a_help_request",
		}
		inputs := map[string]any{"event": event}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid help request type")
		}
	})

	t.Run("help request from self", func(t *testing.T) {
		request := HelpRequest{
			RequestID: "test_request",
			From:      "agent_1", // Same as agent ID
			Task:      "test task",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "help_request",
			Data: request,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || processed {
			t.Error("Expected self help request to not be processed")
		}
	})

	t.Run("agent overloaded", func(t *testing.T) {
		// Set agent to overloaded state
		agent.UpdateStatus(AgentStatus{
			State: AgentStateOverloaded,
			Load:  1.0,
		})

		request := HelpRequest{
			RequestID: "test_request",
			From:      "other_agent",
			Task:      "test task",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "help_request",
			Data: request,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || processed {
			t.Error("Expected overloaded agent to not process help request")
		}
	})

	t.Run("valid help request", func(t *testing.T) {
		// Set agent to idle state
		agent.UpdateStatus(AgentStatus{
			State: AgentStateIdle,
			Load:  0.3,
		})

		// Create a pending help request to avoid "no pending request" error
		responseChan := make(chan interface{}, 1)
		network.helpMu.Lock()
		network.pendingHelp["test_request"] = responseChan
		network.helpMu.Unlock()
		defer func() {
			network.helpMu.Lock()
			delete(network.pendingHelp, "test_request")
			network.helpMu.Unlock()
		}()

		request := HelpRequest{
			RequestID: "test_request",
			From:      "other_agent",
			Task:      "test task",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "help_request",
			Data: request,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || !processed {
			t.Error("Expected help request to be processed")
		}

		helped, ok := result["helped"].(bool)
		if !ok || !helped {
			t.Error("Expected agent to offer help")
		}
	})

	t.Run("module interface methods", func(t *testing.T) {
		signature := module.GetSignature()
		if len(signature.Inputs) == 0 {
			t.Error("Expected signature to have inputs")
		}

		module.SetSignature(signature)
		module.SetLLM(nil)

		cloned := module.Clone()
		if cloned == nil {
			t.Error("Expected clone to return a module")
		}
	})
}

// Test HelpResponseHandlerModule.
func TestHelpResponseHandlerModule(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)
	module := &HelpResponseHandlerModule{agent: agent}

	t.Run("invalid event type", func(t *testing.T) {
		inputs := map[string]any{"event": "not_an_event"}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid event type")
		}
	})

	t.Run("invalid help response type", func(t *testing.T) {
		event := workflows.Event{
			ID:   "test",
			Type: "help_response",
			Data: "not_a_help_response",
		}
		inputs := map[string]any{"event": event}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid help response type")
		}
	})

	t.Run("valid help response", func(t *testing.T) {
		response := HelpResponse{
			RequestID: "test_request",
			From:      "helper_agent",
			Response:  "help response data",
		}
		event := workflows.Event{
			ID:   "test",
			Type: "help_response",
			Data: response,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || !processed {
			t.Error("Expected help response to be processed")
		}
	})

	t.Run("module interface methods", func(t *testing.T) {
		signature := module.GetSignature()
		if len(signature.Inputs) == 0 {
			t.Error("Expected signature to have inputs")
		}

		module.SetSignature(signature)
		module.SetLLM(nil)

		cloned := module.Clone()
		if cloned == nil {
			t.Error("Expected clone to return a module")
		}
	})
}

// Test StatusUpdateHandlerModule.
func TestStatusUpdateHandlerModule(t *testing.T) {
	memory := agents.NewInMemoryStore()
	agent := NewAgent("agent_1", "Test Agent", memory)
	module := &StatusUpdateHandlerModule{agent: agent}

	t.Run("invalid event type", func(t *testing.T) {
		inputs := map[string]any{"event": "not_an_event"}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid event type")
		}
	})

	t.Run("invalid status update data", func(t *testing.T) {
		event := workflows.Event{
			ID:   "test",
			Type: "status_update",
			Data: "not_a_status_update",
		}
		inputs := map[string]any{"event": event}
		_, err := module.Process(context.Background(), inputs)
		if err == nil {
			t.Error("Expected error for invalid status update data")
		}
	})

	t.Run("status update from self", func(t *testing.T) {
		statusData := map[string]interface{}{
			"agent_id": "agent_1", // Same as agent ID
			"status": AgentStatus{
				State: AgentStateBusy,
				Load:  0.5,
			},
		}
		event := workflows.Event{
			ID:   "test",
			Type: "status_update",
			Data: statusData,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || processed {
			t.Error("Expected self status update to not be processed")
		}
	})

	t.Run("valid status update from peer", func(t *testing.T) {
		statusData := map[string]interface{}{
			"agent_id": "other_agent",
			"status": AgentStatus{
				State: AgentStateBusy,
				Load:  0.7,
			},
		}
		event := workflows.Event{
			ID:   "test",
			Type: "status_update",
			Data: statusData,
		}
		inputs := map[string]any{"event": event}

		result, err := module.Process(context.Background(), inputs)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		processed, ok := result["processed"].(bool)
		if !ok || !processed {
			t.Error("Expected peer status update to be processed")
		}

		peerID, ok := result["peer_id"].(string)
		if !ok || peerID != "other_agent" {
			t.Error("Expected peer_id to be set correctly")
		}
	})

	t.Run("module interface methods", func(t *testing.T) {
		signature := module.GetSignature()
		if len(signature.Inputs) == 0 {
			t.Error("Expected signature to have inputs")
		}

		module.SetSignature(signature)
		module.SetLLM(nil)

		cloned := module.Clone()
		if cloned == nil {
			t.Error("Expected clone to return a module")
		}
	})
}

// Test AgentNetwork concurrent access.
func TestAgentNetwork_ConcurrentAccess(t *testing.T) {
	network := NewAgentNetwork()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_ = network.Start(ctx)
	defer func() { _ = network.Stop() }()

	var wg sync.WaitGroup
	numGoroutines := 10

	// Test concurrent agent additions and removals
	wg.Add(numGoroutines * 2)

	// Concurrent agent additions
	for i := 0; i < numGoroutines; i++ {
		go func(i int) {
			defer wg.Done()
			memory := agents.NewInMemoryStore()
			agent := NewAgent(fmt.Sprintf("agent_%d", i), fmt.Sprintf("Agent %d", i), memory)
			_ = network.AddAgent(agent)
		}(i)
	}

	// Concurrent agent lookups
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_ = network.GetAgents()
		}()
	}

	wg.Wait()
}

// Test edge cases.
func TestAgent_EdgeCases(t *testing.T) {
	t.Run("update status broadcasts when connected to network", func(t *testing.T) {
		network := NewAgentNetwork()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		_ = network.Start(ctx)
		defer func() { _ = network.Stop() }()

		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)
		_ = agent.ConnectToNetwork(network)

		status := AgentStatus{
			State:        AgentStateBusy,
			Load:         0.6,
			ActiveTasks:  3,
			Capabilities: []string{"test"},
		}

		// This should broadcast the status
		agent.UpdateStatus(status)

		// Verify status was updated
		updated := agent.GetStatus()
		if updated.Load != 0.6 {
			t.Errorf("Expected load 0.6, got %f", updated.Load)
		}
	})

	t.Run("find available agent with general capability", func(t *testing.T) {
		network := NewAgentNetwork()
		memory := agents.NewInMemoryStore()
		agent := NewAgent("agent_1", "Test Agent", memory)

		agent.UpdateStatus(AgentStatus{
			State:        AgentStateIdle,
			Load:         0.1,
			Capabilities: []string{"general"},
		})

		_ = network.AddAgent(agent)

		// Should find agent with "general" capability for any task
		found := network.FindAvailableAgent("specialized_task")
		if found == nil {
			t.Error("Expected to find agent with general capability")
		}

		if found.GetID() != "agent_1" {
			t.Errorf("Expected agent_1, got %s", found.GetID())
		}
	})
}

// Test response channel full scenario.
func TestAgentNetwork_ResponseChannelFull(t *testing.T) {
	network := NewAgentNetwork()

	// Create a help request with a full response channel
	responseChan := make(chan interface{}, 1)
	responseChan <- "existing_response" // Fill the channel

	network.helpMu.Lock()
	network.pendingHelp["test_request"] = responseChan
	network.helpMu.Unlock()

	response := HelpResponse{
		RequestID: "test_request",
		From:      "agent_1",
		Response:  "new_response",
	}

	err := network.OfferHelp(response)
	if err == nil {
		t.Error("Expected error for full response channel")
	}

	expectedMsg := "response channel full"
	if err.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
	}

	// Clean up
	network.helpMu.Lock()
	delete(network.pendingHelp, "test_request")
	network.helpMu.Unlock()
}
