package communication

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// Agent represents a reactive agent that can communicate with other agents.
type Agent struct {
	// id uniquely identifies this agent
	id string

	// name is a human-readable agent identifier
	name string

	// reactive workflow handles event-driven behavior
	reactive *workflows.ReactiveWorkflow

	// memory provides state persistence
	memory agents.Memory

	// network manages agent-to-agent communication
	network *AgentNetwork

	// config holds agent configuration
	config AgentConfig

	// status tracks current agent state
	status AgentStatus

	// mu protects concurrent access
	mu sync.RWMutex
}

// AgentConfig configures agent behavior.
type AgentConfig struct {
	MaxConcurrentTasks int
	DefaultTimeout     time.Duration
	EnableTracing      bool
	EnableMetrics      bool
	AutoReconnect      bool
	HeartbeatInterval  time.Duration
}

// AgentStatus tracks the current state of an agent.
type AgentStatus struct {
	State        AgentState
	Load         float64 // 0.0 to 1.0
	ActiveTasks  int
	LastSeen     time.Time
	Capabilities []string
}

// AgentState represents the current operational state.
type AgentState int

const (
	AgentStateIdle AgentState = iota
	AgentStateBusy
	AgentStateOverloaded
	AgentStateOffline
	AgentStateError
)

// DefaultAgentConfig returns sensible defaults.
func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		MaxConcurrentTasks: 10,
		DefaultTimeout:     60 * time.Second,
		EnableTracing:      true,
		EnableMetrics:      false,
		AutoReconnect:      true,
		HeartbeatInterval:  30 * time.Second,
	}
}

// NewAgent creates a new reactive agent.
func NewAgent(id, name string, memory agents.Memory) *Agent {
	if memory == nil {
		memory = agents.NewInMemoryStore()
	}

	reactive := workflows.NewReactiveWorkflow(memory)

	agent := &Agent{
		id:       id,
		name:     name,
		reactive: reactive,
		memory:   memory,
		config:   DefaultAgentConfig(),
		status: AgentStatus{
			State:       AgentStateIdle,
			Load:        0.0,
			ActiveTasks: 0,
			LastSeen:    time.Now(),
		},
	}

	// Set up core agent communication handlers
	agent.setupCoreHandlers()

	return agent
}

// GetID returns the agent's unique identifier.
func (a *Agent) GetID() string {
	return a.id
}

// GetName returns the agent's human-readable name.
func (a *Agent) GetName() string {
	return a.name
}

// GetStatus returns the current agent status.
func (a *Agent) GetStatus() AgentStatus {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.status
}

// UpdateStatus updates the agent's status.
func (a *Agent) UpdateStatus(status AgentStatus) {
	a.mu.Lock()
	a.status = status
	a.status.LastSeen = time.Now()
	a.mu.Unlock()

	// Broadcast status update if connected to network
	if a.network != nil {
		_ = a.network.BroadcastStatus(a.id, status)
	}
}

// On registers an event handler for this agent.
func (a *Agent) On(eventType string, workflow workflows.Workflow) *Agent {
	a.reactive.On(eventType, workflow)
	return a
}

// OnModule registers a module as an event handler.
func (a *Agent) OnModule(eventType string, module core.Module) *Agent {
	a.reactive.OnModule(eventType, module)
	return a
}

// SendMessage sends a message to another agent.
func (a *Agent) SendMessage(targetAgentID string, messageType string, data interface{}) error {
	if a.network == nil {
		return fmt.Errorf("agent not connected to network")
	}

	message := AgentMessage{
		From:      a.id,
		To:        targetAgentID,
		Type:      messageType,
		Data:      data,
		Timestamp: time.Now(),
	}

	return a.network.SendMessage(message)
}

// BroadcastMessage sends a message to all agents in the network.
func (a *Agent) BroadcastMessage(messageType string, data interface{}) error {
	if a.network == nil {
		return fmt.Errorf("agent not connected to network")
	}

	message := AgentMessage{
		From:      a.id,
		To:        "broadcast",
		Type:      messageType,
		Data:      data,
		Timestamp: time.Now(),
	}

	return a.network.BroadcastMessage(message)
}

// RequestHelp requests assistance from other agents.
func (a *Agent) RequestHelp(task interface{}, timeout time.Duration) (interface{}, error) {
	if a.network == nil {
		return nil, fmt.Errorf("agent not connected to network")
	}

	helpRequest := HelpRequest{
		RequestID: fmt.Sprintf("help_%s_%d", a.id, time.Now().UnixNano()),
		From:      a.id,
		Task:      task,
		Timeout:   timeout,
	}

	return a.network.RequestHelp(helpRequest)
}

// OfferHelp responds to a help request from another agent.
func (a *Agent) OfferHelp(requestID string, response interface{}) error {
	if a.network == nil {
		return fmt.Errorf("agent not connected to network")
	}

	helpResponse := HelpResponse{
		RequestID: requestID,
		From:      a.id,
		Response:  response,
	}

	return a.network.OfferHelp(helpResponse)
}

// Start begins agent operation.
func (a *Agent) Start(ctx context.Context) error {
	a.mu.Lock()
	a.status.State = AgentStateIdle
	a.mu.Unlock()

	// Start reactive workflow processing
	err := a.reactive.Start(ctx)
	if err != nil {
		return fmt.Errorf("failed to start reactive workflow: %w", err)
	}

	// Start heartbeat if connected to network
	if a.network != nil && a.config.HeartbeatInterval > 0 {
		go a.startHeartbeat(ctx)
	}

	return nil
}

// Stop terminates agent operation.
func (a *Agent) Stop() error {
	a.mu.Lock()
	a.status.State = AgentStateOffline
	a.mu.Unlock()

	return a.reactive.Stop()
}

// ConnectToNetwork connects this agent to an agent network.
func (a *Agent) ConnectToNetwork(network *AgentNetwork) error {
	a.network = network
	return network.AddAgent(a)
}

// DisconnectFromNetwork removes this agent from its network.
func (a *Agent) DisconnectFromNetwork() error {
	if a.network == nil {
		return nil
	}

	err := a.network.RemoveAgent(a.id)
	a.network = nil
	return err
}

// setupCoreHandlers sets up built-in agent communication handlers.
func (a *Agent) setupCoreHandlers() {
	// Handle incoming messages
	a.reactive.OnModule("agent_message", &MessageHandlerModule{agent: a})

	// Handle help requests
	a.reactive.OnModule("help_request", &HelpRequestHandlerModule{agent: a})

	// Handle help responses
	a.reactive.OnModule("help_response", &HelpResponseHandlerModule{agent: a})

	// Handle status updates
	a.reactive.OnModule("status_update", &StatusUpdateHandlerModule{agent: a})
}

// startHeartbeat sends periodic status updates.
func (a *Agent) startHeartbeat(ctx context.Context) {
	ticker := time.NewTicker(a.config.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if a.network != nil {
				status := a.GetStatus()
				_ = a.network.BroadcastStatus(a.id, status)
			}
		case <-ctx.Done():
			return
		}
	}
}

// AgentMessage represents communication between agents.
type AgentMessage struct {
	From      string      `json:"from"`
	To        string      `json:"to"`
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp time.Time   `json:"timestamp"`
}

// HelpRequest represents a request for assistance.
type HelpRequest struct {
	RequestID string        `json:"request_id"`
	From      string        `json:"from"`
	Task      interface{}   `json:"task"`
	Timeout   time.Duration `json:"timeout"`
}

// HelpResponse represents a response to a help request.
type HelpResponse struct {
	RequestID string      `json:"request_id"`
	From      string      `json:"from"`
	Response  interface{} `json:"response"`
}

// AgentNetwork manages communication between multiple agents.
type AgentNetwork struct {
	// agents maps agent IDs to agent instances
	agents map[string]*Agent

	// eventBus handles network-wide event distribution
	eventBus *workflows.EventBus

	// pendingHelp tracks outstanding help requests
	pendingHelp map[string]chan interface{}

	// config holds network configuration
	config NetworkConfig

	// mu protects concurrent access
	mu sync.RWMutex

	// helpMu protects help request map
	helpMu sync.RWMutex
}

// NetworkConfig configures agent network behavior.
type NetworkConfig struct {
	MaxAgents         int
	HelpTimeout       time.Duration
	EnableDiscovery   bool
	EnableLoadBalance bool
}

// DefaultNetworkConfig returns sensible defaults.
func DefaultNetworkConfig() NetworkConfig {
	return NetworkConfig{
		MaxAgents:         100,
		HelpTimeout:       30 * time.Second,
		EnableDiscovery:   true,
		EnableLoadBalance: true,
	}
}

// NewAgentNetwork creates a new agent communication network.
func NewAgentNetwork() *AgentNetwork {
	eventBus := workflows.NewEventBus(workflows.DefaultEventBusConfig())

	return &AgentNetwork{
		agents:      make(map[string]*Agent),
		eventBus:    eventBus,
		pendingHelp: make(map[string]chan interface{}),
		config:      DefaultNetworkConfig(),
	}
}

// Start initializes the agent network.
func (an *AgentNetwork) Start(ctx context.Context) error {
	return an.eventBus.Start(ctx)
}

// Stop terminates the agent network.
func (an *AgentNetwork) Stop() error {
	return an.eventBus.Stop()
}

// AddAgent connects an agent to the network.
func (an *AgentNetwork) AddAgent(agent *Agent) error {
	an.mu.Lock()
	defer an.mu.Unlock()

	if len(an.agents) >= an.config.MaxAgents {
		return fmt.Errorf("network at maximum capacity (%d agents)", an.config.MaxAgents)
	}

	an.agents[agent.GetID()] = agent

	// Connect agent's reactive workflow to network event bus
	agent.reactive.WithEventBus(an.eventBus)

	return nil
}

// RemoveAgent disconnects an agent from the network.
func (an *AgentNetwork) RemoveAgent(agentID string) error {
	an.mu.Lock()
	defer an.mu.Unlock()

	delete(an.agents, agentID)
	return nil
}

// SendMessage routes a message to a specific agent.
func (an *AgentNetwork) SendMessage(message AgentMessage) error {
	event := workflows.Event{
		ID:   fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Type: "agent_message",
		Data: message,
		Context: map[string]interface{}{
			"target_agent": message.To,
		},
	}

	return an.eventBus.Emit(event)
}

// BroadcastMessage sends a message to all agents.
func (an *AgentNetwork) BroadcastMessage(message AgentMessage) error {
	event := workflows.Event{
		ID:   fmt.Sprintf("broadcast_%d", time.Now().UnixNano()),
		Type: "agent_message",
		Data: message,
	}

	return an.eventBus.Broadcast(event)
}

// BroadcastStatus sends status updates to all agents.
func (an *AgentNetwork) BroadcastStatus(agentID string, status AgentStatus) error {
	event := workflows.Event{
		ID:   fmt.Sprintf("status_%s_%d", agentID, time.Now().UnixNano()),
		Type: "status_update",
		Data: map[string]interface{}{
			"agent_id": agentID,
			"status":   status,
		},
	}

	return an.eventBus.Broadcast(event)
}

// RequestHelp broadcasts a help request and waits for responses.
func (an *AgentNetwork) RequestHelp(request HelpRequest) (interface{}, error) {
	// Create response channel
	responseChan := make(chan interface{}, 1)

	an.helpMu.Lock()
	an.pendingHelp[request.RequestID] = responseChan
	an.helpMu.Unlock()

	defer func() {
		an.helpMu.Lock()
		delete(an.pendingHelp, request.RequestID)
		an.helpMu.Unlock()
	}()

	// Broadcast help request
	event := workflows.Event{
		ID:   request.RequestID,
		Type: "help_request",
		Data: request,
	}

	err := an.eventBus.Broadcast(event)
	if err != nil {
		return nil, err
	}

	// Wait for response
	select {
	case response := <-responseChan:
		return response, nil
	case <-time.After(request.Timeout):
		return nil, fmt.Errorf("help request timeout")
	}
}

// OfferHelp sends a response to a help request.
func (an *AgentNetwork) OfferHelp(response HelpResponse) error {
	an.helpMu.RLock()
	responseChan, exists := an.pendingHelp[response.RequestID]
	an.helpMu.RUnlock()

	if !exists {
		return fmt.Errorf("no pending help request found: %s", response.RequestID)
	}

	select {
	case responseChan <- response.Response:
		return nil
	default:
		return fmt.Errorf("response channel full")
	}
}

// GetAgents returns all connected agents.
func (an *AgentNetwork) GetAgents() map[string]*Agent {
	an.mu.RLock()
	defer an.mu.RUnlock()

	// Return copy to prevent external modification
	agents := make(map[string]*Agent)
	for id, agent := range an.agents {
		agents[id] = agent
	}

	return agents
}

// FindAvailableAgent finds an agent that can handle a task.
func (an *AgentNetwork) FindAvailableAgent(taskType string) *Agent {
	an.mu.RLock()
	defer an.mu.RUnlock()

	var bestAgent *Agent
	var lowestLoad = 1.1 // Start above maximum possible load

	for _, agent := range an.agents {
		status := agent.GetStatus()

		// Skip offline or error agents
		if status.State == AgentStateOffline || status.State == AgentStateError {
			continue
		}

		// Skip overloaded agents
		if status.State == AgentStateOverloaded {
			continue
		}

		// Check if agent has required capability
		hasCapability := false
		for _, capability := range status.Capabilities {
			if capability == taskType || capability == "general" {
				hasCapability = true
				break
			}
		}

		if !hasCapability {
			continue
		}

		// Select agent with lowest load
		if status.Load < lowestLoad {
			lowestLoad = status.Load
			bestAgent = agent
		}
	}

	return bestAgent
}

// Handler modules for built-in agent communication

// MessageHandlerModule handles incoming agent messages.
type MessageHandlerModule struct {
	agent *Agent
}

func (m *MessageHandlerModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Convert map[string]any to map[string]interface{} for internal processing
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := m.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}

	return anyResult, nil
}

func (m *MessageHandlerModule) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	message, ok := event.Data.(AgentMessage)
	if !ok {
		return nil, fmt.Errorf("invalid message type")
	}

	// Only process messages intended for this agent or broadcasts
	if message.To != m.agent.GetID() && message.To != "broadcast" {
		return map[string]interface{}{"processed": false}, nil
	}

	// TODO: Implement custom message handling logic
	// For now, just log the message

	return map[string]interface{}{
		"processed": true,
		"message":   message,
	}, nil
}

func (m *MessageHandlerModule) GetSignature() core.Signature {
	inputs := []core.InputField{
		{Field: core.NewField("event")},
	}
	outputs := []core.OutputField{
		{Field: core.NewField("processed")},
	}
	return core.NewSignature(inputs, outputs)
}

func (m *MessageHandlerModule) SetSignature(signature core.Signature) {
	// No-op for handler modules
}

func (m *MessageHandlerModule) SetLLM(llm core.LLM) {
	// No-op for handler modules
}

func (m *MessageHandlerModule) Clone() core.Module {
	return &MessageHandlerModule{agent: m.agent}
}

// HelpRequestHandlerModule handles help requests from other agents.
type HelpRequestHandlerModule struct {
	agent *Agent
}

func (m *HelpRequestHandlerModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	result, err := m.processInternal(ctx, interfaceInputs)
	if err != nil {
		return nil, err
	}

	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}
	return anyResult, nil
}

func (m *HelpRequestHandlerModule) processInternal(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	request, ok := event.Data.(HelpRequest)
	if !ok {
		return nil, fmt.Errorf("invalid help request type")
	}

	// Don't respond to our own requests
	if request.From == m.agent.GetID() {
		return map[string]interface{}{"processed": false}, nil
	}

	// Check if agent can help based on current load and capabilities
	status := m.agent.GetStatus()
	if status.State == AgentStateOverloaded || status.Load > 0.8 {
		return map[string]interface{}{"processed": false}, nil
	}

	// TODO: Implement actual help logic based on task type
	// For now, provide a generic response
	response := fmt.Sprintf("Agent %s acknowledges help request", m.agent.GetID())

	err := m.agent.OfferHelp(request.RequestID, response)
	if err != nil {
		return nil, fmt.Errorf("failed to offer help: %w", err)
	}

	return map[string]interface{}{
		"processed": true,
		"helped":    true,
	}, nil
}

func (m *HelpRequestHandlerModule) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (m *HelpRequestHandlerModule) SetSignature(signature core.Signature) {}
func (m *HelpRequestHandlerModule) SetLLM(llm core.LLM)                   {}

func (m *HelpRequestHandlerModule) Clone() core.Module {
	return &HelpRequestHandlerModule{agent: m.agent}
}

// HelpResponseHandlerModule handles responses to help requests.
type HelpResponseHandlerModule struct {
	agent *Agent
}

func (m *HelpResponseHandlerModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	response, ok := event.Data.(HelpResponse)
	if !ok {
		return nil, fmt.Errorf("invalid help response type")
	}

	// TODO: Handle help response
	// This would typically update agent state or continue workflow

	return map[string]any{
		"processed": true,
		"response":  response,
	}, nil
}

func (m *HelpResponseHandlerModule) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (m *HelpResponseHandlerModule) SetSignature(signature core.Signature) {}
func (m *HelpResponseHandlerModule) SetLLM(llm core.LLM)                   {}

func (m *HelpResponseHandlerModule) Clone() core.Module {
	return &HelpResponseHandlerModule{agent: m.agent}
}

// StatusUpdateHandlerModule handles status updates from other agents.
type StatusUpdateHandlerModule struct {
	agent *Agent
}

func (m *StatusUpdateHandlerModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	event, ok := inputs["event"].(workflows.Event)
	if !ok {
		return nil, fmt.Errorf("invalid event type")
	}

	data, ok := event.Data.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid status update data")
	}

	agentID, _ := data["agent_id"].(string)

	// Don't process our own status updates
	if agentID == m.agent.GetID() {
		return map[string]any{"processed": false}, nil
	}

	// TODO: Handle peer status updates
	// This could be used for load balancing, health monitoring, etc.

	return map[string]any{
		"processed": true,
		"peer_id":   agentID,
	}, nil
}

func (m *StatusUpdateHandlerModule) GetSignature() core.Signature {
	inputs := []core.InputField{{Field: core.NewField("event")}}
	outputs := []core.OutputField{{Field: core.NewField("processed")}}
	return core.NewSignature(inputs, outputs)
}

func (m *StatusUpdateHandlerModule) SetSignature(signature core.Signature) {}
func (m *StatusUpdateHandlerModule) SetLLM(llm core.LLM)                   {}

func (m *StatusUpdateHandlerModule) Clone() core.Module {
	return &StatusUpdateHandlerModule{agent: m.agent}
}
