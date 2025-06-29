package workflows

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// Event represents a reactive event that can trigger workflow execution.
type Event struct {
	// ID uniquely identifies this event instance
	ID string `json:"id"`
	
	// Type categorizes the event (e.g., "pr_created", "user_message")
	Type string `json:"type"`
	
	// Data contains the event payload
	Data interface{} `json:"data"`
	
	// Priority affects event processing order (1=highest, 10=lowest)
	Priority int `json:"priority"`
	
	// Timestamp when the event was created
	Timestamp time.Time `json:"timestamp"`
	
	// Context provides additional metadata
	Context map[string]interface{} `json:"context"`
	
	// Source identifies where the event originated
	Source string `json:"source"`
	
	// CorrelationID links related events
	CorrelationID string `json:"correlation_id"`
}

// EventHandler defines the function signature for handling events.
type EventHandler func(ctx context.Context, event Event) error

// EventFilter determines if an event should be processed.
type EventFilter func(event Event) bool

// EventTransformer modifies events before processing.
type EventTransformer func(event Event) Event

// BackpressureStrategy defines how to handle event overflow.
type BackpressureStrategy int

const (
	BackpressureBlock BackpressureStrategy = iota
	BackpressureDropOldest
	BackpressureDropNewest
	BackpressureDropLowest
)

// EventBus provides centralized event distribution for agent communication.
type EventBus struct {
	// subscribers maps event types to their handlers
	subscribers map[string][]EventHandler
	
	// filters apply to all events before distribution
	filters []EventFilter
	
	// transformers modify events before distribution
	transformers []EventTransformer
	
	// eventChan buffers incoming events
	eventChan chan Event
	
	// responseChan handles request-response patterns
	responseChan map[string]chan interface{}
	
	// config holds bus configuration
	config EventBusConfig
	
	// mu protects concurrent access
	mu sync.RWMutex
	
	// responseMu protects response channel map
	responseMu sync.RWMutex
	
	// running indicates if the bus is active
	running bool
	
	// shutdown signals bus termination
	shutdown chan struct{}
}

// EventBusConfig configures the event bus behavior.
type EventBusConfig struct {
	BufferSize         int
	BackpressureStrategy BackpressureStrategy
	MaxHandlers        int
	HandlerTimeout     time.Duration
	EnablePersistence  bool
	PersistenceStore   EventStore
}

// EventStore interface for persisting events.
type EventStore interface {
	Store(event Event) error
	Retrieve(id string) (Event, error)
	List(filter EventFilter) ([]Event, error)
	Delete(id string) error
}

// DefaultEventBusConfig returns sensible defaults.
func DefaultEventBusConfig() EventBusConfig {
	return EventBusConfig{
		BufferSize:           1000,
		BackpressureStrategy: BackpressureDropOldest,
		MaxHandlers:          100,
		HandlerTimeout:       30 * time.Second,
		EnablePersistence:    false,
	}
}

// NewEventBus creates a new event bus for agent communication.
func NewEventBus(config EventBusConfig) *EventBus {
	return &EventBus{
		subscribers:  make(map[string][]EventHandler),
		filters:      make([]EventFilter, 0),
		transformers: make([]EventTransformer, 0),
		eventChan:    make(chan Event, config.BufferSize),
		responseChan: make(map[string]chan interface{}),
		config:       config,
		shutdown:     make(chan struct{}),
	}
}

// Start begins event processing on the bus.
func (eb *EventBus) Start(ctx context.Context) error {
	eb.mu.Lock()
	if eb.running {
		eb.mu.Unlock()
		return fmt.Errorf("event bus is already running")
	}
	eb.running = true
	eb.mu.Unlock()

	go eb.processEvents(ctx)
	return nil
}

// Stop terminates event processing.
func (eb *EventBus) Stop() error {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	if !eb.running {
		return fmt.Errorf("event bus is not running")
	}
	
	close(eb.shutdown)
	eb.running = false
	return nil
}

// Subscribe registers a handler for events of a specific type.
func (eb *EventBus) Subscribe(eventType string, handler EventHandler) error {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	if len(eb.subscribers[eventType]) >= eb.config.MaxHandlers {
		return fmt.Errorf("maximum handlers reached for event type: %s", eventType)
	}
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
	return nil
}

// Unsubscribe removes handlers for an event type.
func (eb *EventBus) Unsubscribe(eventType string) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	delete(eb.subscribers, eventType)
}

// Emit publishes an event to the bus.
func (eb *EventBus) Emit(event Event) error {
	// Apply transformers
	for _, transformer := range eb.transformers {
		event = transformer(event)
	}
	
	// Apply filters
	for _, filter := range eb.filters {
		if !filter(event) {
			return nil // Event filtered out
		}
	}
	
	// Persist if enabled
	if eb.config.EnablePersistence && eb.config.PersistenceStore != nil {
		if err := eb.config.PersistenceStore.Store(event); err != nil {
			return fmt.Errorf("failed to persist event: %w", err)
		}
	}
	
	// Try to send event, handle backpressure
	select {
	case eb.eventChan <- event:
		return nil
	default:
		return eb.handleBackpressure(event)
	}
}

// Request sends an event and waits for a response.
func (eb *EventBus) Request(event Event, timeout time.Duration) (interface{}, error) {
	responseID := event.ID + "_response"
	responseChan := make(chan interface{}, 1)
	
	eb.responseMu.Lock()
	eb.responseChan[responseID] = responseChan
	eb.responseMu.Unlock()
	
	defer func() {
		eb.responseMu.Lock()
		delete(eb.responseChan, responseID)
		eb.responseMu.Unlock()
	}()
	
	if err := eb.Emit(event); err != nil {
		return nil, err
	}
	
	select {
	case response := <-responseChan:
		return response, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("request timeout after %v", timeout)
	}
}

// Respond sends a response to a request event.
func (eb *EventBus) Respond(requestID string, response interface{}) error {
	responseID := requestID + "_response"
	
	eb.responseMu.RLock()
	responseChan, exists := eb.responseChan[responseID]
	eb.responseMu.RUnlock()
	
	if !exists {
		return fmt.Errorf("no pending request found for ID: %s", requestID)
	}
	
	select {
	case responseChan <- response:
		return nil
	default:
		return fmt.Errorf("response channel full for request: %s", requestID)
	}
}

// Broadcast sends an event to all subscribers regardless of type.
func (eb *EventBus) Broadcast(event Event) error {
	event.Type = "broadcast"
	return eb.Emit(event)
}

// AddFilter adds an event filter to the bus.
func (eb *EventBus) AddFilter(filter EventFilter) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.filters = append(eb.filters, filter)
}

// AddTransformer adds an event transformer to the bus.
func (eb *EventBus) AddTransformer(transformer EventTransformer) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.transformers = append(eb.transformers, transformer)
}

// processEvents handles incoming events from the channel.
func (eb *EventBus) processEvents(ctx context.Context) {
	for {
		select {
		case event := <-eb.eventChan:
			eb.handleEvent(ctx, event)
		case <-eb.shutdown:
			return
		case <-ctx.Done():
			return
		}
	}
}

// handleEvent distributes an event to its subscribers.
func (eb *EventBus) handleEvent(ctx context.Context, event Event) {
	eb.mu.RLock()
	handlers := eb.subscribers[event.Type]
	broadcastHandlers := eb.subscribers["broadcast"]
	eb.mu.RUnlock()
	
	// Handle specific event type subscribers
	for _, handler := range handlers {
		go eb.executeHandler(ctx, handler, event)
	}
	
	// Handle broadcast subscribers
	if event.Type != "broadcast" {
		for _, handler := range broadcastHandlers {
			go eb.executeHandler(ctx, handler, event)
		}
	}
}

// executeHandler runs a handler with timeout protection.
func (eb *EventBus) executeHandler(ctx context.Context, handler EventHandler, event Event) {
	handlerCtx, cancel := context.WithTimeout(ctx, eb.config.HandlerTimeout)
	defer cancel()
	
	done := make(chan error, 1)
	go func() {
		done <- handler(handlerCtx, event)
	}()
	
	select {
	case err := <-done:
		if err != nil {
			// TODO: Add proper logging
			fmt.Printf("Event handler error: %v\n", err)
		}
	case <-handlerCtx.Done():
		// TODO: Add proper logging
		fmt.Printf("Event handler timeout for event: %s\n", event.ID)
	}
}

// handleBackpressure manages event overflow based on strategy.
func (eb *EventBus) handleBackpressure(event Event) error {
	switch eb.config.BackpressureStrategy {
	case BackpressureBlock:
		eb.eventChan <- event // This will block
		return nil
		
	case BackpressureDropOldest:
		// Try to drain oldest event
		select {
		case <-eb.eventChan:
		default:
		}
		eb.eventChan <- event
		return nil
		
	case BackpressureDropNewest:
		return fmt.Errorf("event dropped due to backpressure")
		
	case BackpressureDropLowest:
		// Find and drop lowest priority event in buffer
		return eb.dropLowestPriorityEvent(event)
		
	default:
		return fmt.Errorf("unknown backpressure strategy")
	}
}

// dropLowestPriorityEvent implements priority-based dropping.
func (eb *EventBus) dropLowestPriorityEvent(newEvent Event) error {
	// This is a simplified implementation
	// In practice, you'd need a priority queue for efficient implementation
	
	// For now, just drop the new event if buffer is full
	select {
	case eb.eventChan <- newEvent:
		return nil
	default:
		return fmt.Errorf("event dropped due to backpressure (priority)")
	}
}

// ReactiveWorkflow enables event-driven workflow execution.
type ReactiveWorkflow struct {
	// handlers maps event types to workflow handlers
	handlers map[string]Workflow
	
	// eventBus handles event distribution
	eventBus *EventBus
	
	// memory provides state persistence
	memory agents.Memory
	
	// config holds reactive workflow configuration
	config ReactiveWorkflowConfig
	
	// mu protects concurrent access
	mu sync.RWMutex
}

// ReactiveWorkflowConfig configures reactive workflow behavior.
type ReactiveWorkflowConfig struct {
	AutoStart          bool
	DefaultTimeout     time.Duration
	MaxConcurrentRuns  int
	EnableTracing      bool
	EnableMetrics      bool
}

// DefaultReactiveWorkflowConfig returns sensible defaults.
func DefaultReactiveWorkflowConfig() ReactiveWorkflowConfig {
	return ReactiveWorkflowConfig{
		AutoStart:         true,
		DefaultTimeout:    60 * time.Second,
		MaxConcurrentRuns: 10,
		EnableTracing:     true,
		EnableMetrics:     false,
	}
}

// NewReactiveWorkflow creates a new reactive workflow.
func NewReactiveWorkflow(memory agents.Memory) *ReactiveWorkflow {
	if memory == nil {
		memory = agents.NewInMemoryStore()
	}
	
	eventBus := NewEventBus(DefaultEventBusConfig())
	
	rw := &ReactiveWorkflow{
		handlers: make(map[string]Workflow),
		eventBus: eventBus,
		memory:   memory,
		config:   DefaultReactiveWorkflowConfig(),
	}
	
	return rw
}

// On registers a workflow to handle events of a specific type.
func (rw *ReactiveWorkflow) On(eventType string, workflow Workflow) *ReactiveWorkflow {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	
	rw.handlers[eventType] = workflow
	
	// Register handler with event bus
	_ = rw.eventBus.Subscribe(eventType, rw.createEventHandler(workflow))
	
	return rw
}

// OnModule is a convenience method to register a single module as a workflow.
func (rw *ReactiveWorkflow) OnModule(eventType string, module core.Module) *ReactiveWorkflow {
	workflow := NewChainWorkflow(rw.memory)
	step := &Step{
		ID:     eventType + "_step",
		Module: module,
	}
	_ = workflow.AddStep(step)
	
	return rw.On(eventType, workflow)
}

// WithEventBus allows using a custom event bus.
func (rw *ReactiveWorkflow) WithEventBus(eventBus *EventBus) *ReactiveWorkflow {
	rw.eventBus = eventBus
	
	// Re-register all handlers with new bus
	for eventType, workflow := range rw.handlers {
		_ = rw.eventBus.Subscribe(eventType, rw.createEventHandler(workflow))
	}
	
	return rw
}

// WithConfig sets custom configuration.
func (rw *ReactiveWorkflow) WithConfig(config ReactiveWorkflowConfig) *ReactiveWorkflow {
	rw.config = config
	return rw
}

// WithFilter adds an event filter.
func (rw *ReactiveWorkflow) WithFilter(filter EventFilter) *ReactiveWorkflow {
	rw.eventBus.AddFilter(filter)
	return rw
}

// WithTransformer adds an event transformer.
func (rw *ReactiveWorkflow) WithTransformer(transformer EventTransformer) *ReactiveWorkflow {
	rw.eventBus.AddTransformer(transformer)
	return rw
}

// Start begins reactive event processing.
func (rw *ReactiveWorkflow) Start(ctx context.Context) error {
	return rw.eventBus.Start(ctx)
}

// Stop terminates reactive event processing.
func (rw *ReactiveWorkflow) Stop() error {
	return rw.eventBus.Stop()
}

// Emit publishes an event to trigger workflows.
func (rw *ReactiveWorkflow) Emit(event Event) error {
	return rw.eventBus.Emit(event)
}

// Request sends an event and waits for a response.
func (rw *ReactiveWorkflow) Request(event Event, timeout time.Duration) (interface{}, error) {
	return rw.eventBus.Request(event, timeout)
}

// Respond sends a response to a request event.
func (rw *ReactiveWorkflow) Respond(requestID string, response interface{}) error {
	return rw.eventBus.Respond(requestID, response)
}

// GetEventBus returns the underlying event bus for advanced usage.
func (rw *ReactiveWorkflow) GetEventBus() *EventBus {
	return rw.eventBus
}

// createEventHandler wraps a workflow as an event handler.
func (rw *ReactiveWorkflow) createEventHandler(workflow Workflow) EventHandler {
	return func(ctx context.Context, event Event) error {
		// Convert event data to workflow inputs
		inputs := map[string]interface{}{
			"event":      event,
			"event_type": event.Type,
			"event_data": event.Data,
			"timestamp":  event.Timestamp,
			"context":    event.Context,
		}
		
		// Execute workflow with timeout
		workflowCtx, cancel := context.WithTimeout(ctx, rw.config.DefaultTimeout)
		defer cancel()
		
		_, err := workflow.Execute(workflowCtx, inputs)
		return err
	}
}