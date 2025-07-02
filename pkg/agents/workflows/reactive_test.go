package workflows

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ReactiveMockModule for testing reactive workflows.
type ReactiveMockModule struct {
	processFunc func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)
	callCount   int
	mu          sync.Mutex
}

func NewReactiveMockModule(processFunc func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)) *ReactiveMockModule {
	return &ReactiveMockModule{processFunc: processFunc}
}

func (m *ReactiveMockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	m.mu.Lock()
	m.callCount++
	m.mu.Unlock()

	// Convert map[string]any to map[string]interface{} for our func
	interfaceInputs := make(map[string]interface{})
	for k, v := range inputs {
		interfaceInputs[k] = v
	}

	var result map[string]interface{}
	var err error

	if m.processFunc != nil {
		result, err = m.processFunc(ctx, interfaceInputs)
	} else {
		result = map[string]interface{}{
			"processed":   true,
			"input_count": len(inputs),
			"result":      "mock_result",
		}
	}

	// Convert back to map[string]any
	anyResult := make(map[string]any)
	for k, v := range result {
		anyResult[k] = v
	}

	return anyResult, err
}

func (m *ReactiveMockModule) GetSignature() core.Signature {
	inputs := []core.InputField{
		{Field: core.NewField("event")},
		{Field: core.NewField("event_type")},
		{Field: core.NewField("event_data")},
		{Field: core.NewField("timestamp")},
		{Field: core.NewField("context")},
	}
	outputs := []core.OutputField{
		{Field: core.NewField("processed")},
		{Field: core.NewField("result")},
	}
	return core.NewSignature(inputs, outputs)
}

func (m *ReactiveMockModule) SetSignature(signature core.Signature) {
	// No-op for mock
}

func (m *ReactiveMockModule) SetLLM(llm core.LLM) {
	// No-op for mock
}

func (m *ReactiveMockModule) GetCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

func (m *ReactiveMockModule) Clone() core.Module {
	return &ReactiveMockModule{processFunc: m.processFunc}
}

func (m *ReactiveMockModule) GetDisplayName() string {
	return "ReactiveMockModule"
}

func (m *ReactiveMockModule) GetModuleType() string {
	return "test"
}

func TestEventBus_BasicFunctionality(t *testing.T) {
	config := DefaultEventBusConfig()
	config.BufferSize = 10

	bus := NewEventBus(config)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the bus
	err := bus.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start event bus: %v", err)
	}
	defer func() { _ = bus.Stop() }()

	// Test event handling
	var receivedEvents []Event
	var mu sync.Mutex

	handler := func(ctx context.Context, event Event) error {
		mu.Lock()
		receivedEvents = append(receivedEvents, event)
		mu.Unlock()
		return nil
	}

	// Subscribe to events
	err = bus.Subscribe("test_event", handler)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}

	// Emit test event
	testEvent := Event{
		ID:        "test_1",
		Type:      "test_event",
		Data:      "test_data",
		Priority:  5,
		Timestamp: time.Now(),
		Context:   map[string]interface{}{"test": true},
	}

	err = bus.Emit(testEvent)
	if err != nil {
		t.Fatalf("Failed to emit event: %v", err)
	}

	// Wait for event processing
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	if len(receivedEvents) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(receivedEvents))
	}

	received := receivedEvents[0]
	if received.ID != testEvent.ID {
		t.Errorf("Expected event ID %s, got %s", testEvent.ID, received.ID)
	}
	if received.Type != testEvent.Type {
		t.Errorf("Expected event type %s, got %s", testEvent.Type, received.Type)
	}
	mu.Unlock()
}

func TestEventBus_RequestResponse(t *testing.T) {
	config := DefaultEventBusConfig()
	bus := NewEventBus(config)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := bus.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start event bus: %v", err)
	}
	defer func() { _ = bus.Stop() }()

	// Set up response handler
	handler := func(ctx context.Context, event Event) error {
		response := fmt.Sprintf("Response to %s", event.Data)
		return bus.Respond(event.ID, response)
	}

	err = bus.Subscribe("request_event", handler)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}

	// Send request
	requestEvent := Event{
		ID:   "request_1",
		Type: "request_event",
		Data: "test_request",
	}

	response, err := bus.Request(requestEvent, 5*time.Second)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	expectedResponse := "Response to test_request"
	if response != expectedResponse {
		t.Errorf("Expected response %s, got %v", expectedResponse, response)
	}
}

func TestEventBus_Filters(t *testing.T) {
	config := DefaultEventBusConfig()
	bus := NewEventBus(config)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := bus.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start event bus: %v", err)
	}
	defer func() { _ = bus.Stop() }()

	var receivedEvents []Event
	var mu sync.Mutex

	handler := func(ctx context.Context, event Event) error {
		mu.Lock()
		receivedEvents = append(receivedEvents, event)
		mu.Unlock()
		return nil
	}

	err = bus.Subscribe("filtered_event", handler)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}

	// Add filter that only allows high priority events
	bus.AddFilter(func(event Event) bool {
		return event.Priority <= 3 // Only high priority (1-3)
	})

	// Emit high priority event (should pass)
	highPriorityEvent := Event{
		ID:       "high_1",
		Type:     "filtered_event",
		Priority: 2,
	}

	err = bus.Emit(highPriorityEvent)
	if err != nil {
		t.Fatalf("Failed to emit high priority event: %v", err)
	}

	// Emit low priority event (should be filtered)
	lowPriorityEvent := Event{
		ID:       "low_1",
		Type:     "filtered_event",
		Priority: 8,
	}

	err = bus.Emit(lowPriorityEvent)
	if err != nil {
		t.Fatalf("Failed to emit low priority event: %v", err)
	}

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	if len(receivedEvents) != 1 {
		t.Fatalf("Expected 1 event (filtered), got %d", len(receivedEvents))
	}

	if receivedEvents[0].ID != "high_1" {
		t.Errorf("Expected high priority event, got %s", receivedEvents[0].ID)
	}
	mu.Unlock()
}

func TestReactiveWorkflow_BasicUsage(t *testing.T) {
	memory := agents.NewInMemoryStore()
	reactive := NewReactiveWorkflow(memory)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create mock module that tracks calls
	var processedEvents []Event
	var mu sync.Mutex

	mockModule := NewReactiveMockModule(func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		mu.Lock()
		if event, ok := inputs["event"].(Event); ok {
			processedEvents = append(processedEvents, event)
		}
		mu.Unlock()

		return map[string]interface{}{
			"processed": true,
			"result":    "event_processed",
		}, nil
	})

	// Register event handler
	reactive.OnModule("user_action", mockModule)

	// Start reactive workflow
	err := reactive.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start reactive workflow: %v", err)
	}
	defer func() { _ = reactive.Stop() }()

	// Emit test event
	testEvent := Event{
		ID:   "action_1",
		Type: "user_action",
		Data: map[string]interface{}{
			"action": "click",
			"target": "button_1",
		},
		Timestamp: time.Now(),
	}

	err = reactive.Emit(testEvent)
	if err != nil {
		t.Fatalf("Failed to emit event: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Verify event was processed
	mu.Lock()
	if len(processedEvents) != 1 {
		t.Fatalf("Expected 1 processed event, got %d", len(processedEvents))
	}

	processed := processedEvents[0]
	if processed.ID != testEvent.ID {
		t.Errorf("Expected event ID %s, got %s", testEvent.ID, processed.ID)
	}
	mu.Unlock()

	// Verify module was called
	if mockModule.GetCallCount() != 1 {
		t.Errorf("Expected module to be called once, got %d calls", mockModule.GetCallCount())
	}
}

func TestReactiveWorkflow_MultipleHandlers(t *testing.T) {
	memory := agents.NewInMemoryStore()
	reactive := NewReactiveWorkflow(memory)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Track processed events for each handler
	var handler1Events, handler2Events []Event
	var mu1, mu2 sync.Mutex

	module1 := NewReactiveMockModule(func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		mu1.Lock()
		if event, ok := inputs["event"].(Event); ok {
			handler1Events = append(handler1Events, event)
		}
		mu1.Unlock()
		return map[string]interface{}{"handler": "1", "processed": true, "result": "handler1_result"}, nil
	})

	module2 := NewReactiveMockModule(func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		mu2.Lock()
		if event, ok := inputs["event"].(Event); ok {
			handler2Events = append(handler2Events, event)
		}
		mu2.Unlock()
		return map[string]interface{}{"handler": "2", "processed": true, "result": "handler2_result"}, nil
	})

	// Register different handlers for different event types
	reactive.OnModule("type_a", module1)
	reactive.OnModule("type_b", module2)

	err := reactive.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start reactive workflow: %v", err)
	}
	defer func() { _ = reactive.Stop() }()

	// Emit events of different types
	eventA := Event{ID: "a1", Type: "type_a", Data: "data_a"}
	eventB := Event{ID: "b1", Type: "type_b", Data: "data_b"}

	err = reactive.Emit(eventA)
	if err != nil {
		t.Fatalf("Failed to emit event A: %v", err)
	}

	err = reactive.Emit(eventB)
	if err != nil {
		t.Fatalf("Failed to emit event B: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Verify correct handlers processed correct events
	mu1.Lock()
	if len(handler1Events) != 1 || handler1Events[0].ID != "a1" {
		t.Errorf("Handler 1 should process event A, got %v", handler1Events)
	}
	mu1.Unlock()

	mu2.Lock()
	if len(handler2Events) != 1 || handler2Events[0].ID != "b1" {
		t.Errorf("Handler 2 should process event B, got %v", handler2Events)
	}
	mu2.Unlock()
}

func TestReactiveWorkflow_EventFiltering(t *testing.T) {
	memory := agents.NewInMemoryStore()
	reactive := NewReactiveWorkflow(memory)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var processedEvents []Event
	var mu sync.Mutex

	mockModule := NewReactiveMockModule(func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		mu.Lock()
		if event, ok := inputs["event"].(Event); ok {
			processedEvents = append(processedEvents, event)
		}
		mu.Unlock()
		return map[string]interface{}{"processed": true, "result": "processed"}, nil
	})

	// Add filter for high priority events only
	reactive.WithFilter(func(event Event) bool {
		return event.Priority <= 5
	})

	reactive.OnModule("filtered_event", mockModule)

	err := reactive.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start reactive workflow: %v", err)
	}
	defer func() { _ = reactive.Stop() }()

	// Emit high priority event (should be processed)
	highPriorityEvent := Event{
		ID:       "high_1",
		Type:     "filtered_event",
		Priority: 3,
	}

	// Emit low priority event (should be filtered out)
	lowPriorityEvent := Event{
		ID:       "low_1",
		Type:     "filtered_event",
		Priority: 8,
	}

	err = reactive.Emit(highPriorityEvent)
	if err != nil {
		t.Fatalf("Failed to emit high priority event: %v", err)
	}

	err = reactive.Emit(lowPriorityEvent)
	if err != nil {
		t.Fatalf("Failed to emit low priority event: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Only high priority event should be processed
	mu.Lock()
	if len(processedEvents) != 1 {
		t.Fatalf("Expected 1 processed event, got %d", len(processedEvents))
	}

	if processedEvents[0].ID != "high_1" {
		t.Errorf("Expected high priority event to be processed, got %s", processedEvents[0].ID)
	}
	mu.Unlock()
}

func TestEventBus_Backpressure(t *testing.T) {
	config := DefaultEventBusConfig()
	config.BufferSize = 2 // Small buffer to test backpressure
	config.BackpressureStrategy = BackpressureDropOldest

	bus := NewEventBus(config)
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Don't start the bus so events accumulate in buffer

	// Fill buffer beyond capacity
	for i := 0; i < 5; i++ {
		event := Event{
			ID:   fmt.Sprintf("event_%d", i),
			Type: "test",
			Data: i,
		}

		err := bus.Emit(event)
		if err != nil {
			t.Logf("Event %d emission result: %v", i, err)
		}
	}

	// Buffer should not be completely full due to backpressure handling
	// This test mainly verifies that backpressure doesn't cause panics
}

func TestReactiveWorkflow_EventTransformation(t *testing.T) {
	memory := agents.NewInMemoryStore()
	reactive := NewReactiveWorkflow(memory)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var receivedData []interface{}
	var mu sync.Mutex

	mockModule := NewReactiveMockModule(func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		mu.Lock()
		if event, ok := inputs["event"].(Event); ok {
			receivedData = append(receivedData, event.Data)
		}
		mu.Unlock()
		return map[string]interface{}{"processed": true, "result": "processed"}, nil
	})

	// Add transformer that modifies event data
	reactive.WithTransformer(func(event Event) Event {
		if str, ok := event.Data.(string); ok {
			event.Data = "transformed_" + str
		}
		return event
	})

	reactive.OnModule("transform_test", mockModule)

	err := reactive.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start reactive workflow: %v", err)
	}
	defer func() { _ = reactive.Stop() }()

	// Emit event with string data
	originalEvent := Event{
		ID:   "transform_1",
		Type: "transform_test",
		Data: "original_data",
	}

	err = reactive.Emit(originalEvent)
	if err != nil {
		t.Fatalf("Failed to emit event: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Verify data was transformed
	mu.Lock()
	if len(receivedData) != 1 {
		t.Fatalf("Expected 1 processed event, got %d", len(receivedData))
	}

	transformedData := receivedData[0].(string)
	expected := "transformed_original_data"
	if transformedData != expected {
		t.Errorf("Expected transformed data %s, got %s", expected, transformedData)
	}
	mu.Unlock()
}
