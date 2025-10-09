package communication

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ============================================================================
// Test Helpers
// ============================================================================

// testAgent is a simple agent for server testing.
type testAgent struct {
	name     string
	response map[string]interface{}
	tools    []core.Tool
	delay    time.Duration
	err      error
}

func (a *testAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	if a.delay > 0 {
		time.Sleep(a.delay)
	}
	if a.err != nil {
		return nil, a.err
	}
	return a.response, nil
}

func (a *testAgent) GetCapabilities() []core.Tool {
	return a.tools
}

func (a *testAgent) GetMemory() agents.Memory {
	return nil
}

// createTestServer creates a server with a test agent.
func createTestServer(t *testing.T, agent agents.Agent) *Server {
	if agent == nil {
		agent = &testAgent{
			name: "test-agent",
			response: map[string]interface{}{
				"answer": "Test response",
			},
		}
	}

	server, err := NewServer(agent, ServerConfig{
		Host:        "localhost",
		Port:        0, // Use random port for testing
		Name:        "TestAgent",
		Description: "Test agent for unit tests",
		Version:     "1.0.0",
	})
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}

	return server
}

// ============================================================================
// Server Creation Tests
// ============================================================================

func TestNewServer(t *testing.T) {
	agent := &testAgent{name: "test", response: map[string]interface{}{"answer": "ok"}}

	server, err := NewServer(agent, ServerConfig{
		Host: "localhost",
		Port: 8080,
		Name: "TestAgent",
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if server == nil {
		t.Fatal("expected non-nil server")
	}
	if server.agent != agent {
		t.Error("server should wrap the provided agent")
	}
	if server.agentCard.Name != "TestAgent" {
		t.Errorf("expected agent name 'TestAgent', got '%s'", server.agentCard.Name)
	}
}

func TestNewServer_NilAgent(t *testing.T) {
	_, err := NewServer(nil, ServerConfig{})
	if err == nil {
		t.Error("expected error for nil agent")
	}
}

func TestNewServer_Defaults(t *testing.T) {
	agent := &testAgent{name: "test", response: map[string]interface{}{"answer": "ok"}}

	server, err := NewServer(agent, ServerConfig{})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if server.config.Host != "localhost" {
		t.Error("expected default host 'localhost'")
	}
	if server.config.Port != 8080 {
		t.Error("expected default port 8080")
	}
	if server.config.Name != "dspy-go-agent" {
		t.Error("expected default name")
	}
}

func TestNewServer_WithTools(t *testing.T) {
	tool := &mockTool{
		metadata: &core.ToolMetadata{
			Name:        "search",
			Description: "Search tool",
		},
	}

	agent := &testAgent{
		name:     "test",
		response: map[string]interface{}{"answer": "ok"},
		tools:    []core.Tool{tool},
	}

	server, err := NewServer(agent, ServerConfig{Name: "TestAgent"})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(server.agentCard.Capabilities) == 0 {
		t.Error("expected capabilities from agent tools")
	}
}

// ============================================================================
// HTTP Endpoint Tests
// ============================================================================

func TestHandleAgentCard(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/.well-known/agent.json", nil)
	rec := httptest.NewRecorder()

	server.handleAgentCard(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}

	var card AgentCard
	if err := json.NewDecoder(rec.Body).Decode(&card); err != nil {
		t.Fatalf("failed to decode agent card: %v", err)
	}

	if card.Name != "TestAgent" {
		t.Errorf("expected name 'TestAgent', got '%s'", card.Name)
	}
	if card.Version != "1.0.0" {
		t.Errorf("expected version '1.0.0', got '%s'", card.Version)
	}
}

func TestHandleAgentCard_WrongMethod(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodPost, "/.well-known/agent.json", nil)
	rec := httptest.NewRecorder()

	server.handleAgentCard(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", rec.Code)
	}
}

func TestHandleHealth(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	server.handleHealth(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}

	var response map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}

	if response["status"] != "ok" {
		t.Errorf("expected status 'ok', got '%s'", response["status"])
	}
}

// ============================================================================
// JSON-RPC Handler Tests
// ============================================================================

func TestHandleSendMessage(t *testing.T) {
	server := createTestServer(t, nil)

	// Create JSON-RPC request
	msg := NewUserMessage("What is 2+2?")
	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "sendMessage",
		Params: map[string]interface{}{
			"message": msg,
		},
		ID: "test-1",
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error != nil {
		t.Errorf("unexpected error: %v", response.Error.Message)
	}

	// Verify we got a task ID
	result, ok := response.Result.(map[string]interface{})
	if !ok {
		t.Fatal("expected result to be map")
	}

	taskID, ok := result["taskId"].(string)
	if !ok || taskID == "" {
		t.Error("expected non-empty task ID")
	}
}

func TestHandleGetTask(t *testing.T) {
	server := createTestServer(t, nil)

	// First create a task
	task := server.tasks.create()
	task.UpdateStatus(TaskStateCompleted)
	server.tasks.update(task)

	// Now get the task
	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "getTask",
		Params: map[string]interface{}{
			"taskId": task.ID,
		},
		ID: "test-1",
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rec.Code)
	}

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error != nil {
		t.Errorf("unexpected error: %v", response.Error.Message)
	}
}

func TestHandleGetTask_NotFound(t *testing.T) {
	server := createTestServer(t, nil)

	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "getTask",
		Params: map[string]interface{}{
			"taskId": "nonexistent",
		},
		ID: "test-1",
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error == nil {
		t.Error("expected error for non-existent task")
	}
	if response.Error.Code != RPCErrorCodeInvalidParams {
		t.Errorf("expected invalid params error code, got %d", response.Error.Code)
	}
}

func TestHandleCancelTask(t *testing.T) {
	agent := &testAgent{
		name:     "slow-agent",
		response: map[string]interface{}{"answer": "done"},
		delay:    100 * time.Millisecond,
	}
	server := createTestServer(t, agent)

	// Create a task
	task := server.tasks.create()
	task.UpdateStatus(TaskStateWorking)
	server.tasks.update(task)

	// Cancel it
	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "cancelTask",
		Params: map[string]interface{}{
			"taskId": task.ID,
		},
		ID: "test-1",
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error != nil {
		t.Errorf("unexpected error: %v", response.Error.Message)
	}

	// Verify task is now failed
	updatedTask, _ := server.tasks.get(task.ID)
	if updatedTask.Status.State != TaskStateFailed {
		t.Errorf("expected failed state, got %s", updatedTask.Status.State)
	}
}

func TestHandleRPC_InvalidMethod(t *testing.T) {
	server := createTestServer(t, nil)

	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "unknownMethod",
		Params:  map[string]interface{}{},
		ID:      "test-1",
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error == nil {
		t.Error("expected error for unknown method")
	}
	if response.Error.Code != RPCErrorCodeMethodNotFound {
		t.Errorf("expected method not found error, got code %d", response.Error.Code)
	}
}

func TestHandleRPC_InvalidJSON(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader("invalid json"))
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	var response JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if response.Error == nil {
		t.Error("expected parse error")
	}
	if response.Error.Code != RPCErrorCodeParseError {
		t.Errorf("expected parse error code, got %d", response.Error.Code)
	}
}

func TestHandleRPC_WrongMethod(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/rpc", nil)
	rec := httptest.NewRecorder()

	server.handleRPC(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", rec.Code)
	}
}

// ============================================================================
// Task Processing Tests
// ============================================================================

func TestProcessTask_Success(t *testing.T) {
	agent := &testAgent{
		name:     "test",
		response: map[string]interface{}{"answer": "42"},
	}
	server := createTestServer(t, agent)

	task := server.tasks.create()
	msg := NewUserMessage("What is the meaning of life?")

	// Process task
	server.processTask(context.Background(), task, msg)

	// Wait a bit for async processing
	time.Sleep(10 * time.Millisecond)

	// Verify task completed
	updatedTask, _ := server.tasks.get(task.ID)
	if updatedTask.Status.State != TaskStateCompleted {
		t.Errorf("expected completed state, got %s", updatedTask.Status.State)
	}
	if len(updatedTask.Artifacts) != 1 {
		t.Errorf("expected 1 artifact, got %d", len(updatedTask.Artifacts))
	}
}

func TestProcessTask_AgentError(t *testing.T) {
	agent := &testAgent{
		name: "failing-agent",
		err:  fmt.Errorf("agent failed"),
	}
	server := createTestServer(t, agent)

	task := server.tasks.create()
	msg := NewUserMessage("test")

	server.processTask(context.Background(), task, msg)

	// Wait for processing
	time.Sleep(10 * time.Millisecond)

	// Verify task failed
	updatedTask, _ := server.tasks.get(task.ID)
	if updatedTask.Status.State != TaskStateFailed {
		t.Errorf("expected failed state, got %s", updatedTask.Status.State)
	}
}

// ============================================================================
// Task Registry Tests
// ============================================================================

func TestTaskRegistry_CreateAndGet(t *testing.T) {
	registry := newTaskRegistry()

	task := registry.create()
	if task == nil {
		t.Fatal("expected non-nil task")
	}
	if task.ID == "" {
		t.Error("expected non-empty task ID")
	}

	retrieved, ok := registry.get(task.ID)
	if !ok {
		t.Error("task should be retrievable")
	}
	if retrieved.ID != task.ID {
		t.Error("retrieved task should match created task")
	}
}

func TestTaskRegistry_Update(t *testing.T) {
	registry := newTaskRegistry()

	task := registry.create()
	task.UpdateStatus(TaskStateCompleted)
	registry.update(task)

	retrieved, _ := registry.get(task.ID)
	if retrieved.Status.State != TaskStateCompleted {
		t.Error("task update should be persisted")
	}
}

func TestTaskRegistry_Delete(t *testing.T) {
	registry := newTaskRegistry()

	task := registry.create()
	registry.delete(task.ID)

	_, ok := registry.get(task.ID)
	if ok {
		t.Error("deleted task should not be retrievable")
	}
}

// ============================================================================
// Subscriber Registry Tests
// ============================================================================

func TestSubscriberRegistry_SubscribeAndNotify(t *testing.T) {
	registry := newSubscriberRegistry()

	taskID := "task-123"
	sub := registry.subscribe(taskID)

	if sub == nil {
		t.Fatal("expected non-nil subscriber")
	}
	if sub.taskID != taskID {
		t.Error("subscriber should have correct task ID")
	}

	// Notify
	event := NewTaskStatusUpdateEvent(taskID, NewTaskStatus(TaskStateWorking), false)
	registry.notify(taskID, event)

	// Receive event
	select {
	case received := <-sub.channel:
		if received == nil {
			t.Error("expected non-nil event")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("timeout waiting for event")
	}
}

func TestSubscriberRegistry_Unsubscribe(t *testing.T) {
	registry := newSubscriberRegistry()

	sub := registry.subscribe("task-123")
	registry.unsubscribe(sub)

	// Channel should be closed
	_, ok := <-sub.channel
	if ok {
		t.Error("channel should be closed after unsubscribe")
	}
}

func TestSubscriberRegistry_MultipleSubscribers(t *testing.T) {
	registry := newSubscriberRegistry()

	taskID := "task-123"
	sub1 := registry.subscribe(taskID)
	sub2 := registry.subscribe(taskID)

	event := NewTaskStatusUpdateEvent(taskID, NewTaskStatus(TaskStateCompleted), true)
	registry.notify(taskID, event)

	// Both should receive
	select {
	case <-sub1.channel:
	case <-time.After(100 * time.Millisecond):
		t.Error("sub1 should receive event")
	}

	select {
	case <-sub2.channel:
	case <-time.After(100 * time.Millisecond):
		t.Error("sub2 should receive event")
	}
}

// ============================================================================
// SSE Streaming Tests
// ============================================================================

func TestHandleStream_TaskNotFound(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/stream/nonexistent", nil)
	rec := httptest.NewRecorder()

	server.handleStream(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Errorf("expected status 404, got %d", rec.Code)
	}
}

func TestHandleStream_NoTaskID(t *testing.T) {
	server := createTestServer(t, nil)

	req := httptest.NewRequest(http.MethodGet, "/stream/", nil)
	rec := httptest.NewRecorder()

	server.handleStream(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", rec.Code)
	}
}

func TestHandleStream_CompletedTask(t *testing.T) {
	server := createTestServer(t, nil)

	// Create completed task
	task := server.tasks.create()
	task.UpdateStatus(TaskStateCompleted)
	server.tasks.update(task)

	req := httptest.NewRequest(http.MethodGet, "/stream/"+task.ID, nil)
	rec := httptest.NewRecorder()

	server.handleStream(rec, req)

	// Should get SSE headers
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Error("expected SSE content type")
	}

	// Should receive initial status and then close
	body := rec.Body.String()
	if !contains(body, "event: status") {
		t.Error("expected status event")
	}
}

// ============================================================================
// Cleanup Tests
// ============================================================================

func TestCleanupOldTasks(t *testing.T) {
	server := createTestServer(t, nil)
	server.config.MaxTaskAge = Duration(1 * time.Second)

	// Create old completed task (1 hour old)
	oldTask := server.tasks.create()
	oldTask.UpdateStatus(TaskStateCompleted)
	oldTask.Status.Timestamp = time.Now().Add(-1 * time.Hour).Format(time.RFC3339)
	server.tasks.update(oldTask)

	// Create recent task (just completed, timestamp is current)
	recentTask := server.tasks.create()
	recentTask.UpdateStatus(TaskStateCompleted)
	// Ensure timestamp is very recent
	recentTask.Status.Timestamp = time.Now().Format(time.RFC3339)
	server.tasks.update(recentTask)

	// Verify both exist before cleanup
	_, ok := server.tasks.get(oldTask.ID)
	if !ok {
		t.Fatal("old task should exist before cleanup")
	}
	_, ok = server.tasks.get(recentTask.ID)
	if !ok {
		t.Fatal("recent task should exist before cleanup")
	}

	// Run cleanup
	server.cleanupOldTasks()

	// Old task should be removed (1 hour > 1 second)
	_, ok = server.tasks.get(oldTask.ID)
	if ok {
		t.Error("old task should be cleaned up")
	}

	// Recent task should remain (0 seconds < 1 second)
	_, ok = server.tasks.get(recentTask.ID)
	if !ok {
		t.Error("recent task should not be cleaned up")
	}
}

func TestCleanupOldTasks_OnlyTerminalTasks(t *testing.T) {
	server := createTestServer(t, nil)
	server.config.MaxTaskAge = Duration(1 * time.Millisecond)

	// Create old but still working task
	task := server.tasks.create()
	task.UpdateStatus(TaskStateWorking)
	task.Status.Timestamp = time.Now().Add(-1 * time.Hour).Format(time.RFC3339)
	server.tasks.update(task)

	server.cleanupOldTasks()

	// Should NOT be removed (not terminal)
	_, ok := server.tasks.get(task.ID)
	if !ok {
		t.Error("working task should not be cleaned up")
	}
}

// ============================================================================
// Integration Tests
// ============================================================================

func TestServerFullFlow(t *testing.T) {
	agent := &testAgent{
		name:     "test",
		response: map[string]interface{}{"answer": "The capital is Paris"},
	}
	server := createTestServer(t, agent)

	// 1. Check health
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	server.handleHealth(rec, req)
	if rec.Code != http.StatusOK {
		t.Error("health check failed")
	}

	// 2. Get AgentCard
	req = httptest.NewRequest(http.MethodGet, "/.well-known/agent.json", nil)
	rec = httptest.NewRecorder()
	server.handleAgentCard(rec, req)
	if rec.Code != http.StatusOK {
		t.Error("agent card retrieval failed")
	}

	// 3. Send message
	msg := NewUserMessage("What is the capital of France?")
	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "sendMessage",
		Params:  map[string]interface{}{"message": msg},
		ID:      "test-1",
	}
	body, _ := json.Marshal(reqBody)
	req = httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec = httptest.NewRecorder()
	server.handleRPC(rec, req)

	var sendResp JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&sendResp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	result := sendResp.Result.(map[string]interface{})
	taskID := result["taskId"].(string)

	// Wait for processing
	time.Sleep(50 * time.Millisecond)

	// 4. Get task result
	getReqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "getTask",
		Params:  map[string]interface{}{"taskId": taskID},
		ID:      "test-2",
	}
	body, _ = json.Marshal(getReqBody)
	req = httptest.NewRequest(http.MethodPost, "/rpc", strings.NewReader(string(body)))
	rec = httptest.NewRecorder()
	server.handleRPC(rec, req)

	var getResp JSONRPCResponse
	if err := json.NewDecoder(rec.Body).Decode(&getResp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if getResp.Error != nil {
		t.Errorf("task retrieval failed: %v", getResp.Error.Message)
	}
}
