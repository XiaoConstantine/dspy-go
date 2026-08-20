package communication

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ============================================================================
// Test Helpers
// ============================================================================

type flushSignalRecorder struct {
	*httptest.ResponseRecorder
	flushed chan struct{}
}

func (r *flushSignalRecorder) Flush() {
	r.ResponseRecorder.Flush()
	select {
	case r.flushed <- struct{}{}:
	default:
	}
}

// testAgent is a simple agent for server testing.
type testAgent struct {
	name     string
	response map[string]any
	tools    []core.Tool
	delay    time.Duration
	err      error
}

func (a *testAgent) Execute(ctx context.Context, input map[string]any) (map[string]any, error) {
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

type taskContextKey struct{}

type taskContextAgent struct {
	contexts chan context.Context
}

func (a *taskContextAgent) Execute(ctx context.Context, _ map[string]any) (map[string]any, error) {
	a.contexts <- ctx
	<-ctx.Done()
	return nil, ctx.Err()
}

func (a *taskContextAgent) GetCapabilities() []core.Tool {
	return nil
}

func (a *taskContextAgent) GetMemory() agents.Memory {
	return nil
}

type terminalRaceAgent struct {
	started  chan struct{}
	canceled chan struct{}
	release  chan struct{}
	output   map[string]any
	err      error
}

func (a *terminalRaceAgent) Execute(ctx context.Context, _ map[string]any) (map[string]any, error) {
	close(a.started)
	<-ctx.Done()
	close(a.canceled)
	<-a.release
	return a.output, a.err
}

func (a *terminalRaceAgent) GetCapabilities() []core.Tool {
	return nil
}

func (a *terminalRaceAgent) GetMemory() agents.Memory {
	return nil
}

// createTestServer creates a server with a test agent.
func createTestServer(t *testing.T, agent agents.Agent) *Server {
	if agent == nil {
		agent = &testAgent{
			name: "test-agent",
			response: map[string]any{
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
	agent := &testAgent{name: "test", response: map[string]any{"answer": "ok"}}

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
	agent := &testAgent{name: "test", response: map[string]any{"answer": "ok"}}

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

func TestServerExternalShutdownUnblocksStart(t *testing.T) {
	testutil.CheckGoroutineLeaks(t)

	server := createTestServer(t, nil)
	server.server.Addr = "127.0.0.1:0"

	serveStarted := make(chan struct{})
	server.server.BaseContext = func(net.Listener) context.Context {
		close(serveStarted)
		return context.Background()
	}

	startCtx, cancelStart := context.WithCancel(context.Background())
	defer cancelStart()
	startDone := make(chan error, 1)
	go func() {
		startDone <- server.Start(startCtx)
	}()

	select {
	case <-serveStarted:
	case err := <-startDone:
		t.Fatalf("Start returned before serving: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("Start did not begin serving")
	}

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelShutdown()
	if err := server.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}

	select {
	case err := <-startDone:
		if err != nil {
			t.Fatalf("Start() error after external Shutdown = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("external Shutdown did not unblock Start")
	}

	server.mu.Lock()
	run := server.run
	server.mu.Unlock()
	if run == nil {
		t.Fatal("server run was not initialized")
	}
	select {
	case <-run.cleanupDone:
	default:
		t.Fatal("Shutdown returned before the cleanup loop stopped")
	}
}

func TestWaitForServerShutdownJoinsRunAfterError(t *testing.T) {
	shutdownErr := errors.New("shutdown failed")
	run := &serverRun{
		done:        make(chan struct{}),
		cleanupDone: make(chan struct{}),
	}
	forceCloseCalled := make(chan struct{})
	result := make(chan error, 1)
	go func() {
		result <- waitForServerShutdown(
			run,
			func() error { return shutdownErr },
			func() error {
				close(forceCloseCalled)
				return nil
			},
		)
	}()

	select {
	case <-forceCloseCalled:
	case <-time.After(2 * time.Second):
		t.Fatal("shutdown error did not trigger a forced close")
	}
	select {
	case err := <-result:
		t.Fatalf("waitForServerShutdown returned before run.done: %v", err)
	default:
	}

	close(run.done)
	select {
	case err := <-result:
		t.Fatalf("waitForServerShutdown returned before cleanupDone: %v", err)
	default:
	}

	close(run.cleanupDone)
	select {
	case err := <-result:
		if !errors.Is(err, shutdownErr) {
			t.Fatalf("waitForServerShutdown() error = %v, want %v", err, shutdownErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("waitForServerShutdown did not return after the run stopped")
	}
}

func TestServerShutdownDuringStart(t *testing.T) {
	server := createTestServer(t, nil)
	server.server.Addr = "127.0.0.1:0"

	baseContextEntered := make(chan struct{})
	releaseBaseContext := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseBaseContext) }) }
	defer release()

	server.server.BaseContext = func(net.Listener) context.Context {
		close(baseContextEntered)
		<-releaseBaseContext
		return context.Background()
	}
	shutdownStarted := make(chan struct{})
	server.server.RegisterOnShutdown(func() { close(shutdownStarted) })

	startCtx, cancelStart := context.WithCancel(context.Background())
	defer cancelStart()
	startDone := make(chan error, 1)
	go func() {
		startDone <- server.Start(startCtx)
	}()

	select {
	case <-baseContextEntered:
	case err := <-startDone:
		t.Fatalf("Start returned during startup: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("Serve did not enter BaseContext")
	}

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelShutdown()
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- server.Shutdown(shutdownCtx)
	}()

	select {
	case <-shutdownStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("Shutdown did not overlap server startup")
	}
	release()

	select {
	case err := <-shutdownDone:
		if err != nil {
			t.Fatalf("Shutdown() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Shutdown blocked during server startup")
	}

	select {
	case err := <-startDone:
		if err != nil {
			t.Fatalf("Start() error after startup shutdown = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Start remained blocked after startup shutdown")
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
		response: map[string]any{"answer": "ok"},
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
		Params: map[string]any{
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
	result, ok := response.Result.(map[string]any)
	if !ok {
		t.Fatal("expected result to be map")
	}

	taskID, ok := result["taskId"].(string)
	if !ok || taskID == "" {
		t.Error("expected non-empty task ID")
	}
}

func TestHandleSendMessageTaskUsesServerLifecycle(t *testing.T) {
	agent := &taskContextAgent{contexts: make(chan context.Context, 1)}
	server := createTestServer(t, agent)
	lifecycleCtx, cancelLifecycle := context.WithCancelCause(context.Background())
	defer cancelLifecycle(context.Canceled)
	server.run = &serverRun{ctx: lifecycleCtx}

	requestCtx := context.WithValue(context.Background(), taskContextKey{}, "request-value")
	requestCtx, cancelRequest := context.WithCancelCause(requestCtx)
	resp := server.handleSendMessage(requestCtx, &JSONRPCRequest{
		ID: "request-1",
		Params: map[string]any{
			"message": map[string]any{
				"role":  string(RoleUser),
				"parts": []any{map[string]any{"type": "text", "text": "work"}},
			},
		},
	})
	if resp.Error != nil {
		t.Fatalf("handleSendMessage() error = %v", resp.Error)
	}
	taskID := resp.Result.(map[string]any)["taskId"].(string)
	sub := server.subscribers.subscribe(taskID)
	defer server.subscribers.unsubscribe(sub)

	var taskCtx context.Context
	select {
	case taskCtx = <-agent.contexts:
	case <-time.After(2 * time.Second):
		t.Fatal("agent execution did not start")
	}
	if got := taskCtx.Value(taskContextKey{}); got != "request-value" {
		t.Fatalf("task context value = %v, want request-value", got)
	}
	if got, ok := pprof.Label(taskCtx, "task_id"); !ok || got != taskID {
		t.Fatalf("task pprof label = %q, %v; want %q, true", got, ok, taskID)
	}

	requestErr := errors.New("request ended")
	cancelRequest(requestErr)
	select {
	case <-taskCtx.Done():
		t.Fatal("HTTP request cancellation stopped the background task")
	default:
	}
	if cause := context.Cause(taskCtx); cause != nil {
		t.Fatalf("task cancellation cause = %v after request cancellation, want nil", cause)
	}

	lifecycleErr := errors.New("server stopped")
	cancelLifecycle(lifecycleErr)
	select {
	case <-taskCtx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("server lifecycle cancellation did not stop the background task")
	}
	if cause := context.Cause(taskCtx); !errors.Is(cause, lifecycleErr) {
		t.Fatalf("task cancellation cause = %v, want %v", cause, lifecycleErr)
	}

	deadline := time.After(2 * time.Second)
	for {
		select {
		case event := <-sub.channel:
			statusEvent, ok := event.(*TaskStatusUpdateEvent)
			if !ok || !statusEvent.Final {
				continue
			}
			if statusEvent.Status.State != TaskStateFailed {
				t.Fatalf("final status = %+v, want failed", statusEvent)
			}
			return
		case <-deadline:
			t.Fatal("task did not publish a final status after lifecycle cancellation")
		}
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
		Params: map[string]any{
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
		Params: map[string]any{
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
		response: map[string]any{"answer": "done"},
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
		Params: map[string]any{
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

func TestCancelTaskWinsOnceAgainstLateAgentResult(t *testing.T) {
	for _, tt := range []struct {
		name   string
		output map[string]any
		err    error
	}{
		{name: "completion", output: map[string]any{"answer": "late result"}},
		{name: "failure", err: errors.New("late failure")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			agent := &terminalRaceAgent{
				started:  make(chan struct{}),
				canceled: make(chan struct{}),
				release:  make(chan struct{}),
				output:   tt.output,
				err:      tt.err,
			}
			server := createTestServer(t, agent)
			task := server.tasks.create()
			taskCtx, cancelTask := context.WithCancel(context.Background())
			defer cancelTask()
			task.setExecutionCancel(cancelTask)

			sub := server.subscribers.subscribe(task.ID)
			defer server.subscribers.unsubscribe(sub)

			processDone := make(chan struct{})
			go func() {
				defer close(processDone)
				server.processTask(taskCtx, task, NewUserMessage("work"))
			}()

			select {
			case <-agent.started:
			case <-time.After(2 * time.Second):
				t.Fatal("agent execution did not start")
			}

			cancelReq := &JSONRPCRequest{
				ID:     "cancel-1",
				Params: map[string]any{"taskId": task.ID},
			}
			if resp := server.handleCancelTask(context.Background(), cancelReq); resp.Error != nil {
				t.Fatalf("handleCancelTask() error = %v", resp.Error)
			}
			// A repeated cancellation must not publish another terminal event.
			if resp := server.handleCancelTask(context.Background(), cancelReq); resp.Error != nil {
				t.Fatalf("second handleCancelTask() error = %v", resp.Error)
			}

			select {
			case <-agent.canceled:
			case <-time.After(2 * time.Second):
				t.Fatal("task cancellation did not reach the running agent")
			}
			close(agent.release)

			select {
			case <-processDone:
			case <-time.After(2 * time.Second):
				t.Fatal("task processing did not finish")
			}

			status := task.GetStatus()
			if status.State != TaskStateFailed {
				t.Fatalf("task state = %s, want failed", status.State)
			}
			if got := ExtractTextFromMessage(status.Message); got != "Task cancelled by user" {
				t.Fatalf("task message = %q, want cancellation message", got)
			}
			if artifacts := task.GetArtifacts(); len(artifacts) != 0 {
				t.Fatalf("task has %d late artifacts, want 0", len(artifacts))
			}

			finalEvents := 0
			artifactEvents := 0
		drainEvents:
			for {
				select {
				case event := <-sub.channel:
					switch event := event.(type) {
					case *TaskStatusUpdateEvent:
						if event.Final {
							finalEvents++
							if event.Status.State != TaskStateFailed {
								t.Fatalf("final event state = %s, want failed", event.Status.State)
							}
						}
					case *TaskArtifactUpdateEvent:
						artifactEvents++
					}
				default:
					break drainEvents
				}
			}
			if finalEvents != 1 {
				t.Fatalf("final event count = %d, want 1", finalEvents)
			}
			if artifactEvents != 0 {
				t.Fatalf("artifact event count = %d, want 0", artifactEvents)
			}
		})
	}
}

func TestCancelTaskDoesNotOverwriteCompletion(t *testing.T) {
	server := createTestServer(t, nil)
	task := server.tasks.create()
	sub := server.subscribers.subscribe(task.ID)
	defer server.subscribers.unsubscribe(sub)

	server.processTask(context.Background(), task, NewUserMessage("work"))
	cancelReq := &JSONRPCRequest{ID: "cancel-1", Params: map[string]any{"taskId": task.ID}}
	server.handleCancelTask(context.Background(), cancelReq)
	server.handleCancelTask(context.Background(), cancelReq)

	if status := task.GetStatus(); status.State != TaskStateCompleted {
		t.Fatalf("task state = %s, want completed", status.State)
	}
	if artifacts := task.GetArtifacts(); len(artifacts) != 1 {
		t.Fatalf("artifact count = %d, want 1", len(artifacts))
	}

	finalEvents := 0
	artifactEvents := 0
drainEvents:
	for {
		select {
		case event := <-sub.channel:
			switch event := event.(type) {
			case *TaskStatusUpdateEvent:
				if event.Final {
					finalEvents++
					if event.Status.State != TaskStateCompleted {
						t.Fatalf("final event state = %s, want completed", event.Status.State)
					}
				}
			case *TaskArtifactUpdateEvent:
				artifactEvents++
			}
		default:
			break drainEvents
		}
	}
	if finalEvents != 1 {
		t.Fatalf("final event count = %d, want 1", finalEvents)
	}
	if artifactEvents != 1 {
		t.Fatalf("artifact event count = %d, want 1", artifactEvents)
	}
}

func TestHandleRPC_InvalidMethod(t *testing.T) {
	server := createTestServer(t, nil)

	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "unknownMethod",
		Params:  map[string]any{},
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
		response: map[string]any{"answer": "42"},
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

func TestSubscriberRegistry_ReservesCapacityForFinalStatus(t *testing.T) {
	registry := newSubscriberRegistry()
	taskID := "task-123"
	sub := registry.subscribe(taskID)
	defer registry.unsubscribe(sub)

	for range cap(sub.channel) {
		registry.notify(taskID, NewTaskStatusUpdateEvent(taskID, NewTaskStatus(TaskStateWorking), false))
	}
	if got, want := len(sub.channel), cap(sub.channel)-1; got != want {
		t.Fatalf("queued ordinary updates = %d, want %d", got, want)
	}

	final := NewTaskStatusUpdateEvent(taskID, NewTaskStatus(TaskStateCompleted), true)
	registry.notify(taskID, final)
	if got, want := len(sub.channel), cap(sub.channel); got != want {
		t.Fatalf("queued updates after final status = %d, want %d", got, want)
	}

	for i := range cap(sub.channel) {
		event := <-sub.channel
		if i < cap(sub.channel)-1 {
			if event == final {
				t.Fatalf("final status appeared at queue index %d", i)
			}
			continue
		}
		if event != final {
			t.Fatalf("last queued event = %#v, want final status", event)
		}
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

func TestHandleStream_LastArtifactPrecedesFinalStatus(t *testing.T) {
	server := createTestServer(t, nil)
	task := server.tasks.create()
	task.UpdateStatus(TaskStateWorking)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodGet, "/stream/"+task.ID, nil).WithContext(ctx)
	rec := &flushSignalRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		flushed:          make(chan struct{}, 1),
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		server.handleStream(rec, req)
	}()

	// The initial flush proves that the stream subscribed before publication.
	select {
	case <-rec.flushed:
	case <-time.After(2 * time.Second):
		cancel()
		<-done
		t.Fatal("stream did not emit its initial status")
	}

	artifact := NewArtifact(NewTextPart("final result"))
	if !server.transitionTask(task, NewTaskStatus(TaskStateCompleted), &artifact) {
		cancel()
		<-done
		t.Fatal("completed transition was rejected")
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		cancel()
		<-done
		t.Fatal("stream did not close after the final status")
	}

	frames := strings.Split(strings.TrimSpace(rec.Body.String()), "\n\n")
	if len(frames) != 3 {
		t.Fatalf("SSE frame count = %d, want 3; body:\n%s", len(frames), rec.Body.String())
	}
	if !strings.HasPrefix(frames[0], "event: status\n") {
		t.Fatalf("first SSE frame = %q, want initial status", frames[0])
	}

	frameData := func(index int, eventType string) []byte {
		prefix := "event: " + eventType + "\ndata: "
		if !strings.HasPrefix(frames[index], prefix) {
			t.Fatalf("SSE frame %d = %q, want %s event", index, frames[index], eventType)
		}
		return []byte(strings.TrimPrefix(frames[index], prefix))
	}

	var artifactEvent TaskArtifactUpdateEvent
	if err := json.Unmarshal(frameData(1, "artifact"), &artifactEvent); err != nil {
		t.Fatalf("decode artifact event: %v", err)
	}
	if artifactEvent.TaskID != task.ID || !artifactEvent.LastChunk {
		t.Fatalf("artifact event = %+v, want task %q with lastChunk=true", artifactEvent, task.ID)
	}

	var statusEvent TaskStatusUpdateEvent
	if err := json.Unmarshal(frameData(2, "status"), &statusEvent); err != nil {
		t.Fatalf("decode final status event: %v", err)
	}
	if statusEvent.TaskID != task.ID || !statusEvent.Final || statusEvent.Status.State != TaskStateCompleted {
		t.Fatalf("status event = %+v, want final completed status for task %q", statusEvent, task.ID)
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
		response: map[string]any{"answer": "The capital is Paris"},
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
		Params:  map[string]any{"message": msg},
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
	result := sendResp.Result.(map[string]any)
	taskID := result["taskId"].(string)

	// Wait for processing
	time.Sleep(50 * time.Millisecond)

	// 4. Get task result
	getReqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "getTask",
		Params:  map[string]any{"taskId": taskID},
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
