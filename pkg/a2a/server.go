package a2a

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
)

// ============================================================================
// Server Configuration
// ============================================================================

// ServerConfig holds configuration for the a2a server.
type ServerConfig struct {
	Host        string   // Host address (e.g., "localhost", "0.0.0.0")
	Port        int      // Port number (e.g., 8080)
	Name        string   // Agent name for AgentCard
	Description string   // Agent description for AgentCard
	Version     string   // Agent version
	PathPrefix  string   // Optional path prefix (e.g., "/api/v1")
	MaxTaskAge  Duration // How long to keep completed tasks (default: 1 hour)
}

// Duration is a wrapper around time.Duration for easier JSON unmarshaling.
type Duration time.Duration

// Server exposes a dspy-go agent via the a2a protocol over HTTP.
type Server struct {
	config      ServerConfig
	agent       agents.Agent
	agentCard   AgentCard
	server      *http.Server
	tasks       *taskRegistry
	subscribers *subscriberRegistry
	mu          sync.RWMutex
	running     bool
}

// ============================================================================
// Task Management
// ============================================================================

// taskRegistry manages active and completed tasks.
type taskRegistry struct {
	tasks map[string]*Task
	mu    sync.RWMutex
}

func newTaskRegistry() *taskRegistry {
	return &taskRegistry{
		tasks: make(map[string]*Task),
	}
}

func (r *taskRegistry) create() *Task {
	r.mu.Lock()
	defer r.mu.Unlock()

	task := NewTask()
	r.tasks[task.ID] = task
	return task
}

func (r *taskRegistry) get(id string) (*Task, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	task, ok := r.tasks[id]
	return task, ok
}

func (r *taskRegistry) update(task *Task) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.tasks[task.ID] = task
}

func (r *taskRegistry) delete(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.tasks, id)
}

// ============================================================================
// Subscriber Management (for SSE)
// ============================================================================

// subscriber represents a client subscribed to task updates.
type subscriber struct {
	taskID  string
	channel chan interface{} // Can send TaskStatusUpdateEvent or TaskArtifactUpdateEvent
}

type subscriberRegistry struct {
	subscribers map[string][]*subscriber // taskID -> list of subscribers
	mu          sync.RWMutex
}

func newSubscriberRegistry() *subscriberRegistry {
	return &subscriberRegistry{
		subscribers: make(map[string][]*subscriber),
	}
}

func (r *subscriberRegistry) subscribe(taskID string) *subscriber {
	r.mu.Lock()
	defer r.mu.Unlock()

	sub := &subscriber{
		taskID:  taskID,
		channel: make(chan interface{}, 100), // Buffer to prevent blocking
	}

	r.subscribers[taskID] = append(r.subscribers[taskID], sub)
	return sub
}

func (r *subscriberRegistry) unsubscribe(sub *subscriber) {
	r.mu.Lock()
	defer r.mu.Unlock()

	subs := r.subscribers[sub.taskID]
	for i, s := range subs {
		if s == sub {
			r.subscribers[sub.taskID] = append(subs[:i], subs[i+1:]...)
			close(sub.channel)
			break
		}
	}
}

func (r *subscriberRegistry) notify(taskID string, event interface{}) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for _, sub := range r.subscribers[taskID] {
		select {
		case sub.channel <- event:
		default:
			// Channel full, skip this update
		}
	}
}

// ============================================================================
// Server Creation and Lifecycle
// ============================================================================

// NewServer creates a new a2a server wrapping the given agent.
func NewServer(agent agents.Agent, config ServerConfig) (*Server, error) {
	if agent == nil {
		return nil, fmt.Errorf("agent cannot be nil")
	}

	// Set defaults
	if config.Host == "" {
		config.Host = "localhost"
	}
	if config.Port == 0 {
		config.Port = 8080
	}
	if config.Name == "" {
		config.Name = "dspy-go-agent"
	}
	if config.Version == "" {
		config.Version = "1.0.0"
	}
	if config.MaxTaskAge == 0 {
		config.MaxTaskAge = Duration(time.Hour)
	}

	// Build AgentCard
	baseURL := fmt.Sprintf("http://%s:%d%s", config.Host, config.Port, config.PathPrefix)
	agentCard := AgentCard{
		Name:        config.Name,
		Description: config.Description,
		URL:         baseURL + "/rpc",
		Version:     config.Version,
	}

	// Add capabilities from agent's tools
	if capabilities := agent.GetCapabilities(); len(capabilities) > 0 {
		agentCard.Capabilities = ToolsToCapabilities(capabilities)
	}

	s := &Server{
		config:      config,
		agent:       agent,
		agentCard:   agentCard,
		tasks:       newTaskRegistry(),
		subscribers: newSubscriberRegistry(),
	}

	// Create HTTP server
	mux := http.NewServeMux()
	s.registerHandlers(mux)

	s.server = &http.Server{
		Addr:    fmt.Sprintf("%s:%d", config.Host, config.Port),
		Handler: mux,
	}

	return s, nil
}

// registerHandlers sets up HTTP routes.
func (s *Server) registerHandlers(mux *http.ServeMux) {
	prefix := s.config.PathPrefix

	// AgentCard discovery endpoint
	mux.HandleFunc(prefix+"/.well-known/agent.json", s.handleAgentCard)

	// JSON-RPC endpoint
	mux.HandleFunc(prefix+"/rpc", s.handleRPC)

	// SSE streaming endpoint
	mux.HandleFunc(prefix+"/stream/", s.handleStream)

	// Health check
	mux.HandleFunc(prefix+"/health", s.handleHealth)
}

// Start begins serving the a2a protocol.
func (s *Server) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return fmt.Errorf("server already running")
	}
	s.running = true
	s.mu.Unlock()

	// Start cleanup goroutine for old tasks
	go s.cleanupLoop(ctx)

	// Start server
	errCh := make(chan error, 1)
	go func() {
		if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	fmt.Printf("🚀 a2a server started at %s\n", s.server.Addr)
	fmt.Printf("📋 AgentCard: http://%s%s/.well-known/agent.json\n", s.server.Addr, s.config.PathPrefix)

	// Wait for context cancellation or server error
	select {
	case <-ctx.Done():
		return s.Shutdown(context.Background())
	case err := <-errCh:
		return err
	}
}

// Shutdown gracefully stops the server.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	s.running = false
	return s.server.Shutdown(ctx)
}

// cleanupLoop periodically removes old completed tasks.
func (s *Server) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(15 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.cleanupOldTasks()
		}
	}
}

func (s *Server) cleanupOldTasks() {
	s.tasks.mu.Lock()
	defer s.tasks.mu.Unlock()

	maxAge := time.Duration(s.config.MaxTaskAge)
	now := time.Now()

	for id, task := range s.tasks.tasks {
		status := task.GetStatus()
		if !status.State.IsTerminal() {
			continue
		}

		// Parse timestamp
		ts, err := time.Parse(time.RFC3339, status.Timestamp)
		if err != nil {
			continue
		}

		if now.Sub(ts) > maxAge {
			delete(s.tasks.tasks, id)
		}
	}
}

// ============================================================================
// HTTP Handlers
// ============================================================================

// handleAgentCard serves the agent discovery document.
func (s *Server) handleAgentCard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(s.agentCard); err != nil {
		// Log error but can't change status after writing headers
		fmt.Fprintf(w, `{"error":"encoding failed"}`)
	}
}

// handleHealth is a simple health check endpoint.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"agent":  s.config.Name,
	}); err != nil {
		// Log error but can't change status after writing headers
		fmt.Fprintf(w, `{"error":"encoding failed"}`)
	}
}

// handleRPC processes JSON-RPC requests.
func (s *Server) handleRPC(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse JSON-RPC request
	var req JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		resp := NewJSONRPCError(nil, RPCErrorCodeParseError, "Failed to parse request")
		s.sendJSONResponse(w, resp)
		return
	}

	// Dispatch to handler
	var resp *JSONRPCResponse
	switch req.Method {
	case "sendMessage":
		resp = s.handleSendMessage(r.Context(), &req)
	case "getTask":
		resp = s.handleGetTask(r.Context(), &req)
	case "cancelTask":
		resp = s.handleCancelTask(r.Context(), &req)
	default:
		resp = NewJSONRPCError(req.ID, RPCErrorCodeMethodNotFound, "Method not found")
	}

	s.sendJSONResponse(w, resp)
}

func (s *Server) sendJSONResponse(w http.ResponseWriter, resp *JSONRPCResponse) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		// Log error but can't change status after writing headers
		fmt.Fprintf(w, `{"jsonrpc":"2.0","error":{"code":-32603,"message":"encoding failed"},"id":null}`)
	}
}

// ============================================================================
// JSON-RPC Method Handlers
// ============================================================================

// handleSendMessage processes a sendMessage RPC call.
func (s *Server) handleSendMessage(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
	// Extract message from params
	msgData, ok := req.Params["message"]
	if !ok {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Missing 'message' parameter")
	}

	// Parse message
	msgBytes, _ := json.Marshal(msgData)
	var msg Message
	if err := json.Unmarshal(msgBytes, &msg); err != nil {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Invalid message format")
	}

	// Create task
	task := s.tasks.create()
	task.ContextID = msg.ContextID

	// Process asynchronously
	go s.processTask(ctx, task, &msg)

	// Return task info immediately
	return NewJSONRPCResponse(req.ID, map[string]interface{}{
		"taskId": task.ID,
		"status": task.GetStatus(),
	})
}

// handleGetTask retrieves task status.
func (s *Server) handleGetTask(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
	taskID, ok := req.Params["taskId"].(string)
	if !ok {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Missing or invalid 'taskId'")
	}

	task, ok := s.tasks.get(taskID)
	if !ok {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Task not found")
	}

	return NewJSONRPCResponse(req.ID, task)
}

// handleCancelTask cancels a running task.
func (s *Server) handleCancelTask(ctx context.Context, req *JSONRPCRequest) *JSONRPCResponse {
	taskID, ok := req.Params["taskId"].(string)
	if !ok {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Missing or invalid 'taskId'")
	}

	task, ok := s.tasks.get(taskID)
	if !ok {
		return NewJSONRPCError(req.ID, RPCErrorCodeInvalidParams, "Task not found")
	}

	// Mark as failed
	task.mu.Lock()
	task.Status = NewTaskStatus(TaskStateFailed)
	task.Status.Message = NewAgentMessage("Task cancelled by user")
	task.mu.Unlock()

	s.tasks.update(task)

	// Notify subscribers
	event := NewTaskStatusUpdateEvent(task.ID, task.GetStatus(), true)
	s.subscribers.notify(task.ID, event)

	return NewJSONRPCResponse(req.ID, task)
}

// ============================================================================
// Task Processing
// ============================================================================

// processTask executes the agent and streams results.
func (s *Server) processTask(ctx context.Context, task *Task, msg *Message) {
	// Update to working state
	task.UpdateStatus(TaskStateWorking)
	s.tasks.update(task)
	s.subscribers.notify(task.ID, NewTaskStatusUpdateEvent(task.ID, task.GetStatus(), false))

	// Convert message to agent input
	input, err := MessageToAgentInput(msg)
	if err != nil {
		s.failTask(task, err)
		return
	}

	// Execute agent
	output, err := s.agent.Execute(ctx, input)
	if err != nil {
		s.failTask(task, err)
		return
	}

	// Convert output to artifact
	artifact, err := AgentOutputToArtifact(output)
	if err != nil {
		s.failTask(task, err)
		return
	}

	// Add artifact and complete
	task.AddArtifact(artifact)
	task.UpdateStatus(TaskStateCompleted)
	s.tasks.update(task)

	// Notify subscribers
	s.subscribers.notify(task.ID, NewTaskArtifactUpdateEvent(task.ID, artifact, true))
	s.subscribers.notify(task.ID, NewTaskStatusUpdateEvent(task.ID, task.GetStatus(), true))
}

func (s *Server) failTask(task *Task, err error) {
	task.mu.Lock()
	task.Status = NewTaskStatus(TaskStateFailed)
	task.Status.Message = CreateErrorMessage(err)
	task.mu.Unlock()

	s.tasks.update(task)
	s.subscribers.notify(task.ID, NewTaskStatusUpdateEvent(task.ID, task.GetStatus(), true))
}

// ============================================================================
// SSE Streaming Handler
// ============================================================================

// URL format: /stream/{taskId}.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	// Extract task ID from path
	path := r.URL.Path
	prefix := s.config.PathPrefix + "/stream/"
	if len(path) <= len(prefix) {
		http.Error(w, "Task ID required", http.StatusBadRequest)
		return
	}
	taskID := path[len(prefix):]

	// Verify task exists
	task, ok := s.tasks.get(taskID)
	if !ok {
		http.Error(w, "Task not found", http.StatusNotFound)
		return
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Create flusher
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Subscribe to task updates
	sub := s.subscribers.subscribe(taskID)
	defer s.subscribers.unsubscribe(sub)

	// Send initial task state
	status := task.GetStatus()
	s.sendSSEEvent(w, flusher, "status", status)

	// If task already completed, close immediately
	if status.State.IsTerminal() {
		return
	}

	// Stream updates
	for {
		select {
		case <-r.Context().Done():
			return
		case event, ok := <-sub.channel:
			if !ok {
				return
			}

			// Send event based on type
			switch e := event.(type) {
			case *TaskStatusUpdateEvent:
				s.sendSSEEvent(w, flusher, "status", e)
				if e.Final {
					return
				}
			case *TaskArtifactUpdateEvent:
				s.sendSSEEvent(w, flusher, "artifact", e)
				if e.LastChunk {
					return
				}
			}
		}
	}
}

// sendSSEEvent sends a Server-Sent Event.
func (s *Server) sendSSEEvent(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}

	fmt.Fprintf(w, "event: %s\n", eventType)
	fmt.Fprintf(w, "data: %s\n\n", jsonData)
	flusher.Flush()
}
