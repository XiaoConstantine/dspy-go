// Package a2a implements Google's Agent-to-Agent (a2a) protocol for dspy-go.
// This package provides JSON-RPC over HTTP transport for interoperability with
// Python ADK agents and other a2a-compatible agents.
package communication

import (
	"encoding/json"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// Part Types - Building blocks of messages
// ============================================================================

// Part represents a piece of content in a message.
// Parts can be text, files, or structured data.
type Part struct {
	Type     string                 `json:"type"` // "text", "file", or "data"
	Text     string                 `json:"text,omitempty"`
	File     *FilePart              `json:"file,omitempty"`
	Data     map[string]interface{} `json:"data,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// FilePart represents a file attachment.
// Files can be referenced by URI or embedded as base64-encoded bytes.
type FilePart struct {
	URI      string `json:"uri,omitempty"`   // URI reference to file
	Bytes    string `json:"bytes,omitempty"` // base64-encoded file content
	MimeType string `json:"mimeType"`        // MIME type (e.g., "image/png")
}

// NewTextPart creates a text part.
func NewTextPart(text string) Part {
	return Part{
		Type: "text",
		Text: text,
	}
}

// NewTextPartWithMetadata creates a text part with metadata.
func NewTextPartWithMetadata(text string, metadata map[string]interface{}) Part {
	return Part{
		Type:     "text",
		Text:     text,
		Metadata: metadata,
	}
}

// NewFilePart creates a file part from a URI.
func NewFilePart(uri, mimeType string) Part {
	return Part{
		Type: "file",
		File: &FilePart{
			URI:      uri,
			MimeType: mimeType,
		},
	}
}

// NewFilePartFromBytes creates a file part from base64-encoded bytes.
func NewFilePartFromBytes(bytes, mimeType string) Part {
	return Part{
		Type: "file",
		File: &FilePart{
			Bytes:    bytes,
			MimeType: mimeType,
		},
	}
}

// NewDataPart creates a structured data part.
func NewDataPart(data map[string]interface{}) Part {
	return Part{
		Type: "data",
		Data: data,
	}
}

// ============================================================================
// Message Types - Communication units
// ============================================================================

// Role represents the sender of a message.
type Role string

const (
	RoleUser  Role = "user"
	RoleAgent Role = "agent"
)

// Message represents a message in the a2a protocol.
// Messages contain one or more parts and can maintain context across multiple exchanges.
type Message struct {
	MessageID string                 `json:"messageId"`
	Role      Role                   `json:"role"`
	Parts     []Part                 `json:"parts"`
	ContextID string                 `json:"contextId,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// NewMessage creates a new message with the given role and parts.
func NewMessage(role Role, parts ...Part) *Message {
	return &Message{
		MessageID: generateID(),
		Role:      role,
		Parts:     parts,
	}
}

// NewUserMessage creates a user message with text.
func NewUserMessage(text string) *Message {
	return &Message{
		MessageID: generateID(),
		Role:      RoleUser,
		Parts:     []Part{NewTextPart(text)},
	}
}

// NewAgentMessage creates an agent message with text.
func NewAgentMessage(text string) *Message {
	return &Message{
		MessageID: generateID(),
		Role:      RoleAgent,
		Parts:     []Part{NewTextPart(text)},
	}
}

// WithContext sets the context ID for this message.
func (m *Message) WithContext(contextID string) *Message {
	m.ContextID = contextID
	return m
}

// AddPart adds a part to the message.
func (m *Message) AddPart(part Part) *Message {
	m.Parts = append(m.Parts, part)
	return m
}

// ============================================================================
// Task Types - Execution tracking
// ============================================================================

// TaskState represents the current state of a task.
type TaskState string

const (
	TaskStateSubmitted     TaskState = "submitted"      // Task has been submitted
	TaskStateWorking       TaskState = "working"        // Task is being processed
	TaskStateCompleted     TaskState = "completed"      // Task completed successfully
	TaskStateFailed        TaskState = "failed"         // Task failed with error
	TaskStateInputRequired TaskState = "input_required" // Task requires user input
	TaskStateAuthRequired  TaskState = "auth_required"  // Task requires authentication
)

// IsTerminal returns true if the state is terminal (completed or failed).
func (s TaskState) IsTerminal() bool {
	return s == TaskStateCompleted || s == TaskStateFailed
}

// TaskStatus represents the current status of a task.
type TaskStatus struct {
	State     TaskState `json:"state"`
	Message   *Message  `json:"message,omitempty"`
	Timestamp string    `json:"timestamp"` // RFC3339 format
}

// NewTaskStatus creates a new task status with the current timestamp.
func NewTaskStatus(state TaskState) TaskStatus {
	return TaskStatus{
		State:     state,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
}

// WithMessage adds a message to the task status.
func (ts TaskStatus) WithMessage(msg *Message) TaskStatus {
	ts.Message = msg
	return ts
}

// Task represents a running or completed task.
type Task struct {
	ID        string                 `json:"id"`
	ContextID string                 `json:"contextId,omitempty"`
	Status    TaskStatus             `json:"status"`
	History   []Message              `json:"history,omitempty"`
	Artifacts []Artifact             `json:"artifacts,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	mu        sync.RWMutex           `json:"-"` // Protects concurrent access
}

// NewTask creates a new task with submitted status.
func NewTask() *Task {
	return &Task{
		ID:     generateID(),
		Status: NewTaskStatus(TaskStateSubmitted),
	}
}

// UpdateStatus updates the task status.
func (t *Task) UpdateStatus(state TaskState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Status = NewTaskStatus(state)
}

// AddArtifact adds an artifact to the task.
func (t *Task) AddArtifact(artifact Artifact) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Artifacts = append(t.Artifacts, artifact)
}

// GetStatus returns a copy of the current status (thread-safe).
func (t *Task) GetStatus() TaskStatus {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.Status
}

// GetArtifacts returns a copy of the artifacts (thread-safe).
func (t *Task) GetArtifacts() []Artifact {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return append([]Artifact{}, t.Artifacts...)
}

// ============================================================================
// Artifact Types - Task outputs
// ============================================================================

// Artifact represents output from a task.
type Artifact struct {
	ArtifactID string                 `json:"artifactId"`
	Parts      []Part                 `json:"parts"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// NewArtifact creates a new artifact with the given parts.
func NewArtifact(parts ...Part) Artifact {
	return Artifact{
		ArtifactID: generateID(),
		Parts:      parts,
	}
}

// NewArtifactWithMetadata creates a new artifact with parts and metadata.
func NewArtifactWithMetadata(metadata map[string]interface{}, parts ...Part) Artifact {
	return Artifact{
		ArtifactID: generateID(),
		Parts:      parts,
		Metadata:   metadata,
	}
}

// ============================================================================
// Event Types - Streaming updates
// ============================================================================

// TaskStatusUpdateEvent is sent when a task's status changes.
type TaskStatusUpdateEvent struct {
	TaskID    string                 `json:"taskId"`
	Status    TaskStatus             `json:"status"`
	ContextID string                 `json:"contextId,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Final     bool                   `json:"final"` // true if this is the final status update
}

// NewTaskStatusUpdateEvent creates a status update event.
func NewTaskStatusUpdateEvent(taskID string, status TaskStatus, final bool) *TaskStatusUpdateEvent {
	return &TaskStatusUpdateEvent{
		TaskID: taskID,
		Status: status,
		Final:  final,
	}
}

// TaskArtifactUpdateEvent is sent when a task produces an artifact.
type TaskArtifactUpdateEvent struct {
	TaskID    string                 `json:"taskId"`
	Artifact  Artifact               `json:"artifact"`
	ContextID string                 `json:"contextId,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	LastChunk bool                   `json:"lastChunk"` // true if this is the last artifact chunk
	Append    bool                   `json:"append"`    // true if this should be appended to previous artifact
}

// NewTaskArtifactUpdateEvent creates an artifact update event.
func NewTaskArtifactUpdateEvent(taskID string, artifact Artifact, lastChunk bool) *TaskArtifactUpdateEvent {
	return &TaskArtifactUpdateEvent{
		TaskID:    taskID,
		Artifact:  artifact,
		LastChunk: lastChunk,
	}
}

// ============================================================================
// AgentCard - Agent discovery
// ============================================================================

// AgentCard describes an agent's capabilities and endpoint.
// This is served at /.well-known/agent.json for discovery.
type AgentCard struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	URL          string                 `json:"url"` // RPC endpoint URL
	Version      string                 `json:"version"`
	Capabilities []Capability           `json:"capabilities,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Capability describes a specific capability of an agent.
type Capability struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Type        string                 `json:"type"` // "function", "tool", "service"
	Schema      map[string]interface{} `json:"schema,omitempty"`
}

// NewCapability creates a new capability.
func NewCapability(name, description, capType string) Capability {
	return Capability{
		Name:        name,
		Description: description,
		Type:        capType,
	}
}

// WithSchema adds a schema to the capability.
func (c Capability) WithSchema(schema map[string]interface{}) Capability {
	c.Schema = schema
	return c
}

// ============================================================================
// JSON-RPC Types - Transport protocol
// ============================================================================

// JSONRPCRequest represents a JSON-RPC 2.0 request.
type JSONRPCRequest struct {
	JSONRPC string                 `json:"jsonrpc"` // Must be "2.0"
	Method  string                 `json:"method"`
	Params  map[string]interface{} `json:"params,omitempty"`
	ID      interface{}            `json:"id"` // string, number, or null
}

// NewJSONRPCRequest creates a new JSON-RPC request.
func NewJSONRPCRequest(method string, params map[string]interface{}) *JSONRPCRequest {
	return &JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      generateID(),
	}
}

// JSONRPCResponse represents a JSON-RPC 2.0 response.
type JSONRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"` // Must be "2.0"
	Result  interface{} `json:"result,omitempty"`
	Error   *RPCError   `json:"error,omitempty"`
	ID      interface{} `json:"id"`
}

// NewJSONRPCResponse creates a successful JSON-RPC response.
func NewJSONRPCResponse(id interface{}, result interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		Result:  result,
		ID:      id,
	}
}

// NewJSONRPCError creates an error JSON-RPC response.
func NewJSONRPCError(id interface{}, code int, message string) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		Error: &RPCError{
			Code:    code,
			Message: message,
		},
		ID: id,
	}
}

// RPCError represents a JSON-RPC error.
type RPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Standard JSON-RPC error codes.
const (
	RPCErrorCodeParseError     = -32700
	RPCErrorCodeInvalidRequest = -32600
	RPCErrorCodeMethodNotFound = -32601
	RPCErrorCodeInvalidParams  = -32602
	RPCErrorCodeInternalError  = -32603
)

// NewRPCError creates a new RPC error.
func NewRPCError(code int, message string) *RPCError {
	return &RPCError{
		Code:    code,
		Message: message,
	}
}

// WithData adds data to the RPC error.
func (e *RPCError) WithData(data interface{}) *RPCError {
	e.Data = data
	return e
}

// Error implements the error interface.
func (e *RPCError) Error() string {
	return e.Message
}

// ============================================================================
// Helper functions
// ============================================================================

// generateID generates a unique ID for messages, tasks, and artifacts.
// Uses atomic counter for thread-safety - in production, use UUID or similar.
var idCounter atomic.Int64

func generateID() string {
	count := idCounter.Add(1)
	return time.Now().Format("20060102150405") + "-" + strconv.FormatInt(count, 10)
}

// MarshalJSON provides custom JSON marshaling for Part to ensure proper structure.
func (p Part) MarshalJSON() ([]byte, error) {
	type Alias Part
	return json.Marshal(&struct {
		*Alias
	}{
		Alias: (*Alias)(&p),
	})
}
