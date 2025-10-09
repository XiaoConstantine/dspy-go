package communication

import (
	"encoding/json"
	"testing"
)

// ============================================================================
// Part Tests
// ============================================================================

func TestNewTextPart(t *testing.T) {
	part := NewTextPart("hello world")

	if part.Type != "text" {
		t.Errorf("expected type 'text', got '%s'", part.Type)
	}
	if part.Text != "hello world" {
		t.Errorf("expected text 'hello world', got '%s'", part.Text)
	}
}

func TestNewFilePart(t *testing.T) {
	part := NewFilePart("https://example.com/image.png", "image/png")

	if part.Type != "file" {
		t.Errorf("expected type 'file', got '%s'", part.Type)
	}
	if part.File == nil {
		t.Fatal("expected file to be set")
	}
	if part.File.URI != "https://example.com/image.png" {
		t.Errorf("expected URI 'https://example.com/image.png', got '%s'", part.File.URI)
	}
	if part.File.MimeType != "image/png" {
		t.Errorf("expected MIME type 'image/png', got '%s'", part.File.MimeType)
	}
}

func TestNewDataPart(t *testing.T) {
	data := map[string]interface{}{
		"key1": "value1",
		"key2": 42,
	}
	part := NewDataPart(data)

	if part.Type != "data" {
		t.Errorf("expected type 'data', got '%s'", part.Type)
	}
	if part.Data["key1"] != "value1" {
		t.Errorf("expected data['key1'] to be 'value1'")
	}
	if part.Data["key2"] != 42 {
		t.Errorf("expected data['key2'] to be 42")
	}
}

func TestPartJSONSerialization(t *testing.T) {
	tests := []struct {
		name string
		part Part
	}{
		{
			name: "text part",
			part: NewTextPart("test"),
		},
		{
			name: "file part with URI",
			part: NewFilePart("https://example.com/file.pdf", "application/pdf"),
		},
		{
			name: "data part",
			part: NewDataPart(map[string]interface{}{"foo": "bar"}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Serialize
			data, err := json.Marshal(tt.part)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			// Deserialize
			var decoded Part
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			// Verify type preserved
			if decoded.Type != tt.part.Type {
				t.Errorf("expected type '%s', got '%s'", tt.part.Type, decoded.Type)
			}
		})
	}
}

// ============================================================================
// Message Tests
// ============================================================================

func TestNewUserMessage(t *testing.T) {
	msg := NewUserMessage("What is the weather?")

	if msg.Role != RoleUser {
		t.Errorf("expected role 'user', got '%s'", msg.Role)
	}
	if len(msg.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Parts))
	}
	if msg.Parts[0].Text != "What is the weather?" {
		t.Errorf("unexpected text: %s", msg.Parts[0].Text)
	}
	if msg.MessageID == "" {
		t.Error("expected non-empty message ID")
	}
}

func TestNewAgentMessage(t *testing.T) {
	msg := NewAgentMessage("The weather is sunny")

	if msg.Role != RoleAgent {
		t.Errorf("expected role 'agent', got '%s'", msg.Role)
	}
	if len(msg.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Parts))
	}
}

func TestMessageWithContext(t *testing.T) {
	msg := NewUserMessage("test").WithContext("context-123")

	if msg.ContextID != "context-123" {
		t.Errorf("expected context ID 'context-123', got '%s'", msg.ContextID)
	}
}

func TestMessageAddPart(t *testing.T) {
	msg := NewUserMessage("first part")
	msg.AddPart(NewTextPart("second part"))

	if len(msg.Parts) != 2 {
		t.Errorf("expected 2 parts, got %d", len(msg.Parts))
	}
	if msg.Parts[1].Text != "second part" {
		t.Errorf("unexpected text in second part: %s", msg.Parts[1].Text)
	}
}

func TestMessageJSONSerialization(t *testing.T) {
	original := NewUserMessage("test message").
		WithContext("ctx-1").
		AddPart(NewTextPart("additional"))

	// Serialize
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	// Deserialize
	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	// Verify
	if decoded.Role != original.Role {
		t.Errorf("role mismatch")
	}
	if decoded.ContextID != original.ContextID {
		t.Errorf("context ID mismatch")
	}
	if len(decoded.Parts) != len(original.Parts) {
		t.Errorf("parts count mismatch")
	}
}

// ============================================================================
// Task Tests
// ============================================================================

func TestNewTask(t *testing.T) {
	task := NewTask()

	if task.ID == "" {
		t.Error("expected non-empty task ID")
	}
	if task.Status.State != TaskStateSubmitted {
		t.Errorf("expected state 'submitted', got '%s'", task.Status.State)
	}
	if task.Status.Timestamp == "" {
		t.Error("expected non-empty timestamp")
	}
}

func TestTaskUpdateStatus(t *testing.T) {
	task := NewTask()

	task.UpdateStatus(TaskStateWorking)
	if task.Status.State != TaskStateWorking {
		t.Errorf("expected state 'working', got '%s'", task.Status.State)
	}

	task.UpdateStatus(TaskStateCompleted)
	if task.Status.State != TaskStateCompleted {
		t.Errorf("expected state 'completed', got '%s'", task.Status.State)
	}
}

func TestTaskStateIsTerminal(t *testing.T) {
	tests := []struct {
		state    TaskState
		terminal bool
	}{
		{TaskStateSubmitted, false},
		{TaskStateWorking, false},
		{TaskStateCompleted, true},
		{TaskStateFailed, true},
		{TaskStateInputRequired, false},
		{TaskStateAuthRequired, false},
	}

	for _, tt := range tests {
		t.Run(string(tt.state), func(t *testing.T) {
			if got := tt.state.IsTerminal(); got != tt.terminal {
				t.Errorf("IsTerminal() = %v, want %v", got, tt.terminal)
			}
		})
	}
}

func TestTaskAddArtifact(t *testing.T) {
	task := NewTask()
	artifact := NewArtifact(NewTextPart("result"))

	task.AddArtifact(artifact)

	if len(task.Artifacts) != 1 {
		t.Errorf("expected 1 artifact, got %d", len(task.Artifacts))
	}
	if task.Artifacts[0].ArtifactID != artifact.ArtifactID {
		t.Error("artifact ID mismatch")
	}
}

// ============================================================================
// Artifact Tests
// ============================================================================

func TestNewArtifact(t *testing.T) {
	part1 := NewTextPart("part 1")
	part2 := NewTextPart("part 2")

	artifact := NewArtifact(part1, part2)

	if artifact.ArtifactID == "" {
		t.Error("expected non-empty artifact ID")
	}
	if len(artifact.Parts) != 2 {
		t.Errorf("expected 2 parts, got %d", len(artifact.Parts))
	}
}

func TestNewArtifactWithMetadata(t *testing.T) {
	metadata := map[string]interface{}{
		"source": "test",
		"version": 1,
	}
	artifact := NewArtifactWithMetadata(metadata, NewTextPart("test"))

	if artifact.Metadata["source"] != "test" {
		t.Error("metadata not preserved")
	}
	if artifact.Metadata["version"] != 1 {
		t.Error("metadata not preserved")
	}
}

// ============================================================================
// Event Tests
// ============================================================================

func TestNewTaskStatusUpdateEvent(t *testing.T) {
	status := NewTaskStatus(TaskStateWorking)
	event := NewTaskStatusUpdateEvent("task-123", status, false)

	if event.TaskID != "task-123" {
		t.Errorf("expected task ID 'task-123', got '%s'", event.TaskID)
	}
	if event.Status.State != TaskStateWorking {
		t.Errorf("expected state 'working', got '%s'", event.Status.State)
	}
	if event.Final {
		t.Error("expected Final to be false")
	}
}

func TestNewTaskArtifactUpdateEvent(t *testing.T) {
	artifact := NewArtifact(NewTextPart("result"))
	event := NewTaskArtifactUpdateEvent("task-123", artifact, true)

	if event.TaskID != "task-123" {
		t.Errorf("expected task ID 'task-123', got '%s'", event.TaskID)
	}
	if !event.LastChunk {
		t.Error("expected LastChunk to be true")
	}
}

// ============================================================================
// AgentCard Tests
// ============================================================================

func TestNewCapability(t *testing.T) {
	cap := NewCapability("search", "Search the web", "function")

	if cap.Name != "search" {
		t.Errorf("expected name 'search', got '%s'", cap.Name)
	}
	if cap.Type != "function" {
		t.Errorf("expected type 'function', got '%s'", cap.Type)
	}
}

func TestCapabilityWithSchema(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type": "string",
			},
		},
	}

	cap := NewCapability("search", "Search", "function").WithSchema(schema)

	if cap.Schema == nil {
		t.Fatal("expected schema to be set")
	}
	if cap.Schema["type"] != "object" {
		t.Error("schema not preserved correctly")
	}
}

// ============================================================================
// JSON-RPC Tests
// ============================================================================

func TestNewJSONRPCRequest(t *testing.T) {
	params := map[string]interface{}{
		"message": map[string]interface{}{
			"text": "hello",
		},
	}

	req := NewJSONRPCRequest("sendMessage", params)

	if req.JSONRPC != "2.0" {
		t.Errorf("expected JSONRPC '2.0', got '%s'", req.JSONRPC)
	}
	if req.Method != "sendMessage" {
		t.Errorf("expected method 'sendMessage', got '%s'", req.Method)
	}
	if req.ID == nil {
		t.Error("expected non-nil ID")
	}
}

func TestNewJSONRPCResponse(t *testing.T) {
	result := map[string]interface{}{"success": true}
	resp := NewJSONRPCResponse("req-1", result)

	if resp.JSONRPC != "2.0" {
		t.Errorf("expected JSONRPC '2.0', got '%s'", resp.JSONRPC)
	}
	if resp.Error != nil {
		t.Error("expected no error")
	}
	if resp.Result == nil {
		t.Error("expected result to be set")
	}
}

func TestNewJSONRPCError(t *testing.T) {
	resp := NewJSONRPCError("req-1", RPCErrorCodeInvalidParams, "invalid parameters")

	if resp.Error == nil {
		t.Fatal("expected error to be set")
	}
	if resp.Error.Code != RPCErrorCodeInvalidParams {
		t.Errorf("expected error code %d, got %d", RPCErrorCodeInvalidParams, resp.Error.Code)
	}
	if resp.Error.Message != "invalid parameters" {
		t.Errorf("unexpected error message: %s", resp.Error.Message)
	}
}

func TestRPCErrorImplementsError(t *testing.T) {
	var err error = NewRPCError(123, "test error")

	if err.Error() != "test error" {
		t.Errorf("unexpected error string: %s", err.Error())
	}
}

// ============================================================================
// Integration Tests
// ============================================================================

func TestFullMessageRoundTrip(t *testing.T) {
	// Create a complex message
	original := NewMessage(RoleUser,
		NewTextPart("What is AI?"),
		NewFilePart("https://example.com/doc.pdf", "application/pdf"),
	).WithContext("conversation-1")

	// Serialize to JSON
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	// Deserialize
	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	// Verify structure
	if decoded.Role != original.Role {
		t.Error("role not preserved")
	}
	if decoded.ContextID != original.ContextID {
		t.Error("context ID not preserved")
	}
	if len(decoded.Parts) != 2 {
		t.Errorf("expected 2 parts, got %d", len(decoded.Parts))
	}
}

func TestTaskLifecycle(t *testing.T) {
	// Create task
	task := NewTask()

	// Transition through states
	task.UpdateStatus(TaskStateWorking)
	artifact := NewArtifact(NewTextPart("intermediate result"))
	task.AddArtifact(artifact)

	task.UpdateStatus(TaskStateCompleted)
	finalArtifact := NewArtifact(NewTextPart("final result"))
	task.AddArtifact(finalArtifact)

	// Verify final state
	if task.Status.State != TaskStateCompleted {
		t.Error("task should be completed")
	}
	if len(task.Artifacts) != 2 {
		t.Errorf("expected 2 artifacts, got %d", len(task.Artifacts))
	}
	if !task.Status.State.IsTerminal() {
		t.Error("completed state should be terminal")
	}
}

// ============================================================================
// Additional Helper Tests
// ============================================================================

func TestNewFilePartFromBytes(t *testing.T) {
	part := NewFilePartFromBytes("base64encodeddata", "image/png")

	if part.Type != "file" {
		t.Errorf("expected type 'file', got '%s'", part.Type)
	}
	if part.File == nil {
		t.Fatal("expected file to be set")
	}
	if part.File.Bytes != "base64encodeddata" {
		t.Error("bytes not preserved")
	}
	if part.File.MimeType != "image/png" {
		t.Error("MIME type not preserved")
	}
}

func TestTaskStatusWithMessage(t *testing.T) {
	status := NewTaskStatus(TaskStateWorking)
	msg := NewAgentMessage("Still working...")

	updated := status.WithMessage(msg)

	if updated.Message == nil {
		t.Fatal("expected message to be set")
	}
	if updated.Message.Parts[0].Text != "Still working..." {
		t.Error("message not preserved")
	}
}

func TestRPCErrorWithData(t *testing.T) {
	err := NewRPCError(500, "Internal error")
	data := map[string]interface{}{"details": "Something broke"}

	updated := err.WithData(data)

	if updated.Data == nil {
		t.Fatal("expected data to be set")
	}
	if dataMap, ok := updated.Data.(map[string]interface{}); ok {
		if dataMap["details"] != "Something broke" {
			t.Error("data not preserved")
		}
	} else {
		t.Error("data should be map")
	}
}
