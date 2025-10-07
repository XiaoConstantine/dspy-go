package a2a

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// ============================================================================
// Message to Agent Input Tests
// ============================================================================

func TestMessageToAgentInput_Simple(t *testing.T) {
	msg := NewUserMessage("What is the capital of France?")

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	question, ok := input["question"].(string)
	if !ok {
		t.Fatal("expected 'question' field to be string")
	}
	if question != "What is the capital of France?" {
		t.Errorf("unexpected question: %s", question)
	}
}

func TestMessageToAgentInput_MultipleParts(t *testing.T) {
	msg := NewMessage(RoleUser,
		NewTextPartWithMetadata("Paris", map[string]interface{}{"field": "city"}),
		NewTextPartWithMetadata("France", map[string]interface{}{"field": "country"}),
	)

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if input["city"] != "Paris" {
		t.Errorf("expected city 'Paris', got '%v'", input["city"])
	}
	if input["country"] != "France" {
		t.Errorf("expected country 'France', got '%v'", input["country"])
	}
}

func TestMessageToAgentInput_WithContext(t *testing.T) {
	msg := NewUserMessage("follow-up question").WithContext("ctx-123")

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	contextID, ok := input["_context_id"].(string)
	if !ok || contextID != "ctx-123" {
		t.Error("context ID not preserved")
	}
}

func TestMessageToAgentInput_WithFiles(t *testing.T) {
	msg := NewMessage(RoleUser,
		NewTextPart("Analyze this image"),
		NewFilePart("https://example.com/image.png", "image/png"),
	)

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	files, ok := input["_files"].([]FilePart)
	if !ok {
		t.Fatal("expected '_files' to be []FilePart")
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
	if files[0].URI != "https://example.com/image.png" {
		t.Error("file URI not preserved")
	}
}

func TestMessageToAgentInput_WithData(t *testing.T) {
	data := map[string]interface{}{
		"temperature": 0.7,
		"maxTokens":   100,
	}
	msg := NewMessage(RoleUser,
		NewTextPart("Generate text"),
		NewDataPart(data),
	)

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if input["_data_temperature"] != 0.7 {
		t.Error("data field 'temperature' not preserved")
	}
	if input["_data_maxTokens"] != 100 {
		t.Error("data field 'maxTokens' not preserved")
	}
}

func TestMessageToAgentInput_NilMessage(t *testing.T) {
	_, err := MessageToAgentInput(nil)
	if err == nil {
		t.Error("expected error for nil message")
	}
}

func TestMessageToAgentInput_EmptyParts(t *testing.T) {
	msg := &Message{
		MessageID: "test",
		Role:      RoleUser,
		Parts:     []Part{},
	}

	_, err := MessageToAgentInput(msg)
	if err == nil {
		t.Error("expected error for message with no parts")
	}
}

// ============================================================================
// Agent Output to Message Tests
// ============================================================================

func TestAgentOutputToMessage_Simple(t *testing.T) {
	output := map[string]interface{}{
		"answer": "The capital of France is Paris",
	}

	msg, err := AgentOutputToMessage(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if msg.Role != RoleAgent {
		t.Errorf("expected role 'agent', got '%s'", msg.Role)
	}
	if len(msg.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(msg.Parts))
	}
	if msg.Parts[0].Type != "text" {
		t.Error("expected text part")
	}
	if msg.Parts[0].Text != "The capital of France is Paris" {
		t.Errorf("unexpected text: %s", msg.Parts[0].Text)
	}
}

func TestAgentOutputToMessage_MultipleFields(t *testing.T) {
	output := map[string]interface{}{
		"answer":     "Paris",
		"confidence": "high",
		"source":     "Wikipedia",
	}

	msg, err := AgentOutputToMessage(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(msg.Parts) != 3 {
		t.Errorf("expected 3 parts, got %d", len(msg.Parts))
	}

	// Verify each part has field metadata
	for _, part := range msg.Parts {
		if part.Metadata == nil {
			t.Error("expected metadata on part")
		}
		if part.Metadata["field"] == nil {
			t.Error("expected 'field' in metadata")
		}
	}
}

func TestAgentOutputToMessage_SkipsInternalFields(t *testing.T) {
	output := map[string]interface{}{
		"answer":      "Paris",
		"_context_id": "ctx-123",
		"_metadata":   map[string]interface{}{"internal": true},
		"_some_field": "should be skipped",
	}

	msg, err := AgentOutputToMessage(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should only have 1 part (answer), internal fields excluded
	if len(msg.Parts) != 1 {
		t.Errorf("expected 1 part (internal fields should be skipped), got %d", len(msg.Parts))
	}

	// But context should be preserved
	if msg.ContextID != "ctx-123" {
		t.Error("context ID should be preserved")
	}
}

func TestAgentOutputToMessage_NilOutput(t *testing.T) {
	_, err := AgentOutputToMessage(nil)
	if err == nil {
		t.Error("expected error for nil output")
	}
}

func TestAgentOutputToMessage_EmptyOutput(t *testing.T) {
	output := map[string]interface{}{
		"_only_internal": "value",
	}

	_, err := AgentOutputToMessage(output)
	if err == nil {
		t.Error("expected error when all fields are internal")
	}
}

// ============================================================================
// Agent Output to Artifact Tests
// ============================================================================

func TestAgentOutputToArtifact_Simple(t *testing.T) {
	output := map[string]interface{}{
		"result": "Success",
	}

	artifact, err := AgentOutputToArtifact(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if artifact.ArtifactID == "" {
		t.Error("expected non-empty artifact ID")
	}
	if len(artifact.Parts) != 1 {
		t.Errorf("expected 1 part, got %d", len(artifact.Parts))
	}
}

func TestAgentOutputToArtifact_WithMetadata(t *testing.T) {
	output := map[string]interface{}{
		"result": "Success",
		"_metadata": map[string]interface{}{
			"version":   "1.0",
			"timestamp": "2024-01-01",
		},
	}

	artifact, err := AgentOutputToArtifact(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if artifact.Metadata == nil {
		t.Fatal("expected metadata to be set")
	}
	if artifact.Metadata["version"] != "1.0" {
		t.Error("metadata not preserved")
	}
}

// ============================================================================
// Round-Trip Conversion Tests
// ============================================================================

func TestRoundTrip_SimpleQuestion(t *testing.T) {
	// Start with agent input
	originalInput := map[string]interface{}{
		"question": "What is 2+2?",
	}

	// Convert to message
	msg := NewUserMessage(originalInput["question"].(string))

	// Convert back to input
	convertedInput, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("conversion failed: %v", err)
	}

	// Verify round-trip
	if convertedInput["question"] != originalInput["question"] {
		t.Error("round-trip conversion failed")
	}
}

func TestRoundTrip_AgentOutput(t *testing.T) {
	// Start with agent output
	originalOutput := map[string]interface{}{
		"answer": "The answer is 4",
	}

	// Convert to message
	msg, err := AgentOutputToMessage(originalOutput)
	if err != nil {
		t.Fatalf("to message failed: %v", err)
	}

	// Extract text back
	text := ExtractTextFromMessage(msg)
	if text != "The answer is 4" {
		t.Errorf("round-trip text extraction failed: got '%s'", text)
	}
}

// ============================================================================
// Tool/Capability Conversion Tests
// ============================================================================

// mockTool is a simple Tool implementation for testing.
type mockTool struct {
	metadata *core.ToolMetadata
}

func (m *mockTool) Name() string {
	if m.metadata != nil {
		return m.metadata.Name
	}
	return ""
}

func (m *mockTool) Description() string {
	if m.metadata != nil {
		return m.metadata.Description
	}
	return ""
}

func (m *mockTool) Metadata() *core.ToolMetadata {
	return m.metadata
}

func (m *mockTool) InputSchema() models.InputSchema {
	if m.metadata != nil {
		return m.metadata.InputSchema
	}
	return models.InputSchema{}
}

func (m *mockTool) CanHandle(ctx context.Context, intent string) bool {
	return true
}

func (m *mockTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{}, nil
}

func (m *mockTool) Validate(params map[string]interface{}) error {
	return nil
}

func TestToolsToCapabilities(t *testing.T) {
	tools := []core.Tool{
		&mockTool{
			metadata: &core.ToolMetadata{
				Name:        "search",
				Description: "Search the web",
			},
		},
		&mockTool{
			metadata: &core.ToolMetadata{
				Name:        "calculator",
				Description: "Perform calculations",
			},
		},
	}

	caps := ToolsToCapabilities(tools)

	if len(caps) != 2 {
		t.Fatalf("expected 2 capabilities, got %d", len(caps))
	}

	// Verify first capability
	if caps[0].Name != "search" {
		t.Error("name not preserved")
	}
	if caps[0].Description != "Search the web" {
		t.Error("description not preserved")
	}
	if caps[0].Type != "function" {
		t.Error("expected type 'function'")
	}
	if caps[0].Schema == nil {
		t.Error("schema should not be nil")
	}

	// Verify second capability
	if caps[1].Name != "calculator" {
		t.Error("second capability name not preserved")
	}
}

func TestToolsToCapabilities_Empty(t *testing.T) {
	caps := ToolsToCapabilities([]core.Tool{})
	if caps != nil {
		t.Error("expected nil for empty tools")
	}

	caps = ToolsToCapabilities(nil)
	if caps != nil {
		t.Error("expected nil for nil tools")
	}
}

func TestCapabilitiesToToolMetadata(t *testing.T) {
	caps := []Capability{
		{
			Name:        "search",
			Description: "Search function",
			Type:        "function",
			Schema: map[string]interface{}{
				"type": "object",
			},
		},
		{
			Name:        "service",
			Description: "Some service",
			Type:        "service", // Non-function type
		},
	}

	metadata := CapabilitiesToToolMetadata(caps)

	// Should only convert function-type capabilities
	if len(metadata) != 1 {
		t.Errorf("expected 1 tool metadata (only function type), got %d", len(metadata))
	}
	if metadata[0].Name != "search" {
		t.Error("tool name not preserved")
	}
	if metadata[0].Description != "Search function" {
		t.Error("tool description not preserved")
	}
}

// ============================================================================
// Helper Function Tests
// ============================================================================

func TestCreateErrorMessage(t *testing.T) {
	err := fmt.Errorf("something went wrong")
	msg := CreateErrorMessage(err)

	if msg.Role != RoleAgent {
		t.Error("error message should have agent role")
	}
	if len(msg.Parts) != 1 {
		t.Error("expected 1 part")
	}
	if !contains(msg.Parts[0].Text, "something went wrong") {
		t.Error("error message not included in text")
	}
}

func TestCreateErrorMessage_Nil(t *testing.T) {
	msg := CreateErrorMessage(nil)

	if msg == nil {
		t.Fatal("expected non-nil message")
	}
	if len(msg.Parts) == 0 {
		t.Error("expected at least one part")
	}
}

func TestExtractTextFromMessage(t *testing.T) {
	msg := NewMessage(RoleAgent,
		NewTextPart("Line 1"),
		NewTextPart("Line 2"),
		NewFilePart("file.txt", "text/plain"), // Should be skipped
		NewTextPart("Line 3"),
	)

	text := ExtractTextFromMessage(msg)

	expected := "Line 1\nLine 2\nLine 3"
	if text != expected {
		t.Errorf("expected '%s', got '%s'", expected, text)
	}
}

func TestExtractTextFromMessage_Nil(t *testing.T) {
	text := ExtractTextFromMessage(nil)
	if text != "" {
		t.Error("expected empty string for nil message")
	}
}

func TestExtractTextFromArtifact(t *testing.T) {
	artifact := NewArtifact(
		NewTextPart("Part 1"),
		NewTextPart("Part 2"),
	)

	text := ExtractTextFromArtifact(artifact)

	expected := "Part 1\nPart 2"
	if text != expected {
		t.Errorf("expected '%s', got '%s'", expected, text)
	}
}

func TestCreateSimpleInput(t *testing.T) {
	input := CreateSimpleInput("test question")

	if input["question"] != "test question" {
		t.Error("question not set correctly")
	}
}

func TestCreateSimpleOutput(t *testing.T) {
	output := CreateSimpleOutput("test answer")

	if output["answer"] != "test answer" {
		t.Error("answer not set correctly")
	}
}

// ============================================================================
// Edge Case Tests
// ============================================================================

func TestMessageToAgentInput_ComplexTypes(t *testing.T) {
	// Test with various Go types
	msg := NewMessage(RoleUser,
		NewTextPart("123"),  // numeric string
		NewTextPart("true"), // boolean string
		NewTextPart(""),     // empty string
	)

	input, err := MessageToAgentInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// First part is question
	if input["question"] != "123" {
		t.Error("numeric string not preserved")
	}
}

func TestAgentOutputToMessage_VariousTypes(t *testing.T) {
	output := map[string]interface{}{
		"string": "text",
		"number": 42,
		"float":  3.14,
		"bool":   true,
		"nil":    nil,
	}

	msg, err := AgentOutputToMessage(output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All non-nil, non-internal values should be converted
	// nil values result in empty text and are skipped
	if len(msg.Parts) < 4 {
		t.Errorf("expected at least 4 parts, got %d", len(msg.Parts))
	}
}

func TestIsInternalField(t *testing.T) {
	tests := []struct {
		field    string
		internal bool
	}{
		{"_context_id", true},
		{"_metadata", true},
		{"_anything", true},
		{"answer", false},
		{"question", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.field, func(t *testing.T) {
			if got := isInternalField(tt.field); got != tt.internal {
				t.Errorf("isInternalField(%q) = %v, want %v", tt.field, got, tt.internal)
			}
		})
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ============================================================================
// Additional Helper Tests
// ============================================================================

func TestCreateErrorArtifact(t *testing.T) {
	err := fmt.Errorf("something went wrong")
	artifact := CreateErrorArtifact(err)

	if len(artifact.Parts) != 1 {
		t.Error("expected 1 part")
	}
	text := ExtractTextFromArtifact(artifact)
	if !contains(text, "something went wrong") {
		t.Error("error message not included in artifact")
	}
}

func TestCreateErrorArtifact_Nil(t *testing.T) {
	artifact := CreateErrorArtifact(nil)

	if len(artifact.Parts) == 0 {
		t.Error("expected at least one part")
	}
}
