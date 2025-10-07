package a2a

import (
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ============================================================================
// Message to Agent Input Conversion
// ============================================================================

// MessageToAgentInput converts an a2a Message to dspy-go agent input format.
// Text parts are extracted and mapped to input fields based on metadata or position.
// The first text part without metadata defaults to the "question" field.
func MessageToAgentInput(msg *Message) (map[string]interface{}, error) {
	if msg == nil {
		return nil, fmt.Errorf("message cannot be nil")
	}

	input := make(map[string]interface{})
	textPartCount := 0

	// Process each part
	for _, part := range msg.Parts {
		switch part.Type {
		case "text":
			fieldName := extractFieldName(part, textPartCount)
			input[fieldName] = part.Text
			textPartCount++

		case "file":
			// Store file parts in a separate array for now
			// Agent can access via "_files" key
			if _, exists := input["_files"]; !exists {
				input["_files"] = []FilePart{}
			}
			if part.File != nil {
				input["_files"] = append(input["_files"].([]FilePart), *part.File)
			}

		case "data":
			// Merge data parts into input
			// Prefix keys with "_data_" to avoid collisions
			for key, value := range part.Data {
				input["_data_"+key] = value
			}
		}
	}

	// Add context ID if present
	if msg.ContextID != "" {
		input["_context_id"] = msg.ContextID
	}

	// Add message metadata
	if msg.Metadata != nil {
		input["_message_metadata"] = msg.Metadata
	}

	if len(input) == 0 {
		return nil, fmt.Errorf("message contains no convertible parts")
	}

	return input, nil
}

// extractFieldName determines the field name for a text part.
func extractFieldName(part Part, index int) string {
	// Check metadata for explicit field name
	if part.Metadata != nil {
		if field, ok := part.Metadata["field"].(string); ok && field != "" {
			return field
		}
	}

	// Default naming based on position
	if index == 0 {
		return "question" // First part is usually the question
	}
	return fmt.Sprintf("text_%d", index)
}

// ============================================================================
// Agent Output to Message/Artifact Conversion
// ============================================================================

// AgentOutputToMessage converts dspy-go agent output to an a2a Message.
// Each output field becomes a text part with metadata indicating the field name.
// Internal fields (prefixed with "_") are excluded from conversion.
func AgentOutputToMessage(output map[string]interface{}) (*Message, error) {
	if output == nil {
		return nil, fmt.Errorf("output cannot be nil")
	}

	parts := []Part{}

	// Convert output fields to parts
	for key, value := range output {
		// Skip internal fields
		if isInternalField(key) {
			continue
		}

		// Convert value to text
		text := formatOutputValue(value)
		if text == "" {
			continue
		}

		// Create part with field metadata
		part := NewTextPartWithMetadata(text, map[string]interface{}{
			"field": key,
		})
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		return nil, fmt.Errorf("output contains no convertible fields")
	}

	msg := NewMessage(RoleAgent, parts...)

	// Restore context ID if present
	if contextID, ok := output["_context_id"].(string); ok {
		msg.ContextID = contextID
	}

	return msg, nil
}

// AgentOutputToArtifact converts dspy-go agent output to an a2a Artifact.
// Similar to AgentOutputToMessage but wraps the result in an Artifact structure.
func AgentOutputToArtifact(output map[string]interface{}) (Artifact, error) {
	if output == nil {
		return Artifact{}, fmt.Errorf("output cannot be nil")
	}

	parts := []Part{}
	metadata := make(map[string]interface{})

	// Extract metadata if present
	if meta, ok := output["_metadata"].(map[string]interface{}); ok {
		metadata = meta
	}

	// Convert output fields to parts
	for key, value := range output {
		// Skip internal fields
		if isInternalField(key) {
			continue
		}

		text := formatOutputValue(value)
		if text == "" {
			continue
		}

		part := NewTextPartWithMetadata(text, map[string]interface{}{
			"field": key,
		})
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		return Artifact{}, fmt.Errorf("output contains no convertible fields")
	}

	if len(metadata) > 0 {
		return NewArtifactWithMetadata(metadata, parts...), nil
	}

	return NewArtifact(parts...), nil
}

// formatOutputValue converts any output value to a string representation.
func formatOutputValue(value interface{}) string {
	if value == nil {
		return ""
	}

	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

// isInternalField returns true if the field name indicates an internal field.
// Internal fields start with "_" and are not converted to parts.
func isInternalField(fieldName string) bool {
	if len(fieldName) == 0 {
		return false
	}
	return fieldName[0] == '_'
}

// ============================================================================
// Tool/Capability Conversion
// ============================================================================

// ToolsToCapabilities converts dspy-go Tools to a2a Capabilities.
// Each tool becomes a function-type capability with its schema.
func ToolsToCapabilities(tools []core.Tool) []Capability {
	if len(tools) == 0 {
		return nil
	}

	capabilities := make([]Capability, 0, len(tools))

	for _, tool := range tools {
		metadata := tool.Metadata()
		if metadata == nil {
			continue
		}

		cap := Capability{
			Name:        metadata.Name,
			Description: metadata.Description,
			Type:        "function",
		}

		// InputSchema from MCP is already compatible with JSON Schema format
		// For now, store it as-is; the schema field is map[string]interface{}
		// In the future, we might need to convert InputSchema to map explicitly
		cap.Schema = map[string]interface{}{
			"type": "object",
			// Note: Detailed schema conversion from models.InputSchema to map
			// would go here. For now, we provide a minimal schema.
		}

		capabilities = append(capabilities, cap)
	}

	return capabilities
}

// CapabilitiesToToolMetadata converts a2a Capabilities to dspy-go ToolMetadata.
// Only function-type capabilities are converted.
// Note: The returned metadata describes remote tools but doesn't include implementation.
// Use this when discovering remote agent capabilities via AgentCard.
func CapabilitiesToToolMetadata(capabilities []Capability) []*core.ToolMetadata {
	if len(capabilities) == 0 {
		return nil
	}

	metadata := make([]*core.ToolMetadata, 0, len(capabilities))

	for _, cap := range capabilities {
		// Only convert function-type capabilities
		if cap.Type != "function" {
			continue
		}

		meta := &core.ToolMetadata{
			Name:        cap.Name,
			Description: cap.Description,
		}

		// Note: cap.Schema is map[string]interface{} (JSON Schema)
		// but ToolMetadata.InputSchema is models.InputSchema from MCP-Go
		// For now, we leave InputSchema nil. A full implementation would
		// need to convert JSON Schema to models.InputSchema format.

		metadata = append(metadata, meta)
	}

	return metadata
}

// ============================================================================
// Helper Conversion Functions
// ============================================================================

// CreateErrorMessage creates an a2a Message representing an error.
func CreateErrorMessage(err error) *Message {
	if err == nil {
		return NewAgentMessage("Unknown error occurred")
	}

	return NewAgentMessage(fmt.Sprintf("Error: %v", err))
}

// CreateErrorArtifact creates an a2a Artifact representing an error.
func CreateErrorArtifact(err error) Artifact {
	if err == nil {
		return NewArtifact(NewTextPart("Unknown error occurred"))
	}

	return NewArtifact(NewTextPart(fmt.Sprintf("Error: %v", err)))
}

// ExtractTextFromMessage extracts all text content from a message.
// Returns concatenated text from all text parts, separated by newlines.
func ExtractTextFromMessage(msg *Message) string {
	if msg == nil || len(msg.Parts) == 0 {
		return ""
	}

	var text string
	for i, part := range msg.Parts {
		if part.Type == "text" && part.Text != "" {
			if i > 0 && text != "" {
				text += "\n"
			}
			text += part.Text
		}
	}

	return text
}

// ExtractTextFromArtifact extracts all text content from an artifact.
func ExtractTextFromArtifact(artifact Artifact) string {
	if len(artifact.Parts) == 0 {
		return ""
	}

	var text string
	for i, part := range artifact.Parts {
		if part.Type == "text" && part.Text != "" {
			if i > 0 && text != "" {
				text += "\n"
			}
			text += part.Text
		}
	}

	return text
}

// CreateSimpleInput creates agent input from a simple text question.
func CreateSimpleInput(question string) map[string]interface{} {
	return map[string]interface{}{
		"question": question,
	}
}

// CreateSimpleOutput creates agent output from a simple text answer.
func CreateSimpleOutput(answer string) map[string]interface{} {
	return map[string]interface{}{
		"answer": answer,
	}
}
