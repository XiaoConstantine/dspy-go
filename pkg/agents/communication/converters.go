package communication

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// ============================================================================
// Message to Agent Input Conversion
// ============================================================================

// MessageToAgentInput converts an a2a Message to dspy-go agent input format.
// Text parts are extracted and mapped to input fields based on metadata or position.
// The first text part without metadata defaults to the "question" field.
func MessageToAgentInput(msg *Message) (map[string]any, error) {
	return messageToAgentInput(msg, agents.AgentContract{PrimaryInput: "question"}, false)
}

// MessageToAgentInputWithContract maps unlabelled text to the contract's
// primary input and validates required named fields.
func MessageToAgentInputWithContract(msg *Message, contract agents.AgentContract) (map[string]any, error) {
	if err := contract.Validate(); err != nil {
		return nil, err
	}
	return messageToAgentInput(msg, contract, true)
}

func messageToAgentInput(msg *Message, contract agents.AgentContract, validateRequired bool) (map[string]any, error) {
	if msg == nil {
		return nil, fmt.Errorf("message cannot be nil")
	}

	input := make(map[string]any)
	textPartCount := 0

	// Process each part
	for _, part := range msg.Parts {
		switch part.Type {
		case "text":
			fieldName := extractFieldName(part, textPartCount, contract.PrimaryInput)
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
			for key, value := range part.Data {
				if contractHasInput(contract, key) {
					input[key] = value
				} else if validateRequired {
					return nil, fmt.Errorf("message data field %q is not accepted by the agent", key)
				} else {
					input["_data_"+key] = value
				}
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
	if validateRequired {
		for name := range input {
			if !strings.HasPrefix(name, "_") && !contractHasInput(contract, name) {
				return nil, fmt.Errorf("message field %q is not accepted by the agent", name)
			}
		}
		for _, field := range contract.Inputs {
			if field.Required {
				if _, exists := input[field.Name]; !exists {
					return nil, fmt.Errorf("message is missing required agent input %q", field.Name)
				}
			}
		}
	}

	return input, nil
}

// extractFieldName determines the field name for a text part.
func extractFieldName(part Part, index int, primaryInput string) string {
	// Check metadata for explicit field name
	if part.Metadata != nil {
		if field, ok := part.Metadata["field"].(string); ok && field != "" {
			return field
		}
	}

	// Default naming based on position
	if index == 0 {
		if primaryInput != "" {
			return primaryInput
		}
		return "question"
	}
	return fmt.Sprintf("text_%d", index)
}

func contractHasInput(contract agents.AgentContract, name string) bool {
	for _, field := range contract.Inputs {
		if field.Name == name {
			return true
		}
	}
	return false
}

// ============================================================================
// Agent Output to Message/Artifact Conversion
// ============================================================================

// AgentOutputToMessage converts dspy-go agent output to an a2a Message.
// Each output field becomes a text part with metadata indicating the field name.
// Internal fields (prefixed with "_") are excluded from conversion.
func AgentOutputToMessage(output map[string]any) (*Message, error) {
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
		part := NewTextPartWithMetadata(text, map[string]any{
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
func AgentOutputToArtifact(output map[string]any) (Artifact, error) {
	if output == nil {
		return Artifact{}, fmt.Errorf("output cannot be nil")
	}

	parts := []Part{}
	metadata := make(map[string]any)

	// Extract metadata if present
	if meta, ok := output["_metadata"].(map[string]any); ok {
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

		part := NewTextPartWithMetadata(text, map[string]any{
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
func formatOutputValue(value any) string {
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

		// Convert tool's InputSchema to JSON schema map for full discoverability
		var schemaMap map[string]any
		schema := tool.InputSchema()
		// Attempt to convert the tool's input schema to a map.
		if schemaBytes, err := json.Marshal(schema); err == nil {
			if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
				// Log schema conversion failures to help identify malformed tool schemas
				logger := logging.GetLogger()
				logger.Error(context.Background(), "Failed to unmarshal tool schema for %s: %v", metadata.Name, err)
			}
		}
		cap.Schema = schemaMap

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

		// Convert schema map back to InputSchema struct for full round-trip support
		if cap.Schema != nil {
			// Attempt to convert schema map back to InputSchema struct.
			if schemaBytes, err := json.Marshal(cap.Schema); err == nil {
				if err := json.Unmarshal(schemaBytes, &meta.InputSchema); err != nil {
					// Log schema conversion failures to help identify malformed capability schemas
					logger := logging.GetLogger()
					logger.Error(context.Background(), "Failed to unmarshal capability schema for %s: %v", cap.Name, err)
				}
			}
		}

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

	var texts []string
	for _, part := range msg.Parts {
		if part.Type == "text" && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	return strings.Join(texts, "\n")
}

// ExtractTextFromArtifact extracts all text content from an artifact.
func ExtractTextFromArtifact(artifact Artifact) string {
	if len(artifact.Parts) == 0 {
		return ""
	}

	var texts []string
	for _, part := range artifact.Parts {
		if part.Type == "text" && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	return strings.Join(texts, "\n")
}

// CreateSimpleInput creates agent input from a simple text question.
func CreateSimpleInput(question string) map[string]any {
	return map[string]any{
		"question": question,
	}
}

// CreateSimpleOutput creates agent output from a simple text answer.
func CreateSimpleOutput(answer string) map[string]any {
	return map[string]any{
		"answer": answer,
	}
}
