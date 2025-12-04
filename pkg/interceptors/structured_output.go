// Package interceptors provides middleware components for dspy-go modules and agents.
package interceptors

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// StructuredOutputConfig configures the structured output interceptor.
type StructuredOutputConfig struct {
	// StrictSchema requires all output fields to be present in the response.
	// When false, missing fields are allowed and will be empty in the output.
	StrictSchema bool

	// IncludeDescriptions adds field descriptions to the JSON schema.
	// This helps the LLM understand what each field should contain.
	IncludeDescriptions bool

	// CustomInstructions are prepended to the prompt to guide JSON generation.
	CustomInstructions string
}

// DefaultStructuredOutputConfig returns sensible defaults.
func DefaultStructuredOutputConfig() StructuredOutputConfig {
	return StructuredOutputConfig{
		StrictSchema:        false,
		IncludeDescriptions: true,
		CustomInstructions:  "",
	}
}

// StructuredOutputInterceptor creates a module interceptor that uses native JSON
// output capabilities instead of text-based parsing.
//
// This interceptor transforms the Predict module by:
// 1. Converting output signature fields into a JSON schema
// 2. Using GenerateWithJSON instead of Generate + parsing
// 3. Mapping the JSON response directly to output fields
//
// Benefits over text-based parsing:
// - No parsing errors from malformed prefixes
// - Strongly typed output from the LLM
// - Works with any signature without custom prefix configuration
// - More reliable extraction of multiple output fields
//
// Usage:
//
//	config := interceptors.DefaultStructuredOutputConfig()
//	interceptor := interceptors.StructuredOutputInterceptor(config)
//	predict.SetInterceptors([]core.ModuleInterceptor{interceptor})
//
// Or use the convenience method on Predict:
//
//	predict := modules.NewPredict(signature).WithStructuredOutput()
func StructuredOutputInterceptor(config StructuredOutputConfig) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		logger := logging.GetLogger()

		// Check if the LLM supports JSON output
		llm := core.GlobalConfig.DefaultLLM
		if llm == nil {
			logger.Debug(ctx, "No LLM configured, falling back to text-based parsing")
			return handler(ctx, inputs, opts...)
		}

		if !supportsJSONOutput(llm) {
			logger.Debug(ctx, "LLM %s does not support JSON output, falling back to text-based parsing", llm.ProviderName())
			return handler(ctx, inputs, opts...)
		}

		// Get signature from module info
		if info == nil || len(info.Signature.Outputs) == 0 {
			logger.Debug(ctx, "No signature or output fields, falling back to text-based parsing")
			return handler(ctx, inputs, opts...)
		}

		signature := info.Signature

		// Build the prompt with JSON schema instructions
		prompt := buildStructuredPrompt(inputs, signature, config)

		logger.Debug(ctx, "Using structured JSON output for %d output fields", len(signature.Outputs))

		// Call LLM with JSON output
		result, err := llm.GenerateWithJSON(ctx, prompt)
		if err != nil {
			logger.Warn(ctx, "GenerateWithJSON failed: %v, falling back to text-based parsing", err)
			// Fall back to handler on error - the text-based parsing might still work
			return handler(ctx, inputs, opts...)
		}

		// Transform the JSON result into the expected output format
		return transformJSONResult(result, signature, config)
	}
}

// supportsJSONOutput checks if the LLM supports native JSON output.
func supportsJSONOutput(llm core.LLM) bool {
	capabilities := llm.Capabilities()
	return slices.Contains(capabilities, core.CapabilityJSON)
}

// buildStructuredPrompt constructs a prompt that instructs the LLM to output JSON.
func buildStructuredPrompt(inputs map[string]any, signature core.Signature, config StructuredOutputConfig) string {
	var sb strings.Builder

	// Add custom instructions if provided
	if config.CustomInstructions != "" {
		sb.WriteString(config.CustomInstructions)
		sb.WriteString("\n\n")
	}

	// Add signature instruction
	if signature.Instruction != "" {
		sb.WriteString(signature.Instruction)
		sb.WriteString("\n\n")
	}

	// Add input fields
	sb.WriteString("## Inputs\n")
	for _, inputField := range signature.Inputs {
		if value, ok := inputs[inputField.Name]; ok {
			sb.WriteString(fmt.Sprintf("**%s**: %v\n", inputField.Name, value))
		}
	}
	sb.WriteString("\n")

	// Add JSON schema instructions
	sb.WriteString("## Required Output Format\n")
	sb.WriteString("Respond with a JSON object containing the following fields:\n\n")
	sb.WriteString("```json\n{\n")

	for i, outputField := range signature.Outputs {
		fieldType := getJSONType(outputField.Type)
		if config.IncludeDescriptions && outputField.Description != "" {
			sb.WriteString(fmt.Sprintf("  // %s\n", outputField.Description))
		}
		sb.WriteString(fmt.Sprintf("  \"%s\": <%s>", outputField.Name, fieldType))
		if i < len(signature.Outputs)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}

	sb.WriteString("}\n```\n\n")

	// Add field descriptions as additional context
	if config.IncludeDescriptions {
		hasDescriptions := false
		for _, outputField := range signature.Outputs {
			if outputField.Description != "" {
				if !hasDescriptions {
					sb.WriteString("### Field Descriptions\n")
					hasDescriptions = true
				}
				sb.WriteString(fmt.Sprintf("- **%s**: %s\n", outputField.Name, outputField.Description))
			}
		}
		if hasDescriptions {
			sb.WriteString("\n")
		}
	}

	sb.WriteString("Respond ONLY with the JSON object, no additional text or markdown code blocks.")

	return sb.String()
}

// getJSONType maps FieldType to JSON type description.
func getJSONType(fieldType core.FieldType) string {
	switch fieldType {
	case core.FieldTypeText:
		return "string"
	case core.FieldTypeImage:
		return "string (base64 or URL)"
	case core.FieldTypeAudio:
		return "string (base64 or URL)"
	default:
		return "string"
	}
}

// transformJSONResult maps the JSON response to the expected output format.
func transformJSONResult(result map[string]interface{}, signature core.Signature, config StructuredOutputConfig) (map[string]any, error) {
	output := make(map[string]any)

	// Map each output field from the JSON result
	for _, outputField := range signature.Outputs {
		if value, ok := result[outputField.Name]; ok {
			output[outputField.Name] = value
		} else if config.StrictSchema {
			return nil, fmt.Errorf("missing required field '%s' in JSON response", outputField.Name)
		} else {
			// Set empty string for missing optional fields
			output[outputField.Name] = ""
		}
	}

	// Also include any extra fields from the response (for flexibility)
	for key, value := range result {
		if _, exists := output[key]; !exists {
			output[key] = value
		}
	}

	return output, nil
}

// ChainOfThoughtStructuredConfig extends StructuredOutputConfig for CoT-specific settings.
type ChainOfThoughtStructuredConfig struct {
	StructuredOutputConfig

	// ReasoningFieldName is the name of the field containing the reasoning.
	// Defaults to "reasoning" if empty.
	ReasoningFieldName string

	// IncludeReasoningInOutput determines if reasoning should be in the final output.
	IncludeReasoningInOutput bool
}

// DefaultChainOfThoughtStructuredConfig returns defaults for ChainOfThought.
func DefaultChainOfThoughtStructuredConfig() ChainOfThoughtStructuredConfig {
	return ChainOfThoughtStructuredConfig{
		StructuredOutputConfig:   DefaultStructuredOutputConfig(),
		ReasoningFieldName:       "reasoning",
		IncludeReasoningInOutput: true,
	}
}

// ChainOfThoughtStructuredInterceptor creates an interceptor specifically for ChainOfThought.
// It ensures the reasoning field is always included in the output schema.
func ChainOfThoughtStructuredInterceptor(config ChainOfThoughtStructuredConfig) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		logger := logging.GetLogger()

		llm := core.GlobalConfig.DefaultLLM
		if llm == nil || !supportsJSONOutput(llm) {
			return handler(ctx, inputs, opts...)
		}

		if info == nil {
			return handler(ctx, inputs, opts...)
		}

		signature := info.Signature
		reasoningField := config.ReasoningFieldName
		if reasoningField == "" {
			reasoningField = "reasoning"
		}

		// Build enhanced prompt with explicit reasoning request
		prompt := buildCoTStructuredPrompt(inputs, signature, reasoningField, config)

		logger.Debug(ctx, "Using structured JSON output for ChainOfThought with reasoning field '%s'", reasoningField)

		result, err := llm.GenerateWithJSON(ctx, prompt)
		if err != nil {
			logger.Warn(ctx, "GenerateWithJSON failed for CoT: %v, falling back", err)
			return handler(ctx, inputs, opts...)
		}

		// Transform result, ensuring reasoning is captured
		output, err := transformJSONResult(result, signature, config.StructuredOutputConfig)
		if err != nil {
			return nil, err
		}

		// Add reasoning to output if present and configured
		if reasoning, ok := result[reasoningField]; ok && config.IncludeReasoningInOutput {
			output[reasoningField] = reasoning
		}

		return output, nil
	}
}

// buildCoTStructuredPrompt builds a prompt specifically for ChainOfThought with reasoning.
func buildCoTStructuredPrompt(inputs map[string]any, signature core.Signature, reasoningField string, config ChainOfThoughtStructuredConfig) string {
	var sb strings.Builder

	// Add custom instructions
	if config.CustomInstructions != "" {
		sb.WriteString(config.CustomInstructions)
		sb.WriteString("\n\n")
	}

	// Add signature instruction
	if signature.Instruction != "" {
		sb.WriteString(signature.Instruction)
		sb.WriteString("\n\n")
	}

	// Add input fields
	sb.WriteString("## Inputs\n")
	for _, inputField := range signature.Inputs {
		if value, ok := inputs[inputField.Name]; ok {
			sb.WriteString(fmt.Sprintf("**%s**: %v\n", inputField.Name, value))
		}
	}
	sb.WriteString("\n")

	// Add thinking instructions
	sb.WriteString("## Instructions\n")
	sb.WriteString("Think through this step-by-step. First explain your reasoning, then provide the answer.\n\n")

	// Build JSON schema with reasoning field
	sb.WriteString("## Required Output Format\n")
	sb.WriteString("Respond with a JSON object in this exact format:\n\n")
	sb.WriteString("```json\n{\n")

	// Always include reasoning first
	sb.WriteString(fmt.Sprintf("  \"%s\": \"<your step-by-step reasoning>\",\n", reasoningField))

	// Then add output fields
	for i, outputField := range signature.Outputs {
		// Skip if this is the reasoning field (already added)
		if outputField.Name == reasoningField {
			continue
		}
		fieldType := getJSONType(outputField.Type)
		sb.WriteString(fmt.Sprintf("  \"%s\": <%s>", outputField.Name, fieldType))
		if i < len(signature.Outputs)-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}

	sb.WriteString("}\n```\n\n")

	// Add field descriptions
	if config.IncludeDescriptions {
		sb.WriteString("### Field Descriptions\n")
		sb.WriteString(fmt.Sprintf("- **%s**: Your detailed step-by-step reasoning process\n", reasoningField))
		for _, outputField := range signature.Outputs {
			if outputField.Name != reasoningField && outputField.Description != "" {
				sb.WriteString(fmt.Sprintf("- **%s**: %s\n", outputField.Name, outputField.Description))
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("Respond ONLY with the JSON object, no additional text.")

	return sb.String()
}

// StructuredOutputAdapter provides a high-level interface for enabling structured output.
type StructuredOutputAdapter struct {
	config      StructuredOutputConfig
	interceptor core.ModuleInterceptor
}

// NewStructuredOutputAdapter creates an adapter for structured JSON output.
func NewStructuredOutputAdapter() *StructuredOutputAdapter {
	config := DefaultStructuredOutputConfig()
	return &StructuredOutputAdapter{
		config:      config,
		interceptor: StructuredOutputInterceptor(config),
	}
}

// WithStrictSchema enables strict schema validation.
func (a *StructuredOutputAdapter) WithStrictSchema() *StructuredOutputAdapter {
	a.config.StrictSchema = true
	a.interceptor = StructuredOutputInterceptor(a.config)
	return a
}

// WithCustomInstructions adds custom instructions to the prompt.
func (a *StructuredOutputAdapter) WithCustomInstructions(instructions string) *StructuredOutputAdapter {
	a.config.CustomInstructions = instructions
	a.interceptor = StructuredOutputInterceptor(a.config)
	return a
}

// WithoutDescriptions disables field descriptions in the schema.
func (a *StructuredOutputAdapter) WithoutDescriptions() *StructuredOutputAdapter {
	a.config.IncludeDescriptions = false
	a.interceptor = StructuredOutputInterceptor(a.config)
	return a
}

// GetInterceptor returns the configured module interceptor.
func (a *StructuredOutputAdapter) GetInterceptor() core.ModuleInterceptor {
	return a.interceptor
}
