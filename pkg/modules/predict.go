package modules

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

type Predict struct {
	core.BaseModule
	Demos          []core.Example
	LLM            core.LLM
	defaultOptions *core.ModuleOptions

	// XML output configuration
	xmlConfig     *interceptors.XMLConfig
	enableXMLMode bool
}

// Ensure Predict implements core.Module.
var _ core.Module = (*Predict)(nil)
var _ core.InterceptableModule = (*Predict)(nil)

// Ensure Predict implements InterceptableModule.
var _ core.InterceptableModule = (*Predict)(nil)

// Ensure Predict implements demo interfaces for saving/loading.
var _ core.DemoProvider = (*Predict)(nil)
var _ core.DemoConsumer = (*Predict)(nil)

// Ensure Predict implements LMConfigProvider for saving/loading.
var _ core.LMConfigProvider = (*Predict)(nil)

func NewPredict(signature core.Signature) *Predict {
	baseModule := core.NewModule(signature)
	baseModule.ModuleType = "Predict"
	baseModule.DisplayName = "" // Will be set by user or derived from context

	p := &Predict{
		BaseModule: *baseModule,
		Demos:      []core.Example{},
		LLM:        core.GetDefaultLLM(),
	}

	// Enable XML output by default for structured responses
	// Users can override with WithTextOutput() if they prefer traditional parsing
	if shouldUseXMLByDefault(signature) {
		defaultXMLConfig := interceptors.DefaultXMLConfig()
		p.WithXMLOutput(defaultXMLConfig)
	}

	return p
}

// WithName sets a semantic name for this module instance.
func (p *Predict) WithName(name string) *Predict {
	p.DisplayName = name
	return p
}

func (p *Predict) WithDefaultOptions(opts ...core.Option) *Predict {
	options := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(options)
	}
	p.defaultOptions = options
	return p
}

// IsXMLModeEnabled returns true if XML mode is enabled for this module.
func (p *Predict) IsXMLModeEnabled() bool {
	return p.enableXMLMode
}

// WithXMLOutput enables XML interceptor-based output formatting.
// This provides structured XML output with validation, security features, and error handling.
func (p *Predict) WithXMLOutput(config interceptors.XMLConfig) *Predict {
	p.enableXMLMode = true
	p.xmlConfig = &config

	// Apply the combined XML interceptor (format + parse)
	xmlInterceptor := interceptors.XMLModuleInterceptor(config)

	// Combine with existing interceptors if any
	existingInterceptors := p.GetInterceptors()
	allInterceptors := append(existingInterceptors, xmlInterceptor)
	p.SetInterceptors(allInterceptors)

	return p
}

// WithStructuredOutput enables native JSON structured output instead of text parsing.
// This uses the LLM's GenerateWithJSON capability to produce structured responses
// that map directly to the signature's output fields, eliminating parsing errors.
//
// Benefits:
//   - No parsing errors from malformed prefixes
//   - Strongly typed output from the LLM
//   - Works with any signature without custom prefix configuration
//   - More reliable extraction of multiple output fields
//
// Requirements:
//   - The LLM must support CapabilityJSON
//   - Falls back to text-based parsing if not supported
//
// Usage:
//
//	predict := modules.NewPredict(signature).WithStructuredOutput()
func (p *Predict) WithStructuredOutput() *Predict {
	config := interceptors.DefaultStructuredOutputConfig()
	interceptor := interceptors.StructuredOutputInterceptor(config)
	p.SetInterceptors([]core.ModuleInterceptor{interceptor})
	return p
}

// WithStructuredOutputConfig enables structured output with custom configuration.
func (p *Predict) WithStructuredOutputConfig(config interceptors.StructuredOutputConfig) *Predict {
	interceptor := interceptors.StructuredOutputInterceptor(config)
	p.SetInterceptors([]core.ModuleInterceptor{interceptor})
	return p
}

// WithTextOutput disables XML output and uses traditional text-based parsing.
// This is an escape hatch for users who prefer the original behavior.
//
// IMPORTANT: This method currently removes ALL interceptors from the module,
// not just XML-related ones. This means any custom interceptors you've configured
// (such as logging, caching, or metrics) will also be removed.
//
// TODO(#interceptor-preservation): Implement selective removal of only XML interceptors.
// This requires an interceptor identification mechanism since interceptors are function types.
// Possible solutions:
//   1. Wrap interceptors in a struct with metadata
//   2. Maintain a separate list of XML interceptor indices
//   3. Use a registry pattern for interceptor management
//
// Until this is fixed, if you need to preserve custom interceptors:
//   1. Save your interceptors before calling WithTextOutput()
//   2. Re-add them after calling WithTextOutput()
//
// Example workaround:
//   customInterceptors := predict.GetInterceptors()[:2] // Save first 2 custom interceptors
//   predict.WithTextOutput()
//   predict.SetInterceptors(customInterceptors) // Re-add them
func (p *Predict) WithTextOutput() *Predict {
	p.enableXMLMode = false
	p.xmlConfig = nil

	// WARNING: This clears ALL interceptors, not just XML ones
	// See TODO above for future improvement
	p.SetInterceptors([]core.ModuleInterceptor{})

	return p
}

func (p *Predict) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	// If XML mode is enabled, automatically use ProcessWithInterceptors for proper XML handling
	if p.enableXMLMode {
		return p.ProcessWithInterceptors(ctx, inputs, nil, opts...)
	}

	logger := logging.GetLogger()
	callOptions := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(callOptions)
	}

	finalOptions := p.defaultOptions.MergeWith(callOptions)

	if callOptions.StreamHandler != nil {
		return p.processWithStreaming(ctx, inputs, callOptions.StreamHandler, finalOptions)
	}
	// Use semantic name if set, otherwise fall back to operation name
	displayName := p.GetDisplayName()
	if displayName == "" || displayName == "BaseModule" {
		displayName = "Predict"
	}

	metadata := map[string]interface{}{
		"module_type":   p.GetModuleType(),
		"module_config": p.GetSignature().String(),
	}
	ctx, span := core.StartSpanWithContext(ctx, "Predict", displayName, metadata)
	defer core.EndSpan(ctx)
	span.WithAnnotation("inputs", inputs)

	if err := p.ValidateInputs(inputs); err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "input validation failed"),
			errors.Fields{
				"module": "Predict",
				"inputs": inputs,
			})
	}

	signature := p.GetSignature()

	// Check if inputs contain multimodal content
	if core.IsMultimodalContent(signature, inputs) {
		// Use structured content approach
		content := core.ConvertInputsToContentBlocks(signature, inputs)
		logger.Debug(ctx, "Using multimodal content generation with %d blocks", len(content))

		resp, err := p.LLM.GenerateWithContent(ctx, content, finalOptions.GenerateOptions...)
		if err != nil {
			span.WithError(err)
			return nil, errors.WithFields(
				errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate multimodal prediction"),
				errors.Fields{
					"module":         "Predict",
					"content_blocks": len(content),
					"model":          p.LLM,
				})
		}

		logger.Debug(ctx, "LLM Multimodal Completion: %v", resp.Content)

		if resp.Usage != nil {
			if state := core.GetExecutionState(ctx); state != nil {
				state.WithTokenUsage(&core.TokenUsage{
					PromptTokens:     resp.Usage.PromptTokens,
					CompletionTokens: resp.Usage.CompletionTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				})
			}
		}

		// Parse the response (same as text-based)
		cleaned := stripMarkdown(resp.Content, signature)
		outputs := parseCompletion(cleaned, signature)
		formattedOutputs := p.FormatOutputs(outputs)

		return formattedOutputs, nil
	}

	// Fall back to traditional text-based approach
	prompt := formatPrompt(signature, p.Demos, inputs)
	logger.Debug(ctx, "Generated prompt with prompt: %v", prompt)

	resp, err := p.LLM.Generate(ctx, prompt, finalOptions.GenerateOptions...)
	if err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate prediction"),
			errors.Fields{
				"module": "Predict",
				"prompt": prompt,
				"model":  p.LLM,
			})
	}
	logger.Debug(ctx, "LLM Completion: %v", resp.Content)

	if resp.Usage != nil {
		if state := core.GetExecutionState(ctx); state != nil {
			state.WithTokenUsage(&core.TokenUsage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
			})
		}
		logger.Debug(ctx, "LLM Completion total token usage: %d, %d, %d", resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}

	// Traditional parsing for non-XML mode
	cleaned := stripMarkdown(resp.Content, p.GetSignature())
	logger.Debug(ctx, "Normalized completion: %s", cleaned)
	outputs := parseCompletion(cleaned, signature)
	logger.Debug(ctx, "Parsed LLM Completion: %v", outputs)
	formattedOutputs := p.FormatOutputs(outputs)
	logger.Debug(ctx, "Formatted LLM Completion: %v", outputs)

	// Include raw response for XML interceptor processing
	if p.IsXMLModeEnabled() {
		formattedOutputs["__raw_response"] = resp.Content
	}

	span.WithAnnotation("outputs", formattedOutputs)

	return formattedOutputs, nil
}

// ProcessWithInterceptors executes the Predict module's logic with interceptor support.
func (p *Predict) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	// Use the helper method from BaseModule, but pass our core processing method (without XML logic)
	return p.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, p.processCore, opts...)
}

// processCore handles the core prediction logic without XML interceptor routing.
func (p *Predict) processCore(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	callOptions := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(callOptions)
	}

	finalOptions := p.defaultOptions.MergeWith(callOptions)

	if callOptions.StreamHandler != nil {
		return p.processWithStreaming(ctx, inputs, callOptions.StreamHandler, finalOptions)
	}
	// Use semantic name if set, otherwise fall back to operation name
	displayName := p.GetDisplayName()
	if displayName == "" || displayName == "BaseModule" {
		displayName = "Predict"
	}

	metadata := map[string]interface{}{
		"module_type":   p.GetModuleType(),
		"module_config": p.GetSignature().String(),
	}
	ctx, span := core.StartSpanWithContext(ctx, "Predict", displayName, metadata)
	defer core.EndSpan(ctx)
	span.WithAnnotation("inputs", inputs)

	if err := p.ValidateInputs(inputs); err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "input validation failed"),
			errors.Fields{
				"module": "Predict",
				"inputs": inputs,
			})
	}

	signature := p.GetSignature()

	// Check if inputs contain multimodal content
	if core.IsMultimodalContent(signature, inputs) {
		// Use structured content approach
		content := core.ConvertInputsToContentBlocks(signature, inputs)
		logger.Debug(ctx, "Using multimodal content generation with %d blocks", len(content))

		resp, err := p.LLM.GenerateWithContent(ctx, content, finalOptions.GenerateOptions...)
		if err != nil {
			span.WithError(err)
			return nil, errors.WithFields(
				errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate multimodal prediction"),
				errors.Fields{
					"module":         "Predict",
					"content_blocks": len(content),
					"model":          p.LLM,
				})
		}

		logger.Debug(ctx, "LLM Multimodal Completion: %v", resp.Content)

		if resp.Usage != nil {
			if state := core.GetExecutionState(ctx); state != nil {
				state.WithTokenUsage(&core.TokenUsage{
					PromptTokens:     resp.Usage.PromptTokens,
					CompletionTokens: resp.Usage.CompletionTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				})
			}
		}

		// For XML mode, return raw LLM content; for non-XML mode, parse normally
		if p.enableXMLMode {
			// Return raw LLM response for XML interceptor to parse
			rawOutputs := map[string]interface{}{
				"response": resp.Content,
			}
			return rawOutputs, nil
		}

		// Traditional parsing for non-XML mode
		cleaned := stripMarkdown(resp.Content, signature)
		outputs := parseCompletion(cleaned, signature)
		formattedOutputs := p.FormatOutputs(outputs)

		return formattedOutputs, nil
	}

	// Traditional text-based approach
	prompt := formatPrompt(signature, p.Demos, inputs)
	logger.Debug(ctx, "Generated prompt with prompt: %v", prompt)

	resp, err := p.LLM.Generate(ctx, prompt, finalOptions.GenerateOptions...)
	if err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate prediction"),
			errors.Fields{
				"module": "Predict",
				"prompt": prompt,
				"model":  p.LLM,
			})
	}
	logger.Debug(ctx, "LLM Completion: %v", resp.Content)

	if resp.Usage != nil {
		if state := core.GetExecutionState(ctx); state != nil {
			state.WithTokenUsage(&core.TokenUsage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
			})
		}
		logger.Debug(ctx, "LLM Completion total token usage: %d, %d, %d", resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}

	// For XML mode, return raw LLM content so XML interceptor can parse it
	// For non-XML mode, use traditional parsing
	if p.enableXMLMode {
		// Return raw LLM response for XML interceptor to parse
		rawOutputs := map[string]interface{}{
			"response": resp.Content, // Use standard field name that XML parser looks for
		}
		return rawOutputs, nil
	}

	// Traditional parsing for non-XML mode
	cleaned := stripMarkdown(resp.Content, p.GetSignature())
	logger.Debug(ctx, "Normalized completion: %s", cleaned)
	outputs := parseCompletion(cleaned, signature)
	logger.Debug(ctx, "Parsed LLM Completion: %v", outputs)
	formattedOutputs := p.FormatOutputs(outputs)
	logger.Debug(ctx, "Formatted LLM Completion: %v", outputs)

	// Include raw response for XML interceptor processing
	if p.IsXMLModeEnabled() {
		formattedOutputs["__raw_response"] = resp.Content
	}

	span.WithAnnotation("outputs", formattedOutputs)

	return formattedOutputs, nil
}

func (p *Predict) Clone() core.Module {
	cloned := &Predict{
		BaseModule:     *p.BaseModule.Clone().(*core.BaseModule),
		Demos:          append([]core.Example{}, p.Demos...),
		LLM:            p.LLM,
		defaultOptions: p.defaultOptions,
		enableXMLMode:  p.enableXMLMode,
	}

	// Deep copy XML config if present
	if p.xmlConfig != nil {
		configCopy := *p.xmlConfig
		cloned.xmlConfig = &configCopy
	}

	return cloned
}

func (p *Predict) GetDemos() []core.Example {
	return p.Demos
}


// GetXMLConfig returns the XML configuration if XML mode is enabled.
func (p *Predict) GetXMLConfig() *interceptors.XMLConfig {
	if p.enableXMLMode {
		return p.xmlConfig
	}
	return nil
}

func (p *Predict) processWithStreaming(ctx context.Context, inputs map[string]interface{},
	handler core.StreamHandler, opts *core.ModuleOptions) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	// Use semantic name if set, otherwise fall back to operation name
	displayName := p.GetDisplayName()
	if displayName == p.GetModuleType() {
		displayName = "PredictStream"
	}

	metadata := map[string]interface{}{
		"module_type":   p.GetModuleType(),
		"module_config": p.GetSignature().String(),
	}
	ctx, span := core.StartSpanWithContext(ctx, "PredictStream", displayName, metadata)
	defer core.EndSpan(ctx)

	// Validate inputs and build prompt (same as regular Process)
	if err := p.ValidateInputs(inputs); err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "input validation failed"),
			errors.Fields{
				"module": "Predict",
				"inputs": inputs,
			})
	}

	signature := p.GetSignature()
	prompt := formatPrompt(signature, p.Demos, inputs)

	// Use StreamGenerate instead of Generate
	stream, err := p.LLM.StreamGenerate(ctx, prompt, opts.GenerateOptions...)
	if err != nil {
		span.WithError(err)
		return nil, errors.WithFields(
			errors.Wrap(err, errors.LLMGenerationFailed, "failed to generate prediction"),
			errors.Fields{
				"module": "Predict",
				"prompt": prompt,
				"model":  p.LLM,
			})
	}

	// Collect the full response while streaming chunks
	var fullContent strings.Builder
	var tokenUsage core.TokenInfo

	for chunk := range stream.ChunkChannel {
		logger.Debug(ctx, "Predict received chunk: Done=%v, Error=%v, ContentLen=%d",
			chunk.Done, chunk.Error != nil, len(chunk.Content))
		// Update token usage if available
		if chunk.Usage != nil {
			tokenUsage.PromptTokens = chunk.Usage.PromptTokens
			tokenUsage.CompletionTokens += chunk.Usage.CompletionTokens
			tokenUsage.TotalTokens = tokenUsage.PromptTokens + tokenUsage.CompletionTokens
		}

		// Handle errors
		if chunk.Error != nil {
			span.WithError(chunk.Error)
			return nil, chunk.Error
		}

		// Check if done
		if chunk.Done {
			if err := handler(chunk); err != nil {
				stream.Cancel()
				return nil, err
			}
			break
		}

		// Append content to the full response
		fullContent.WriteString(chunk.Content)

		// Call the handler
		if err := handler(chunk); err != nil {
			stream.Cancel() // Cancel the stream
			return nil, err
		}
	}

	logger.Debug(ctx, "Stream channel closed, ensuring Done signal is sent")
	if err := handler(core.StreamChunk{Done: true}); err != nil {
		return nil, err
	}

	// Process the complete response just like in the normal flow
	content := fullContent.String()
	logger.Debug(ctx, "Complete response: %s", content)

	cleaned := stripMarkdown(content, p.GetSignature())
	outputs := parseCompletion(cleaned, signature)
	formattedOutputs := p.FormatOutputs(outputs)

	// Include raw response for XML interceptor processing
	if p.IsXMLModeEnabled() {
		formattedOutputs["__raw_response"] = content
	}

	// Update execution state with token usage
	if state := core.GetExecutionState(ctx); state != nil {
		state.WithTokenUsage(&core.TokenUsage{
			PromptTokens:     tokenUsage.PromptTokens,
			CompletionTokens: tokenUsage.CompletionTokens,
			TotalTokens:      tokenUsage.TotalTokens,
		})
	}

	return formattedOutputs, nil
}

func formatPrompt(signature core.Signature, demos []core.Example, inputs map[string]any) string {
	var sb strings.Builder

	// Write the instruction
	sb.WriteString(fmt.Sprintf("Given the fields '%s', produce the fields '%s'.\n\n",
		joinFieldNames(inputFieldsToFields(signature.Inputs)),
		joinFieldNames(outputFieldsToFields(signature.Outputs)),
	))

	for _, field := range signature.Outputs {
		if field.Prefix != "" {
			sb.WriteString(fmt.Sprintf("The %s field should start with '%s' followed by the content on new lines.\n",
				field.Name, field.Prefix))
		}
		if field.Description != "" {
			sb.WriteString(fmt.Sprintf(" %s", field.Description))
			sb.WriteString("\n")
		}
	}
	sb.WriteString("\n")

	// Add the instruction if present
	if signature.Instruction != "" {
		sb.WriteString(signature.Instruction + "\n\n")
	}
	if context, ok := inputs["conversation_context"].(string); ok && context != "" {
		sb.WriteString("===== CONVERSATION HISTORY =====\n")
		sb.WriteString(context)
		sb.WriteString("\n===== END HISTORY =====\n\n")
	}

	// Write the demonstrations
	for _, demo := range demos {
		sb.WriteString("---\n\n")
		for _, field := range signature.Inputs {
			sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, demo.Inputs[field.Name]))
		}
		for _, field := range signature.Outputs {
			sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, demo.Outputs[field.Name]))
		}
		sb.WriteString("\n")
	}

	// Write the current input
	sb.WriteString("---\n\n")
	for _, field := range signature.Inputs {
		value := inputs[field.Name]

		valueStr := fmt.Sprintf("%v", value)
		// If the value is a ContentBlock of the correct type, use its custom string representation.
		if block, ok := value.(core.ContentBlock); ok && block.Type == field.Type {
			valueStr = block.String()
		}

		sb.WriteString(fmt.Sprintf("%s: %s\n", field.Name, valueStr))
	}

	return sb.String()
}

func stripMarkdown(content string, signature core.Signature) string {
	// First, try to parse as JSON if it looks like JSON
	if strings.Contains(content, "```json") && strings.Contains(content, "```") {
		return parseJSONResponse(content, signature)
	}

	// Fall back to original prefix-based parsing
	// Define a map to track each field's content
	fieldContents := make(map[string][]string)

	// Helper to find which field a prefix belongs to
	findField := func(line string) (string, string) {
		for _, field := range signature.Outputs {
			prefix := strings.TrimSpace(field.Prefix)
			if prefix != "" && strings.HasPrefix(strings.ToLower(line), strings.ToLower(prefix)) {
				// Return the field name and the content after the prefix
				content := strings.TrimPrefix(strings.TrimSpace(line), prefix)
				return field.Name, content
			}
		}
		return "", ""
	}

	// Process the content line by line
	var currentField string
	lines := strings.Split(content, "\n")

	for _, line := range lines {
		// First, clean basic markdown formatting
		cleaned := cleanBasicMarkdown(line)

		// Check if this line starts a new field
		if fieldName, remainingContent := findField(cleaned); fieldName != "" {
			currentField = fieldName
			// If there's content after the prefix, add it
			if remainingContent != "" {
				fieldContents[currentField] = append(fieldContents[currentField], remainingContent)
			}
			continue
		}

		// If we're in a field, add the content
		if currentField != "" {
			// Skip empty lines at the start of a section
			if len(fieldContents[currentField]) == 0 && strings.TrimSpace(cleaned) == "" {
				continue
			}
			fieldContents[currentField] = append(fieldContents[currentField], cleaned)
		}
	}

	// Rebuild the content with clean formatting
	var result strings.Builder
	for _, field := range signature.Outputs {
		content := fieldContents[field.Name]
		if len(content) > 0 {
			// Add the field prefix
			result.WriteString(field.Prefix)
			result.WriteString("\n")

			// Process the content based on its structure
			if isStructuredContent(content[0]) {
				// Preserve structure (like XML or JSON)
				result.WriteString(preserveStructure(content))
			} else {
				// Clean unstructured content
				result.WriteString(cleanUnstructuredContentNew(content))
			}
			result.WriteString("\n\n")
		}
	}

	return strings.TrimSpace(result.String())
}

func cleanBasicMarkdown(line string) string {
	// Remove bold markers
	line = strings.ReplaceAll(line, "**", "")

	// Remove code block markers
	line = strings.ReplaceAll(line, "```xml", "")
	line = strings.ReplaceAll(line, "```markdown", "")
	line = strings.ReplaceAll(line, "```", "")

	// Remove bullet points
	line = strings.TrimPrefix(strings.TrimSpace(line), "-")
	line = strings.TrimPrefix(strings.TrimSpace(line), "*")

	return strings.TrimSpace(line)
}

func isStructuredContent(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "<") ||
		strings.HasPrefix(trimmed, "{") ||
		strings.HasPrefix(trimmed, "[")
}

func preserveStructure(content []string) string {
	// Join with newlines to maintain structure
	return strings.Join(content, "\n")
}

func parseCompletion(completion string, signature core.Signature) map[string]any {
	outputs := make(map[string]any)

	// We'll track sections and their content
	type section struct {
		name       string   // The field name
		startLine  int      // Where the section content begins
		content    []string // The accumulated content lines
		inProgress bool     // Whether we're currently collecting this section
	}

	// Initialize our sections based on signature
	sections := make(map[string]*section)
	for _, field := range signature.Outputs {
		sections[field.Name] = &section{
			name:       field.Name,
			content:    make([]string, 0),
			inProgress: false,
		}
	}

	// Process the content line by line, keeping track of section boundaries
	lines := strings.Split(completion, "\n")
	var currentSection *section

	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}

		// Check if this line starts a new section
		foundNewSection := false
		for _, field := range signature.Outputs {
			prefix := strings.TrimSpace(field.Prefix)
			if prefix != "" && strings.HasPrefix(strings.ToLower(line), strings.ToLower(prefix)) {
				// If we were collecting another section, finalize it
				if currentSection != nil {
					currentSection.inProgress = false
				}

				// Start the new section
				section := sections[field.Name]
				section.startLine = i + 1 // Content starts on next line
				section.inProgress = true
				currentSection = section
				foundNewSection = true
				break
			}
		}

		// If this isn't a section marker and we have a current section, add the content
		if !foundNewSection && currentSection != nil && currentSection.inProgress {
			currentSection.content = append(currentSection.content, line)
		}
	}

	// Process each section's content
	for name, section := range sections {
		if len(section.content) > 0 {
			// Join the content lines while preserving structure
			content := strings.Join(section.content, "\n")

			// Clean up any remaining markdown artifacts
			content = strings.TrimSpace(content)

			// Handle common content patterns (XML, JSON, etc.)
			content = preserveStructuredContent(content)

			outputs[name] = content
		}
	}

	return outputs
}

// preserveStructuredContent handles various types of structured content.
func preserveStructuredContent(content string) string {
	// First, detect if this is structured content by checking common patterns
	trimmed := strings.TrimSpace(content)

	// Handle XML-like content
	if strings.HasPrefix(trimmed, "<") && strings.HasSuffix(trimmed, ">") {
		// Preserve XML indentation and structure
		return preserveXMLStructure(content)
	}

	return cleanUnstructuredContent(content)
}

func preserveXMLStructure(content string) string {
	// Preserve XML formatting while cleaning any artifacts
	lines := strings.Split(content, "\n")
	var cleaned []string

	for _, line := range lines {
		// Preserve indentation
		indent := countLeadingSpaces(line)
		trimmed := strings.TrimSpace(line)

		// Skip empty lines
		if trimmed == "" {
			continue
		}

		// Rebuild the line with proper indentation
		cleaned = append(cleaned, strings.Repeat(" ", indent)+trimmed)
	}

	return strings.Join(cleaned, "\n")
}

func countLeadingSpaces(s string) int {
	return len(s) - len(strings.TrimLeft(s, " \t"))
}

func cleanUnstructuredContent(content string) string {
	// Handle unstructured content (like plain text or lists)
	lines := strings.Split(content, "\n")
	var cleaned []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}

	return strings.Join(cleaned, "\n")
}
func cleanUnstructuredContentNew(content []string) string {
	var cleaned []string
	for _, line := range content {
		if strings.TrimSpace(line) != "" {
			cleaned = append(cleaned, line)
		}
	}
	return strings.Join(cleaned, "\n")
}
func joinFieldNames(fields []core.Field) string {
	names := make([]string, len(fields))
	for i, field := range fields {
		names[i] = field.Name
	}
	return strings.Join(names, ", ")
}

func inputFieldsToFields(inputs []core.InputField) []core.Field {
	fields := make([]core.Field, len(inputs))
	for i, input := range inputs {
		fields[i] = input.Field
	}
	return fields
}

func outputFieldsToFields(outputs []core.OutputField) []core.Field {
	fields := make([]core.Field, len(outputs))
	for i, output := range outputs {
		fields[i] = output.Field
	}
	return fields
}

func (p *Predict) FormatOutputs(outputs map[string]interface{}) map[string]interface{} {
	formattedOutputs := make(map[string]interface{})
	for _, field := range p.GetSignature().Outputs {
		if value, ok := outputs[field.Name]; ok {
			formattedOutputs[field.Name] = value
		}
	}
	return formattedOutputs
}

func (p *Predict) GetSignature() core.Signature {
	return p.BaseModule.GetSignature()
}

func (p *Predict) ValidateInputs(inputs map[string]interface{}) error {
	signature := p.GetSignature()
	for _, field := range signature.Inputs {
		if _, ok := inputs[field.Name]; !ok {
			return fmt.Errorf("missing required input: %s", field.Name)
		}
	}
	return nil
}

func (p *Predict) SetDemos(demos []core.Example) {
	p.Demos = demos
}

func (p *Predict) SetLLM(llm core.LLM) {
	p.LLM = llm
}

// GetLLMIdentifier implements the LMConfigProvider interface.
func (p *Predict) GetLLMIdentifier() map[string]string {
	if p.LLM == nil {
		return nil // Or return an empty map? Depends on desired behavior
	}
	return map[string]string{
		"provider": p.LLM.ProviderName(),
		"model":    p.LLM.ModelID(),
		// Add other identifiers like BaseURL for Ollama if needed/possible
	}
}

// parseJSONResponse extracts JSON content from markdown code blocks and formats it for field parsing.
func parseJSONResponse(content string, signature core.Signature) string {
	// Extract JSON from markdown code blocks
	jsonStart := strings.Index(content, "```json")
	if jsonStart == -1 {
		return content // Fall back to original content if no JSON block found
	}

	jsonStart += len("```json")
	jsonEnd := strings.Index(content[jsonStart:], "```")
	if jsonEnd == -1 {
		return content // Fall back if no closing ```
	}

	jsonContent := strings.TrimSpace(content[jsonStart : jsonStart+jsonEnd])

	// Parse the JSON
	var jsonData map[string]interface{}
	if err := json.Unmarshal([]byte(jsonContent), &jsonData); err != nil {
		return content // Fall back if JSON parsing fails
	}

	// Convert JSON to field prefix format expected by parseCompletion
	var result strings.Builder
	for _, field := range signature.Outputs {
		if value, ok := jsonData[field.Name]; ok {
			// Add the field with its prefix
			result.WriteString(field.Prefix)
			result.WriteString("\n")

			// Add the value, handling both string and other types
			switch v := value.(type) {
			case string:
				result.WriteString(v)
			default:
				result.WriteString(fmt.Sprintf("%v", v))
			}
			result.WriteString("\n\n")
		}
	}

	return strings.TrimSpace(result.String())
}

// ProcessTyped provides type-safe processing with compile-time type validation.
func ProcessTyped[TInput, TOutput any](ctx context.Context, predict *Predict, inputs TInput, opts ...core.Option) (TOutput, error) {
	var zero TOutput

	// Convert typed inputs to legacy format
	legacyInputs, err := utils.ConvertTypedInputsToLegacy(inputs)
	if err != nil {
		return zero, errors.WithFields(
			errors.Wrap(err, errors.InvalidInput, "failed to convert typed inputs"),
			errors.Fields{
				"module": "Predict",
				"type":   fmt.Sprintf("%T", inputs),
			})
	}

	// Call the legacy Process method
	legacyOutputs, err := predict.Process(ctx, legacyInputs, opts...)
	if err != nil {
		return zero, err
	}

	// Convert legacy outputs to typed format
	typedOutputs, err := utils.ConvertLegacyOutputsToTyped[TOutput](legacyOutputs)
	if err != nil {
		return zero, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to convert legacy outputs"),
			errors.Fields{
				"module":  "Predict",
				"type":    fmt.Sprintf("%T", zero),
				"outputs": legacyOutputs,
			})
	}

	return typedOutputs, nil
}

// ProcessTypedWithValidation provides type-safe processing with input and output validation.
func ProcessTypedWithValidation[TInput, TOutput any](ctx context.Context, predict *Predict, inputs TInput, opts ...core.Option) (TOutput, error) {
	var zero TOutput

	// Create typed signature for validation (cached for performance)
	typedSig := core.NewTypedSignatureCached[TInput, TOutput]()

	// Validate inputs
	if err := typedSig.ValidateInput(inputs); err != nil {
		return zero, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "typed input validation failed"),
			errors.Fields{
				"module": "Predict",
				"type":   fmt.Sprintf("%T", inputs),
			})
	}

	// Process with type conversion
	result, err := ProcessTyped[TInput, TOutput](ctx, predict, inputs, opts...)
	if err != nil {
		return zero, err
	}

	// Validate outputs
	if err := typedSig.ValidateOutput(result); err != nil {
		return zero, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "typed output validation failed"),
			errors.Fields{
				"module": "Predict",
				"type":   fmt.Sprintf("%T", result),
			})
	}

	return result, nil
}

// NewTypedPredict creates a new type-safe Predict module from a typed signature.
// Typed modules use text-based parsing by default since they typically rely on prefixes.
func NewTypedPredict[TInput, TOutput any]() *Predict {
	typedSig := core.NewTypedSignatureCached[TInput, TOutput]()
	legacySig := typedSig.ToLegacySignature()

	predict := NewPredict(legacySig).WithTextOutput() // Typed modules use prefix-based parsing
	// Use clearer variable names for type display
	var i TInput
	var o TOutput
	predict.DisplayName = fmt.Sprintf("TypedPredict[%T,%T]", i, o)

	return predict
}

// shouldUseXMLByDefault determines if XML output should be enabled by default
// based on the signature characteristics.
func shouldUseXMLByDefault(signature core.Signature) bool {
	// Enable XML by default for signatures with multiple output fields
	// as they benefit most from structured output
	if len(signature.Outputs) > 1 {
		return true
	}

	// For single output fields, be more conservative to avoid breaking existing code
	// Only enable for fields that clearly suggest structured content
	if len(signature.Outputs) == 1 {
		output := signature.Outputs[0]
		// Check if the field name suggests structured content
		structuredNames := []string{
			"summary", "analysis", "report", "response", "result",
			"evaluation", "assessment", "recommendations", "findings",
			"conclusion", "review", "feedback", "details", "information",
		}

		for _, name := range structuredNames {
			if strings.Contains(strings.ToLower(output.Name), name) {
				return true
			}
		}
	}

	// Default to text mode for backward compatibility with single simple fields
	// Users can opt in with WithXMLOutput() if they want structured output
	return false
}
