package modules

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

type Predict struct {
	core.BaseModule
	Demos          []core.Example
	LLM            core.LLM
	defaultOptions *core.ModuleOptions
}

// Ensure Predict implements core.Module.
var _ core.Module = (*Predict)(nil)

func NewPredict(signature core.Signature) *Predict {
	return &Predict{
		BaseModule: *core.NewModule(signature),
		Demos:      []core.Example{},
		LLM:        core.GetDefaultLLM(),
	}
}

func (p *Predict) WithDefaultOptions(opts ...core.Option) *Predict {
	options := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(options)
	}
	p.defaultOptions = options
	return p
}

func (p *Predict) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	callOptions := &core.ModuleOptions{}
	for _, opt := range opts {
		opt(callOptions)
	}

	finalOptions := p.defaultOptions.MergeWith(callOptions)
	ctx, span := core.StartSpan(ctx, "Predict")
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

	cleaned := stripMarkdown(resp.Content, p.GetSignature())

	logger.Debug(ctx, "Normalized completion: %s", cleaned)
	outputs := parseCompletion(cleaned, signature)

	logger.Debug(ctx, "Parsed LLM Completion: %v", outputs)
	formattedOutputs := p.FormatOutputs(outputs)
	logger.Debug(ctx, "Formatted LLM Completion: %v", outputs)

	span.WithAnnotation("outputs", formattedOutputs)

	return formattedOutputs, nil
}

func (p *Predict) Clone() core.Module {
	return &Predict{
		BaseModule: *p.BaseModule.Clone().(*core.BaseModule),
		Demos:      append([]core.Example{}, p.Demos...),
		LLM:        p.LLM,
	}
}

func (p *Predict) GetDemos() []core.Example {
	return p.Demos
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
	}
	sb.WriteString("\n")

	// Add the instruction if present
	if signature.Instruction != "" {
		sb.WriteString(signature.Instruction + "\n\n")
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
		sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, inputs[field.Name]))
	}

	return sb.String()
}

//	func stripMarkdown(content string) string {
//		// First, split into lines to handle line-by-line patterns
//		lines := strings.Split(content, "\n")
//		var cleanedLines []string
//
//		// Track if we're inside a code block
//		inCodeBlock := false
//		var xmlContent strings.Builder
//
//		for _, line := range lines {
//			// Handle code block boundaries
//			if strings.Contains(line, "```") {
//				inCodeBlock = !inCodeBlock
//				continue // Skip the boundary line
//			}
//
//			// If we're in a code block, collect XML content
//			if inCodeBlock {
//				xmlContent.WriteString(line + "\n")
//				continue
//			}
//
//			// Clean line-level markdown patterns
//			cleaned := line
//
//			// Remove bold markers
//			cleaned = strings.ReplaceAll(cleaned, "**", "")
//
//			// Remove header markers
//			cleaned = strings.TrimLeft(cleaned, "#")
//			cleaned = strings.TrimSpace(cleaned)
//
//			// Remove list markers
//			cleaned = strings.TrimPrefix(cleaned, "- ")
//			cleaned = strings.TrimPrefix(cleaned, "* ")
//			cleaned = strings.TrimPrefix(cleaned, "> ")
//
//			// Only add non-empty lines
//			if cleaned != "" {
//				cleanedLines = append(cleanedLines, cleaned)
//			}
//		}
//
//		// Handle the case where we found XML content
//		if xmlContent.Len() > 0 {
//			return xmlContent.String()
//		}
//
//		// Join cleaned lines back together
//		return strings.TrimSpace(strings.Join(cleanedLines, "\n"))
//	}
func stripMarkdown(content string, signature core.Signature) string {
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

//	func parseCompletion(completion string, signature core.Signature) map[string]any {
//		outputs := make(map[string]any)
//		completion = stripMarkdown(completion)
//		logging.GetLogger().Info(context.Background(), "Normalized completion:  %s", completion)
//
//		lines := strings.Split(strings.TrimSpace(completion), "\n")
//		var currentField *core.OutputField
//		var contentLines []string
//
//		for i, line := range lines {
//			line = strings.TrimSpace(line)
//			if line == "" {
//				continue
//			}
//
//			// Try to match a field prefix
//			for _, field := range signature.Outputs {
//				prefix := strings.TrimSpace(field.Prefix)
//				if prefix != "" && strings.HasPrefix(strings.ToLower(line), strings.ToLower(prefix)) {
//					// If we were collecting content for a previous field, save it
//					if currentField != nil && len(contentLines) > 0 {
//						outputs[currentField.Name] = strings.TrimSpace(strings.Join(contentLines, "\n"))
//					}
//
//					// Start collecting content for this new field
//					currentField = &field
//					contentLines = nil
//					content := strings.TrimPrefix(line, field.Prefix)
//					if content != "" {
//						contentLines = append(contentLines, content)
//					}
//					continue
//				}
//			}
//
//			// If we have a current field and this isn't a prefix line, collect the content
//			if currentField != nil {
//				// Don't add the prefix line itself to the content
//				if !strings.HasPrefix(strings.ToLower(line), strings.ToLower(currentField.Prefix)) {
//					contentLines = append(contentLines, line)
//				}
//			}
//
//			// If this is the last line, save any remaining content
//			if i == len(lines)-1 && currentField != nil && len(contentLines) > 0 {
//				outputs[currentField.Name] = strings.TrimSpace(strings.Join(contentLines, "\n"))
//			}
//		}
//
//		return outputs
//	}
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

	// Handle JSON-like content
	// if (strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
	// 	(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]")) {
	// 	// Preserve JSON formatting
	// 	return preserveJSONStructure(content)
	// }
	//
	// // Handle YAML-like content
	// if strings.Contains(trimmed, ":") && !strings.Contains(trimmed, "<") {
	// 	// Preserve YAML indentation
	// 	return preserveYAMLStructure(content)
	// }

	// For unstructured content, clean up any remaining formatting issues
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
