package structured

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/fatih/color"
)

// PromptComponent represents a detected component in the prompt
type PromptComponent struct {
	Type       string
	Content    string
	StartIndex int
	EndIndex   int
	Confidence float64
}

// ComponentPattern defines patterns for detecting prompt components
type ComponentPattern struct {
	Name     string
	Patterns []string
	Keywords []string
	Color    color.Attribute
}

// PromptAnalyzer analyzes prompts and maps them to the 10-component structure
type PromptAnalyzer struct {
	patterns []ComponentPattern
}

// NewPromptAnalyzer creates a new prompt analyzer
func NewPromptAnalyzer() *PromptAnalyzer {
	return &PromptAnalyzer{
		patterns: initializePatterns(),
	}
}

// initializePatterns sets up the detection patterns for each component
func initializePatterns() []ComponentPattern {
	return []ComponentPattern{
		{
			Name: "Task Context",
			Patterns: []string{
				`(?i)you\s+are\s+(?:a|an)\s+[\w\s]+`,
				`(?i)act\s+as\s+(?:a|an)\s+[\w\s]+`,
				`(?i)you\s+will\s+be\s+(?:acting|playing|serving)\s+as`,
				`(?i)your\s+role\s+is`,
			},
			Keywords: []string{"role", "acting as", "you are", "identity", "persona"},
			Color:    color.FgYellow,
		},
		{
			Name: "Tone Context",
			Patterns: []string{
				`(?i)(?:maintain|use|adopt)\s+a\s+[\w\s]*\s*(?:friendly|professional|formal|casual|helpful)[\w\s]*tone`,
				`(?i)(?:friendly|professional|formal|casual|helpful)[\w\s]*(?:tone|style|manner)`,
				`(?i)communication\s+style`,
				`(?i)tone[\w\s]*(?:friendly|professional|formal|casual|helpful)`,
			},
			Keywords: []string{"tone", "friendly", "professional", "customer service", "style", "manner"},
			Color:    color.FgCyan,
		},
		{
			Name: "Background Data",
			Patterns: []string{
				`(?i)(?:given|using|with)\s+(?:the\s+)?(?:following\s+)?(?:data|information|context)`,
				`(?i)reference\s+(?:document|material|guide)`,
				`(?i)here\s+(?:is|are)\s+(?:the|some)\s+[\w\s]+\s+(?:data|information)`,
			},
			Keywords: []string{"document", "data", "information", "context", "reference", "guide"},
			Color:    color.FgGreen,
		},
		{
			Name: "Task Rules",
			Patterns: []string{
				`(?i)(?:important\s+)?rules?\s+for\s+the\s+interaction`,
				`(?i)(?:important\s+)?rules?:`,
				`(?i)always\s+stay\s+in\s+character`,
				`(?i)(?:always|never|must|should|don't|do\s+not)`,
				`(?i)(?:ensure|make\s+sure)\s+(?:that|to)`,
				`(?i)constraints?:`,
			},
			Keywords: []string{"rules", "always", "never", "must", "should", "ensure", "constraint", "interaction"},
			Color:    color.FgRed,
		},
		{
			Name: "Examples",
			Patterns: []string{
				`(?i)(?:for\s+)?example:`,
				`(?i)(?:here|these)\s+(?:is|are)\s+(?:an?\s+)?examples?`,
				`(?i)<example>`,
				`(?i)user:\s*[\w\s]+\s*(?:assistant|ai|bot):`,
			},
			Keywords: []string{"example", "e.g.", "for instance", "such as", "demonstration"},
			Color:    color.FgMagenta,
		},
		{
			Name: "Conversation History",
			Patterns: []string{
				`(?i)(?:conversation|chat)\s+history`,
				`(?i)previous\s+(?:messages|interactions|context)`,
				`(?i)(?:here\s+is|below\s+is)\s+(?:the\s+)?(?:conversation|history)`,
			},
			Keywords: []string{"history", "previous", "conversation", "context", "earlier"},
			Color:    color.FgBlue,
		},
		{
			Name: "User Request",
			Patterns: []string{
				`(?i)(?:user\s+)?(?:question|request|query|asks?):\s*`,
				`(?i)(?:please|can\s+you|could\s+you|would\s+you)`,
				`(?i)(?:help\s+me|I\s+need|I\s+want|I'd\s+like)`,
			},
			Keywords: []string{"question", "request", "query", "help", "please", "need"},
			Color:    color.FgWhite,
		},
		{
			Name: "Thinking Steps",
			Patterns: []string{
				`(?i)think\s+about\s+your\s+answer`,
				`(?i)(?:think|reason)\s+(?:step\s+by\s+step|through)`,
				`(?i)(?:first|then|finally|before)\s+you\s+respond`,
				`(?i)(?:break\s+down|analyze|consider)`,
				`(?i)take\s+a\s+deep\s+breath`,
			},
			Keywords: []string{"think", "step", "reason", "analyze", "consider", "first", "before you respond", "think about"},
			Color:    color.FgYellow,
		},
		{
			Name: "Output Format",
			Patterns: []string{
				`(?i)put\s+your\s+response\s+in`,
				`(?i)(?:format|structure)\s+(?:your\s+)?(?:response|output)`,
				`(?i)(?:provide|give|return)\s+(?:the\s+)?(?:answer|response)\s+(?:in|as)`,
				`(?i)(?:use|follow)\s+(?:this|the\s+following)\s+(?:format|structure)`,
				`<[\w]+>[^<]*</[\w]+>`,
			},
			Keywords: []string{"format", "structure", "output", "response", "template", "tags"},
			Color:    color.FgGreen,
		},
		{
			Name: "Prefilled Response",
			Patterns: []string{
				`(?i)(?:start\s+with|begin\s+with)`,
				`(?i)(?:here's|here\s+is)\s+(?:a\s+)?(?:template|starter)`,
				`(?i)assistant:\s*$`,
			},
			Keywords: []string{"start with", "begin", "template", "prefilled"},
			Color:    color.FgCyan,
		},
	}
}

// AnalyzePrompt analyzes a prompt and identifies its components
func (pa *PromptAnalyzer) AnalyzePrompt(prompt string) []PromptComponent {
	components := []PromptComponent{}
	lines := strings.Split(prompt, "\n")

	// Track which parts have been identified
	identified := make([]bool, len(prompt))

	for _, pattern := range pa.patterns {
		// Check patterns
		for _, p := range pattern.Patterns {
			re, err := regexp.Compile(p)
			if err != nil {
				continue
			}

			matches := re.FindAllStringIndex(prompt, -1)
			for _, match := range matches {
				// Skip if this part was already identified
				alreadyIdentified := false
				for i := match[0]; i < match[1]; i++ {
					if identified[i] {
						alreadyIdentified = true
						break
					}
				}
				if alreadyIdentified {
					continue
				}

				// Mark as identified
				for i := match[0]; i < match[1]; i++ {
					identified[i] = true
				}

				components = append(components, PromptComponent{
					Type:       pattern.Name,
					Content:    prompt[match[0]:match[1]],
					StartIndex: match[0],
					EndIndex:   match[1],
					Confidence: 0.8,
				})
			}
		}

		// Check keywords
		for _, keyword := range pattern.Keywords {
			idx := strings.Index(strings.ToLower(prompt), strings.ToLower(keyword))
			if idx != -1 {
				// Find the sentence containing the keyword
				start := idx
				end := idx + len(keyword)

				// Expand to sentence boundaries
				for start > 0 && prompt[start-1] != '.' && prompt[start-1] != '\n' {
					start--
				}
				for end < len(prompt) && prompt[end] != '.' && prompt[end] != '\n' {
					end++
				}

				// Skip if already identified
				alreadyIdentified := false
				for i := start; i < end; i++ {
					if i < len(identified) && identified[i] {
						alreadyIdentified = true
						break
					}
				}
				if alreadyIdentified {
					continue
				}

				// Mark as identified with lower confidence
				for i := start; i < end && i < len(identified); i++ {
					identified[i] = true
				}

				if start < len(prompt) && end <= len(prompt) {
					components = append(components, PromptComponent{
						Type:       pattern.Name,
						Content:    strings.TrimSpace(prompt[start:end]),
						StartIndex: start,
						EndIndex:   end,
						Confidence: 0.5,
					})
				}
			}
		}
	}

	// Handle unidentified parts as potential user requests or general content
	for i, lineContent := range lines {
		lineStart := 0
		for j := 0; j < i; j++ {
			lineStart += len(lines[j]) + 1 // +1 for newline
		}
		lineEnd := lineStart + len(lineContent)

		// Check if this line is unidentified
		isUnidentified := true
		for j := lineStart; j < lineEnd && j < len(identified); j++ {
			if identified[j] {
				isUnidentified = false
				break
			}
		}

		if isUnidentified && strings.TrimSpace(lineContent) != "" {
			components = append(components, PromptComponent{
				Type:       "User Request", // Default unidentified content to user request
				Content:    lineContent,
				StartIndex: lineStart,
				EndIndex:   lineEnd,
				Confidence: 0.3,
			})
		}
	}

	return components
}

// ConvertToSignature converts analyzed components to a DSPy signature
func (pa *PromptAnalyzer) ConvertToSignature(components []PromptComponent) core.Signature {
	// Group components by type
	componentGroups := make(map[string][]PromptComponent)
	for _, comp := range components {
		componentGroups[comp.Type] = append(componentGroups[comp.Type], comp)
	}

	// Build input fields based on detected components
	inputs := []core.InputField{}
	outputs := []core.OutputField{}

	// Map components to input fields
	componentToField := map[string]string{
		"Task Context":         "role_context",
		"Tone Context":         "tone_guidelines",
		"Background Data":      "background_info",
		"Task Rules":          "constraints",
		"Examples":            "demonstrations",
		"Conversation History": "chat_history",
		"User Request":        "user_query",
		"Thinking Steps":      "reasoning_steps",
	}

	for componentType, fieldName := range componentToField {
		if comps, ok := componentGroups[componentType]; ok {
			// Combine all content for this component type
			content := []string{}
			for _, comp := range comps {
				content = append(content, comp.Content)
			}

			inputs = append(inputs, core.InputField{
				Field: core.Field{
					Name:        fieldName,
					Type:        "string",
					Description: fmt.Sprintf("%s extracted from prompt", componentType),
				},
			})
		}
	}

	// Add default input if no specific inputs found
	if len(inputs) == 0 {
		inputs = append(inputs, core.InputField{
			Field: core.Field{
				Name:        "prompt",
				Type:        "string",
				Description: "The input prompt to process",
			},
		})
	}

	// Define outputs based on detected output format
	if _, hasFormat := componentGroups["Output Format"]; hasFormat {
		outputs = append(outputs, core.OutputField{
			Field: core.Field{
				Name:        "structured_response",
				Type:        "string",
				Prefix:      "Response:",
				Description: "Structured response following the specified format",
			},
		})
	} else {
		outputs = append(outputs, core.OutputField{
			Field: core.Field{
				Name:        "response",
				Type:        "string",
				Prefix:      "Response:",
				Description: "Generated response",
			},
		})
	}

	// Build instruction from components
	instruction := buildInstructionFromComponents(componentGroups)

	return core.Signature{
		Inputs:      inputs,
		Outputs:     outputs,
		Instruction: instruction,
	}
}

// OptimizeToFullStructure takes a signature and optimizes it to include all 10 components
func (pa *PromptAnalyzer) OptimizeToFullStructure(ctx context.Context, sig core.Signature) core.Signature {
	// Analyze current signature to understand what components are present
	existingComponents := pa.analyzeSignatureComponents(sig)
	missingComponents := pa.findMissingComponents(existingComponents)

	// TODO(human): Implement the intelligent optimization logic
	// This function should intelligently generate content for missing components
	// based on the existing components and component relationships
	// Consider:
	// - Component interdependencies (e.g., tone should match task context)
	// - Domain-specific requirements based on detected task type
	// - Appropriate sophistication level based on existing complexity
	optimizedSig := pa.generateOptimizedSignature(sig, existingComponents, missingComponents)

	return optimizedSig
}

// Helper method to analyze which components are present in current signature
func (pa *PromptAnalyzer) analyzeSignatureComponents(sig core.Signature) map[string]bool {
	components := make(map[string]bool)

	// Check input fields for component presence
	for _, input := range sig.Inputs {
		switch input.Name {
		case "task_context":
			components["Task Context"] = true
		case "tone_context":
			components["Tone Context"] = true
		case "background_data":
			components["Background Data"] = true
		case "task_rules":
			components["Task Rules"] = true
		case "examples":
			components["Examples"] = true
		case "conversation_history":
			components["Conversation History"] = true
		case "user_request":
			components["User Request"] = true
		case "thinking_steps":
			components["Thinking Steps"] = true
		}
	}

	// Check instruction for output format and prefilled response indicators
	instruction := sig.Instruction
	if strings.Contains(strings.ToLower(instruction), "format") ||
	   strings.Contains(strings.ToLower(instruction), "structure") ||
	   strings.Contains(strings.ToLower(instruction), "json") ||
	   strings.Contains(strings.ToLower(instruction), "xml") {
		components["Output Format"] = true
	}

	if strings.Contains(strings.ToLower(instruction), "start with") ||
	   strings.Contains(strings.ToLower(instruction), "begin by") ||
	   strings.Contains(strings.ToLower(instruction), "response:") {
		components["Prefilled Response"] = true
	}

	return components
}

// Helper method to identify missing components
func (pa *PromptAnalyzer) findMissingComponents(existing map[string]bool) []string {
	allComponents := []string{
		"Task Context", "Tone Context", "Background Data", "Task Rules",
		"Examples", "Conversation History", "User Request", "Thinking Steps",
		"Output Format", "Prefilled Response",
	}

	missing := []string{}
	for _, component := range allComponents {
		if !existing[component] {
			missing = append(missing, component)
		}
	}

	return missing
}

func (pa *PromptAnalyzer) generateOptimizedSignature(original core.Signature, existing map[string]bool, missing []string) core.Signature {
	// Start with original inputs and outputs
	optimizedInputs := make([]core.InputField, len(original.Inputs))
	copy(optimizedInputs, original.Inputs)

	optimizedOutputs := make([]core.OutputField, len(original.Outputs))
	copy(optimizedOutputs, original.Outputs)

	// Detect domain and complexity from original signature
	domain := pa.detectDomain(original)
	complexity := pa.detectComplexity(original)
	tone := pa.detectTone(original)

	// Add missing input components intelligently
	for _, missingComponent := range missing {
		switch missingComponent {
		case "Task Context":
			if !pa.hasInputField(optimizedInputs, "task_context") {
				content := pa.generateTaskContext(domain, complexity)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "task_context", Type: core.FieldTypeText, Description: content},
				})
			}
		case "Tone Context":
			if !pa.hasInputField(optimizedInputs, "tone_context") {
				content := pa.generateToneContext(tone, domain)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "tone_context", Type: core.FieldTypeText, Description: content},
				})
			}
		case "Background Data":
			if !pa.hasInputField(optimizedInputs, "background_data") {
				content := pa.generateBackgroundContext(domain)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "background_data", Type: core.FieldTypeText, Description: content},
				})
			}
		case "Task Rules":
			if !pa.hasInputField(optimizedInputs, "task_rules") {
				content := pa.generateTaskRules(domain, complexity)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "task_rules", Type: core.FieldTypeText, Description: content},
				})
			}
		case "Examples":
			if !pa.hasInputField(optimizedInputs, "examples") {
				content := pa.generateExamplesContext(domain)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "examples", Type: core.FieldTypeText, Description: content},
				})
			}
		case "Conversation History":
			if !pa.hasInputField(optimizedInputs, "conversation_history") {
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "conversation_history", Type: core.FieldTypeText, Description: "Previous conversation context"},
				})
			}
		case "User Request":
			if !pa.hasInputField(optimizedInputs, "user_request") {
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "user_request", Type: core.FieldTypeText, Description: "Current user query or task"},
				})
			}
		case "Thinking Steps":
			if !pa.hasInputField(optimizedInputs, "thinking_steps") {
				content := pa.generateThinkingSteps(complexity)
				optimizedInputs = append(optimizedInputs, core.InputField{
					Field: core.Field{Name: "thinking_steps", Type: core.FieldTypeText, Description: content},
				})
			}
		}
	}

	// Ensure we have appropriate outputs
	if len(optimizedOutputs) == 0 {
		optimizedOutputs = append(optimizedOutputs, core.OutputField{
			Field: core.Field{Name: "response", Type: core.FieldTypeText, Prefix: "Response:"},
		})
	}

	// Generate optimized instruction
	optimizedInstruction := pa.generateOptimizedInstruction(original.Instruction, domain, complexity, tone, missing)

	return core.Signature{
		Inputs:      optimizedInputs,
		Outputs:     optimizedOutputs,
		Instruction: optimizedInstruction,
	}
}

// Helper method to check if input field exists
func (pa *PromptAnalyzer) hasInputField(inputs []core.InputField, fieldName string) bool {
	for _, input := range inputs {
		if input.Name == fieldName {
			return true
		}
	}
	return false
}

// Domain detection from signature content
func (pa *PromptAnalyzer) detectDomain(sig core.Signature) string {
	content := strings.ToLower(sig.Instruction)
	for _, input := range sig.Inputs {
		content += " " + strings.ToLower(input.Description)
	}

	if strings.Contains(content, "code") || strings.Contains(content, "programming") ||
	   strings.Contains(content, "developer") || strings.Contains(content, "technical") {
		return "technical"
	}
	if strings.Contains(content, "creative") || strings.Contains(content, "writing") ||
	   strings.Contains(content, "story") || strings.Contains(content, "design") {
		return "creative"
	}
	if strings.Contains(content, "business") || strings.Contains(content, "analysis") ||
	   strings.Contains(content, "strategy") || strings.Contains(content, "finance") {
		return "business"
	}
	if strings.Contains(content, "education") || strings.Contains(content, "teaching") ||
	   strings.Contains(content, "learning") || strings.Contains(content, "explain") {
		return "educational"
	}
	return "general"
}

// Complexity detection from signature structure
func (pa *PromptAnalyzer) detectComplexity(sig core.Signature) string {
	inputCount := len(sig.Inputs)
	instructionLength := len(sig.Instruction)

	if inputCount >= 5 || instructionLength > 200 {
		return "high"
	}
	if inputCount >= 3 || instructionLength > 100 {
		return "medium"
	}
	return "low"
}

// Tone detection from signature content
func (pa *PromptAnalyzer) detectTone(sig core.Signature) string {
	content := strings.ToLower(sig.Instruction)

	if strings.Contains(content, "professional") || strings.Contains(content, "formal") {
		return "professional"
	}
	if strings.Contains(content, "friendly") || strings.Contains(content, "casual") {
		return "friendly"
	}
	if strings.Contains(content, "helpful") || strings.Contains(content, "supportive") {
		return "helpful"
	}
	return "balanced"
}

// Generate context-aware task context
func (pa *PromptAnalyzer) generateTaskContext(domain, complexity string) string {
	baseContext := "Define your role and identity for this task"

	switch domain {
	case "technical":
		return "You are a technical expert providing accurate, detailed guidance on programming and technology topics"
	case "creative":
		return "You are a creative professional helping with writing, design, and artistic endeavors"
	case "business":
		return "You are a business consultant providing strategic advice and analysis"
	case "educational":
		return "You are an educational assistant helping explain concepts clearly and effectively"
	default:
		return baseContext
	}
}

// Generate tone context based on detected tone and domain
func (pa *PromptAnalyzer) generateToneContext(tone, domain string) string {
	switch tone {
	case "professional":
		return "Maintain a professional, formal tone appropriate for business communications"
	case "friendly":
		return "Use a warm, approachable tone that makes users feel comfortable and supported"
	case "helpful":
		return "Adopt a helpful, supportive tone focused on solving problems and providing value"
	default:
		return "Use a balanced tone that is both professional and approachable"
	}
}

// Generate background context based on domain
func (pa *PromptAnalyzer) generateBackgroundContext(domain string) string {
	switch domain {
	case "technical":
		return "Relevant technical documentation, code examples, or system specifications"
	case "creative":
		return "Creative briefs, style guides, or inspirational references"
	case "business":
		return "Market data, business requirements, or strategic context"
	case "educational":
		return "Learning objectives, student background, or curriculum context"
	default:
		return "Supporting documents, references, or contextual information"
	}
}

// Generate task rules based on domain and complexity
func (pa *PromptAnalyzer) generateTaskRules(domain, complexity string) string {
	baseRules := "Follow these specific constraints and requirements"

	if complexity == "high" {
		return baseRules + ": Provide comprehensive analysis, consider edge cases, validate assumptions"
	}
	if complexity == "medium" {
		return baseRules + ": Be thorough but concise, focus on key points"
	}
	return baseRules + ": Keep responses clear and direct"
}

// Generate examples context based on domain
func (pa *PromptAnalyzer) generateExamplesContext(domain string) string {
	switch domain {
	case "technical":
		return "Code samples, technical demonstrations, or implementation examples"
	case "creative":
		return "Creative samples, style examples, or inspirational works"
	case "business":
		return "Case studies, business examples, or analytical samples"
	case "educational":
		return "Learning examples, practice problems, or educational demonstrations"
	default:
		return "Relevant examples that demonstrate the desired approach or output"
	}
}

// Generate thinking steps based on complexity
func (pa *PromptAnalyzer) generateThinkingSteps(complexity string) string {
	switch complexity {
	case "high":
		return "Follow systematic reasoning: analyze, plan, execute, validate, refine"
	case "medium":
		return "Use structured thinking: understand, plan, implement, review"
	default:
		return "Think step by step to ensure accuracy and completeness"
	}
}

// Generate optimized instruction that incorporates missing components
func (pa *PromptAnalyzer) generateOptimizedInstruction(original, domain, complexity, tone string, missing []string) string {
	if original != "" && len(missing) == 0 {
		return original
	}

	instruction := "Using the provided context"

	// Add component-specific instruction elements
	for _, missingComponent := range missing {
		switch missingComponent {
		case "Task Context":
			instruction += ", following your defined role"
		case "Tone Context":
			instruction += ", maintaining the specified tone"
		case "Background Data":
			instruction += ", referencing relevant background information"
		case "Task Rules":
			instruction += ", adhering to all specified rules and constraints"
		case "Examples":
			instruction += ", using provided examples as guidance"
		case "Thinking Steps":
			instruction += ", applying systematic reasoning"
		case "Output Format":
			instruction += ", formatting output according to specifications"
		}
	}

	instruction += ", generate a comprehensive response that addresses the user request effectively."

	// Add complexity-specific guidance
	if complexity == "high" {
		instruction += " Provide detailed analysis and consider multiple perspectives."
	} else if complexity == "medium" {
		instruction += " Balance thoroughness with clarity."
	} else {
		instruction += " Keep the response clear and direct."
	}

	return instruction
}

// Helper functions
func sortComponentsByIndex(components []PromptComponent) []PromptComponent {
	// Simple bubble sort for component ordering
	sorted := make([]PromptComponent, len(components))
	copy(sorted, components)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].StartIndex < sorted[i].StartIndex {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	return sorted
}

func buildInstructionFromComponents(groups map[string][]PromptComponent) string {
	var instruction strings.Builder

	// Add task context if present
	if comps, ok := groups["Task Context"]; ok {
		for _, comp := range comps {
			instruction.WriteString(comp.Content)
			instruction.WriteString("\n")
		}
	}

	// Add tone context
	if comps, ok := groups["Tone Context"]; ok {
		for _, comp := range comps {
			instruction.WriteString(comp.Content)
			instruction.WriteString("\n")
		}
	}

	// Add rules
	if comps, ok := groups["Task Rules"]; ok {
		instruction.WriteString("\nRules:\n")
		for _, comp := range comps {
			instruction.WriteString("• ")
			instruction.WriteString(comp.Content)
			instruction.WriteString("\n")
		}
	}

	// Add thinking guidance
	if comps, ok := groups["Thinking Steps"]; ok {
		instruction.WriteString("\nApproach:\n")
		for _, comp := range comps {
			instruction.WriteString(comp.Content)
			instruction.WriteString("\n")
		}
	}

	return instruction.String()
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}
