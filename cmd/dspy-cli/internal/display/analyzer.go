package display

import (
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/structured"
)

// Signature represents a simplified DSPy signature for display
type Signature struct {
	Inputs      []InputField
	Outputs     []OutputField
	Instruction string
}

type InputField struct {
	Name        string
	Type        string
	Description string
}

type OutputField struct {
	Name        string
	Type        string
	Prefix      string
	Description string
}

// FormatPromptAnalysis formats the prompt analysis results
func FormatPromptAnalysis(prompt string, components []structured.PromptComponent) string {
	var output strings.Builder

	// Header
	output.WriteString(fmt.Sprintf("%s%sPrompt Structure Analysis%s\n", ColorBold, ColorBlue, ColorReset))
	output.WriteString(strings.Repeat("=", 50) + "\n\n")

	// Component detection summary
	output.WriteString(formatComponentSummary(components))
	output.WriteString("\n")

	// Detailed analysis
	output.WriteString(formatDetailedAnalysis(prompt, components))
	output.WriteString("\n")

	// Recommendations
	output.WriteString(formatRecommendations(components))

	return output.String()
}

// formatComponentSummary creates a summary of detected components
func formatComponentSummary(components []structured.PromptComponent) string {
	var output strings.Builder

	componentTypes := make(map[string]bool)
	for _, comp := range components {
		componentTypes[comp.Type] = true
	}

	foundCount := len(componentTypes)
	totalCount := 10
	percentage := float64(foundCount) / float64(totalCount) * 100

	output.WriteString(fmt.Sprintf("%sStructure Completeness:%s %.0f%% (%d/10 components)\n",
		ColorCyan, ColorReset, percentage, foundCount))

	// Component checklist
	componentOrder := []string{
		"Task Context", "Tone Context", "Background Data", "Task Rules",
		"Examples", "Conversation History", "User Request",
		"Thinking Steps", "Output Format", "Prefilled Response",
	}

	output.WriteString("\n" + ColorBold + "Components Found:" + ColorReset + "\n")
	for i, componentType := range componentOrder {
		status := ColorRed + "✗" + ColorReset
		if componentTypes[componentType] {
			status = ColorGreen + "✓" + ColorReset
		}
		output.WriteString(fmt.Sprintf("  %s %d. %s\n", status, i+1, componentType))
	}

	return output.String()
}

// formatDetailedAnalysis shows the prompt with annotations
func formatDetailedAnalysis(prompt string, components []structured.PromptComponent) string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%sDetected Components:%s\n", ColorBold, ColorReset))

	// Group components by type and show content
	componentGroups := make(map[string][]structured.PromptComponent)
	for _, comp := range components {
		componentGroups[comp.Type] = append(componentGroups[comp.Type], comp)
	}

	for componentType, comps := range componentGroups {
		componentColor := getComponentColor(componentType)
		output.WriteString(fmt.Sprintf("\n%s%s%s:%s\n", ColorBold, componentColor, componentType, ColorReset))
		for _, comp := range comps {
			preview := truncateString(comp.Content, 80)
			output.WriteString(fmt.Sprintf("  %s\"%s...\"%s\n", componentColor, preview, ColorReset))
		}
	}

	return output.String()
}

// formatRecommendations shows optimization suggestions
func formatRecommendations(components []structured.PromptComponent) string {
	var output strings.Builder

	componentTypes := make(map[string]bool)
	for _, comp := range components {
		componentTypes[comp.Type] = true
	}

	foundCount := len(componentTypes)
	percentage := float64(foundCount) / float64(10) * 100

	output.WriteString(fmt.Sprintf("%sOptimization Potential:%s\n", ColorBold, ColorReset))

	if percentage < 100 {
		improvement := (100 - percentage) * 0.8
		output.WriteString(fmt.Sprintf("  %sPotential Performance Gain:%s +%.0f%%\n",
			ColorGreen, ColorReset, improvement))

		// Missing components
		missingComponents := []string{}
		allComponents := []string{
			"Task Context", "Tone Context", "Background Data", "Task Rules",
			"Examples", "Conversation History", "User Request",
			"Thinking Steps", "Output Format", "Prefilled Response",
		}

		for _, comp := range allComponents {
			if !componentTypes[comp] {
				missingComponents = append(missingComponents, comp)
			}
		}

		if len(missingComponents) > 0 {
			output.WriteString(fmt.Sprintf("\n%sMissing Components:%s\n", ColorYellow, ColorReset))
			for _, missing := range missingComponents {
				output.WriteString(fmt.Sprintf("  • %s\n", missing))
			}
		}
	} else {
		output.WriteString(fmt.Sprintf("  %s✓ Prompt already has optimal structure!%s\n",
			ColorGreen, ColorReset))
	}

	output.WriteString(fmt.Sprintf("\n%sTip:%s Use 'dspy-cli analyze --optimize' to see the optimized version\n",
		ColorPurple, ColorReset))

	return output.String()
}

// FormatOptimizerRecommendations formats optimizer recommendations based on prompt analysis
func FormatOptimizerRecommendations(recommendations []structured.OptimizerRecommendation) string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%s%s🚀 Optimizer Recommendations%s\n", ColorBold, ColorGreen, ColorReset))
	output.WriteString(strings.Repeat("=", 40) + "\n\n")

	if len(recommendations) == 0 {
		output.WriteString("No specific recommendations available for this prompt structure.\n")
		return output.String()
	}

	for i, rec := range recommendations {
		// Confidence indicator
		confidenceColor := ColorRed
		confidenceIcon := "⚠️"
		if rec.Confidence >= 0.8 {
			confidenceColor = ColorGreen
			confidenceIcon = "✅"
		} else if rec.Confidence >= 0.6 {
			confidenceColor = ColorYellow
			confidenceIcon = "⭐"
		}

		// Header with optimizer name and confidence
		output.WriteString(fmt.Sprintf("%s%d. %s%s %s%s (%.0f%% confidence)%s\n",
			ColorBold, i+1, ColorCyan, rec.Name, confidenceIcon, confidenceColor,
			rec.Confidence*100, ColorReset))

		// Reasoning
		output.WriteString(fmt.Sprintf("   %s\n", rec.Reasoning))

		// Details
		output.WriteString(fmt.Sprintf("   %sComplexity:%s %s | %sCost:%s %s\n",
			ColorPurple, ColorReset, rec.Complexity, ColorPurple, ColorReset, rec.Cost))

		// Best for
		if len(rec.BestFor) > 0 {
			output.WriteString(fmt.Sprintf("   %sBest for:%s %s\n",
				ColorBlue, ColorReset, strings.Join(rec.BestFor, ", ")))
		}

		if i < len(recommendations)-1 {
			output.WriteString("\n")
		}
	}

	output.WriteString(fmt.Sprintf("\n%s💡 Try:%s dspy-cli try <optimizer> --dataset gsm8k --max-examples 5\n",
		ColorPurple, ColorReset))

	return output.String()
}

// FormatSignatureDetails formats DSPy signature information
func FormatSignatureDetails(sig Signature) string {
	var output strings.Builder

	output.WriteString(fmt.Sprintf("%s%sDSPy Signature%s\n", ColorBold, ColorBlue, ColorReset))
	output.WriteString(strings.Repeat("=", 30) + "\n\n")

	// Inputs
	output.WriteString(fmt.Sprintf("%sInputs:%s\n", ColorCyan, ColorReset))
	for _, input := range sig.Inputs {
		output.WriteString(fmt.Sprintf("  • %s%s%s (%s)\n",
			ColorGreen, input.Name, ColorReset, input.Type))
		if input.Description != "" {
			output.WriteString(fmt.Sprintf("    %s\n", truncateString(input.Description, 60)))
		}
	}

	// Outputs
	output.WriteString(fmt.Sprintf("\n%sOutputs:%s\n", ColorCyan, ColorReset))
	for _, output_field := range sig.Outputs {
		prefix := ""
		if output_field.Prefix != "" {
			prefix = fmt.Sprintf(" [%s]", output_field.Prefix)
		}
		output.WriteString(fmt.Sprintf("  • %s%s%s (%s)%s\n",
			ColorGreen, output_field.Name, ColorReset, output_field.Type, prefix))
	}

	// Instruction
	if sig.Instruction != "" {
		output.WriteString(fmt.Sprintf("\n%sInstruction:%s\n", ColorCyan, ColorReset))
		lines := strings.Split(sig.Instruction, "\n")
		for i, line := range lines {
			if i < 3 && strings.TrimSpace(line) != "" {
				output.WriteString(fmt.Sprintf("  %s\n", truncateString(line, 70)))
			}
		}
		if len(lines) > 3 {
			output.WriteString(fmt.Sprintf("  ... (%d more lines)\n", len(lines)-3))
		}
	}

	return output.String()
}

// getComponentColor returns the color code for each component type
func getComponentColor(componentType string) string {
	colorMap := map[string]string{
		"Task Context":         ColorBlue,    // Blue for identity/role
		"Tone Context":         ColorPurple,  // Purple for style/tone
		"Background Data":      ColorCyan,    // Cyan for information/data
		"Task Rules":          ColorRed,      // Red for rules/constraints
		"Examples":            ColorGreen,    // Green for demonstrations
		"Conversation History": ColorYellow,  // Yellow for context/history
		"User Request":        ColorPurple,  // Purple for user input
		"Thinking Steps":      ColorCyan,    // Cyan for reasoning
		"Output Format":       ColorBlue,     // Blue for structure
		"Prefilled Response":  ColorGreen,    // Green for output starters
	}

	if color, exists := colorMap[componentType]; exists {
		return color
	}
	return ColorReset // Default to no color
}

// Helper functions
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
