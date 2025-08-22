package structured

import (
	"fmt"
	"strings"

	"github.com/fatih/color"
)

// ComponentStyle defines the visual style for each component
type ComponentStyle struct {
	Number      int
	Name        string
	BGColor     color.Attribute
	FGColor     color.Attribute
	Symbol      string
	Description string
}

// PromptVisualizer creates beautiful visual representations of prompt structures
type PromptVisualizer struct {
	styles []ComponentStyle
}

// NewPromptVisualizer creates a new visualizer with predefined styles
func NewPromptVisualizer() *PromptVisualizer {
	return &PromptVisualizer{
		styles: initializeComponentStyles(),
	}
}

// initializeComponentStyles sets up the visual styles matching the image
func initializeComponentStyles() []ComponentStyle {
	return []ComponentStyle{
		{1, "Task context", color.BgRed, color.FgWhite, "📋", "Role and identity definition"},
		{2, "Tone context", color.BgRed, color.FgWhite, "🎭", "Communication style"},
		{3, "Background data, documents, and images", color.BgGreen, color.FgBlack, "📚", "Supporting information"},
		{4, "Detailed task description & rules", color.BgCyan, color.FgBlack, "📝", "Requirements and constraints"},
		{5, "Examples", color.BgCyan, color.FgBlack, "💡", "Demonstration interactions"},
		{6, "Conversation history", color.BgMagenta, color.FgWhite, "💬", "Previous context"},
		{7, "Immediate task description or request", color.BgYellow, color.FgBlack, "⚡", "Current user query"},
		{8, "Thinking step by step / take a deep breath", color.BgYellow, color.FgBlack, "🧠", "Reasoning guidance"},
		{9, "Output formatting", color.BgBlue, color.FgWhite, "📐", "Response structure"},
		{10, "Prefilled response (if any)", color.BgWhite, color.FgBlack, "✏️", "Template starter"},
	}
}

// VisualizePromptStructure creates a visual representation like the image
func (v *PromptVisualizer) VisualizePromptStructure(prompt string, components []PromptComponent) string {
	var output strings.Builder

	// Clear screen for better visual impact
	output.WriteString("\033[H\033[2J")

	// Title
	titleBar := color.New(color.Bold, color.FgCyan)
	output.WriteString(titleBar.Sprint("╔══════════════════════════════════════════════════════════════════════╗\n"))
	output.WriteString(titleBar.Sprint("║                    PROMPT STRUCTURE VISUALIZATION                    ║\n"))
	output.WriteString(titleBar.Sprint("╚══════════════════════════════════════════════════════════════════════╝\n\n"))

	// Create the visual blocks layout (similar to the image)
	output.WriteString(v.createBlocksVisualization(components))

	// Separator
	output.WriteString("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

	// Show the actual prompt with color-coded highlighting
	output.WriteString(v.createAnnotatedPrompt(prompt, components))

	// Analysis summary
	output.WriteString("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
	output.WriteString(v.createAnalysisSummary(components))

	return output.String()
}

// createBlocksVisualization creates the colored blocks layout
func (v *PromptVisualizer) createBlocksVisualization(components []PromptComponent) string {
	var output strings.Builder

	output.WriteString("📊 PROMPT STRUCTURE COMPONENTS\n")
	output.WriteString("─────────────────────────────────\n\n")

	// Create a map to track which components are present
	presentComponents := make(map[string]bool)
	for _, comp := range components {
		presentComponents[comp.Type] = true
	}

	// Map component types to style numbers
	typeToNumber := map[string]int{
		"Task Context":         1,
		"Tone Context":         2,
		"Background Data":      3,
		"Task Rules":          4,
		"Examples":            5,
		"Conversation History": 6,
		"User Request":        7,
		"Thinking Steps":      8,
		"Output Format":       9,
		"Prefilled Response":  10,
	}

	// Display each component as a colored block
	for _, style := range v.styles {
		// Determine if this component is present
		isPresent := false
		componentType := ""

		for typeName, number := range typeToNumber {
			if number == style.Number {
				componentType = typeName
				isPresent = presentComponents[typeName]
				break
			}
		}

		// Create the colored block
		blockColor := color.New(style.BGColor, style.FGColor, color.Bold)
		dimColor := color.New(color.FgHiBlack)

		if isPresent {
			// Highlighted block for present components
			output.WriteString(fmt.Sprintf("  %s ", style.Symbol))
			output.WriteString(blockColor.Sprintf(" %2d. %-50s ", style.Number, style.Name))
			output.WriteString(color.New(color.FgGreen, color.Bold).Sprint(" ✓\n"))

			// Show extracted content preview
			for _, comp := range components {
				if comp.Type == componentType {
					preview := truncateString(comp.Content, 60)
					output.WriteString(color.New(color.FgHiBlack).Sprintf("      └─ \"%s...\"\n", preview))
					break
				}
			}
		} else {
			// Dimmed block for missing components
			output.WriteString(fmt.Sprintf("  %s ", dimColor.Sprint(style.Symbol)))
			output.WriteString(dimColor.Sprintf(" %2d. %-50s ", style.Number, style.Name))
			output.WriteString(color.New(color.FgRed).Sprint(" ✗\n"))
		}
	}

	return output.String()
}

// createAnnotatedPrompt shows the prompt with inline color annotations
func (v *PromptVisualizer) createAnnotatedPrompt(prompt string, components []PromptComponent) string {
	var output strings.Builder

	output.WriteString("📝 ANNOTATED PROMPT\n")
	output.WriteString("─────────────────────\n\n")

	// Map component types to colors
	typeToStyle := make(map[string]ComponentStyle)
	typeToNumber := map[string]int{
		"Task Context":         1,
		"Tone Context":         2,
		"Background Data":      3,
		"Task Rules":          4,
		"Examples":            5,
		"Conversation History": 6,
		"User Request":        7,
		"Thinking Steps":      8,
		"Output Format":       9,
		"Prefilled Response":  10,
	}

	for typeName, number := range typeToNumber {
		for _, style := range v.styles {
			if style.Number == number {
				typeToStyle[typeName] = style
				break
			}
		}
	}

	// Sort components by position
	sorted := sortComponentsByIndex(components)

	// Create the annotated view
	lastEnd := 0

	for _, comp := range sorted {
		// Print any unmarked text before this component
		if comp.StartIndex > lastEnd {
			unmarked := prompt[lastEnd:comp.StartIndex]
			output.WriteString(color.New(color.FgHiBlack).Sprint(unmarked))
		}

		// Get the style for this component
		if style, ok := typeToStyle[comp.Type]; ok {
			// Create inline annotation
			c := color.New(style.BGColor, style.FGColor, color.Bold)

			// Add component number and symbol
			output.WriteString(c.Sprintf("[%d.%s]", style.Number, style.Symbol))

			// Add the content with background color
			contentColor := color.New(style.BGColor, style.FGColor)
			output.WriteString(contentColor.Sprint(comp.Content))
		} else {
			// Default styling for unknown components
			output.WriteString(fmt.Sprintf("[%s] %s", comp.Type, comp.Content))
		}

		lastEnd = comp.EndIndex
	}

	// Print any remaining text
	if lastEnd < len(prompt) {
		output.WriteString(color.New(color.FgHiBlack).Sprint(prompt[lastEnd:]))
	}

	return output.String()
}

// createAnalysisSummary provides analysis metrics
func (v *PromptVisualizer) createAnalysisSummary(components []PromptComponent) string {
	var output strings.Builder

	output.WriteString("📈 ANALYSIS SUMMARY\n")
	output.WriteString("─────────────────────\n\n")

	// Count present components
	componentTypes := make(map[string]bool)
	for _, comp := range components {
		componentTypes[comp.Type] = true
	}

	foundCount := len(componentTypes)
	totalCount := 10
	percentage := float64(foundCount) / float64(totalCount) * 100

	// Create progress bar
	barLength := 40
	filledLength := int(percentage / 100 * float64(barLength))

	output.WriteString("  Structure Completeness: ")
	output.WriteString("[")

	// Filled portion
	filled := color.New(color.FgGreen, color.Bold)
	for i := 0; i < filledLength; i++ {
		output.WriteString(filled.Sprint("█"))
	}

	// Empty portion
	empty := color.New(color.FgHiBlack)
	for i := filledLength; i < barLength; i++ {
		output.WriteString(empty.Sprint("░"))
	}

	output.WriteString("] ")
	output.WriteString(fmt.Sprintf("%.0f%% (%d/10)\n\n", percentage, foundCount))

	// Component breakdown
	output.WriteString("  Component Analysis:\n")

	// Categories
	categories := []struct {
		name       string
		components []int
		symbol     string
		color      color.Attribute
	}{
		{"Context Setup", []int{1, 2}, "🎯", color.FgRed},
		{"Information", []int{3, 4, 5, 6}, "📚", color.FgGreen},
		{"Task & Logic", []int{7, 8}, "⚡", color.FgYellow},
		{"Output", []int{9, 10}, "📐", color.FgBlue},
	}

	typeToNumber := map[string]int{
		"Task Context":         1,
		"Tone Context":         2,
		"Background Data":      3,
		"Task Rules":          4,
		"Examples":            5,
		"Conversation History": 6,
		"User Request":        7,
		"Thinking Steps":      8,
		"Output Format":       9,
		"Prefilled Response":  10,
	}

	for _, cat := range categories {
		// Count present components in this category
		presentInCategory := 0
		for _, compNum := range cat.components {
			for typeName, num := range typeToNumber {
				if num == compNum && componentTypes[typeName] {
					presentInCategory++
					break
				}
			}
		}

		catColor := color.New(cat.color, color.Bold)
		output.WriteString(fmt.Sprintf("    %s ", cat.symbol))
		output.WriteString(catColor.Sprintf("%-15s", cat.name))
		output.WriteString(fmt.Sprintf(" %d/%d", presentInCategory, len(cat.components)))

		// Mini progress bar
		output.WriteString(" [")
		for i := 0; i < len(cat.components); i++ {
			if i < presentInCategory {
				output.WriteString(color.New(color.FgGreen).Sprint("●"))
			} else {
				output.WriteString(color.New(color.FgRed).Sprint("○"))
			}
		}
		output.WriteString("]\n")
	}

	// Recommendations
	output.WriteString("\n💡 OPTIMIZATION POTENTIAL\n")
	output.WriteString("──────────────────────────\n")

	if percentage < 100 {
		output.WriteString("  Missing components that DSPy can add:\n")

		for _, style := range v.styles {
			found := false
			for typeName, num := range typeToNumber {
				if num == style.Number && componentTypes[typeName] {
					found = true
					break
				}
			}

			if !found {
				improvementColor := color.New(color.FgYellow)
				output.WriteString(improvementColor.Sprintf("    + %s %s\n", style.Symbol, style.Name))
			}
		}

		output.WriteString(fmt.Sprintf("\n  🚀 DSPy can optimize this prompt to achieve ~%.0f%% better performance\n",
			(100-percentage)*0.8)) // Estimate 80% of missing structure translates to performance
	} else {
		successColor := color.New(color.FgGreen, color.Bold)
		output.WriteString(successColor.Sprint("  ✅ This prompt already has optimal structure!\n"))
	}

	return output.String()
}

// CreateInteractiveVisualization creates an interactive view for the CLI
func (v *PromptVisualizer) CreateInteractiveVisualization(prompt string) string {
	analyzer := NewPromptAnalyzer()
	components := analyzer.AnalyzePrompt(prompt)
	return v.VisualizePromptStructure(prompt, components)
}
