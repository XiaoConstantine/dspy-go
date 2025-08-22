package models

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/structured"
)

// PromptAnalyzerModel represents the prompt analyzer screen
type PromptAnalyzerModel struct {
	width          int
	height         int
	nextScreen     string
	analyzer       *structured.PromptAnalyzer
	currentPrompt  string
	analysis       []structured.PromptComponent
	showResults    bool
	mode           string // "input" or "results"
	inputCursor    int
	inputLines     []string
	scrollOffset   int
	lastInputLen   int // Track input length to detect paste operations
}

// NewPromptAnalyzerModel creates a new prompt analyzer model
func NewPromptAnalyzerModel() PromptAnalyzerModel {
	return PromptAnalyzerModel{
		width:        80,
		height:       24,
		analyzer:     structured.NewPromptAnalyzer(),
		mode:         "input",
		inputLines:   []string{""},
		inputCursor:  0,
		scrollOffset: 0,
	}
}

// Init initializes the model
func (m PromptAnalyzerModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m PromptAnalyzerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		// Handle global ctrl+c for exit
		if msg.String() == "ctrl+c" {
			m.nextScreen = "back"
			return m, nil
		}

		switch m.mode {
		case "input":
			return m.updateInput(msg)
		case "results":
			return m.updateResults(msg)
		}
	}
	return m, nil
}

// updateInput handles input mode key presses
func (m PromptAnalyzerModel) updateInput(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc":
		m.nextScreen = "back"
		return m, nil
	case "enter":
		// Add new line at current cursor position
		currentLine := m.inputLines[m.inputCursor]
		m.inputLines[m.inputCursor] = currentLine
		// Insert new empty line after current
		newLines := make([]string, len(m.inputLines)+1)
		copy(newLines[:m.inputCursor+1], m.inputLines[:m.inputCursor+1])
		newLines[m.inputCursor+1] = ""
		copy(newLines[m.inputCursor+2:], m.inputLines[m.inputCursor+1:])
		m.inputLines = newLines
		m.inputCursor++
	case "backspace":
		currentLine := m.inputLines[m.inputCursor]
		if len(currentLine) > 0 {
			// Remove last character from current line
			m.inputLines[m.inputCursor] = currentLine[:len(currentLine)-1]
		} else if m.inputCursor > 0 {
			// Remove current empty line and move cursor up
			newLines := make([]string, len(m.inputLines)-1)
			copy(newLines[:m.inputCursor], m.inputLines[:m.inputCursor])
			copy(newLines[m.inputCursor:], m.inputLines[m.inputCursor+1:])
			m.inputLines = newLines
			m.inputCursor--
		}
	case "delete":
		currentLine := m.inputLines[m.inputCursor]
		if len(currentLine) > 0 {
			// For now, just remove last character (simple delete)
			m.inputLines[m.inputCursor] = currentLine[:len(currentLine)-1]
		}
	case "up":
		if m.inputCursor > 0 {
			m.inputCursor--
		}
	case "down":
		if m.inputCursor < len(m.inputLines)-1 {
			m.inputCursor++
		}
	case "ctrl+a":
		// Select all - for now, just move to first line
		m.inputCursor = 0
	case "ctrl+e":
		// End - move to last line
		m.inputCursor = len(m.inputLines) - 1
	case "ctrl+d", "ctrl+enter", "tab":
		// Analyze the prompt
		m.currentPrompt = strings.Join(m.inputLines, "\n")
		if strings.TrimSpace(m.currentPrompt) != "" {
			m.analysis = m.analyzer.AnalyzePrompt(m.currentPrompt)
			m.mode = "results"
			m.showResults = true
		}
	case "ctrl+l":
		// Clear input
		m.inputLines = []string{""}
		m.inputCursor = 0
	default:
		// Handle character input and detect potential paste operations
		msgStr := msg.String()

		// Check for multi-character paste (common when paste contains multiple chars)
		if len(msgStr) > 1 {
			// This might be a paste operation - split into lines
			lines := strings.Split(msgStr, "\n")
			if len(lines) > 1 {
				// Multi-line paste
				currentLine := m.inputLines[m.inputCursor]
				m.inputLines[m.inputCursor] = currentLine + lines[0]

				// Add remaining lines
				for i := 1; i < len(lines); i++ {
					m.inputLines = append(m.inputLines, lines[i])
				}
				m.inputCursor += len(lines) - 1
			} else {
				// Single line paste
				if m.inputCursor < len(m.inputLines) {
					m.inputLines[m.inputCursor] += msgStr
				}
			}
		} else if len(msgStr) == 1 && msgStr != "\x00" {
			// Single character input
			if m.inputCursor < len(m.inputLines) {
				m.inputLines[m.inputCursor] += msgStr
			}
		}
	}
	return m, nil
}

// updateResults handles results mode key presses
func (m PromptAnalyzerModel) updateResults(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "esc", "q":
		m.nextScreen = "back"
		return m, nil
	case "n", "enter":
		// New analysis
		m.mode = "input"
		m.showResults = false
		m.inputLines = []string{""}
		m.inputCursor = 0
		m.currentPrompt = ""
		m.analysis = nil
		m.scrollOffset = 0 // Reset scroll
	case "up", "k":
		if m.scrollOffset > 0 {
			m.scrollOffset--
		}
	case "down", "j":
		m.scrollOffset++
	case "home":
		m.scrollOffset = 0
	case "end":
		// Will be bounded in render function
		m.scrollOffset = 9999
	}
	return m, nil
}

// View renders the prompt analyzer screen
func (m PromptAnalyzerModel) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	switch m.mode {
	case "input":
		return m.renderInputView()
	case "results":
		return m.renderResultsView()
	default:
		return "Unknown mode"
	}
}

// renderInputView renders the input interface
func (m PromptAnalyzerModel) renderInputView() string {
	// Calculate safe dimensions
	safeWidth := max(40, m.width-4)
	safeHeight := max(20, m.height-4)

	var content []string

	// Header
	header := styles.HeroStyle.Render("🔍 Prompt Structure Analyzer")
	content = append(content, header)
	content = append(content, "")

	subtitle := styles.BodyStyle.Render("Enter your prompt below. I'll analyze its structure and suggest improvements.")
	content = append(content, subtitle)
	content = append(content, "")

	// Input area with better sizing
	inputBoxWidth := min(safeWidth-4, 80)
	inputStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("62")).
		Padding(1, 2).
		Width(inputBoxWidth).
		Height(12)

	// Build input content with better display
	var inputLines []string
	hasContent := false

	for i, line := range m.inputLines {
		if strings.TrimSpace(line) != "" {
			hasContent = true
		}

		if i == m.inputCursor {
			cursor := "█"
			if len(line) == 0 {
				inputLines = append(inputLines, "▶ "+cursor)
			} else {
				inputLines = append(inputLines, "▶ "+line+cursor)
			}
		} else {
			inputLines = append(inputLines, "  "+line)
		}
	}

	var inputContent string
	if !hasContent && len(m.inputLines) == 1 && m.inputLines[0] == "" {
		inputContent = styles.InfoStyle.Render("Type your prompt here...\n\nTip: You can paste text using your terminal's paste (usually Ctrl+Shift+V)")
	} else {
		inputContent = strings.Join(inputLines, "\n")
	}

	inputBox := inputStyle.Render(inputContent)
	content = append(content, inputBox)
	content = append(content, "")

	// Instructions with better formatting
	instructions := []string{
		"[↑↓] Navigate lines",
		"[Enter] New line",
		"[Backspace] Delete",
		"[Tab] Analyze",
		"[Ctrl+L] Clear",
		"[Ctrl+C] Exit",
	}

	instructionText := "💡 " + strings.Join(instructions, " • ")
	footer := styles.FooterStyle.Render(instructionText)
	content = append(content, footer)

	// Join content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Apply container styling
	container := lipgloss.NewStyle().
		Width(safeWidth).
		Height(safeHeight).
		Align(lipgloss.Center, lipgloss.Center).
		Render(fullContent)

	return lipgloss.Place(
		m.width,
		m.height,
		lipgloss.Center,
		lipgloss.Center,
		container,
	)
}

// renderResultsView renders the analysis results
func (m PromptAnalyzerModel) renderResultsView() string {
	var content []string

	// Header
	header := styles.HeroStyle.Render("📊 Analysis Results")
	content = append(content, header)
	content = append(content, "")

	// Color legend at the top for immediate reference
	content = append(content, styles.HeadingStyle.Render("🎨 Component Colors:"))
	allComponentTypes := []string{
		"Task Context", "Tone Context", "Background Data", "Task Rules", "Examples",
		"Conversation History", "User Request", "Thinking Steps", "Output Format", "Prefilled Response",
	}

	// Group into rows of 2 for better layout
	for i := 0; i < len(allComponentTypes); i += 2 {
		var row []string
		for j := i; j < i+2 && j < len(allComponentTypes); j++ {
			componentColor := getComponentColor(allComponentTypes[j])
			componentStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(componentColor)).Bold(true)
			coloredName := componentStyle.Render(allComponentTypes[j])
			row = append(row, coloredName)
		}
		legendLine := "  " + strings.Join(row, "  •  ")
		content = append(content, legendLine)
	}
	content = append(content, "")

	// Component detection summary - count unique types, not instances
	componentTypes := make(map[string]bool)
	for _, comp := range m.analysis {
		componentTypes[comp.Type] = true
	}

	foundCount := len(componentTypes)
	totalInstances := len(m.analysis)
	totalCount := 10
	percentage := int(float64(foundCount) / float64(totalCount) * 100)

	summary := fmt.Sprintf("Structure Completeness: %d%% (%d/%d components)",
		percentage, foundCount, totalCount)

	// Show instance details if there are multiple instances
	if totalInstances > foundCount {
		summary += fmt.Sprintf(" • %d total instances found", totalInstances)
	}
	content = append(content, styles.HeadingStyle.Render(summary))
	content = append(content, "")

	// Components found with color coding
	if len(m.analysis) > 0 {
		content = append(content, styles.HeadingStyle.Render("✅ Components Found:"))
		for _, component := range m.analysis {
			componentColor := getComponentColor(component.Type)
			componentStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(componentColor)).Bold(true)

			coloredType := componentStyle.Render(component.Type)
			componentLine := fmt.Sprintf("  • %s: %s",
				coloredType,
				truncateString(component.Content, 50))
			content = append(content, componentLine)
		}
		content = append(content, "")
	}

	// Missing components
	allComponents := []string{
		"Task Context", "Tone Context", "Background Data", "Task Rules",
		"Examples", "Conversation History", "User Request", "Thinking Steps",
		"Output Format", "Prefilled Response",
	}

	foundTypes := make(map[string]bool)
	for _, comp := range m.analysis {
		foundTypes[comp.Type] = true
	}

	var missing []string
	for _, comp := range allComponents {
		if !foundTypes[comp] {
			missing = append(missing, comp)
		}
	}

	if len(missing) > 0 {
		content = append(content, styles.HeadingStyle.Render("⚠️  Missing Components:"))
		for _, comp := range missing {
			content = append(content, styles.InfoStyle.Render("  • "+comp))
		}
		content = append(content, "")
	}

	// Optimization tip
	if percentage < 100 {
		tip := fmt.Sprintf("💡 Tip: Adding %d more components could improve performance by +%d%%",
			len(missing), len(missing)*8)
		content = append(content, styles.InfoStyle.Render(tip))
		content = append(content, "")
	} else {
		tip := "🎉 Excellent! Your prompt has all 10 essential components!"
		if totalInstances > foundCount {
			tip += fmt.Sprintf(" (%d instances show great detail)", totalInstances)
		}
		content = append(content, styles.InfoStyle.Render(tip))
		content = append(content, "")
	}

	// Footer instructions
	instructions := []string{
		"[N] New Analysis",
		"[↑↓] Scroll",
		"[Q] Back to Menu",
	}
	footer := styles.FooterStyle.Render(strings.Join(instructions, "  "))
	content = append(content, footer)

	// Join content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Calculate safe dimensions for consistent box size
	safeWidth := max(40, m.width-4)
	safeHeight := max(20, m.height-4)
	boxWidth := min(safeWidth-4, 80)
	boxHeight := safeHeight - 4

	// Apply scrolling if content is too long
	lines := strings.Split(fullContent, "\n")
	maxLines := boxHeight - 2 // Account for padding

	var scrolledContent string
	if len(lines) > maxLines {
		// Ensure scroll offset is within bounds
		maxScroll := max(0, len(lines)-maxLines)
		if m.scrollOffset > maxScroll {
			m.scrollOffset = maxScroll
		}
		if m.scrollOffset < 0 {
			m.scrollOffset = 0
		}

		start := m.scrollOffset
		end := start + maxLines
		if end > len(lines) {
			end = len(lines)
		}
		scrolledContent = strings.Join(lines[start:end], "\n")

		// Add scroll indicator
		scrollInfo := fmt.Sprintf(" [%d-%d of %d lines]", start+1, end, len(lines))
		scrolledContent += "\n" + styles.InfoStyle.Render(scrollInfo)
	} else {
		scrolledContent = fullContent
		m.scrollOffset = 0 // Reset if not needed
	}

	// Apply fixed-size box
	boxed := styles.BoxStyle.Copy().
		Width(boxWidth).
		Height(boxHeight).
		Render(scrolledContent)

	return lipgloss.Place(
		m.width,
		m.height,
		lipgloss.Center,
		lipgloss.Center,
		boxed,
	)
}

// truncateString truncates a string to specified length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// getComponentColor returns the color for each component type using lipgloss colors
func getComponentColor(componentType string) string {
	colorMap := map[string]string{
		"Task Context":         "#6BB6FF",  // Light blue for identity/role
		"Tone Context":         "#B19CD9",  // Purple for style/tone
		"Background Data":      "#87CEEB",  // Sky blue for information/data
		"Task Rules":          "#FFB6C1",   // Light pink for rules/constraints
		"Examples":            "#98FB98",   // Pale green for demonstrations
		"Conversation History": "#F0E68C",  // Khaki for context/history
		"User Request":        "#DDA0DD",   // Plum for user input
		"Thinking Steps":      "#D3D3D3",   // Light gray for reasoning
		"Output Format":       "#87CEFA",   // Light sky blue for structure
		"Prefilled Response":  "#90EE90",   // Light green for output starters
	}

	if color, exists := colorMap[componentType]; exists {
		return color
	}
	return "#FFFFFF" // Default to white
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// GetNextScreen returns the next screen to navigate to
func (m PromptAnalyzerModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation clears the navigation state
func (m *PromptAnalyzerModel) ResetNavigation() {
	m.nextScreen = ""
}
