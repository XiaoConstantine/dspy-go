package models

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
)

// WelcomeModel represents the welcome screen state
type WelcomeModel struct {
	selectedOption int
	options        []TaskOption
	width          int
	height         int
	recommendation string
	nextScreen     string
}

// TaskOption represents a task type the user can select
type TaskOption struct {
	Icon        string
	Title       string
	Description string
	Optimizer   string // Recommended optimizer for this task
}

// NewWelcomeModel creates a new welcome screen model
func NewWelcomeModel() WelcomeModel {
	return WelcomeModel{
		selectedOption: 0,
		width:          80,  // Default width to prevent loading screen
		height:         24,  // Default height to prevent loading screen
		options: []TaskOption{
			{
				Icon:        "🧮",
				Title:       "Math problem solving",
				Description: "Algebra, word problems, step-by-step reasoning",
				Optimizer:   "mipro",
			},
			{
				Icon:        "🤔",
				Title:       "Complex reasoning",
				Description: "Multi-step logic, advanced problem solving",
				Optimizer:   "simba",
			},
			{
				Icon:        "📚",
				Title:       "Question answering",
				Description: "Factual questions, research, information synthesis",
				Optimizer:   "bootstrap",
			},
			{
				Icon:        "🔗",
				Title:       "Multi-module workflows",
				Description: "Complex pipelines, collaborative optimization",
				Optimizer:   "copro",
			},
			{
				Icon:        "🚀",
				Title:       "Quick prototyping",
				Description: "Fast iteration, minimal setup, immediate results",
				Optimizer:   "bootstrap",
			},
			{
				Icon:        "🔬",
				Title:       "Research & experimentation",
				Description: "Cutting-edge optimization, advanced features",
				Optimizer:   "gepa",
			},
			{
				Icon:        "🎲",
				Title:       "Explore everything",
				Description: "Try all optimizers, compare performance",
				Optimizer:   "comparison",
			},
			{
				Icon:        "🔍",
				Title:       "Analyze prompt structure",
				Description: "Visualize and optimize existing prompts",
				Optimizer:   "analyze",
			},
			{
				Icon:        "❓",
				Title:       "Help me decide",
				Description: "Get personalized recommendations",
				Optimizer:   "wizard",
			},
		},
	}
}

// Init initializes the model
func (m WelcomeModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m WelcomeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.selectedOption > 0 {
				m.selectedOption--
			} else {
				// Wrap to bottom
				m.selectedOption = len(m.options) - 1
			}
		case "down", "j":
			if m.selectedOption < len(m.options)-1 {
				m.selectedOption++
			} else {
				// Wrap to top
				m.selectedOption = 0
			}
		case "enter", " ":
			// Process the user's selection and set up the next screen
			m.handleSelection()
			return m, nil
		case "q", "ctrl+c":
			return m, tea.Quit
		}
	}
	return m, nil
}

// handleSelection processes the user's choice and sets up the next screen
func (m *WelcomeModel) handleSelection() {
	selected := m.options[m.selectedOption]

	// Map the selected option to appropriate recommendations and next screen
	switch selected.Optimizer {
	case "mipro":
		m.recommendation = "MIPRO - Perfect for systematic optimization with balanced performance"
		m.nextScreen = "optimizer_detail"
	case "simba":
		m.recommendation = "SIMBA - Advanced introspective optimization for complex reasoning"
		m.nextScreen = "optimizer_detail"
	case "bootstrap":
		m.recommendation = "Bootstrap - Fast and simple, great for getting started quickly"
		m.nextScreen = "optimizer_detail"
	case "copro":
		m.recommendation = "COPRO - Ideal for multi-module collaborative optimization"
		m.nextScreen = "optimizer_detail"
	case "gepa":
		m.recommendation = "GEPA - Cutting-edge evolutionary optimization for research"
		m.nextScreen = "optimizer_detail"
	case "comparison":
		m.recommendation = "Compare all optimizers side-by-side"
		m.nextScreen = "comparison_studio"
	case "wizard":
		m.recommendation = "Let me ask you a few more questions to find the perfect fit"
		m.nextScreen = "recommendation_wizard"
	case "analyze":
		m.recommendation = "Analyze and optimize your prompt structure"
		m.nextScreen = "prompt_analyzer"
	default:
		m.recommendation = selected.Optimizer
		m.nextScreen = "optimizer_detail"
	}
}

// View renders the welcome screen
func (m WelcomeModel) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	// Build the welcome screen
	var content []string

	// Hero section
	hero := styles.HeroStyle.Render("🚀 Welcome to DSPy-Go Optimizer Explorer")
	content = append(content, hero)
	content = append(content, "")

	// Subtitle
	subtitle := styles.BodyStyle.Render("I'm your AI optimization assistant! Let's find the perfect optimizer for your task.")
	content = append(content, subtitle)
	content = append(content, "")

	// Question prompt
	prompt := styles.HeadingStyle.Render("🎯 What best describes your task?")
	content = append(content, prompt)
	content = append(content, "")

	// Options list
	for i, option := range m.options {
		var optionStr string
		if i == m.selectedOption {
			// Selected option with arrow and highlighting
			optionStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s %s",
				styles.IconSelected,
				option.Icon,
				option.Title))
			// Add description on new line, indented
			optionStr += "\n   " + styles.InfoStyle.Render(option.Description)
		} else {
			// Unselected option
			optionStr = styles.UnselectedStyle.Render(fmt.Sprintf("  %s %s",
				option.Icon,
				option.Title))
		}
		content = append(content, optionStr)
	}

	content = append(content, "")
	content = append(content, "")

	// Footer with shortcuts
	shortcuts := []string{
		"[↑↓/jk] Navigate",
		"[Enter] Select",
		"[Tab] Advanced",
		"[?] Help",
		"[q] Quit",
	}
	footer := styles.FooterStyle.Render(strings.Join(shortcuts, "  "))
	content = append(content, footer)

	// Join all content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Apply box styling and center
	boxed := styles.BoxStyle.Copy().
		Width(min(m.width-4, 80)).
		Render(fullContent)

	// Center the box
	return lipgloss.Place(
		m.width,
		m.height,
		lipgloss.Center,
		lipgloss.Center,
		boxed,
	)
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// GetRecommendation returns the recommended optimizer based on selection
func (m WelcomeModel) GetRecommendation() string {
	return m.recommendation
}

// GetNextScreen returns the next screen to navigate to
func (m WelcomeModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation clears the next screen state
func (m *WelcomeModel) ResetNavigation() {
	m.nextScreen = ""
}
