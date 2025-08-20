package models

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
)

// WizardModel represents the AI-powered recommendation wizard
type WizardModel struct {
	currentQuestion int
	selectedOption  int
	answers         map[string]interface{}
	questions       []Question
	recommendation  *Recommendation
	width           int
	height          int
	completed       bool
	nextScreen      string
}

// Question represents a single question in the wizard
type Question struct {
	ID          string
	Text        string
	Type        QuestionType
	Options     []string
	Description string
}

// QuestionType defines the type of question
type QuestionType int

const (
	QuestionTypeChoice QuestionType = iota
	QuestionTypeScale
)

// Recommendation represents the final recommendation from the wizard
type Recommendation struct {
	Optimizer     string
	Confidence    int
	Reasoning     string
	Alternatives  []string
	Configuration map[string]interface{}
}

// NewWizardModel creates a new wizard model
func NewWizardModel() WizardModel {
	return WizardModel{
		currentQuestion: 0,
		answers:         make(map[string]interface{}),
		width:           80,
		height:          24,
		completed:       false,
		nextScreen:      "config",
		questions: []Question{
			{
				ID:          "task_type",
				Text:        "What type of task are you optimizing?",
				Type:        QuestionTypeChoice,
				Description: "Different optimizers excel at different types of problems",
				Options: []string{
					"Math & calculations",
					"Question answering",
					"Complex reasoning",
					"Creative writing",
					"Data analysis",
					"Code generation",
				},
			},
			{
				ID:          "expertise",
				Text:        "How experienced are you with prompt optimization?",
				Type:        QuestionTypeChoice,
				Description: "This helps us recommend the right complexity level",
				Options: []string{
					"Beginner - I'm just getting started",
					"Intermediate - I know the basics",
					"Advanced - I want full control",
					"Expert - Show me everything",
				},
			},
			{
				ID:          "priority",
				Text:        "What's most important for your use case?",
				Type:        QuestionTypeChoice,
				Description: "Different optimizers balance speed vs accuracy differently",
				Options: []string{
					"Speed - I need fast results",
					"Balanced - Good mix of speed and quality",
					"Accuracy - I need the best possible results",
					"Research - Cutting-edge experimental features",
				},
			},
			{
				ID:          "data_size",
				Text:        "How much training data do you have available?",
				Type:        QuestionTypeChoice,
				Description: "Some optimizers work better with larger datasets",
				Options: []string{
					"Small (< 50 examples)",
					"Medium (50-200 examples)",
					"Large (200+ examples)",
					"Very large (1000+ examples)",
				},
			},
			{
				ID:          "time_budget",
				Text:        "How much time can you spend on optimization?",
				Type:        QuestionTypeChoice,
				Description: "This affects which optimization strategy we recommend",
				Options: []string{
					"Quick (< 5 minutes)",
					"Standard (5-30 minutes)",
					"Thorough (30+ minutes)",
					"Research (hours)",
				},
			},
		},
	}
}

// Init initializes the wizard model
func (m WizardModel) Init() tea.Cmd {
	return nil
}

// Update handles messages for the wizard
func (m WizardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		if m.completed {
			// Handle completed state
			switch msg.String() {
			case "enter", " ":
				// Set nextScreen for app to detect navigation
				m.nextScreen = "config"
				return m, nil
			case "b", "esc":
				// Set nextScreen for app to detect navigation
				m.nextScreen = "welcome"
				return m, nil
			}
			return m, nil
		}

		// Handle question navigation
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "b", "esc":
			if m.currentQuestion > 0 {
				m.currentQuestion--
			} else {
				// Set nextScreen for app to detect navigation
				m.nextScreen = "welcome"
				return m, nil
			}
		case "enter", " ":
			// TODO(human): Implement answer selection logic
			// This function should handle the user's selection and advance to next question
			// When all questions are answered, call m.generateRecommendation()
			//
			// Guidance: Track selected option index, store answer in m.answers map,
			// increment m.currentQuestion, and check if we're done
			return m, nil
		case "up", "k":
			// Handle option selection (implement selection state)
		case "down", "j":
			// Handle option selection (implement selection state)
		}
	}

	return m, nil
}

// View renders the wizard interface
func (m WizardModel) View() string {
	if m.completed && m.recommendation != nil {
		return m.renderRecommendation()
	}

	if m.currentQuestion >= len(m.questions) {
		// Generate recommendation if we've answered all questions
		m.generateRecommendation()
		m.completed = true
		return m.renderRecommendation()
	}

	return m.renderQuestion()
}

// renderQuestion renders the current question
func (m WizardModel) renderQuestion() string {
	question := m.questions[m.currentQuestion]

	var b strings.Builder

	// Header
	b.WriteString(styles.TitleStyle.Render("🧙 AI Recommendation Wizard"))
	b.WriteString("\n\n")

	// Progress bar
	progress := styles.RenderProgressBar(m.currentQuestion+1, len(m.questions), 40)
	b.WriteString(fmt.Sprintf("Progress: %s (%d/%d)\n\n", progress, m.currentQuestion+1, len(m.questions)))

	// Question
	b.WriteString(styles.HeadingStyle.Render(question.Text))
	b.WriteString("\n")
	b.WriteString(styles.CaptionStyle.Render(question.Description))
	b.WriteString("\n\n")

	// Options
	for i, option := range question.Options {
		optionStyle := styles.BodyStyle
		cursor := "  "

		// TODO(human): Add selection highlighting
		// Implement selectedOption tracking and highlight the current selection
		// Use styles.SelectedStyle for selected option and cursor "> " for selected

		line := fmt.Sprintf("%s%s", cursor, option)
		b.WriteString(optionStyle.Render(line))
		b.WriteString("\n")
		_ = i // TODO(human): Use this for selection highlighting
	}

	// Help text
	b.WriteString("\n")
	helpText := "↑/↓ Navigate • Enter Select • B Back • Q Quit"
	b.WriteString(styles.CaptionStyle.Render(helpText))

	return lipgloss.NewStyle().
		Width(m.width).
		Align(lipgloss.Left).
		Render(b.String())
}

// renderRecommendation renders the final recommendation
func (m WizardModel) renderRecommendation() string {
	if m.recommendation == nil {
		return "Generating recommendation..."
	}

	var b strings.Builder

	// Header with celebration
	b.WriteString(styles.TitleStyle.Render("🎉 Perfect Match Found!"))
	b.WriteString("\n\n")

	// Main recommendation
	recBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1).
		Width(60)

	recContent := fmt.Sprintf("%s %s\n\n%s",
		m.getOptimizerIcon(m.recommendation.Optimizer),
		styles.HeadingStyle.Render(m.recommendation.Optimizer),
		m.recommendation.Reasoning)

	b.WriteString(recBox.Render(recContent))
	b.WriteString("\n\n")

	// Confidence
	confidenceBar := styles.RenderProgressBar(m.recommendation.Confidence, 100, 30)
	b.WriteString(fmt.Sprintf("Confidence: %s %d%%\n\n", confidenceBar, m.recommendation.Confidence))

	// Alternatives
	if len(m.recommendation.Alternatives) > 0 {
		b.WriteString(styles.HeadingStyle.Render("Alternative Options:"))
		b.WriteString("\n")
		for _, alt := range m.recommendation.Alternatives {
			b.WriteString(fmt.Sprintf("• %s\n", alt))
		}
		b.WriteString("\n")
	}

	// Next steps
	b.WriteString(styles.BodyStyle.Render("Ready to configure ") +
		styles.HighlightStyle.Render(m.recommendation.Optimizer) +
		styles.BodyStyle.Render(" with optimized settings?"))
	b.WriteString("\n\n")

	// Help text
	helpText := "Enter Continue • B Back to Questions • Q Quit"
	b.WriteString(styles.CaptionStyle.Render(helpText))

	return lipgloss.NewStyle().
		Width(m.width).
		Align(lipgloss.Left).
		Render(b.String())
}

// generateRecommendation creates a recommendation based on answers
func (m *WizardModel) generateRecommendation() {
	// TODO(human): Implement recommendation algorithm
	// This should analyze the answers map and determine the best optimizer
	//
	// Guidance: Create decision tree based on task_type, expertise, priority, etc.
	// Return Recommendation struct with optimizer, confidence, reasoning, and alternatives
	// Consider combinations like: math+speed=bootstrap, reasoning+accuracy=simba, etc.

	// Placeholder implementation
	m.recommendation = &Recommendation{
		Optimizer:     "MIPRO",
		Confidence:    85,
		Reasoning:     "Based on your answers, MIPRO offers the best balance of performance and ease of use for your needs.",
		Alternatives:  []string{"Bootstrap (faster setup)", "SIMBA (higher accuracy)"},
		Configuration: make(map[string]interface{}),
	}
}

// getOptimizerIcon returns an appropriate icon for the optimizer
func (m WizardModel) getOptimizerIcon(optimizer string) string {
	icons := map[string]string{
		"MIPRO":     "🎯",
		"SIMBA":     "🦁",
		"Bootstrap": "⚡",
		"COPRO":     "🤝",
		"GEPA":      "🧬",
	}

	if icon, exists := icons[optimizer]; exists {
		return icon
	}
	return "🔧"
}

// GetNextScreen returns the next screen to navigate to
func (m WizardModel) GetNextScreen() string {
	return m.nextScreen
}

// GetRecommendation returns the generated recommendation
func (m WizardModel) GetRecommendation() *Recommendation {
	return m.recommendation
}

// ResetNavigation clears the navigation state to prevent infinite loops
func (m *WizardModel) ResetNavigation() {
	m.nextScreen = ""
}
