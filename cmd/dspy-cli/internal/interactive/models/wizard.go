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
			// Store the selected answer
			question := m.questions[m.currentQuestion]
			selectedAnswer := question.Options[m.selectedOption]
			m.answers[question.ID] = selectedAnswer

			// Move to next question
			m.currentQuestion++
			m.selectedOption = 0 // Reset selection for next question

			// Check if we've completed all questions
			if m.currentQuestion >= len(m.questions) {
				m.generateRecommendation()
				m.completed = true
			}
			return m, nil
		case "up", "k":
			// Move selection up
			if m.selectedOption > 0 {
				m.selectedOption--
			} else {
				// Wrap to bottom
				m.selectedOption = len(m.questions[m.currentQuestion].Options) - 1
			}
		case "down", "j":
			// Move selection down
			if m.selectedOption < len(m.questions[m.currentQuestion].Options)-1 {
				m.selectedOption++
			} else {
				// Wrap to top
				m.selectedOption = 0
			}
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
		var optionStyle lipgloss.Style
		var cursor string

		// Highlight selected option
		if i == m.selectedOption {
			optionStyle = styles.SelectedStyle
			cursor = styles.IconSelected + " "
		} else {
			optionStyle = styles.UnselectedStyle
			cursor = styles.IconUnselected + " "
		}

		line := fmt.Sprintf("%s%s", cursor, option)
		b.WriteString(optionStyle.Render(line))
		b.WriteString("\n")
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
	// Extract answers for decision making
	taskType := m.answers["task_type"].(string)
	expertise := m.answers["expertise"].(string)
	priority := m.answers["priority"].(string)
	dataSize := m.answers["data_size"].(string)
	timeBudget := m.answers["time_budget"].(string)

	// Decision tree algorithm
	var optimizer string
	var confidence int
	var reasoning string
	var alternatives []string

	// Primary decision based on task type and priority
	switch {
	case strings.Contains(taskType, "Math") && strings.Contains(priority, "Speed"):
		optimizer = "Bootstrap"
		confidence = 90
		reasoning = "Bootstrap is perfect for mathematical tasks when speed is critical. Its fast convergence makes it ideal for quick optimization cycles."
		alternatives = []string{"MIPRO (more robust)", "COPRO (collaborative approach)"}

	case strings.Contains(taskType, "Complex reasoning") && strings.Contains(priority, "Accuracy"):
		optimizer = "SIMBA"
		confidence = 95
		reasoning = "SIMBA excels at complex reasoning tasks with its introspective approach, delivering highest accuracy for challenging problems."
		alternatives = []string{"MIPRO (balanced approach)", "GEPA (experimental features)"}

	case strings.Contains(taskType, "Question answering") && strings.Contains(dataSize, "Large"):
		optimizer = "MIPRO"
		confidence = 88
		reasoning = "MIPRO is optimized for question-answering with large datasets, providing systematic optimization with proven results."
		alternatives = []string{"COPRO (multi-module)", "Bootstrap (faster iteration)"}

	case strings.Contains(taskType, "Creative writing") || strings.Contains(taskType, "Code generation"):
		optimizer = "COPRO"
		confidence = 85
		reasoning = "COPRO's collaborative multi-module approach excels at creative and generative tasks requiring diverse optimization strategies."
		alternatives = []string{"SIMBA (deeper reasoning)", "GEPA (cutting-edge methods)"}

	case strings.Contains(priority, "Research") || strings.Contains(expertise, "Expert"):
		optimizer = "GEPA"
		confidence = 80
		reasoning = "GEPA offers cutting-edge evolutionary optimization perfect for research applications and expert users seeking latest methodologies."
		alternatives = []string{"SIMBA (proven accuracy)", "MIPRO (production-ready)"}

	case strings.Contains(timeBudget, "Quick") || strings.Contains(expertise, "Beginner"):
		optimizer = "Bootstrap"
		confidence = 92
		reasoning = "Bootstrap provides the fastest path to results with minimal complexity, perfect for quick experiments and learning."
		alternatives = []string{"MIPRO (more features)", "COPRO (scalable approach)"}

	default:
		// Balanced recommendation for mixed requirements
		optimizer = "MIPRO"
		confidence = 85
		reasoning = "MIPRO provides the best balance of performance, reliability, and ease of use for your varied requirements."
		alternatives = []string{"Bootstrap (faster)", "SIMBA (more accurate)", "COPRO (collaborative)"}
	}

	// Adjust confidence based on additional factors
	if strings.Contains(expertise, "Expert") {
		confidence += 5 // Expert users can maximize any optimizer
	}
	if strings.Contains(dataSize, "Very large") && optimizer != "MIPRO" {
		confidence -= 10 // Large datasets favor MIPRO
	}
	if strings.Contains(timeBudget, "Research") && optimizer == "Bootstrap" {
		confidence -= 15 // Research time allows for more sophisticated methods
	}

	// Ensure confidence stays in reasonable range
	if confidence > 95 {
		confidence = 95
	}
	if confidence < 70 {
		confidence = 70
	}

	m.recommendation = &Recommendation{
		Optimizer:     optimizer,
		Confidence:    confidence,
		Reasoning:     reasoning,
		Alternatives:  alternatives,
		Configuration: m.generateOptimalConfig(optimizer, expertise, priority),
	}
}

// generateOptimalConfig creates optimized configuration based on user profile
func (m *WizardModel) generateOptimalConfig(optimizer, expertise, priority string) map[string]interface{} {
	config := make(map[string]interface{})

	// Base configuration based on optimizer
	switch optimizer {
	case "Bootstrap":
		config["max_bootstrapped_demos"] = 4
		config["max_labeled_demos"] = 16
	case "MIPRO":
		config["num_candidates"] = 50
		config["init_temperature"] = 1.0
	case "SIMBA":
		config["num_threads"] = 8
		config["max_examples"] = 20
	case "COPRO":
		config["breadth"] = 10
		config["depth"] = 3
	case "GEPA":
		config["population_size"] = 50
		config["generations"] = 20
	}

	// Adjust based on expertise level
	if strings.Contains(expertise, "Beginner") {
		// Simpler, more conservative settings
		if optimizer == "MIPRO" {
			config["num_candidates"] = 20
		}
		if optimizer == "GEPA" {
			config["population_size"] = 25
			config["generations"] = 10
		}
	} else if strings.Contains(expertise, "Expert") {
		// More aggressive settings for experts
		if optimizer == "MIPRO" {
			config["num_candidates"] = 100
		}
		if optimizer == "GEPA" {
			config["population_size"] = 100
			config["generations"] = 50
		}
	}

	// Adjust based on priority
	if strings.Contains(priority, "Speed") {
		// Reduce iteration counts for faster results
		if optimizer == "MIPRO" {
			config["num_candidates"] = int(config["num_candidates"].(int) / 2)
		}
		if optimizer == "GEPA" {
			config["generations"] = int(config["generations"].(int) / 2)
		}
	} else if strings.Contains(priority, "Accuracy") {
		// Increase iterations for better results
		if optimizer == "MIPRO" {
			config["num_candidates"] = int(config["num_candidates"].(int) * 2)
		}
		if optimizer == "GEPA" {
			config["generations"] = int(config["generations"].(int) * 2)
		}
	}

	return config
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
