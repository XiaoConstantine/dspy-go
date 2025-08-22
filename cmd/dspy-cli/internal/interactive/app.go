package interactive

import (
	"strings"

	"github.com/charmbracelet/bubbletea"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/models"
)

// Screen names for navigation
const (
	ScreenWelcome          = "welcome"
	ScreenOptimizerDetail  = "optimizer_detail"
	ScreenConfig           = "config"
	ScreenComparisonStudio = "comparison_studio"
	ScreenWizard          = "recommendation_wizard"
	ScreenLiveOptimization = "live_optimization"
	ScreenResults         = "results"
	ScreenHelp            = "help"
	ScreenPromptAnalyzer  = "prompt_analyzer"
)

// AppModel is the main application model that manages all screens
type AppModel struct {
	currentScreen    string
	welcome          models.WelcomeModel
	optimizerDetail  models.OptimizerDetailModel
	config           models.ConfigModel
	comparison       models.ComparisonModel
	wizard           models.WizardModel
	liveOptimization models.LiveOptimizationModel
	results          models.ResultsModel
	help             models.HelpModel
	promptAnalyzer   models.PromptAnalyzerModel
	width            int
	height           int
	quitting         bool
	previousScreen   string // For help navigation
}

// NewApp creates a new interactive application
func NewApp() AppModel {
	return AppModel{
		currentScreen: ScreenWelcome,
		welcome:       models.NewWelcomeModel(),
		quitting:      false,
	}
}

// Init initializes the application
func (m AppModel) Init() tea.Cmd {
	// Initialize with the welcome screen
	return m.welcome.Init()
}

// Update handles messages and updates the application state
func (m AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Handle global messages first
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Update all models with new size
		welcomeModel, _ := m.welcome.Update(msg)
		m.welcome = welcomeModel.(models.WelcomeModel)

		// Update optimizer detail model if it exists
		if m.currentScreen == ScreenOptimizerDetail {
			optimizerModel, _ := m.optimizerDetail.Update(msg)
			m.optimizerDetail = optimizerModel.(models.OptimizerDetailModel)
		}

		// Update config model if it exists
		if m.currentScreen == ScreenConfig {
			configModel, _ := m.config.Update(msg)
			m.config = configModel.(models.ConfigModel)
		}

		// Update comparison model if it exists
		if m.currentScreen == ScreenComparisonStudio {
			comparisonModel, _ := m.comparison.Update(msg)
			m.comparison = comparisonModel.(models.ComparisonModel)
		}

		return m, nil

	case tea.KeyMsg:
		// Global quit command
		if msg.String() == "ctrl+c" {
			m.quitting = true
			return m, tea.Quit
		}

		// Global help command
		if msg.String() == "?" || msg.String() == "F1" {
			m.previousScreen = m.currentScreen
			m.help = models.NewHelpModel()
			// Update help model with current window size
			if m.width > 0 && m.height > 0 {
				windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
				newModel, _ := m.help.Update(windowMsg)
				m.help = newModel.(models.HelpModel)
			}
			m.currentScreen = ScreenHelp
			return m, nil
		}
	}

	// Route update to current screen
	switch m.currentScreen {
	case ScreenWelcome:
		newModel, cmd := m.welcome.Update(msg)
		m.welcome = newModel.(models.WelcomeModel)

		// Check if we need to transition to a new screen
		if nextScreen := m.welcome.GetNextScreen(); nextScreen != "" {
			if nextScreen == "optimizer_detail" {
				// Extract optimizer name from welcome screen selection
				optimizer := m.extractOptimizerFromSelection()
				m.optimizerDetail = models.NewOptimizerDetailModel(optimizer)
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.optimizerDetail.Update(windowMsg)
					m.optimizerDetail = newModel.(models.OptimizerDetailModel)
				}
				m.currentScreen = ScreenOptimizerDetail
			} else if nextScreen == "comparison_studio" {
				m.comparison = models.NewComparisonModel()
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.comparison.Update(windowMsg)
					m.comparison = newModel.(models.ComparisonModel)
				}
				m.currentScreen = ScreenComparisonStudio
			} else if nextScreen == "recommendation_wizard" {
				m.wizard = models.NewWizardModel()
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.wizard.Update(windowMsg)
					m.wizard = newModel.(models.WizardModel)
				}
				m.currentScreen = ScreenWizard
			} else if nextScreen == "prompt_analyzer" {
				m.promptAnalyzer = models.NewPromptAnalyzerModel()
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.promptAnalyzer.Update(windowMsg)
					m.promptAnalyzer = newModel.(models.PromptAnalyzerModel)
				}
				m.currentScreen = ScreenPromptAnalyzer
			} else {
				m.currentScreen = nextScreen
			}
			// Reset navigation state to prevent infinite loops
			m.welcome.ResetNavigation()
		}

		return m, cmd

	case ScreenOptimizerDetail:
		newModel, cmd := m.optimizerDetail.Update(msg)
		m.optimizerDetail = newModel.(models.OptimizerDetailModel)

		// Check for navigation
		if nextScreen := m.optimizerDetail.GetNextScreen(); nextScreen != "" {
			if nextScreen == "back" {
				m.currentScreen = ScreenWelcome
				m.welcome = models.NewWelcomeModel()
				// Ensure the new welcome model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.welcome.Update(windowMsg)
					m.welcome = newModel.(models.WelcomeModel)
				}
			} else if nextScreen == "config" {
				m.config = models.NewConfigModel(m.optimizerDetail.GetOptimizer())
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.config.Update(windowMsg)
					m.config = newModel.(models.ConfigModel)
				}
				m.currentScreen = ScreenConfig
			} else if nextScreen == "comparison_studio" {
				m.comparison = models.NewComparisonModel()
				// Ensure the new model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.comparison.Update(windowMsg)
					m.comparison = newModel.(models.ComparisonModel)
				}
				m.currentScreen = ScreenComparisonStudio
			} else {
				m.currentScreen = nextScreen
			}
			// Reset navigation state to prevent infinite loops
			m.optimizerDetail.ResetNavigation()
		}

		return m, cmd

	case ScreenConfig:
		newModel, cmd := m.config.Update(msg)
		m.config = newModel.(models.ConfigModel)

		// Check for navigation
		if nextScreen := m.config.GetNextScreen(); nextScreen != "" {
			if nextScreen == "back" {
				m.currentScreen = ScreenOptimizerDetail
			} else if nextScreen == "run" {
				// TODO: Run with custom config
				m.currentScreen = ScreenOptimizerDetail
			}
			// Reset navigation state to prevent infinite loops
			m.config.ResetNavigation()
		}

		return m, cmd

	case ScreenComparisonStudio:
		newModel, cmd := m.comparison.Update(msg)
		m.comparison = newModel.(models.ComparisonModel)

		// Check for navigation
		if nextScreen := m.comparison.GetNextScreen(); nextScreen != "" {
			if nextScreen == "back" {
				m.currentScreen = ScreenWelcome
				m.welcome = models.NewWelcomeModel()
				// Ensure the new welcome model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.welcome.Update(windowMsg)
					m.welcome = newModel.(models.WelcomeModel)
				}
			}
			// Reset navigation state to prevent infinite loops
			m.comparison.ResetNavigation()
		}

		return m, cmd

	case ScreenWizard:
		newModel, cmd := m.wizard.Update(msg)
		m.wizard = newModel.(models.WizardModel)

		// Check for navigation
		if nextScreen := m.wizard.GetNextScreen(); nextScreen != "" {
			if nextScreen == "welcome" {
				m.currentScreen = ScreenWelcome
				m.welcome = models.NewWelcomeModel()
				// Ensure the new welcome model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.welcome.Update(windowMsg)
					m.welcome = newModel.(models.WelcomeModel)
				}
			} else if nextScreen == "config" {
				// Get recommended optimizer from wizard
				if rec := m.wizard.GetRecommendation(); rec != nil {
					m.config = models.NewConfigModel(rec.Optimizer)
					// Ensure the new model gets the current window size
					if m.width > 0 && m.height > 0 {
						windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
						newModel, _ := m.config.Update(windowMsg)
						m.config = newModel.(models.ConfigModel)
					}
					m.currentScreen = ScreenConfig
				}
			}
			// Reset navigation state to prevent infinite loops
			m.wizard.ResetNavigation()
		}

		return m, cmd

	case ScreenHelp:
		newModel, cmd := m.help.Update(msg)
		m.help = newModel.(models.HelpModel)

		// Check for navigation back
		if nextScreen := m.help.GetNextScreen(); nextScreen != "" {
			if nextScreen == "back" {
				m.currentScreen = m.previousScreen
				m.help.ResetNavigation()
			}
		}

		return m, cmd

	case ScreenPromptAnalyzer:
		newModel, cmd := m.promptAnalyzer.Update(msg)
		m.promptAnalyzer = newModel.(models.PromptAnalyzerModel)

		// Check for navigation back
		if nextScreen := m.promptAnalyzer.GetNextScreen(); nextScreen != "" {
			if nextScreen == "back" {
				m.currentScreen = ScreenWelcome
				m.welcome = models.NewWelcomeModel()
				// Ensure the new welcome model gets the current window size
				if m.width > 0 && m.height > 0 {
					windowMsg := tea.WindowSizeMsg{Width: m.width, Height: m.height}
					newModel, _ := m.welcome.Update(windowMsg)
					m.welcome = newModel.(models.WelcomeModel)
				}
			}
			// Reset navigation state to prevent infinite loops
			m.promptAnalyzer.ResetNavigation()
		}

		return m, cmd

	// Add more screens as we implement them
	default:
		return m, nil
	}
}

// View renders the current screen
func (m AppModel) View() string {
	if m.quitting {
		return "\nThanks for using DSPy-CLI! Happy optimizing! 🚀\n"
	}

	switch m.currentScreen {
	case ScreenWelcome:
		return m.welcome.View()

	// Actual screens
	case ScreenOptimizerDetail:
		return m.optimizerDetail.View()
	case ScreenConfig:
		return m.config.View()
	case ScreenComparisonStudio:
		return m.comparison.View()
	case ScreenWizard:
		return m.wizard.View()
	case ScreenHelp:
		return m.help.View()
	case ScreenPromptAnalyzer:
		return m.promptAnalyzer.View()
	default:
		return "Loading..."
	}
}

// extractOptimizerFromSelection extracts the optimizer name from the current welcome selection
func (m AppModel) extractOptimizerFromSelection() string {
	// Get the recommendation and extract optimizer name
	recommendation := m.welcome.GetRecommendation()
	// Parse the recommendation to get optimizer name
	if strings.Contains(recommendation, "MIPRO") {
		return "mipro"
	} else if strings.Contains(recommendation, "SIMBA") {
		return "simba"
	} else if strings.Contains(recommendation, "Bootstrap") {
		return "bootstrap"
	} else if strings.Contains(recommendation, "COPRO") {
		return "copro"
	} else if strings.Contains(recommendation, "GEPA") {
		return "gepa"
	}
	// Default fallback
	return "bootstrap"
}


func (m AppModel) renderWizard() string {
	return `
🧙 Recommendation Wizard

Let's find the perfect optimizer for you!

Coming soon...

[Press q to go back]
`
}
