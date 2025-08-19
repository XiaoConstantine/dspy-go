package interactive

import (
	"fmt"

	"github.com/charmbracelet/bubbletea"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/models"
)

// Screen names for navigation
const (
	ScreenWelcome          = "welcome"
	ScreenOptimizerDetail  = "optimizer_detail"
	ScreenComparisonStudio = "comparison_studio"
	ScreenWizard          = "recommendation_wizard"
	ScreenLiveOptimization = "live_optimization"
	ScreenResults         = "results"
)

// AppModel is the main application model that manages all screens
type AppModel struct {
	currentScreen string
	welcome       models.WelcomeModel
	width         int
	height        int
	quitting      bool
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
		return m, nil

	case tea.KeyMsg:
		// Global quit command
		if msg.String() == "ctrl+c" {
			m.quitting = true
			return m, tea.Quit
		}
	}

	// Route update to current screen
	switch m.currentScreen {
	case ScreenWelcome:
		newModel, cmd := m.welcome.Update(msg)
		m.welcome = newModel.(models.WelcomeModel)

		// Check if we need to transition to a new screen
		if nextScreen := m.welcome.GetNextScreen(); nextScreen != "" {
			m.currentScreen = nextScreen
		}

		return m, cmd

	case ScreenOptimizerDetail:
		// Handle navigation in optimizer detail screen
		if keyMsg, ok := msg.(tea.KeyMsg); ok {
			switch keyMsg.String() {
			case "b", "esc":
				// Go back to welcome
				m.currentScreen = ScreenWelcome
				m.welcome = models.NewWelcomeModel()
			case "q":
				return m, tea.Quit
			case "enter":
				// TODO: Launch optimizer
				return m, nil
			case "c":
				m.currentScreen = ScreenComparisonStudio
			}
		}
		return m, nil

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

	// Placeholder for other screens
	case ScreenOptimizerDetail:
		return m.renderOptimizerDetail()
	case ScreenComparisonStudio:
		return m.renderComparisonStudio()
	case ScreenWizard:
		return m.renderWizard()
	default:
		return "Loading..."
	}
}

// Placeholder methods for other screens (to be implemented)
func (m AppModel) renderOptimizerDetail() string {
	recommendation := m.welcome.GetRecommendation()

	// For now, return a better formatted detail view
	return fmt.Sprintf(`
┌─ DSPy-CLI Optimizer Detail ─────────────────────────────────────────┐
│                                                                     │
│ 🎯 Recommended Optimizer                                            │
│                                                                     │
│ %s                                                                  │
│                                                                     │
│ 📊 Quick Actions:                                                  │
│ • Press [Enter] to try this optimizer with sample data             │
│ • Press [c] to compare with other optimizers                       │
│ • Press [d] for detailed documentation                             │
│ • Press [b] to go back                                             │
│ • Press [q] to quit                                                │
│                                                                     │
│ Coming soon: Live optimization visualization!                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

`, recommendation)
}

func (m AppModel) renderComparisonStudio() string {
	return `
⚔️ Comparison Studio

Compare all optimizers side-by-side!

Coming soon...

[Press q to go back]
`
}

func (m AppModel) renderWizard() string {
	return `
🧙 Recommendation Wizard

Let's find the perfect optimizer for you!

Coming soon...

[Press q to go back]
`
}
