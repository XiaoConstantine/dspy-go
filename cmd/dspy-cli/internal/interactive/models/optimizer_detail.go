package models

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/samples"
)

// OptimizerDetailModel represents the optimizer detail screen state
type OptimizerDetailModel struct {
	optimizer        string
	optimizerInfo    optimizers.OptimizerInfo
	selectedDataset  int
	datasets         []string
	width           int
	height          int
	isRunning       bool
	runResult       *runner.RunResult
	errorMessage    string
	runStartTime    time.Time
	nextScreen      string
	spinnerFrame    int
}

// NewOptimizerDetailModel creates a new optimizer detail model
func NewOptimizerDetailModel(optimizerName string) OptimizerDetailModel {
	info, err := optimizers.GetOptimizer(optimizerName)
	if err != nil {
		// Fallback if optimizer not found
		info = optimizers.OptimizerInfo{
			Name:        optimizerName,
			Description: "Optimizer information not available",
		}
	}

	return OptimizerDetailModel{
		optimizer:       optimizerName,
		optimizerInfo:   info,
		selectedDataset: 0,
		datasets:        samples.ListAvailableDatasets(),
		isRunning:       false,
		width:          80,  // Default width to prevent loading screen
		height:         24,  // Default height to prevent loading screen
	}
}

// Init initializes the model
func (m OptimizerDetailModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m OptimizerDetailModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if !m.isRunning && m.selectedDataset > 0 {
				m.selectedDataset--
			}
		case "down", "j":
			if !m.isRunning && m.selectedDataset < len(m.datasets)-1 {
				m.selectedDataset++
			}
		case "enter", " ":
			if !m.isRunning {
				return m, m.runOptimizer()
			}
		case "b", "esc":
			if !m.isRunning {
				m.nextScreen = "back"
			}
		case "q":
			return m, tea.Quit
		case "c":
			if !m.isRunning {
				m.nextScreen = "comparison_studio"
			}
		case "s":
			if !m.isRunning {
				m.nextScreen = "config"
			}
		}

	case runCompleteMsg:
		m.isRunning = false
		if msg.err != nil {
			m.errorMessage = msg.err.Error()
			m.runResult = nil
		} else {
			m.runResult = msg.result
			m.errorMessage = ""
		}
		return m, nil

	case runProgressMsg:
		// Handle progress updates if needed
		return m, nil

	case spinnerTickMsg:
		if m.isRunning {
			m.spinnerFrame++
			return m, m.tickSpinner()
		}
		return m, nil
	}

	return m, nil
}

// runOptimizer starts the optimization process asynchronously
func (m *OptimizerDetailModel) runOptimizer() tea.Cmd {
	m.isRunning = true
	m.runStartTime = time.Now()
	m.runResult = nil
	m.errorMessage = ""
	m.spinnerFrame = 0

	selectedDatasetDisplay := m.datasets[m.selectedDataset]
	selectedDataset := extractDatasetKey(selectedDatasetDisplay)

	return tea.Batch(
		// Start the optimization
		tea.Cmd(func() tea.Msg {
			config := runner.OptimizerConfig{
				OptimizerName: m.optimizer,
				DatasetName:   selectedDataset,
				APIKey:        "", // Auto-detect from environment
				MaxExamples:   5,  // Keep it small for quick testing
				Verbose:       false,
				SuppressLogs:  true, // Suppress console output for clean TUI
			}

			result, err := runner.RunOptimizer(config)
			return runCompleteMsg{result: result, err: err}
		}),
		// Start the spinner animation
		m.tickSpinner(),
	)
}

// View renders the optimizer detail screen
func (m OptimizerDetailModel) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	var content []string

	// Header
	header := styles.TitleStyle.Render(fmt.Sprintf("🎯 %s", m.optimizerInfo.Name))
	content = append(content, header)
	content = append(content, "")

	// Description
	description := styles.BodyStyle.Render(m.optimizerInfo.Description)
	content = append(content, description)
	content = append(content, "")

	// Optimizer details section
	detailsTitle := styles.HeadingStyle.Render("📊 Optimizer Details")
	content = append(content, detailsTitle)
	content = append(content, "")

	// Details grid
	details := []string{
		fmt.Sprintf("Complexity: %s", styles.InfoStyle.Render(m.optimizerInfo.Complexity)),
		fmt.Sprintf("Compute Cost: %s", styles.InfoStyle.Render(m.optimizerInfo.ComputeCost)),
		fmt.Sprintf("Convergence: %s", styles.InfoStyle.Render(m.optimizerInfo.Convergence)),
	}
	content = append(content, strings.Join(details, "  "))
	content = append(content, "")

	// Best for section
	if len(m.optimizerInfo.BestFor) > 0 {
		bestForTitle := styles.HeadingStyle.Render("✨ Best For")
		content = append(content, bestForTitle)
		for _, item := range m.optimizerInfo.BestFor {
			content = append(content, fmt.Sprintf("  %s %s", styles.IconDot, styles.BodyStyle.Render(item)))
		}
		content = append(content, "")
	}

	// Dataset selection section
	if !m.isRunning && m.runResult == nil {
		datasetTitle := styles.HeadingStyle.Render("🗂️ Select Test Dataset")
		content = append(content, datasetTitle)
		content = append(content, "")

		for i, dataset := range m.datasets {
			var datasetStr string
			if i == m.selectedDataset {
				datasetStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s", styles.IconSelected, dataset))
			} else {
				datasetStr = styles.UnselectedStyle.Render(fmt.Sprintf("  %s", dataset))
			}
			content = append(content, datasetStr)
		}
		content = append(content, "")
	}

	// Running state
	if m.isRunning {
		runningTitle := styles.HeadingStyle.Render("🚀 Running Optimization...")
		content = append(content, runningTitle)
		content = append(content, "")

		elapsed := time.Since(m.runStartTime)
		progress := styles.InfoStyle.Render(fmt.Sprintf("Elapsed time: %v", elapsed.Round(time.Second)))
		content = append(content, progress)
		content = append(content, "")

		spinner := styles.HighlightStyle.Render(fmt.Sprintf("%s Optimizing...", getSpinnerFrame(m.spinnerFrame)))
		content = append(content, spinner)
		content = append(content, "")
	}

	// Results section
	if m.runResult != nil {
		resultsTitle := styles.HeadingStyle.Render("📈 Optimization Results")
		content = append(content, resultsTitle)
		content = append(content, "")

		if m.runResult.Success {
			// Success metrics
			metrics := []string{
				fmt.Sprintf("Initial Accuracy: %s", styles.MutedStyle.Render(fmt.Sprintf("%.1f%%", m.runResult.InitialAccuracy*100))),
				fmt.Sprintf("Final Accuracy: %s", styles.SuccessStyle.Render(fmt.Sprintf("%.1f%%", m.runResult.FinalAccuracy*100))),
				fmt.Sprintf("Improvement: %s", styles.HighlightStyle.Render(fmt.Sprintf("+%.1f%%", m.runResult.ImprovementPct))),
				fmt.Sprintf("Duration: %s", styles.InfoStyle.Render(m.runResult.Duration.Round(time.Second).String())),
			}
			content = append(content, strings.Join(metrics, "  "))
		} else {
			// Error display
			errorMsg := styles.ErrorStyle.Render(fmt.Sprintf("❌ Optimization failed: %s", m.runResult.ErrorMessage))
			content = append(content, errorMsg)
		}
		content = append(content, "")
	}

	// Error message
	if m.errorMessage != "" {
		errorMsg := styles.ErrorStyle.Render(fmt.Sprintf("❌ Error: %s", m.errorMessage))
		content = append(content, errorMsg)
		content = append(content, "")
	}

	// Footer with controls
	var controls []string
	if !m.isRunning {
		if m.runResult == nil {
			controls = append(controls, "[↑↓/jk] Select dataset", "[Enter] Run optimizer")
		}
		controls = append(controls, "[s] Settings", "[c] Compare optimizers", "[b] Back", "[q] Quit")
	} else {
		controls = append(controls, "Please wait...", "[q] Quit")
	}

	footer := styles.FooterStyle.Render(strings.Join(controls, "  "))
	content = append(content, footer)

	// Join all content
	fullContent := lipgloss.JoinVertical(lipgloss.Left, content...)

	// Apply box styling and center
	boxed := styles.BoxStyle.Copy().
		Width(min(m.width-4, 100)).
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

// GetNextScreen returns the next screen to navigate to
func (m OptimizerDetailModel) GetNextScreen() string {
	return m.nextScreen
}

// GetOptimizer returns the optimizer name
func (m OptimizerDetailModel) GetOptimizer() string {
	return m.optimizer
}

// ResetNavigation clears the next screen state
func (m *OptimizerDetailModel) ResetNavigation() {
	m.nextScreen = ""
}

// tickSpinner returns a command that sends a spinner tick after a delay
func (m OptimizerDetailModel) tickSpinner() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(time.Time) tea.Msg {
		return spinnerTickMsg{}
	})
}

// Message types for async operations
type runCompleteMsg struct {
	result *runner.RunResult
	err    error
}

type runProgressMsg struct {
	message string
}
