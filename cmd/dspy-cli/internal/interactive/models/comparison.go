package models

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/samples"
)

// ComparisonModel represents the comparison studio state
type ComparisonModel struct {
	selectedDataset int
	datasets        []string
	width          int
	height         int
	isRunning      bool
	results        map[string]*runner.RunResult
	runningStatus  map[string]bool
	startTime      time.Time
	completed      int
	total          int
	nextScreen     string
	spinnerFrame   int
	sortBy         string // "name", "accuracy", "improvement", "duration"
	mu             sync.RWMutex
}

// ComparisonResult holds the result with optimizer info for display
type ComparisonResult struct {
	OptimizerName string
	DisplayName   string
	Result        *runner.RunResult
	Status        string // "pending", "running", "completed", "failed"
}

// NewComparisonModel creates a new comparison studio model
func NewComparisonModel() ComparisonModel {
	return ComparisonModel{
		selectedDataset: 0,
		datasets:        samples.ListAvailableDatasets(),
		results:         make(map[string]*runner.RunResult),
		runningStatus:   make(map[string]bool),
		width:          80,  // Default width
		height:         24,  // Default height
		sortBy:         "accuracy", // Default sort by accuracy
	}
}

// Init initializes the model
func (m ComparisonModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m ComparisonModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
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
				return m, m.startComparison()
			}
		case "s":
			if !m.isRunning {
				// Cycle through sort options
				switch m.sortBy {
				case "accuracy":
					m.sortBy = "improvement"
				case "improvement":
					m.sortBy = "duration"
				case "duration":
					m.sortBy = "name"
				default:
					m.sortBy = "accuracy"
				}
			}
		case "b", "esc":
			if !m.isRunning {
				m.nextScreen = "back"
			}
		case "q":
			return m, tea.Quit
		}

	case comparisonCompleteMsg:
		m.mu.Lock()
		m.results[msg.optimizer] = msg.result
		m.runningStatus[msg.optimizer] = false
		m.completed++

		// Check if all optimizers are done
		if m.completed >= m.total {
			m.isRunning = false
		}
		m.mu.Unlock()

		return m, nil

	case comparisonStartMsg:
		m.mu.Lock()
		m.runningStatus[msg.optimizer] = true
		m.mu.Unlock()
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

// startComparison begins running all optimizers in parallel
func (m *ComparisonModel) startComparison() tea.Cmd {
	m.isRunning = true
	m.startTime = time.Now()
	m.results = make(map[string]*runner.RunResult)
	m.runningStatus = make(map[string]bool)
	m.completed = 0
	m.spinnerFrame = 0

	// Get all available optimizers
	allOptimizers := []string{"bootstrap", "mipro", "simba", "gepa", "copro"}
	m.total = len(allOptimizers)

	selectedDatasetDisplay := m.datasets[m.selectedDataset]
	selectedDataset := extractDatasetKey(selectedDatasetDisplay)

	var commands []tea.Cmd

	// Add spinner
	commands = append(commands, m.tickSpinner())

	// Start all optimizers in parallel
	for _, optimizer := range allOptimizers {
		commands = append(commands, m.runSingleOptimizer(optimizer, selectedDataset))
	}

	return tea.Batch(commands...)
}

// runSingleOptimizer runs a single optimizer and returns its result
func (m ComparisonModel) runSingleOptimizer(optimizer, dataset string) tea.Cmd {
	return tea.Cmd(func() tea.Msg {
		// Run the optimizer
		config := runner.OptimizerConfig{
			OptimizerName: optimizer,
			DatasetName:   dataset,
			APIKey:        "", // Auto-detect from environment
			MaxExamples:   5,  // Keep it small for quick comparison
			Verbose:       false,
			SuppressLogs:  true, // Suppress console output for clean TUI
		}

		result, err := runner.RunOptimizer(config)

		return comparisonCompleteMsg{
			optimizer: optimizer,
			result:    result,
			err:       err,
		}
	})
}

// View renders the comparison studio screen
func (m ComparisonModel) View() string {
	if m.width == 0 {
		return "Loading..."
	}

	var content []string

	// Header
	header := styles.TitleStyle.Render("⚔️ Optimizer Comparison Studio")
	content = append(content, header)
	content = append(content, "")

	// Description
	description := styles.BodyStyle.Render("Compare all DSPy optimizers side-by-side to find the best performer for your task")
	content = append(content, description)
	content = append(content, "")

	// Dataset selection (if not running)
	if !m.isRunning && m.completed == 0 {
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

	// Progress section (if running or completed)
	if m.isRunning || m.completed > 0 {
		progressTitle := styles.HeadingStyle.Render("📊 Comparison Progress")
		content = append(content, progressTitle)
		content = append(content, "")

		if m.isRunning {
			elapsed := time.Since(m.startTime)
			progress := styles.InfoStyle.Render(fmt.Sprintf("Elapsed: %v • Completed: %d/%d",
				elapsed.Round(time.Second), m.completed, m.total))
			content = append(content, progress)

			spinner := styles.HighlightStyle.Render(fmt.Sprintf("%s Running comparisons...", getSpinnerFrame(m.spinnerFrame)))
			content = append(content, spinner)
		} else {
			completedMsg := styles.SuccessStyle.Render(fmt.Sprintf("✅ Comparison completed! (%d optimizers tested)", m.total))
			content = append(content, completedMsg)
		}
		content = append(content, "")

		// Results leaderboard
		leaderboardTitle := styles.HeadingStyle.Render(fmt.Sprintf("🏆 Leaderboard (sorted by %s)", m.sortBy))
		content = append(content, leaderboardTitle)
		content = append(content, "")

		results := m.getSortedResults()

		// Table header
		headerRow := fmt.Sprintf("%-12s %-12s %-12s %-12s %-10s",
			"Optimizer", "Status", "Accuracy", "Improvement", "Duration")
		content = append(content, styles.HeadingStyle.Render(headerRow))
		content = append(content, strings.Repeat("─", 70))

		// Table rows
		for i, result := range results {
			var row string

			// Rank styling
			rankStyle := styles.BodyStyle
			if i == 0 && result.Status == "completed" {
				rankStyle = styles.SuccessStyle // Winner
			} else if i == 1 && result.Status == "completed" {
				rankStyle = styles.InfoStyle // Second place
			}

			row = rankStyle.Render(fmt.Sprintf("%-12s %-12s %-12s %-12s %-10s",
				result.DisplayName,
				result.Status,
				formatAccuracy(result.Result),
				formatImprovement(result.Result),
				formatDuration(result.Result),
			))

			content = append(content, row)
		}
		content = append(content, "")
	}

	// Footer with controls
	var controls []string
	if !m.isRunning {
		if m.completed == 0 {
			controls = append(controls, "[↑↓/jk] Select dataset", "[Enter] Start comparison")
		} else {
			controls = append(controls, "[s] Sort by", "[Enter] Run again")
		}
		controls = append(controls, "[b] Back", "[q] Quit")
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

// getSortedResults returns results sorted by the current sort criteria
func (m ComparisonModel) getSortedResults() []ComparisonResult {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []ComparisonResult

	// Get all optimizers (including those not yet started)
	allOptimizers := []string{"bootstrap", "mipro", "simba", "gepa", "copro"}

	for _, optimizer := range allOptimizers {
		info, _ := optimizers.GetOptimizer(optimizer)

		var status string
		var result *runner.RunResult

		if r, exists := m.results[optimizer]; exists {
			result = r
			if r.Success {
				status = "completed"
			} else {
				status = "failed"
			}
		} else if running, exists := m.runningStatus[optimizer]; exists && running {
			status = "running"
		} else {
			status = "pending"
		}

		results = append(results, ComparisonResult{
			OptimizerName: optimizer,
			DisplayName:   strings.TrimSuffix(info.Name, " ✅"),
			Result:        result,
			Status:        status,
		})
	}

	// Sort results
	sort.Slice(results, func(i, j int) bool {
		switch m.sortBy {
		case "accuracy":
			// Completed results first, then by accuracy descending
			if results[i].Status != "completed" && results[j].Status == "completed" {
				return false
			}
			if results[i].Status == "completed" && results[j].Status != "completed" {
				return true
			}
			if results[i].Result != nil && results[j].Result != nil {
				return results[i].Result.FinalAccuracy > results[j].Result.FinalAccuracy
			}
			return results[i].OptimizerName < results[j].OptimizerName

		case "improvement":
			if results[i].Status != "completed" && results[j].Status == "completed" {
				return false
			}
			if results[i].Status == "completed" && results[j].Status != "completed" {
				return true
			}
			if results[i].Result != nil && results[j].Result != nil {
				return results[i].Result.ImprovementPct > results[j].Result.ImprovementPct
			}
			return results[i].OptimizerName < results[j].OptimizerName

		case "duration":
			if results[i].Status != "completed" && results[j].Status == "completed" {
				return false
			}
			if results[i].Status == "completed" && results[j].Status != "completed" {
				return true
			}
			if results[i].Result != nil && results[j].Result != nil {
				return results[i].Result.Duration < results[j].Result.Duration
			}
			return results[i].OptimizerName < results[j].OptimizerName

		default: // "name"
			return results[i].OptimizerName < results[j].OptimizerName
		}
	})

	return results
}

// Helper functions for formatting
func formatAccuracy(result *runner.RunResult) string {
	if result == nil || !result.Success {
		return "—"
	}
	return fmt.Sprintf("%.1f%%", result.FinalAccuracy*100)
}

func formatImprovement(result *runner.RunResult) string {
	if result == nil || !result.Success {
		return "—"
	}
	return fmt.Sprintf("+%.1f%%", result.ImprovementPct)
}

func formatDuration(result *runner.RunResult) string {
	if result == nil || !result.Success {
		return "—"
	}
	return result.Duration.Round(time.Second).String()
}

// GetNextScreen returns the next screen to navigate to
func (m ComparisonModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation clears the next screen state
func (m *ComparisonModel) ResetNavigation() {
	m.nextScreen = ""
}

// tickSpinner returns a command that sends a spinner tick after a delay
func (m ComparisonModel) tickSpinner() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(time.Time) tea.Msg {
		return spinnerTickMsg{}
	})
}

// Message types for comparison operations
type comparisonCompleteMsg struct {
	optimizer string
	result    *runner.RunResult
	err       error
}

type comparisonStartMsg struct {
	optimizer string
}
