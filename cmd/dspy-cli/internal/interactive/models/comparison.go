package models

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
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
	runningStatus  map[string]OptimizerStatus
	startTime      time.Time
	completed      int
	total          int
	nextScreen     string
	spinnerFrame   int
	sortBy         string // "name", "accuracy", "improvement", "duration"
	mu             sync.RWMutex

	// Enhanced features
	selectedOptimizers map[string]bool // Which optimizers to compare
	liveProgress      map[string]float64 // Real-time progress per optimizer
	errorMessages     map[string]string  // Error messages per optimizer
	leaderboard       []LeaderboardEntry // Sorted results
	showDetails       bool              // Show detailed view
	animationFrame    int               // For visual effects
	maxExamples       int               // Configurable examples count
}

// OptimizerStatus represents the current status of an optimizer
type OptimizerStatus struct {
	State    string    // "pending", "initializing", "running", "completed", "failed"
	Progress float64   // 0.0 to 1.0
	Message  string    // Current status message
	StartTime time.Time
	EndTime   time.Time
}

// LeaderboardEntry represents a single entry in the results leaderboard
type LeaderboardEntry struct {
	Rank         int
	OptimizerName string
	Icon         string
	DisplayName  string
	Accuracy     float64
	Improvement  float64
	Duration     time.Duration
	Status       string
	TrendIcon    string // ↗️ ↘️ ➡️
}

// ComparisonResult holds the result with optimizer info for display
type ComparisonResult struct {
	OptimizerName string
	DisplayName   string
	Result        *runner.RunResult
	Status        string // "pending", "running", "completed", "failed"
}

// Message types for comparison events
type comparisonCompleteMsg struct {
	optimizer string
	result    *runner.RunResult
	err       error
}

type comparisonStartMsg struct {
	optimizer string
}

type comparisonProgressMsg struct {
	optimizer string
	progress  float64
	message   string
}

type spinnerTickMsg struct{}

type animationTickMsg struct{}

// NewComparisonModel creates a new comparison studio model
func NewComparisonModel() ComparisonModel {
	// Default to all optimizers selected
	selectedOptimizers := map[string]bool{
		"bootstrap": true,
		"mipro":     true,
		"simba":     true,
		"gepa":      true,
		"copro":     true,
	}

	return ComparisonModel{
		selectedDataset:    0,
		datasets:          samples.ListAvailableDatasets(),
		results:           make(map[string]*runner.RunResult),
		runningStatus:     make(map[string]OptimizerStatus),
		selectedOptimizers: selectedOptimizers,
		liveProgress:      make(map[string]float64),
		errorMessages:     make(map[string]string),
		leaderboard:       []LeaderboardEntry{},
		width:            80,  // Default width
		height:           24,  // Default height
		sortBy:           "accuracy", // Default sort by accuracy
		maxExamples:      5,   // Default examples for quick comparison
		showDetails:      false,
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

		// Update status
		status := m.runningStatus[msg.optimizer]
		if msg.err != nil {
			status.State = "failed"
			status.Message = msg.err.Error()
			m.errorMessages[msg.optimizer] = msg.err.Error()
		} else {
			status.State = "completed"
			status.Progress = 1.0
			status.Message = "Battle completed successfully"
		}
		status.EndTime = time.Now()
		m.runningStatus[msg.optimizer] = status

		m.completed++

		// Update leaderboard
		m.updateLeaderboard()

		// Check if all optimizers are done
		if m.completed >= m.total {
			m.isRunning = false
		}
		m.mu.Unlock()

		return m, nil

	case comparisonStartMsg:
		m.mu.Lock()
		status := OptimizerStatus{
			State:     "initializing",
			Progress:  0.0,
			Message:   "Preparing for battle...",
			StartTime: time.Now(),
		}
		m.runningStatus[msg.optimizer] = status
		m.mu.Unlock()
		return m, nil

	case comparisonProgressMsg:
		m.mu.Lock()
		status := m.runningStatus[msg.optimizer]
		status.State = "running"
		status.Progress = msg.progress
		status.Message = msg.message
		m.runningStatus[msg.optimizer] = status
		m.liveProgress[msg.optimizer] = msg.progress
		m.mu.Unlock()
		return m, nil

	case spinnerTickMsg:
		if m.isRunning {
			m.spinnerFrame++
			return m, m.tickSpinner()
		}
		return m, nil

	case animationTickMsg:
		m.animationFrame++
		if m.isRunning || m.completed > 0 {
			return m, m.tickAnimation()
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
	m.runningStatus = make(map[string]OptimizerStatus)
	m.liveProgress = make(map[string]float64)
	m.errorMessages = make(map[string]string)
	m.completed = 0
	m.spinnerFrame = 0
	m.animationFrame = 0
	m.leaderboard = []LeaderboardEntry{}

	// Get selected optimizers
	var selectedOptimizers []string
	for optimizer, selected := range m.selectedOptimizers {
		if selected {
			selectedOptimizers = append(selectedOptimizers, optimizer)
		}
	}
	m.total = len(selectedOptimizers)

	selectedDatasetDisplay := m.datasets[m.selectedDataset]
	selectedDataset := extractDatasetKey(selectedDatasetDisplay)

	var commands []tea.Cmd

	// Add animation tickers
	commands = append(commands, m.tickSpinner())
	commands = append(commands, m.tickAnimation())

	// Start all optimizers in parallel
	for _, optimizer := range selectedOptimizers {
		// Initialize status
		m.runningStatus[optimizer] = OptimizerStatus{
			State:     "pending",
			Progress:  0.0,
			Message:   "Waiting to enter arena...",
			StartTime: time.Now(),
		}

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

	var sections []string

	// Header with battle theme
	header := m.renderHeader()
	sections = append(sections, header)

	if !m.isRunning && m.completed == 0 {
		// Pre-battle setup screen
		setup := m.renderSetupScreen()
		sections = append(sections, setup)
	} else {
		// Live battle arena
		arena := m.renderBattleArena()
		sections = append(sections, arena)
	}

	// Footer with controls
	footer := m.renderFooter()
	sections = append(sections, footer)

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// renderHeader renders the comparison header with battle theme
func (m ComparisonModel) renderHeader() string {
	var headerText string
	if m.isRunning {
		headerText = "⚔️ OPTIMIZER BATTLE IN PROGRESS"
	} else if m.completed > 0 {
		headerText = "🏆 BATTLE RESULTS - CHAMPIONS DECIDED"
	} else {
		headerText = "⚔️ OPTIMIZER BATTLE ARENA"
	}

	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Background(lipgloss.Color(styles.DSPyBlue)).
		Foreground(lipgloss.Color(styles.White)).
		Width(m.width).
		Padding(0, 2)

	subtitle := "Compare all DSPy optimizers in head-to-head competition"
	subtitleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(styles.MediumGray)).
		Italic(true).
		Align(lipgloss.Center).
		Width(m.width)

	return lipgloss.JoinVertical(lipgloss.Left,
		headerStyle.Render(headerText),
		subtitleStyle.Render(subtitle))
}

// renderSetupScreen renders the pre-battle configuration
func (m ComparisonModel) renderSetupScreen() string {
	var content []string

	// Dataset selection
	content = append(content, styles.HeadingStyle.Render("🗂️ Choose Your Battleground (Dataset)"))
	content = append(content, "")

	for i, dataset := range m.datasets {
		var datasetStr string
		if i == m.selectedDataset {
			datasetStr = styles.SelectedStyle.Render(fmt.Sprintf("%s %s",
				styles.IconSelected, dataset))
		} else {
			datasetStr = styles.UnselectedStyle.Render(fmt.Sprintf("  %s", dataset))
		}
		content = append(content, datasetStr)
	}
	content = append(content, "")

	// Configuration summary
	content = append(content, styles.HeadingStyle.Render("⚙️ Battle Configuration"))
	content = append(content, "")

	configBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1, 2).
		Width(m.width - 4)

	var configContent []string
	configContent = append(configContent, fmt.Sprintf("🎯 Competitors: %d optimizers", m.countSelectedOptimizers()))
	configContent = append(configContent, fmt.Sprintf("📊 Examples per battle: %d", m.maxExamples))
	configContent = append(configContent, fmt.Sprintf("⚡ Execution: Parallel (all at once)"))
	configContent = append(configContent, fmt.Sprintf("🏆 Victory condition: Highest accuracy"))

	content = append(content, configBox.Render(strings.Join(configContent, "\n")))
	content = append(content, "")

	// Ready to battle prompt
	readyPrompt := styles.HighlightStyle.Render("🚀 Press Enter to begin the ultimate optimizer showdown!")
	content = append(content, readyPrompt)

	return strings.Join(content, "\n")
}

// renderBattleArena renders the live battle progress and leaderboard
func (m ComparisonModel) renderBattleArena() string {
	var content []string

	if m.isRunning {
		// Live battle progress
		content = append(content, m.renderLiveBattle())
	} else {
		// Final results
		content = append(content, m.renderFinalResults())
	}

	return strings.Join(content, "\n")
}

// renderLiveBattle renders the real-time battle progress
func (m ComparisonModel) renderLiveBattle() string {
	var sections []string

	// Battle status header
	elapsed := time.Since(m.startTime)
	statusStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color(styles.ProcessingBlue))

	status := fmt.Sprintf("🔥 BATTLE TIME: %v • COMPLETED: %d/%d",
		elapsed.Round(time.Second), m.completed, m.total)
	sections = append(sections, statusStyle.Render(status))
	sections = append(sections, "")

	// Live competitor status
	sections = append(sections, styles.SubheadStyle.Render("🥊 Live Competitor Status"))
	sections = append(sections, "")

	competitors := m.renderCompetitorStatus()
	sections = append(sections, competitors)
	sections = append(sections, "")

	// Current leaderboard (as results come in)
	if len(m.leaderboard) > 0 {
		sections = append(sections, styles.SubheadStyle.Render("🏆 Live Leaderboard"))
		sections = append(sections, "")
		leaderboard := m.renderLiveLeaderboard()
		sections = append(sections, leaderboard)
	}

	return strings.Join(sections, "\n")
}

// renderCompetitorStatus shows each optimizer's current battle status
func (m ComparisonModel) renderCompetitorStatus() string {
	var competitors []string

	optimizerNames := []string{"bootstrap", "mipro", "simba", "gepa", "copro"}

	for _, optimizer := range optimizerNames {
		if !m.selectedOptimizers[optimizer] {
			continue
		}

		status := m.runningStatus[optimizer]
		icon := m.getOptimizerIcon(optimizer)
		name := strings.ToUpper(optimizer)

		var statusLine string
		switch status.State {
		case "pending":
			statusLine = fmt.Sprintf("%s %s: %s Waiting to enter arena...",
				icon, name, styles.MutedStyle.Render("⏳"))
		case "initializing":
			statusLine = fmt.Sprintf("%s %s: %s Preparing for battle...",
				icon, name, styles.InfoStyle.Render("🔄"))
		case "running":
			progress := m.renderProgressBar(status.Progress, 20)
			statusLine = fmt.Sprintf("%s %s: %s %s",
				icon, name, styles.ProcessingBlue, progress)
		case "completed":
			result := m.results[optimizer]
			if result != nil && result.Success {
				statusLine = fmt.Sprintf("%s %s: %s %.1f%% accuracy (+%.1f%%)",
					icon, name,
					styles.SuccessStyle.Render("✅"),
					result.FinalAccuracy*100,
					result.ImprovementPct)
			} else {
				statusLine = fmt.Sprintf("%s %s: %s Battle lost",
					icon, name, styles.ErrorStyle.Render("❌"))
			}
		case "failed":
			statusLine = fmt.Sprintf("%s %s: %s %s",
				icon, name,
				styles.ErrorStyle.Render("💥"),
				styles.ErrorStyle.Render("Crashed in battle"))
		default:
			statusLine = fmt.Sprintf("%s %s: Unknown status", icon, name)
		}

		competitors = append(competitors, statusLine)
	}

	return strings.Join(competitors, "\n")
}

// renderLiveLeaderboard shows current rankings as results come in
func (m ComparisonModel) renderLiveLeaderboard() string {
	if len(m.leaderboard) == 0 {
		return styles.MutedStyle.Render("🏁 Waiting for first competitor to finish...")
	}

	var leaderboard []string

	for i, entry := range m.leaderboard {
		if i >= 3 { // Show top 3 during live updates
			break
		}

		var medal string
		switch i {
		case 0:
			medal = "🥇"
		case 1:
			medal = "🥈"
		case 2:
			medal = "🥉"
		}

		line := fmt.Sprintf("%s %s %s: %.1f%% accuracy (+%.1f%% improvement)",
			medal, entry.Icon, entry.DisplayName,
			entry.Accuracy*100, entry.Improvement)

		if i == 0 {
			leaderboard = append(leaderboard, styles.SuccessStyle.Render(line))
		} else {
			leaderboard = append(leaderboard, line)
		}
	}

	return strings.Join(leaderboard, "\n")
}

// renderFinalResults shows the complete battle results
func (m ComparisonModel) renderFinalResults() string {
	var sections []string

	// Battle summary
	duration := time.Since(m.startTime)
	summaryStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color(styles.DSPyGreen))

	summary := fmt.Sprintf("🏁 BATTLE COMPLETE! Duration: %v", duration.Round(time.Second))
	sections = append(sections, summaryStyle.Render(summary))
	sections = append(sections, "")

	// Final leaderboard
	sections = append(sections, styles.TitleStyle.Render("🏆 FINAL LEADERBOARD"))
	sections = append(sections, "")

	leaderboard := m.renderFullLeaderboard()
	sections = append(sections, leaderboard)

	if m.showDetails {
		sections = append(sections, "")
		sections = append(sections, styles.SubheadStyle.Render("📊 Detailed Statistics"))
		sections = append(sections, "")
		details := m.renderDetailedStats()
		sections = append(sections, details)
	}

	return strings.Join(sections, "\n")
}

// renderFullLeaderboard shows complete final rankings
func (m ComparisonModel) renderFullLeaderboard() string {
	if len(m.leaderboard) == 0 {
		return styles.ErrorStyle.Render("❌ No results available")
	}

	// Create leaderboard table
	leaderboardBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1, 2).
		Width(m.width - 4)

	var rows []string

	// Header
	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(styles.DSPyBlue))
	header := fmt.Sprintf("%-4s %-12s %-10s %-12s %-10s",
		"RANK", "OPTIMIZER", "ACCURACY", "IMPROVEMENT", "TIME")
	rows = append(rows, headerStyle.Render(header))
	rows = append(rows, strings.Repeat("─", 60))

	// Entries
	for _, entry := range m.leaderboard {
		var rankIcon string
		var style lipgloss.Style = lipgloss.NewStyle()

		switch entry.Rank {
		case 1:
			rankIcon = "🥇"
			style = style.Foreground(lipgloss.Color("#FFD700")) // Gold
		case 2:
			rankIcon = "🥈"
			style = style.Foreground(lipgloss.Color("#C0C0C0")) // Silver
		case 3:
			rankIcon = "🥉"
			style = style.Foreground(lipgloss.Color("#CD7F32")) // Bronze
		default:
			rankIcon = fmt.Sprintf("#%d", entry.Rank)
		}

		row := fmt.Sprintf("%-4s %-12s %-10.1f%% %-12.1f%% %-10s",
			rankIcon,
			entry.DisplayName,
			entry.Accuracy*100,
			entry.Improvement,
			m.formatDuration(entry.Duration))

		rows = append(rows, style.Render(row))
	}

	return leaderboardBox.Render(strings.Join(rows, "\n"))
}

// tickSpinner returns a command to tick the spinner
func (m ComparisonModel) tickSpinner() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg {
		return spinnerTickMsg{}
	})
}

// tickAnimation returns a command to tick the animation
func (m ComparisonModel) tickAnimation() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(time.Time) tea.Msg {
		return animationTickMsg{}
	})
}


// GetNextScreen returns the next screen to navigate to
func (m ComparisonModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation resets the navigation state
func (m *ComparisonModel) ResetNavigation() {
	m.nextScreen = ""
}
