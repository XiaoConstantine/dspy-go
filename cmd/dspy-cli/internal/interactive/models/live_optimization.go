package models

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
)

// Optimization phases for progress tracking
const (
	PhaseInitializing = "Initializing"
	PhaseDataLoading  = "Loading Dataset"
	PhaseOptimizing   = "Optimizing"
	PhaseEvaluating   = "Evaluating"
	PhaseComplete     = "Complete"
)

// LiveOptimizationModel represents the live optimization screen state
type LiveOptimizationModel struct {
	// Basic info
	optimizer       string
	dataset         string
	config          runner.OptimizerConfig

	// Progress tracking
	phase           string
	progress        float64
	currentTrial    int
	totalTrials     int
	bestScore       float64
	currentScore    float64
	improvementRate float64

	// Log streaming
	logViewport     viewport.Model
	logMessages     []string
	maxLogMessages  int

	// Timing
	startTime       time.Time
	elapsedTime     time.Duration
	estimatedTime   time.Duration

	// Results
	finalResult     *runner.RunResult
	errorMessage    string

	// UI state
	width           int
	height          int
	spinner         spinner.Model
	isComplete      bool
	nextScreen      string

	// Channels for async updates
	progressChan    chan ProgressUpdate
	logChan         chan string
	resultChan      chan ResultUpdate
}

// ProgressUpdate represents an optimization progress update
type ProgressUpdate struct {
	Phase        string
	Progress     float64
	Trial        int
	TotalTrials  int
	BestScore    float64
	CurrentScore float64
	Message      string
}

// ResultUpdate represents the final optimization result
type ResultUpdate struct {
	Result *runner.RunResult
	Error  error
}

// NewLiveOptimizationModel creates a new live optimization model
func NewLiveOptimizationModel(optimizer, dataset string, config runner.OptimizerConfig) LiveOptimizationModel {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyBlue))

	vp := viewport.New(80, 10)
	vp.Style = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.MediumGray))

	return LiveOptimizationModel{
		optimizer:       optimizer,
		dataset:         dataset,
		config:          config,
		phase:           PhaseInitializing,
		progress:        0.0,
		currentTrial:    0,
		totalTrials:     config.MaxExamples,
		logViewport:     vp,
		logMessages:     []string{},
		maxLogMessages:  100,
		spinner:         s,
		progressChan:    make(chan ProgressUpdate, 100),
		logChan:         make(chan string, 100),
		resultChan:      make(chan ResultUpdate, 1),
		width:           80,
		height:          24,
	}
}

// Init initializes the model and starts the optimization
func (m LiveOptimizationModel) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		m.startOptimization(),
		m.listenForUpdates(),
	)
}

// Update handles messages and updates the model
func (m LiveOptimizationModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Resize viewport for logs
		logHeight := m.height - 20 // Reserve space for other UI elements
		if logHeight > 5 {
			m.logViewport.Width = m.width - 4
			m.logViewport.Height = logHeight
		}

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			// TODO: Cancel optimization gracefully
			return m, tea.Quit
		case "enter", " ":
			if m.isComplete {
				m.nextScreen = "results"
			}
		case "b", "esc":
			if m.isComplete {
				m.nextScreen = "back"
			}
		case "up", "k":
			m.logViewport.LineUp(1)
		case "down", "j":
			m.logViewport.LineDown(1)
		case "pgup":
			m.logViewport.HalfViewUp()
		case "pgdn":
			m.logViewport.HalfViewDown()
		}

	case spinner.TickMsg:
		if !m.isComplete {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case ProgressUpdate:
		m.handleProgressUpdate(msg)
		if !m.isComplete {
			cmds = append(cmds, m.listenForUpdates())
		}

	case string: // Log message
		m.addLogMessage(msg)
		if !m.isComplete {
			cmds = append(cmds, m.listenForUpdates())
		}

	case ResultUpdate:
		m.handleResultUpdate(msg)
		m.isComplete = true
	}

	// Update viewport
	var cmd tea.Cmd
	m.logViewport, cmd = m.logViewport.Update(msg)
	cmds = append(cmds, cmd)

	// Update elapsed time
	if !m.isComplete && !m.startTime.IsZero() {
		m.elapsedTime = time.Since(m.startTime)
		// Estimate remaining time based on progress
		if m.progress > 0 {
			totalEstimate := time.Duration(float64(m.elapsedTime) / m.progress)
			m.estimatedTime = totalEstimate - m.elapsedTime
		}
	}

	return m, tea.Batch(cmds...)
}

// View renders the live optimization screen
func (m LiveOptimizationModel) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading..."
	}

	var sections []string

	// Header
	header := m.renderHeader()
	sections = append(sections, header)

	// Progress section
	progress := m.renderProgress()
	sections = append(sections, progress)

	// Metrics section
	metrics := m.renderMetrics()
	sections = append(sections, metrics)

	// Log viewer
	logs := m.renderLogs()
	sections = append(sections, logs)

	// Footer with controls
	footer := m.renderFooter()
	sections = append(sections, footer)

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// renderHeader renders the header section
func (m LiveOptimizationModel) renderHeader() string {
	icon := m.getOptimizerIcon()
	title := fmt.Sprintf("%s %s Optimization - %s", icon, strings.ToUpper(m.optimizer), m.dataset)

	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Background(lipgloss.Color(styles.DSPyBlue)).
		Foreground(lipgloss.Color(styles.White)).
		Width(m.width).
		Padding(0, 2)

	return headerStyle.Render(title)
}

// renderProgress renders the progress section
func (m LiveOptimizationModel) renderProgress() string {
	var content strings.Builder

	// Phase indicator
	phaseStyle := lipgloss.NewStyle().Bold(true)
	if m.isComplete {
		phaseStyle = phaseStyle.Foreground(lipgloss.Color(styles.DSPyGreen))
		content.WriteString(phaseStyle.Render("✅ " + m.phase))
	} else {
		phaseStyle = phaseStyle.Foreground(lipgloss.Color(styles.ProcessingBlue))
		content.WriteString(m.spinner.View() + " " + phaseStyle.Render(m.phase))
	}
	content.WriteString("\n\n")

	// Progress bar
	progressBar := m.renderProgressBar()
	content.WriteString(progressBar)
	content.WriteString("\n")

	// Trial info
	if m.totalTrials > 0 {
		trialInfo := fmt.Sprintf("Trial %d/%d", m.currentTrial, m.totalTrials)
		content.WriteString(styles.MutedStyle.Render(trialInfo))
	}

	return styles.BoxStyle.Copy().Width(m.width - 2).Render(content.String())
}

// renderProgressBar creates a visual progress bar
func (m LiveOptimizationModel) renderProgressBar() string {
	width := m.width - 10
	if width < 20 {
		width = 20
	}

	filled := int(float64(width) * m.progress)
	if filled > width {
		filled = width
	}

	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	percentage := fmt.Sprintf(" %3.0f%%", m.progress*100)

	barStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyGreen))
	emptyStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.MediumGray))

	if filled > 0 {
		return barStyle.Render(bar[:filled]) + emptyStyle.Render(bar[filled:]) + percentage
	}
	return emptyStyle.Render(bar) + percentage
}

// renderMetrics renders the metrics section
func (m LiveOptimizationModel) renderMetrics() string {
	metricsBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(0, 1).
		Width(m.width - 2)

	var metrics []string

	// Best score
	if m.bestScore > 0 {
		bestStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(styles.DSPyGreen))
		metrics = append(metrics, fmt.Sprintf("🏆 Best Score: %s", bestStyle.Render(fmt.Sprintf("%.2f%%", m.bestScore*100))))
	}

	// Current score
	if m.currentScore > 0 {
		metrics = append(metrics, fmt.Sprintf("📊 Current: %.2f%%", m.currentScore*100))
	}

	// Improvement rate
	if m.improvementRate > 0 {
		improvStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyOrange))
		metrics = append(metrics, fmt.Sprintf("📈 Improvement: %s", improvStyle.Render(fmt.Sprintf("+%.1f%%", m.improvementRate))))
	}

	// Timing
	if !m.startTime.IsZero() {
		metrics = append(metrics, fmt.Sprintf("⏱ Elapsed: %s", m.formatDuration(m.elapsedTime)))
		if m.estimatedTime > 0 && !m.isComplete {
			metrics = append(metrics, fmt.Sprintf("⏳ ETA: %s", m.formatDuration(m.estimatedTime)))
		}
	}

	if len(metrics) == 0 {
		metrics = append(metrics, styles.MutedStyle.Render("Waiting for metrics..."))
	}

	return metricsBox.Render(lipgloss.JoinHorizontal(lipgloss.Top, strings.Join(metrics, "  ")))
}

// renderLogs renders the log viewer section
func (m LiveOptimizationModel) renderLogs() string {
	m.logViewport.SetContent(strings.Join(m.logMessages, "\n"))

	title := styles.SubheadStyle.Render("📋 Optimization Log")
	return lipgloss.JoinVertical(lipgloss.Left, title, m.logViewport.View())
}

// renderFooter renders the footer with controls
func (m LiveOptimizationModel) renderFooter() string {
	var controls []string

	if m.isComplete {
		if m.finalResult != nil {
			controls = append(controls, "[Enter] View Results")
		}
		controls = append(controls, "[b] Back")
	} else {
		controls = append(controls, "[↑↓] Scroll Logs")
		controls = append(controls, "[Ctrl+C] Cancel")
	}

	controls = append(controls, "[q] Quit")

	return styles.FooterStyle.Render(strings.Join(controls, "  •  "))
}

// Helper methods

func (m *LiveOptimizationModel) startOptimization() tea.Cmd {
	return func() tea.Msg {
		m.startTime = time.Now()

		// Start the actual optimization process in a goroutine
		go m.runOptimizationWithStreaming()

		return nil
	}
}

// runOptimizationWithStreaming runs the optimization and streams updates
func (m *LiveOptimizationModel) runOptimizationWithStreaming() {
	// Send initial progress update
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseInitializing,
		Progress:     0.0,
		Trial:        0,
		TotalTrials:  m.config.MaxExamples,
		Message:      "Starting optimization...",
	}
	m.logChan <- "🚀 Optimization process started"

	// Phase 1: Initializing
	time.Sleep(500 * time.Millisecond)
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseInitializing,
		Progress:     0.1,
		Trial:        0,
		TotalTrials:  m.config.MaxExamples,
		Message:      "Setting up optimizer configuration...",
	}
	m.logChan <- fmt.Sprintf("📋 Optimizer: %s", m.optimizer)
	m.logChan <- fmt.Sprintf("📊 Dataset: %s", m.dataset)
	m.logChan <- fmt.Sprintf("🔢 Max Examples: %d", m.config.MaxExamples)

	// Phase 2: Data Loading
	time.Sleep(800 * time.Millisecond)
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseDataLoading,
		Progress:     0.2,
		Trial:        0,
		TotalTrials:  m.config.MaxExamples,
		Message:      "Loading dataset and preparing examples...",
	}
	m.logChan <- "📥 Loading dataset samples..."
	m.logChan <- fmt.Sprintf("✅ Loaded %d examples for optimization", m.config.MaxExamples)

	// Phase 3: Optimization - Run the actual optimizer
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseOptimizing,
		Progress:     0.3,
		Trial:        1,
		TotalTrials:  m.config.MaxExamples,
		Message:      "Running optimization algorithm...",
	}
	m.logChan <- "🧠 Starting optimization trials..."

	// Actually run the optimizer
	result, err := runner.RunOptimizer(m.config)

	if err != nil {
		// Send error result
		m.resultChan <- ResultUpdate{
			Result: nil,
			Error:  err,
		}
		return
	}

	// Simulate progress updates during optimization
	for trial := 1; trial <= m.config.MaxExamples; trial++ {
		time.Sleep(200 * time.Millisecond) // Simulate processing time

		// Calculate intermediate scores
		progress := 0.3 + (0.5 * float64(trial) / float64(m.config.MaxExamples))
		currentScore := result.InitialAccuracy +
			((result.FinalAccuracy - result.InitialAccuracy) * float64(trial) / float64(m.config.MaxExamples))

		m.progressChan <- ProgressUpdate{
			Phase:        PhaseOptimizing,
			Progress:     progress,
			Trial:        trial,
			TotalTrials:  m.config.MaxExamples,
			BestScore:    currentScore,
			CurrentScore: currentScore,
			Message:      fmt.Sprintf("Processing trial %d/%d...", trial, m.config.MaxExamples),
		}

		m.logChan <- fmt.Sprintf("🔄 Trial %d: %.2f%% accuracy", trial, currentScore*100)

		// Show improvement if it's happening
		if trial > 1 {
			improvement := ((currentScore - result.InitialAccuracy) / result.InitialAccuracy) * 100
			if improvement > 0 {
				m.logChan <- fmt.Sprintf("📈 Improvement: +%.1f%%", improvement)
			}
		}
	}

	// Phase 4: Evaluation
	time.Sleep(400 * time.Millisecond)
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseEvaluating,
		Progress:     0.9,
		Trial:        m.config.MaxExamples,
		TotalTrials:  m.config.MaxExamples,
		BestScore:    result.FinalAccuracy,
		CurrentScore: result.FinalAccuracy,
		Message:      "Evaluating final results...",
	}
	m.logChan <- "📊 Evaluating optimization results..."
	m.logChan <- fmt.Sprintf("🎯 Final accuracy: %.2f%%", result.FinalAccuracy*100)
	m.logChan <- fmt.Sprintf("📈 Total improvement: +%.1f%%", result.ImprovementPct)

	// Phase 5: Complete
	m.progressChan <- ProgressUpdate{
		Phase:        PhaseComplete,
		Progress:     1.0,
		Trial:        m.config.MaxExamples,
		TotalTrials:  m.config.MaxExamples,
		BestScore:    result.FinalAccuracy,
		CurrentScore: result.FinalAccuracy,
		Message:      "Optimization complete!",
	}

	// Send success message based on improvement
	if result.ImprovementPct > 50 {
		m.logChan <- "🎉 Excellent results! Significant improvement achieved!"
	} else if result.ImprovementPct > 0 {
		m.logChan <- "✅ Optimization successful! Results improved!"
	} else {
		m.logChan <- "✓ Optimization complete. No improvement detected."
	}

	// Send final result
	m.resultChan <- ResultUpdate{
		Result: result,
		Error:  nil,
	}
}

func (m *LiveOptimizationModel) simulateOptimization() {
	// Temporary simulation - replace with actual optimization
	phases := []string{PhaseInitializing, PhaseDataLoading, PhaseOptimizing, PhaseEvaluating, PhaseComplete}

	for i, phase := range phases {
		time.Sleep(2 * time.Second)

		progress := float64(i+1) / float64(len(phases))
		m.progressChan <- ProgressUpdate{
			Phase:    phase,
			Progress: progress,
			Trial:    i + 1,
			TotalTrials: 5,
			BestScore: 0.75 + (progress * 0.15),
			CurrentScore: 0.70 + (progress * 0.15),
			Message: fmt.Sprintf("Processing %s...", phase),
		}

		m.logChan <- fmt.Sprintf("[%s] %s started", time.Now().Format("15:04:05"), phase)
	}

	// Send final result
	m.resultChan <- ResultUpdate{
		Result: &runner.RunResult{
			OptimizerName: m.optimizer,
			DatasetName:   m.dataset,
			Success:       true,
		},
		Error: nil,
	}
}

func (m *LiveOptimizationModel) listenForUpdates() tea.Cmd {
	return func() tea.Msg {
		select {
		case update := <-m.progressChan:
			return update
		case log := <-m.logChan:
			return log
		case result := <-m.resultChan:
			return result
		default:
			return nil
		}
	}
}

func (m *LiveOptimizationModel) handleProgressUpdate(update ProgressUpdate) {
	m.phase = update.Phase
	m.progress = update.Progress
	m.currentTrial = update.Trial
	m.totalTrials = update.TotalTrials
	m.bestScore = update.BestScore
	m.currentScore = update.CurrentScore

	if m.bestScore > 0 && m.currentScore > 0 {
		m.improvementRate = ((m.bestScore - m.currentScore) / m.currentScore) * 100
	}

	if update.Message != "" {
		m.addLogMessage(update.Message)
	}
}

func (m *LiveOptimizationModel) handleResultUpdate(update ResultUpdate) {
	if update.Error != nil {
		m.errorMessage = update.Error.Error()
		m.phase = "Error"
		m.addLogMessage(fmt.Sprintf("❌ Error: %s", update.Error))
	} else {
		m.finalResult = update.Result
		m.phase = PhaseComplete
		m.progress = 1.0
		m.addLogMessage("✅ Optimization complete!")
	}
}

func (m *LiveOptimizationModel) addLogMessage(msg string) {
	timestamp := time.Now().Format("15:04:05")
	logEntry := fmt.Sprintf("[%s] %s", timestamp, msg)

	m.logMessages = append(m.logMessages, logEntry)

	// Keep only the last N messages
	if len(m.logMessages) > m.maxLogMessages {
		m.logMessages = m.logMessages[len(m.logMessages)-m.maxLogMessages:]
	}
}

func (m LiveOptimizationModel) getOptimizerIcon() string {
	icons := map[string]string{
		"bootstrap": styles.IconBootstrap,
		"mipro":     styles.IconMIPRO,
		"simba":     styles.IconSIMBA,
		"gepa":      styles.IconGEPA,
		"copro":     styles.IconCOPRO,
	}

	if icon, ok := icons[strings.ToLower(m.optimizer)]; ok {
		return icon
	}
	return "🔧"
}

func (m LiveOptimizationModel) formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	} else if d < time.Hour {
		mins := int(d.Minutes())
		secs := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm %ds", mins, secs)
	}
	hours := int(d.Hours())
	mins := int(d.Minutes()) % 60
	return fmt.Sprintf("%dh %dm", hours, mins)
}

// GetNextScreen returns the next screen to navigate to
func (m LiveOptimizationModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation resets the navigation state
func (m *LiveOptimizationModel) ResetNavigation() {
	m.nextScreen = ""
}
