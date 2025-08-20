package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
)

// ResultsModel represents the results visualization screen
type ResultsModel struct {
	result         *runner.RunResult
	comparisonData []runner.RunResult // For comparing multiple runs

	// UI state
	width          int
	height         int
	viewport       viewport.Model
	currentView    string // "summary", "details", "export"
	selectedExport int
	nextScreen     string

	// Animation
	animationFrame int
	animating      bool
}

// Export formats
var exportFormats = []string{
	"JSON",
	"CSV",
	"Markdown",
	"Copy to Clipboard",
}

// NewResultsModel creates a new results visualization model
func NewResultsModel(result *runner.RunResult) ResultsModel {
	vp := viewport.New(80, 20)
	vp.Style = lipgloss.NewStyle().
		BorderStyle(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue))

	return ResultsModel{
		result:      result,
		viewport:    vp,
		currentView: "summary",
		animating:   true,
		width:       80,
		height:      24,
	}
}

// Init initializes the model
func (m ResultsModel) Init() tea.Cmd {
	return m.animate()
}

// Update handles messages and updates the model
func (m ResultsModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		// Resize viewport
		m.viewport.Width = m.width - 4
		m.viewport.Height = m.height - 15

	case tea.KeyMsg:
		switch msg.String() {
		case "tab":
			// Cycle through views
			switch m.currentView {
			case "summary":
				m.currentView = "details"
			case "details":
				m.currentView = "export"
			case "export":
				m.currentView = "summary"
			}
			m.updateViewportContent()

		case "s":
			m.currentView = "summary"
			m.updateViewportContent()

		case "d":
			m.currentView = "details"
			m.updateViewportContent()

		case "e":
			m.currentView = "export"
			m.updateViewportContent()

		case "up", "k":
			if m.currentView == "export" {
				if m.selectedExport > 0 {
					m.selectedExport--
				}
			} else {
				m.viewport.LineUp(1)
			}

		case "down", "j":
			if m.currentView == "export" {
				if m.selectedExport < len(exportFormats)-1 {
					m.selectedExport++
				}
			} else {
				m.viewport.LineDown(1)
			}

		case "pgup":
			m.viewport.HalfViewUp()

		case "pgdn":
			m.viewport.HalfViewDown()

		case "enter", " ":
			if m.currentView == "export" {
				m.exportResults()
			}

		case "b", "esc":
			m.nextScreen = "back"

		case "r":
			// Run again
			m.nextScreen = "run_again"

		case "c":
			// Compare with another run
			m.nextScreen = "comparison"

		case "q":
			return m, tea.Quit
		}

	case animationMsg:
		if m.animating {
			m.animationFrame++
			if m.animationFrame > 10 {
				m.animating = false
			} else {
				cmds = append(cmds, m.animate())
			}
		}
	}

	// Update viewport
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

// View renders the results screen
func (m ResultsModel) View() string {
	if m.width == 0 || m.height == 0 {
		return "Loading results..."
	}

	var sections []string

	// Header
	header := m.renderHeader()
	sections = append(sections, header)

	// Success/Failure banner
	banner := m.renderResultBanner()
	sections = append(sections, banner)

	// Main content based on current view
	switch m.currentView {
	case "summary":
		content := m.renderSummary()
		sections = append(sections, content)
	case "details":
		content := m.renderDetails()
		sections = append(sections, content)
	case "export":
		content := m.renderExport()
		sections = append(sections, content)
	}

	// Footer
	footer := m.renderFooter()
	sections = append(sections, footer)

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// renderHeader renders the header section
func (m ResultsModel) renderHeader() string {
	title := fmt.Sprintf("📊 Optimization Results - %s on %s",
		strings.ToUpper(m.result.OptimizerName),
		m.result.DatasetName)

	headerStyle := lipgloss.NewStyle().
		Bold(true).
		Background(lipgloss.Color(styles.DSPyBlue)).
		Foreground(lipgloss.Color(styles.White)).
		Width(m.width).
		Padding(0, 2)

	// View tabs
	var tabs []string
	for _, view := range []string{"summary", "details", "export"} {
		tabStyle := lipgloss.NewStyle().Padding(0, 2)
		if view == m.currentView {
			tabStyle = tabStyle.
				Bold(true).
				Foreground(lipgloss.Color(styles.DSPyGreen))
			tabs = append(tabs, tabStyle.Render("▶ "+strings.Title(view)))
		} else {
			tabStyle = tabStyle.Foreground(lipgloss.Color(styles.MediumGray))
			tabs = append(tabs, tabStyle.Render(strings.Title(view)))
		}
	}

	tabBar := lipgloss.JoinHorizontal(lipgloss.Top, tabs...)

	return lipgloss.JoinVertical(lipgloss.Left, headerStyle.Render(title), tabBar)
}

// renderResultBanner renders success/failure banner with animation
func (m ResultsModel) renderResultBanner() string {
	if m.result.Success {
		// Success animation
		var icon string
		if m.animating {
			icons := []string{"🎯", "🎉", "✨", "🚀", "💪", "🏆"}
			icon = icons[m.animationFrame%len(icons)]
		} else {
			icon = "✅"
		}

		improvement := m.result.ImprovementPct
		var message string
		if improvement > 100 {
			message = fmt.Sprintf("%s AMAZING! %.1f%% improvement achieved!", icon, improvement)
		} else if improvement > 50 {
			message = fmt.Sprintf("%s EXCELLENT! %.1f%% improvement achieved!", icon, improvement)
		} else if improvement > 0 {
			message = fmt.Sprintf("%s SUCCESS! %.1f%% improvement achieved!", icon, improvement)
		} else {
			message = fmt.Sprintf("%s Optimization complete (no improvement)", icon)
		}

		bannerStyle := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color(styles.DSPyGreen)).
			Border(lipgloss.DoubleBorder()).
			BorderForeground(lipgloss.Color(styles.DSPyGreen)).
			Padding(1, 2).
			Width(m.width - 2).
			Align(lipgloss.Center)

		return bannerStyle.Render(message)
	} else {
		// Error banner
		bannerStyle := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color(styles.DSPyRed)).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color(styles.DSPyRed)).
			Padding(1, 2).
			Width(m.width - 2)

		return bannerStyle.Render(fmt.Sprintf("❌ Optimization failed: %s", m.result.ErrorMessage))
	}
}

// renderSummary renders the summary view with charts
func (m ResultsModel) renderSummary() string {
	var content strings.Builder

	// Key metrics cards
	metricsBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1, 2).
		Width(m.width - 2)

	// Accuracy chart
	accuracyChart := m.renderAccuracyChart()
	content.WriteString(styles.SubheadStyle.Render("📈 Accuracy Improvement"))
	content.WriteString("\n\n")
	content.WriteString(accuracyChart)
	content.WriteString("\n\n")

	// Performance metrics grid
	metrics := m.renderMetricsGrid()
	content.WriteString(styles.SubheadStyle.Render("📊 Key Metrics"))
	content.WriteString("\n\n")
	content.WriteString(metrics)
	content.WriteString("\n\n")

	// Time and efficiency
	efficiency := m.renderEfficiencyMetrics()
	content.WriteString(styles.SubheadStyle.Render("⚡ Efficiency"))
	content.WriteString("\n\n")
	content.WriteString(efficiency)

	return metricsBox.Render(content.String())
}

// renderAccuracyChart creates a visual bar chart for accuracy comparison
func (m ResultsModel) renderAccuracyChart() string {
	initial := m.result.InitialAccuracy
	final := m.result.FinalAccuracy

	maxWidth := 40
	initialWidth := int(initial * float64(maxWidth))
	finalWidth := int(final * float64(maxWidth))

	// Create bars
	initialBar := strings.Repeat("█", initialWidth) + strings.Repeat("░", maxWidth-initialWidth)
	finalBar := strings.Repeat("█", finalWidth) + strings.Repeat("░", maxWidth-finalWidth)

	initialStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.MediumGray))
	finalStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyGreen))

	var chart strings.Builder
	chart.WriteString(fmt.Sprintf("Initial: %s %.1f%%\n",
		initialStyle.Render(initialBar), initial*100))
	chart.WriteString(fmt.Sprintf("Final:   %s %.1f%%",
		finalStyle.Render(finalBar), final*100))

	// Add improvement arrow if there's improvement
	if m.result.ImprovementPct > 0 {
		improvementStyle := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color(styles.DSPyOrange))
		chart.WriteString("\n")
		chart.WriteString(improvementStyle.Render(fmt.Sprintf("         ↑ +%.1f%% improvement", m.result.ImprovementPct)))
	}

	return chart.String()
}

// renderMetricsGrid creates a grid of key metrics
func (m ResultsModel) renderMetricsGrid() string {
	metricStyle := lipgloss.NewStyle().
		Border(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color(styles.MediumGray)).
		Padding(0, 1).
		Width(20).
		Height(3).
		Align(lipgloss.Center)

	valueStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(styles.DSPyBlue))
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.MediumGray))

	// Create metric cards
	metrics := []string{
		metricStyle.Render(
			valueStyle.Render(fmt.Sprintf("%.1f%%", m.result.FinalAccuracy*100)) + "\n" +
			labelStyle.Render("Final Accuracy")),
		metricStyle.Render(
			valueStyle.Render(fmt.Sprintf("+%.1f%%", m.result.ImprovementPct)) + "\n" +
			labelStyle.Render("Improvement")),
		metricStyle.Render(
			valueStyle.Render(fmt.Sprintf("%d", m.result.ExamplesUsed)) + "\n" +
			labelStyle.Render("Examples Used")),
	}

	return lipgloss.JoinHorizontal(lipgloss.Top, metrics...)
}

// renderEfficiencyMetrics renders efficiency statistics
func (m ResultsModel) renderEfficiencyMetrics() string {
	var efficiency strings.Builder

	// Calculate efficiency score
	efficiencyScore := m.calculateEfficiencyScore()

	// Time taken
	timeStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.ProcessingBlue))
	efficiency.WriteString(fmt.Sprintf("⏱ Time: %s\n", timeStyle.Render(m.formatDuration(m.result.Duration))))

	// Examples per second
	if m.result.Duration > 0 {
		examplesPerSec := float64(m.result.ExamplesUsed) / m.result.Duration.Seconds()
		efficiency.WriteString(fmt.Sprintf("📝 Processing Rate: %.1f examples/sec\n", examplesPerSec))
	}

	// Efficiency rating
	var rating string
	var ratingStyle lipgloss.Style
	if efficiencyScore > 0.8 {
		rating = "⚡ Highly Efficient"
		ratingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyGreen))
	} else if efficiencyScore > 0.5 {
		rating = "✓ Good Efficiency"
		ratingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyBlue))
	} else {
		rating = "🐢 Could be optimized"
		ratingStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyOrange))
	}
	efficiency.WriteString(ratingStyle.Render(rating))

	return efficiency.String()
}

// renderDetails renders detailed statistics view
func (m ResultsModel) renderDetails() string {
	m.viewport.SetContent(m.getDetailedStats())
	return m.viewport.View()
}

// getDetailedStats generates detailed statistics content
func (m ResultsModel) getDetailedStats() string {
	var details strings.Builder

	// Optimizer configuration
	details.WriteString(styles.HeadingStyle.Render("🔧 Optimizer Configuration"))
	details.WriteString("\n\n")
	details.WriteString(fmt.Sprintf("  • Optimizer: %s\n", m.result.OptimizerName))
	details.WriteString(fmt.Sprintf("  • Dataset: %s\n", m.result.DatasetName))
	details.WriteString(fmt.Sprintf("  • Examples: %d\n", m.result.ExamplesUsed))
	details.WriteString("\n")

	// Performance breakdown
	details.WriteString(styles.HeadingStyle.Render("📊 Performance Breakdown"))
	details.WriteString("\n\n")
	details.WriteString(fmt.Sprintf("  • Initial Accuracy: %.2f%%\n", m.result.InitialAccuracy*100))
	details.WriteString(fmt.Sprintf("  • Final Accuracy: %.2f%%\n", m.result.FinalAccuracy*100))
	details.WriteString(fmt.Sprintf("  • Absolute Improvement: %.2f%%\n", (m.result.FinalAccuracy-m.result.InitialAccuracy)*100))
	details.WriteString(fmt.Sprintf("  • Relative Improvement: %.2f%%\n", m.result.ImprovementPct))
	details.WriteString("\n")

	// Statistical analysis
	details.WriteString(styles.HeadingStyle.Render("📈 Statistical Analysis"))
	details.WriteString("\n\n")

	// Calculate confidence interval (simplified)
	sampleSize := m.result.ExamplesUsed
	accuracy := m.result.FinalAccuracy
	marginOfError := 1.96 * math.Sqrt((accuracy*(1-accuracy))/float64(sampleSize))

	details.WriteString(fmt.Sprintf("  • 95%% Confidence Interval: %.2f%% ± %.2f%%\n",
		accuracy*100, marginOfError*100))
	details.WriteString(fmt.Sprintf("  • Sample Size: %d examples\n", sampleSize))

	// Add statistical significance indicator
	if m.result.ImprovementPct > marginOfError*100 {
		details.WriteString(styles.SuccessStyle.Render("  • ✓ Statistically significant improvement\n"))
	} else {
		details.WriteString(styles.WarningStyle.Render("  • ⚠ Improvement within margin of error\n"))
	}
	details.WriteString("\n")

	// Resource usage
	details.WriteString(styles.HeadingStyle.Render("💾 Resource Usage"))
	details.WriteString("\n\n")
	details.WriteString(fmt.Sprintf("  • Total Duration: %s\n", m.formatDuration(m.result.Duration)))
	details.WriteString(fmt.Sprintf("  • Avg Time per Example: %s\n",
		m.formatDuration(m.result.Duration/time.Duration(m.result.ExamplesUsed))))

	// Estimated API calls (assuming 2 calls per example for optimization)
	estimatedAPICalls := m.result.ExamplesUsed * 2
	details.WriteString(fmt.Sprintf("  • Estimated API Calls: ~%d\n", estimatedAPICalls))

	return details.String()
}

// renderExport renders the export options view
func (m ResultsModel) renderExport() string {
	var content strings.Builder

	content.WriteString(styles.SubheadStyle.Render("💾 Export Results"))
	content.WriteString("\n\n")
	content.WriteString(styles.BodyStyle.Render("Choose an export format:"))
	content.WriteString("\n\n")

	for i, format := range exportFormats {
		var icon string
		switch format {
		case "JSON":
			icon = "📋"
		case "CSV":
			icon = "📊"
		case "Markdown":
			icon = "📝"
		case "Copy to Clipboard":
			icon = "📎"
		}

		if i == m.selectedExport {
			content.WriteString(styles.SelectedStyle.Render(fmt.Sprintf("%s %s %s",
				styles.IconSelected, icon, format)))
		} else {
			content.WriteString(styles.UnselectedStyle.Render(fmt.Sprintf("  %s %s",
				icon, format)))
		}
		content.WriteString("\n")
	}

	content.WriteString("\n")
	content.WriteString(styles.CaptionStyle.Render("Press Enter to export in selected format"))

	exportBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(styles.DSPyBlue)).
		Padding(1, 2).
		Width(m.width - 2)

	return exportBox.Render(content.String())
}

// renderFooter renders the footer with controls
func (m ResultsModel) renderFooter() string {
	var controls []string

	switch m.currentView {
	case "summary":
		controls = append(controls, "[Tab] Switch View", "[r] Run Again", "[c] Compare")
	case "details":
		controls = append(controls, "[↑↓] Scroll", "[Tab] Switch View")
	case "export":
		controls = append(controls, "[↑↓] Select", "[Enter] Export", "[Tab] Switch View")
	}

	controls = append(controls, "[b] Back", "[q] Quit")

	return styles.FooterStyle.Render(strings.Join(controls, " • "))
}

// Helper methods

func (m *ResultsModel) updateViewportContent() {
	if m.currentView == "details" {
		m.viewport.SetContent(m.getDetailedStats())
	}
}

func (m ResultsModel) calculateEfficiencyScore() float64 {
	// Simple efficiency calculation based on improvement per time
	if m.result.Duration.Seconds() == 0 {
		return 0
	}

	improvementPerSecond := m.result.ImprovementPct / m.result.Duration.Seconds()
	// Normalize to 0-1 scale (assuming 10% improvement per second is excellent)
	return math.Min(improvementPerSecond/10.0, 1.0)
}

func (m ResultsModel) formatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	} else if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else if d < time.Hour {
		mins := int(d.Minutes())
		secs := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm %ds", mins, secs)
	}
	hours := int(d.Hours())
	mins := int(d.Minutes()) % 60
	return fmt.Sprintf("%dh %dm", hours, mins)
}

func (m *ResultsModel) exportResults() {
	format := exportFormats[m.selectedExport]

	switch format {
	case "JSON":
		m.exportAsJSON()
	case "CSV":
		m.exportAsCSV()
	case "Markdown":
		m.exportAsMarkdown()
	case "Copy to Clipboard":
		m.copyToClipboard()
	}
}

func (m *ResultsModel) exportAsJSON() {
	// Create JSON structure
	exportData := map[string]interface{}{
		"optimizer":        m.result.OptimizerName,
		"dataset":         m.result.DatasetName,
		"initial_accuracy": fmt.Sprintf("%.2f%%", m.result.InitialAccuracy*100),
		"final_accuracy":   fmt.Sprintf("%.2f%%", m.result.FinalAccuracy*100),
		"improvement":      fmt.Sprintf("%.2f%%", m.result.ImprovementPct),
		"duration":         m.result.Duration.String(),
		"examples_used":    m.result.ExamplesUsed,
		"success":          m.result.Success,
		"timestamp":        time.Now().Format(time.RFC3339),
	}

	// Write to file
	filename := fmt.Sprintf("dspy_results_%s_%s.json",
		m.result.OptimizerName,
		time.Now().Format("20060102_150405"))

	jsonData, _ := json.MarshalIndent(exportData, "", "  ")
	os.WriteFile(filename, jsonData, 0644)

	// Update UI to show success
	m.addExportNotification(fmt.Sprintf("✅ Exported to %s", filename))
}

func (m *ResultsModel) exportAsCSV() {
	// Create CSV content
	var csvContent strings.Builder
	csvContent.WriteString("Metric,Value\n")
	csvContent.WriteString(fmt.Sprintf("Optimizer,%s\n", m.result.OptimizerName))
	csvContent.WriteString(fmt.Sprintf("Dataset,%s\n", m.result.DatasetName))
	csvContent.WriteString(fmt.Sprintf("Initial Accuracy,%.2f%%\n", m.result.InitialAccuracy*100))
	csvContent.WriteString(fmt.Sprintf("Final Accuracy,%.2f%%\n", m.result.FinalAccuracy*100))
	csvContent.WriteString(fmt.Sprintf("Improvement,%.2f%%\n", m.result.ImprovementPct))
	csvContent.WriteString(fmt.Sprintf("Duration,%s\n", m.result.Duration))
	csvContent.WriteString(fmt.Sprintf("Examples Used,%d\n", m.result.ExamplesUsed))
	csvContent.WriteString(fmt.Sprintf("Success,%v\n", m.result.Success))

	// Write to file
	filename := fmt.Sprintf("dspy_results_%s_%s.csv",
		m.result.OptimizerName,
		time.Now().Format("20060102_150405"))

	os.WriteFile(filename, []byte(csvContent.String()), 0644)

	// Update UI to show success
	m.addExportNotification(fmt.Sprintf("✅ Exported to %s", filename))
}

func (m *ResultsModel) exportAsMarkdown() {
	// Create Markdown report
	var mdContent strings.Builder
	mdContent.WriteString(fmt.Sprintf("# DSPy Optimization Results\n\n"))
	mdContent.WriteString(fmt.Sprintf("**Date:** %s\n\n", time.Now().Format("January 2, 2006 15:04:05")))

	mdContent.WriteString("## Configuration\n\n")
	mdContent.WriteString(fmt.Sprintf("- **Optimizer:** %s\n", m.result.OptimizerName))
	mdContent.WriteString(fmt.Sprintf("- **Dataset:** %s\n", m.result.DatasetName))
	mdContent.WriteString(fmt.Sprintf("- **Examples Used:** %d\n\n", m.result.ExamplesUsed))

	mdContent.WriteString("## Results\n\n")
	mdContent.WriteString("| Metric | Value |\n")
	mdContent.WriteString("|--------|-------|\n")
	mdContent.WriteString(fmt.Sprintf("| Initial Accuracy | %.2f%% |\n", m.result.InitialAccuracy*100))
	mdContent.WriteString(fmt.Sprintf("| Final Accuracy | %.2f%% |\n", m.result.FinalAccuracy*100))
	mdContent.WriteString(fmt.Sprintf("| **Improvement** | **%.2f%%** |\n", m.result.ImprovementPct))
	mdContent.WriteString(fmt.Sprintf("| Duration | %s |\n\n", m.result.Duration))

	if m.result.Success {
		mdContent.WriteString("✅ **Status:** Optimization completed successfully\n\n")
	} else {
		mdContent.WriteString(fmt.Sprintf("❌ **Status:** Failed - %s\n\n", m.result.ErrorMessage))
	}

	// Statistical significance
	sampleSize := m.result.ExamplesUsed
	accuracy := m.result.FinalAccuracy
	marginOfError := 1.96 * math.Sqrt((accuracy*(1-accuracy))/float64(sampleSize))

	mdContent.WriteString("## Statistical Analysis\n\n")
	mdContent.WriteString(fmt.Sprintf("- 95%% Confidence Interval: %.2f%% ± %.2f%%\n",
		accuracy*100, marginOfError*100))

	if m.result.ImprovementPct > marginOfError*100 {
		mdContent.WriteString("- ✓ Statistically significant improvement\n")
	} else {
		mdContent.WriteString("- ⚠ Improvement within margin of error\n")
	}

	// Write to file
	filename := fmt.Sprintf("dspy_report_%s_%s.md",
		m.result.OptimizerName,
		time.Now().Format("20060102_150405"))

	os.WriteFile(filename, []byte(mdContent.String()), 0644)

	// Update UI to show success
	m.addExportNotification(fmt.Sprintf("✅ Exported to %s", filename))
}

func (m *ResultsModel) copyToClipboard() {
	// Create formatted text for clipboard
	var clipboardText strings.Builder
	clipboardText.WriteString(fmt.Sprintf("DSPy Optimization Results\n"))
	clipboardText.WriteString(fmt.Sprintf("========================\n\n"))
	clipboardText.WriteString(fmt.Sprintf("Optimizer: %s\n", m.result.OptimizerName))
	clipboardText.WriteString(fmt.Sprintf("Dataset: %s\n", m.result.DatasetName))
	clipboardText.WriteString(fmt.Sprintf("Initial: %.2f%%\n", m.result.InitialAccuracy*100))
	clipboardText.WriteString(fmt.Sprintf("Final: %.2f%%\n", m.result.FinalAccuracy*100))
	clipboardText.WriteString(fmt.Sprintf("Improvement: +%.2f%%\n", m.result.ImprovementPct))
	clipboardText.WriteString(fmt.Sprintf("Duration: %s\n", m.result.Duration))
	clipboardText.WriteString(fmt.Sprintf("Examples: %d\n", m.result.ExamplesUsed))

	// For clipboard, we'll write to a temp file and tell user to copy
	// (Full clipboard integration would require external package)
	tempFile := "/tmp/dspy_results.txt"
	os.WriteFile(tempFile, []byte(clipboardText.String()), 0644)

	// Update UI with instructions
	m.addExportNotification(fmt.Sprintf("📋 Results saved to %s\nUse: pbcopy < %s (macOS) or xclip -sel clip < %s (Linux)",
		tempFile, tempFile, tempFile))
}

func (m *ResultsModel) addExportNotification(message string) {
	// This would update the UI to show export success
	// For now, we'll just append to the current content
	currentContent := m.getDetailedStats()
	newContent := currentContent + "\n\n" + styles.SuccessStyle.Render(message)
	m.viewport.SetContent(newContent)
}

// Animation message
type animationMsg struct{}

func (m ResultsModel) animate() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg {
		return animationMsg{}
	})
}

// GetNextScreen returns the next screen to navigate to
func (m ResultsModel) GetNextScreen() string {
	return m.nextScreen
}

// ResetNavigation resets the navigation state
func (m *ResultsModel) ResetNavigation() {
	m.nextScreen = ""
}

// SetComparisonData sets comparison data for multiple runs
func (m *ResultsModel) SetComparisonData(data []runner.RunResult) {
	m.comparisonData = data
}
