package models

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive/styles"
)

// Helper methods for the comparison model

// countSelectedOptimizers returns the number of selected optimizers
func (m ComparisonModel) countSelectedOptimizers() int {
	count := 0
	for _, selected := range m.selectedOptimizers {
		if selected {
			count++
		}
	}
	return count
}

// getOptimizerIcon returns the icon for a given optimizer
func (m ComparisonModel) getOptimizerIcon(optimizer string) string {
	icons := map[string]string{
		"bootstrap": styles.IconBootstrap,
		"mipro":     styles.IconMIPRO,
		"simba":     styles.IconSIMBA,
		"gepa":      styles.IconGEPA,
		"copro":     styles.IconCOPRO,
	}

	if icon, ok := icons[optimizer]; ok {
		return icon
	}
	return "🔧"
}

// getOptimizerDisplayName returns a human-readable name for the optimizer
func (m ComparisonModel) getOptimizerDisplayName(optimizer string) string {
	names := map[string]string{
		"bootstrap": "Bootstrap",
		"mipro":     "MIPRO",
		"simba":     "SIMBA",
		"gepa":      "GEPA",
		"copro":     "COPRO",
	}

	if name, ok := names[optimizer]; ok {
		return name
	}
	return strings.ToUpper(optimizer)
}

// renderProgressBar creates a visual progress bar
func (m ComparisonModel) renderProgressBar(progress float64, width int) string {
	filled := int(float64(width) * progress)
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}

	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	percentage := fmt.Sprintf(" %.0f%%", progress*100)

	barStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.DSPyGreen))
	emptyStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(styles.MediumGray))

	if filled > 0 {
		return barStyle.Render(bar[:filled]) + emptyStyle.Render(bar[filled:]) + percentage
	}
	return emptyStyle.Render(bar) + percentage
}

// formatDuration formats a duration for display
func (m ComparisonModel) formatDuration(d time.Duration) string {
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

// updateLeaderboard updates the leaderboard with current results
func (m *ComparisonModel) updateLeaderboard() {
	var entries []LeaderboardEntry

	for optimizer, result := range m.results {
		if result != nil && result.Success {
			entries = append(entries, LeaderboardEntry{
				OptimizerName: optimizer,
				Icon:         m.getOptimizerIcon(optimizer),
				DisplayName:  m.getOptimizerDisplayName(optimizer),
				Accuracy:     result.FinalAccuracy,
				Improvement:  result.ImprovementPct,
				Duration:     result.Duration,
				Status:       "completed",
				TrendIcon:    m.getTrendIcon(result.ImprovementPct),
			})
		}
	}

	// Sort by accuracy (highest first)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Accuracy > entries[j].Accuracy
	})

	// Assign ranks
	for i := range entries {
		entries[i].Rank = i + 1
	}

	m.leaderboard = entries
}

// getTrendIcon returns an icon representing the improvement trend
func (m ComparisonModel) getTrendIcon(improvement float64) string {
	if improvement > 20 {
		return "🚀" // Excellent
	} else if improvement > 5 {
		return "📈" // Good
	} else if improvement > 0 {
		return "↗️" // Slight improvement
	} else if improvement == 0 {
		return "➡️" // No change
	} else {
		return "↘️" // Decline
	}
}

// renderDetailedStats renders detailed comparison statistics
func (m ComparisonModel) renderDetailedStats() string {
	var sections []string

	// Performance statistics
	sections = append(sections, m.renderPerformanceStats())
	sections = append(sections, "")

	// Speed analysis
	sections = append(sections, m.renderSpeedAnalysis())
	sections = append(sections, "")

	// Recommendations
	sections = append(sections, m.renderRecommendations())

	return strings.Join(sections, "\n")
}

// renderPerformanceStats shows performance comparison
func (m ComparisonModel) renderPerformanceStats() string {
	if len(m.leaderboard) == 0 {
		return "No performance data available"
	}

	var stats []string
	stats = append(stats, "📊 Performance Analysis:")

	best := m.leaderboard[0]
	worst := m.leaderboard[len(m.leaderboard)-1]

	stats = append(stats, fmt.Sprintf("  🏆 Best Performer: %s (%.1f%% accuracy)",
		best.DisplayName, best.Accuracy*100))
	stats = append(stats, fmt.Sprintf("  📉 Lowest Score: %s (%.1f%% accuracy)",
		worst.DisplayName, worst.Accuracy*100))

	// Calculate accuracy spread
	spread := (best.Accuracy - worst.Accuracy) * 100
	stats = append(stats, fmt.Sprintf("  📏 Accuracy Range: %.1f%% spread", spread))

	// Average improvement
	totalImprovement := 0.0
	for _, entry := range m.leaderboard {
		totalImprovement += entry.Improvement
	}
	avgImprovement := totalImprovement / float64(len(m.leaderboard))
	stats = append(stats, fmt.Sprintf("  📈 Average Improvement: %.1f%%", avgImprovement))

	return strings.Join(stats, "\n")
}

// renderSpeedAnalysis shows speed comparison
func (m ComparisonModel) renderSpeedAnalysis() string {
	if len(m.leaderboard) == 0 {
		return "No speed data available"
	}

	var stats []string
	stats = append(stats, "⚡ Speed Analysis:")

	// Find fastest and slowest
	fastest := m.leaderboard[0]
	slowest := m.leaderboard[0]

	for _, entry := range m.leaderboard {
		if entry.Duration < fastest.Duration {
			fastest = entry
		}
		if entry.Duration > slowest.Duration {
			slowest = entry
		}
	}

	stats = append(stats, fmt.Sprintf("  🏃 Fastest: %s (%s)",
		fastest.DisplayName, m.formatDuration(fastest.Duration)))
	stats = append(stats, fmt.Sprintf("  🐌 Slowest: %s (%s)",
		slowest.DisplayName, m.formatDuration(slowest.Duration)))

	// Speed ratio
	ratio := float64(slowest.Duration) / float64(fastest.Duration)
	stats = append(stats, fmt.Sprintf("  📊 Speed Ratio: %.1fx difference", ratio))

	return strings.Join(stats, "\n")
}

// renderRecommendations provides intelligent recommendations
func (m ComparisonModel) renderRecommendations() string {
	if len(m.leaderboard) == 0 {
		return "No recommendations available"
	}

	var recommendations []string
	recommendations = append(recommendations, "💡 Smart Recommendations:")

	best := m.leaderboard[0]

	// Performance recommendation
	recommendations = append(recommendations, fmt.Sprintf("  🎯 For best accuracy: Use %s", best.DisplayName))

	// Speed recommendation
	fastest := m.leaderboard[0]
	for _, entry := range m.leaderboard {
		if entry.Duration < fastest.Duration {
			fastest = entry
		}
	}
	recommendations = append(recommendations, fmt.Sprintf("  ⚡ For fastest results: Use %s", fastest.DisplayName))

	// Balanced recommendation
	var balanced LeaderboardEntry
	bestScore := 0.0
	for _, entry := range m.leaderboard {
		// Calculate balanced score (accuracy + speed bonus)
		speedBonus := 1.0 / entry.Duration.Seconds() * 100
		score := entry.Accuracy*100 + speedBonus
		if score > bestScore {
			bestScore = score
			balanced = entry
		}
	}
	recommendations = append(recommendations, fmt.Sprintf("  ⚖️ For balanced performance: Use %s", balanced.DisplayName))

	// Beginner recommendation
	recommendations = append(recommendations, "  🔰 For beginners: Bootstrap (simple and reliable)")

	return strings.Join(recommendations, "\n")
}

// renderFooter renders the comparison footer with controls
func (m ComparisonModel) renderFooter() string {
	var controls []string

	if !m.isRunning && m.completed == 0 {
		// Setup phase
		controls = append(controls, "[↑↓] Select Dataset", "[Enter] Start Battle")
		controls = append(controls, "[s] Sort Options", "[b] Back")
	} else if m.isRunning {
		// Running phase
		controls = append(controls, "[⏳] Battle in Progress...")
		controls = append(controls, "[Ctrl+C] Cancel", "[q] Quit")
	} else {
		// Results phase
		controls = append(controls, "[d] Toggle Details", "[s] Sort")
		controls = append(controls, "[r] Run Again", "[b] Back")
	}

	controls = append(controls, "[?] Help")

	return styles.FooterStyle.Render(strings.Join(controls, " • "))
}

// extractDatasetKey extracts the dataset key from display string
func extractDatasetKey(datasetDisplay string) string {
	// Extract dataset key from display string (e.g., "gsm8k (Grade School Math 8K)" -> "gsm8k")
	if idx := strings.Index(datasetDisplay, " ("); idx != -1 {
		return datasetDisplay[:idx]
	}
	return datasetDisplay
}

// getSpinnerFrame returns the current spinner frame
func getSpinnerFrame(frame int) string {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	return frames[frame%len(frames)]
}
