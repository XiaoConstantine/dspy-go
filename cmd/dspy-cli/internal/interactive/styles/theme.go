package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Brand Colors
const (
	DSPyBlue   = "#4A90E2"
	DSPyGreen  = "#7ED321"
	DSPyOrange = "#F5A623"
	DSPyRed    = "#D0021B"
	DSPyPurple = "#9013FE"
	DarkBlue   = "#2C3E50"
	MediumGray = "#9B9B9B"
	LightGray  = "#F8F9FA"
	White      = "#FFFFFF"

	// Semantic Colors
	OptimizedGreen = "#00D09C"
	ProcessingBlue = "#007AFF"
	WarningAmber   = "#FF9500"
	ErrorCrimson   = "#FF3B30"
)

// Typography Styles
var (
	// Hierarchy
	HeroStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(DSPyBlue)).MarginBottom(1)
	TitleStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(DarkBlue))
	HeadingStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(DarkBlue))
	SubheadStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(MediumGray))
	BodyStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(DarkBlue))
	CaptionStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(MediumGray)).Italic(true)

	// Semantic
	SuccessStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyGreen)).Bold(true)
	WarningStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyOrange)).Bold(true)
	ErrorStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyRed)).Bold(true)
	InfoStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyBlue))
	MutedStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(MediumGray))
	HighlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyPurple)).Bold(true)

	// Interactive Elements
	SelectedStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(DSPyGreen)).Bold(true)
	UnselectedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(MediumGray))
	FocusedStyle    = lipgloss.NewStyle().BorderStyle(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(DSPyBlue))
	BlurredStyle    = lipgloss.NewStyle().BorderStyle(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(MediumGray))

	// Layout Components
	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color(DSPyBlue)).
			Padding(1, 2).
			MarginTop(1).
			MarginBottom(1)

	HeaderStyle = lipgloss.NewStyle().
			Bold(true).
			Background(lipgloss.Color(DSPyBlue)).
			Foreground(lipgloss.Color(White)).
			Padding(0, 2).
			Width(80)

	FooterStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(MediumGray)).
			MarginTop(1)

	// Progress Indicators
	ProgressBarStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(DSPyGreen)).
				Background(lipgloss.Color(MediumGray))

	ProgressCompleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(DSPyGreen)).
				Bold(true)

	ProgressIncompleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(MediumGray))
)

// Optimizer Icons and Symbols
const (
	// Optimizer Icons
	IconBootstrap = "🚀"
	IconMIPRO     = "🧠"
	IconSIMBA     = "⚡"
	IconGEPA      = "🔬"
	IconCOPRO     = "🤝"

	// Status Icons
	IconSuccess  = "✅"
	IconProgress = "🔄"
	IconWarning  = "⚠️"
	IconError    = "❌"
	IconInfo     = "ℹ️"
	IconPending  = "⏳"

	// Action Icons
	IconPlay     = "▶️"
	IconPause    = "⏸️"
	IconStop     = "⏹️"
	IconSettings = "⚙️"
	IconHelp     = "❓"
	IconHome     = "🏠"
	IconBack     = "⬅️"
	IconNext     = "➡️"

	// UI Elements
	IconSelected   = "❯"
	IconUnselected = " "
	IconCheck      = "✓"
	IconCross      = "✗"
	IconDot        = "•"
	IconArrowRight = "→"
	IconArrowDown  = "↓"

	// Progress Characters
	ProgressFull  = "█"
	ProgressEmpty = "░"
	ProgressHalf  = "▒"
)

// Helper functions for common styling patterns
func RenderProgressBar(current, total int, width int) string {
	if total == 0 {
		return strings.Repeat(ProgressEmpty, width)
	}

	filled := int(float64(current) / float64(total) * float64(width))
	if filled > width {
		filled = width
	}

	return ProgressBarStyle.Render(
		strings.Repeat(ProgressFull, filled) +
			strings.Repeat(ProgressEmpty, width-filled),
	)
}

func RenderPercentage(current, total int) string {
	if total == 0 {
		return "0%"
	}
	percentage := int(float64(current) / float64(total) * 100)
	return fmt.Sprintf("%d%%", percentage)
}

// Render a status badge with appropriate styling
func RenderStatusBadge(status string) string {
	switch status {
	case "success", "completed", "done":
		return SuccessStyle.Render(IconSuccess + " " + status)
	case "error", "failed":
		return ErrorStyle.Render(IconError + " " + status)
	case "warning", "caution":
		return WarningStyle.Render(IconWarning + " " + status)
	case "running", "progress", "processing":
		return InfoStyle.Render(IconProgress + " " + status)
	default:
		return MutedStyle.Render(IconPending + " " + status)
	}
}

// Create a consistent box with title
func RenderTitledBox(title, content string, width int) string {
	titleBar := TitleStyle.Copy().
		Background(lipgloss.Color(DSPyBlue)).
		Foreground(lipgloss.Color(White)).
		Width(width - 4). // Account for border and padding
		Padding(0, 1).
		Render(title)

	contentBox := lipgloss.NewStyle().
		Width(width - 4).
		Padding(1).
		Render(content)

	combined := lipgloss.JoinVertical(lipgloss.Left, titleBar, contentBox)

	return BoxStyle.Copy().Width(width).Render(combined)
}
