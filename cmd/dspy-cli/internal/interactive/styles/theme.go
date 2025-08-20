package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Ghost in the Shell + gh-dash Inspired Cyberpunk Theme
const (
	// Primary Cyberpunk Palette
	CyberCyan     = "#00F5FF"  // Neon cyan - primary accent
	CyberPurple   = "#6B46C1"  // Deep purple - secondary
	CyberViolet   = "#A855F7"  // Bright violet - highlights
	CyberPink     = "#FF006E"  // Magenta - warnings/alerts
	CyberGreen    = "#00FF88"  // Matrix green - success
	CyberOrange   = "#FF8500"  // Neon orange - warnings

	// Background & Structure
	TerminalBlack = "#0F0F0F"  // Deep black background
	MatrixDark    = "#1A1B26"  // Tokyo Night inspired dark
	BorderGray    = "#2D3748"  // Subtle borders
	TextPrimary   = "#E2E8F0"  // High contrast text
	TextSecondary = "#94A3B8"  // Muted text
	TextTertiary  = "#64748B"  // Subtle text

	// Legacy compatibility (mapped to new colors)
	DSPyBlue   = CyberCyan
	DSPyGreen  = CyberGreen
	DSPyPurple = CyberViolet
	DarkBlue   = MatrixDark
	White      = TextPrimary

	// Status Colors (GitS inspired)
	StatusSuccess = "#00FF88"  // Bright green
	StatusError   = "#FF006E"  // Bright magenta
	StatusWarning = "#FF8500"  // Neon orange
	StatusInfo    = "#00F5FF"  // Cyber cyan
	StatusPending = "#A855F7"  // Violet
)

// Cyberpunk Typography Styles
var (
	// Primary Hierarchy (Ghost in the Shell inspired)
	HeroStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(CyberCyan)).MarginBottom(1).Underline(true)
	TitleStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(CyberViolet)).Background(lipgloss.Color(MatrixDark))
	HeadingStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(TextPrimary))
	SubheadStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(CyberPurple))
	BodyStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(TextPrimary))
	CaptionStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(TextSecondary)).Italic(true)

	// Enhanced Semantic Styles
	SuccessStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(StatusSuccess)).Bold(true)
	WarningStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(StatusWarning)).Bold(true)
	ErrorStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(StatusError)).Bold(true)
	InfoStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color(StatusInfo))
	MutedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(TextTertiary))
	HighlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberViolet)).Bold(true)

	// Cyberpunk Interactive Elements
	SelectedStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberCyan)).Bold(true).Background(lipgloss.Color(CyberPurple))
	UnselectedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(TextSecondary))
	FocusedStyle    = lipgloss.NewStyle().BorderStyle(lipgloss.ThickBorder()).BorderForeground(lipgloss.Color(CyberCyan))
	BlurredStyle    = lipgloss.NewStyle().BorderStyle(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color(BorderGray))

	// New Cyberpunk Specific Styles
	GlitchStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberPink)).Bold(true).Strikethrough(true)
	MatrixStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberGreen)).Background(lipgloss.Color(TerminalBlack))
	NeonStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberCyan)).Bold(true).Underline(true)
	TerminalStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color(CyberGreen)).Background(lipgloss.Color(TerminalBlack)).Padding(0, 1)

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
