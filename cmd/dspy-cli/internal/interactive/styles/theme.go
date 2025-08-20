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
	DSPyBlue       = CyberCyan
	DSPyGreen      = CyberGreen
	DSPyOrange     = CyberOrange
	DSPyRed        = StatusError
	DSPyPurple     = CyberViolet
	DarkBlue       = MatrixDark
	MediumGray     = TextSecondary
	LightGray      = TextPrimary
	White          = TextPrimary

	// Additional legacy mappings
	OptimizedGreen = CyberGreen
	ProcessingBlue = CyberCyan
	WarningAmber   = CyberOrange
	ErrorCrimson   = StatusError

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

	// Cyberpunk Layout Components
	BoxStyle = lipgloss.NewStyle().
			Border(lipgloss.ThickBorder()).
			BorderForeground(lipgloss.Color(CyberCyan)).
			Padding(1, 2).
			MarginTop(1).
			MarginBottom(1).
			Background(lipgloss.Color(MatrixDark))

	HeaderStyle = lipgloss.NewStyle().
			Bold(true).
			Background(lipgloss.Color(CyberPurple)).
			Foreground(lipgloss.Color(TextPrimary)).
			Padding(0, 2).
			Width(80).
			Underline(true)

	FooterStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(TextSecondary)).
			Background(lipgloss.Color(TerminalBlack)).
			MarginTop(1).
			Padding(0, 1)

	// Enhanced Progress Indicators (Matrix-style)
	ProgressBarStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(CyberGreen)).
				Background(lipgloss.Color(TerminalBlack))

	ProgressCompleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(CyberGreen)).
				Bold(true)

	ProgressIncompleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color(BorderGray))

	// New Cyberpunk Components
	PanelStyle = lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForeground(lipgloss.Color(CyberViolet)).
			Background(lipgloss.Color(MatrixDark)).
			Padding(1)

	AlertStyle = lipgloss.NewStyle().
			Border(lipgloss.ThickBorder()).
			BorderForeground(lipgloss.Color(CyberPink)).
			Background(lipgloss.Color(TerminalBlack)).
			Padding(1, 2).
			Bold(true)
)

// Cyberpunk Icons and Symbols (Ghost in the Shell inspired)
const (
	// Optimizer Icons (Cyberpunk themed)
	IconBootstrap = "⚡"   // Lightning for speed
	IconMIPRO     = "🧠"   // Brain for intelligence
	IconSIMBA     = "🦾"   // Cybernetic arm for strength
	IconGEPA      = "🧬"   // DNA for evolution
	IconCOPRO     = "🤖"   // Robot for collaboration

	// Cyberpunk Status Icons
	IconSuccess  = "▓"     // Block for success
	IconProgress = "▒"     // Loading block
	IconWarning  = "⚠"     // Warning triangle
	IconError    = "✗"     // X mark
	IconInfo     = "◉"     // Circle dot
	IconPending  = "○"     // Empty circle

	// Action Icons
	IconPlay     = "▶️"
	IconPause    = "⏸️"
	IconStop     = "⏹️"
	IconSettings = "⚙️"
	IconHelp     = "❓"
	IconHome     = "🏠"
	IconBack     = "⬅️"
	IconNext     = "➡️"

	// Cyberpunk UI Elements
	IconSelected   = "►"     // Solid arrow
	IconUnselected = "▷"     // Hollow arrow
	IconCheck      = "◆"     // Diamond
	IconCross      = "◇"     // Hollow diamond
	IconDot        = "▪"     // Small block
	IconArrowRight = "▶"     // Right arrow
	IconArrowDown  = "▼"     // Down arrow
	IconWizard     = "◈"     // Special wizard icon
	IconMatrix     = "▓"     // Matrix block

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

// Create a cyberpunk-styled box with title
func RenderTitledBox(title, content string, width int) string {
	titleBar := TitleStyle.Copy().
		Background(lipgloss.Color(CyberPurple)).
		Foreground(lipgloss.Color(TextPrimary)).
		Width(width - 4). // Account for border and padding
		Padding(0, 1).
		Render(title)

	contentBox := lipgloss.NewStyle().
		Width(width - 4).
		Padding(1).
		Foreground(lipgloss.Color(TextPrimary)).
		Background(lipgloss.Color(MatrixDark)).
		Render(content)

	combined := lipgloss.JoinVertical(lipgloss.Left, titleBar, contentBox)

	return BoxStyle.Copy().Width(width).Render(combined)
}

// Render a cyberpunk-styled panel (new function)
func RenderCyberPanel(title, content string, width int) string {
	titleBar := NeonStyle.Copy().
		Width(width - 4).
		Padding(0, 1).
		Render(IconMatrix + " " + title)

	contentBox := TerminalStyle.Copy().
		Width(width - 4).
		Padding(1).
		Render(content)

	combined := lipgloss.JoinVertical(lipgloss.Left, titleBar, contentBox)

	return PanelStyle.Copy().Width(width).Render(combined)
}
