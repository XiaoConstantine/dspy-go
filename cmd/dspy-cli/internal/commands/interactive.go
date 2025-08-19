package commands

import (
	"fmt"
	"os"

	"github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/interactive"
)

// NewInteractiveCommand creates the interactive command
func NewInteractiveCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "interactive",
		Short: "Enter interactive mode for guided optimization exploration",
		Long: `Launch the interactive DSPy-CLI interface for a guided experience
in exploring and using optimizers. This mode provides:

• AI-guided optimizer selection based on your task
• Real-time visualization of optimization progress
• Side-by-side optimizer comparisons
• Built-in tutorials and help

Simply run 'dspy-cli interactive' or 'dspy-cli' without any arguments.`,
		Aliases: []string{"i"},
		RunE:    runInteractive,
	}
}

func runInteractive(cmd *cobra.Command, args []string) error {
	// Check if terminal supports TUI
	if !isTerminal() {
		return fmt.Errorf("interactive mode requires a terminal")
	}

	// Create the interactive app
	app := interactive.NewApp()

	// Create and run the bubbletea program
	p := tea.NewProgram(
		app,
		tea.WithAltScreen(),       // Use alternate screen buffer
		tea.WithMouseCellMotion(), // Enable mouse support
	)

	// Run the program
	model, err := p.Run()
	if err != nil {
		return fmt.Errorf("error running interactive mode: %w", err)
	}

	// Check if user selected something
	if finalModel, ok := model.(interactive.AppModel); ok {
		// Could potentially return selected optimizer or action here
		_ = finalModel
	}

	return nil
}

// isTerminal checks if we're running in a terminal
func isTerminal() bool {
	fileInfo, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fileInfo.Mode() & os.ModeCharDevice) != 0
}
