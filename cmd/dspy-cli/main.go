package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/commands"
)

var rootCmd = &cobra.Command{
	Use:   "dspy-cli",
	Short: "DSPy-Go CLI for exploring and running optimizers",
	Long: `A command-line interface for the DSPy-Go framework that makes it easy to
explore, compare, and run different optimizers without writing boilerplate code.

The CLI provides:
- Quick optimizer exploration and comparison
- Built-in sample datasets for experimentation
- Guided optimization workflows
- Results visualization and export`,
	Version: "0.1.0",
}

func init() {
	rootCmd.AddCommand(commands.NewListCommand())
	rootCmd.AddCommand(commands.NewDescribeCommand())
	rootCmd.AddCommand(commands.NewRecommendCommand())
	rootCmd.AddCommand(commands.NewTryCommand())
	rootCmd.AddCommand(commands.NewInteractiveCommand())
	rootCmd.AddCommand(commands.NewAnalyzeCommand())
	rootCmd.AddCommand(commands.NewViewerCommand())

	// Make interactive mode the default when no command is specified
	rootCmd.RunE = func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			// Run interactive mode by default
			interactiveCmd := commands.NewInteractiveCommand()
			return interactiveCmd.RunE(cmd, args)
		}
		return cmd.Help()
	}
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
