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
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
