package commands

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/display"
)

func NewListCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all available optimizers",
		Long: `Display a comprehensive list of all available DSPy-Go optimizers with their
key characteristics, complexity levels, and use cases.

This command helps you discover what optimizers are available and get a quick
overview of their strengths and computational requirements.`,
		Example: `  # List all optimizers
  dspy-cli list

  # Pipe to grep for filtering
  dspy-cli list | grep -i "low complexity"`,
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Print(display.FormatOptimizerList())
		},
	}
}
