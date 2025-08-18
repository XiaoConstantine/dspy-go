package commands

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/display"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
)

func NewDescribeCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "describe <optimizer>",
		Short: "Get detailed information about a specific optimizer",
		Long: `Display comprehensive details about a specific optimizer including:
- Algorithm description and approach
- Complexity and computational cost analysis
- Best use cases and scenarios
- Example applications
- Quick start command suggestions

This helps you understand whether an optimizer is suitable for your specific needs.`,
		Example: `  # Describe the MIPRO optimizer
  dspy-cli describe mipro

  # Describe the Bootstrap optimizer
  dspy-cli describe bootstrap

  # Describe GEPA (case insensitive)
  dspy-cli describe GEPA`,
		Args: cobra.ExactArgs(1),
		ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return optimizers.ListAll(), cobra.ShellCompDirectiveNoFileComp
		},
		Run: func(cmd *cobra.Command, args []string) {
			optimizerName := strings.ToLower(args[0])

			info, err := optimizers.GetOptimizer(optimizerName)
			if err != nil {
				fmt.Printf("Error: %s\n\n", err)
				fmt.Println("Available optimizers:")
				for _, name := range optimizers.ListAll() {
					fmt.Printf("  - %s\n", name)
				}
				return
			}

			fmt.Print(display.FormatOptimizerDetails(info))
		},
	}
}
