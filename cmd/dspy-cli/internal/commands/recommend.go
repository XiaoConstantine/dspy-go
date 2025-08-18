package commands

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/display"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
)

func NewRecommendCommand() *cobra.Command {
	var useCase string
	var interactive bool

	cmd := &cobra.Command{
		Use:   "recommend",
		Short: "Get optimizer recommendations based on your use case",
		Long: `Get personalized optimizer recommendations based on your specific requirements.
You can either specify a use case directly or use interactive mode to answer
a few questions and get tailored recommendations.

The recommendation engine considers factors like:
- Task complexity
- Computational budget
- Performance requirements
- Experience level`,
		Example: `  # Get recommendations for simple tasks
  dspy-cli recommend --use-case simple

  # Interactive recommendation wizard
  dspy-cli recommend --interactive

  # Quick recommendation for research work
  dspy-cli recommend --use-case research`,
		Run: func(cmd *cobra.Command, args []string) {
			if interactive {
				runInteractiveRecommendation()
				return
			}

			if useCase == "" {
				fmt.Println("Please specify a use case with --use-case or use --interactive mode")
				fmt.Println("\nAvailable use cases: simple, balanced, advanced, multi-module")
				return
			}

			recommendations := optimizers.GetRecommendation(useCase)
			if len(recommendations) == 0 {
				fmt.Printf("No specific recommendations for '%s'. Try: simple, balanced, advanced, or multi-module\n", useCase)
				return
			}

			fmt.Print(display.FormatRecommendations(recommendations))
		},
	}

	cmd.Flags().StringVar(&useCase, "use-case", "", "Specify your use case (simple, balanced, advanced, multi-module)")
	cmd.Flags().BoolVar(&interactive, "interactive", false, "Use interactive recommendation wizard")

	return cmd
}

func runInteractiveRecommendation() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Printf("%s🧙 DSPy-Go Optimizer Recommendation Wizard%s\n", display.ColorBold+display.ColorBlue, display.ColorReset)
	fmt.Println(strings.Repeat("=", 45))
	fmt.Println()

	// Question 1: Experience level
	fmt.Printf("%sWhat's your experience with DSPy optimizers?%s\n", display.ColorBold, display.ColorReset)
	fmt.Println("1. New to DSPy/prompt optimization")
	fmt.Println("2. Some experience with optimization")
	fmt.Println("3. Advanced user/researcher")
	fmt.Print("\nEnter choice (1-3): ")

	experience, _ := reader.ReadString('\n')
	experience = strings.TrimSpace(experience)

	// Question 2: Task complexity
	fmt.Printf("\n%sHow complex is your task?%s\n", display.ColorBold, display.ColorReset)
	fmt.Println("1. Simple (Q&A, basic classification)")
	fmt.Println("2. Moderate (reasoning, multi-step tasks)")
	fmt.Println("3. Complex (research-level, multi-objective)")
	fmt.Print("\nEnter choice (1-3): ")

	complexity, _ := reader.ReadString('\n')
	complexity = strings.TrimSpace(complexity)

	// Question 3: Computational budget
	fmt.Printf("\n%sWhat's your computational budget?%s\n", display.ColorBold, display.ColorReset)
	fmt.Println("1. Limited (prefer fast, efficient optimization)")
	fmt.Println("2. Moderate (balanced cost/performance)")
	fmt.Println("3. High (performance over cost)")
	fmt.Print("\nEnter choice (1-3): ")

	budget, _ := reader.ReadString('\n')
	budget = strings.TrimSpace(budget)

	fmt.Println()

	// Generate recommendations based on answers
	var recommendedUseCase string

	if experience == "1" || (complexity == "1" && budget == "1") {
		recommendedUseCase = "simple"
	} else if experience == "3" || (complexity == "3" && budget == "3") {
		recommendedUseCase = "advanced"
	} else {
		recommendedUseCase = "balanced"
	}

	recommendations := optimizers.GetRecommendation(recommendedUseCase)
	fmt.Print(display.FormatRecommendations(recommendations))

	fmt.Printf("%sTip:%s Try the recommended optimizers with 'dspy-cli try <optimizer>'\n",
		display.ColorPurple, display.ColorReset)
}
