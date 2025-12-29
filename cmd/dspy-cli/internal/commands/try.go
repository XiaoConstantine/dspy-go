package commands

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/display"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/optimizers"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/runner"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/samples"
)

func NewTryCommand() *cobra.Command {
	var dataset string
	var apiKey string
	var provider string
	var model string
	var baseURL string
	var maxExamples int
	var verbose bool

	cmd := &cobra.Command{
		Use:   "try <optimizer>",
		Short: "Try an optimizer with sample data (eliminates all boilerplate)",
		Long: `Run an optimizer with built-in sample datasets without writing any code.
This command eliminates all the boilerplate setup (logging, LLM configuration,
dataset loading, signature creation) and lets you quickly experiment with different
optimizers to see how they perform.

Perfect for:
- Quick experimentation with optimizers
- Understanding optimizer behavior without coding
- Comparing performance across different approaches
- Learning how optimizers work with real data`,
		Example: `  # Try MIPRO with math problems
  dspy-cli try mipro --dataset gsm8k

  # Try Bootstrap with simple Q&A (fastest)
  dspy-cli try bootstrap --dataset qa --max-examples 3

  # Try SIMBA with multi-hop reasoning
  dspy-cli try simba --dataset hotpotqa --verbose

  # Use custom API key
  dspy-cli try gepa --dataset gsm8k --api-key your-key-here
  
  # Use local LLM (e.g. LM Studio)
  dspy-cli try simba --dataset gsm8k --provider local --base-url http://localhost:1234/v1 --model local-model`,
		Args: cobra.ExactArgs(1),
		ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return optimizers.ListAll(), cobra.ShellCompDirectiveNoFileComp
		},
		Run: func(cmd *cobra.Command, args []string) {
			optimizerName := strings.ToLower(args[0])

			// Validate optimizer
			if _, err := optimizers.GetOptimizer(optimizerName); err != nil {
				fmt.Printf("%sError:%s Unknown optimizer '%s'\n\n", display.ColorRed, display.ColorReset, optimizerName)
				fmt.Println("Available optimizers:")
				for _, name := range optimizers.ListAll() {
					fmt.Printf("  - %s\n", name)
				}
				return
			}

			// Validate dataset
			if _, exists := samples.GetSampleDataset(dataset); !exists {
				fmt.Printf("%sError:%s Unknown dataset '%s'\n\n", display.ColorRed, display.ColorReset, dataset)
				fmt.Println("Available datasets:")
				for _, ds := range samples.ListAvailableDatasets() {
					fmt.Printf("  - %s\n", ds)
				}
				return
			}

			// Show what we're about to do
			optimizerInfo, _ := optimizers.GetOptimizer(optimizerName)
			sampleDataset, _ := samples.GetSampleDataset(dataset)

			fmt.Printf("%s%s🚀 Running %s Optimizer%s\n", display.ColorBold, display.ColorBlue, optimizerInfo.Name, display.ColorReset)
			fmt.Println(strings.Repeat("=", 50))
			fmt.Printf("%sDataset:%s %s (%s)\n", display.ColorCyan, display.ColorReset, sampleDataset.Name, sampleDataset.Description)
			fmt.Printf("%sComplexity:%s %s | %sCost:%s %s\n",
				display.ColorCyan, display.ColorReset, optimizerInfo.Complexity,
				display.ColorCyan, display.ColorReset, optimizerInfo.ComputeCost)

			if maxExamples > 0 {
				fmt.Printf("%sExamples:%s Limited to %d examples\n", display.ColorCyan, display.ColorReset, maxExamples)
			} else {
				fmt.Printf("%sExamples:%s Using all %d examples\n", display.ColorCyan, display.ColorReset, len(sampleDataset.Examples))
			}
			fmt.Println()

			// Run the optimizer
			config := runner.OptimizerConfig{
				OptimizerName: optimizerName,
				DatasetName:   dataset,
				APIKey:        apiKey,
				Provider:      provider,
				Model:         model,
				BaseURL:       baseURL,
				MaxExamples:   maxExamples,
				Verbose:       verbose,
			}

			fmt.Printf("%sStarting optimization...%s\n", display.ColorYellow, display.ColorReset)
			result, err := runner.RunOptimizer(config)

			if err != nil {
				fmt.Printf("\n%s❌ Optimization Failed%s\n", display.ColorRed, display.ColorReset)
				fmt.Printf("%sError:%s %s\n", display.ColorRed, display.ColorReset, err.Error())
				if result != nil && result.ErrorMessage != "" {
					fmt.Printf("%sDetails:%s %s\n", display.ColorRed, display.ColorReset, result.ErrorMessage)
				}
				return
			}

			// Display results
			displayResults(result)
		},
	}

	cmd.Flags().StringVar(&dataset, "dataset", "qa", "Dataset to use (qa, gsm8k, hotpotqa)")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "API key (or set GEMINI_API_KEY env var)")
	cmd.Flags().StringVar(&provider, "provider", "google", "LLM provider (google, openai, local)")
	cmd.Flags().StringVar(&model, "model", "gemini-2.0-flash", "Model ID to use")
	cmd.Flags().StringVar(&baseURL, "base-url", "", "Base URL for the LLM provider (optional)")
	cmd.Flags().IntVar(&maxExamples, "max-examples", 0, "Limit number of examples (0 = use all)")
	cmd.Flags().BoolVar(&verbose, "verbose", false, "Enable verbose logging")

	return cmd
}

func displayResults(result *runner.RunResult) {
	fmt.Printf("\n%s%s✅ Optimization Complete%s\n", display.ColorBold, display.ColorGreen, display.ColorReset)
	fmt.Println(strings.Repeat("=", 40))

	fmt.Printf("%sOptimizer:%s %s\n", display.ColorCyan, display.ColorReset, result.OptimizerName)
	fmt.Printf("%sDataset:%s %s\n", display.ColorCyan, display.ColorReset, result.DatasetName)
	fmt.Printf("%sDuration:%s %v\n", display.ColorCyan, display.ColorReset, result.Duration.Round(100))
	fmt.Printf("%sExamples Used:%s %d\n", display.ColorCyan, display.ColorReset, result.ExamplesUsed)

	fmt.Println()
	fmt.Printf("%s📊 Performance Results%s\n", display.ColorBold, display.ColorBlue)
	fmt.Println(strings.Repeat("-", 25))

	fmt.Printf("%sInitial Accuracy:%s %.1f%%\n", display.ColorYellow, display.ColorReset, result.InitialAccuracy*100)
	fmt.Printf("%sFinal Accuracy:%s %.1f%%\n", display.ColorGreen, display.ColorReset, result.FinalAccuracy*100)

	// Show improvement with appropriate color
	improvementColor := display.ColorGreen
	improvementSymbol := "📈"
	if result.ImprovementPct < 0 {
		improvementColor = display.ColorRed
		improvementSymbol = "📉"
	} else if result.ImprovementPct == 0 {
		improvementColor = display.ColorYellow
		improvementSymbol = "➡️"
	}

	fmt.Printf("%sImprovement:%s %s%+.1f%% %s%s\n",
		display.ColorCyan, display.ColorReset,
		improvementColor, result.ImprovementPct, improvementSymbol, display.ColorReset)

	fmt.Println()

	// Provide contextual insights
	if result.ImprovementPct > 10 {
		fmt.Printf("%s💡 Great improvement! This optimizer works well for this type of task.%s\n",
			display.ColorGreen, display.ColorReset)
	} else if result.ImprovementPct > 0 {
		fmt.Printf("%s💡 Modest improvement. Try with more examples or a different optimizer.%s\n",
			display.ColorYellow, display.ColorReset)
	} else {
		fmt.Printf("%s💡 No improvement. This task might need a different approach or more data.%s\n",
			display.ColorYellow, display.ColorReset)
	}

	fmt.Printf("\n%sNext steps:%s\n", display.ColorPurple, display.ColorReset)
	fmt.Printf("• Try other optimizers: %sdspy-cli recommend%s\n", display.ColorCyan, display.ColorReset)
	fmt.Printf("• Get detailed info: %sdspy-cli describe %s%s\n", display.ColorCyan, result.OptimizerName, display.ColorReset)
	fmt.Printf("• Compare optimizers: %sdspy-cli try <other-optimizer> --dataset %s%s\n",
		display.ColorCyan, result.DatasetName, display.ColorReset)
}
