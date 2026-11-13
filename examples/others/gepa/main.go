package main

import (
	"context"
	"flag"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func main() {
	apiKey := flag.String("api-key", "", "API Key for the Gemini LLM provider")
	dataset := flag.String("dataset", "gsm8k", "Dataset to use (gsm8k or hotpotqa)")
	generations := flag.Int("generations", 8, "Maximum number of generations")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for GEPA evolution")
	flag.Parse()

	if *apiKey == "" {
		fmt.Println("Please provide a Gemini API key using -api-key flag")
		fmt.Println("Usage: go run main.go -api-key YOUR_GEMINI_API_KEY")
		return
	}

	// Run the GEPA example
	RunGEPAExample(*apiKey, *dataset, *generations, *verbose)
}

func RunGEPAExample(apiKey, datasetName string, generations int, verbose bool) {
	// Setup enhanced logging for GEPA evolution tracking
	logLevel := logging.INFO
	if verbose {
		logLevel = logging.DEBUG
	}

	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logLevel,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	// Create context with execution state
	ctx := core.WithExecutionState(context.Background())

	logger.Info(ctx, "🧬 Starting GEPA (Generative Evolutionary Prompt Adaptation) Example")
	logger.Info(ctx, "Dataset: %s, Generation cap: %d", datasetName, generations)

	// Configure Gemini LLM (as specified in the request)
	llms.EnsureFactory()
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to setup LLM: %v", err)
	}
	logger.Info(ctx, "✅ Configured Gemini Flash model for GEPA optimization")

	// Load and prepare training dataset
	examples, err := loadDataset(datasetName)
	if err != nil {
		logger.Fatalf(ctx, "Failed to load dataset: %v", err)
	}

	// Use a subset for demonstration (GEPA can handle larger datasets)
	trainingSize := 50
	if len(examples) > trainingSize {
		examples = examples[:trainingSize]
	}
	logger.Info(ctx, "📊 Loaded %d training examples from %s dataset", len(examples), datasetName)

	// Create the signature and module based on dataset
	signature, metricFunc := createTaskSignature(datasetName)

	// Create the module to optimize with native JSON structured output
	// for reliable multi-field extraction (reasoning + answer)
	module := modules.NewChainOfThought(*signature).WithStructuredOutput()

	// Create the program to be optimized
	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{"reasoner": module},
		func(bound map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			module := bound["reasoner"]
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				return module.Process(ctx, inputs)
			}
		},
	)

	// Keep the default metric-call budget and attempt one proposal per iteration.
	config := optimizers.DefaultGEPAConfig()
	config.MaxGenerations = generations

	logger.Info(ctx, "🔧 Creating GEPA optimizer with a %d metric-call budget", config.MaxMetricCalls)
	gepa, err := optimizers.NewGEPA(config)
	if err != nil {
		logger.Fatalf(ctx, "Failed to create GEPA optimizer: %v", err)
	}

	// Set up progress reporting
	gepa.SetProgressReporter(&ProgressReporter{logger: logger, ctx: ctx})

	logger.Info(ctx, "🚀 Starting GEPA optimization:")
	logger.Info(ctx, "   • One seed program, retaining accepted children and their parents")
	logger.Info(ctx, "   • One reflection proposal from each selected parent's minibatch evidence")
	logger.Info(ctx, "   • Strict minibatch improvement before full validation")
	logger.Info(ctx, "   • Pareto selection using per-example user-metric scores")

	startTime := time.Now()

	// Perform GEPA optimization. The metric is a pure function, so GEPA may
	// call it concurrently.
	optimizedProgram, err := gepa.CompileWithScorer(ctx, program, examples, core.MetricScorer(metricFunc))
	if err != nil {
		logger.Fatalf(ctx, "GEPA optimization failed: %v", err)
	}

	optimizationDuration := time.Since(startTime)
	logger.Info(ctx, "✅ GEPA optimization completed in %v", optimizationDuration)

	// Analyze and display results
	displayOptimizationResults(ctx, logger, gepa, datasetName, optimizationDuration)

	// Test the optimized program on new examples
	testOptimizedProgram(ctx, logger, optimizedProgram, examples, datasetName, metricFunc)

	// Demonstrate paper's key contribution: access to diverse elite solutions
	demonstrateParetoArchive(ctx, logger, gepa)
}

func loadDataset(datasetName string) ([]core.Example, error) {
	switch datasetName {
	case "gsm8k":
		gsm8kExamples, err := datasets.LoadGSM8K()
		if err != nil {
			return nil, err
		}
		// Convert to core.Example
		examples := make([]core.Example, len(gsm8kExamples))
		for i, ex := range gsm8kExamples {
			examples[i] = core.Example{
				Inputs:  map[string]any{"question": ex.Question},
				Outputs: map[string]any{"answer": ex.Answer},
			}
		}
		return examples, nil
	case "hotpotqa":
		hotpotExamples, err := datasets.LoadHotpotQA()
		if err != nil {
			return nil, err
		}
		// Convert to core.Example
		examples := make([]core.Example, len(hotpotExamples))
		for i, ex := range hotpotExamples {
			examples[i] = core.Example{
				Inputs:  map[string]any{"question": ex.Question},
				Outputs: map[string]any{"answer": ex.Answer},
			}
		}
		return examples, nil
	default:
		return loadDataset("gsm8k") // Default to GSM8K
	}
}

func createTaskSignature(datasetName string) (*core.Signature, func(map[string]any, map[string]any) float64) {
	switch datasetName {
	case "gsm8k":
		// Math reasoning signature
		signature := core.NewSignature(
			[]core.InputField{
				{Field: core.NewField("question", core.WithDescription("Math problem to solve"))},
			},
			[]core.OutputField{
				{Field: core.NewField("reasoning", core.WithDescription("Step-by-step reasoning process"))},
				{Field: core.NewField("answer", core.WithDescription("Final numerical answer"))},
			},
		).WithInstruction("Solve the math problem step by step, showing your reasoning clearly.")

		// Math-specific metric
		metricFunc := func(expected, actual map[string]any) float64 {
			expectedAns := extractNumber(fmt.Sprintf("%v", expected["answer"]))
			actualAns := extractNumber(fmt.Sprintf("%v", actual["answer"]))

			if expectedAns == actualAns && expectedAns != "" {
				return 1.0
			}
			return 0.0
		}

		return &signature, metricFunc

	case "hotpotqa":
		// Question answering signature
		signature := core.NewSignature(
			[]core.InputField{
				{Field: core.NewField("question", core.WithDescription("Question to answer"))},
			},
			[]core.OutputField{
				{Field: core.NewField("reasoning", core.WithDescription("Reasoning process for the answer"))},
				{Field: core.NewField("answer", core.WithDescription("Final answer"))},
			},
		).WithInstruction("Answer the question with clear reasoning.")

		// String similarity metric
		metricFunc := func(expected, actual map[string]any) float64 {
			expectedAns := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", expected["answer"])))
			actualAns := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", actual["answer"])))

			if expectedAns == actualAns {
				return 1.0
			}
			// Partial credit for similar answers
			if strings.Contains(actualAns, expectedAns) || strings.Contains(expectedAns, actualAns) {
				return 0.5
			}
			return 0.0
		}

		return &signature, metricFunc

	default:
		// Default to GSM8K
		return createTaskSignature("gsm8k")
	}
}

func extractNumber(text string) string {
	re := regexp.MustCompile(`-?\d+\.?\d*`)
	matches := re.FindAllString(text, -1)
	if len(matches) > 0 {
		return matches[len(matches)-1] // Return the last number found
	}
	return ""
}

func displayOptimizationResults(ctx context.Context, logger *logging.Logger, gepa *optimizers.GEPA, datasetName string, duration time.Duration) {
	state := gepa.GetOptimizationState()

	logger.Info(ctx, "")
	logger.Info(ctx, "🏆 GEPA Optimization Results:")
	logger.Info(ctx, "%s", "="+strings.Repeat("=", 50))
	logger.Info(ctx, "Dataset: %s", datasetName)
	logger.Info(ctx, "Total Generations: %d", state.CurrentGeneration)
	logger.Info(ctx, "Optimization Duration: %v", duration)
	logger.Info(ctx, "Best Fitness Achieved: %.4f", state.BestFitness)
	logger.Info(ctx, "")

	if state.BestCandidate != nil {
		logger.Info(ctx, "🥇 Best Evolved Prompt:")
		logger.Info(ctx, "Generation: %d", state.BestCandidate.Generation)
		logger.Info(ctx, "Instruction: %s", state.BestCandidate.Instruction)
		logger.Info(ctx, "")
	}

	// Show Pareto archive statistics (key paper contribution)
	archive := state.GetParetoArchive()
	logger.Info(ctx, "📊 Pareto Archive (Elite Solutions):")
	logger.Info(ctx, "Elite solutions preserved: %d", len(archive))
	logger.Info(ctx, "These candidates cover different examples under the supplied metric.")
	logger.Info(ctx, "")

	// Show evolution statistics
	logger.Info(ctx, "🧬 Evolution Statistics:")
	logger.Info(ctx, "Population History: %d generations", len(state.PopulationHistory))
	logger.Info(ctx, "Metric Calls: %d", state.MetricCallCount())
	logger.Info(ctx, "")
}

func testOptimizedProgram(ctx context.Context, logger *logging.Logger, program core.Program, examples []core.Example, datasetName string, metricFunc func(map[string]any, map[string]any) float64) {
	logger.Info(ctx, "🧪 Testing Optimized Program:")
	logger.Info(ctx, "%s", "-"+strings.Repeat("-", 30))

	// Test on a few examples
	testCount := 3
	if len(examples) < testCount {
		testCount = len(examples)
	}

	totalScore := 0.0
	for i := 0; i < testCount; i++ {
		example := examples[i]

		result, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			logger.Error(ctx, "Test %d failed: %v", i+1, err)
			continue
		}

		score := metricFunc(example.Outputs, result)
		totalScore += score

		logger.Info(ctx, "Test %d:", i+1)
		if datasetName == "gsm8k" {
			logger.Info(ctx, "  Question: %s", example.Inputs["question"])
			logger.Info(ctx, "  Expected: %s", example.Outputs["answer"])
			logger.Info(ctx, "  Actual: %s", result["answer"])
		} else {
			logger.Info(ctx, "  Question: %s", example.Inputs["question"])
			logger.Info(ctx, "  Expected: %s", example.Outputs["answer"])
			logger.Info(ctx, "  Actual: %s", result["answer"])
		}
		logger.Info(ctx, "  Score: %.2f", score)
		logger.Info(ctx, "")
	}

	avgScore := totalScore / float64(testCount)
	logger.Info(ctx, "📈 Average Test Score: %.4f", avgScore)
	logger.Info(ctx, "")
}

func demonstrateParetoArchive(ctx context.Context, logger *logging.Logger, gepa *optimizers.GEPA) {
	state := gepa.GetOptimizationState()
	archive := state.GetParetoArchive()

	logger.Info(ctx, "🎯 GEPA Paper's Key Innovation - Pareto Archive Analysis:")
	logger.Info(ctx, "%s", "="+strings.Repeat("=", 55))

	if len(archive) == 0 {
		logger.Info(ctx, "No solutions in Pareto archive")
		return
	}

	logger.Info(ctx, "The archive contains %d solutions covering different examples:", len(archive))

	// Show up to 3 diverse solutions from the archive
	showCount := 3
	if len(archive) < showCount {
		showCount = len(archive)
	}

	for i := 0; i < showCount; i++ {
		candidate := archive[i]
		logger.Info(ctx, "")
		logger.Info(ctx, "Elite Solution %d:", i+1)
		logger.Info(ctx, "  Generation: %d", candidate.Generation)
		logger.Info(ctx, "  Fitness: %.4f", candidate.Fitness)
		logger.Info(ctx, "  Instruction: %s", candidate.Instruction)
	}

	logger.Info(ctx, "")
	logger.Info(ctx, "The returned program maximizes aggregate validation score; the archive retains alternatives.")
}

// ProgressReporter implements core.ProgressReporter for GEPA.
type ProgressReporter struct {
	logger *logging.Logger
	ctx    context.Context
}

func (pr *ProgressReporter) Report(operation string, current, total int) {
	if current%2 == 0 || current == total {
		percentage := float64(current) / float64(total) * 100
		pr.logger.Info(pr.ctx, "🧬 %s: %d/%d (%.1f%%)", operation, current, total, percentage)
	}
}
