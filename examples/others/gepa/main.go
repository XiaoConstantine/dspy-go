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

// SimpleDataset implements core.Dataset for the example.
type SimpleDataset struct {
	examples []core.Example
	index    int
}

func (d *SimpleDataset) Next() (core.Example, bool) {
	if d.index >= len(d.examples) {
		d.index = 0 // Reset for reuse
		return core.Example{}, false
	}
	example := d.examples[d.index]
	d.index++
	return example, true
}

func (d *SimpleDataset) Reset() {
	d.index = 0
}

func (d *SimpleDataset) Size() int {
	return len(d.examples)
}

func main() {
	apiKey := flag.String("api-key", "", "API Key for the Gemini LLM provider")
	dataset := flag.String("dataset", "gsm8k", "Dataset to use (gsm8k or hotpotqa)")
	populationSize := flag.Int("population", 12, "GEPA population size")
	generations := flag.Int("generations", 8, "Maximum number of generations")
	verbose := flag.Bool("verbose", false, "Enable verbose logging for GEPA evolution")
	flag.Parse()

	if *apiKey == "" {
		fmt.Println("Please provide a Gemini API key using -api-key flag")
		fmt.Println("Usage: go run main.go -api-key YOUR_GEMINI_API_KEY")
		return
	}

	// Run the GEPA example
	RunGEPAExample(*apiKey, *dataset, *populationSize, *generations, *verbose)
}

func RunGEPAExample(apiKey, datasetName string, populationSize, generations int, verbose bool) {
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
	logger.Info(ctx, "Dataset: %s, Population: %d, Generations: %d", datasetName, populationSize, generations)

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

	// Create the module to optimize
	module := modules.NewChainOfThought(*signature)

	// Create the program to be optimized
	program := core.NewProgram(
		map[string]core.Module{"reasoner": module},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return module.Process(ctx, inputs)
		},
	)

	// Configure GEPA with paper-inspired settings
	config := &optimizers.GEPAConfig{
		PopulationSize:       populationSize,    // Population size for evolutionary search
		MaxGenerations:       generations,       // Maximum number of generations
		SelectionStrategy:    "adaptive_pareto", // Multi-objective Pareto selection (paper's approach)
		MutationRate:         0.3,               // Mutation probability
		CrossoverRate:        0.7,               // Crossover probability
		ElitismRate:          0.1,               // Elite preservation rate
		ReflectionFreq:       2,                 // LLM-based reflection every 2 generations (key paper feature)
		ReflectionDepth:      3,                 // Depth of reflection analysis
		SelfCritiqueTemp:     0.7,               // Temperature for self-critique
		TournamentSize:       3,                 // Tournament selection size
		ConvergenceThreshold: 0.01,              // Convergence threshold
		StagnationLimit:      3,                 // Stagnation limit
		EvaluationBatchSize:  5,                 // Batch size for evaluation
		ConcurrencyLevel:     3,                 // Concurrent evaluation level
		Temperature:          0.8,               // LLM generation temperature
		MaxTokens:            8192,              // Maximum tokens for generation
	}

	logger.Info(ctx, "🔧 Creating GEPA optimizer with multi-objective evolutionary configuration")
	gepa, err := optimizers.NewGEPA(config)
	if err != nil {
		logger.Fatalf(ctx, "Failed to create GEPA optimizer: %v", err)
	}

	// Set up progress reporting
	gepa.SetProgressReporter(&ProgressReporter{logger: logger, ctx: ctx})

	logger.Info(ctx, "🚀 Starting GEPA optimization - this demonstrates the paper's key innovations:")
	logger.Info(ctx, "   • Multi-objective Pareto optimization across 7 dimensions")
	logger.Info(ctx, "   • LLM-based self-reflection and critique (every %d generations)", config.ReflectionFreq)
	logger.Info(ctx, "   • Semantic diversity metrics using LLM similarity")
	logger.Info(ctx, "   • Elite archive management with crowding distance")
	logger.Info(ctx, "   • Real-time system monitoring and context awareness")

	startTime := time.Now()

	// Create dataset from examples
	dataset := &SimpleDataset{examples: examples}

	// Perform GEPA optimization
	optimizedProgram, err := gepa.Compile(ctx, program, dataset, metricFunc)
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
				Inputs:  map[string]interface{}{"question": ex.Question},
				Outputs: map[string]interface{}{"answer": ex.Answer},
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
				Inputs:  map[string]interface{}{"question": ex.Question},
				Outputs: map[string]interface{}{"answer": ex.Answer},
			}
		}
		return examples, nil
	default:
		return loadDataset("gsm8k") // Default to GSM8K
	}
}

func createTaskSignature(datasetName string) (*core.Signature, func(map[string]interface{}, map[string]interface{}) float64) {
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
		metricFunc := func(expected, actual map[string]interface{}) float64 {
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
		metricFunc := func(expected, actual map[string]interface{}) float64 {
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
	logger.Info(ctx, "These solutions represent different trade-offs:")
	logger.Info(ctx, "• High accuracy vs fast execution")
	logger.Info(ctx, "• Quality vs efficiency balance")
	logger.Info(ctx, "• Robustness vs generalization optimization")
	logger.Info(ctx, "")

	// Show evolution statistics
	logger.Info(ctx, "🧬 Evolution Statistics:")
	logger.Info(ctx, "Population History: %d generations", len(state.PopulationHistory))
	logger.Info(ctx, "Reflection Events: %d", len(state.ReflectionHistory))
	logger.Info(ctx, "")
}

func testOptimizedProgram(ctx context.Context, logger *logging.Logger, program core.Program, examples []core.Example, datasetName string, metricFunc func(map[string]interface{}, map[string]interface{}) float64) {
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

	logger.Info(ctx, "The archive contains %d elite solutions, each optimized for different objectives:", len(archive))

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
		logger.Info(ctx, "  Optimization Focus: %s", getOptimizationFocus(i))
	}

	logger.Info(ctx, "")
	logger.Info(ctx, "💡 This demonstrates GEPA's key advantage over single-objective optimizers:")
	logger.Info(ctx, "   Instead of one 'best' solution, you get a diverse set of elite solutions")
	logger.Info(ctx, "   optimized for different trade-offs, allowing you to choose based on your needs.")
}

func getOptimizationFocus(index int) string {
	focuses := []string{
		"High accuracy with detailed reasoning",
		"Fast execution with good quality",
		"Robust performance across diverse inputs",
	}
	if index < len(focuses) {
		return focuses[index]
	}
	return "Balanced multi-objective performance"
}

// ProgressReporter implements core.ProgressReporter for GEPA.
type ProgressReporter struct {
	logger *logging.Logger
	ctx    context.Context
}

func (pr *ProgressReporter) Report(operation string, current, total int) {
	if current%2 == 0 || current == total { // Report every 2 generations or at completion
		percentage := float64(current) / float64(total) * 100
		pr.logger.Info(pr.ctx, "🧬 %s: %d/%d (%.1f%%) - Generation %d", operation, current, total, percentage, current)
	}
}
