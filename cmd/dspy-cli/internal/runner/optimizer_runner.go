package runner

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/XiaoConstantine/dspy-go/cmd/dspy-cli/internal/samples"
)

// OptimizerConfig holds configuration for running an optimizer
type OptimizerConfig struct {
	OptimizerName string
	DatasetName   string
	APIKey        string
	MaxExamples   int
	Verbose       bool
	SuppressLogs  bool // Suppress console output for TUI mode
	Parameters    map[string]interface{} // Optimizer-specific parameters
}

// RunResult holds the results of an optimizer run
type RunResult struct {
	OptimizerName    string
	DatasetName      string
	InitialAccuracy  float64
	FinalAccuracy    float64
	ImprovementPct   float64
	Duration         time.Duration
	ExamplesUsed     int
	Success          bool
	ErrorMessage     string
}

// RunOptimizer eliminates all boilerplate and runs an optimizer with sample data
func RunOptimizer(config OptimizerConfig) (*RunResult, error) {
	startTime := time.Now()

	result := &RunResult{
		OptimizerName: config.OptimizerName,
		DatasetName:   config.DatasetName,
	}

	// Setup logging
	setupLogging(config.Verbose, config.SuppressLogs)

	// Create context
	ctx := core.WithExecutionState(context.Background())

	// Setup LLM
	if err := setupLLM(config.APIKey); err != nil {
		result.ErrorMessage = fmt.Sprintf("LLM setup failed: %v", err)
		return result, err
	}

	// Get sample dataset
	dataset, exists := samples.GetSampleDataset(config.DatasetName)
	if !exists {
		result.ErrorMessage = fmt.Sprintf("unknown dataset: %s", config.DatasetName)
		return result, fmt.Errorf("unknown dataset: %s", config.DatasetName)
	}

	// Limit examples if requested
	examples := dataset.Examples
	if config.MaxExamples > 0 && len(examples) > config.MaxExamples {
		examples = examples[:config.MaxExamples]
	}
	result.ExamplesUsed = len(examples)

	// Create signature based on dataset and optimizer
	var signature core.Signature
	var module core.Module

	if config.OptimizerName == "mipro" {
		// Convert question/answer data to prompt/completion format for MIPRO
		for i := range examples {
			if question, ok := examples[i].Inputs["question"]; ok {
				examples[i].Inputs["prompt"] = question
				delete(examples[i].Inputs, "question")
			}
			if answer, ok := examples[i].Outputs["answer"]; ok {
				examples[i].Outputs["completion"] = answer
				delete(examples[i].Outputs, "answer")
			}
		}
		signature = createSignatureForMIPRO()
		module = modules.NewChainOfThought(signature)
	} else {
		signature = createSignature()
		module = modules.NewChainOfThought(signature)
	}

	// Test initial performance
	var initialAccuracy float64
	if config.OptimizerName == "mipro" {
		initialAccuracy = evaluateModuleMIPRO(ctx, module, examples)
	} else {
		initialAccuracy = evaluateModule(ctx, module, examples)
	}
	result.InitialAccuracy = initialAccuracy

	// Create optimizer
	optimizer, err := createOptimizer(config.OptimizerName)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("optimizer creation failed: %v", err)
		return result, err
	}

	// Create program
	var program core.Program
	if config.OptimizerName == "mipro" {
		// Create extractor module for MIPRO (2-module architecture)
		extractorSignature := core.NewSignature(
			[]core.InputField{{Field: core.Field{Name: "prompt"}}},
			[]core.OutputField{{Field: core.NewField("completion")}},
		).WithInstruction("Extract the final numerical answer from the reasoning.")
		extractor := modules.NewChainOfThought(extractorSignature)

		program = core.NewProgram(
			map[string]core.Module{
				"cot":       module,
				"extractor": extractor,
			},
			func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				// First run chain of thought
				cotResult, err := module.Process(ctx, inputs)
				if err != nil {
					return nil, fmt.Errorf("chain of thought failed: %w", err)
				}

				extractorInput := map[string]interface{}{
					"prompt": cotResult["completion"],
				}

				// Then extract the answer
				return extractor.Process(ctx, extractorInput)
			},
		)
	} else {
		program = core.NewProgram(
			map[string]core.Module{
				"main": module,
			},
			func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return module.Process(ctx, inputs)
			},
		)
	}

	// Split data for training/validation
	trainExamples, valExamples := splitDataset(examples)

	// Create datasets
	trainDataset := createDataset(trainExamples)

	// Create metric function
	metricFunc := createMetric()

	// Run optimization
	logger := logging.GetLogger()
	logger.Info(ctx, fmt.Sprintf("Starting %s optimization with %d training examples...",
		config.OptimizerName, len(trainExamples)))

	optimizedProgram, err := optimizer.Compile(ctx, program, trainDataset, metricFunc)
	if err != nil {
		result.ErrorMessage = fmt.Sprintf("optimization failed: %v", err)
		return result, err
	}

	// Test final performance
	var finalAccuracy float64
	if config.OptimizerName == "mipro" {
		finalAccuracy = evaluateProgramMIPRO(ctx, optimizedProgram, valExamples)
	} else {
		finalAccuracy = evaluateProgram(ctx, optimizedProgram, valExamples)
	}
	result.FinalAccuracy = finalAccuracy

	// Calculate improvement
	if initialAccuracy > 0 {
		result.ImprovementPct = ((finalAccuracy - initialAccuracy) / initialAccuracy) * 100
	} else {
		result.ImprovementPct = finalAccuracy * 100
	}
	result.Duration = time.Since(startTime)
	result.Success = true

	return result, nil
}

func setupLogging(verbose bool, suppressLogs bool) {
	severity := logging.INFO
	if verbose {
		severity = logging.DEBUG
	}

	// If suppressing logs (TUI mode), use a no-op logger
	if suppressLogs {
		logger := logging.NewLogger(logging.Config{
			Severity: logging.ERROR, // Only show critical errors
			Outputs:  []logging.Output{}, // No outputs
		})
		logging.SetLogger(logger)
		return
	}

	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: severity,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)
}

func setupLLM(apiKey string) error {
	// Auto-detect API key from environment if not provided
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
		}
		if apiKey == "" {
			apiKey = os.Getenv("DSPY_API_KEY")
		}
		if apiKey == "" {
			return fmt.Errorf("API key required. Set GEMINI_API_KEY, GOOGLE_API_KEY, or DSPY_API_KEY environment variable")
		}
	}

	llms.EnsureFactory()
	return core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
}

func createSignature() core.Signature {
	// Standard question->answer signature for most tasks
	return core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
}

func createSignatureForMIPRO() core.Signature {
	// MIPRO-compatible signature using prompt/completion format
	return core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "prompt"}}},
		[]core.OutputField{{Field: core.NewField("completion")}},
	).WithInstruction("Think step by step to answer this question.")
}

func createOptimizer(name string) (core.Optimizer, error) {
	switch name {
	case "bootstrap":
		metricFunc := func(example, prediction map[string]interface{}, ctx context.Context) bool {
			expected, ok1 := example["answer"].(string)
			predicted, ok2 := prediction["answer"].(string)
			return ok1 && ok2 && predicted == expected
		}
		return optimizers.NewBootstrapFewShot(metricFunc, 3), nil
	case "mipro":
		metricFunc := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
			expected, ok1 := example["completion"].(string)
			predicted, ok2 := prediction["completion"].(string)
			if !ok1 || !ok2 {
				return 0.0
			}
			if predicted == expected {
				return 1.0
			}
			return 0.0
		}
		return optimizers.NewMIPRO(
			metricFunc,
			optimizers.WithMode(optimizers.LightMode),
			optimizers.WithNumTrials(3),
			optimizers.WithMiniBatchSize(2),
			optimizers.WithTPEGamma(0.25),
			optimizers.WithNumModules(2), // Match working example
			optimizers.WithMaxLabeledDemos(3),
			optimizers.WithNumCandidates(3),
		), nil
	case "simba":
		return optimizers.NewSIMBA(
			optimizers.WithSIMBABatchSize(2),
			optimizers.WithFastMode(true),
		), nil
	case "gepa":
		config := &optimizers.GEPAConfig{
			PopulationSize:    4,
			MaxGenerations:    2,
			SelectionStrategy: "adaptive_pareto",
			MutationRate:      0.3,
			CrossoverRate:     0.7,
			ReflectionFreq:    5, // Avoid division by zero
		}
		return optimizers.NewGEPA(config)
	case "copro":
		metricFunc := func(expected, actual map[string]interface{}) float64 {
			expectedAnswer, ok1 := expected["answer"].(string)
			actualAnswer, ok2 := actual["answer"].(string)
			if !ok1 || !ok2 {
				return 0.0
			}
			if actualAnswer == expectedAnswer {
				return 1.0
			}
			return 0.0
		}
		return optimizers.NewCOPRO(metricFunc), nil
	default:
		return nil, fmt.Errorf("optimizer '%s' not yet implemented in CLI", name)
	}
}

func createMetric() core.Metric {
	return func(expected, actual map[string]interface{}) float64 {
		expectedAnswer, ok1 := expected["answer"].(string)
		actualAnswer, ok2 := actual["answer"].(string)
		if !ok1 || !ok2 {
			return 0.0
		}
		if actualAnswer == expectedAnswer {
			return 1.0
		}
		return 0.0
	}
}

func splitDataset(examples []core.Example) ([]core.Example, []core.Example) {
	// Split 70/30 for train/validation
	splitPoint := int(float64(len(examples)) * 0.7)
	if splitPoint == 0 {
		splitPoint = 1
	}
	if splitPoint >= len(examples) {
		// If very small dataset, use all for both train and val
		return examples, examples
	}

	return examples[:splitPoint], examples[splitPoint:]
}

func createDataset(examples []core.Example) core.Dataset {
	return &exampleDataset{examples: examples}
}

// Custom implementation of Dataset interface
type exampleDataset struct {
	examples []core.Example
	position int
}

func (d *exampleDataset) Next() (core.Example, bool) {
	if d.position >= len(d.examples) {
		return core.Example{}, false
	}
	example := d.examples[d.position]
	d.position++
	return example, true
}

func (d *exampleDataset) Reset() {
	d.position = 0
}

func evaluateModule(ctx context.Context, module core.Module, examples []core.Example) float64 {
	if len(examples) == 0 {
		return 0.0
	}

	correct := 0
	for _, example := range examples {
		result, err := module.Process(ctx, example.Inputs)
		if err != nil {
			continue
		}

		if predicted, ok := result["answer"].(string); ok {
			if expected, ok := example.Outputs["answer"].(string); ok {
				if predicted == expected {
					correct++
				}
			}
		}
	}

	return float64(correct) / float64(len(examples))
}

func evaluateProgram(ctx context.Context, program core.Program, examples []core.Example) float64 {
	if len(examples) == 0 {
		return 0.0
	}

	correct := 0
	for _, example := range examples {
		result, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			continue
		}

		if predicted, ok := result["answer"].(string); ok {
			if expected, ok := example.Outputs["answer"].(string); ok {
				if predicted == expected {
					correct++
				}
			}
		}
	}

	return float64(correct) / float64(len(examples))
}

func evaluateModuleMIPRO(ctx context.Context, module core.Module, examples []core.Example) float64 {
	if len(examples) == 0 {
		return 0.0
	}

	correct := 0
	for _, example := range examples {
		result, err := module.Process(ctx, example.Inputs)
		if err != nil {
			continue
		}

		if predicted, ok := result["completion"].(string); ok {
			if expected, ok := example.Outputs["completion"].(string); ok {
				if predicted == expected {
					correct++
				}
			}
		}
	}

	return float64(correct) / float64(len(examples))
}

func evaluateProgramMIPRO(ctx context.Context, program core.Program, examples []core.Example) float64 {
	if len(examples) == 0 {
		return 0.0
	}

	correct := 0
	for _, example := range examples {
		result, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			continue
		}

		if predicted, ok := result["completion"].(string); ok {
			if expected, ok := example.Outputs["completion"].(string); ok {
				if predicted == expected {
					correct++
				}
			}
		}
	}

	return float64(correct) / float64(len(examples))
}
