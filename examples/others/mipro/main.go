package main

import (
	"context"
	"flag"
	"fmt"
	"regexp"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func main() {
	apiKey := flag.String("api-key", "", "API Key for the LLM provider")
	flag.Parse()

	// Run the MIPRO example
	RunMIPROExample(*apiKey)
}

func RunMIPROExample(apiKey string) {
	// Setup logging
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	// Create context with execution state
	ctx := core.WithExecutionState(context.Background())

	// Configure LLM
	llms.EnsureFactory()
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to setup LLM: %v", err)
	}

	// Load dataset (using GSM8K as an example)
	examples, err := datasets.LoadGSM8K()
	if err != nil {
		logger.Fatalf(ctx, "Failed to load dataset: %v", err)
	}

	// Create a train/validation split - using fewer examples for faster execution
	valExamples := examples[:10]

	// Convert to dspy-go Dataset format
	valDataset := createDataset(valExamples)

	// Create modules for our program
	// 1. Chain of Thought for reasoning
	cotSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "prompt"}}},
		[]core.OutputField{{Field: core.NewField("completion")}},
	).WithInstruction("Think step by step to solve this math problem.")

	cot := modules.NewChainOfThought(cotSignature)

	// 2. Answer extractor to get the final answer
	extractorSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "prompt"}}},
		[]core.OutputField{{Field: core.NewField("completion")}},
	).WithInstruction("Extract the final numerical answer from the reasoning.")

	extractor := modules.NewChainOfThought(extractorSignature)

	// Create the program
	program := core.NewProgram(
		map[string]core.Module{
			"cot":       cot,
			"extractor": extractor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			// First run chain of thought
			cotResult, err := cot.Process(ctx, inputs)
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

	// Define a metric function for evaluation
	metricFunc := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
		// Simple exact match metric
		expectedAnswer, ok1 := example["completion"].(string)
		predictedAnswer, ok2 := prediction["completion"].(string)
		if !ok1 || !ok2 {
			return 0.0
		}

		if predictedAnswer == expectedAnswer {
			return 1.0
		}
		return 0.0
	}

	// Create MIPRO optimizer with simplified configuration for the demo
	optimizer := optimizers.NewMIPRO(
		metricFunc,
		// Use light mode for faster demo
		optimizers.WithMode(optimizers.LightMode),
		// Set number of trials to a small number for the demo
		optimizers.WithNumTrials(3),
		// Set minimum batch size for faster evaluation
		optimizers.WithMiniBatchSize(2),
		// Configure TPE parameters
		optimizers.WithTPEGamma(0.25),
		// Explicitly set the number of modules
		optimizers.WithNumModules(2),
		// Set max labeled demos
		optimizers.WithMaxLabeledDemos(3),

		optimizers.WithNumCandidates(3),
	)

	// Compile the program with MIPRO
	logger.Info(ctx, "Starting MIPRO optimization...")
	optimizedProgram, err := optimizer.Compile(ctx, program, valDataset, nil)
	if err != nil {
		logger.Fatalf(ctx, "Optimization failed: %v", err)
	}

	logger.Info(ctx, "Examining optimized program...")

	if len(optimizedProgram.Modules) == 0 {
		logger.Fatal(ctx, "Optimized program has no modules!")
	}
	// Check if modules exist
	if len(optimizedProgram.Modules) == 0 {
		logger.Warn(ctx, "Optimized program has no modules!")
	}

	// Check each module
	for name, module := range optimizedProgram.Modules {
		logger.Info(ctx, "Found module %s in optimized program", name)
		if module == nil {
			logger.Warn(ctx, "Module %s is nil!", name)
		} else {
			signature := module.GetSignature()
			logger.Info(ctx, "Module %s signature: %+v", name, signature)
		}
	}
	optimizedCot := optimizedProgram.Modules["cot"]
	optimizedExtractor := optimizedProgram.Modules["extractor"]

	// Create a new program with these modules and a proper forward function
	finalProgram := core.NewProgram(
		map[string]core.Module{
			"cot":       optimizedCot,
			"extractor": optimizedExtractor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			// First run the optimized chain of thought
			cotResult, err := optimizedCot.Process(ctx, inputs)
			if err != nil {
				return nil, fmt.Errorf("chain of thought failed: %w", err)
			}

			// Map field names properly
			extractorInput := map[string]interface{}{
				"prompt": cotResult["completion"],
			}

			// Then extract the answer
			return optimizedExtractor.Process(ctx, extractorInput)
		},
	)

	logger.Info(ctx, "Optimization complete!")

	// Use finalProgram instead of optimizedProgram for evaluation
	runEvaluation(ctx, finalProgram, valExamples, logger)
}

// Helper function to create a Dataset from examples.
func createDataset(examples []datasets.GSM8KExample) core.Dataset {
	// Create a slice of core.Example
	data := make([]core.Example, len(examples))
	for i, ex := range examples {
		data[i] = core.Example{
			Inputs: map[string]interface{}{
				"prompt": ex.Question,
			},
			Outputs: map[string]interface{}{
				"completion": ex.Answer,
			},
		}
	}

	// Return a new dataset implementation
	return &exampleDataset{examples: data}
}

// Custom implementation of Dataset interface.
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

func extractFinalAnswer(answerString string, isActualAnswer bool) string {
	if isActualAnswer {
		// Regex to find "#### <number>" pattern, allowing for negative numbers and decimals
		re := regexp.MustCompile(`####\s*(-?\d+(\.\d+)?)`)
		matches := re.FindStringSubmatch(answerString)
		if len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
		// If #### pattern not found in actual answer, try to clean it as a last resort,
		// but prioritize the #### pattern for ground truth.
	}

	// General cleaner for predicted answers or fallback for actual if #### not found.
	cleaned := strings.TrimSpace(answerString)
	// Remove dollar signs, as they are common in currency values.
	cleaned = strings.ReplaceAll(cleaned, "$", "")
	// Remove commas from numbers (e.g., 1,000 -> 1000)
	cleaned = strings.ReplaceAll(cleaned, ",", "")

	// Attempt to extract a number if it's the primary content.
	// This regex tries to find a number at the beginning possibly, or the main numeric part.
	numRe := regexp.MustCompile(`^-?\d+(\.\d+)?`)
	if numMatch := numRe.FindString(cleaned); numMatch != "" {
		// Check if the cleaned string is ONLY the number (with optional surrounding non-alphanumeric noise)
		// This avoids extracting "3" from "3 dogs" if we want the full "3 dogs"
		// For GSM8K, we typically want just the number.
		return numMatch
	}
	// Fallback if no leading number is found, return the cleaned string.
	// This might be useful if the answer is non-numeric (e.g. "yes", "no") or if the number is embedded differently.
	// For GSM8K, we expect a number, so if #### isn't found and this cleaning doesn't isolate one, it might be an issue.
	return cleaned
}

// Evaluate the program on test examples.
func runEvaluation(ctx context.Context, program core.Program, examples []datasets.GSM8KExample, logger *logging.Logger) {
	correct := 0
	total := 0

	logger.Info(ctx, "Running evaluation on test examples...")
	for _, ex := range examples {
		result, err := program.Execute(ctx, map[string]interface{}{"prompt": ex.Question})
		if err != nil {
			logger.Warn(ctx, "Error executing program: %v", err)
			continue
		}

		predictedAnswerRaw, ok := result["completion"].(string)
		if !ok {
			logger.Warn(ctx, "Invalid predicted answer format from program result")
			continue
		}

		// Extract numerical/key part from answers
		actualFinalAnswer := extractFinalAnswer(ex.Answer, true)              // true indicates it's the ground truth
		predictedFinalAnswer := extractFinalAnswer(predictedAnswerRaw, false) // false for predicted

		isCorrect := false
		if actualFinalAnswer == "" {
			logger.Warn(ctx, "Could not extract a final answer from actual (ground truth) answer: %s", ex.Answer)
			// Decide how to score this: if actual is unparseable, is any prediction wrong? Or skip?
			// For now, if actual is unparseable, we can't reliably score. Let's mark as incorrect.
		} else if predictedFinalAnswer == "" {
			logger.Warn(ctx, "Could not extract a final answer from predicted answer: %s", predictedAnswerRaw)
		} else {
			isCorrect = strings.TrimSpace(predictedFinalAnswer) == strings.TrimSpace(actualFinalAnswer)
		}

		if isCorrect {
			correct++
		}
		total++

		logger.Info(ctx, "Question: %s", ex.Question)
		logger.Info(ctx, "Predicted Raw: %s, Predicted Extracted: %s, Actual Raw (Full): %s, Actual Extracted: %s, Correct: %v",
			predictedAnswerRaw, predictedFinalAnswer, ex.Answer, actualFinalAnswer, isCorrect)
		logger.Info(ctx, "-------------------")
	}

	if total > 0 {
		accuracy := float64(correct) / float64(total) * 100
		logger.Info(ctx, "Final Accuracy: %.2f%% (%d/%d correct)", accuracy, correct, total)
	}
}
