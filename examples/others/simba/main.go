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
	apiKey := flag.String("api-key", "", "API Key for the LLM provider")
	flag.Parse()

	// Run the SIMBA example
	RunSIMBAExample(*apiKey)
}

func RunSIMBAExample(apiKey string) {
	// Setup enhanced logging for SIMBA introspection
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

	// Load dataset - using a subset of GSM8K for demonstration
	examples, err := datasets.LoadGSM8K()
	if err != nil {
		logger.Fatalf(ctx, "Failed to load dataset: %v", err)
	}

	// Create a smaller dataset for faster SIMBA optimization demonstration
	trainExamples := examples[:15]  // Small training set for mini-batch optimization
	testExamples := examples[15:25] // Small test set for evaluation

	// Convert to dspy-go Dataset format
	trainDataset := createDataset(trainExamples)

	// Create a reasoning program optimized for mathematical problem solving
	// This showcases how SIMBA can iteratively improve instruction quality
	reasoner := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("reasoning")}, {Field: core.NewField("answer")}},
	).WithInstruction("Solve this math problem step by step, showing your reasoning clearly and providing the final numerical answer."))

	// Create the program
	program := core.NewProgram(
		map[string]core.Module{
			"reasoner": reasoner,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			// Process the question through our reasoner
			result, err := reasoner.Process(ctx, inputs)
			if err != nil {
				return nil, fmt.Errorf("reasoning failed: %w", err)
			}

			// Extract and clean the answer
			if reasoning, ok := result["reasoning"].(string); ok {
				result["reasoning"] = reasoning
			}
			if answer, ok := result["answer"].(string); ok {
				result["answer"] = extractFinalAnswer(answer)
			}

			return result, nil
		},
	)

	// Define a metric function that evaluates both correctness and reasoning quality
	metricFunc := func(expected, actual map[string]interface{}) float64 {
		expectedAnswer, ok1 := expected["completion"].(string)
		actualAnswer, ok2 := actual["answer"].(string)
		if !ok1 || !ok2 {
			return 0.0
		}

		// Extract numerical answers for comparison
		expectedNum := extractFinalAnswer(expectedAnswer)
		actualNum := extractFinalAnswer(actualAnswer)

		// Perfect match gets full score
		if strings.TrimSpace(expectedNum) == strings.TrimSpace(actualNum) {
			return 1.0
		}

		// Partial credit for reasonable answers (within 10% for non-zero)
		if expectedNum != "0" && actualNum != "" {
			// Simple heuristic for partial credit - could be enhanced
			if len(actualNum) > 0 && actualNum[0] == expectedNum[0] {
				return 0.3
			}
		}

		return 0.0
	}

	// Create SIMBA optimizer with configuration optimized for demonstration
	// This showcases SIMBA's key features and introspective capabilities
	logger.Info(ctx, "Creating SIMBA optimizer with introspective learning enabled...")
	
	optimizer := optimizers.NewSIMBA(
		// Configure for demonstration - smaller values for faster execution
		optimizers.WithSIMBABatchSize(4),           // Small batches for stochastic optimization
		optimizers.WithSIMBAMaxSteps(6),            // Enough steps to show convergence
		optimizers.WithSIMBANumCandidates(4),       // Generate multiple instruction variants
		optimizers.WithSamplingTemperature(0.3),    // Moderate exploration vs exploitation
	)

	// Display initial program state
	logger.Info(ctx, "Initial program configuration:")
	displayProgramInfo(ctx, program, logger)

	// Run SIMBA optimization with detailed progress tracking
	logger.Info(ctx, "Starting SIMBA optimization with introspective learning...")
	logger.Info(ctx, "SIMBA will:")
	logger.Info(ctx, "  • Generate instruction variants using LLM-based perturbation")
	logger.Info(ctx, "  • Evaluate candidates on mini-batches for efficiency")
	logger.Info(ctx, "  • Use temperature-controlled sampling for exploration")
	logger.Info(ctx, "  • Perform introspective analysis every 2 steps")
	logger.Info(ctx, "  • Detect convergence automatically")
	logger.Info(ctx, "")

	startTime := time.Now()
	optimizedProgram, err := optimizer.Compile(ctx, program, trainDataset, metricFunc)
	optimizationTime := time.Since(startTime)

	if err != nil {
		logger.Fatalf(ctx, "SIMBA optimization failed: %v", err)
	}

	logger.Info(ctx, "SIMBA optimization completed successfully!")
	logger.Info(ctx, "Total optimization time: %v", optimizationTime)

	// Display SIMBA's introspective insights
	displaySIMBAInsights(ctx, optimizer, logger)

	// Display optimized program information
	logger.Info(ctx, "Optimized program configuration:")
	displayProgramInfo(ctx, optimizedProgram, logger)

	// Evaluate both original and optimized programs
	logger.Info(ctx, "Comparing original vs optimized performance...")
	
	logger.Info(ctx, "\n=== ORIGINAL PROGRAM EVALUATION ===")
	originalScore := runEvaluation(ctx, program, testExamples, logger, "Original")
	
	logger.Info(ctx, "\n=== OPTIMIZED PROGRAM EVALUATION ===")
	optimizedScore := runEvaluation(ctx, optimizedProgram, testExamples, logger, "Optimized")

	// Display improvement summary
	improvement := optimizedScore - originalScore
	improvementPercent := improvement * 100

	logger.Info(ctx, "\n=== SIMBA OPTIMIZATION RESULTS ===")
	logger.Info(ctx, "Original accuracy:  %.1f%%", originalScore*100)
	logger.Info(ctx, "Optimized accuracy: %.1f%%", optimizedScore*100)
	if improvement > 0 {
		logger.Info(ctx, "Improvement: +%.1f%% (SIMBA optimization successful!)", improvementPercent)
	} else if improvement == 0 {
		logger.Info(ctx, "No change in accuracy (program already well-optimized)")
	} else {
		logger.Info(ctx, "Decrease: %.1f%% (optimization may need different parameters)", improvementPercent)
	}
	
	logger.Info(ctx, "Optimization completed in %v", optimizationTime)
}

// Helper function to create a Dataset from examples
func createDataset(examples []datasets.GSM8KExample) core.Dataset {
	data := make([]core.Example, len(examples))
	for i, ex := range examples {
		data[i] = core.Example{
			Inputs: map[string]interface{}{
				"question": ex.Question,
			},
			Outputs: map[string]interface{}{
				"completion": ex.Answer,
			},
		}
	}
	return &exampleDataset{examples: data}
}

// Custom Dataset implementation
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

// Enhanced answer extraction for GSM8K format
func extractFinalAnswer(answerString string) string {
	// Clean the input
	cleaned := strings.TrimSpace(answerString)
	
	// Look for #### pattern first (GSM8K standard)
	re := regexp.MustCompile(`####\s*(-?\d+(?:\.\d+)?)`)
	if matches := re.FindStringSubmatch(cleaned); len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}
	
	// Fallback to finding any number in the string
	cleaned = strings.ReplaceAll(cleaned, "$", "")
	cleaned = strings.ReplaceAll(cleaned, ",", "")
	
	numRe := regexp.MustCompile(`-?\d+(?:\.\d+)?`)
	if numMatch := numRe.FindString(cleaned); numMatch != "" {
		return numMatch
	}
	
	return cleaned
}

// Display detailed program information
func displayProgramInfo(ctx context.Context, program core.Program, logger *logging.Logger) {
	for name, module := range program.Modules {
		signature := module.GetSignature()
		logger.Info(ctx, "  Module '%s': %s", name, signature.Instruction)
		logger.Info(ctx, "    Inputs: %v", getInputFieldNames(signature.Inputs))
		logger.Info(ctx, "    Outputs: %v", getOutputFieldNames(signature.Outputs))
	}
}

// Helper to extract input field names for display
func getInputFieldNames(fields []core.InputField) []string {
	names := make([]string, len(fields))
	for i, field := range fields {
		names[i] = field.Field.Name
	}
	return names
}

// Helper to extract output field names for display
func getOutputFieldNames(fields []core.OutputField) []string {
	names := make([]string, len(fields))
	for i, field := range fields {
		names[i] = field.Field.Name
	}
	return names
}

// Display SIMBA's introspective insights and analysis
func displaySIMBAInsights(ctx context.Context, optimizer *optimizers.SIMBA, logger *logging.Logger) {
	state := optimizer.GetState()
	config := optimizer.GetConfig()

	logger.Info(ctx, "\n=== SIMBA INTROSPECTIVE ANALYSIS ===")
	logger.Info(ctx, "Optimization Statistics:")
	logger.Info(ctx, "  • Total steps executed: %d/%d", state.CurrentStep+1, config.MaxSteps)
	logger.Info(ctx, "  • Final best score: %.4f", state.BestScore)
	logger.Info(ctx, "  • Total candidates evaluated: %d", len(state.CandidateHistory))
	logger.Info(ctx, "  • Optimization duration: %v", time.Since(state.StartTime))

	if len(state.PerformanceLog) > 0 {
		logger.Info(ctx, "\nOptimization Progress:")
		for _, step := range state.PerformanceLog {
			logger.Info(ctx, "  Step %d: Score=%.4f, Improvement=%.4f, Temperature=%.3f, Batch=%d", 
				step.Step, step.BestScore, step.Improvement, step.Temperature, step.BatchSize)
			
			if step.Introspection != "" {
				logger.Info(ctx, "    Introspection: %s", truncateString(step.Introspection, 100))
			}
		}
	}

	if len(state.IntrospectionLog) > 0 {
		logger.Info(ctx, "\nIntrospective Insights:")
		for i, insight := range state.IntrospectionLog {
			logger.Info(ctx, "  Analysis %d: %s", i+1, truncateString(insight, 150))
		}
	}

	// Show convergence analysis
	if len(state.PerformanceLog) >= 3 {
		recentSteps := state.PerformanceLog[max(0, len(state.PerformanceLog)-3):]
		logger.Info(ctx, "\nConvergence Analysis:")
		
		totalImprovement := 0.0
		for _, step := range recentSteps {
			totalImprovement += step.Improvement
		}
		avgImprovement := totalImprovement / float64(len(recentSteps))
		
		if avgImprovement < config.ConvergenceThreshold {
			logger.Info(ctx, "  ✓ Converged: Average improvement (%.6f) below threshold (%.6f)", 
				avgImprovement, config.ConvergenceThreshold)
		} else {
			logger.Info(ctx, "  ○ Still improving: Average improvement (%.6f) above threshold (%.6f)", 
				avgImprovement, config.ConvergenceThreshold)
		}
	}
}

// Helper function to truncate long strings for display
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// Helper function for max
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Enhanced evaluation function with detailed reporting
func runEvaluation(ctx context.Context, program core.Program, examples []datasets.GSM8KExample, logger *logging.Logger, programType string) float64 {
	correct := 0
	total := 0

	logger.Info(ctx, "Evaluating %s program on %d test examples...", programType, len(examples))
	
	for i, ex := range examples {
		result, err := program.Execute(ctx, map[string]interface{}{"question": ex.Question})
		if err != nil {
			logger.Warn(ctx, "Error executing program on example %d: %v", i+1, err)
			total++
			continue
		}

		predictedAnswer, ok := result["answer"].(string)
		if !ok {
			logger.Warn(ctx, "Invalid answer format from program on example %d", i+1)
			total++
			continue
		}

		// Extract numerical answers
		actualAnswer := extractFinalAnswer(ex.Answer)
		predictedAnswer = extractFinalAnswer(predictedAnswer)

		isCorrect := strings.TrimSpace(predictedAnswer) == strings.TrimSpace(actualAnswer)
		if isCorrect {
			correct++
		}
		total++

		// Log first few examples for visibility
		if i < 3 {
			logger.Info(ctx, "Example %d:", i+1)
			logger.Info(ctx, "  Question: %s", truncateString(ex.Question, 80))
			logger.Info(ctx, "  Expected: %s", actualAnswer)
			logger.Info(ctx, "  Predicted: %s", predictedAnswer)
			logger.Info(ctx, "  Correct: %v", isCorrect)
			
			if reasoning, ok := result["reasoning"].(string); ok && reasoning != "" {
				logger.Info(ctx, "  Reasoning: %s", truncateString(reasoning, 100))
			}
			logger.Info(ctx, "")
		}
	}

	accuracy := float64(correct) / float64(total)
	logger.Info(ctx, "%s program accuracy: %.1f%% (%d/%d correct)", 
		programType, accuracy*100, correct, total)

	return accuracy
}