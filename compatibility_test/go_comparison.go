package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

// ComparisonMetrics tracks metrics for comparison.
type ComparisonMetrics struct {
	Scores         []float64 `json:"scores"`
	ExecutionTimes []float64 `json:"execution_times"`
	TokenUsage     []int     `json:"token_usage"`
}

func (cm *ComparisonMetrics) AddScore(score float64) {
	cm.Scores = append(cm.Scores, score)
}

func (cm *ComparisonMetrics) AddExecutionTime(time float64) {
	cm.ExecutionTimes = append(cm.ExecutionTimes, time)
}

func (cm *ComparisonMetrics) AddTokenUsage(tokens int) {
	cm.TokenUsage = append(cm.TokenUsage, tokens)
}

func (cm *ComparisonMetrics) GetAverageScore() float64 {
	if len(cm.Scores) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, score := range cm.Scores {
		sum += score
	}
	return sum / float64(len(cm.Scores))
}

func (cm *ComparisonMetrics) GetTotalTokens() int {
	sum := 0
	for _, tokens := range cm.TokenUsage {
		sum += tokens
	}
	return sum
}

func (cm *ComparisonMetrics) GetTotalTime() float64 {
	sum := 0.0
	for _, time := range cm.ExecutionTimes {
		sum += time
	}
	return sum
}

// SimpleDataset implements core.Dataset interface for testing.
type SimpleDataset struct {
	examples []core.Example
	current  int
}

func NewSimpleDataset(examples []core.Example) *SimpleDataset {
	return &SimpleDataset{
		examples: examples,
		current:  0,
	}
}

func (sd *SimpleDataset) Next() (core.Example, bool) {
	if sd.current >= len(sd.examples) {
		return core.Example{}, false
	}
	example := sd.examples[sd.current]
	sd.current++
	return example, true
}

func (sd *SimpleDataset) Reset() {
	sd.current = 0
}

// OptimizerComparison handles the comparison between Go and Python implementations.
type OptimizerComparison struct {
	ModelName string
	LLM       core.LLM

	BootstrapMetrics *ComparisonMetrics
	MIPROMetrics     *ComparisonMetrics
	SIMBAMetrics     *ComparisonMetrics
}

func NewOptimizerComparison(modelName string) *OptimizerComparison {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatalf("GEMINI_API_KEY environment variable is required")
	}

	// Initialize factory
	llms.EnsureFactory()

	// Configure default LLM
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		log.Fatalf("Failed to configure default LLM: %v", err)
	}

	// Configure teacher LLM for optimization
	err = core.ConfigureTeacherLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		log.Fatalf("Failed to configure teacher LLM: %v", err)
	}

	// Get the configured LLM for reference
	llm, err := llms.NewLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	return &OptimizerComparison{
		ModelName:        modelName,
		LLM:              llm,
		BootstrapMetrics: &ComparisonMetrics{},
		MIPROMetrics:     &ComparisonMetrics{},
		SIMBAMetrics:     &ComparisonMetrics{},
	}
}

func (oc *OptimizerComparison) CreateSampleDataset(size int) []core.Example {
	sampleData := []struct {
		question string
		answer   string
	}{
		{"What is the capital of France?", "Paris"},
		{"What is 2 + 2?", "4"},
		{"What color is the sky?", "Blue"},
		{"What is the largest planet?", "Jupiter"},
		{"What is the smallest prime number?", "2"},
		{"What is the chemical symbol for water?", "H2O"},
		{"What is the speed of light?", "299,792,458 m/s"},
		{"What year did World War II end?", "1945"},
		{"What is the square root of 16?", "4"},
		{"What is the boiling point of water?", "100°C"},
	}

	examples := make([]core.Example, size)
	for i := 0; i < size; i++ {
		data := sampleData[i%len(sampleData)]
		examples[i] = core.Example{
			Inputs: map[string]interface{}{
				"question": data.question,
				"prompt":   data.question, // Add prompt field for MIPRO compatibility
			},
			Outputs: map[string]interface{}{
				"answer": data.answer,
			},
		}
	}

	return examples
}

func (oc *OptimizerComparison) AccuracyMetric(example, prediction map[string]interface{}, ctx context.Context) float64 {
	expected, ok := example["answer"].(string)
	if !ok {
		return 0.0
	}

	predicted, ok := prediction["answer"].(string)
	if !ok {
		return 0.0
	}

	// Simple substring matching for demo purposes
	if len(expected) > 0 && len(predicted) > 0 {
		if expected == predicted {
			return 1.0
		}
		// Check if one contains the other (case-insensitive)
		expectedLower := expected
		predictedLower := predicted
		if expectedLower == predictedLower {
			return 1.0
		}
	}

	return 0.0
}

func (oc *OptimizerComparison) TestBootstrapFewShot(ctx context.Context, dataset []core.Example, maxBootstrappedDemos int) (core.Program, map[string]interface{}) {
	log.Println("Testing BootstrapFewShot optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := modules.NewPredict(signature)

	program := core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Convert examples to map format expected by optimizer
	trainset := make([]map[string]interface{}, len(trainExamples))
	for i, example := range trainExamples {
		trainset[i] = map[string]interface{}{
			"question": example.Inputs["question"],
			"prompt":   example.Inputs["prompt"], // Include prompt for MIPRO
			"answer":   example.Outputs["answer"],
		}
	}

	// Create optimizer
	optimizer := optimizers.NewBootstrapFewShot(
		func(example, prediction map[string]interface{}, ctx context.Context) bool {
			return oc.AccuracyMetric(example, prediction, ctx) > 0.5
		},
		maxBootstrappedDemos,
	)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.Compile(ctx, program, program, trainset)
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.BootstrapMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":              "BootstrapFewShot",
		"compilation_time":       compilationTime,
		"average_score":          avgScore,
		"total_examples":         len(valset),
		"max_bootstrapped_demos": maxBootstrappedDemos,
		"demonstrations":         oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("BootstrapFewShot results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestMIPRO(ctx context.Context, dataset []core.Example, numTrials int, maxBootstrappedDemos int) (core.Program, map[string]interface{}) {
	log.Println("Testing MIPRO optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := modules.NewPredict(signature)

	program := core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create optimizer
	optimizer := optimizers.NewMIPRO(
		func(example, prediction map[string]interface{}, ctx context.Context) float64 {
			return oc.AccuracyMetric(example, prediction, ctx)
		},
		optimizers.WithMode(optimizers.LightMode),
		optimizers.WithNumTrials(numTrials),
		optimizers.WithMaxLabeledDemos(maxBootstrappedDemos),
	)

	// Create dataset interface for MIPRO
	trainDataset := NewSimpleDataset(trainExamples)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.Compile(ctx, program, trainDataset, nil)
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.MIPROMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":              "MIPRO",
		"compilation_time":       compilationTime,
		"average_score":          avgScore,
		"total_examples":         len(valset),
		"num_trials":             numTrials,
		"max_bootstrapped_demos": maxBootstrappedDemos,
		"demonstrations":         oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("MIPRO results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestSIMBA(ctx context.Context, dataset []core.Example, batchSize int, maxSteps int) (core.Program, map[string]interface{}) {
	log.Println("Testing SIMBA optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := modules.NewPredict(signature)

	program := core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create optimizer
	optimizer := optimizers.NewSIMBA(
		optimizers.WithSIMBABatchSize(batchSize),
		optimizers.WithSIMBAMaxSteps(maxSteps),
		optimizers.WithSIMBANumCandidates(4),
		optimizers.WithSamplingTemperature(0.2),
	)

	// Create dataset interface for SIMBA
	trainDataset := NewSimpleDataset(trainExamples)

	// Define metric function for SIMBA
	metricFunc := func(example, prediction map[string]interface{}) float64 {
		return oc.AccuracyMetric(example, prediction, ctx)
	}

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.Compile(ctx, program, trainDataset, metricFunc)
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.SIMBAMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":        "SIMBA",
		"compilation_time": compilationTime,
		"average_score":    avgScore,
		"total_examples":   len(valset),
		"batch_size":       batchSize,
		"max_steps":        maxSteps,
		"demonstrations":   oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("SIMBA results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) extractDemonstrations(program core.Program) []map[string]interface{} {
	demonstrations := []map[string]interface{}{}

	// Try to extract demos from predictors
	for _, module := range program.Modules {
		if predictor, ok := module.(*modules.Predict); ok {
			demos := predictor.GetDemos()
			for _, demo := range demos {
				demonstrations = append(demonstrations, map[string]interface{}{
					"inputs":  demo.Inputs,
					"outputs": demo.Outputs,
				})
			}
		}
	}

	return demonstrations
}

func (oc *OptimizerComparison) RunComparison(ctx context.Context, datasetSize int) map[string]interface{} {
	log.Println("Starting optimizer comparison")

	// Create dataset
	dataset := oc.CreateSampleDataset(datasetSize)

	// Test optimizers
	_, bootstrapResults := oc.TestBootstrapFewShot(ctx, dataset, 4)
	_, miproResults := oc.TestMIPRO(ctx, dataset, 5, 3)
	_, simbaResults := oc.TestSIMBA(ctx, dataset, 4, 6)

	// Compare results
	bootstrapScore := bootstrapResults["average_score"].(float64)
	miproScore := miproResults["average_score"].(float64)
	simbaScore := simbaResults["average_score"].(float64)
	bootstrapTime := bootstrapResults["compilation_time"].(float64)
	miproTime := miproResults["compilation_time"].(float64)
	simbaTime := simbaResults["compilation_time"].(float64)

	// Find best optimizer
	bestOptimizer := "BootstrapFewShot"
	bestScore := bootstrapScore
	if miproScore > bestScore {
		bestOptimizer = "MIPRO"
		bestScore = miproScore
	}
	if simbaScore > bestScore {
		bestOptimizer = "SIMBA"
		bestScore = simbaScore
	}

	comparisonResults := map[string]interface{}{
		"dataset_size":      datasetSize,
		"model":             oc.ModelName,
		"bootstrap_fewshot": bootstrapResults,
		"mipro":             miproResults,
		"simba":             simbaResults,
		"comparison": map[string]interface{}{
			"bootstrap_vs_mipro_score_diff": miproScore - bootstrapScore,
			"bootstrap_vs_simba_score_diff": simbaScore - bootstrapScore,
			"mipro_vs_simba_score_diff":     simbaScore - miproScore,
			"bootstrap_vs_mipro_time_diff":  miproTime - bootstrapTime,
			"bootstrap_vs_simba_time_diff":  simbaTime - bootstrapTime,
			"mipro_vs_simba_time_diff":      simbaTime - miproTime,
			"best_optimizer":                bestOptimizer,
			"best_score":                    bestScore,
		},
	}

	return comparisonResults
}

func (oc *OptimizerComparison) SaveResults(results map[string]interface{}, filename string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return err
	}

	log.Printf("Results saved to %s", filename)
	return nil
}

func main() {
	var optimizer = flag.String("optimizer", "all", "Optimizer to test: bootstrap, mipro, simba, or all")
	var datasetSize = flag.Int("dataset-size", 20, "Dataset size for testing")
	flag.Parse()

	ctx := context.Background()

	// Initialize comparison
	comparison := NewOptimizerComparison("gemini-1.5-flash")

	// Create dataset
	dataset := comparison.CreateSampleDataset(*datasetSize)

	results := map[string]interface{}{
		"dataset_size":     *datasetSize,
		"model":            comparison.ModelName,
		"optimizer_tested": *optimizer,
	}

	// Test individual optimizers based on parameter
	if *optimizer == "bootstrap" || *optimizer == "all" {
		fmt.Println("Testing BootstrapFewShot...")
		_, bootstrapResults := comparison.TestBootstrapFewShot(ctx, dataset, 4)
		results["bootstrap_fewshot"] = bootstrapResults
	}

	if *optimizer == "mipro" || *optimizer == "all" {
		fmt.Println("Testing MIPRO...")
		_, miproResults := comparison.TestMIPRO(ctx, dataset, 5, 3)
		results["mipro"] = miproResults
	}

	if *optimizer == "simba" || *optimizer == "all" {
		fmt.Println("Testing SIMBA...")
		_, simbaResults := comparison.TestSIMBA(ctx, dataset, 4, 6)
		results["simba"] = simbaResults
	}

	// Add comparison section if testing multiple optimizers
	if *optimizer == "all" {
		bootstrapResults := results["bootstrap_fewshot"].(map[string]interface{})
		miproResults := results["mipro"].(map[string]interface{})
		simbaResults := results["simba"].(map[string]interface{})

		bootstrapScore := bootstrapResults["average_score"].(float64)
		miproScore := miproResults["average_score"].(float64)
		simbaScore := simbaResults["average_score"].(float64)
		bootstrapTime := bootstrapResults["compilation_time"].(float64)
		miproTime := miproResults["compilation_time"].(float64)
		simbaTime := simbaResults["compilation_time"].(float64)

		// Find best optimizer
		bestOptimizer := "BootstrapFewShot"
		bestScore := bootstrapScore
		if miproScore > bestScore {
			bestOptimizer = "MIPRO"
			bestScore = miproScore
		}
		if simbaScore > bestScore {
			bestOptimizer = "SIMBA"
			bestScore = simbaScore
		}

		results["comparison"] = map[string]interface{}{
			"bootstrap_vs_mipro_score_diff": miproScore - bootstrapScore,
			"bootstrap_vs_simba_score_diff": simbaScore - bootstrapScore,
			"mipro_vs_simba_score_diff":     simbaScore - miproScore,
			"bootstrap_vs_mipro_time_diff":  miproTime - bootstrapTime,
			"bootstrap_vs_simba_time_diff":  simbaTime - bootstrapTime,
			"mipro_vs_simba_time_diff":      simbaTime - miproTime,
			"best_optimizer":                bestOptimizer,
			"best_score":                    bestScore,
		}
	}

	// Save results
	err := comparison.SaveResults(results, "go_comparison_results.json")
	if err != nil {
		log.Printf("Error saving results: %v", err)
	}

	// Print summary
	fmt.Println("\n=== dspy-go Optimizer Comparison Results ===")
	fmt.Printf("Dataset size: %v\n", results["dataset_size"])
	fmt.Printf("Model: %v\n", results["model"])
	fmt.Printf("Optimizer tested: %v\n", results["optimizer_tested"])

	if bootstrapResults, ok := results["bootstrap_fewshot"].(map[string]interface{}); ok {
		fmt.Printf("\nBootstrapFewShot:\n")
		fmt.Printf("  - Average score: %.3f\n", bootstrapResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", bootstrapResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(bootstrapResults["demonstrations"].([]map[string]interface{})))
	}

	if miproResults, ok := results["mipro"].(map[string]interface{}); ok {
		fmt.Printf("\nMIPRO:\n")
		fmt.Printf("  - Average score: %.3f\n", miproResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", miproResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(miproResults["demonstrations"].([]map[string]interface{})))
	}

	if simbaResults, ok := results["simba"].(map[string]interface{}); ok {
		fmt.Printf("\nSIMBA:\n")
		fmt.Printf("  - Average score: %.3f\n", simbaResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", simbaResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(simbaResults["demonstrations"].([]map[string]interface{})))
	}

	if comparisonResults, ok := results["comparison"].(map[string]interface{}); ok {
		fmt.Printf("\nBest optimizer: %v (%.3f)\n", comparisonResults["best_optimizer"], comparisonResults["best_score"])
	}
}

// Helper function for min.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
