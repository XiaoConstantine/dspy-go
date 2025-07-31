package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
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


// OptimizerComparison handles the comparison between Go and Python implementations.
type OptimizerComparison struct {
	ModelName string
	LLM       core.LLM

	BootstrapMetrics *ComparisonMetrics
	MIPROMetrics     *ComparisonMetrics
	SIMBAMetrics     *ComparisonMetrics
	CoProMetrics     *ComparisonMetrics
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
		CoProMetrics:     &ComparisonMetrics{},
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

	// Simple substring matching for demo purposes - match Python DSPy behavior
	if len(expected) > 0 && len(predicted) > 0 {
		expectedLower := strings.ToLower(strings.TrimSpace(expected))
		predictedLower := strings.ToLower(strings.TrimSpace(predicted))

		// Bidirectional substring matching: expected in predicted OR predicted in expected
		if strings.Contains(expectedLower, predictedLower) || strings.Contains(predictedLower, expectedLower) {
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

	// Create dataset interface for BootstrapFewShot
	trainDataset := datasets.NewSimpleDataset(trainExamples)

	// Define metric function for BootstrapFewShot
	metricFunc := func(expected, actual map[string]interface{}) float64 {
		return oc.AccuracyMetric(expected, actual, ctx)
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
	trainDataset := datasets.NewSimpleDataset(trainExamples)

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

func (oc *OptimizerComparison) TestCOPRO(ctx context.Context, dataset []core.Example, breadth int, depth int) (core.Program, map[string]interface{}) {
	log.Println("Testing COPRO optimizer")

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

	// Create dataset interface for COPRO
	trainDataset := datasets.NewSimpleDataset(trainExamples)

	// Define metric function for COPRO
	metricFunc := func(expected, actual map[string]interface{}) float64 {
		return oc.AccuracyMetric(expected, actual, ctx)
	}

	// Create COPRO optimizer (prompt optimization algorithm)
	coproOptimizer := optimizers.NewCOPRO(
		metricFunc,
		optimizers.WithBreadth(breadth),        // Number of prompt candidates
		optimizers.WithDepth(depth),            // Refinement iterations
		optimizers.WithInitTemperature(1.2),    // Exploration randomness
		optimizers.WithTrackStats(false),
	)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := coproOptimizer.Compile(ctx, program, trainDataset, metricFunc)
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
		oc.CoProMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":        "COPRO",
		"compilation_time": compilationTime,
		"average_score":    avgScore,
		"total_examples":   len(valset),
		"breadth":          breadth,
		"depth":            depth,
		"demonstrations":   oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("COPRO results: %+v", results)
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

	// Create optimizer with fast mode for better compatibility test performance
	optimizer := optimizers.NewSIMBA(
		optimizers.WithSIMBABatchSize(batchSize),
		optimizers.WithSIMBAMaxSteps(maxSteps),
		optimizers.WithSIMBANumCandidates(4),
		optimizers.WithSamplingTemperature(0.2),
		optimizers.WithFastMode(true), // Enable fast mode for compatibility tests
	)

	// Create dataset interface for SIMBA
	trainDataset := datasets.NewSimpleDataset(trainExamples)

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
	var optimizer = flag.String("optimizer", "all", "Optimizer to test: bootstrap, mipro, simba, copro, or all")
	var datasetSize = flag.Int("dataset-size", 20, "Dataset size for testing")
	var enableCache = flag.Bool("enable-cache", false, "Enable caching for performance comparison")
	var cacheType = flag.String("cache-type", "memory", "Cache type: memory or sqlite")
	var withoutCache = flag.Bool("without-cache", false, "Run without cache for baseline comparison")
	flag.Parse()

	ctx := context.Background()

	// Configure caching based on flags
	if *enableCache && !*withoutCache {
		fmt.Printf("Enabling %s cache for performance testing...\n", *cacheType)
		// Set environment variables to enable caching
		os.Setenv("DSPY_CACHE_ENABLED", "true")
		os.Setenv("DSPY_CACHE_TYPE", *cacheType)
		os.Setenv("DSPY_CACHE_TTL", "1h")

		if *cacheType == "sqlite" {
			homeDir, _ := os.UserHomeDir()
			cacheDir := filepath.Join(homeDir, ".dspy-go")
			if err := os.MkdirAll(cacheDir, 0755); err != nil {
				log.Printf("Warning: could not create cache directory: %v", err)
			}
			os.Setenv("DSPY_CACHE_PATH", filepath.Join(cacheDir, "cache.db"))
		} else {
			os.Setenv("DSPY_CACHE_MAX_SIZE", "100MB")
		}
	} else if *withoutCache {
		fmt.Println("Running without cache for baseline comparison...")
		os.Setenv("DSPY_CACHE_ENABLED", "false")
	}

	// Initialize comparison
	comparison := NewOptimizerComparison("gemini-2.0-flash")

	// Log cache status
	if *enableCache && !*withoutCache {
		fmt.Printf("Cache enabled: %s\n", *cacheType)
		// Clear cache at start to ensure clean test
		if err := cache.ClearGlobalCache(ctx); err != nil {
			log.Printf("Warning: Could not clear cache: %v", err)
		}
	}

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

	if *optimizer == "copro" || *optimizer == "all" {
		fmt.Println("Testing COPRO...")
		_, coproResults := comparison.TestCOPRO(ctx, dataset, 5, 2) // breadth=5, depth=2
		results["copro"] = coproResults
	}

	// Add comparison section if testing multiple optimizers
	if *optimizer == "all" {
		bootstrapResults := results["bootstrap_fewshot"].(map[string]interface{})
		miproResults := results["mipro"].(map[string]interface{})
		simbaResults := results["simba"].(map[string]interface{})
		coproResults := results["copro"].(map[string]interface{})

		bootstrapScore := bootstrapResults["average_score"].(float64)
		miproScore := miproResults["average_score"].(float64)
		simbaScore := simbaResults["average_score"].(float64)
		coproScore := coproResults["average_score"].(float64)
		bootstrapTime := bootstrapResults["compilation_time"].(float64)
		miproTime := miproResults["compilation_time"].(float64)
		simbaTime := simbaResults["compilation_time"].(float64)
		coproTime := coproResults["compilation_time"].(float64)

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
		if coproScore > bestScore {
			bestOptimizer = "COPRO"
			bestScore = coproScore
		}

		results["comparison"] = map[string]interface{}{
			"bootstrap_vs_mipro_score_diff": miproScore - bootstrapScore,
			"bootstrap_vs_simba_score_diff": simbaScore - bootstrapScore,
			"bootstrap_vs_copro_score_diff": coproScore - bootstrapScore,
			"mipro_vs_simba_score_diff":     simbaScore - miproScore,
			"mipro_vs_copro_score_diff":     coproScore - miproScore,
			"simba_vs_copro_score_diff":     coproScore - simbaScore,
			"bootstrap_vs_mipro_time_diff":  miproTime - bootstrapTime,
			"bootstrap_vs_simba_time_diff":  simbaTime - bootstrapTime,
			"bootstrap_vs_copro_time_diff":  coproTime - bootstrapTime,
			"mipro_vs_simba_time_diff":      simbaTime - miproTime,
			"mipro_vs_copro_time_diff":      coproTime - miproTime,
			"simba_vs_copro_time_diff":      coproTime - simbaTime,
			"best_optimizer":                bestOptimizer,
			"best_score":                    bestScore,
		}
	}

	// Add cache statistics if caching was enabled
	if *enableCache && !*withoutCache {
		cacheStats := cache.GetGlobalCacheStats()
		results["cache_stats"] = map[string]interface{}{
			"hits":     cacheStats.Hits,
			"misses":   cacheStats.Misses,
			"sets":     cacheStats.Sets,
			"size":     cacheStats.Size,
			"hit_rate": float64(cacheStats.Hits) / float64(cacheStats.Hits+cacheStats.Misses),
		}
		fmt.Printf("\nCache Statistics:\n")
		fmt.Printf("  - Hits: %d\n", cacheStats.Hits)
		fmt.Printf("  - Misses: %d\n", cacheStats.Misses)
		fmt.Printf("  - Hit Rate: %.2f%%\n", float64(cacheStats.Hits)/float64(cacheStats.Hits+cacheStats.Misses)*100)
		fmt.Printf("  - Cache Size: %d bytes\n", cacheStats.Size)
	}

	// Save results
	filename := "go_comparison_results.json"
	if *enableCache && !*withoutCache {
		filename = "go_comparison_results_cached.json"
	} else if *withoutCache {
		filename = "go_comparison_results_nocache.json"
	}

	err := comparison.SaveResults(results, filename)
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

	if coproResults, ok := results["copro"].(map[string]interface{}); ok {
		fmt.Printf("\nCOPRO:\n")
		if avgScore, exists := coproResults["average_score"]; exists && avgScore != nil {
			fmt.Printf("  - Average score: %.3f\n", avgScore)
		} else {
			fmt.Printf("  - Average score: N/A (compilation failed)\n")
		}
		if compTime, exists := coproResults["compilation_time"]; exists && compTime != nil {
			fmt.Printf("  - Compilation time: %.2fs\n", compTime)
		} else {
			fmt.Printf("  - Compilation time: N/A\n")
		}
		if breadth, exists := coproResults["breadth"]; exists && breadth != nil {
			fmt.Printf("  - Breadth: %v\n", breadth)
		}
		if depth, exists := coproResults["depth"]; exists && depth != nil {
			fmt.Printf("  - Depth: %v\n", depth)
		}
		if demos, exists := coproResults["demonstrations"]; exists && demos != nil {
			if demoSlice, ok := demos.([]map[string]interface{}); ok {
				fmt.Printf("  - Demonstrations: %d\n", len(demoSlice))
			} else {
				fmt.Printf("  - Demonstrations: 0 (error extracting)\n")
			}
		} else {
			fmt.Printf("  - Demonstrations: 0\n")
		}
		if err, exists := coproResults["error"]; exists {
			fmt.Printf("  - Error: %s\n", err)
		}
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
