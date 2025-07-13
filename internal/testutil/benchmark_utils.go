package testutil

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

// ConcurrencyTestMu protects concurrent access to global concurrency settings
// during benchmark tests to prevent race conditions.
var ConcurrencyTestMu sync.Mutex

// BenchmarkDataset represents different dataset sizes for benchmarking.
type BenchmarkDataset struct {
	Name     string
	Size     int
	Examples []core.Example
}

// BenchmarkMetrics tracks performance metrics during benchmarking.
type BenchmarkMetrics struct {
	CompilationTime float64
	AverageScore    float64
	TotalExamples   int
	TokensUsed      int
}

// CreateBenchmarkDatasets creates standardized datasets for benchmarking optimizers.
func CreateBenchmarkDatasets() map[string]BenchmarkDataset {
	datasets := make(map[string]BenchmarkDataset)

	// Question-Answer pairs for benchmarking
	baseQA := []struct {
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
		{"Who wrote Romeo and Juliet?", "William Shakespeare"},
		{"What is the chemical symbol for gold?", "Au"},
		{"How many continents are there?", "7"},
		{"What is the largest ocean?", "Pacific Ocean"},
		{"What is the smallest country in the world?", "Vatican City"},
		{"What is the longest river in the world?", "Nile River"},
		{"What is the highest mountain in the world?", "Mount Everest"},
		{"What is the currency of Japan?", "Yen"},
		{"What is the largest mammal?", "Blue whale"},
		{"What is the hottest planet in our solar system?", "Venus"},
		{"What is the fastest land animal?", "Cheetah"},
		{"What is the most abundant gas in Earth's atmosphere?", "Nitrogen"},
		{"What is the hardest natural substance?", "Diamond"},
		{"What is the capital of Australia?", "Canberra"},
		{"What is the smallest bone in the human body?", "Stapes"},
		{"What is the most spoken language in the world?", "Mandarin Chinese"},
		{"What is the largest desert in the world?", "Antarctic Desert"},
		{"What is the deepest ocean trench?", "Mariana Trench"},
		{"What is the most common blood type?", "O positive"},
		{"What is the study of earthquakes called?", "Seismology"},
	}

	// Create different sized datasets
	sizes := map[string]int{
		"tiny":   10,
		"small":  20,
		"medium": 50,
		"large":  100,
	}

	for name, size := range sizes {
		examples := make([]core.Example, size)
		for i := 0; i < size; i++ {
			qa := baseQA[i%len(baseQA)]
			examples[i] = core.Example{
				Inputs: map[string]interface{}{
					"question": qa.question,
					"prompt":   qa.question, // For MIPRO compatibility
				},
				Outputs: map[string]interface{}{
					"answer": qa.answer,
				},
			}
		}

		datasets[name] = BenchmarkDataset{
			Name:     fmt.Sprintf("%s_dataset", name),
			Size:     size,
			Examples: examples,
		}
	}

	return datasets
}

// CreateBenchmarkProgram creates a standard program for benchmarking optimizers.
// The predictor parameter should be a modules.Predict instance to avoid import cycles.
func CreateBenchmarkProgram(predictor core.Module) core.Program {
	return core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)
}

// BenchmarkAccuracyMetric provides a standard accuracy metric for benchmarking.
func BenchmarkAccuracyMetric(expected, actual map[string]interface{}) float64 {
	expectedAnswer, ok := expected["answer"].(string)
	if !ok {
		return 0.0
	}

	actualAnswer, ok := actual["answer"].(string)
	if !ok {
		return 0.0
	}

	// Simple substring matching for benchmarking
	expectedLower := strings.ToLower(strings.TrimSpace(expectedAnswer))
	actualLower := strings.ToLower(strings.TrimSpace(actualAnswer))

	// Bidirectional substring matching
	if strings.Contains(expectedLower, actualLower) || strings.Contains(actualLower, expectedLower) {
		return 1.0
	}

	return 0.0
}

// BenchmarkDatasetFromExamples converts examples to a dataset interface.
func BenchmarkDatasetFromExamples(examples []core.Example) core.Dataset {
	return datasets.NewSimpleDataset(examples)
}

// BenchmarkConfig represents common benchmark configuration.
type BenchmarkConfig struct {
	DatasetSize int
	Breadth     int
	Depth       int
	MaxSteps    int
	BatchSize   int
	NumTrials   int
	Temperature float64
	Timeout     int // seconds
}

// StandardBenchmarkConfigs provides predefined benchmark configurations.
func StandardBenchmarkConfigs() map[string]BenchmarkConfig {
	return map[string]BenchmarkConfig{
		"fast": {
			DatasetSize: 10,
			Breadth:     3,
			Depth:       2,
			MaxSteps:    3,
			BatchSize:   2,
			NumTrials:   3,
			Temperature: 1.0,
			Timeout:     30,
		},
		"standard": {
			DatasetSize: 20,
			Breadth:     5,
			Depth:       2,
			MaxSteps:    6,
			BatchSize:   4,
			NumTrials:   5,
			Temperature: 1.2,
			Timeout:     60,
		},
		"comprehensive": {
			DatasetSize: 50,
			Breadth:     8,
			Depth:       3,
			MaxSteps:    10,
			BatchSize:   8,
			NumTrials:   8,
			Temperature: 1.5,
			Timeout:     120,
		},
	}
}

// BenchmarkResult captures the results of a benchmark run.
type BenchmarkResult struct {
	OptimizerName   string
	Config          BenchmarkConfig
	CompilationTime float64 // seconds
	AverageScore    float64
	TotalExamples   int
	Success         bool
	Error           string
}

// LogBenchmarkResult logs benchmark results in a structured format.
func LogBenchmarkResult(result BenchmarkResult) {
	status := "SUCCESS"
	if !result.Success {
		status = "FAILED"
	}

	fmt.Printf("BENCHMARK [%s] %s: %.3fs, score: %.3f, examples: %d\n",
		status,
		result.OptimizerName,
		result.CompilationTime,
		result.AverageScore,
		result.TotalExamples,
	)

	if result.Error != "" {
		fmt.Printf("  Error: %s\n", result.Error)
	}
}
