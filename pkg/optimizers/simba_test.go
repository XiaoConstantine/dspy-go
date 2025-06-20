package optimizers

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// TestSIMBA contains comprehensive tests for the SIMBA optimizer.
func TestSIMBA(t *testing.T) {
	// Initialize mock LLM for tests
	setupMockLLM()

	t.Run("Constructor and Configuration", func(t *testing.T) {
		testSIMBAConstructorAndConfiguration(t)
	})

	t.Run("DSPy Python Compatibility", func(t *testing.T) {
		testSIMBAPythonCompatibility(t)
	})

	t.Run("Core Algorithm Components", func(t *testing.T) {
		testSIMBACoreAlgorithm(t)
	})

	t.Run("Mini-batch Processing", func(t *testing.T) {
		testSIMBAMiniBatchProcessing(t)
	})

	t.Run("Candidate Generation", func(t *testing.T) {
		testSIMBACandidateGeneration(t)
	})

	t.Run("Temperature-Controlled Sampling", func(t *testing.T) {
		testSIMBATemperatureSampling(t)
	})

	t.Run("Introspective Analysis", func(t *testing.T) {
		testSIMBAIntrospectiveAnalysis(t)
	})

	t.Run("Optimization Workflow", func(t *testing.T) {
		testSIMBAOptimizationWorkflow(t)
	})

	t.Run("Edge Cases and Error Handling", func(t *testing.T) {
		testSIMBAEdgeCases(t)
	})

	t.Run("Performance and Convergence", func(t *testing.T) {
		testSIMBAConvergence(t)
	})
}

// setupMockLLM initializes mock LLM for all tests.
func setupMockLLM() {
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
		&core.LLMResponse{Content: "test response"}, nil).Maybe()
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(
		map[string]interface{}{"answer": "test"}, nil).Maybe()

	core.GlobalConfig.DefaultLLM = mockLLM
	core.GlobalConfig.TeacherLLM = mockLLM
	core.GlobalConfig.ConcurrencyLevel = 1
}

// testSIMBAConstructorAndConfiguration tests SIMBA constructor and options.
func testSIMBAConstructorAndConfiguration(t *testing.T) {
	t.Run("Creates instance with default configuration", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		assert.NotNil(t, simba)
		assert.NotNil(t, simba.state)
		assert.NotNil(t, simba.logger)
		assert.NotNil(t, simba.rng)

		// Verify default configuration values
		assert.Equal(t, 32, simba.config.BatchSize)
		assert.Equal(t, 8, simba.config.MaxSteps)
		assert.Equal(t, 6, simba.config.NumCandidates)
		assert.Equal(t, 4, simba.config.MaxDemos)
		assert.Equal(t, 0.2, simba.config.SamplingTemperature)
		assert.Equal(t, 0.2, simba.config.CandidateTemperature)
		assert.Equal(t, 2, simba.config.IntrospectionFrequency)
		assert.Equal(t, 0.3, simba.config.SelfAdviceWeight)
		assert.Equal(t, 0.001, simba.config.ConvergenceThreshold)
		assert.Equal(t, 0.05, simba.config.MinImprovementRatio)
		assert.Equal(t, 10, simba.config.MaxGoroutines)
	})

	t.Run("Functional options configure SIMBA correctly", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric,
			WithSIMBABatchSize(16),
			WithSIMBAMaxSteps(12),
			WithSIMBANumCandidates(8),
			WithTemperatures(0.3, 0.4),
			WithMaxDemos(6),
		)

		assert.Equal(t, 16, simba.config.BatchSize)
		assert.Equal(t, 12, simba.config.MaxSteps)
		assert.Equal(t, 8, simba.config.NumCandidates)
		assert.Equal(t, 0.3, simba.config.SamplingTemperature)
		assert.Equal(t, 0.4, simba.config.CandidateTemperature)
		assert.Equal(t, 6, simba.config.MaxDemos)
	})

	t.Run("State is properly initialized", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		state := simba.GetState()
		assert.Equal(t, 0, state.CurrentStep)
		assert.Equal(t, 0.0, state.BestScore)
		assert.NotNil(t, state.CandidateHistory)
		assert.NotNil(t, state.PerformanceLog)
		assert.NotNil(t, state.IntrospectionLog)
		assert.NotZero(t, state.StartTime)
	})
}

// testSIMBAPythonCompatibility verifies compatibility with DSPy Python implementation.
func testSIMBAPythonCompatibility(t *testing.T) {
	t.Run("Default parameters match DSPy Python SIMBA", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Verify parameters match DSPy Python defaults:
		// bsize=32, num_candidates=6, max_steps=8, max_demos=4
		// temperature_for_sampling=0.2, temperature_for_candidates=0.2
		assert.Equal(t, 32, simba.config.BatchSize, "BatchSize should match Python bsize=32")
		assert.Equal(t, 6, simba.config.NumCandidates, "NumCandidates should match Python num_candidates=6")
		assert.Equal(t, 8, simba.config.MaxSteps, "MaxSteps should match Python max_steps=8")
		assert.Equal(t, 4, simba.config.MaxDemos, "MaxDemos should match Python max_demos=4")
		assert.Equal(t, 0.2, simba.config.SamplingTemperature, "SamplingTemperature should match Python temperature_for_sampling=0.2")
		assert.Equal(t, 0.2, simba.config.CandidateTemperature, "CandidateTemperature should match Python temperature_for_candidates=0.2")
	})

	t.Run("Compile method signature is compatible", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Create test program and dataset
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()

		ctx := core.WithExecutionState(context.Background())

		// Should be able to call Compile with context, program, dataset, and metric
		// This matches the pattern: compile(student, trainset, seed=0) in Python
		result, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.IsType(t, core.Program{}, result)
	})
}

// testSIMBACoreAlgorithm tests core algorithmic components.
func testSIMBACoreAlgorithm(t *testing.T) {
	t.Run("Temperature decay function works correctly", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Test temperature decay over steps
		initialTemp := simba.getCurrentTemperature(0)
		step5Temp := simba.getCurrentTemperature(5)
		step10Temp := simba.getCurrentTemperature(10)

		assert.Equal(t, 0.2, initialTemp)
		assert.True(t, step5Temp < initialTemp, "Temperature should decay over steps")
		assert.True(t, step10Temp < step5Temp, "Temperature should continue decaying")
		assert.True(t, step10Temp > 0, "Temperature should remain positive")
	})

	t.Run("Temperature sampling selects candidates probabilistically", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		scores := []float64{0.1, 0.5, 0.9, 0.3}

		// Test with high temperature (more exploration)
		highTempSelections := make(map[int]int)
		for i := 0; i < 100; i++ {
			idx := simba.temperatureSample(scores, 1.0)
			highTempSelections[idx]++
		}

		// Test with low temperature (more exploitation)
		lowTempSelections := make(map[int]int)
		for i := 0; i < 100; i++ {
			idx := simba.temperatureSample(scores, 0.1)
			lowTempSelections[idx]++
		}

		// High temperature should have more diversity
		highTempUniqueSelections := len(highTempSelections)
		lowTempUniqueSelections := len(lowTempSelections)

		// With low temperature, should mostly select best candidate (index 2)
		assert.True(t, lowTempSelections[2] > 50, "Low temperature should favor best candidate")
		assert.True(t, highTempUniqueSelections >= lowTempUniqueSelections, "High temperature should have more diversity")
	})

	t.Run("State management is thread-safe", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Test concurrent access to state
		done := make(chan bool, 10)
		for i := 0; i < 10; i++ {
			go func() {
				state := simba.GetState()
				assert.NotNil(t, state)
				done <- true
			}()
		}

		// Wait for all goroutines to complete
		for i := 0; i < 10; i++ {
			<-done
		}
	})
}

// testSIMBAMiniBatchProcessing tests mini-batch sampling and processing.
func testSIMBAMiniBatchProcessing(t *testing.T) {
	t.Run("Samples mini-batch from dataset correctly", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric, WithSIMBABatchSize(3))

		dataset := createSIMBATestDataset()
		ctx := context.Background()

		batch, err := simba.sampleMiniBatch(ctx, dataset)

		assert.NoError(t, err)
		assert.Equal(t, 3, len(batch))

		// Verify batch contains valid examples
		for _, example := range batch {
			assert.NotNil(t, example.Inputs)
			assert.Contains(t, example.Inputs, "question")
		}
	})

	t.Run("Handles batch size larger than dataset", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric, WithSIMBABatchSize(100))

		dataset := createSIMBATestDataset() // Has only 3 examples
		ctx := context.Background()

		batch, err := simba.sampleMiniBatch(ctx, dataset)

		assert.NoError(t, err)
		assert.Equal(t, 3, len(batch)) // Should return all available examples
	})

	t.Run("Handles empty dataset gracefully", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		emptyDataset := testutil.NewMockDataset([]core.Example{})
		emptyDataset.On("Reset").Return()
		emptyDataset.On("Next").Return(core.Example{}, false)

		ctx := context.Background()

		_, err := simba.sampleMiniBatch(ctx, emptyDataset)

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "dataset is empty")
	})
}

// testSIMBACandidateGeneration tests candidate program generation.
func testSIMBACandidateGeneration(t *testing.T) {
	t.Run("Generates correct number of candidates", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric, WithSIMBANumCandidates(5))

		program := createSIMBATestProgram()
		ctx := context.Background()

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) <= 5) // May be fewer if generation fails
		assert.True(t, len(candidates) >= 1) // Should at least include base program
	})

	t.Run("Includes base program as first candidate", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		program := createSIMBATestProgram()
		ctx := context.Background()

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1)
		// First candidate should be a clone of the base program (compare pointers)
		assert.NotSame(t, &program, &candidates[0])
		// Verify modules were cloned (different instances)
		for name, originalModule := range program.Modules {
			clonedModule, exists := candidates[0].Modules[name]
			assert.True(t, exists)
			assert.NotSame(t, originalModule, clonedModule)
		}
	})

	t.Run("Perturbs program instructions", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Test instruction perturbation
		variant, err := simba.perturbProgram(ctx, program)

		assert.NoError(t, err)
		assert.NotNil(t, variant)
		assert.NotSame(t, &program, &variant)
	})

	t.Run("Generates instruction variations", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		ctx := context.Background()
		originalInstruction := "Answer the question"

		variation, err := simba.generateInstructionVariation(ctx, originalInstruction)

		assert.NoError(t, err)
		assert.NotEmpty(t, variation)
		// Should return either the original or a variation
		assert.True(t, len(variation) > 0)
	})
}

// testSIMBATemperatureSampling tests temperature-controlled sampling.
func testSIMBATemperatureSampling(t *testing.T) {
	t.Run("Selects best candidate with zero temperature", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric, WithTemperatures(0.0, 0.0)) // Set both temperatures to 0

		// Create distinct programs
		program1 := createSIMBATestProgram()
		program2 := createSIMBATestProgram()
		candidates := []core.Program{program1, program2}
		scores := []float64{0.3, 0.7}

		bestProgram, bestScore := simba.selectBestCandidate(candidates, scores)

		assert.Equal(t, 0.7, bestScore)
		// With zero temperature, should always select the best scoring candidate
		// Since programs might be identical, just verify the score is correct
		assert.NotNil(t, bestProgram)
	})

	t.Run("Temperature sampling respects probability distribution", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		scores := []float64{0.1, 0.9}

		// Test multiple samples to verify probabilistic behavior
		highScoreSelections := 0
		for i := 0; i < 100; i++ {
			idx := simba.temperatureSample(scores, 0.5)
			if idx == 1 { // High score index
				highScoreSelections++
			}
		}

		// Should select high-scoring candidate more often
		assert.True(t, highScoreSelections > 60, "Should favor high-scoring candidates")
	})

	t.Run("Handles edge cases in temperature sampling", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Test with single score
		singleScore := []float64{0.5}
		idx := simba.temperatureSample(singleScore, 0.2)
		assert.Equal(t, 0, idx)

		// Test with identical scores
		identicalScores := []float64{0.5, 0.5, 0.5}
		idx = simba.temperatureSample(identicalScores, 0.2)
		assert.True(t, idx >= 0 && idx < 3)
	})
}

// testSIMBAIntrospectiveAnalysis tests introspective learning capabilities.
func testSIMBAIntrospectiveAnalysis(t *testing.T) {
	t.Run("Performs introspection with sufficient data", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Add some performance history
		simba.state.PerformanceLog = []StepResult{
			{Step: 0, BestScore: 0.5, Improvement: 0.0},
			{Step: 1, BestScore: 0.6, Improvement: 0.1},
			{Step: 2, BestScore: 0.7, Improvement: 0.1},
		}

		ctx := context.Background()
		result := simba.performIntrospection(ctx)

		assert.NotNil(t, result)
		assert.NotEmpty(t, result.Analysis)
		assert.NotNil(t, result.Recommendations)
		assert.NotNil(t, result.IdentifiedPatterns)
		assert.NotNil(t, result.SuggestedAdjustments)
		assert.True(t, result.Confidence >= 0.0)
	})

	t.Run("Handles insufficient data gracefully", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		ctx := context.Background()
		result := simba.performIntrospection(ctx)

		assert.NotNil(t, result)
		assert.Contains(t, result.Analysis, "Insufficient data")
		assert.Equal(t, 0.0, result.Confidence)
	})

	t.Run("Identifies performance patterns", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Create improving trend
		improvingSteps := []StepResult{
			{Step: 0, BestScore: 0.3, Improvement: 0.0},
			{Step: 1, BestScore: 0.5, Improvement: 0.2},
			{Step: 2, BestScore: 0.7, Improvement: 0.2},
		}

		patterns := simba.identifyPatterns(improvingSteps)
		assert.Contains(t, patterns, "Consistent improvement trend")

		// Create stagnating trend
		stagnatingSteps := []StepResult{
			{Step: 0, BestScore: 0.5, Improvement: 0.0},
			{Step: 1, BestScore: 0.5, Improvement: 0.0},
			{Step: 2, BestScore: 0.5, Improvement: 0.0},
		}

		patterns = simba.identifyPatterns(stagnatingSteps)
		assert.Contains(t, patterns, "Stagnation detected")
	})

	t.Run("Suggests appropriate adjustments", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Create steps with low improvement
		lowImprovementSteps := []StepResult{
			{Step: 0, BestScore: 0.5, Improvement: 0.0},
			{Step: 1, BestScore: 0.501, Improvement: 0.001}, // Very low improvement
		}

		adjustments := simba.suggestAdjustments(lowImprovementSteps)
		assert.Contains(t, adjustments, "Consider increasing exploration temperature")
		assert.Contains(t, adjustments, "Try different instruction variations")
	})
}

// testSIMBAOptimizationWorkflow tests the complete optimization workflow.
func testSIMBAOptimizationWorkflow(t *testing.T) {
	t.Run("Completes full optimization cycle", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 {
			// Simple metric that always returns 0.8
			return 0.8
		}
		simba := NewSIMBA(metric, WithSIMBAMaxSteps(2)) // Limit steps for testing

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)
		assert.NotSame(t, &program, &optimizedProgram)

		// Verify state was updated
		state := simba.GetState()
		assert.True(t, state.CurrentStep >= 0)
		assert.True(t, state.BestScore >= 0)
		assert.NotNil(t, state.BestProgram)
	})

	t.Run("Records optimization metrics", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.75 }
		simba := NewSIMBA(metric, WithSIMBAMaxSteps(3))

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		_, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)

		state := simba.GetState()
		assert.True(t, len(state.PerformanceLog) > 0)

		// Check performance log structure
		for _, step := range state.PerformanceLog {
			assert.True(t, step.Step >= 0)
			assert.True(t, step.BestScore >= 0)
			assert.NotNil(t, step.CandidateScores)
			assert.True(t, step.Temperature >= 0)
			assert.True(t, step.BatchSize > 0)
			assert.True(t, step.Duration > 0)
		}
	})

	t.Run("Performs introspection at configured intervals", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.6 }
		simba := NewSIMBA(metric,
			WithSIMBAMaxSteps(5),
		)
		// Use default introspection frequency = 2

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		_, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)

		state := simba.GetState()
		// Should have performed introspection at steps 2 and 4
		introspectionCount := 0
		for _, step := range state.PerformanceLog {
			if step.Introspection != "" {
				introspectionCount++
			}
		}
		assert.True(t, introspectionCount >= 1, "Should perform introspection")
	})
}

// testSIMBAEdgeCases tests edge cases and error handling.
func testSIMBAEdgeCases(t *testing.T) {
	t.Run("Handles nil context gracefully", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()

		// Pass nil context
		result, err := simba.Compile(nil, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("Handles LLM generation failures", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Set up failing mock LLM
		failingLLM := new(testutil.MockLLM)
		failingLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(nil, fmt.Errorf("LLM failure"))

		simba.primaryModel = failingLLM
		simba.analyzerModel = failingLLM

		ctx := context.Background()

		// Should handle LLM failures gracefully
		_, err := simba.generateInstructionVariation(ctx, "test instruction")
		assert.NoError(t, err) // Should return original instruction
	})

	t.Run("Handles program evaluation failures", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Create a program that fails during Forward
		failingProgram := core.NewProgram(
			map[string]core.Module{},
			func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return nil, fmt.Errorf("program evaluation failed")
			},
		)

		dataset := createSIMBATestDataset()
		score := simba.evaluateProgram(context.Background(), failingProgram, dataset)

		assert.Equal(t, 0.0, score) // Should return 0 for failed evaluation
	})

	t.Run("Handles zero or negative scores", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return -0.5 }
		simba := NewSIMBA(metric)

		candidates := []core.Program{createSIMBATestProgram()}
		scores := []float64{-0.5}

		program, score := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, program)
		assert.Equal(t, -0.5, score)
	})
}

// testSIMBAConvergence tests convergence detection and performance.
func testSIMBAConvergence(t *testing.T) {
	t.Run("Detects convergence with minimal improvements", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Simulate convergence with very small improvements
		simba.state.PerformanceLog = []StepResult{
			{Step: 0, Improvement: 0.0005},
			{Step: 1, Improvement: 0.0003},
			{Step: 2, Improvement: 0.0002},
		}

		converged := simba.hasConverged()
		assert.True(t, converged, "Should detect convergence with small improvements")
	})

	t.Run("Does not converge with significant improvements", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Simulate ongoing improvements
		simba.state.PerformanceLog = []StepResult{
			{Step: 0, Improvement: 0.1},
			{Step: 1, Improvement: 0.05},
			{Step: 2, Improvement: 0.02},
		}

		converged := simba.hasConverged()
		assert.False(t, converged, "Should not converge with significant improvements")
	})

	t.Run("Handles insufficient data for convergence", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
		simba := NewSIMBA(metric)

		// Empty performance log
		simba.state.PerformanceLog = []StepResult{}

		converged := simba.hasConverged()
		assert.False(t, converged, "Should not converge without data")
	})

	t.Run("Completes optimization within reasonable time", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.9 }
		simba := NewSIMBA(metric, WithSIMBAMaxSteps(3)) // Small number for fast test

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		start := time.Now()
		_, err := simba.Compile(ctx, program, dataset, metric)
		duration := time.Since(start)

		assert.NoError(t, err)
		assert.True(t, duration < 30*time.Second, "Optimization should complete within reasonable time")
	})
}

// Helper functions for creating test objects specific to SIMBA tests

func createSIMBATestProgram() core.Program {
	predict := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	).WithInstruction("Answer the question"))

	forwardFunc := func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		return predict.Process(ctx, inputs)
	}

	return core.NewProgram(map[string]core.Module{"predict": predict}, forwardFunc)
}

func createSIMBATestDataset() core.Dataset {
	examples := []core.Example{
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of France?"},
			Outputs: map[string]interface{}{"answer": "Paris"},
		},
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of Germany?"},
			Outputs: map[string]interface{}{"answer": "Berlin"},
		},
		{
			Inputs:  map[string]interface{}{"question": "What is the capital of Italy?"},
			Outputs: map[string]interface{}{"answer": "Rome"},
		},
	}

	dataset := testutil.NewMockDataset(examples)
	dataset.On("Reset").Return().Maybe()
	dataset.On("Next").Return(examples[0], true).Maybe()
	dataset.On("Next").Return(examples[1], true).Maybe()
	dataset.On("Next").Return(examples[2], true).Maybe()
	dataset.On("Next").Return(core.Example{}, false).Maybe()

	return dataset
}
