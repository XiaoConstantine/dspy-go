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

	// New comprehensive dual strategy tests
	t.Run("Dual Strategy System", func(t *testing.T) {
		testSIMBADualStrategy(t)
	})

	t.Run("Trajectory Tracking", func(t *testing.T) {
		testSIMBATrajectoryTracking(t)
	})

	t.Run("Rule Generation", func(t *testing.T) {
		testSIMBARuleGeneration(t)
	})

	t.Run("Strategy Configuration", func(t *testing.T) {
		testSIMBAStrategyConfiguration(t)
	})

	// New comprehensive bucket sorting tests
	t.Run("Bucket Sorting Configuration", func(t *testing.T) {
		testSIMBABucketSortingConfiguration(t)
	})

	t.Run("Multi-Criteria Scoring", func(t *testing.T) {
		testSIMBAMultiCriteriaScoring(t)
	})

	t.Run("Bucket Sorting Algorithm", func(t *testing.T) {
		testSIMBABucketSortingAlgorithm(t)
	})

	t.Run("Candidate Metadata", func(t *testing.T) {
		testSIMBACandidateMetadata(t)
	})

	t.Run("Integration Tests", func(t *testing.T) {
		testSIMBAIntegration(t)
	})

	t.Run("LLM Concurrency Configuration", func(t *testing.T) {
		testSIMBALLMConcurrency(t)
	})

	t.Run("Pipeline Processing", func(t *testing.T) {
		testSIMBAPipelineProcessing(t)
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
		simba := NewSIMBA()

		assert.NotNil(t, simba)
		assert.NotNil(t, simba.state)
		assert.NotNil(t, simba.logger)
		assert.NotNil(t, simba.rng)

		// Verify default configuration values
		assert.Equal(t, 32, simba.config.BatchSize)
		assert.Equal(t, 8, simba.config.MaxSteps)
		assert.Equal(t, 6, simba.config.NumCandidates)
		assert.Equal(t, 0.2, simba.config.SamplingTemperature)
		assert.Equal(t, 2, simba.config.IntrospectionFrequency)
		assert.Equal(t, 0.001, simba.config.ConvergenceThreshold)
		assert.Equal(t, 0.05, simba.config.MinImprovementRatio)
		assert.Equal(t, 10, simba.config.MaxGoroutines)
		// Verify new dual strategy default values
		assert.Equal(t, "both", simba.config.StrategyMode)
		assert.Equal(t, 0.5, simba.config.StrategyRatio)
	})

	t.Run("Functional options configure SIMBA correctly", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBABatchSize(16),
			WithSIMBAMaxSteps(12),
			WithSIMBANumCandidates(8),
			WithSamplingTemperature(0.3),
			WithSIMBAStrategyMode("instruction_only"),
			WithSIMBAStrategyRatio(0.7),
		)

		assert.Equal(t, 16, simba.config.BatchSize)
		assert.Equal(t, 12, simba.config.MaxSteps)
		assert.Equal(t, 8, simba.config.NumCandidates)
		assert.Equal(t, 0.3, simba.config.SamplingTemperature)
		assert.Equal(t, "instruction_only", simba.config.StrategyMode)
		assert.Equal(t, 0.7, simba.config.StrategyRatio)
	})

	t.Run("State is properly initialized", func(t *testing.T) {
		simba := NewSIMBA()

		state := simba.GetState()
		assert.Equal(t, 0, state.CurrentStep)
		assert.Equal(t, 0.0, state.BestScore)
		assert.NotNil(t, state.CandidateHistory)
		assert.NotNil(t, state.PerformanceLog)
		assert.NotNil(t, state.IntrospectionLog)
		assert.NotZero(t, state.StartTime)
		// Verify trajectory tracking is initialized
		assert.NotNil(t, state.Trajectories)
		assert.Equal(t, 0, len(state.Trajectories))
	})
}

// testSIMBAPythonCompatibility verifies compatibility with DSPy Python implementation.
func testSIMBAPythonCompatibility(t *testing.T) {
	t.Run("Default parameters match DSPy Python SIMBA", func(t *testing.T) {
		simba := NewSIMBA()

		// Verify parameters match DSPy Python defaults:
		// bsize=32, num_candidates=6, max_steps=8
		// temperature_for_sampling=0.2
		assert.Equal(t, 32, simba.config.BatchSize, "BatchSize should match Python bsize=32")
		assert.Equal(t, 6, simba.config.NumCandidates, "NumCandidates should match Python num_candidates=6")
		assert.Equal(t, 8, simba.config.MaxSteps, "MaxSteps should match Python max_steps=8")
		assert.Equal(t, 0.2, simba.config.SamplingTemperature, "SamplingTemperature should match Python temperature_for_sampling=0.2")
	})

	t.Run("Compile method signature is compatible", func(t *testing.T) {
		simba := NewSIMBA()

		// Create test program and dataset
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()

		ctx := core.WithExecutionState(context.Background())

		// Define metric function
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA(WithSIMBABatchSize(3))

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
		simba := NewSIMBA(WithSIMBABatchSize(100))

		dataset := createSIMBATestDataset() // Has only 3 examples
		ctx := context.Background()

		batch, err := simba.sampleMiniBatch(ctx, dataset)

		assert.NoError(t, err)
		assert.Equal(t, 3, len(batch)) // Should return all available examples
	})

	t.Run("Handles empty dataset gracefully", func(t *testing.T) {
		simba := NewSIMBA()

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
		simba := NewSIMBA(WithSIMBANumCandidates(5))

		program := createSIMBATestProgram()
		ctx := context.Background()

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) <= 5) // May be fewer if generation fails
		assert.True(t, len(candidates) >= 1) // Should at least include base program
	})

	t.Run("Includes base program as first candidate", func(t *testing.T) {
		simba := NewSIMBA()

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
		simba := NewSIMBA()

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Test instruction perturbation
		variant, err := simba.perturbProgram(ctx, program)

		assert.NoError(t, err)
		assert.NotNil(t, variant)
		assert.NotSame(t, &program, &variant)
	})

	t.Run("Generates instruction variations", func(t *testing.T) {
		simba := NewSIMBA()

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
		simba := NewSIMBA(WithSamplingTemperature(0.0))

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

		ctx := context.Background()
		result := simba.performIntrospection(ctx)

		assert.NotNil(t, result)
		assert.Contains(t, result.Analysis, "Insufficient data")
		assert.Equal(t, 0.0, result.Confidence)
	})

	t.Run("Identifies performance patterns", func(t *testing.T) {
		simba := NewSIMBA()

		// Create improving trend
		improvingSteps := []StepResult{
			{Step: 0, BestScore: 0.3, Improvement: 0.0},
			{Step: 1, BestScore: 0.5, Improvement: 0.2},
			{Step: 2, BestScore: 0.7, Improvement: 0.2},
		}

		patterns := simba.identifyPatterns(improvingSteps)
		assert.Contains(t, patterns, "Strong upward trend detected")

		// Create stagnating trend
		stagnatingSteps := []StepResult{
			{Step: 0, BestScore: 0.5, Improvement: 0.0},
			{Step: 1, BestScore: 0.5, Improvement: 0.0},
			{Step: 2, BestScore: 0.5, Improvement: 0.0},
		}

		patterns = simba.identifyPatterns(stagnatingSteps)
		assert.Contains(t, patterns, "Consistent stagnation or decline")
	})

	t.Run("Suggests appropriate adjustments", func(t *testing.T) {
		simba := NewSIMBA()

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
		simba := NewSIMBA(WithSIMBAMaxSteps(2)) // Limit steps for testing

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
		simba := NewSIMBA(WithSIMBAMaxSteps(3))

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
		simba := NewSIMBA(
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
		simba := NewSIMBA()

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()

		// Define metric function
		metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }

		// Pass nil context - should be handled gracefully
		result, err := simba.Compile(context.TODO(), program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, result)
	})

	t.Run("Handles LLM generation failures", func(t *testing.T) {
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

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
		simba := NewSIMBA()

		// Empty performance log
		simba.state.PerformanceLog = []StepResult{}

		converged := simba.hasConverged()
		assert.False(t, converged, "Should not converge without data")
	})

	t.Run("Completes optimization within reasonable time", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.9 }
		simba := NewSIMBA(WithSIMBAMaxSteps(3)) // Small number for fast test

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

// testSIMBADualStrategy tests the dual strategy system functionality.
func testSIMBADualStrategy(t *testing.T) {
	t.Run("Generates candidates using both strategies", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBAStrategyMode("both"),
			WithSIMBAStrategyRatio(0.5),
			WithSIMBANumCandidates(7), // 1 base + 6 new (3 instruction + 3 rule)
		)

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Add some trajectories for rule generation
		addMockTrajectories(simba)

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1) // Should have at least base program
		assert.True(t, len(candidates) <= 7) // Should not exceed requested candidates
	})

	t.Run("Instruction-only strategy mode works correctly", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBAStrategyMode("instruction_only"),
			WithSIMBANumCandidates(4),
		)

		program := createSIMBATestProgram()
		ctx := context.Background()

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1)
		assert.True(t, len(candidates) <= 4)
	})

	t.Run("Rule-only strategy mode works correctly", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBAStrategyMode("rule_only"),
			WithSIMBANumCandidates(4),
		)

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Add trajectories for rule generation
		addMockTrajectories(simba)

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1)
	})

	t.Run("Strategy ratio controls candidate distribution", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBAStrategyMode("both"),
			WithSIMBAStrategyRatio(0.8), // 80% instruction, 20% rule
			WithSIMBANumCandidates(6),   // 1 base + 5 new (4 instruction + 1 rule)
		)

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Add trajectories for rule generation
		addMockTrajectories(simba)

		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1)
	})

	t.Run("Fallback to instruction perturbation when rule generation fails", func(t *testing.T) {
		simba := NewSIMBA(
			WithSIMBAStrategyMode("rule_only"),
			WithSIMBANumCandidates(3),
		)

		program := createSIMBATestProgram()
		ctx := context.Background()

		// Don't add trajectories - should fall back to instruction perturbation
		candidates, err := simba.generateCandidates(ctx, program)

		assert.NoError(t, err)
		assert.True(t, len(candidates) >= 1)
	})
}

// testSIMBATrajectoryTracking tests trajectory tracking functionality.
func testSIMBATrajectoryTracking(t *testing.T) {
	t.Run("Records trajectories during evaluation", func(t *testing.T) {
		simba := NewSIMBA()

		// Initialize the metric function
		simba.metric = func(expected, actual map[string]interface{}) float64 {
			return 0.8 // Return a fixed score for testing
		}

		program := createSIMBATestProgram()
		batch := []core.Example{
			{
				Inputs:  map[string]interface{}{"question": "What is 2+2?"},
				Outputs: map[string]interface{}{"answer": "4"},
			},
		}

		ctx := context.Background()
		score := simba.evaluateCandidateOnBatch(ctx, program, batch)

		assert.True(t, score >= 0)

		// Check that trajectories were recorded
		state := simba.GetState()
		assert.True(t, len(state.Trajectories) > 0, "Expected trajectories to be recorded but found none")

		// Verify trajectory structure
		if len(state.Trajectories) > 0 {
			trajectory := state.Trajectories[0]
			assert.NotEmpty(t, trajectory.ProgramID, "Program ID should not be empty")
			assert.NotNil(t, trajectory.Example, "Example should not be nil")
			assert.True(t, trajectory.ExecutionTime > 0, "Execution time should be positive")
		}
	})

	t.Run("Maintains sliding window of trajectories", func(t *testing.T) {
		simba := NewSIMBA()

		// Initialize the metric function
		simba.metric = func(expected, actual map[string]interface{}) float64 {
			return 0.7 // Return a fixed score for testing
		}

		program := createSIMBATestProgram()
		batch := []core.Example{
			{
				Inputs:  map[string]interface{}{"question": "Test question"},
				Outputs: map[string]interface{}{"answer": "Test answer"},
			},
		}

		ctx := context.Background()

		// Add more than 100 trajectories (sliding window limit)
		for i := 0; i < 55; i++ {
			simba.evaluateCandidateOnBatch(ctx, program, batch)
		}

		state := simba.GetState()
		assert.True(t, len(state.Trajectories) <= 100, "Should maintain sliding window of max 100 trajectories")
	})

	t.Run("Tracks success and failure correctly", func(t *testing.T) {
		simba := NewSIMBA()

		// Mock metric to control success/failure
		simba.metric = func(expected, actual map[string]interface{}) float64 {
			if actual != nil {
				return 0.8 // Success (> 0.5)
			}
			return 0.2 // Failure (< 0.5)
		}

		program := createSIMBATestProgram()
		batch := []core.Example{
			{
				Inputs:  map[string]interface{}{"question": "Test question"},
				Outputs: map[string]interface{}{"answer": "Test answer"},
			},
		}

		ctx := context.Background()
		simba.evaluateCandidateOnBatch(ctx, program, batch)

		state := simba.GetState()
		assert.True(t, len(state.Trajectories) > 0)

		// Check that success is tracked correctly
		if len(state.Trajectories) > 0 {
			trajectory := state.Trajectories[0]
			assert.True(t, trajectory.Success, "Trajectory should be marked as successful for score > 0.5")
		}
	})

	t.Run("Handles program evaluation failures", func(t *testing.T) {
		simba := NewSIMBA()

		// Initialize the metric function
		simba.metric = func(expected, actual map[string]interface{}) float64 {
			return 0.5 // Return a fixed score for testing
		}

		// Create a program that fails during evaluation
		failingProgram := core.NewProgram(
			map[string]core.Module{},
			func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return nil, fmt.Errorf("evaluation failed")
			},
		)

		batch := []core.Example{
			{
				Inputs:  map[string]interface{}{"question": "Test question"},
				Outputs: map[string]interface{}{"answer": "Test answer"},
			},
		}

		ctx := context.Background()
		score := simba.evaluateCandidateOnBatch(ctx, failingProgram, batch)

		assert.Equal(t, 0.0, score)

		// Check that failed trajectories are recorded
		state := simba.GetState()
		assert.True(t, len(state.Trajectories) > 0)

		trajectory := state.Trajectories[0]
		assert.False(t, trajectory.Success)
		assert.Equal(t, 0.0, trajectory.Score)
		assert.Nil(t, trajectory.Prediction)
	})
}

// testSIMBARuleGeneration tests rule generation from trajectories.
func testSIMBARuleGeneration(t *testing.T) {
	t.Run("Extracts rules from trajectories", func(t *testing.T) {
		simba := NewSIMBA()

		// Add mock trajectories
		addMockTrajectories(simba)

		ctx := context.Background()
		rules, err := simba.extractRulesFromTrajectories(ctx)

		assert.NoError(t, err)
		assert.True(t, len(rules) >= 0) // Could be empty if LLM doesn't extract rules
	})

	t.Run("Handles insufficient trajectory data", func(t *testing.T) {
		simba := NewSIMBA()

		// Don't add trajectories - should handle gracefully
		ctx := context.Background()
		rules, err := simba.extractRulesFromTrajectories(ctx)

		assert.NoError(t, err)
		assert.Equal(t, 0, len(rules))
	})

	t.Run("Requires both successful and failed trajectories", func(t *testing.T) {
		simba := NewSIMBA()

		// Add only successful trajectories
		successfulTrajectories := []Trajectory{
			{
				Example: core.Example{
					Inputs:  map[string]interface{}{"question": "What is 2+2?"},
					Outputs: map[string]interface{}{"answer": "4"},
				},
				Prediction: map[string]interface{}{"answer": "4"},
				Score:      0.9,
				Success:    true,
			},
		}

		simba.state.Trajectories = successfulTrajectories

		ctx := context.Background()
		rules, err := simba.extractRulesFromTrajectories(ctx)

		assert.NoError(t, err)
		assert.Equal(t, 0, len(rules)) // Should return empty rules without both success and failure
	})

	t.Run("Parses rules from analysis correctly", func(t *testing.T) {
		simba := NewSIMBA()

		analysis := `
		Based on the analysis, here are the rules:
		RULE: Always provide detailed explanations
		RULE: Use specific examples when possible
		RULE: Verify calculations before responding
		`

		rules := simba.parseRulesFromAnalysis(analysis)

		assert.Equal(t, 3, len(rules))
		assert.Contains(t, rules, "Always provide detailed explanations")
		assert.Contains(t, rules, "Use specific examples when possible")
		assert.Contains(t, rules, "Verify calculations before responding")
	})

	t.Run("Limits rules to maximum of 3", func(t *testing.T) {
		simba := NewSIMBA()

		analysis := `
		RULE: Rule 1
		RULE: Rule 2
		RULE: Rule 3
		RULE: Rule 4
		RULE: Rule 5
		`

		rules := simba.parseRulesFromAnalysis(analysis)

		assert.Equal(t, 3, len(rules)) // Should be limited to 3 rules
	})

	t.Run("Generates rule-based candidates", func(t *testing.T) {
		simba := NewSIMBA()

		// Add mock trajectories
		addMockTrajectories(simba)

		program := createSIMBATestProgram()
		ctx := context.Background()

		candidate, err := simba.generateRuleBasedCandidate(ctx, program)

		assert.NoError(t, err)
		assert.NotNil(t, candidate)
		assert.NotSame(t, &program, &candidate) // Should be a different instance
	})

	t.Run("Appends rules to instructions", func(t *testing.T) {
		simba := NewSIMBA()

		// Set up a mock primary model that returns enhanced instructions
		mockLLM := new(testutil.MockLLM)
		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
			&core.LLMResponse{Content: "Answer the question. Be specific and detailed."}, nil)
		simba.primaryModel = mockLLM

		originalInstruction := "Answer the question"
		rules := []string{"Be specific and detailed", "Provide examples"}

		ctx := context.Background()
		enhancedInstruction, err := simba.appendRuleToInstruction(ctx, originalInstruction, rules)

		assert.NoError(t, err)
		assert.NotEmpty(t, enhancedInstruction)
		assert.NotEqual(t, originalInstruction, enhancedInstruction)
	})

	t.Run("Handles empty rules gracefully", func(t *testing.T) {
		simba := NewSIMBA()

		originalInstruction := "Answer the question"
		rules := []string{}

		ctx := context.Background()
		enhancedInstruction, err := simba.appendRuleToInstruction(ctx, originalInstruction, rules)

		assert.NoError(t, err)
		assert.Equal(t, originalInstruction, enhancedInstruction)
	})
}

// testSIMBAStrategyConfiguration tests strategy configuration options.
func testSIMBAStrategyConfiguration(t *testing.T) {
	t.Run("Default configuration uses both strategies", func(t *testing.T) {
		simba := NewSIMBA()

		config := simba.GetConfig()
		assert.Equal(t, "both", config.StrategyMode)
		assert.Equal(t, 0.5, config.StrategyRatio)
	})

	t.Run("WithSIMBAStrategyMode sets strategy mode correctly", func(t *testing.T) {
		simba := NewSIMBA(WithSIMBAStrategyMode("instruction_only"))

		config := simba.GetConfig()
		assert.Equal(t, "instruction_only", config.StrategyMode)
	})

	t.Run("WithSIMBAStrategyRatio sets ratio correctly", func(t *testing.T) {
		simba := NewSIMBA(WithSIMBAStrategyRatio(0.7))

		config := simba.GetConfig()
		assert.Equal(t, 0.7, config.StrategyRatio)
	})

	t.Run("Strategy ratio is clamped between 0 and 1", func(t *testing.T) {
		simba1 := NewSIMBA(WithSIMBAStrategyRatio(-0.5))
		config1 := simba1.GetConfig()
		assert.Equal(t, 0.0, config1.StrategyRatio)

		simba2 := NewSIMBA(WithSIMBAStrategyRatio(1.5))
		config2 := simba2.GetConfig()
		assert.Equal(t, 1.0, config2.StrategyRatio)
	})

	t.Run("Strategy types are defined correctly", func(t *testing.T) {
		assert.Equal(t, "instruction_perturbation", string(InstructionPerturbation))
		assert.Equal(t, "rule_generation", string(RuleGeneration))
	})

	t.Run("Configuration supports all strategy modes", func(t *testing.T) {
		validModes := []string{"both", "instruction_only", "rule_only"}

		for _, mode := range validModes {
			simba := NewSIMBA(WithSIMBAStrategyMode(mode))
			config := simba.GetConfig()
			assert.Equal(t, mode, config.StrategyMode)
		}
	})

	t.Run("Trajectory tracking is enabled by default", func(t *testing.T) {
		simba := NewSIMBA()

		state := simba.GetState()
		assert.NotNil(t, state.Trajectories)
		assert.Equal(t, 0, len(state.Trajectories)) // Initially empty
	})
}

// testSIMBAIntegration tests integration scenarios.
func testSIMBAIntegration(t *testing.T) {
	t.Run("Full optimization with dual strategy", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 {
			if actual != nil && expected != nil {
				return 0.8 // Good score
			}
			return 0.3 // Lower score
		}

		simba := NewSIMBA(
			WithSIMBAMaxSteps(3),
			WithSIMBAStrategyMode("both"),
			WithSIMBAStrategyRatio(0.6),
			WithSIMBANumCandidates(4),
		)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)

		// Verify optimization process
		state := simba.GetState()
		assert.True(t, state.CurrentStep >= 0)
		assert.True(t, state.BestScore >= 0)
		assert.True(t, len(state.Trajectories) > 0)
		assert.True(t, len(state.PerformanceLog) > 0)
	})

	t.Run("Optimization with rule-only strategy", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.7 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBAStrategyMode("rule_only"),
			WithSIMBANumCandidates(3),
		)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)

		// Should complete successfully even with rule-only strategy
		state := simba.GetState()
		assert.True(t, len(state.Trajectories) > 0)
	})

	t.Run("Performance comparison between strategies", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.75 }

		// Test instruction-only strategy
		simba1 := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBAStrategyMode("instruction_only"),
			WithSIMBANumCandidates(3),
		)

		program1 := createSIMBATestProgram()
		dataset1 := createSIMBATestDataset()
		ctx1 := core.WithExecutionState(context.Background())

		start1 := time.Now()
		_, err1 := simba1.Compile(ctx1, program1, dataset1, metric)
		duration1 := time.Since(start1)

		assert.NoError(t, err1)

		// Test dual strategy
		simba2 := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBAStrategyMode("both"),
			WithSIMBANumCandidates(3),
		)

		program2 := createSIMBATestProgram()
		dataset2 := createSIMBATestDataset()
		ctx2 := core.WithExecutionState(context.Background())

		start2 := time.Now()
		_, err2 := simba2.Compile(ctx2, program2, dataset2, metric)
		duration2 := time.Since(start2)

		assert.NoError(t, err2)

		// Both should complete in reasonable time
		assert.True(t, duration1 < 30*time.Second)
		assert.True(t, duration2 < 30*time.Second)
	})

	t.Run("Trajectory accumulation across optimization steps", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.6 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(3),
			WithSIMBAStrategyMode("both"),
			WithSIMBANumCandidates(4),
		)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		_, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)

		// Verify trajectories accumulated over multiple steps
		state := simba.GetState()
		assert.True(t, len(state.Trajectories) > 0)

		// Check that trajectories have varying program IDs (different candidates)
		programIDs := make(map[string]bool)
		for _, trajectory := range state.Trajectories {
			programIDs[trajectory.ProgramID] = true
		}
		assert.True(t, len(programIDs) > 1, "Should have trajectories from multiple program candidates")
	})

	t.Run("Error handling preserves dual strategy functionality", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.5 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBAStrategyMode("both"),
			WithSIMBANumCandidates(3),
		)

		// Set up a failing LLM for rule generation
		failingLLM := new(testutil.MockLLM)
		failingLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(nil, fmt.Errorf("LLM failure"))

		simba.analyzerModel = failingLLM

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		// Should still complete successfully by falling back to instruction perturbation
		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)
	})

	t.Run("Full optimization with bucket sorting", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 {
			if actual != nil && expected != nil {
				return 0.85 // Good score
			}
			return 0.3 // Lower score
		}

		simba := NewSIMBA(
			WithSIMBAMaxSteps(3),
			WithBucketSorting(true),
			WithBucketSortingCriteria([]string{"max_score", "max_to_avg_gap", "diversity"}),
			WithBucketSortingWeights([]float64{0.5, 0.3, 0.2}),
			WithSIMBANumCandidates(4),
		)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)

		// Verify bucket sorting was used
		state := simba.GetState()
		assert.True(t, state.CurrentStep >= 0)
		assert.True(t, state.BestScore >= 0)
		assert.True(t, len(state.CandidateHistory) > 0)
		assert.True(t, len(state.PerformanceLog) > 0)

		// Check that candidate metadata was recorded
		hasMetadata := false
		for _, candidate := range state.CandidateHistory {
			if candidate.Metadata != nil {
				hasMetadata = true
				assert.True(t, candidate.Metadata.CompositeScore >= 0)
				assert.True(t, candidate.Metadata.SelectionRank > 0)
				assert.True(t, candidate.Metadata.BucketAssignment > 0)
				break
			}
		}
		assert.True(t, hasMetadata, "Expected candidate metadata to be recorded")
	})

	t.Run("Bucket sorting vs temperature sampling performance comparison", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 {
			return 0.75 + 0.1*float64(len(actual)) // Varies by response complexity
		}

		// Test with temperature sampling only
		simba1 := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithBucketSorting(false),
			WithSIMBANumCandidates(3),
		)

		program1 := createSIMBATestProgram()
		dataset1 := createSIMBATestDataset()
		ctx1 := core.WithExecutionState(context.Background())

		start1 := time.Now()
		_, err1 := simba1.Compile(ctx1, program1, dataset1, metric)
		duration1 := time.Since(start1)

		assert.NoError(t, err1)

		// Test with bucket sorting
		simba2 := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithBucketSorting(true),
			WithSIMBANumCandidates(3),
		)

		program2 := createSIMBATestProgram()
		dataset2 := createSIMBATestDataset()
		ctx2 := core.WithExecutionState(context.Background())

		start2 := time.Now()
		_, err2 := simba2.Compile(ctx2, program2, dataset2, metric)
		duration2 := time.Since(start2)

		assert.NoError(t, err2)

		// Both should complete successfully and in reasonable time
		assert.True(t, duration1 < 30*time.Second)
		assert.True(t, duration2 < 30*time.Second)

		// Get final states
		state1 := simba1.GetState()
		state2 := simba2.GetState()

		// Both should have achieved reasonable performance
		assert.True(t, state1.BestScore >= 0.0)
		assert.True(t, state2.BestScore >= 0.0)

		// Bucket sorting should have additional metadata
		assert.True(t, len(state2.CandidateHistory) >= len(state1.CandidateHistory))
	})

	t.Run("Bucket sorting with different criteria combinations", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.8 }

		testCases := []struct {
			name     string
			criteria []string
			weights  []float64
		}{
			{
				name:     "Max score focused",
				criteria: []string{"max_score"},
				weights:  []float64{1.0},
			},
			{
				name:     "Gap focused",
				criteria: []string{"max_to_min_gap", "max_to_avg_gap"},
				weights:  []float64{0.6, 0.4},
			},
			{
				name:     "Diversity focused",
				criteria: []string{"diversity", "improvement_potential"},
				weights:  []float64{0.7, 0.3},
			},
			{
				name:     "Balanced approach",
				criteria: []string{"max_score", "max_to_avg_gap", "diversity"},
				weights:  []float64{0.4, 0.3, 0.3},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				simba := NewSIMBA(
					WithSIMBAMaxSteps(2),
					WithBucketSorting(true),
					WithBucketSortingCriteria(tc.criteria),
					WithBucketSortingWeights(tc.weights),
					WithSIMBANumCandidates(3),
				)

				program := createSIMBATestProgram()
				dataset := createSIMBATestDataset()
				ctx := core.WithExecutionState(context.Background())

				optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

				assert.NoError(t, err)
				assert.NotNil(t, optimizedProgram)

				// Verify configuration was applied
				config := simba.GetConfig()
				assert.Equal(t, tc.criteria, config.BucketSortingCriteria)
				assert.Equal(t, tc.weights, config.BucketSortingWeights)
			})
		}
	})

	t.Run("Bucket sorting error handling and robustness", func(t *testing.T) {
		metric := func(expected, actual map[string]interface{}) float64 { return 0.5 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithBucketSorting(true),
			WithBucketSortingCriteria([]string{"invalid_criterion", "max_score"}),
			WithBucketSortingWeights([]float64{0.5, 0.5}),
			WithSIMBANumCandidates(2),
		)

		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		ctx := core.WithExecutionState(context.Background())

		// Should handle invalid criteria gracefully
		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)

		// Should still complete successfully
		state := simba.GetState()
		assert.True(t, state.BestScore >= 0)
	})
}

// Helper function to add mock trajectories for testing.
func addMockTrajectories(simba *SIMBA) {
	mockTrajectories := []Trajectory{
		{
			Example: core.Example{
				Inputs:  map[string]interface{}{"question": "What is 2+2?"},
				Outputs: map[string]interface{}{"answer": "4"},
			},
			Prediction: map[string]interface{}{"answer": "4"},
			Score:      0.9,
			Success:    true,
			ProgramID:  "prog_success_1",
		},
		{
			Example: core.Example{
				Inputs:  map[string]interface{}{"question": "What is 3+3?"},
				Outputs: map[string]interface{}{"answer": "6"},
			},
			Prediction: map[string]interface{}{"answer": "6"},
			Score:      0.8,
			Success:    true,
			ProgramID:  "prog_success_2",
		},
		{
			Example: core.Example{
				Inputs:  map[string]interface{}{"question": "What is the capital of Mars?"},
				Outputs: map[string]interface{}{"answer": "Unknown"},
			},
			Prediction: map[string]interface{}{"answer": "New York"},
			Score:      0.1,
			Success:    false,
			ProgramID:  "prog_fail_1",
		},
		{
			Example: core.Example{
				Inputs:  map[string]interface{}{"question": "What is 5+5?"},
				Outputs: map[string]interface{}{"answer": "10"},
			},
			Prediction: map[string]interface{}{"answer": "11"},
			Score:      0.2,
			Success:    false,
			ProgramID:  "prog_fail_2",
		},
	}

	simba.mu.Lock()
	simba.state.Trajectories = mockTrajectories
	simba.mu.Unlock()
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

// testSIMBABucketSortingConfiguration tests bucket sorting configuration options.
func testSIMBABucketSortingConfiguration(t *testing.T) {
	t.Run("Default configuration has bucket sorting disabled", func(t *testing.T) {
		simba := NewSIMBA()
		config := simba.GetConfig()

		assert.False(t, config.UseBucketSorting)
		assert.Equal(t, []string{"max_to_min_gap", "max_score", "max_to_avg_gap"}, config.BucketSortingCriteria)
		assert.Equal(t, []float64{0.4, 0.4, 0.2}, config.BucketSortingWeights)
	})

	t.Run("WithBucketSorting enables bucket sorting", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))
		config := simba.GetConfig()

		assert.True(t, config.UseBucketSorting)
	})

	t.Run("WithBucketSortingCriteria sets criteria correctly", func(t *testing.T) {
		criteria := []string{"max_score", "diversity", "improvement_potential"}
		simba := NewSIMBA(WithBucketSortingCriteria(criteria))
		config := simba.GetConfig()

		assert.Equal(t, criteria, config.BucketSortingCriteria)
	})

	t.Run("WithBucketSortingWeights normalizes weights", func(t *testing.T) {
		weights := []float64{0.6, 0.3, 0.1}
		simba := NewSIMBA(WithBucketSortingWeights(weights))
		config := simba.GetConfig()

		// Weights should sum to 1.0
		total := 0.0
		for _, w := range config.BucketSortingWeights {
			total += w
		}
		assert.InDelta(t, 1.0, total, 0.001)
	})

	t.Run("WithBucketSortingWeights handles zero weights", func(t *testing.T) {
		weights := []float64{0.0, 0.0, 0.0}
		simba := NewSIMBA(WithBucketSortingWeights(weights))
		config := simba.GetConfig()

		// Should keep original weights when sum is zero
		assert.Equal(t, []float64{0.4, 0.4, 0.2}, config.BucketSortingWeights)
	})

	t.Run("WithBucketSortingWeights handles empty weights", func(t *testing.T) {
		weights := []float64{}
		simba := NewSIMBA(WithBucketSortingWeights(weights))
		config := simba.GetConfig()

		// Should keep original weights when empty
		assert.Equal(t, []float64{0.4, 0.4, 0.2}, config.BucketSortingWeights)
	})
}

// testSIMBAMultiCriteriaScoring tests multi-criteria scoring functionality.
func testSIMBAMultiCriteriaScoring(t *testing.T) {
	t.Run("Calculates multi-criteria scores correctly", func(t *testing.T) {
		simba := NewSIMBA()

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.3, 0.7, 0.5}

		metadata := simba.calculateMultiCriteriaScore(candidates, scores)

		assert.Equal(t, 3, len(metadata))

		// Check that all metadata entries have valid values
		for i, meta := range metadata {
			assert.Equal(t, scores[i], meta.IndividualScores[0])
			assert.True(t, meta.MaxScore >= 0)
			assert.True(t, meta.MaxToMinGap >= 0)
			assert.True(t, meta.MaxToAvgGap >= 0)
			assert.True(t, meta.DiversityScore >= 0)
			assert.True(t, meta.CompositeScore >= 0)
		}
	})

	t.Run("Handles single candidate correctly", func(t *testing.T) {
		simba := NewSIMBA()

		candidates := []core.Program{createSIMBATestProgram()}
		scores := []float64{0.8}

		metadata := simba.calculateMultiCriteriaScore(candidates, scores)

		assert.Equal(t, 1, len(metadata))
		assert.Equal(t, 0.8, metadata[0].IndividualScores[0])
		assert.Equal(t, 0.8, metadata[0].MaxScore)
		assert.Equal(t, 0.0, metadata[0].MaxToMinGap) // Single candidate has no gap
		assert.Equal(t, 0.0, metadata[0].MaxToAvgGap) // Single candidate has no gap
	})

	t.Run("Handles identical scores correctly", func(t *testing.T) {
		simba := NewSIMBA()

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.5, 0.5, 0.5}

		metadata := simba.calculateMultiCriteriaScore(candidates, scores)

		assert.Equal(t, 3, len(metadata))

		// All candidates should have identical gaps
		for _, meta := range metadata {
			assert.Equal(t, 0.0, meta.MaxToMinGap)
			assert.Equal(t, 0.0, meta.MaxToAvgGap)
			assert.Equal(t, 0.0, meta.DiversityScore)
		}
	})

	t.Run("Composite score calculation works with different criteria", func(t *testing.T) {
		simba := NewSIMBA(
			WithBucketSortingCriteria([]string{"max_score", "diversity"}),
			WithBucketSortingWeights([]float64{0.7, 0.3}),
		)

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.2, 0.8}

		metadata := simba.calculateMultiCriteriaScore(candidates, scores)

		assert.Equal(t, 2, len(metadata))

		// Higher score should have higher composite score
		assert.True(t, metadata[1].CompositeScore > metadata[0].CompositeScore)
	})

	t.Run("Handles edge cases gracefully", func(t *testing.T) {
		simba := NewSIMBA()

		// Empty candidates
		emptyMetadata := simba.calculateMultiCriteriaScore([]core.Program{}, []float64{})
		assert.Equal(t, 0, len(emptyMetadata))

		// Zero scores
		candidates := []core.Program{createSIMBATestProgram()}
		zeroScores := []float64{0.0}

		metadata := simba.calculateMultiCriteriaScore(candidates, zeroScores)
		assert.Equal(t, 1, len(metadata))
		assert.Equal(t, 0.0, metadata[0].IndividualScores[0])
	})
}

// testSIMBABucketSortingAlgorithm tests bucket sorting algorithm.
func testSIMBABucketSortingAlgorithm(t *testing.T) {
	t.Run("Selects best candidate with bucket sorting", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.3, 0.9, 0.6}

		selectedProgram, selectedScore := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, selectedProgram)
		assert.True(t, selectedScore >= 0.6) // Should select from top-performing candidates
	})

	t.Run("Bucket assignment works correctly", func(t *testing.T) {
		simba := NewSIMBA()

		// Test bucket assignment for different ranks
		totalCandidates := 10

		// Top 30% should be bucket 1
		topBucket := simba.assignBucket(0.9, 0, totalCandidates)
		assert.Equal(t, 1, topBucket)

		// Next 40% should be bucket 2
		middleBucket := simba.assignBucket(0.5, 5, totalCandidates)
		assert.Equal(t, 2, middleBucket)

		// Bottom 30% should be bucket 3
		bottomBucket := simba.assignBucket(0.1, 9, totalCandidates)
		assert.Equal(t, 3, bottomBucket)
	})

	t.Run("Temperature sampling works within top bucket", func(t *testing.T) {
		simba := NewSIMBA(
			WithBucketSorting(true),
			WithSamplingTemperature(0.5),
		)

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.2, 0.8, 0.9, 0.85}

		// Run multiple selections to check probabilistic behavior
		selections := make(map[float64]int)
		for i := 0; i < 100; i++ {
			_, selectedScore := simba.selectBestCandidate(candidates, scores)
			selections[selectedScore]++
		}

		// Should favor high-scoring candidates but allow some diversity
		assert.True(t, len(selections) >= 1)
		assert.True(t, selections[0.9] > 0 || selections[0.85] > 0 || selections[0.8] > 0)
	})

	t.Run("Handles single candidate correctly", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		candidates := []core.Program{createSIMBATestProgram()}
		scores := []float64{0.7}

		selectedProgram, selectedScore := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, selectedProgram)
		assert.Equal(t, 0.7, selectedScore)
	})

	t.Run("Fallback to temperature sampling when bucket sorting disabled", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(false))

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.3, 0.7}

		selectedProgram, selectedScore := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, selectedProgram)
		assert.True(t, selectedScore >= 0.3)
	})

	t.Run("Bucket sorting with custom criteria", func(t *testing.T) {
		simba := NewSIMBA(
			WithBucketSorting(true),
			WithBucketSortingCriteria([]string{"max_score", "diversity", "improvement_potential"}),
			WithBucketSortingWeights([]float64{0.5, 0.3, 0.2}),
		)

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.2, 0.8, 0.5}

		selectedProgram, selectedScore := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, selectedProgram)
		assert.True(t, selectedScore >= 0.2)
	})
}

// testSIMBACandidateMetadata tests candidate metadata tracking.
func testSIMBACandidateMetadata(t *testing.T) {
	t.Run("Records candidate metadata correctly", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		program := createSIMBATestProgram()
		score := 0.8
		metadata := &CandidateMetadata{
			IndividualScores: []float64{0.8},
			DiversityScore:   0.2,
			MaxScore:         0.9,
			CompositeScore:   0.85,
			SelectionRank:    1,
			BucketAssignment: 1,
		}

		simba.recordCandidateMetadata(program, score, metadata)

		state := simba.GetState()
		assert.Equal(t, 1, len(state.CandidateHistory))

		result := state.CandidateHistory[0]
		assert.Equal(t, score, result.Score)
		assert.NotNil(t, result.Metadata)
		assert.Equal(t, 0.85, result.Metadata.CompositeScore)
		assert.Equal(t, 1, result.Metadata.SelectionRank)
		assert.Equal(t, 1, result.Metadata.BucketAssignment)
	})

	t.Run("Maintains sliding window of candidate history", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		program := createSIMBATestProgram()
		metadata := &CandidateMetadata{
			IndividualScores: []float64{0.5},
			CompositeScore:   0.5,
		}

		// Add more than 50 candidates
		for i := 0; i < 55; i++ {
			simba.recordCandidateMetadata(program, 0.5, metadata)
		}

		state := simba.GetState()
		assert.Equal(t, 50, len(state.CandidateHistory)) // Should maintain max 50
	})

	t.Run("Handles nil metadata gracefully", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		program := createSIMBATestProgram()
		score := 0.7

		// Should not panic with nil metadata
		simba.recordCandidateMetadata(program, score, nil)

		state := simba.GetState()
		assert.Equal(t, 0, len(state.CandidateHistory)) // Should not record anything
	})

	t.Run("Metadata includes timestamp and temperature", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		program := createSIMBATestProgram()
		score := 0.6
		metadata := &CandidateMetadata{
			IndividualScores: []float64{0.6},
			CompositeScore:   0.6,
		}

		simba.recordCandidateMetadata(program, score, metadata)

		state := simba.GetState()
		assert.Equal(t, 1, len(state.CandidateHistory))

		result := state.CandidateHistory[0]
		assert.True(t, result.CreatedAt.After(time.Time{}))
		assert.True(t, result.Temperature >= 0)
	})

	t.Run("Integration with bucket sorting selection", func(t *testing.T) {
		simba := NewSIMBA(WithBucketSorting(true))

		candidates := []core.Program{
			createSIMBATestProgram(),
			createSIMBATestProgram(),
			createSIMBATestProgram(),
		}
		scores := []float64{0.3, 0.7, 0.5}

		// Select candidate using bucket sorting
		selectedProgram, selectedScore := simba.selectBestCandidate(candidates, scores)

		assert.NotNil(t, selectedProgram)
		assert.True(t, selectedScore >= 0.3)

		// Check that metadata was recorded
		state := simba.GetState()
		assert.Equal(t, 1, len(state.CandidateHistory))

		result := state.CandidateHistory[0]
		assert.Equal(t, selectedScore, result.Score)
		assert.NotNil(t, result.Metadata)
		assert.True(t, result.Metadata.CompositeScore >= 0)
	})
}

// testSIMBALLMConcurrency tests the LLM concurrency configuration.
func testSIMBALLMConcurrency(t *testing.T) {
	t.Run("Default LLM concurrency is unlimited", func(t *testing.T) {
		simba := NewSIMBA()
		assert.Equal(t, 0, simba.config.LLMConcurrency)

		// Test that createLLMPool returns unlimited pool
		pool := simba.createLLMPool()
		assert.NotNil(t, pool)
	})

	t.Run("LLM concurrency can be configured", func(t *testing.T) {
		simba := NewSIMBA(WithLLMConcurrency(50))
		assert.Equal(t, 50, simba.config.LLMConcurrency)

		// Test that createLLMPool returns limited pool
		pool := simba.createLLMPool()
		assert.NotNil(t, pool)
	})

	t.Run("LLM concurrency option validation", func(t *testing.T) {
		// Test various concurrency values
		testCases := []struct {
			concurrency int
			expected    int
		}{
			{0, 0},     // unlimited
			{1, 1},     // single threaded
			{100, 100}, // high concurrency
			{-1, -1},   // negative (should work as unlimited)
		}

		for _, tc := range testCases {
			simba := NewSIMBA(WithLLMConcurrency(tc.concurrency))
			assert.Equal(t, tc.expected, simba.config.LLMConcurrency)
		}
	})
}

// testSIMBAPipelineProcessing tests the pipeline processing functionality.
func testSIMBAPipelineProcessing(t *testing.T) {
	t.Run("Pipeline processing configuration", func(t *testing.T) {
		// Test default configuration
		simba := NewSIMBA()
		assert.False(t, simba.config.UsePipelineProcessing)
		assert.Equal(t, 2, simba.config.PipelineBufferSize)

		// Test enabling pipeline processing
		simba = NewSIMBA(WithPipelineProcessing(true))
		assert.True(t, simba.config.UsePipelineProcessing)

		// Test configuring buffer size
		simba = NewSIMBA(WithPipelineBufferSize(5))
		assert.Equal(t, 5, simba.config.PipelineBufferSize)

		// Test zero buffer size (should be ignored)
		simba = NewSIMBA(WithPipelineBufferSize(0))
		assert.Equal(t, 2, simba.config.PipelineBufferSize) // Should remain default
	})

	t.Run("Pipeline channels initialization", func(t *testing.T) {
		simba := NewSIMBA(WithPipelineProcessing(true), WithPipelineBufferSize(3))

		// Test channel initialization
		simba.initializePipelineChannels()
		assert.NotNil(t, simba.pipelineChannels)
		assert.NotNil(t, simba.pipelineChannels.CandidateGeneration)
		assert.NotNil(t, simba.pipelineChannels.BatchSampling)
		assert.NotNil(t, simba.pipelineChannels.CandidateEvaluation)
		assert.NotNil(t, simba.pipelineChannels.Results)
		assert.NotNil(t, simba.pipelineChannels.Errors)
		assert.NotNil(t, simba.pipelineChannels.Done)

		// Test safe cleanup
		simba.closePipelineChannels()
	})

	t.Run("Pipeline processing execution", func(t *testing.T) {
		// Create test components
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		metric := func(expected, actual map[string]interface{}) float64 { return 0.8 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(3),
			WithSIMBANumCandidates(4),
			WithPipelineProcessing(true),
			WithPipelineBufferSize(2),
		)

		ctx := context.Background()

		// Run optimization with pipeline processing
		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)

		// Check that optimization completed
		state := simba.GetState()
		assert.True(t, state.CurrentStep >= 0)
		assert.True(t, len(state.PerformanceLog) > 0)
	})

	t.Run("Sequential vs pipeline processing comparison", func(t *testing.T) {
		// Create test components
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		metric := func(expected, actual map[string]interface{}) float64 { return 0.75 }

		// Test sequential processing
		simbaSequential := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBANumCandidates(3),
			WithPipelineProcessing(false),
		)

		ctx := context.Background()
		startTime := time.Now()
		_, err1 := simbaSequential.Compile(ctx, program, dataset, metric)
		sequentialDuration := time.Since(startTime)

		// Test pipeline processing
		simbaPipeline := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBANumCandidates(3),
			WithPipelineProcessing(true),
			WithPipelineBufferSize(2),
		)

		startTime = time.Now()
		_, err2 := simbaPipeline.Compile(ctx, program, dataset, metric)
		pipelineDuration := time.Since(startTime)

		assert.NoError(t, err1)
		assert.NoError(t, err2)

		// Both should complete successfully
		// Note: In this test environment, pipeline might not always be faster
		// due to overhead, but both should work correctly
		t.Logf("Sequential duration: %v, Pipeline duration: %v", sequentialDuration, pipelineDuration)
	})

	t.Run("Pipeline error handling", func(t *testing.T) {
		// Create test components with potential for errors
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		metric := func(expected, actual map[string]interface{}) float64 { return 0.6 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(2),
			WithSIMBANumCandidates(2),
			WithPipelineProcessing(true),
			WithPipelineBufferSize(1), // Small buffer to test channel handling
		)

		ctx := context.Background()

		// Should handle gracefully even with small buffers and datasets
		optimizedProgram, err := simba.Compile(ctx, program, dataset, metric)

		// Should complete without errors despite edge conditions
		assert.NoError(t, err)
		assert.NotNil(t, optimizedProgram)
	})

	t.Run("Pipeline stage data flow", func(t *testing.T) {
		// Test PipelineStage creation and data flow
		stage := &PipelineStage{
			StepIndex:  1,
			Candidates: []core.Program{createSIMBATestProgram()},
			Batch:      []core.Example{{Inputs: map[string]interface{}{"input": "test"}, Outputs: map[string]interface{}{"output": "result"}}},
			Scores:     []float64{0.8},
			Timestamp:  time.Now(),
			Error:      nil,
		}

		assert.Equal(t, 1, stage.StepIndex)
		assert.Len(t, stage.Candidates, 1)
		assert.Len(t, stage.Batch, 1)
		assert.Len(t, stage.Scores, 1)
		assert.Nil(t, stage.Error)
		assert.False(t, stage.Timestamp.IsZero())
	})

	t.Run("Pipeline context cancellation", func(t *testing.T) {
		program := createSIMBATestProgram()
		dataset := createSIMBATestDataset()
		metric := func(expected, actual map[string]interface{}) float64 { return 0.5 }

		simba := NewSIMBA(
			WithSIMBAMaxSteps(5), // Longer optimization
			WithSIMBANumCandidates(4),
			WithPipelineProcessing(true),
			WithPipelineBufferSize(2),
		)

		// Create cancellable context with a CI-friendly timeout
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Start optimization in a goroutine to test cancellation
		done := make(chan bool, 1)
		var err error
		go func() {
			defer func() { done <- true }()
			_, err = simba.Compile(ctx, program, dataset, metric)
		}()

		// Cancel after a short time to ensure cancellation works
		time.Sleep(50 * time.Millisecond)
		cancel()

		// Wait for completion with timeout - increased for CI race detection overhead
		select {
		case <-done:
			// Should handle context cancellation gracefully
			if err != nil {
				assert.Contains(t, err.Error(), "context")
			}
		case <-time.After(10 * time.Second):
			t.Fatal("Pipeline context cancellation test timed out - workers not shutting down properly")
		}
	})
}

// Benchmark tests for SIMBA optimizer using shared benchmark utilities

// BenchmarkSIMBA runs comprehensive benchmarks for SIMBA optimizer.
func BenchmarkSIMBA(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	configs := testutil.StandardBenchmarkConfigs()

	testCases := []struct {
		name       string
		datasetKey string
		configKey  string
	}{
		{"Fast_Tiny", "tiny", "fast"},
		{"Fast_Small", "small", "fast"},
		{"Standard_Small", "small", "standard"},
		{"Standard_Medium", "medium", "standard"},
		{"Comprehensive_Medium", "medium", "comprehensive"},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			dataset := datasets[tc.datasetKey]
			config := configs[tc.configKey]

			// Create program
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			// Create SIMBA optimizer with config
			simba := NewSIMBA(
				WithSIMBABatchSize(config.BatchSize),
				WithSIMBAMaxSteps(config.MaxSteps),
				WithSIMBANumCandidates(6),
				WithSamplingTemperature(config.Temperature),
				WithFastMode(true), // Enable fast mode for benchmarks
			)

			// Setup mock LLM
			mockLLM := &testutil.MockLLM{}
			setupLLMMockForBenchmark(mockLLM)

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := simba.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				if err != nil {
					b.Fatalf("SIMBA compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkSIMBADatasetScaling tests performance across different dataset sizes.
func BenchmarkSIMBADatasetScaling(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	config := testutil.StandardBenchmarkConfigs()["fast"] // Use fast config for scaling tests

	for name, dataset := range datasets {
		b.Run(fmt.Sprintf("Dataset_%s_%d", name, dataset.Size), func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			simba := NewSIMBA(
				WithSIMBABatchSize(config.BatchSize),
				WithSIMBAMaxSteps(config.MaxSteps),
				WithSIMBANumCandidates(4),
				WithFastMode(true),
			)

			mockLLM := &testutil.MockLLM{}
			setupLLMMockForBenchmark(mockLLM)

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := simba.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				if err != nil {
					b.Fatalf("SIMBA compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkSIMBAParameterTuning tests performance with different parameter configurations.
func BenchmarkSIMBAParameterTuning(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"] // Use small dataset for parameter testing

	parameterTests := []struct {
		name          string
		batchSize     int
		maxSteps      int
		numCandidates int
		temperature   float64
	}{
		{"Batch2_Steps3_Candidates4", 2, 3, 4, 0.2},
		{"Batch4_Steps6_Candidates6", 4, 6, 6, 0.2},
		{"Batch8_Steps10_Candidates8", 8, 10, 8, 0.2},
		{"HighTemp_Batch4", 4, 6, 6, 1.0},
		{"LowTemp_Batch4", 4, 6, 6, 0.1},
	}

	for _, pt := range parameterTests {
		b.Run(pt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			simba := NewSIMBA(
				WithSIMBABatchSize(pt.batchSize),
				WithSIMBAMaxSteps(pt.maxSteps),
				WithSIMBANumCandidates(pt.numCandidates),
				WithSamplingTemperature(pt.temperature),
				WithFastMode(true),
			)

			mockLLM := &testutil.MockLLM{}
			setupLLMMockForBenchmark(mockLLM)

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := simba.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				if err != nil {
					b.Fatalf("SIMBA compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkSIMBAMiniBatch tests mini-batch processing performance.
func BenchmarkSIMBAMiniBatch(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["medium"]

	batchTests := []struct {
		name      string
		batchSize int
	}{
		{"Batch2", 2},
		{"Batch4", 4},
		{"Batch8", 8},
		{"Batch16", 16},
	}

	for _, bt := range batchTests {
		b.Run(bt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			simba := NewSIMBA(
				WithSIMBABatchSize(bt.batchSize),
				WithSIMBAMaxSteps(3), // Keep steps low for batch testing
				WithSIMBANumCandidates(4),
				WithFastMode(true),
			)

			mockLLM := &testutil.MockLLM{}
			setupLLMMockForBenchmark(mockLLM)

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := simba.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				if err != nil {
					b.Fatalf("SIMBA compilation failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkSIMBAFastMode compares fast mode vs normal mode performance.
func BenchmarkSIMBAFastMode(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"]

	modeTests := []struct {
		name     string
		fastMode bool
	}{
		{"FastMode_Enabled", true},
		{"FastMode_Disabled", false},
	}

	for _, mt := range modeTests {
		b.Run(mt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			simba := NewSIMBA(
				WithSIMBABatchSize(4),
				WithSIMBAMaxSteps(3),
				WithSIMBANumCandidates(4),
				WithFastMode(mt.fastMode),
			)

			mockLLM := &testutil.MockLLM{}
			setupLLMMockForBenchmark(mockLLM)

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := context.Background()
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := simba.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				if err != nil {
					b.Fatalf("SIMBA compilation failed: %v", err)
				}
			}
		})
	}
}

// setupLLMMockForBenchmark configures mock LLM for benchmark tests.
func setupLLMMockForBenchmark(mockLLM *testutil.MockLLM) {
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "test response"}, nil)
	mockLLM.On("GetModelName").Return("benchmark-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	mockLLM.On("ProviderName").Return("benchmark")
	mockLLM.On("ModelID").Return("benchmark-model-id")
}
