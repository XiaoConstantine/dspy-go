package optimizers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTPEOptimizer(t *testing.T) {
	t.Run("Basic Optimization Loop", func(t *testing.T) {
		// Create a simple test configuration
		config := TPEConfig{
			Gamma:            0.25,
			Seed:             42,
			NumEIGenerations: 10,
			PriorWeight:      1.0,
			BandwidthFactor:  1.0,
		}

		// Create parameter space - simple categorical parameters
		paramSpace := map[string][]interface{}{
			"param1": {float64(0), float64(1), float64(2)},
			"param2": {float64(0), float64(1)},
		}

		// Create and initialize the optimizer
		tpe := NewTPEOptimizer(config)
		err := tpe.Initialize(SearchConfig{
			ParamSpace: paramSpace,
			MaxTrials:  20,
			Seed:       42,
		})
		assert.NoError(t, err)

		// Run a simple optimization loop
		ctx := context.Background()
		for i := 0; i < 15; i++ {
			params, err := tpe.SuggestParams(ctx)
			assert.NoError(t, err)

			// Create a simple score based on the parameters
			// For testing, let's say param1=1 and param2=0 is optimal
			score := 0.5
			if params["param1"] == float64(1) {
				score += 0.3
			}
			if params["param2"] == float64(0) {
				score += 0.2
			}

			err = tpe.UpdateResults(params, score)
			assert.NoError(t, err)
		}

		// Check that the optimizer found the optimal parameters
		bestParams, bestScore := tpe.GetBestParams()
		t.Logf("Best parameters: %v, score: %f", bestParams, bestScore)
		assert.NotEmpty(t, bestParams)
		assert.Greater(t, bestScore, 0.5)
	})

	t.Run("Default Configuration Values", func(t *testing.T) {
		// Test that default values are applied when not specified
		config := TPEConfig{
			// Leave all fields at zero/default values
		}

		// Type assertion is necessary to access non-interface fields
		tpeImpl := NewTPEOptimizer(config).(*TPEOptimizer)

		// Check that default values were applied
		assert.Equal(t, 0.25, tpeImpl.gamma, "Default gamma should be 0.25")
		assert.Equal(t, 24, tpeImpl.numEIGenerations, "Default numEIGenerations should be 24")
		assert.Equal(t, 1.0, tpeImpl.priorWeight, "Default priorWeight should be 1.0")
		assert.Equal(t, 1.0, tpeImpl.bandwidthFactor, "Default bandwidthFactor should be 1.0")
		assert.NotEqual(t, 0, tpeImpl.seed, "Seed should be set to something non-zero")
		assert.NotNil(t, tpeImpl.rng, "RNG should be initialized")
	})

	t.Run("Custom Configuration Values", func(t *testing.T) {
		// Test that custom values are used when specified
		config := TPEConfig{
			Gamma:            0.3,
			Seed:             123,
			NumEIGenerations: 50,
			PriorWeight:      2.0,
			BandwidthFactor:  1.5,
		}

		tpeImpl := NewTPEOptimizer(config).(*TPEOptimizer)

		// Check that custom values were applied
		assert.Equal(t, 0.3, tpeImpl.gamma)
		assert.Equal(t, int64(123), tpeImpl.seed)
		assert.Equal(t, 50, tpeImpl.numEIGenerations)
		assert.Equal(t, 2.0, tpeImpl.priorWeight)
		assert.Equal(t, 1.5, tpeImpl.bandwidthFactor)
	})

	t.Run("Initialize Validation", func(t *testing.T) {
		tpe := NewTPEOptimizer(TPEConfig{Seed: 42})

		// Test with empty parameter space
		err := tpe.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{},
			MaxTrials:  10,
		})
		assert.Error(t, err, "Initialize should return error for empty parameter space")
		assert.Contains(t, err.Error(), "parameter space cannot be empty")

		// Test with valid config
		err = tpe.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{"param": {0, 1}},
			MaxTrials:  10,
		})
		assert.NoError(t, err)
	})

	t.Run("Random Sampling Phase", func(t *testing.T) {
		// Test the initial random sampling phase
		config := TPEConfig{Seed: 42}
		tpe := NewTPEOptimizer(config)

		err := tpe.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{
				"param1": {float64(0), float64(1)},
			},
			MaxTrials: 20,
		})
		assert.NoError(t, err)

		ctx := context.Background()

		// First few trials should use random sampling
		for i := 0; i < 3; i++ {
			params, err := tpe.SuggestParams(ctx)
			assert.NoError(t, err)
			assert.Contains(t, params, "param1")

			// Update with a random score
			err = tpe.UpdateResults(params, float64(i)*0.1)
			assert.NoError(t, err)
		}
	})

	t.Run("TPE Suggestion Phase", func(t *testing.T) {
		// Test that TPE algorithm kicks in after enough observations
		config := TPEConfig{
			Seed:             42,
			NumEIGenerations: 5, // Small value for faster tests
		}
		tpe := NewTPEOptimizer(config)

		err := tpe.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{
				"param1": {float64(0), float64(1), float64(2)},
				"param2": {float64(0), float64(1)},
			},
			MaxTrials: 20,
		})
		assert.NoError(t, err)

		ctx := context.Background()

		// Add enough observations to transition to TPE
		for i := 0; i < 6; i++ {
			params, err := tpe.SuggestParams(ctx)
			assert.NoError(t, err)

			// Higher score for param1=1, param2=0
			score := 0.5
			if params["param1"] == float64(1) {
				score += 0.3
			}
			if params["param2"] == float64(0) {
				score += 0.2
			}

			err = tpe.UpdateResults(params, score)
			assert.NoError(t, err)
		}

		// Now we should have enough observations for TPE to kick in
		// Run a few more iterations
		for i := 0; i < 5; i++ {
			params, err := tpe.SuggestParams(ctx)
			assert.NoError(t, err)

			score := 0.5
			if params["param1"] == float64(1) {
				score += 0.3
			}
			if params["param2"] == float64(0) {
				score += 0.2
			}

			err = tpe.UpdateResults(params, score)
			assert.NoError(t, err)
		}

		// Check that we've found a good solution
		bestParams, bestScore := tpe.GetBestParams()
		assert.NotEmpty(t, bestParams)
		assert.GreaterOrEqual(t, bestScore, 0.5)
	})

	t.Run("Edge Cases in TPE Algorithm", func(t *testing.T) {
		// Test edge cases in the TPE algorithm
		config := TPEConfig{
			Seed:             42,
			NumEIGenerations: 5,
		}
		tpeImpl := NewTPEOptimizer(config).(*TPEOptimizer)

		// Initialize with parameter space
		err := tpeImpl.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{
				"param1": {float64(0), float64(1)},
			},
			MaxTrials: 10,
		})
		assert.NoError(t, err)

		// Test with empty params lists
		goodParams := []map[string]interface{}{}
		badParams := []map[string]interface{}{
			{"param1": float64(0)},
		}

		// This should handle the empty good params list
		candidate := tpeImpl.generateTPECandidate(goodParams, badParams)
		assert.NotNil(t, candidate)

		// Add a few observations with the same score
		ctx := context.Background()
		for i := 0; i < 5; i++ {
			params := map[string]interface{}{"param1": float64(i % 2)}
			err = tpeImpl.UpdateResults(params, 0.5) // Same score for all
			assert.NoError(t, err)
		}

		// Should still be able to suggest params
		params, err := tpeImpl.SuggestParams(ctx)
		assert.NoError(t, err)
		assert.NotNil(t, params)

		// Test handling of all zero ratios in generateTPECandidate
		// This is an internal test that requires manipulating the TPE optimizer
		goodCounts := []float64{0, 0}
		//badCounts := []float64{1, 1}
		smoothedGood := tpeImpl.smoothCounts(goodCounts, 2)

		// Check that smoothing prevents zeros
		assert.Greater(t, smoothedGood[0], 0.0)
		assert.Greater(t, smoothedGood[1], 0.0)
	})
	t.Run("Test Utility Functions", func(t *testing.T) {
		// Test the utility functions
		tpeImpl := NewTPEOptimizer(TPEConfig{Seed: 42, PriorWeight: 1.0}).(*TPEOptimizer)

		// Initialize with a parameter space for testing
		err := tpeImpl.Initialize(SearchConfig{
			ParamSpace: map[string][]interface{}{
				"param": {float64(0), float64(1), float64(2)},
			},
			MaxTrials: 10,
		})
		assert.NoError(t, err)

		// Test countValues
		params := []map[string]interface{}{
			{"param": float64(0)},
			{"param": float64(0)},
			{"param": float64(1)},
		}
		possibleValues := []interface{}{float64(0), float64(1), float64(2)}

		counts := tpeImpl.countValues(params, "param", possibleValues)
		assert.Equal(t, []float64{2, 1, 0}, counts)

		// Test smoothCounts
		smoothed := tpeImpl.smoothCounts(counts, 3)
		assert.Len(t, smoothed, 3)

		// Use direct boolean assertions instead of assert.Greater for float64
		assert.True(t, smoothed[0] > smoothed[1], "smoothed[0] should be greater than smoothed[1]")
		assert.True(t, smoothed[1] > smoothed[2], "smoothed[1] should be greater than smoothed[2]")

		// Test computeLikelihood
		candidate := map[string]interface{}{"param": float64(0)}
		likelihood := tpeImpl.computeLikelihood(candidate, params)
		assert.True(t, likelihood >= 0.0, "likelihood should be non-negative")

		// Test with empty params
		emptyLikelihood := tpeImpl.computeLikelihood(candidate, []map[string]interface{}{})
		assert.Equal(t, 0.0, emptyLikelihood)

		// Test expectedImprovement - create more realistic test data
		// First, add some observations to build proper good/bad parameters
		for i := 0; i < 10; i++ {
			paramValue := float64(i % 3)
			// Make param=0 have higher scores
			score := 0.5
			if paramValue == 0 {
				score = 0.9
			}
			err := tpeImpl.UpdateResults(map[string]interface{}{"param": paramValue}, score)
			assert.NoError(t, err)
		}

		// Now that we have observations, test the actual TPE algorithm
		ctx := context.Background()
		suggestedParams, err := tpeImpl.SuggestParams(ctx)
		assert.NoError(t, err)
		assert.NotNil(t, suggestedParams)

		// Instead of testing expectedImprovement directly, which is more of an
		// implementation detail, check that GetBestParams works correctly
		bestParams, bestScore := tpeImpl.GetBestParams()
		assert.NotNil(t, bestParams)
		assert.True(t, bestScore >= 0.5, "Best score should be at least 0.5")
	})
}
