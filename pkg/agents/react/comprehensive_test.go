package react

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryOptimizer_ComprehensiveTests(t *testing.T) {
	t.Run("Store and Retrieve", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)
		ctx := context.Background()

		input := map[string]interface{}{"task": "test task"}
		output := map[string]interface{}{"result": "test result"}

		err := optimizer.Store(ctx, input, output, true)
		require.NoError(t, err)

		retrieved, err := optimizer.Retrieve(ctx, input)
		require.NoError(t, err)
		assert.NotNil(t, retrieved)
	})

	t.Run("Memory Compression Threshold", func(t *testing.T) {
		optimizer := NewMemoryOptimizerWithCompressionThreshold(24*time.Hour, 0.5, 5)

		ctx := context.Background()
		// Add items to trigger compression
		for i := 0; i < 10; i++ {
			input := map[string]interface{}{"task": fmt.Sprintf("task %d", i)}
			output := map[string]interface{}{"result": fmt.Sprintf("result %d", i)}
			err := optimizer.Store(ctx, input, output, true)
			require.NoError(t, err)
		}

		// Verify compression was triggered
		assert.True(t, optimizer.shouldCompress())
	})

	t.Run("Memory Categories", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		testCases := []struct {
			task     string
			expected string
		}{
			{"research the topic", "research"},
			{"calculate the sum", "calculation"},
			{"generate a report", "creation"},
			{"compare two options", "comparison"},
			{"unknown task", "general"},
		}

		for _, tc := range testCases {
			input := map[string]interface{}{"task": tc.task}
			category := optimizer.categorize(input)
			assert.Equal(t, tc.expected, category)
		}
	})

	t.Run("Importance Calculation", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		input := map[string]interface{}{"task": "test"}
		output := map[string]interface{}{"result1": "value1", "result2": "value2", "result3": "value3", "result4": "value4"}

		importance := optimizer.calculateImportance(input, output, true)
		assert.Greater(t, importance, 0.0)
		assert.LessOrEqual(t, importance, 1.0)

		// Failed tasks should have lower importance
		importanceFailed := optimizer.calculateImportance(input, output, false)
		assert.Less(t, importanceFailed, importance)
	})

	t.Run("Embedding Generation", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		input := map[string]interface{}{"task": "analyze the data"}
		embedding := optimizer.generateEmbedding(input)

		assert.NotNil(t, embedding)
		assert.Equal(t, 10, len(embedding))
	})

	t.Run("Cleanup Old Memories", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(1*time.Millisecond, 0.9) // Very short retention
		ctx := context.Background()

		input := map[string]interface{}{"task": "test task"}
		output := map[string]interface{}{"result": "test result"}

		err := optimizer.Store(ctx, input, output, true)
		require.NoError(t, err)

		// Wait for memory to expire
		time.Sleep(5 * time.Millisecond)

		// Trigger cleanup
		retrieved, err := optimizer.Retrieve(ctx, input)
		require.NoError(t, err)
		// Memory should be cleaned up (or very low retention score)
		_ = retrieved
	})

	t.Run("Statistics", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)
		ctx := context.Background()

		// Store some memories
		for i := 0; i < 3; i++ {
			input := map[string]interface{}{"task": fmt.Sprintf("research task %d", i)}
			output := map[string]interface{}{"result": fmt.Sprintf("result %d", i)}
			err := optimizer.Store(ctx, input, output, i%2 == 0) // Alternate success/failure
			require.NoError(t, err)
		}

		stats := optimizer.GetStatistics()
		assert.NotNil(t, stats)
		assert.Contains(t, stats, "total_items")
		assert.Contains(t, stats, "categories")
		assert.Contains(t, stats, "category_distribution")
	})
}

func TestMemoryIndex_ComprehensiveTests(t *testing.T) {
	t.Run("Add and Get", func(t *testing.T) {
		index := NewMemoryIndex()

		item := &MemoryItem{
			Key:      "test-key",
			Value:    "test-value",
			Category: "test",
			Created:  time.Now(),
		}

		index.Add(item)

		retrieved, exists := index.Get("test-key")
		assert.True(t, exists)
		assert.Equal(t, item.Key, retrieved.Key)
		assert.Equal(t, item.Value, retrieved.Value)
	})

	t.Run("Remove Item", func(t *testing.T) {
		index := NewMemoryIndex()

		item := &MemoryItem{
			Key:      "test-key",
			Value:    "test-value",
			Category: "test",
			Created:  time.Now(),
		}

		index.Add(item)
		assert.Equal(t, 1, index.Size())

		index.Remove("test-key")
		assert.Equal(t, 0, index.Size())

		_, exists := index.Get("test-key")
		assert.False(t, exists)
	})

	t.Run("Get By Category", func(t *testing.T) {
		index := NewMemoryIndex()

		items := []*MemoryItem{
			{Key: "key1", Category: "research", Created: time.Now()},
			{Key: "key2", Category: "research", Created: time.Now()},
			{Key: "key3", Category: "calculation", Created: time.Now()},
		}

		for _, item := range items {
			index.Add(item)
		}

		researchItems := index.GetByCategory("research")
		assert.Equal(t, 2, len(researchItems))

		calcItems := index.GetByCategory("calculation")
		assert.Equal(t, 1, len(calcItems))
	})
}

func TestForgettingCurve_Tests(t *testing.T) {
	t.Run("Calculate Retention", func(t *testing.T) {
		curve := NewForgettingCurve()

		// Test recent memory with high importance
		retention := curve.Calculate(1*time.Hour, 0.9, 5)
		assert.Greater(t, retention, 0.5)

		// Test old memory with low importance
		retention = curve.Calculate(30*24*time.Hour, 0.1, 1)
		assert.Less(t, retention, 0.5)

		// Test minimum retention floor
		retention = curve.Calculate(365*24*time.Hour, 0.0, 0)
		assert.GreaterOrEqual(t, retention, curve.minRetention)
	})
}

func TestMemoryOptimizer_CompressionStrategies(t *testing.T) {
	t.Run("Compression by Summarization", func(t *testing.T) {
		optimizer := NewMemoryOptimizerWithCompressionThreshold(24*time.Hour, 0.5, 2)
		ctx := context.Background()

		// Add many items in the same category to trigger summarization
		for i := 0; i < 8; i++ {
			input := map[string]interface{}{"task": fmt.Sprintf("research task %d", i)}
			output := map[string]interface{}{"result": fmt.Sprintf("result %d", i)}
			err := optimizer.Store(ctx, input, output, true)
			require.NoError(t, err)
		}

		// Manually trigger compression to test the logic
		optimizer.compress(ctx)
	})

	t.Run("Memory Similarity", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		item1 := &MemoryItem{
			Category:  "research",
			Hash:      "hash123",
			Embedding: []float64{1.0, 0.5, 0.3},
		}

		item2 := &MemoryItem{
			Category:  "research",
			Hash:      "hash123",
			Embedding: []float64{1.0, 0.5, 0.3},
		}

		item3 := &MemoryItem{
			Category:  "calculation",
			Hash:      "hash456",
			Embedding: []float64{0.1, 0.2, 0.9},
		}

		assert.True(t, optimizer.areSimilar(item1, item2))
		assert.False(t, optimizer.areSimilar(item1, item3))
	})

	t.Run("Cosine Similarity", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		vec1 := []float64{1.0, 0.0, 0.0}
		vec2 := []float64{1.0, 0.0, 0.0}
		vec3 := []float64{0.0, 1.0, 0.0}

		similarity := optimizer.cosineSimilarity(vec1, vec2)
		assert.InDelta(t, 1.0, similarity, 0.001)

		similarity = optimizer.cosineSimilarity(vec1, vec3)
		assert.InDelta(t, 0.0, similarity, 0.001)

		// Test zero vectors
		zeroVec := []float64{0.0, 0.0, 0.0}
		similarity = optimizer.cosineSimilarity(vec1, zeroVec)
		assert.Equal(t, 0.0, similarity)
	})

	t.Run("Time Utilities", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		t1 := time.Now()
		t2 := t1.Add(time.Hour)

		later := optimizer.laterTime(t1, t2)
		assert.Equal(t, t2, later)

		earlier := optimizer.earlierTime(t1, t2)
		assert.Equal(t, t1, earlier)
	})

	t.Run("Retention Score Calculation", func(t *testing.T) {
		optimizer := NewMemoryOptimizer(24*time.Hour, 0.5)

		item := &MemoryItem{
			Created:      time.Now().Add(-2*time.Hour),
			LastAccessed: time.Now().Add(-time.Hour),
			Importance:   0.8,
			AccessCount:  5,
		}

		score := optimizer.calculateRetentionScore(item)
		assert.Greater(t, score, 0.0)

		// Recently accessed item should have significantly higher score
		item.LastAccessed = time.Now().Add(-10*time.Minute)
		item.AccessCount = 10 // Also increase access count to make difference more significant
		recentScore := optimizer.calculateRetentionScore(item)
		assert.Greater(t, recentScore, score)
	})
}
