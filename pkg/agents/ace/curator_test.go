package ace

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello World", "hello world"},
		{"  spaces  ", "spaces"},
		{"multiple   spaces", "multiple spaces"},
		{"UPPER lower", "upper lower"},
		{"\ttabs\nand\nnewlines", "tabs and newlines"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, normalize(tt.input))
		})
	}
}

func TestTokenize(t *testing.T) {
	tokens := tokenize("Check nil after type assertion")
	expected := map[string]bool{
		"check": true, "nil": true, "after": true,
		"type": true, "assertion": true,
	}
	assert.Equal(t, expected, tokens)
}

func TestJaccardSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     map[string]bool
		expected float64
	}{
		{"identical", map[string]bool{"a": true, "b": true}, map[string]bool{"a": true, "b": true}, 1.0},
		{"disjoint", map[string]bool{"a": true}, map[string]bool{"b": true}, 0.0},
		{"half overlap", map[string]bool{"a": true, "b": true}, map[string]bool{"b": true, "c": true}, 1.0 / 3.0},
		{"empty both", map[string]bool{}, map[string]bool{}, 1.0},
		{"empty one", map[string]bool{"a": true}, map[string]bool{}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.expected, jaccardSimilarity(tt.a, tt.b), 0.001)
		})
	}
}

func TestCurator(t *testing.T) {
	config := DefaultConfig()
	config.MinConfidence = 0.5
	config.SimilarityThreshold = 0.8

	t.Run("add new insight", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))
		curator := NewCurator(config)

		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			TrajectoryID: "test-1",
			SuccessPatterns: []InsightCandidate{
				{Content: "Always check errors", Category: "strategies", Confidence: 0.9},
			},
		})

		require.NoError(t, err)
		assert.Len(t, result.Added, 1)
		assert.Equal(t, "Always check errors", result.Added[0].Content)
	})

	t.Run("deduplicate exact match", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		// Seed with existing learning
		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Check errors"},
		})
		require.NoError(t, err)

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			SuccessPatterns: []InsightCandidate{
				{Content: "Check errors", Confidence: 0.9},
			},
		})

		require.NoError(t, err)
		assert.Empty(t, result.Added)
	})

	t.Run("deduplicate normalized match", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Check errors"},
		})
		require.NoError(t, err)

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			SuccessPatterns: []InsightCandidate{
				{Content: "  CHECK   ERRORS  ", Confidence: 0.9},
			},
		})

		require.NoError(t, err)
		assert.Empty(t, result.Added)
	})

	t.Run("deduplicate similar content", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Always check nil after type assertion", Helpful: 3},
		})
		require.NoError(t, err)

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			SuccessPatterns: []InsightCandidate{
				{Content: "Check nil after type assertion always", Confidence: 0.9},
			},
		})

		require.NoError(t, err)
		assert.Empty(t, result.Added)
		assert.Len(t, result.Merged, 1)
	})

	t.Run("apply learning updates", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Test", Helpful: 5, Harmful: 1},
		})
		require.NoError(t, err)

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			LearningUpdates: []LearningUpdate{
				{LearningID: "strategies-00001", Delta: DeltaHelpful},
				{LearningID: "L001", Delta: DeltaHarmful}, // short code
			},
		})

		require.NoError(t, err)
		assert.Len(t, result.Updated, 2)

		// Verify file was updated
		learnings, _ := file.Load()
		require.Len(t, learnings, 1)
		assert.Equal(t, 6, learnings[0].Helpful)
		assert.Equal(t, 2, learnings[0].Harmful)
	})

	t.Run("prune low performers", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		config := DefaultConfig()
		config.PruneMinRatio = 0.3
		config.PruneMinUsage = 5

		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Good", Helpful: 8, Harmful: 2},
			{ID: "strategies-00002", Category: "strategies", Content: "Bad", Helpful: 1, Harmful: 9},
		})
		require.NoError(t, err)

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{})

		require.NoError(t, err)
		assert.Contains(t, result.Pruned, "strategies-00002")

		learnings, _ := file.Load()
		assert.Len(t, learnings, 1)
		assert.Equal(t, "strategies-00001", learnings[0].ID)
	})

	t.Run("filter low confidence", func(t *testing.T) {
		tmpDir := t.TempDir()
		file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

		config := DefaultConfig()
		config.MinConfidence = 0.7

		curator := NewCurator(config)
		result, err := curator.Curate(context.Background(), file, &ReflectionResult{
			SuccessPatterns: []InsightCandidate{
				{Content: "High confidence", Confidence: 0.9},
				{Content: "Low confidence", Confidence: 0.5},
			},
		})

		require.NoError(t, err)
		assert.Len(t, result.Added, 1)
		assert.Equal(t, "High confidence", result.Added[0].Content)
	})
}

func TestCurationResult(t *testing.T) {
	result := &CurationResult{
		FilePath:     "/test/path",
		TokensBefore: 1000,
		TokensAfter:  900,
		ProcessedAt:  time.Now(),
	}

	assert.Equal(t, "/test/path", result.FilePath)
	assert.Equal(t, 1000, result.TokensBefore)
	assert.Equal(t, 900, result.TokensAfter)
}

func TestAggressivePrune(t *testing.T) {
	config := DefaultConfig()
	curator := NewCurator(config)

	t.Run("empty list", func(t *testing.T) {
		kept, pruned := curator.aggressivePrune([]Learning{}, 100)
		assert.Empty(t, kept)
		assert.Empty(t, pruned)
	})

	t.Run("prune lowest performers first", func(t *testing.T) {
		learnings := []Learning{
			{ID: "s-1", Helpful: 8, Harmful: 2, Content: "Good one"},    // 80%
			{ID: "s-2", Helpful: 1, Harmful: 9, Content: "Bad one"},     // 10%
			{ID: "s-3", Helpful: 5, Harmful: 5, Content: "Average one"}, // 50%
		}

		// Token counts: s-1=9, s-2=9, s-3=10 (based on String() length / 4)
		// Request exactly 9 tokens to prune just s-2 (the worst performer)
		kept, pruned := curator.aggressivePrune(learnings, 9)

		// Should prune the worst one first (s-2)
		assert.Len(t, pruned, 1)
		assert.Equal(t, "s-2", pruned[0])
		assert.Len(t, kept, 2)
	})

	t.Run("prune multiple until target", func(t *testing.T) {
		learnings := []Learning{
			{ID: "s-1", Helpful: 9, Harmful: 1, Content: "Best"},
			{ID: "s-2", Helpful: 1, Harmful: 9, Content: "Worst"},
			{ID: "s-3", Helpful: 2, Harmful: 8, Content: "Bad"},
			{ID: "s-4", Helpful: 5, Harmful: 5, Content: "Medium"},
		}

		// Request removal of many tokens to force pruning all
		kept, pruned := curator.aggressivePrune(learnings, 1000)

		// Should prune all of them (worst first) to reach target
		assert.Len(t, pruned, 4)
		assert.Empty(t, kept)
	})

	t.Run("stop when target reached", func(t *testing.T) {
		learnings := []Learning{
			{ID: "s-1", Helpful: 9, Harmful: 1, Content: "Best"},
			{ID: "s-2", Helpful: 1, Harmful: 9, Content: "Worst"},
		}

		// Token counts: s-1=8, s-2=8 (based on String() length / 4)
		// Request exactly 8 tokens to prune just s-2 (the worst performer)
		kept, pruned := curator.aggressivePrune(learnings, 8)

		// Should prune the worst one first and stop
		assert.Len(t, pruned, 1)
		assert.Equal(t, "s-2", pruned[0])
		assert.Len(t, kept, 1)
	})
}

func TestCuratorNilReflectionResult(t *testing.T) {
	config := DefaultConfig()
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))
	curator := NewCurator(config)

	// Should handle nil gracefully
	result, err := curator.Curate(context.Background(), file, nil)
	require.NoError(t, err)
	assert.NotNil(t, result)
}

func TestCuratorEmptyReflectionResult(t *testing.T) {
	config := DefaultConfig()
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))
	curator := NewCurator(config)

	result, err := curator.Curate(context.Background(), file, &ReflectionResult{})
	require.NoError(t, err)
	assert.Empty(t, result.Added)
	assert.Empty(t, result.Updated)
}

func TestCuratorFailurePatterns(t *testing.T) {
	config := DefaultConfig()
	config.MinConfidence = 0.5
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))
	curator := NewCurator(config)

	result, err := curator.Curate(context.Background(), file, &ReflectionResult{
		FailurePatterns: []InsightCandidate{
			{Content: "Don't forget to close files", Category: "mistakes", Confidence: 0.8},
		},
	})

	require.NoError(t, err)
	assert.Len(t, result.Added, 1)
	assert.Equal(t, "mistakes", result.Added[0].Category)
}

func TestCuratorUnknownLearningUpdate(t *testing.T) {
	config := DefaultConfig()
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

	// Save one learning
	err := file.Save([]Learning{
		{ID: "strategies-00001", Content: "Test", Helpful: 1, Harmful: 0},
	})
	require.NoError(t, err)

	curator := NewCurator(config)
	result, err := curator.Curate(context.Background(), file, &ReflectionResult{
		LearningUpdates: []LearningUpdate{
			{LearningID: "unknown-99999", Delta: DeltaHelpful},
		},
	})

	require.NoError(t, err)
	// Unknown ID should be skipped
	assert.Empty(t, result.Updated)
}

func TestNewCuratorDefaultConfig(t *testing.T) {
	config := DefaultConfig()
	curator := NewCurator(config)
	assert.NotNil(t, curator)
	assert.Equal(t, config.MinConfidence, curator.config.MinConfidence)
}

func TestNewCuratorZeroSimilarityThreshold(t *testing.T) {
	// Test that zero similarity threshold gets set to default
	config := DefaultConfig()
	config.SimilarityThreshold = 0 // Zero should trigger default

	curator := NewCurator(config)
	assert.NotNil(t, curator)
	assert.Equal(t, 0.85, curator.similarityThreshold)
}

func TestNewCuratorNegativeSimilarityThreshold(t *testing.T) {
	// Test that negative similarity threshold gets set to default
	config := DefaultConfig()
	config.SimilarityThreshold = -0.5

	curator := NewCurator(config)
	assert.NotNil(t, curator)
	assert.Equal(t, 0.85, curator.similarityThreshold)
}

func TestCuratorAddOrMergeExactDuplicate(t *testing.T) {
	config := DefaultConfig()
	config.MinConfidence = 0.5
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

	// Seed with existing learning
	err := file.Save([]Learning{
		{ID: "strategies-00001", Category: "strategies", Content: "Always check errors before using result"},
	})
	require.NoError(t, err)

	curator := NewCurator(config)
	result, err := curator.Curate(context.Background(), file, &ReflectionResult{
		SuccessPatterns: []InsightCandidate{
			{Content: "Always check errors before using result", Confidence: 0.9}, // Exact match
		},
	})

	require.NoError(t, err)
	assert.Empty(t, result.Added) // Should not add duplicate
}

func TestCuratorTokenBudgetAggressive(t *testing.T) {
	config := DefaultConfig()
	config.MaxTokens = 10 // Very small budget to force aggressive pruning
	config.MinConfidence = 0.5
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

	curator := NewCurator(config)

	// Add multiple learnings that will exceed the token budget
	result, err := curator.Curate(context.Background(), file, &ReflectionResult{
		SuccessPatterns: []InsightCandidate{
			{Content: "First long learning content that takes many tokens", Confidence: 0.9},
			{Content: "Second long learning content that takes many tokens", Confidence: 0.8},
			{Content: "Third long learning content that takes many tokens", Confidence: 0.7},
		},
	})

	require.NoError(t, err)
	// Some learnings should have been added
	assert.NotEmpty(t, result.Added)
}

func TestCuratorInsightWithEmptyCategory(t *testing.T) {
	config := DefaultConfig()
	config.MinConfidence = 0.5
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))
	curator := NewCurator(config)

	result, err := curator.Curate(context.Background(), file, &ReflectionResult{
		SuccessPatterns: []InsightCandidate{
			{Content: "Insight without category", Category: "", Confidence: 0.9},
		},
	})

	require.NoError(t, err)
	assert.Len(t, result.Added, 1)
	// Should default to "strategies" category
	assert.Equal(t, "strategies", result.Added[0].Category)
}

func TestJaccardSimilarityEdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		a, b     map[string]bool
		expected float64
	}{
		{"single common", map[string]bool{"a": true}, map[string]bool{"a": true}, 1.0},
		{"subset", map[string]bool{"a": true, "b": true}, map[string]bool{"a": true, "b": true, "c": true}, 2.0 / 3.0},
		{"large sets", map[string]bool{"a": true, "b": true, "c": true, "d": true}, map[string]bool{"a": true, "b": true, "e": true, "f": true}, 2.0 / 6.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := jaccardSimilarity(tt.a, tt.b)
			assert.InDelta(t, tt.expected, result, 0.001)
		})
	}
}
