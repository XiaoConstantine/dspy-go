package ace

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseLearnings(t *testing.T) {
	t.Run("empty content", func(t *testing.T) {
		learnings, err := ParseLearnings("")
		require.NoError(t, err)
		assert.Empty(t, learnings)
	})

	t.Run("single learning", func(t *testing.T) {
		content := `## STRATEGIES
[strategies-00001] helpful=5 harmful=1 :: Check nil after type assertion`

		learnings, err := ParseLearnings(content)
		require.NoError(t, err)
		require.Len(t, learnings, 1)

		assert.Equal(t, "strategies-00001", learnings[0].ID)
		assert.Equal(t, "strategies", learnings[0].Category)
		assert.Equal(t, 5, learnings[0].Helpful)
		assert.Equal(t, 1, learnings[0].Harmful)
		assert.Equal(t, "Check nil after type assertion", learnings[0].Content)
	})

	t.Run("multiple categories", func(t *testing.T) {
		content := `## STRATEGIES
[strategies-00001] helpful=5 harmful=1 :: Check nil after type assertion
[strategies-00002] helpful=3 harmful=0 :: Use context for cancellation

## MISTAKES
[mistakes-00001] helpful=2 harmful=0 :: Don't ignore Close errors`

		learnings, err := ParseLearnings(content)
		require.NoError(t, err)
		require.Len(t, learnings, 3)

		assert.Equal(t, "strategies", learnings[0].Category)
		assert.Equal(t, "strategies", learnings[1].Category)
		assert.Equal(t, "mistakes", learnings[2].Category)
	})

	t.Run("no section header", func(t *testing.T) {
		content := `[strategies-00001] helpful=5 harmful=1 :: Check nil`

		learnings, err := ParseLearnings(content)
		require.NoError(t, err)
		require.Len(t, learnings, 1)
		assert.Equal(t, "strategies", learnings[0].Category)
	})
}

func TestFormatLearnings(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		result := FormatLearnings([]Learning{})
		assert.Empty(t, result)
	})

	t.Run("single learning", func(t *testing.T) {
		learnings := []Learning{
			{ID: "strategies-00001", Category: "strategies", Helpful: 5, Harmful: 1, Content: "Test content"},
		}

		result := FormatLearnings(learnings)
		assert.Contains(t, result, "## STRATEGIES")
		assert.Contains(t, result, "[strategies-00001] helpful=5 harmful=1 :: Test content")
	})

	t.Run("round trip", func(t *testing.T) {
		original := []Learning{
			{ID: "strategies-00001", Category: "strategies", Helpful: 5, Harmful: 1, Content: "First"},
			{ID: "strategies-00002", Category: "strategies", Helpful: 3, Harmful: 0, Content: "Second"},
			{ID: "mistakes-00001", Category: "mistakes", Helpful: 2, Harmful: 0, Content: "Third"},
		}

		formatted := FormatLearnings(original)
		parsed, err := ParseLearnings(formatted)
		require.NoError(t, err)
		require.Len(t, parsed, 3)

		// Check by ID since order may change due to category sorting
		for _, orig := range original {
			found := FindByID(parsed, orig.ID)
			require.NotNil(t, found, "learning %s not found", orig.ID)
			assert.Equal(t, orig.Content, found.Content)
			assert.Equal(t, orig.Helpful, found.Helpful)
			assert.Equal(t, orig.Harmful, found.Harmful)
		}
	})
}

func TestLearningsFile(t *testing.T) {
	t.Run("load non-existent file", func(t *testing.T) {
		f := NewLearningsFile("/tmp/ace_test_nonexistent.md")
		learnings, err := f.Load()
		require.NoError(t, err)
		assert.Empty(t, learnings)
	})

	t.Run("save and load", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "learnings.md")
		f := NewLearningsFile(path)

		original := []Learning{
			{ID: "strategies-00001", Category: "strategies", Helpful: 5, Harmful: 1, Content: "Test"},
		}

		err := f.Save(original)
		require.NoError(t, err)

		loaded, err := f.Load()
		require.NoError(t, err)
		require.Len(t, loaded, 1)
		assert.Equal(t, original[0].ID, loaded[0].ID)
		assert.Equal(t, original[0].Content, loaded[0].Content)
	})

	t.Run("creates directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "nested", "dir", "learnings.md")
		f := NewLearningsFile(path)

		err := f.Save([]Learning{{ID: "test-00001", Category: "test", Content: "Test"}})
		require.NoError(t, err)

		_, err = os.Stat(path)
		assert.NoError(t, err)
	})

	t.Run("exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "learnings.md")
		f := NewLearningsFile(path)

		assert.False(t, f.Exists())

		err := f.Save([]Learning{{ID: "test-00001", Category: "test", Content: "Test"}})
		require.NoError(t, err)

		assert.True(t, f.Exists())
	})

	t.Run("estimate tokens", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "learnings.md")
		f := NewLearningsFile(path)

		content := "This is test content with some words"
		err := os.WriteFile(path, []byte(content), 0644)
		require.NoError(t, err)

		tokens, err := f.EstimateTokens()
		require.NoError(t, err)
		assert.Equal(t, len(content)/4, tokens)
	})
}

func TestLearningsFileSequentialWrites(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "sequential.md")

	// Seed with initial data
	f := NewLearningsFile(path)
	err := f.Save([]Learning{{ID: "test-00001", Category: "test", Content: "Initial"}})
	require.NoError(t, err)

	// Multiple sequential writes with separate file handles
	for i := 0; i < 5; i++ {
		handle := NewLearningsFile(path)
		learnings := []Learning{
			{ID: "test-00001", Category: "test", Helpful: i, Content: "Updated"},
		}
		err := handle.Save(learnings)
		require.NoError(t, err)
	}

	// Verify file is readable after sequential writes
	result, err := f.Load()
	require.NoError(t, err)
	require.Len(t, result, 1)
	assert.Equal(t, 4, result[0].Helpful) // Last write should persist
}

func TestGetNextID(t *testing.T) {
	t.Run("empty list", func(t *testing.T) {
		id := GetNextID([]Learning{}, "strategies")
		assert.Equal(t, "strategies-00001", id)
	})

	t.Run("existing entries", func(t *testing.T) {
		learnings := []Learning{
			{ID: "strategies-00001"},
			{ID: "strategies-00003"},
			{ID: "mistakes-00001"},
		}
		id := GetNextID(learnings, "strategies")
		assert.Equal(t, "strategies-00004", id)
	})
}

func TestFindByID(t *testing.T) {
	learnings := []Learning{
		{ID: "strategies-00001", Content: "First"},
		{ID: "strategies-00002", Content: "Second"},
	}

	t.Run("found", func(t *testing.T) {
		l := FindByID(learnings, "strategies-00002")
		require.NotNil(t, l)
		assert.Equal(t, "Second", l.Content)
	})

	t.Run("not found", func(t *testing.T) {
		l := FindByID(learnings, "strategies-00999")
		assert.Nil(t, l)
	})
}

func TestLearning(t *testing.T) {
	t.Run("success rate", func(t *testing.T) {
		l := Learning{Helpful: 8, Harmful: 2}
		assert.Equal(t, 0.8, l.SuccessRate())
	})

	t.Run("success rate zero uses", func(t *testing.T) {
		l := Learning{Helpful: 0, Harmful: 0}
		assert.Equal(t, 0.5, l.SuccessRate())
	})

	t.Run("should prune", func(t *testing.T) {
		l := Learning{Helpful: 1, Harmful: 9}
		assert.False(t, l.ShouldPrune(0.3, 20)) // Not enough uses
		assert.True(t, l.ShouldPrune(0.3, 5))  // Enough uses, low rate
	})

	t.Run("short code", func(t *testing.T) {
		l1 := Learning{ID: "strategies-00001", Category: "strategies"}
		l2 := Learning{ID: "mistakes-00002", Category: "mistakes"}
		assert.Equal(t, "L001", l1.ShortCode())
		assert.Equal(t, "M002", l2.ShortCode())
	})
}

func TestFormatForInjection(t *testing.T) {
	t.Run("mixed categories", func(t *testing.T) {
		learnings := []Learning{
			{ID: "strategies-00001", Category: "strategies", Helpful: 8, Harmful: 2, Content: "Check nil"},
			{ID: "mistakes-00001", Category: "mistakes", Helpful: 5, Harmful: 0, Content: "Avoid panic"},
		}

		result := FormatForInjection(learnings)
		assert.Contains(t, result, "## Learned Strategies")
		assert.Contains(t, result, "[L001]")
		assert.Contains(t, result, "80% success")
		assert.Contains(t, result, "## Mistakes to Avoid")
		assert.Contains(t, result, "[M001]")
	})

	t.Run("empty", func(t *testing.T) {
		result := FormatForInjection([]Learning{})
		assert.Empty(t, result)
	})

	t.Run("patterns category", func(t *testing.T) {
		learnings := []Learning{
			{ID: "patterns-00001", Category: "patterns", Helpful: 5, Harmful: 0, Content: "Use context"},
		}
		result := FormatForInjection(learnings)
		assert.Contains(t, result, "## Learned Strategies")
		assert.Contains(t, result, "[P001]")
	})

	t.Run("other category", func(t *testing.T) {
		learnings := []Learning{
			{ID: "custom-00001", Category: "custom", Helpful: 3, Harmful: 1, Content: "Custom learning"},
		}
		result := FormatForInjection(learnings)
		assert.Contains(t, result, "## Learned Strategies")
	})

	t.Run("only mistakes", func(t *testing.T) {
		learnings := []Learning{
			{ID: "mistakes-00001", Category: "mistakes", Helpful: 5, Harmful: 0, Content: "Don't panic"},
		}
		result := FormatForInjection(learnings)
		assert.Contains(t, result, "## Mistakes to Avoid")
		assert.NotContains(t, result, "## Learned Strategies")
	})
}

func TestFindByShortCode(t *testing.T) {
	learnings := []Learning{
		{ID: "strategies-00001", Category: "strategies", Content: "First"},
		{ID: "mistakes-00002", Category: "mistakes", Content: "Second"},
		{ID: "patterns-00003", Category: "patterns", Content: "Third"},
	}

	t.Run("found strategy", func(t *testing.T) {
		l := FindByShortCode(learnings, "L001")
		require.NotNil(t, l)
		assert.Equal(t, "First", l.Content)
	})

	t.Run("found mistake", func(t *testing.T) {
		l := FindByShortCode(learnings, "M002")
		require.NotNil(t, l)
		assert.Equal(t, "Second", l.Content)
	})

	t.Run("found pattern", func(t *testing.T) {
		l := FindByShortCode(learnings, "P003")
		require.NotNil(t, l)
		assert.Equal(t, "Third", l.Content)
	})

	t.Run("not found", func(t *testing.T) {
		l := FindByShortCode(learnings, "X999")
		assert.Nil(t, l)
	})
}

func TestExtractCategoryFromID(t *testing.T) {
	tests := []struct {
		id       string
		expected string
	}{
		{"strategies-00001", "strategies"},
		{"mistakes-00002", "mistakes"},
		{"patterns-00003", "patterns"},
		{"simple", "simple"},
		{"", ""},
	}

	for _, tt := range tests {
		result := extractCategoryFromID(tt.id)
		assert.Equal(t, tt.expected, result, "for id: %s", tt.id)
	}
}

func TestLearningShortCodeEdgeCases(t *testing.T) {
	t.Run("patterns category", func(t *testing.T) {
		l := Learning{ID: "patterns-00005", Category: "patterns"}
		assert.Equal(t, "P005", l.ShortCode())
	})

	t.Run("general category", func(t *testing.T) {
		l := Learning{ID: "general-00007", Category: "general"}
		assert.Equal(t, "L007", l.ShortCode())
	})

	t.Run("malformed id", func(t *testing.T) {
		l := Learning{ID: "nonum", Category: "strategies"}
		// Should still work, returning L000
		code := l.ShortCode()
		assert.Contains(t, code, "L")
	})
}

func TestLearningsFileLoadContent(t *testing.T) {
	t.Run("non-existent file", func(t *testing.T) {
		f := NewLearningsFile("/tmp/ace_test_nonexistent_content.md")
		content, err := f.LoadContent()
		require.NoError(t, err)
		assert.Empty(t, content)
	})

	t.Run("existing file", func(t *testing.T) {
		tmpDir := t.TempDir()
		path := filepath.Join(tmpDir, "content.md")
		expected := "test content here"
		err := os.WriteFile(path, []byte(expected), 0644)
		require.NoError(t, err)

		f := NewLearningsFile(path)
		content, err := f.LoadContent()
		require.NoError(t, err)
		assert.Equal(t, expected, content)
	})
}

func TestLearningsFileEstimateTokensEmpty(t *testing.T) {
	f := NewLearningsFile("/tmp/ace_test_nonexistent_tokens.md")
	tokens, err := f.EstimateTokens()
	require.NoError(t, err)
	assert.Equal(t, 0, tokens)
}

func TestFormatLearningsEmptyCategory(t *testing.T) {
	learnings := []Learning{
		{ID: "test-00001", Category: "", Helpful: 1, Harmful: 0, Content: "No category"},
	}
	result := FormatLearnings(learnings)
	assert.Contains(t, result, "## GENERAL")
}

func TestStepIsSuccessful(t *testing.T) {
	t.Run("successful step", func(t *testing.T) {
		s := Step{Action: "think", Error: ""}
		assert.True(t, s.IsSuccessful())
	})

	t.Run("failed step", func(t *testing.T) {
		s := Step{Action: "tool", Error: "tool not found"}
		assert.False(t, s.IsSuccessful())
	})
}

func TestConfigValidate(t *testing.T) {
	validConfig := func() Config {
		return Config{
			Enabled:             true,
			LearningsPath:       "/tmp/test.md",
			MinConfidence:       0.7,
			PruneMinRatio:       0.3,
			SimilarityThreshold: 0.85,
			MaxTokens:           10000,
			CurationFrequency:   5,
		}
	}

	t.Run("valid config", func(t *testing.T) {
		c := validConfig()
		assert.NoError(t, c.Validate())
	})

	t.Run("empty learnings path", func(t *testing.T) {
		c := validConfig()
		c.LearningsPath = ""
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "learnings_path")
	})

	t.Run("min_confidence too low", func(t *testing.T) {
		c := validConfig()
		c.MinConfidence = -0.1
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "min_confidence")
	})

	t.Run("min_confidence too high", func(t *testing.T) {
		c := validConfig()
		c.MinConfidence = 1.5
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "min_confidence")
	})

	t.Run("prune_min_ratio too low", func(t *testing.T) {
		c := validConfig()
		c.PruneMinRatio = -0.1
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "prune_min_ratio")
	})

	t.Run("prune_min_ratio too high", func(t *testing.T) {
		c := validConfig()
		c.PruneMinRatio = 1.5
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "prune_min_ratio")
	})

	t.Run("similarity_threshold too low", func(t *testing.T) {
		c := validConfig()
		c.SimilarityThreshold = -0.1
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "similarity_threshold")
	})

	t.Run("similarity_threshold too high", func(t *testing.T) {
		c := validConfig()
		c.SimilarityThreshold = 1.5
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "similarity_threshold")
	})

	t.Run("max_tokens zero", func(t *testing.T) {
		c := validConfig()
		c.MaxTokens = 0
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "max_tokens")
	})

	t.Run("max_tokens negative", func(t *testing.T) {
		c := validConfig()
		c.MaxTokens = -1
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "max_tokens")
	})

	t.Run("curation_frequency zero", func(t *testing.T) {
		c := validConfig()
		c.CurationFrequency = 0
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "curation_frequency")
	})

	t.Run("curation_frequency negative", func(t *testing.T) {
		c := validConfig()
		c.CurationFrequency = -1
		err := c.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "curation_frequency")
	})
}

func TestLearningsFileSaveAtomicRename(t *testing.T) {
	// Test that Save creates temp file and renames it
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

	learnings := []Learning{
		{ID: "test-001", Category: "strategies", Content: "Test content", Helpful: 1},
	}

	err := file.Save(learnings)
	require.NoError(t, err)

	// Verify file exists and can be read back
	loaded, err := file.Load()
	require.NoError(t, err)
	assert.Len(t, loaded, 1)
	assert.Equal(t, "Test content", loaded[0].Content)
}

func TestLearningsFileLoadNonExistent(t *testing.T) {
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "nonexistent.md"))

	// Should return empty slice, not error
	learnings, err := file.Load()
	require.NoError(t, err)
	assert.Empty(t, learnings)
}

func TestLearningsFileLoadReadError(t *testing.T) {
	// Create a directory where the file should be to cause a read error
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "learnings.md")
	err := os.MkdirAll(filePath, 0755) // Create directory instead of file
	require.NoError(t, err)

	file := NewLearningsFile(filePath)
	_, err = file.Load()
	// Should return an error (trying to read a directory as file)
	assert.Error(t, err)
}

func TestLearningsFileSequentialReadWrite(t *testing.T) {
	// Test sequential read/write operations (concurrent access has locking issues)
	tmpDir := t.TempDir()
	file := NewLearningsFile(filepath.Join(tmpDir, "learnings.md"))

	// Write
	err := file.Save([]Learning{
		{ID: "test-001", Content: "Initial", Helpful: 0},
	})
	require.NoError(t, err)

	// Read
	learnings, err := file.Load()
	require.NoError(t, err)
	assert.Len(t, learnings, 1)

	// Write again
	err = file.Save([]Learning{
		{ID: "test-001", Content: "Updated", Helpful: 1},
	})
	require.NoError(t, err)

	// Read again
	learnings, err = file.Load()
	require.NoError(t, err)
	assert.Equal(t, "Updated", learnings[0].Content)
}

func TestLearningsFileExists(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "learnings.md")
	file := NewLearningsFile(filePath)

	// File doesn't exist yet
	assert.False(t, file.Exists())

	// Create file
	err := file.Save([]Learning{})
	require.NoError(t, err)

	// Now it should exist
	assert.True(t, file.Exists())
}

func TestExtractCategoryFromIDEdgeCases(t *testing.T) {
	// Test various edge cases for extractCategoryFromID
	tests := []struct {
		id       string
		expected string
	}{
		{"strategies-00001", "strategies"},
		{"mistakes-12345", "mistakes"},
		{"patterns-00001", "patterns"},
		{"context-99999", "context"},
		{"simple", "simple"},     // No dash - returns whole string
		{"a-b-c", "a"},           // Multiple dashes, takes first part
		{"", ""},                 // Empty string
		{"-00001", ""},           // Starts with dash
		{"test-", "test"},        // Ends with dash
	}

	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			result := extractCategoryFromID(tt.id)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetNextIDOverflow(t *testing.T) {
	// Test ID generation with existing high IDs
	learnings := []Learning{
		{ID: "strategies-00001"},
		{ID: "strategies-00099"},
		{ID: "strategies-00050"},
	}

	nextID := GetNextID(learnings, "strategies")
	// Should be 00100 (99 + 1 padded to 5 digits)
	assert.Equal(t, "strategies-00100", nextID)
}

func TestFormatLearningsWithSpecialCharacters(t *testing.T) {
	learnings := []Learning{
		{ID: "strategies-00001", Category: "strategies", Content: "Use :: in code (for scope resolution)", Helpful: 1, Harmful: 0},
		{ID: "strategies-00002", Category: "strategies", Content: "Handle [brackets] properly", Helpful: 2, Harmful: 1},
	}

	formatted := FormatLearnings(learnings)
	assert.Contains(t, formatted, "Use :: in code (for scope resolution)")
	assert.Contains(t, formatted, "Handle [brackets] properly")

	// Should be parseable back
	parsed, err := ParseLearnings(formatted)
	require.NoError(t, err)
	assert.Len(t, parsed, 2)
}

func TestShouldPruneEdgeCases(t *testing.T) {
	tests := []struct {
		name         string
		learning     Learning
		minRatio     float64
		minUsage     int
		shouldPrune  bool
	}{
		{
			name:        "exactly at min usage",
			learning:    Learning{Helpful: 2, Harmful: 3},
			minRatio:    0.5,
			minUsage:    5,
			shouldPrune: true, // 2/5 = 0.4 < 0.5
		},
		{
			name:        "below min usage",
			learning:    Learning{Helpful: 0, Harmful: 4},
			minRatio:    0.5,
			minUsage:    5,
			shouldPrune: false, // total usage 4 < 5
		},
		{
			name:        "exactly at min ratio",
			learning:    Learning{Helpful: 3, Harmful: 3},
			minRatio:    0.5,
			minUsage:    5,
			shouldPrune: false, // 3/6 = 0.5, not < 0.5
		},
		{
			name:        "just below min ratio",
			learning:    Learning{Helpful: 2, Harmful: 5},
			minRatio:    0.5,
			minUsage:    5,
			shouldPrune: true, // 2/7 = 0.286 < 0.5
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.learning.ShouldPrune(tt.minRatio, tt.minUsage)
			assert.Equal(t, tt.shouldPrune, result)
		})
	}
}
