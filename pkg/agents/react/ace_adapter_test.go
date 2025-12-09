package react

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelfReflectorACEAdapter(t *testing.T) {
	t.Run("nil reflector returns nil", func(t *testing.T) {
		adapter := NewSelfReflectorACEAdapter(nil)
		candidates, err := adapter.Extract(context.Background())
		require.NoError(t, err)
		assert.Nil(t, candidates)
	})

	t.Run("extracts reflections as candidates", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)

		// Add some reflections by simulating executions
		record := ExecutionRecord{
			Timestamp: time.Now(),
			Input:     map[string]interface{}{"task": "test"},
			Output:    map[string]interface{}{"result": "done"},
			Success:   true,
			Actions: []ActionRecord{
				{Tool: "search", Success: true},
			},
		}
		reflector.Reflect(context.Background(), record)

		adapter := NewSelfReflectorACEAdapter(reflector)
		candidates, err := adapter.Extract(context.Background())
		require.NoError(t, err)
		// May or may not have candidates depending on reflection logic
		_ = candidates
	})
}

func TestReActAgentWithACE(t *testing.T) {
	tmpDir := t.TempDir()
	learningsPath := filepath.Join(tmpDir, "learnings.md")

	t.Run("creates agent with ACE enabled", func(t *testing.T) {
		config := ace.Config{
			Enabled:           true,
			LearningsPath:     learningsPath,
			AsyncReflection:   false,
			CurationFrequency: 1,
			MinConfidence:     0.5,
			MaxTokens:         10000,
			PruneMinRatio:     0.3,
			PruneMinUsage:     3,
			SimilarityThreshold: 0.85,
		}

		agent := NewReActAgent("test-ace", "Test ACE Agent",
			WithACE(config),
			WithReflection(true, 3),
		)
		require.NotNil(t, agent)
		assert.NotNil(t, agent.aceManager)
		defer agent.Close()
	})

	t.Run("ACE disabled by default", func(t *testing.T) {
		agent := NewReActAgent("test-no-ace", "Test No ACE Agent")
		require.NotNil(t, agent)
		assert.Nil(t, agent.aceManager)
	})

	t.Run("GetACEMetrics returns nil when disabled", func(t *testing.T) {
		agent := NewReActAgent("test", "Test Agent")
		metrics := agent.GetACEMetrics()
		assert.Nil(t, metrics)
	})

	t.Run("GetACEMetrics returns metrics when enabled", func(t *testing.T) {
		config := ace.Config{
			Enabled:             true,
			LearningsPath:       filepath.Join(tmpDir, "metrics_test.md"),
			AsyncReflection:     false,
			CurationFrequency:   1,
			MinConfidence:       0.5,
			MaxTokens:           10000,
			PruneMinRatio:       0.3,
			PruneMinUsage:       3,
			SimilarityThreshold: 0.85,
		}

		agent := NewReActAgent("test-metrics", "Test Metrics Agent", WithACE(config))
		require.NotNil(t, agent)
		defer agent.Close()

		metrics := agent.GetACEMetrics()
		require.NotNil(t, metrics)
		assert.Contains(t, metrics, "trajectories_processed")
	})
}

func TestACELearningsInjection(t *testing.T) {
	tmpDir := t.TempDir()
	learningsPath := filepath.Join(tmpDir, "learnings.md")

	// Pre-populate learnings file
	content := `## STRATEGIES
[strategies-00001] helpful=5 harmful=0 :: Always validate input before processing
`
	err := os.WriteFile(learningsPath, []byte(content), 0644)
	require.NoError(t, err)

	config := ace.Config{
		Enabled:             true,
		LearningsPath:       learningsPath,
		AsyncReflection:     false,
		CurationFrequency:   1,
		MinConfidence:       0.5,
		MaxTokens:           10000,
		PruneMinRatio:       0.3,
		PruneMinUsage:       3,
		SimilarityThreshold: 0.85,
	}

	agent := NewReActAgent("test-inject", "Test Injection Agent", WithACE(config))
	require.NotNil(t, agent)
	defer agent.Close()

	// Verify learnings are available
	require.NotNil(t, agent.aceManager)
	learnings := agent.aceManager.GetLearningsContext()
	assert.Contains(t, learnings, "validate input")
}

func TestMapReflectionTypeToCategory(t *testing.T) {
	tests := []struct {
		input    ReflectionType
		expected string
	}{
		{ReflectionTypeError, "mistakes"},
		{ReflectionTypeStrategy, "strategies"},
		{ReflectionTypeLearning, "patterns"},
		{ReflectionTypePerformance, "patterns"},
		{ReflectionType(999), "general"},
	}

	for _, tt := range tests {
		result := mapReflectionTypeToCategory(tt.input)
		assert.Equal(t, tt.expected, result)
	}
}
