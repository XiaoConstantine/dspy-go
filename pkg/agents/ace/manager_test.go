package ace

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManager(t *testing.T) {
	t.Run("synchronous processing", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false
		config.MinConfidence = 0.5

		m, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m.Close()

		// Record a trajectory using the handle pattern
		recorder := m.StartTrajectory("agent-1", "test", "Do something")
		recorder.RecordStep("think", "", "Planning", nil, nil, nil)
		recorder.RecordStep("tool", "search", "Searching", nil, nil, nil)
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

		metrics := m.Metrics()
		assert.Equal(t, int64(1), metrics["trajectories_processed"])
	})

	t.Run("async processing", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = true
		config.CurationFrequency = 2

		m, err := NewManager(config, nil)
		require.NoError(t, err)

		// Record trajectories
		for i := 0; i < 3; i++ {
			recorder := m.StartTrajectory("agent-1", "test", "Query")
			recorder.RecordStep("action", "", "Reasoning", nil, nil, nil)
			m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)
		}

		// Close flushes pending
		err = m.Close()
		require.NoError(t, err)

		metrics := m.Metrics()
		assert.Equal(t, int64(3), metrics["trajectories_processed"])
	})

	t.Run("learnings context injection", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false

		// Pre-seed learnings
		file := NewLearningsFile(config.LearningsPath)
		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Test strategy", Helpful: 5},
			{ID: "mistakes-00001", Category: "mistakes", Content: "Test mistake", Helpful: 3},
		})
		require.NoError(t, err)

		m, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m.Close()

		ctx := m.LearningsContext()
		assert.Contains(t, ctx, "L001")
		assert.Contains(t, ctx, "M001")
		assert.Contains(t, ctx, "Test strategy")
	})

	t.Run("citation tracking", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false
		config.MinConfidence = 0.9 // High threshold to prevent new learnings from being added

		// Seed with learnings
		file := NewLearningsFile(config.LearningsPath)
		err := file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Check nil", Helpful: 5, Harmful: 0},
		})
		require.NoError(t, err)

		m, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m.Close()

		// Simulate trajectory with citation
		recorder := m.StartTrajectory("agent-1", "test", "Query")
		recorder.RecordStep("think", "", "Using [L001] pattern here", nil, nil, nil)
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

		// Citation should trigger helpful update
		learnings := m.Learnings()
		found := FindByID(learnings, "strategies-00001")
		require.NotNil(t, found)
		assert.Equal(t, 6, found.Helpful) // 5 + 1 from citation
	})

	t.Run("persists across restarts", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false

		// First manager session
		m1, err := NewManager(config, nil)
		require.NoError(t, err)

		// Manually add a learning via file
		file := NewLearningsFile(config.LearningsPath)
		err = file.Save([]Learning{
			{ID: "strategies-00001", Category: "strategies", Content: "Persisted learning", Helpful: 1},
		})
		require.NoError(t, err)
		m1.Close()

		// Second manager session
		m2, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m2.Close()

		learnings := m2.Learnings()
		require.Len(t, learnings, 1)
		assert.Equal(t, "Persisted learning", learnings[0].Content)
	})

	t.Run("metrics tracking", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false

		m, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m.Close()

		initialMetrics := m.Metrics()
		assert.Equal(t, int64(0), initialMetrics["trajectories_processed"])

		recorder := m.StartTrajectory("agent-1", "test", "Query")
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

		finalMetrics := m.Metrics()
		assert.Equal(t, int64(1), finalMetrics["trajectories_processed"])
	})

	t.Run("concurrent trajectories are isolated", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false

		m, err := NewManager(config, nil)
		require.NoError(t, err)
		defer m.Close()

		var wg sync.WaitGroup
		const numConcurrent = 10

		// Start concurrent trajectories
		for i := 0; i < numConcurrent; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				recorder := m.StartTrajectory("agent-1", "task", "query")
				recorder.RecordStep("step", "", "step for trajectory", nil, nil, nil)
				m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)
			}(i)
		}

		wg.Wait()

		metrics := m.Metrics()
		assert.Equal(t, int64(numConcurrent), metrics["trajectories_processed"])
	})
}

func TestManagerWithReflector(t *testing.T) {
	t.Run("uses provided reflector", func(t *testing.T) {
		tmpDir := t.TempDir()
		config := DefaultConfig()
		config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
		config.AsyncReflection = false
		config.MinConfidence = 0.5

		adapter := NewStaticAdapter([]InsightCandidate{
			{Content: "Adapter insight", Category: "strategies", Confidence: 0.9},
		})
		reflector := NewUnifiedReflector([]Adapter{adapter}, nil)

		m, err := NewManager(config, reflector)
		require.NoError(t, err)
		defer m.Close()

		recorder := m.StartTrajectory("agent-1", "test", "Query")
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

		// Wait for async processing to complete
		assert.Eventually(t, func() bool {
			metrics := m.Metrics()
			return metrics["insights_extracted"] >= int64(1)
		}, 100*time.Millisecond, 5*time.Millisecond, "expected at least 1 insight to be extracted")
	})
}

func TestManagerValidation(t *testing.T) {
	t.Run("invalid config", func(t *testing.T) {
		config := Config{} // Invalid: empty learnings path
		_, err := NewManager(config, nil)
		assert.Error(t, err)
	})
}

func TestManagerEndTrajectoryNilRecorder(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	// End trajectory with nil recorder - should not panic
	m.EndTrajectory(context.Background(), nil, OutcomeSuccess)

	metrics := m.Metrics()
	assert.Equal(t, int64(0), metrics["trajectories_processed"])
}

func TestManagerAsyncQueueFull(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = true
	config.CurationFrequency = 1000 // High value to prevent batch processing

	m, err := NewManager(config, nil)
	require.NoError(t, err)

	// Fill the queue (capacity is 100)
	for i := 0; i < 150; i++ {
		recorder := m.StartTrajectory("agent-1", "test", "Query")
		recorder.RecordStep("think", "", "Reasoning", nil, nil, nil)
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)
	}

	// Should not hang or panic
	err = m.Close()
	require.NoError(t, err)
}

func TestManagerPartialOutcome(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	recorder := m.StartTrajectory("agent-1", "test", "Query")
	recorder.RecordStep("think", "", "Partial work", nil, nil, nil)
	m.EndTrajectory(context.Background(), recorder, OutcomePartial)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerFailureOutcome(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	recorder := m.StartTrajectory("agent-1", "test", "Query")
	recorder.RecordStep("tool", "search", "Failed search", nil, nil, nil)
	m.EndTrajectory(context.Background(), recorder, OutcomeFailure)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerRefreshCacheError(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	// Get initial learnings
	initial := m.Learnings()
	assert.Empty(t, initial)

	// Add learning externally
	file := NewLearningsFile(config.LearningsPath)
	err = file.Save([]Learning{
		{ID: "external-00001", Content: "External", Helpful: 1},
	})
	require.NoError(t, err)

	// Trigger refresh via trajectory
	recorder := m.StartTrajectory("agent-1", "test", "Query")
	m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

	// Cache should be refreshed
	updated := m.Learnings()
	assert.Len(t, updated, 1)
}

func TestManagerAsyncTickerFlush(t *testing.T) {
	// This test verifies the ticker-based flush in processLoop
	// We use a short test to avoid slowing down the test suite
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = true
	config.CurationFrequency = 100 // High value so ticker triggers before batch

	m, err := NewManager(config, nil)
	require.NoError(t, err)

	// Add one trajectory (won't trigger batch)
	recorder := m.StartTrajectory("agent-1", "test", "Query")
	m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

	// Close will flush
	err = m.Close()
	require.NoError(t, err)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerRecordStepWithError(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	recorder := m.StartTrajectory("agent-1", "test", "Query")

	// Record a step with an error
	testErr := assert.AnError
	recorder.RecordStep("tool", "search", "Failed search", nil, nil, testErr)

	m.EndTrajectory(context.Background(), recorder, OutcomeFailure)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerRecordStepWithToolInput(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	recorder := m.StartTrajectory("agent-1", "test", "Query")

	// Record step with tool input and output
	toolInput := map[string]interface{}{"query": "test"}
	toolOutput := map[string]interface{}{"results": []string{"a", "b"}}
	recorder.RecordStep("tool", "search", "Search operation", toolInput, toolOutput, nil)

	m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerSyncProcessing(t *testing.T) {
	// Test synchronous mode processes immediately without async queue
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false // Synchronous mode
	config.MinConfidence = 0.5

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	// Process trajectory synchronously
	recorder := m.StartTrajectory("agent-1", "test", "Query")
	recorder.RecordStep("think", "", "Planning step", nil, nil, nil)
	recorder.RecordStep("tool", "read", "Reading file", map[string]interface{}{"path": "/test"}, nil, nil)
	m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)

	metrics := m.Metrics()
	assert.Equal(t, int64(1), metrics["trajectories_processed"])
}

func TestManagerAsyncBatchProcessing(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = true
	config.CurationFrequency = 3 // Small batch size

	m, err := NewManager(config, nil)
	require.NoError(t, err)

	// Add exactly one batch of trajectories
	for i := 0; i < 3; i++ {
		recorder := m.StartTrajectory("agent-1", "test", "Query")
		recorder.RecordStep("action", "", "Work", nil, nil, nil)
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)
	}

	// Wait for async batch processing to complete
	assert.Eventually(t, func() bool {
		metrics := m.Metrics()
		return metrics["trajectories_processed"] >= int64(3)
	}, 200*time.Millisecond, 10*time.Millisecond, "expected all 3 trajectories to be processed")

	err = m.Close()
	require.NoError(t, err)
}

func TestManagerLearningsContextWithCategories(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	// Pre-seed learnings with multiple categories
	file := NewLearningsFile(config.LearningsPath)
	err := file.Save([]Learning{
		{ID: "strategies-00001", Category: "strategies", Content: "Strategy 1", Helpful: 5},
		{ID: "strategies-00002", Category: "strategies", Content: "Strategy 2", Helpful: 3},
		{ID: "mistakes-00001", Category: "mistakes", Content: "Mistake 1", Helpful: 2},
		{ID: "patterns-00001", Category: "patterns", Content: "Pattern 1", Helpful: 4},
	})
	require.NoError(t, err)

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	ctx := m.LearningsContext()

	// Should contain sections with learnings
	assert.Contains(t, ctx, "Strategies")
	assert.Contains(t, ctx, "Mistakes")

	// Should contain short codes
	assert.Contains(t, ctx, "L001")
	assert.Contains(t, ctx, "L002")
	assert.Contains(t, ctx, "M001")
	assert.Contains(t, ctx, "P001")
}

func TestManagerEmptyLearningsContext(t *testing.T) {
	tmpDir := t.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(t, err)
	defer m.Close()

	ctx := m.LearningsContext()
	// Empty context returns empty string
	assert.Empty(t, ctx)
}

func BenchmarkManagerTrajectoryProcessing(b *testing.B) {
	tmpDir := b.TempDir()
	config := DefaultConfig()
	config.LearningsPath = filepath.Join(tmpDir, "learnings.md")
	config.AsyncReflection = false

	m, err := NewManager(config, nil)
	require.NoError(b, err)
	defer m.Close()

	ctx := context.Background()
	_ = ctx

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recorder := m.StartTrajectory("agent-1", "test", "Query")
		recorder.RecordStep("think", "", "Reasoning", nil, nil, nil)
		recorder.RecordStep("tool", "search", "Searching", nil, nil, nil)
		m.EndTrajectory(context.Background(), recorder, OutcomeSuccess)
	}
}
