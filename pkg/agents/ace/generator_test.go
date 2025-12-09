package ace

import (
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerator(t *testing.T) {
	t.Run("full lifecycle", func(t *testing.T) {
		g := NewGenerator()
		recorder := g.Start("agent-1", "code_review", "Review this PR")

		recorder.SetInjectedLearnings([]string{"L001", "L002", "M001"})

		recorder.RecordStep("think", "", "I'll check the diff first", nil, nil, nil)
		recorder.RecordStep("tool", "git_diff", "Using [L001] pattern", map[string]any{"path": "."}, map[string]any{"diff": "..."}, nil)
		recorder.RecordStep("tool", "lint", "Based on [M001] avoid this mistake", nil, nil, errors.New("lint failed"))

		traj := recorder.End(OutcomePartial, 0.7)

		require.NotNil(t, traj)
		assert.Equal(t, "agent-1", traj.AgentID)
		assert.Equal(t, "code_review", traj.TaskType)
		assert.Len(t, traj.Steps, 3)
		assert.Equal(t, OutcomePartial, traj.FinalOutcome)
		assert.Equal(t, 0.7, traj.Quality)
		assert.True(t, traj.Duration > 0)

		// Check context tracking
		assert.ElementsMatch(t, []string{"L001", "L002", "M001"}, traj.Context.InjectedLearnings)
		assert.ElementsMatch(t, []string{"L001", "M001"}, traj.Context.CitedLearnings)
	})

	t.Run("nil recorder handling", func(t *testing.T) {
		// Create a recorder and end it
		g := NewGenerator()
		recorder := g.Start("agent-1", "task", "query")
		recorder.End(OutcomeSuccess, 1.0)

		// Operations after End should not panic
		recorder.RecordStep("test", "", "", nil, nil, nil)
		recorder.SetInjectedLearnings([]string{"L001"})
		traj := recorder.End(OutcomeSuccess, 1.0)

		assert.Nil(t, traj)
	})

	t.Run("current returns in-progress trajectory", func(t *testing.T) {
		g := NewGenerator()
		recorder := g.Start("agent-1", "task", "query")

		current := recorder.Current()
		require.NotNil(t, current)
		assert.Equal(t, "agent-1", current.AgentID)

		recorder.End(OutcomeSuccess, 1.0)
		assert.Nil(t, recorder.Current())
	})

	t.Run("concurrent trajectories are isolated", func(t *testing.T) {
		g := NewGenerator()

		var wg sync.WaitGroup
		results := make(chan *Trajectory, 10)

		// Start 10 concurrent trajectories
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()

				recorder := g.Start("agent-1", "task", "query")
				recorder.RecordStep("step", "", "step for trajectory", nil, nil, nil)
				traj := recorder.End(OutcomeSuccess, 1.0)
				results <- traj
			}(i)
		}

		wg.Wait()
		close(results)

		// All trajectories should be independent and complete
		count := 0
		for traj := range results {
			require.NotNil(t, traj)
			assert.Len(t, traj.Steps, 1)
			count++
		}
		assert.Equal(t, 10, count)
	})
}

func TestDetectCitations(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		expected []string
	}{
		{"no citations", "Plain text without references", nil},
		{"single L citation", "Using [L001] here", []string{"L001"}},
		{"single M citation", "Avoid [M003] mistake", []string{"M003"}},
		{"single P citation", "Apply [P012] pattern", []string{"P012"}},
		{"multiple citations", "Based on [L001] and [M002], also [L003]", []string{"L001", "M002", "L003"}},
		{"duplicate citations", "[L001] and again [L001]", []string{"L001"}},
		{"mixed with text", "First [L001] then some text [M001] more text [P002] end", []string{"L001", "M001", "P002"}},
		{"invalid format", "[l001] [L01] [X001] [L0001]", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DetectCitations(tt.text)
			assert.Equal(t, tt.expected, result)
		})
	}
}
