package ace

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerator(t *testing.T) {
	t.Run("full lifecycle", func(t *testing.T) {
		g := NewGenerator()
		g.Start("agent-1", "code_review", "Review this PR")

		g.SetInjectedLearnings([]string{"L001", "L002", "M001"})

		g.RecordStep("think", "", "I'll check the diff first", nil, nil, nil)
		g.RecordStep("tool", "git_diff", "Using [L001] pattern", map[string]any{"path": "."}, map[string]any{"diff": "..."}, nil)
		g.RecordStep("tool", "lint", "Based on [M001] avoid this mistake", nil, nil, errors.New("lint failed"))

		traj := g.End(OutcomePartial, 0.7)

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

	t.Run("nil trajectory handling", func(t *testing.T) {
		g := NewGenerator()

		// Operations before Start should not panic
		g.RecordStep("test", "", "", nil, nil, nil)
		g.SetInjectedLearnings([]string{"L001"})
		traj := g.End(OutcomeSuccess, 1.0)

		assert.Nil(t, traj)
	})

	t.Run("current returns in-progress trajectory", func(t *testing.T) {
		g := NewGenerator()
		assert.Nil(t, g.Current())

		g.Start("agent-1", "task", "query")
		current := g.Current()
		require.NotNil(t, current)
		assert.Equal(t, "agent-1", current.AgentID)

		g.End(OutcomeSuccess, 1.0)
		assert.Nil(t, g.Current())
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
