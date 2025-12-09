package ace

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedReflector(t *testing.T) {
	t.Run("correlate feedback on success", func(t *testing.T) {
		ur := NewUnifiedReflector(nil, nil)

		trajectory := &Trajectory{
			ID:           "test-1",
			FinalOutcome: OutcomeSuccess,
			Quality:      0.9,
			Context: TrajectoryContext{
				CitedLearnings: []string{"L001", "M002"},
			},
		}

		result, err := ur.Reflect(context.Background(), trajectory, nil)
		require.NoError(t, err)

		assert.Len(t, result.LearningUpdates, 2)
		for _, update := range result.LearningUpdates {
			assert.Equal(t, DeltaHelpful, update.Delta)
		}
	})

	t.Run("correlate feedback on failure", func(t *testing.T) {
		ur := NewUnifiedReflector(nil, nil)

		trajectory := &Trajectory{
			ID:           "test-2",
			FinalOutcome: OutcomeFailure,
			Quality:      0.2,
			Context: TrajectoryContext{
				CitedLearnings: []string{"L001"},
			},
		}

		result, err := ur.Reflect(context.Background(), trajectory, nil)
		require.NoError(t, err)

		assert.Len(t, result.LearningUpdates, 1)
		assert.Equal(t, DeltaHarmful, result.LearningUpdates[0].Delta)
	})

	t.Run("correlate feedback on low-quality partial", func(t *testing.T) {
		ur := NewUnifiedReflector(nil, nil)

		trajectory := &Trajectory{
			ID:           "test-3",
			FinalOutcome: OutcomePartial,
			Quality:      0.3,
			Context: TrajectoryContext{
				CitedLearnings: []string{"L001"},
			},
		}

		result, err := ur.Reflect(context.Background(), trajectory, nil)
		require.NoError(t, err)

		assert.Equal(t, DeltaHarmful, result.LearningUpdates[0].Delta)
	})

	t.Run("combine adapter insights", func(t *testing.T) {
		adapter := NewStaticAdapter([]InsightCandidate{
			{Content: "Test strategy", Category: "strategies", Confidence: 0.8},
			{Content: "Test mistake", Category: "mistakes", Confidence: 0.7},
		})

		ur := NewUnifiedReflector([]Adapter{adapter}, nil)

		trajectory := &Trajectory{ID: "test-4", FinalOutcome: OutcomeSuccess}
		result, err := ur.Reflect(context.Background(), trajectory, nil)

		require.NoError(t, err)
		assert.Len(t, result.SuccessPatterns, 1)
		assert.Len(t, result.FailurePatterns, 1)
	})

	t.Run("combine with domain reflector", func(t *testing.T) {
		adapter := NewStaticAdapter([]InsightCandidate{
			{Content: "Adapter insight", Category: "strategies"},
		})

		domain := NewSimpleReflector()

		ur := NewUnifiedReflector([]Adapter{adapter}, domain)

		trajectory := &Trajectory{
			ID:           "test-5",
			FinalOutcome: OutcomeSuccess,
			Steps:        []Step{{Action: "a"}, {Action: "b"}},
		}

		result, err := ur.Reflect(context.Background(), trajectory, nil)

		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(result.SuccessPatterns), 2)
	})
}

func TestSimpleReflector(t *testing.T) {
	t.Run("success with few steps", func(t *testing.T) {
		sr := NewSimpleReflector()

		trajectory := &Trajectory{
			ID:           "test-1",
			FinalOutcome: OutcomeSuccess,
			Steps:        []Step{{Action: "a"}, {Action: "b"}},
		}

		result, err := sr.Reflect(context.Background(), trajectory, nil)
		require.NoError(t, err)
		assert.Len(t, result.SuccessPatterns, 1)
	})

	t.Run("failure extracts error", func(t *testing.T) {
		sr := NewSimpleReflector()

		trajectory := &Trajectory{
			ID:           "test-2",
			FinalOutcome: OutcomeFailure,
			Steps: []Step{
				{Action: "a"},
				{Action: "b", Error: "something went wrong"},
			},
		}

		result, err := sr.Reflect(context.Background(), trajectory, nil)
		require.NoError(t, err)
		assert.Len(t, result.FailurePatterns, 1)
		assert.Contains(t, result.FailurePatterns[0].Content, "something went wrong")
	})
}
