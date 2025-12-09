package ace

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQualityCalculator(t *testing.T) {
	t.Run("nil trajectory", func(t *testing.T) {
		qc := NewQualityCalculator()
		assert.Equal(t, 0.0, qc.Calculate(nil))
	})

	t.Run("perfect execution", func(t *testing.T) {
		qc := NewQualityCalculator()
		traj := &Trajectory{
			FinalOutcome: OutcomeSuccess,
			Steps: []Step{
				{Action: "think"},
				{Action: "tool", Tool: "search"},
				{Action: "tool", Tool: "read"},
			},
		}

		score := qc.Calculate(traj)
		assert.InDelta(t, 1.0, score, 0.01)
	})

	t.Run("failure", func(t *testing.T) {
		qc := NewQualityCalculator()
		traj := &Trajectory{
			FinalOutcome: OutcomeFailure,
			Steps: []Step{
				{Action: "tool", Tool: "search", Error: "not found"},
			},
		}

		score := qc.Calculate(traj)
		assert.Less(t, score, 0.3)
	})

	t.Run("partial success with many steps", func(t *testing.T) {
		qc := NewQualityCalculator()
		steps := make([]Step, 12)
		for i := range steps {
			steps[i] = Step{Action: "think"}
		}
		traj := &Trajectory{
			FinalOutcome: OutcomePartial,
			Steps:        steps,
		}

		score := qc.Calculate(traj)
		assert.InDelta(t, 0.5, score, 0.2)
	})

	t.Run("custom weights", func(t *testing.T) {
		qc := NewQualityCalculator().WithWeights(QualityWeights{
			Outcome:    1.0,
			Efficiency: 0.0,
			ToolSuccess: 0.0,
			ErrorFree:  0.0,
		})

		success := &Trajectory{FinalOutcome: OutcomeSuccess, Steps: make([]Step, 20)}
		failure := &Trajectory{FinalOutcome: OutcomeFailure, Steps: []Step{{Action: "x"}}}

		assert.Equal(t, 1.0, qc.Calculate(success))
		assert.Equal(t, 0.0, qc.Calculate(failure))
	})

	t.Run("efficiency scoring", func(t *testing.T) {
		qc := NewQualityCalculator().WithExpectedSteps(3, 10).WithWeights(QualityWeights{
			Outcome:    0.0,
			Efficiency: 1.0,
			ToolSuccess: 0.0,
			ErrorFree:  0.0,
		})

		fast := &Trajectory{Steps: make([]Step, 2)}
		expected := &Trajectory{Steps: make([]Step, 3)}
		slow := &Trajectory{Steps: make([]Step, 10)}

		assert.Equal(t, 1.0, qc.Calculate(fast))
		assert.Equal(t, 1.0, qc.Calculate(expected))
		assert.InDelta(t, 0.2, qc.Calculate(slow), 0.01)
	})

	t.Run("tool success rate", func(t *testing.T) {
		qc := NewQualityCalculator().WithWeights(QualityWeights{
			Outcome:    0.0,
			Efficiency: 0.0,
			ToolSuccess: 1.0,
			ErrorFree:  0.0,
		})

		allSuccess := &Trajectory{
			Steps: []Step{
				{Tool: "a"},
				{Tool: "b"},
			},
		}

		halfSuccess := &Trajectory{
			Steps: []Step{
				{Tool: "a"},
				{Tool: "b", Error: "failed"},
			},
		}

		noTools := &Trajectory{
			Steps: []Step{
				{Action: "think"},
			},
		}

		assert.Equal(t, 1.0, qc.Calculate(allSuccess))
		assert.Equal(t, 0.5, qc.Calculate(halfSuccess))
		assert.Equal(t, 1.0, qc.Calculate(noTools))
	})
}
