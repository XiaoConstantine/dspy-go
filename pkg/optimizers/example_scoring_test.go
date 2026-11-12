package optimizers

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// overlapProgram echoes its "q" input and holds every call until level calls
// are in flight, so peak reports how many executions really overlapped.
type overlapProgram struct {
	mu             sync.Mutex
	level          int
	inFlight, peak int
	full           chan struct{}
}

func newOverlapProgram(level int) *overlapProgram {
	return &overlapProgram{level: level, full: make(chan struct{})}
}

func (o *overlapProgram) program() core.Program {
	return core.NewProgram(map[string]core.Module{}, func(_ context.Context, inputs map[string]any) (map[string]any, error) {
		o.mu.Lock()
		o.inFlight++
		o.peak = max(o.peak, o.inFlight)
		if o.inFlight == o.level {
			select {
			case <-o.full:
			default:
				close(o.full)
			}
		}
		o.mu.Unlock()
		select {
		case <-o.full:
		case <-time.After(2 * time.Second):
		}
		o.mu.Lock()
		o.inFlight--
		o.mu.Unlock()
		if inputs["q"] == "fail" {
			return nil, errors.New("boom")
		}
		return map[string]any{"output": inputs["q"]}, nil
	})
}

func overlapExamples(n int) []core.Example {
	examples := make([]core.Example, n)
	for i := range examples {
		q := fmt.Sprintf("question-%d", i)
		examples[i] = core.Example{Inputs: map[string]any{"q": q}, Outputs: map[string]any{"output": q}}
	}
	return examples
}

// positionMetric scores question-3 lower than the rest, so a result stored at
// the wrong index changes the outcome.
func positionMetric(expected, actual map[string]any) float64 {
	if actual["output"] == "question-3" {
		return 0.25
	}
	return exactOutputMetric(expected, actual)
}

func TestScoreExamplesKeepsOrderAndStopsAfterFailure(t *testing.T) {
	examples := overlapExamples(7)
	examples[4].Inputs["q"] = "fail"
	var started []string
	score := func(example core.Example) (float64, error) {
		q := example.Inputs["q"].(string)
		started = append(started, q)
		if q == "fail" {
			return 0, errors.New("boom")
		}
		return positionMetric(example.Outputs, map[string]any{"output": q}), nil
	}

	results := scoreExamples(context.Background(), 1, true, examples, score)

	require.Len(t, results, 7)
	assert.Len(t, started, 5, "one worker stops at the first failure like a plain loop")
	assert.Equal(t, []float64{1, 1, 1, 0.25}, []float64{results[0].Score, results[1].Score, results[2].Score, results[3].Score})
	assert.Error(t, results[4].Err)
	assert.False(t, results[5].Ran)
	assert.False(t, results[6].Ran)

	started = nil
	results = scoreExamples(context.Background(), 1, false, examples, score)
	assert.Len(t, started, 7, "without stopOnError a failure does not skip later examples")
	assert.True(t, results[6].Ran)
}

func TestMIPROEvaluationHonorsPackageConcurrency(t *testing.T) {
	const level = 3
	previous := core.GetConcurrencyLevel()
	core.SetConcurrencyOptions(level)
	t.Cleanup(func() { core.SetConcurrencyOptions(previous) })

	metric := func(expected, actual map[string]any, _ context.Context) float64 {
		return positionMetric(expected, actual)
	}
	m := NewMIPRO(metric)

	full := newOverlapProgram(level)
	score, err := m.evaluateCandidate(context.Background(), full.program(), overlapExamples(7), metric)
	require.NoError(t, err)
	assert.Equal(t, level, full.peak)
	assert.InDelta(t, 6.25/7, score, 1e-12)

	batch := newOverlapProgram(level)
	score, err = m.evaluateOnMinibatch(context.Background(), batch.program(), overlapExamples(7), 4, metric)
	require.NoError(t, err)
	assert.Equal(t, level, batch.peak)
	assert.InDelta(t, 3.25/4, score, 1e-12)
}

func TestSIMBAFullEvaluationHonorsLLMConcurrency(t *testing.T) {
	for _, level := range []int{1, 3} {
		simba := NewSIMBA(WithLLMConcurrency(level))
		tracked := newOverlapProgram(level)

		score, err := simba.evaluateProgramOnExamples(context.Background(), tracked.program(), overlapExamples(7), positionMetric)

		require.NoError(t, err)
		assert.Equal(t, level, tracked.peak)
		assert.InDelta(t, 6.25/7, score, 1e-12)
	}
}
