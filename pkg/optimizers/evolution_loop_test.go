package optimizers

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunEvolutionLoop_OrdersStagesAndSkipsGenerationZeroReflection(t *testing.T) {
	var calls []string
	var progress []string

	err := RunEvolutionLoop(context.Background(), EvolutionLoopConfig{
		MaxGenerations: 3,
		ReflectionFreq: 1,
		PhaseName:      "GEPA Evolution",
	}, EvolutionLoopHooks{
		Initialize: func(ctx context.Context) error {
			calls = append(calls, "init")
			return nil
		},
		BeforeGeneration: func(ctx context.Context, generation int) error {
			calls = append(calls, "before")
			return nil
		},
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			calls = append(calls, "evaluate")
			return nil
		},
		AfterEvaluation: func(ctx context.Context, generation int) error {
			calls = append(calls, "after")
			return nil
		},
		Reflect: func(ctx context.Context, generation int) error {
			calls = append(calls, "reflect")
			return nil
		},
		HasConverged: func(ctx context.Context, generation int) bool {
			calls = append(calls, "converged")
			return false
		},
		Evolve: func(ctx context.Context, generation int) error {
			calls = append(calls, "evolve")
			return nil
		},
		ReportProgress: func(phase string, current, total int) {
			progress = append(progress, phase)
		},
		Finalize: func(ctx context.Context) error {
			calls = append(calls, "finalize")
			return nil
		},
	})

	require.NoError(t, err)
	assert.Equal(t, []string{
		"init",
		"before", "evaluate", "after", "converged", "evolve",
		"before", "evaluate", "after", "reflect", "converged", "evolve",
		"before", "evaluate", "after", "reflect", "converged",
		"finalize",
	}, calls)
	assert.Equal(t, []string{"GEPA Evolution", "GEPA Evolution", "GEPA Evolution"}, progress)
}

func TestRunEvolutionLoop_ContinuesAfterNonFatalReflectionError(t *testing.T) {
	var nonFatal []string
	evaluateCalls := 0

	err := RunEvolutionLoop(context.Background(), EvolutionLoopConfig{
		MaxGenerations: 2,
		ReflectionFreq: 1,
	}, EvolutionLoopHooks{
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			evaluateCalls++
			return nil
		},
		Reflect: func(ctx context.Context, generation int) error {
			return errors.New("reflection failed")
		},
		OnNonFatalError: func(ctx context.Context, stage string, generation int, err error) {
			nonFatal = append(nonFatal, stage)
		},
	})

	require.NoError(t, err)
	assert.Equal(t, 2, evaluateCalls)
	assert.Equal(t, []string{"reflection"}, nonFatal)
}

func TestRunEvolutionLoop_StopsAfterConvergence(t *testing.T) {
	evaluateCalls := 0
	evolveCalls := 0

	err := RunEvolutionLoop(context.Background(), EvolutionLoopConfig{
		MaxGenerations: 5,
		ReflectionFreq: 2,
	}, EvolutionLoopHooks{
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			evaluateCalls++
			return nil
		},
		HasConverged: func(ctx context.Context, generation int) bool {
			return generation == 1
		},
		Evolve: func(ctx context.Context, generation int) error {
			evolveCalls++
			return nil
		},
	})

	require.NoError(t, err)
	assert.Equal(t, 2, evaluateCalls)
	assert.Equal(t, 1, evolveCalls)
}

func TestRunEvolutionLoop_ZeroReflectionFreqDoesNotCallReflect(t *testing.T) {
	err := RunEvolutionLoop(context.Background(), EvolutionLoopConfig{
		MaxGenerations: 2,
		ReflectionFreq: 0,
	}, EvolutionLoopHooks{
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			return nil
		},
		Reflect: func(ctx context.Context, generation int) error {
			return errors.New("reflect should not be called")
		},
	})

	require.NoError(t, err)
}

func TestRunEvolutionLoop_RespectsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	evaluateCalls := 0

	err := RunEvolutionLoop(ctx, EvolutionLoopConfig{
		MaxGenerations: 10,
	}, EvolutionLoopHooks{
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			evaluateCalls++
			if generation == 1 {
				cancel()
			}
			return nil
		},
	})

	require.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, 2, evaluateCalls)
}
