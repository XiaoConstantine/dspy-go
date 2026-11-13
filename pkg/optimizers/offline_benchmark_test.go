package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/require"
)

// Fix the contract before measuring optimizers: 0.25 baseline, at least +0.625
// improvement on every seed. A partial codebook (0.75) is not enough. Replaying
// each seed must produce the same held-out predictions, not just the same mean.
const offlineMargin = 0.625

type offlineCase struct {
	name       string
	new        func(int64) core.Optimizer
	quarantine string
}

func offlineCases() []offlineCase {
	return []offlineCase{
		{name: "BootstrapFewShot", new: func(int64) core.Optimizer {
			// The constructor has no default capacity; four is an explicit required
			// argument, not a reduced iteration budget. All other settings are default.
			return NewBootstrapFewShot(func(want, got map[string]any, _ context.Context) bool {
				return offlineMetric(want, got) == 1
			}, 4)
		}},
		{name: "MIPRO", new: func(seed int64) core.Optimizer {
			return NewMIPRO(nil, WithRandomSeed(seed))
		}},
		{name: "COPRO", new: func(int64) core.Optimizer {
			return NewCOPRO(offlineMetric)
		}},
		{name: "SIMBA", new: func(seed int64) core.Optimizer {
			s := NewSIMBA()
			// SIMBA has no public seed option. This is the only private optimizer
			// state changed by the fixture; no strategy or budget is overridden.
			s.rng = rand.New(rand.NewSource(seed))
			return s
		}},
		{name: "GEPA_default", new: func(seed int64) core.Optimizer {
			config := DefaultGEPAConfig()
			config.RandomSeed = seed
			optimizer, err := NewGEPA(config)
			if err != nil {
				panic(err)
			}
			return optimizer
		}},
		{name: "GEPA_tournament", new: func(seed int64) core.Optimizer {
			config := DefaultGEPAConfig()
			config.RandomSeed = seed
			config.SelectionStrategy = "tournament"
			optimizer, err := NewGEPA(config)
			if err != nil {
				panic(err)
			}
			return optimizer
		}},
		{name: "TPE", new: func(seed int64) core.Optimizer {
			return offlineTPE{seed: seed}
		}},
	}
}

// TPE is a SearchStrategy, not a core.Optimizer. The adapter only supplies its
// required categorical domain and 25-trial budget, evaluates real programs, and
// applies GetBestParams. All TPE hyperparameters retain their constructor defaults.
type offlineTPE struct{ seed int64 }

func (o offlineTPE) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	strategy := NewTPEOptimizer(TPEConfig{Seed: o.seed})
	choices := []string{offlineDecoy, offlineCorrect, offlinePartial}
	const trials = 25
	if err := strategy.Initialize(SearchConfig{
		ParamSpace: map[string][]any{"instruction": {0, 1, 2}}, MaxTrials: trials, Seed: o.seed,
	}); err != nil {
		return program, err
	}
	examples := core.MaterializeDataset(dataset)
	for range trials {
		params, err := strategy.SuggestParams(ctx)
		if err != nil {
			return program, err
		}
		instruction := choices[params["instruction"].(int)]
		model := program.Modules["route"].(*modules.Predict).LLM.(*offlineLLM)
		model.mu.Lock()
		model.propose(instruction)
		model.mu.Unlock()
		candidate := offlineWithInstruction(program, instruction)
		score, _, err := offlineScore(ctx, candidate, examples, metric)
		if err != nil {
			return program, err
		}
		if err := strategy.UpdateResults(params, score); err != nil {
			return program, err
		}
	}
	params, _ := strategy.GetBestParams()
	return offlineWithInstruction(program, choices[params["instruction"].(int)]), nil
}

func offlineScore(ctx context.Context, program core.Program, examples []core.Example, metric core.Metric) (float64, []string, error) {
	var total float64
	var predictions []string
	for _, example := range examples {
		output, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			return 0, nil, err
		}
		total += metric(example.Outputs, output)
		predictions = append(predictions, fmt.Sprint(output["completion"]))
	}
	return total / float64(len(examples)), predictions, nil
}

func offlineGate(baseline, optimized float64) error {
	if math.IsNaN(optimized) || math.IsInf(optimized, 0) || optimized-baseline < offlineMargin {
		return fmt.Errorf("score gate: baseline=%.3f optimized=%.3f gain=%.3f; require gain >= %.3f", baseline, optimized, optimized-baseline, offlineMargin)
	}
	return nil
}

type offlineResult struct {
	baseline    float64
	optimized   float64
	random      float64
	predictions []string
}

func runOffline(t *testing.T, tc offlineCase, seed int64) offlineResult {
	t.Helper()
	model := newOfflineLLM()
	oldDefault, oldTeacher := core.GetDefaultLLM(), core.GetTeacherLLM()
	core.SetDefaultLLM(model)
	core.SetTeacherLLM(model)
	defer func() {
		core.SetDefaultLLM(oldDefault)
		core.SetTeacherLLM(oldTeacher)
	}()
	if tc.name == "BootstrapFewShot" {
		core.SetTeacherLLM(&offlineTeacher{model})
	}
	ctx := context.Background()
	base := offlineProgram(model)
	train, heldout := offlineExamples(seed, 0), offlineExamples(seed+1000, 10000)
	baseline, _, err := offlineScore(ctx, base, heldout, offlineMetric)
	require.NoError(t, err)
	require.Equal(t, 0.25, baseline, "fixture baseline changed")
	optimized, err := tc.new(seed).Compile(ctx, base, datasets.NewSimpleDataset(train), offlineMetric)
	require.NoError(t, err)
	unchanged, _, err := offlineScore(ctx, base, heldout, offlineMetric)
	require.NoError(t, err)
	require.Equal(t, baseline, unchanged, "Compile mutated the original program")
	// Replace the final choice with a uniform draw from the actual emitted
	// candidates (including the original), after allowing the real optimizer its
	// entire generation/evaluation budget. This mutant ignores candidate scores.
	randomProgram := model.randomCandidate(base, seed, tc.name == "BootstrapFewShot")
	randomScore, _, err := offlineScore(ctx, randomProgram, heldout, offlineMetric)
	require.NoError(t, err)
	// Mutation runs replace the returned program, not the metric or its score.
	// Both mutants get the real optimizer's entire candidate-generation budget.
	switch os.Getenv("DSPY_OFFLINE_MUTANT") {
	case "noop":
		optimized = base.Clone()
	case "random":
		optimized = randomProgram
	}
	score, predictions, err := offlineScore(ctx, optimized, heldout, offlineMetric)
	require.NoError(t, err)
	require.Empty(t, model.unexpected, "unknown prompts must not silently trigger fallback")
	if seed == 1 && score < baseline {
		predict := optimized.Modules["route"].(*modules.Predict)
		t.Logf("regressed program: instruction=%q demos=%v first prediction=%q",
			predict.GetSignature().Instruction, predict.GetDemos(), predictions[0])
	}
	return offlineResult{baseline, score, randomScore, predictions}
}

func TestOfflineOptimizerImprovement(t *testing.T) {
	// Global model/logger configuration is restored, and these tests intentionally
	// do not use t.Parallel. Optimizers still execute with their default concurrency.
	logger := logging.GetLogger()
	logging.SetLogger(logging.NewLogger(logging.Config{Severity: logging.ERROR}))
	t.Cleanup(func() { logging.SetLogger(logger) })
	started := time.Now()
	t.Cleanup(func() { t.Logf("offline score suite duration: %s", time.Since(started)) })
	mutant := os.Getenv("DSPY_OFFLINE_MUTANT")
	require.Contains(t, []string{"", "noop", "random"}, mutant)
	for _, tc := range offlineCases() {
		t.Run(tc.name, func(t *testing.T) {
			var failures []error
			var randomRejected int
			var scores, randomScores []float64
			for seed := int64(1); seed <= 8; seed++ {
				result := runOffline(t, tc, seed)
				replay := runOffline(t, tc, seed)
				require.Equal(t, result, replay, "seed %d did not replay deterministically", seed)
				require.Error(t, offlineGate(result.baseline, result.baseline), "no-op survived")
				if offlineGate(result.baseline, result.random) != nil {
					randomRejected++
				}
				scores = append(scores, result.optimized)
				randomScores = append(randomScores, result.random)
				if err := offlineGate(result.baseline, result.optimized); err != nil {
					failures = append(failures, fmt.Errorf("seed %d: %w", seed, err))
				}
			}
			t.Logf("baseline=0.250 optimized=%v random=%v no-op rejected=8/8 random rejected=%d/8", scores, randomScores, randomRejected)
			require.Positive(t, randomRejected, "random selection survived the score gate")
			if tc.quarantine != "" && mutant == "" && os.Getenv("DSPY_OFFLINE_STRICT") != "1" {
				require.NotEmpty(t, failures, "quarantined optimizer now passes; remove its quarantine")
				t.Skipf("KNOWN FAILURE: %s; set DSPY_OFFLINE_STRICT=1 to enforce: %v", tc.quarantine, failures)
			}
			require.Empty(t, failures)
		})
	}
}

func TestOfflineTaskContract(t *testing.T) {
	model := newOfflineLLM()
	program := offlineProgram(model)
	examples := offlineExamples(1, 10000)
	for _, tc := range []struct {
		instruction string
		score       float64
	}{
		{offlineBaseline, 0.25}, {offlinePartial, 0.75}, {offlineCorrect, 1},
		{offlineDecoy, 0}, {offlineOther, 0},
	} {
		score, _, err := offlineScore(context.Background(), offlineWithInstruction(program, tc.instruction), examples, offlineMetric)
		require.NoError(t, err)
		require.Equal(t, tc.score, score)
	}
	train := offlineExamples(1, 0)
	for _, example := range examples {
		require.False(t, slices.ContainsFunc(train, func(e core.Example) bool {
			return e.Inputs["prompt"] == example.Inputs["prompt"]
		}), "held-out input leaked into training")
	}
}
