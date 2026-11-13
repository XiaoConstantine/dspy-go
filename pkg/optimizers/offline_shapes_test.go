package optimizers

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/stretchr/testify/require"
)

func offlineShapeOptimizer(tc offlineCase, seed int64, model *offlineShapeLLM) core.Optimizer {
	if tc.name == "TPE" {
		return offlineShapeTPE{seed: seed, model: model}
	}
	return tc.new(seed)
}

func offlineShapeCases(shape string) []offlineCase {
	if shape != "late" {
		return offlineCases()
	}
	var cases []offlineCase
	for _, tc := range offlineCases() {
		switch tc.name {
		case "BootstrapFewShot", "TPE":
			continue
		case "COPRO":
			// Default depth two cannot reach a fourth refinement. This is a
			// documented budget increase, not a weaker score threshold.
			tc.new = func(int64) core.Optimizer { return NewCOPRO(offlineMetric, WithDepth(4)) }
		case "MIPRO":
			tc.new = func(seed int64) core.Optimizer {
				return NewMIPRO(nil, WithRandomSeed(seed), WithSearchStrategy(&offlineDelayedSearch{
					SearchStrategy: offlineShapeMIPROSearch(seed),
				}))
			}
		}
		cases = append(cases, tc)
	}
	return cases
}

func offlineShapeMIPROSearch(seed int64) SearchStrategy {
	// Match MIPRO's default TPE configuration, changing only the first three
	// suggestions through its public SearchStrategy option.
	return NewTPEOptimizer(TPEConfig{Seed: seed, Gamma: 0.25, NumEIGenerations: 20, PriorWeight: 1, BandwidthFactor: 1})
}

func offlineShapeModels(t *testing.T, shape string, late, bootstrap bool) *offlineShapeLLM {
	t.Helper()
	model := newOfflineShapeLLM(shape, late)
	oldDefault, oldTeacher := core.GetDefaultLLM(), core.GetTeacherLLM()
	core.SetDefaultLLM(model)
	core.SetTeacherLLM(model)
	if bootstrap {
		core.SetTeacherLLM(&offlineShapeTeacher{model})
	}
	t.Cleanup(func() {
		core.SetDefaultLLM(oldDefault)
		core.SetTeacherLLM(oldTeacher)
	})
	return model
}

type offlineShapeResult struct {
	offlineResult
	compileError string
}

func runOfflineShape(t *testing.T, tc offlineCase, shape string, late bool, seed int64) offlineShapeResult {
	t.Helper()
	model := offlineShapeModels(t, shape, late, tc.name == "BootstrapFewShot")
	ctx := context.Background()
	base := offlineShapeProgram(model)
	train, heldout := offlineExamples(seed, 0), offlineExamples(seed+1000, 10000)
	baseline, _, err := offlineScore(ctx, base, heldout, offlineMetric)
	require.NoError(t, err)
	require.Equal(t, 0.25, baseline, "fixture baseline changed")
	optimized, compileErr := offlineShapeOptimizer(tc, seed, model).Compile(ctx, base, datasets.NewSimpleDataset(train), offlineMetric)
	unchanged, _, err := offlineScore(ctx, base, heldout, offlineMetric)
	require.NoError(t, err)
	require.Equal(t, baseline, unchanged, "Compile mutated the original program")
	randomProgram := model.randomProgram(base, seed, tc.name == "BootstrapFewShot")
	randomScore, _, err := offlineScore(ctx, randomProgram, heldout, offlineMetric)
	require.NoError(t, err)
	switch os.Getenv("DSPY_OFFLINE_MUTANT") {
	case "noop":
		optimized = base.Clone()
	case "random":
		optimized = randomProgram
	}
	score, predictions, err := offlineScore(ctx, optimized, heldout, offlineMetric)
	require.NoError(t, err)
	require.Empty(t, model.unexpected, "unknown prompts must not silently trigger fallback")
	result := offlineShapeResult{offlineResult: offlineResult{baseline, score, randomScore, predictions}}
	if compileErr != nil {
		// Error metadata is a map; its rendered field order is not a replay
		// contract. Retain the full cause, without that unordered annotation.
		result.compileError, _, _ = strings.Cut(compileErr.Error(), " [")
	}
	return result
}

// Shape cases share the original gate's exact thresholds, seeds, replay and
// negative controls.
func TestOfflineOptimizerShapes(t *testing.T) {
	logger := logging.GetLogger()
	logging.SetLogger(logging.NewLogger(logging.Config{Severity: logging.ERROR}))
	t.Cleanup(func() { logging.SetLogger(logger) })
	require.Contains(t, []string{"", "noop", "random"}, os.Getenv("DSPY_OFFLINE_MUTANT"))
	for _, shape := range []string{"cot", "pipeline", "late"} {
		t.Run(shape, func(t *testing.T) {
			for _, tc := range offlineShapeCases(shape) {
				t.Run(tc.name, func(t *testing.T) {
					var failures []error
					var scores, randomScores []float64
					var randomRejected int
					for seed := int64(1); seed <= 8; seed++ {
						result := runOfflineShape(t, tc, shape, shape == "late", seed)
						replay := runOfflineShape(t, tc, shape, shape == "late", seed)
						require.Equal(t, result, replay, "seed %d did not replay deterministically", seed)
						require.Error(t, offlineGate(result.baseline, result.baseline), "no-op survived")
						if offlineGate(result.baseline, result.random) != nil {
							randomRejected++
						}
						scores = append(scores, result.optimized)
						randomScores = append(randomScores, result.random)
						if result.compileError != "" {
							failures = append(failures, fmt.Errorf("seed %d Compile: %s", seed, result.compileError))
						}
						if err := offlineGate(result.baseline, result.optimized); err != nil {
							failures = append(failures, fmt.Errorf("seed %d: %w", seed, err))
						}
					}
					t.Logf("baseline=0.250 optimized=%v random=%v no-op rejected=8/8 random rejected=%d/8", scores, randomScores, randomRejected)
					require.Positive(t, randomRejected, "random selection survived the score gate")
					require.Empty(t, failures)
				})
			}
		})
	}
}

func TestOfflineOptimizerRequiresForwardFactory(t *testing.T) {
	logger := logging.GetLogger()
	logging.SetLogger(logging.NewLogger(logging.Config{Severity: logging.ERROR}))
	t.Cleanup(func() { logging.SetLogger(logger) })
	for _, tc := range offlineCases() {
		if tc.name == "TPE" {
			// TPE is a SearchStrategy driven by a test adapter, not a program
			// optimizer, so it has no Compile entry point to reject programs.
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			var failures []error
			var silentScores []float64
			for seed := int64(1); seed <= 8; seed++ {
				model := offlineShapeModels(t, "plain", false, tc.name == "BootstrapFewShot")
				base := offlineShapeProgram(model)
				// Keep the exact closure but deliberately lose rebinding metadata.
				unbound := core.NewProgram(base.Modules, base.Forward)
				score, _, err := offlineScore(context.Background(), unbound, offlineExamples(seed, 10000), offlineMetric)
				require.NoError(t, err)
				require.Equal(t, 0.25, score)
				optimized, err := offlineShapeOptimizer(tc, seed, model).Compile(context.Background(), unbound,
					datasets.NewSimpleDataset(offlineExamples(seed, 0)), offlineMetric)
				if err == nil {
					failures = append(failures, fmt.Errorf("seed %d silently accepted a non-rebindable program", seed))
					score, _, scoreErr := offlineScore(context.Background(), optimized, offlineExamples(seed, 10000), offlineMetric)
					require.NoError(t, scoreErr)
					silentScores = append(silentScores, score)
				} else {
					require.ErrorContains(t, err, "NewProgramWithForwardFactory", "seed %d returned an unrelated error", seed)
				}
			}
			t.Logf("rejected=%d/8 silently returned scores=%v", 8-len(failures), silentScores)
			require.Empty(t, failures)
		})
	}
}

func TestOfflineShapeTaskContract(t *testing.T) {
	ctx := context.Background()
	examples := offlineExamples(1, 10000)
	for _, shape := range []string{"cot", "pipeline", "late"} {
		t.Run(shape, func(t *testing.T) {
			model := newOfflineShapeLLM(shape, shape == "late")
			base := offlineShapeProgram(model)
			for _, tc := range []struct {
				instruction string
				score       float64
			}{{offlineBaseline, 0.25}, {offlinePartial, 0.75}, {offlineCorrect, 1}, {offlineDecoy, 0}} {
				score, _, err := offlineScore(ctx, offlineWithInstruction(base, tc.instruction), examples, offlineMetric)
				require.NoError(t, err)
				require.Equal(t, tc.score, score)
			}
			if shape == "pipeline" {
				lastOnly := base.Clone()
				lastOnly.Modules["write"].SetSignature(lastOnly.Modules["write"].GetSignature().WithInstruction(offlineCorrect))
				score, _, err := offlineScore(ctx, lastOnly, examples, offlineMetric)
				require.NoError(t, err)
				require.Equal(t, 0.25, score, "changing only the last predictor must not solve the task")
				require.Error(t, offlineGate(0.25, score))
				// A labeled program example is not a route-local prompt -> query demo.
				lastOnly.Modules["route"].(interface{ SetDemos([]core.Example) }).SetDemos(examples[:1])
				_, err = lastOnly.Execute(ctx, examples[1].Inputs)
				require.ErrorContains(t, err, "invalid prompt -> query demonstration")
			}
		})
	}
}

func TestOfflineShapeReflectionEvidence(t *testing.T) {
	const evidence = "1. Inputs: {\"prompt\":\"ticket=9 symbols=dcba\"}\n   Expected: {\"completion\":\"dusk coral blue amber\"}\n   Actual: "
	for _, tc := range []struct {
		name, evidence, want string
	}{
		{"complete", evidence + `{"completion":"amber amber amber amber","rationale":"reason"}`, offlinePartial},
		{"truncated metadata", evidence + `{"completion":"amber amber amber amber","response":"<response...`, offlinePartial},
		{"truncated answer", evidence + `{"completion":"amber amber...`, offlineDecoy},
		{"missing actual", evidence, offlineDecoy},
		{"answer without input", `Expected: {"completion":"dusk coral blue amber"}`, offlineDecoy},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newOfflineShapeLLM("cot", false)
			prompt := fmt.Sprintf("You are improving a GEPA instruction using reflection-guided evidence.\n\nCURRENT INSTRUCTION: %q\nMODULE: route\n\nEXAMPLE-LEVEL EVIDENCE:\n%s", offlineBaseline, tc.evidence)
			response, err := model.Generate(context.Background(), prompt)
			require.NoError(t, err)
			require.Equal(t, tc.want, response.Content)
		})
	}
}

// This control executes real optimizers with deliberately insufficient budgets;
// if it improves, the delayed fixture no longer tests survival past a plateau.
func TestOfflineLateRejectsThreeStepBudget(t *testing.T) {
	for _, tc := range offlineCases() {
		switch tc.name {
		case "COPRO":
			tc.new = func(int64) core.Optimizer { return NewCOPRO(offlineMetric, WithDepth(3)) }
		case "MIPRO":
			tc.new = func(seed int64) core.Optimizer {
				return NewMIPRO(nil, WithRandomSeed(seed), WithNumTrials(3), WithSearchStrategy(&offlineDelayedSearch{
					SearchStrategy: offlineShapeMIPROSearch(seed),
				}))
			}
		case "SIMBA":
			tc.new = func(seed int64) core.Optimizer {
				s := NewSIMBA(WithSIMBAMaxSteps(3))
				s.rng = rand.New(rand.NewSource(seed))
				return s
			}
		case "GEPA_default", "GEPA_tournament":
			strategy := "adaptive_pareto"
			if tc.name == "GEPA_tournament" {
				strategy = "tournament"
			}
			tc.new = func(seed int64) core.Optimizer {
				config := DefaultGEPAConfig()
				config.RandomSeed, config.MaxGenerations, config.SelectionStrategy = seed, 3, strategy
				g, err := NewGEPA(config)
				require.NoError(t, err)
				return g
			}
		default:
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			for seed := int64(1); seed <= 8; seed++ {
				result := runOfflineShape(t, tc, "late", true, seed)
				require.Empty(t, result.compileError)
				require.Equal(t, 0.25, result.optimized)
				require.Error(t, offlineGate(result.baseline, result.optimized))
			}
		})
	}
}
