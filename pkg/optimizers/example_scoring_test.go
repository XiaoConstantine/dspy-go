package optimizers

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
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
	score, err := m.evaluateCandidate(context.Background(), full.program(), overlapExamples(7), miproMetricScorer(metric))
	require.NoError(t, err)
	assert.Equal(t, level, full.peak)
	assert.InDelta(t, 6.25/7, score, 1e-12)

	batch := newOverlapProgram(level)
	score, err = m.evaluateOnMinibatch(context.Background(), batch.program(), overlapExamples(7), 4, miproMetricScorer(metric))
	require.NoError(t, err)
	assert.Equal(t, level, batch.peak)
	assert.InDelta(t, 3.25/4, score, 1e-12)
}

func TestSIMBAFullEvaluationHonorsLLMConcurrency(t *testing.T) {
	for _, level := range []int{1, 3} {
		simba := NewSIMBA(WithLLMConcurrency(level))
		tracked := newOverlapProgram(level)

		score, err := simba.evaluateProgramOnExamples(context.Background(), tracked.program(), overlapExamples(7), core.MetricScorer(positionMetric))

		require.NoError(t, err)
		assert.Equal(t, level, tracked.peak)
		assert.InDelta(t, 6.25/7, score, 1e-12)
	}
}

// overlapScorer scores like positionMetric and records how many Score calls
// ran at once. It cannot score question-5.
type overlapScorer struct {
	mu             sync.Mutex
	inFlight, peak int
}

var errUnscorable = errors.New("judge unavailable")

func (o *overlapScorer) Score(_ context.Context, example core.Example, prediction map[string]any) (float64, error) {
	o.mu.Lock()
	o.inFlight++
	o.peak = max(o.peak, o.inFlight)
	o.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
	o.mu.Lock()
	o.inFlight--
	o.mu.Unlock()
	if example.Inputs["q"] == "question-5" {
		return 0, errUnscorable
	}
	return positionMetric(example.Outputs, prediction), nil
}

func echoProgram() core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"alpha": newStaticCandidateTestModule("alpha"),
	}, func(map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(_ context.Context, inputs map[string]any) (map[string]any, error) {
			return map[string]any{"output": inputs["q"]}, nil
		}
	})
}

func TestGEPACompileSerializesMetricAndCompileWithScorerDoesNot(t *testing.T) {
	const level = 3
	examples := overlapExamples(7)
	newOptimizer := func(t *testing.T) *GEPA {
		llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
			return nil, errors.New("no proposals in this test")
		}}
		config := DefaultGEPAConfig()
		config.GenerationLLM, config.ReflectionLLM = llm, llm
		config.ConcurrencyLevel, config.RandomSeed = level, 1
		config.MaxMetricCalls = len(examples) // Seed validation only.
		optimizer, err := NewGEPA(config)
		require.NoError(t, err)
		return optimizer
	}

	legacy := &overlapScorer{}
	metric := func(expected, actual map[string]any) float64 {
		score, _ := legacy.Score(context.Background(), core.Example{Outputs: expected}, actual)
		return score
	}
	optimizer := newOptimizer(t)
	_, err := optimizer.Compile(context.Background(), echoProgram(), newGEPAExampleDataset(examples), metric)
	require.NoError(t, err)
	assert.Equal(t, len(examples), optimizer.GetOptimizationState().MetricCallCount())
	assert.Equal(t, 1, legacy.peak, "Compile must not overlap calls to a core.Metric")

	scorer := &overlapScorer{}
	optimizer = newOptimizer(t)
	_, err = optimizer.CompileWithScorer(context.Background(), echoProgram(), examples, scorer)
	require.NoError(t, err)
	assert.Equal(t, len(examples), optimizer.GetOptimizationState().MetricCallCount())
	assert.Equal(t, level, scorer.peak, "CompileWithScorer runs the scorer at ConcurrencyLevel")
}

func TestUnscorableExampleNeverRaisesScore(t *testing.T) {
	examples := overlapExamples(7)
	scorer := &overlapScorer{}

	t.Run("GEPA", func(t *testing.T) {
		gepa := &GEPA{config: DefaultGEPAConfig(), state: NewGEPAState()}
		candidate := &GEPACandidate{ID: "candidate", ModuleName: "alpha", Instruction: "alpha"}
		adapter := newGEPAScorerAdapter(echoProgram(), examples, scorer, true)
		evaluation := gepa.evaluateCandidateWithAdapter(gepa.withGEPAState(context.Background()), candidate, adapter)
		require.Len(t, evaluation.Cases, 7)
		assert.Equal(t, 0.25, evaluation.Cases[3].Score, "scores keep batch order")
		assert.InDelta(t, 5.25/7, evaluation.AverageScore, 1e-12, "the unscorable example counts as zero, not dropped")
	})

	t.Run("COPRO", func(t *testing.T) {
		model := newOfflineLLM()
		predictor := offlineProgram(model).Modules["route"].(*modules.Predict)
		train := offlineExamples(1, 0)
		// A judge that fails on all but one example must not report that one
		// score as the candidate's average.
		judge := core.ScorerFunc(func(_ context.Context, example core.Example, _ map[string]any) (float64, error) {
			if example.Inputs["prompt"] != train[0].Inputs["prompt"] {
				return 0, errUnscorable
			}
			return 1, nil
		})
		score, err := NewCOPRO(nil).evaluateCandidate(context.Background(), predictor, PromptCandidate{Instruction: offlineBaseline}, train, judge)
		require.NoError(t, err)
		assert.InDelta(t, 1.0/float64(len(train)), score, 1e-12)
	})

	t.Run("MIPRO", func(t *testing.T) {
		m := NewMIPRO(nil)
		_, err := m.evaluateCandidate(context.Background(), echoProgram(), examples, scorer)
		assert.ErrorIs(t, err, errUnscorable)

		score, err := m.evaluateOnMinibatch(context.Background(), echoProgram(), examples, 7, scorer)
		require.NoError(t, err)
		assert.InDelta(t, 5.25/7, score, 1e-12)
	})

	t.Run("SIMBA", func(t *testing.T) {
		simba := NewSIMBA(WithLLMConcurrency(3))
		score, err := simba.evaluateProgramOnExamples(context.Background(), echoProgram(), examples, scorer)
		require.NoError(t, err)
		assert.True(t, math.IsInf(score, -1), "same result as a failed program execution")

		score, err = simba.evaluateCandidateOnBatch(context.Background(), echoProgram(), examples, scorer)
		require.NoError(t, err)
		assert.True(t, math.IsInf(score, -1), "same result as a failed program execution")
	})
}

// A cancelled evaluation must fail, not return the average of the examples
// that happened to finish.
func TestMIPROEvaluationReportsCancellation(t *testing.T) {
	newProgram := func(cancel context.CancelFunc) core.Program {
		calls := 0
		return core.NewProgram(map[string]core.Module{}, func(_ context.Context, inputs map[string]any) (map[string]any, error) {
			if calls++; calls == 3 {
				cancel()
			}
			return map[string]any{"output": inputs["q"]}, nil
		})
	}
	scorer := core.MetricScorer(exactOutputMetric)
	m := NewMIPRO(nil)

	ctx, cancel := context.WithCancel(context.Background())
	_, err := m.evaluateCandidate(ctx, newProgram(cancel), overlapExamples(7), scorer)
	assert.ErrorIs(t, err, context.Canceled)

	ctx, cancel = context.WithCancel(context.Background())
	_, err = m.evaluateOnMinibatch(ctx, newProgram(cancel), overlapExamples(7), 7, scorer)
	assert.ErrorIs(t, err, context.Canceled)
}

// The full-dataset pass was sequential before it shared the scoring pool, so
// it must stay sequential unless LLMConcurrency asks for more.
func TestSIMBAFullEvaluationIsSequentialByDefault(t *testing.T) {
	scorer := &wiringScorer{}
	_, err := NewSIMBA().evaluateProgramOnExamples(context.Background(), echoProgram(), overlapExamples(40), scorer)
	require.NoError(t, err)
	assert.Equal(t, 40, scorer.calls)
	assert.Equal(t, 1, scorer.peak)
}

// Only CompileWithScorer may score the full dataset concurrently when
// LLMConcurrency is unset; a core.Metric never ran concurrently there.
func TestSIMBAFullEvaluationIsConcurrentOnlyForScorers(t *testing.T) {
	model := newOfflineLLM()
	oldDefault, oldTeacher := core.GetDefaultLLM(), core.GetTeacherLLM()
	core.SetDefaultLLM(model)
	core.SetTeacherLLM(model)
	t.Cleanup(func() {
		core.SetDefaultLLM(oldDefault)
		core.SetTeacherLLM(oldTeacher)
	})
	train := offlineExamples(1, 0)
	ctx := context.Background()
	fullEvaluationPeak := func(simba *SIMBA) int {
		scorer := &wiringScorer{}
		_, err := simba.evaluateProgramOnExamples(ctx, echoProgram(), overlapExamples(40), scorer)
		require.NoError(t, err)
		return scorer.peak
	}

	simba := NewSIMBA()
	_, err := simba.CompileWithScorer(ctx, offlineProgram(model), train, core.MetricScorer(offlineMetric))
	require.NoError(t, err)
	assert.Greater(t, fullEvaluationPeak(simba), 1)

	_, err = simba.CompileExamples(ctx, offlineProgram(model), train, offlineMetric)
	require.NoError(t, err)
	assert.Equal(t, 1, fullEvaluationPeak(simba))

	simba = NewSIMBA()
	_, err = simba.CompileWithScorer(ctx, offlineProgram(model), train, core.MetricScorer(offlineMetric))
	require.NoError(t, err)
	_, err = simba.Compile(ctx, offlineProgram(model), datasets.NewSimpleDataset(train), offlineMetric)
	require.NoError(t, err)
	assert.Equal(t, 1, fullEvaluationPeak(simba))
}

// wiringScorer scores like offlineMetric, or exact output match outside the
// offline fixture, and records how its calls arrived.
type wiringScorer struct {
	mu                    sync.Mutex
	inFlight, peak, calls int
	withoutCallerContext  int
}

type wiringContextKey struct{}

func (w *wiringScorer) Score(ctx context.Context, example core.Example, prediction map[string]any) (float64, error) {
	w.mu.Lock()
	w.calls++
	w.inFlight++
	w.peak = max(w.peak, w.inFlight)
	if ctx.Value(wiringContextKey{}) == nil {
		w.withoutCallerContext++
	}
	w.mu.Unlock()
	time.Sleep(2 * time.Millisecond)
	w.mu.Lock()
	w.inFlight--
	w.mu.Unlock()
	if _, offline := example.Outputs["completion"]; offline {
		return offlineMetric(example.Outputs, prediction), nil
	}
	return exactOutputMetric(example.Outputs, prediction), nil
}

func (w *wiringScorer) metric(expected, actual map[string]any) float64 {
	score, _ := w.Score(context.Background(), core.Example{Outputs: expected}, actual)
	return score
}

// These run each optimizer through its public entry points on the offline
// fixture, so they fail if an entry point is wired to the wrong concurrency or
// drops the caller's context on the way to the scorer.
func TestCompileEntryPointsScoreAsDocumented(t *testing.T) {
	const level = 3
	previous := core.GetConcurrencyLevel()
	core.SetConcurrencyOptions(level)
	model := newOfflineLLM()
	oldDefault, oldTeacher := core.GetDefaultLLM(), core.GetTeacherLLM()
	core.SetDefaultLLM(model)
	core.SetTeacherLLM(model)
	t.Cleanup(func() {
		core.SetConcurrencyOptions(previous)
		core.SetDefaultLLM(oldDefault)
		core.SetTeacherLLM(oldTeacher)
	})
	train := offlineExamples(1, 0)
	ctx := context.WithValue(context.Background(), wiringContextKey{}, true)

	newGEPA := func() *GEPA {
		config := DefaultGEPAConfig()
		config.RandomSeed, config.ConcurrencyLevel = 1, level
		optimizer, err := NewGEPA(config)
		require.NoError(t, err)
		return optimizer
	}
	newMIPRO := func(metric miproMetric) *MIPRO {
		return NewMIPRO(metric, WithRandomSeed(1), WithNumTrials(3), WithMiniBatchSize(6))
	}

	// GEPA and MIPRO never overlapped calls to a metric.
	serialized := map[string]func(*wiringScorer) error{
		"GEPA.Compile": func(w *wiringScorer) error {
			_, err := newGEPA().Compile(ctx, offlineProgram(model), datasets.NewSimpleDataset(train), w.metric)
			return err
		},
		"MIPRO.Compile": func(w *wiringScorer) error {
			_, err := newMIPRO(nil).Compile(ctx, offlineProgram(model), datasets.NewSimpleDataset(train), w.metric)
			return err
		},
		"MIPRO.CompileExamples": func(w *wiringScorer) error {
			_, err := newMIPRO(nil).CompileExamples(ctx, offlineProgram(model), train, w.metric)
			return err
		},
		"MIPRO constructor metric": func(w *wiringScorer) error {
			metric := func(expected, actual map[string]any, _ context.Context) float64 { return w.metric(expected, actual) }
			_, err := newMIPRO(metric).CompileExamples(ctx, offlineProgram(model), train, nil)
			return err
		},
	}
	for name, compile := range serialized {
		t.Run(name, func(t *testing.T) {
			legacy := &wiringScorer{}
			require.NoError(t, compile(legacy))
			assert.Positive(t, legacy.calls)
			assert.Equal(t, 1, legacy.peak, "calls to a core.Metric must not overlap")
		})
	}

	compilers := map[string]core.ScorerCompiler{
		"GEPA":  newGEPA(),
		"MIPRO": newMIPRO(nil),
		"SIMBA": NewSIMBA(WithLLMConcurrency(level)),
		"COPRO": NewCOPRO(nil),
	}
	for name, compiler := range compilers {
		t.Run(name+".CompileWithScorer", func(t *testing.T) {
			scorer := &wiringScorer{}
			_, err := compiler.CompileWithScorer(ctx, offlineProgram(model), train, scorer)
			require.NoError(t, err)
			assert.Greater(t, scorer.peak, 1, "a Scorer is called concurrently")
			assert.Zero(t, scorer.withoutCallerContext, "every Score call receives the caller's context")
		})
		t.Run(name+" rejects a nil metric", func(t *testing.T) {
			_, err := compiler.CompileWithScorer(ctx, offlineProgram(model), train, core.MetricScorer(nil))
			assert.Error(t, err)
		})
	}
}

// instructionProgram answers correctly only once its module's instruction is
// "better", so a baseline and a proposal score differently on the same batch.
func instructionProgram() core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{
		"alpha": newStaticCandidateTestModule("alpha"),
	}, func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(_ context.Context, inputs map[string]any) (map[string]any, error) {
			switch {
			case inputs["q"] == "fail":
				return nil, errors.New("boom")
			case modules["alpha"].GetSignature().Instruction == "better":
				return map[string]any{"output": inputs["q"]}, nil
			}
			return map[string]any{"output": "wrong"}, nil
		}
	})
}

func TestGEPAEvaluationFailures(t *testing.T) {
	examples := overlapExamples(4)
	better := func() *GEPACandidate {
		return &GEPACandidate{ID: "better", ModuleName: "alpha", Instruction: "better"}
	}
	newGEPA := func() *GEPA {
		config := DefaultGEPAConfig()
		config.ConcurrencyLevel = 1
		return &GEPA{config: config, state: NewGEPAState()}
	}

	t.Run("a scorer error counts as zero and is retried", func(t *testing.T) {
		gepa := newGEPA()
		var mu sync.Mutex
		calls := 0
		// The judge is down for the first evaluation. A score returned with
		// an error must be ignored.
		judge := core.ScorerFunc(func(_ context.Context, example core.Example, prediction map[string]any) (float64, error) {
			mu.Lock()
			defer mu.Unlock()
			calls++
			if calls <= len(examples) {
				return 1, errUnscorable
			}
			return exactOutputMetric(example.Outputs, prediction), nil
		})
		adapter := newGEPAScorerAdapter(instructionProgram(), examples, judge, true)
		ctx := gepa.withGEPAState(context.Background())

		first := gepa.evaluateCandidateWithAdapter(ctx, better(), adapter)
		require.Len(t, first.Cases, len(examples))
		assert.Zero(t, first.TotalScore)
		assert.Zero(t, first.AverageScore)

		second := gepa.evaluateCandidateWithAdapter(ctx, better(), adapter)
		assert.Equal(t, 1.0, second.AverageScore, "the failed scores were not cached")
		assert.Equal(t, 2*len(examples), calls)
	})

	t.Run("a failed execution counts as zero in the average", func(t *testing.T) {
		gepa := newGEPA()
		batch := append(overlapExamples(1), core.Example{Inputs: map[string]any{"q": "fail"}}, core.Example{Inputs: map[string]any{"q": "fail"}})
		adapter := newGEPAScorerAdapter(instructionProgram(), batch, core.MetricScorer(exactOutputMetric), true)
		evaluation := gepa.evaluateCandidateWithAdapter(gepa.withGEPAState(context.Background()), better(), adapter)
		assert.Equal(t, 1.0, evaluation.TotalScore)
		assert.InDelta(t, 1.0/3, evaluation.AverageScore, 1e-12)
	})

	// cancelOn returns a scorer that cancels the run while scoring question.
	cancelOn := func(question string, cancel context.CancelFunc) core.Scorer {
		return core.ScorerFunc(func(ctx context.Context, example core.Example, prediction map[string]any) (float64, error) {
			if example.Inputs["q"] == question && prediction["output"] == question {
				cancel()
				return 0, ctx.Err()
			}
			return exactOutputMetric(example.Outputs, prediction), nil
		})
	}

	t.Run("a cancelled score stays an error", func(t *testing.T) {
		gepa := newGEPA()
		ctx, cancel := context.WithCancel(gepa.withGEPAState(context.Background()))
		defer cancel()
		adapter := newGEPAScorerAdapter(instructionProgram(), examples, cancelOn("question-2", cancel), true)
		evaluation := gepa.evaluateCandidateWithAdapter(ctx, better(), adapter)
		require.Len(t, evaluation.Cases, 3, "examples after the cancellation do not run")
		assert.ErrorIs(t, evaluation.Cases[2].Err, context.Canceled)
		assert.False(t, gepa.evaluationIsPerfect(evaluation))
	})

	t.Run("a proposal evaluated only in part is rejected", func(t *testing.T) {
		gepa := newGEPA()
		ctx, cancel := context.WithCancel(gepa.withGEPAState(context.Background()))
		defer cancel()
		adapter := newGEPAScorerAdapter(instructionProgram(), examples, cancelOn("question-2", cancel), true)
		baseline := &GEPACandidate{ID: "baseline", ModuleName: "alpha", Instruction: "alpha"}
		// The proposal scores 2 on the two examples it finished; the baseline
		// scores 0 on all four.
		accepted := gepa.acceptCandidateProposalWithAdapter(ctx, baseline, better(), adapter, nil)
		assert.Same(t, baseline, accepted)
	})
}
