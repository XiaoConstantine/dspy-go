package optimizers

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/adapters"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// policyModel answers question q<i> with gold-q<i>. It gives an unparseable
// reply to the questions in unparseable and fails to answer those in
// unreachable.
type policyModel struct {
	core.LLM
	unparseable, unreachable map[string]bool
	calls                    atomic.Int64
}

var errUnreachable = errors.New("connection reset")

func (m *policyModel) ModelID() string      { return "policy" }
func (m *policyModel) ProviderName() string { return "test" }
func (m *policyModel) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *policyModel) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	m.calls.Add(1)
	// The question asked is the last one in the prompt, after any demonstrations.
	_, question, _ := strings.Cut(prompt[strings.LastIndex(prompt, "<question>"):], "\n")
	question, _, _ = strings.Cut(question, "\n")
	switch {
	case m.unreachable[question]:
		return nil, errUnreachable
	case m.unparseable[question]:
		return &core.LLMResponse{Content: "I would rather not use tags."}, nil
	}
	return &core.LLMResponse{Content: "<answer>gold-" + question + "</answer>"}, nil
}

func questions(indexes ...int) map[string]bool {
	set := make(map[string]bool, len(indexes))
	for _, i := range indexes {
		set[fmt.Sprintf("q%d", i)] = true
	}
	return set
}

// policyProgram is one predictor that uses the XML adapter with no fallback,
// so an unparseable reply is an output error after exactly one call.
func policyProgram(model *policyModel) core.Program {
	predictor := modules.NewPredict(scriptQASignature("base")).
		WithAdapter(adapters.NewXMLAdapter().WithJSONFallback(false))
	predictor.SetLLM(model)
	return coproSinglePredictorProgram(predictor)
}

// Eight examples. The policy scores an output error 0.5 and allows three.
var (
	policyUnderTest = EvalPolicy{FailureScore: 0.5, MaxErrors: 3}
	policyScorer    = core.MetricScorer(scriptExactMatch)
)

func requireMaxErrors(t *testing.T, err error) {
	t.Helper()
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	require.ErrorIs(t, err, ErrMaxErrors)
	assert.ErrorContains(t, err, "max_errors")
	assert.ErrorContains(t, err, "4 output errors > limit 3")
}

func TestCOPROEvaluationAppliesEvalPolicy(t *testing.T) {
	evaluate := func(model *policyModel) (float64, error) {
		copro := NewCOPRO(scriptExactMatch, WithCOPROEvalPolicy(policyUnderTest))
		return copro.evaluateCandidate(context.Background(), policyProgram(model), "predictor",
			PromptCandidate{Instruction: "base"}, scriptQAExamples(8), policyScorer)
	}

	score, err := evaluate(&policyModel{unparseable: questions(1, 5, 6)})
	require.NoError(t, err, "three output errors are within the limit")
	assert.InDelta(t, (5+3*0.5)/8, score, 1e-12)

	_, err = evaluate(&policyModel{unparseable: questions(1, 2, 5, 6)})
	requireMaxErrors(t, err)

	score, err = evaluate(&policyModel{unreachable: questions(1, 2, 5, 6)})
	require.NoError(t, err, "other failures count as zero and never stop the run, as before")
	assert.InDelta(t, 4.0/8, score, 1e-12)
}

func TestSIMBAEvaluationAppliesEvalPolicy(t *testing.T) {
	evaluations := map[string]func(*SIMBA, core.Program) (float64, error){
		"whole program": func(s *SIMBA, program core.Program) (float64, error) {
			return s.evaluateProgramOnExamples(context.Background(), program, scriptQAExamples(8), policyScorer)
		},
		"candidate batch": func(s *SIMBA, program core.Program) (float64, error) {
			return s.evaluateCandidateOnBatch(context.Background(), program, scriptQAExamples(8), policyScorer)
		},
	}
	for name, evaluate := range evaluations {
		t.Run(name, func(t *testing.T) {
			run := func(model *policyModel) (float64, error) {
				return evaluate(NewSIMBA(WithSIMBAEvalPolicy(policyUnderTest)), policyProgram(model))
			}

			score, err := run(&policyModel{unparseable: questions(1, 5, 6)})
			require.NoError(t, err)
			assert.InDelta(t, (5+3*0.5)/8, score, 1e-12, "an output error is a low score, not an unusable program")

			_, err = run(&policyModel{unparseable: questions(1, 2, 5, 6)})
			requireMaxErrors(t, err)

			score, err = run(&policyModel{unreachable: questions(1)})
			require.NoError(t, err)
			assert.True(t, math.IsInf(score, -1), "any other failure makes the program unusable, as before")
		})
	}
}

func TestGEPAEvaluationAppliesEvalPolicy(t *testing.T) {
	newGEPA := func(t *testing.T) *GEPA {
		llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
			return nil, errors.New("no proposals in this test")
		}}
		config := DefaultGEPAConfig()
		config.EvalPolicy = policyUnderTest
		config.GenerationLLM, config.ReflectionLLM = llm, llm
		config.RandomSeed = 1
		config.MaxMetricCalls = 8 // Seed validation only.
		optimizer, err := NewGEPA(config)
		require.NoError(t, err)
		return optimizer
	}

	t.Run("an output error scores the failure score", func(t *testing.T) {
		gepa := &GEPA{config: DefaultGEPAConfig(), state: NewGEPAState()}
		gepa.config.EvalPolicy = policyUnderTest
		adapter := newGEPAScorerAdapter(policyProgram(&policyModel{unparseable: questions(1, 5, 6)}), scriptQAExamples(8), policyScorer, true)
		candidate := &GEPACandidate{ID: "candidate", ModuleName: "predictor", Instruction: "base"}

		evaluation := gepa.evaluateCandidateWithAdapter(gepa.withGEPAState(context.Background()), candidate, adapter)
		require.Len(t, evaluation.Cases, 8)
		assert.InDelta(t, (5+3*0.5)/8, evaluation.AverageScore, 1e-12)
		require.ErrorIs(t, evaluation.Cases[1].Err, dspyerrors.ErrParseFailed)
	})

	t.Run("output errors within the limit do not stop a compilation", func(t *testing.T) {
		_, err := newGEPA(t).CompileWithScorer(context.Background(),
			policyProgram(&policyModel{unparseable: questions(1, 5, 6)}), scriptQAExamples(8), policyScorer)
		require.NoError(t, err)
	})

	t.Run("output errors over the limit stop the compilation", func(t *testing.T) {
		program := policyProgram(&policyModel{unparseable: questions(1, 2, 5, 6)})
		compiled, err := newGEPA(t).CompileWithScorer(context.Background(), program, scriptQAExamples(8), policyScorer)
		requireMaxErrors(t, err)
		assert.Equal(t, "base", compiled.Modules["predictor"].GetSignature().Instruction, "the original program is returned")
	})
}

func TestBootstrapFewShotLimitsTeacherOutputErrors(t *testing.T) {
	compile := func(policy EvalPolicy, unparseable ...int) (core.Program, error) {
		optimizer := NewBootstrapFewShot(keepEverything, 8)
		optimizer.EvalPolicy = policy
		model := &policyModel{unparseable: questions(unparseable...)}
		return optimizer.CompileLegacy(context.Background(), policyProgram(model), policyProgram(model), bootstrapTrainset(8))
	}

	compiled, err := compile(policyUnderTest, 1, 5, 6)
	require.NoError(t, err)
	assert.Equal(t, []any{"q0", "q2", "q3", "q4", "q7"}, bootstrapQuestions(t, compiled), "a failed teacher run yields no demonstration")

	_, err = compile(policyUnderTest, 1, 2, 5, 6)
	requireMaxErrors(t, err)

	compiled, err = compile(EvalPolicy{MaxErrors: -1}, 1, 2, 5, 6)
	require.NoError(t, err, "a negative MaxErrors sets no limit")
	assert.Len(t, bootstrapQuestions(t, compiled), 4)

	_, err = compile(EvalPolicy{}, 5)
	require.ErrorIs(t, err, ErrMaxErrors, "the zero policy stops at the first output error")
	assert.ErrorContains(t, err, "1 output errors > limit 0")

	compiled, err = compile(EvalPolicy{})
	require.NoError(t, err, "the zero policy changes nothing for a run without output errors")
	assert.Len(t, bootstrapQuestions(t, compiled), 8)
}
