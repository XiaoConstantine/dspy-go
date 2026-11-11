package optimizers

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/require"
)

type gepaLoopLLM struct {
	core.LLM
	generate func(string) (*core.LLMResponse, error)
}

func (m *gepaLoopLLM) ModelID() string { return "gepa-loop-test" }

func (m *gepaLoopLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	return m.generate(prompt)
}

func TestGEPACompileRetainsParentsAndValidatesOnlyAcceptedChildren(t *testing.T) {
	var proposals int
	llm := &gepaLoopLLM{generate: func(prompt string) (*core.LLMResponse, error) {
		require.Contains(t, prompt, "EXAMPLE-LEVEL EVIDENCE:")
		require.Contains(t, prompt, "Cases Evaluated: 2")
		require.Contains(t, prompt, "Trace: module=alpha")
		require.NotContains(t, prompt, "validation-only")
		proposals++
		if proposals == 5 {
			return nil, fmt.Errorf("reflection unavailable")
		}
		return &core.LLMResponse{Content: []string{"tied instruction", "worse instruction", "overfit instruction", "improved instruction"}[proposals-1]}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize, config.RandomSeed = 6, 2, 1
	for i := range 3 {
		config.ValidationExamples = append(config.ValidationExamples, core.Example{
			Inputs: map[string]any{"case": fmt.Sprintf("validation-only-%d", i)}, Outputs: map[string]any{"validation": true},
		})
	}
	var train []core.Example
	for i := range 4 {
		train = append(train, core.Example{Inputs: map[string]any{"case": i}})
	}
	validationCalls := make(map[string]int)
	metric := func(expected, actual map[string]any) float64 {
		instruction := actual["output"].(string)
		if expected["validation"] == true {
			validationCalls[instruction]++
			return map[string]float64{"seed instruction": 0.6, "overfit instruction": 0.4, "improved instruction": 0.9}[instruction]
		}
		return map[string]float64{"seed instruction": 0.25, "tied instruction": 0.25, "worse instruction": 0.1, "overfit instruction": 0.5, "improved instruction": 0.75}[instruction]
	}
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	base := core.NewProgramWithForwardFactory(map[string]core.Module{"alpha": newStaticCandidateTestModule("seed instruction")},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				outputs := map[string]any{"output": modules["alpha"].GetSignature().Instruction}
				if state := GetGEPAState(ctx); state != nil {
					state.AddTraceContext(ctx, &ExecutionTrace{
						CandidateID: CurrentCandidateID(ctx), ModuleName: "alpha",
						Inputs: map[string]any{"inner": inputs["case"]}, Outputs: outputs,
					})
				}
				return outputs, nil
			}
		})
	optimized, err := optimizer.Compile(context.Background(), base, datasets.NewSimpleDataset(train), metric)
	require.NoError(t, err)
	require.Equal(t, "improved instruction", optimized.Modules["alpha"].GetSignature().Instruction)
	require.Equal(t, "seed instruction", base.Modules["alpha"].GetSignature().Instruction)
	require.Equal(t, 5, proposals, "one proposal call per attempted iteration, no initialization, critique, or failure fallback")
	require.Equal(t, map[string]int{"seed instruction": 3, "overfit instruction": 3, "improved instruction": 3}, validationCalls)
	population := optimizer.CurrentPopulation()
	require.Len(t, population.Candidates, 3)
	seed := population.Candidates[0]
	require.Equal(t, "seed instruction", seed.Instruction)
	for _, candidate := range population.Candidates[1:] {
		require.Equal(t, []string{seed.ID}, candidate.ParentIDs, "the validation-best parent remains selectable after accepting an overfit child")
	}
	require.InDelta(t, 0.9, optimizer.GetOptimizationState().BestValidationFitness, 1e-9)
}

func TestGEPACompileBudgetRequiresCompleteValidation(t *testing.T) {
	for _, budget := range []int{2, 3, 4, 6} {
		t.Run(fmt.Sprint(budget), func(t *testing.T) {
			llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
				return &core.LLMResponse{Content: "improved instruction"}, nil
			}}
			config := DefaultGEPAConfig()
			config.GenerationLLM, config.ReflectionLLM = llm, llm
			config.MaxMetricCalls, config.EvaluationBatchSize = budget, 1
			config.RandomSeed = 1
			var examples []core.Example
			for i := range 3 {
				examples = append(examples, core.Example{Inputs: map[string]any{"case": i}})
			}
			calls := 0
			optimizer, err := NewGEPA(config)
			require.NoError(t, err)
			optimized, err := optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("seed instruction"), datasets.NewSimpleDataset(examples), func(_, actual map[string]any) float64 {
				calls++
				if actual["output"] == "improved instruction" {
					return 1
				}
				return 0.25
			})
			require.NoError(t, err)
			require.LessOrEqual(t, calls, budget)
			require.Equal(t, calls, optimizer.GetOptimizationState().MetricCallCount())
			want := "seed instruction"
			if budget == 6 {
				want = "improved instruction"
				require.Equal(t, 6, calls)
			} else {
				require.Equal(t, gepaStopReasonMetricBudget, optimizer.GetOptimizationState().StopReason)
				require.Len(t, optimizer.CurrentPopulation().Candidates, 1)
			}
			require.Equal(t, want, optimized.Modules["alpha"].GetSignature().Instruction)
		})
	}
}

func TestGEPACompileSkipsPerfectMinibatches(t *testing.T) {
	proposals := 0
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		proposals++
		return nil, fmt.Errorf("proposal unavailable")
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize = 3, 1
	config.PerfectScore = 10 // Do not assume the user's metric has a 0–1 scale.
	config.ValidationExamples = []core.Example{{Outputs: map[string]any{"score": 5.0}}}
	examples := []core.Example{{Outputs: map[string]any{"score": 10.0}}, {Outputs: map[string]any{"score": 9.0}}}
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	_, err = optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("seed instruction"), datasets.NewSimpleDataset(examples), func(expected, _ map[string]any) float64 {
		return expected["score"].(float64)
	})
	require.NoError(t, err)
	require.Equal(t, 1, proposals, "skip the solved case, not the entire epoch")
	require.Len(t, optimizer.CurrentPopulation().Candidates, 1)
}

func TestGEPACompileWithoutProposalEvidenceFailsOfflineGate(t *testing.T) {
	model := newOfflineLLM()
	reflection := &gepaLoopLLM{generate: func(prompt string) (*core.LLMResponse, error) {
		withoutEvidence, _, _ := strings.Cut(prompt, "EXAMPLE-LEVEL EVIDENCE:")
		return model.Generate(context.Background(), withoutEvidence)
	}}
	config := DefaultGEPAConfig()
	config.RandomSeed = 1
	config.GenerationLLM, config.ReflectionLLM = model, reflection
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	optimized, err := optimizer.Compile(context.Background(), offlineProgram(model), datasets.NewSimpleDataset(offlineExamples(1, 0)), offlineMetric)
	require.NoError(t, err)
	score, _, err := offlineScore(context.Background(), optimized, offlineExamples(1001, 10000), offlineMetric)
	require.NoError(t, err)
	require.Equal(t, 0.25, score)
	require.Error(t, offlineGate(0.25, score))
	t.Logf("no-evidence optimizer control: baseline=0.250 optimized=%.3f, score gate rejected", score)
}

func TestGEPACompileProposalFormats(t *testing.T) {
	for _, tc := range []struct {
		name, selector, response, classifier, generator string
	}{
		{"multiline", "round_robin", "```text\nclassifier improved\nKeep this second line.\n```", "classifier improved\nKeep this second line.", "generator base"},
		{"all", "all", `{"classifier":"classifier improved","generator":"generator improved"}`, "classifier improved", "generator improved"},
		{"malformed", "all", "not JSON", "classifier base", "generator base"},
		{"missing module", "all", `{"classifier":"classifier improved"}`, "classifier base", "generator base"},
		{"empty", "round_robin", "", "classifier base", "generator base"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
				calls++
				return &core.LLMResponse{Content: tc.response}, nil
			}}
			config := DefaultGEPAConfig()
			config.GenerationLLM, config.ReflectionLLM = llm, llm
			config.MaxGenerations, config.ComponentSelection = 2, tc.selector
			optimizer, err := NewGEPA(config)
			require.NoError(t, err)
			optimized, err := optimizer.Compile(context.Background(), newCompatFixtureProgram("classifier base", "generator base"), datasets.NewSimpleDataset([]core.Example{{}}), compatInstructionProgressMetric)
			require.NoError(t, err)
			require.Equal(t, 1, calls)
			require.Equal(t, tc.classifier, optimized.Modules["classifier"].GetSignature().Instruction)
			require.Equal(t, tc.generator, optimized.Modules["generator"].GetSignature().Instruction)
		})
	}
}
