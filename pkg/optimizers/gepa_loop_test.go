package optimizers

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
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

func TestGEPACompileRetriesTransientExecutionErrors(t *testing.T) {
	var failed atomic.Bool
	var attempts atomic.Int64
	program := core.NewProgramWithForwardFactory(map[string]core.Module{"alpha": newStaticCandidateTestModule("seed instruction")},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(_ context.Context, inputs map[string]any) (map[string]any, error) {
				if inputs["case"] == 0 && modules["alpha"].GetSignature().Instruction == "seed instruction" {
					attempts.Add(1)
					if failed.CompareAndSwap(false, true) {
						return nil, fmt.Errorf("transient rate limit")
					}
				}
				return map[string]any{"output": modules["alpha"].GetSignature().Instruction}, nil
			}
		})
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "worse instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize, config.RandomSeed = 4, 2, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	examples := []core.Example{{Inputs: map[string]any{"case": 0}}, {Inputs: map[string]any{"case": 1}}}
	scorer := core.ScorerFunc(func(_ context.Context, _ core.Example, actual map[string]any) (float64, error) {
		if actual["output"] == "seed instruction" {
			return 1, nil
		}
		return 0.75, nil // Strictly worse on every example, even the one that initially failed.
	})
	optimized, err := optimizer.CompileWithScorer(context.Background(), program, examples, scorer)
	require.NoError(t, err)
	require.Equal(t, "seed instruction", optimized.Modules["alpha"].GetSignature().Instruction)
	require.Equal(t, int64(2), attempts.Load(), "retry the failed execution, then cache its successful result")
	require.Len(t, optimizer.CurrentPopulation().Candidates, 1)
}

func TestGEPACompileRetainsDistinctCandidates(t *testing.T) {
	llm := &gepaLoopLLM{generate: func(prompt string) (*core.LLMResponse, error) {
		proposal := "specialist A"
		if strings.Contains(prompt, `CURRENT INSTRUCTION: "specialist A"`) {
			proposal = "specialist B"
		}
		return &core.LLMResponse{Content: proposal}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize, config.RandomSeed = 20, 1, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	examples := []core.Example{{Outputs: map[string]any{"specialist": "specialist A"}}, {Outputs: map[string]any{"specialist": "specialist B"}}}
	_, err = optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("seed instruction"), datasets.NewSimpleDataset(examples), func(expected, actual map[string]any) float64 {
		if actual["output"] == expected["specialist"] {
			return 0.9
		}
		if actual["output"] == "seed instruction" {
			return 0.1
		}
		return 0.2
	})
	require.NoError(t, err)
	var instructions []string
	for _, candidate := range optimizer.CurrentPopulation().Candidates {
		instructions = append(instructions, candidate.Instruction)
	}
	require.ElementsMatch(t, []string{"seed instruction", "specialist A", "specialist B"}, instructions)
}

func TestGEPACompileStagnationSkipsMissingChildren(t *testing.T) {
	for _, kind := range []string{"empty", "unchanged", "failure"} {
		t.Run(kind, func(t *testing.T) {
			calls := 0
			llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
				calls++
				if calls > 3 {
					return &core.LLMResponse{Content: "improved instruction"}, nil
				}
				switch kind {
				case "unchanged":
					return &core.LLMResponse{Content: "seed instruction"}, nil
				case "failure":
					return nil, fmt.Errorf("reflection unavailable")
				default:
					return &core.LLMResponse{}, nil
				}
			}}
			config := DefaultGEPAConfig()
			config.GenerationLLM, config.ReflectionLLM = llm, llm
			config.MaxGenerations, config.StagnationLimit = 5, 3
			optimizer, err := NewGEPA(config)
			require.NoError(t, err)
			optimized, err := optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("seed instruction"), datasets.NewSimpleDataset([]core.Example{{}}), func(_, actual map[string]any) float64 {
				if actual["output"] == "improved instruction" {
					return 1
				}
				return 0.5
			})
			require.NoError(t, err)
			require.Equal(t, "improved instruction", optimized.Modules["alpha"].GetSignature().Instruction)
			require.Zero(t, optimizer.GetOptimizationState().ConvergenceStatus.StagnationCount)
		})
	}
}

func TestGEPACompileStagnationResetsOnSmallGains(t *testing.T) {
	steps := 0
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		steps++
		return &core.LLMResponse{Content: fmt.Sprintf("fix %d", steps)}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.StagnationLimit, config.RandomSeed = 8, 3, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	optimized, err := optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("fix 0"), datasets.NewSimpleDataset([]core.Example{{}}), func(_, actual map[string]any) float64 {
		var step int
		_, err := fmt.Sscanf(actual["output"].(string), "fix %d", &step)
		require.NoError(t, err)
		return 0.5 + 0.005*float64(step)
	})
	require.NoError(t, err)
	require.Equal(t, "fix 7", optimized.Modules["alpha"].GetSignature().Instruction)
	require.InDelta(t, 0.535, optimizer.GetOptimizationState().BestValidationFitness, 1e-9)
	require.Zero(t, optimizer.GetOptimizationState().ConvergenceStatus.StagnationCount)
}

func TestGEPACompileStagnationIsNotWallClockTimeout(t *testing.T) {
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "rejected instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.StagnationLimit = 10, 3
	config.StopCallbacks = []GEPAStopper{func(_ context.Context, optimizer *GEPA) *GEPAStopDecision {
		// Simulate a slow model without sleeping: elapsed time is governed by
		// MaxRuntime, not by a count of unsuccessful proposals.
		state := optimizer.GetOptimizationState()
		state.LastImprovement = time.Now().Add(-time.Hour)
		return nil
	}}
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	_, err = optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("seed instruction"), datasets.NewSimpleDataset([]core.Example{{}}), func(_, _ map[string]any) float64 { return 0.5 })
	require.NoError(t, err)
	require.Equal(t, 3, optimizer.GetOptimizationState().CurrentGeneration)
	require.True(t, optimizer.GetOptimizationState().ConvergenceStatus.IsConverged)
}

func TestGEPACompileStartsFreshWithoutCheckpoint(t *testing.T) {
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "improved instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.RandomSeed = 3, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	metric := func(_, actual map[string]any) float64 {
		switch actual["output"] {
		case "second program seed":
			return 1
		case "improved instruction":
			return 0.9
		default:
			return 0.1
		}
	}
	first, err := optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("first program seed"), datasets.NewSimpleDataset([]core.Example{{}}), metric)
	require.NoError(t, err)
	require.Equal(t, "improved instruction", first.Modules["alpha"].GetSignature().Instruction)
	second, err := optimizer.Compile(context.Background(), newCandidateEvaluationTestProgram("second program seed"), datasets.NewSimpleDataset([]core.Example{{}}), metric)
	require.NoError(t, err)
	require.Equal(t, "second program seed", second.Modules["alpha"].GetSignature().Instruction)
	require.Len(t, optimizer.CurrentPopulation().Candidates, 1)
	require.Equal(t, 1, optimizer.GetOptimizationState().MetricCallCount())
}

func TestGEPACompilePreservesUserInterceptors(t *testing.T) {
	predict := modules.NewPredict(core.NewSignature(nil, []core.OutputField{{Field: core.NewField("answer")}}).WithInstruction("seed instruction")).WithTextOutput()
	predict.DisplayName = "alpha"
	predict.SetLLM(&gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "answer: missing interceptors"}, nil
	}})
	predict.SetInterceptors([]core.ModuleInterceptor{
		func(ctx context.Context, inputs map[string]any, _ *core.ModuleInfo, next core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
			output, err := next(ctx, inputs, opts...)
			if err == nil {
				output["answer"] = output["answer"].(string) + " outer"
			}
			return output, err
		},
		func(_ context.Context, _ map[string]any, info *core.ModuleInfo, _ core.ModuleHandler, _ ...core.Option) (map[string]any, error) {
			// Short-circuit the model, as structured-output interceptors can.
			return map[string]any{"answer": info.Signature.Instruction}, nil
		},
	})
	program := core.NewProgramWithForwardFactory(map[string]core.Module{"alpha": predict}, func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			return modules["alpha"].(core.InterceptableModule).ProcessWithInterceptors(ctx, inputs, nil)
		}
	})
	var reflectionPrompt string
	llm := &gepaLoopLLM{generate: func(prompt string) (*core.LLMResponse, error) {
		reflectionPrompt = prompt
		return &core.LLMResponse{Content: "improved instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM, config.MaxGenerations = llm, llm, 2
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	optimized, err := optimizer.Compile(context.Background(), program, datasets.NewSimpleDataset([]core.Example{{}}), func(_, actual map[string]any) float64 {
		if actual["answer"] == "improved instruction outer" {
			return 1
		}
		return 0
	})
	require.NoError(t, err)
	output, err := optimized.Execute(context.Background(), nil)
	require.NoError(t, err)
	require.Equal(t, "improved instruction outer", output["answer"])
	require.Contains(t, reflectionPrompt, "Trace: module=alpha")
	require.Contains(t, reflectionPrompt, "seed instruction outer")
	require.Len(t, optimized.Modules["alpha"].(core.InterceptableModule).GetInterceptors(), 2, "return only the caller's interceptors, without GEPA tracking")
	require.Len(t, predict.GetInterceptors(), 2)
	require.Equal(t, "seed instruction", predict.GetSignature().Instruction)
}

func TestGEPACompileProposalFormats(t *testing.T) {
	for _, tc := range []struct {
		name, selector, response, classifier, generator string
	}{
		{"multiline", "round_robin", "```text\nclassifier improved\nKeep this second line.\n```", "classifier improved\nKeep this second line.", "generator base"},
		{"all", "all", `{"classifier":"classifier improved","generator":"generator improved"}`, "classifier improved", "generator improved"},
		{"preamble and fence", "round_robin", "Here is the improved instruction:\n```\nClassify improved.\n```\nLet me know if you want changes.", "Classify improved.", "generator base"},
		{"numbered steps", "round_robin", "1. Read the improved question.\n2. Answer briefly.", "1. Read the improved question.\n2. Answer briefly.", "generator base"},
		{"single numbered proposal", "round_robin", "1. classifier improved", "classifier improved", "generator base"},
		{"quoted word", "round_robin", `improved: reply "yes" or "no"`, `improved: reply "yes" or "no"`, "generator base"},
		{"enclosing quotes", "round_robin", `"classifier improved"`, "classifier improved", "generator base"},
		{"markdown label", "round_robin", "**Rewritten instruction:** classifier improved", "classifier improved", "generator base"},
		{"markdown label colon outside", "round_robin", "**Instruction**: classifier improved", "classifier improved", "generator base"},
		{"all fenced preamble", "all", "Here are the revised instructions:\n```json\n{\"classifier\":\"classifier improved\",\"generator\":\"generator improved\"}\n```\nThese should help.", "classifier improved", "generator improved"},
		{"all preamble extra key", "all", `Here are the instructions: {"classifier":"classifier improved","generator":"generator improved","notes":{"reason":"tightened both"}} All done.`, "classifier improved", "generator improved"},
		{"nonstring module", "all", `{"classifier":{},"generator":"generator improved"}`, "classifier base", "generator base"},
		{"empty module", "all", `{"classifier":"classifier improved","generator":" "}`, "classifier base", "generator base"},
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

// gepaInstructionProgram answers every input with its module's instruction.
func gepaInstructionProgram() core.Program {
	return core.NewProgramWithForwardFactory(map[string]core.Module{"alpha": newStaticCandidateTestModule("seed instruction")},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(context.Context, map[string]any) (map[string]any, error) {
				return map[string]any{"output": modules["alpha"].GetSignature().Instruction}, nil
			}
		})
}

func TestGEPACompileChargesDuplicateExamplesOnce(t *testing.T) {
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "child instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxMetricCalls, config.ConcurrencyLevel, config.RandomSeed = 1, 3, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	var calls atomic.Int64
	scorer := core.ScorerFunc(func(context.Context, core.Example, map[string]any) (float64, error) {
		calls.Add(1)
		time.Sleep(2 * time.Millisecond) // keep the copies in flight together
		return 0.5, nil
	})
	same := core.Example{Inputs: map[string]any{"case": 0}}

	_, err = optimizer.CompileWithScorer(context.Background(), gepaInstructionProgram(), []core.Example{same, same, same}, scorer)

	require.NoError(t, err)
	require.EqualValues(t, 1, calls.Load(), "three copies of one example fit a budget of one metric call")
}

func TestGEPACompileUnscorableChildDoesNotBeatNegativeScore(t *testing.T) {
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "child instruction"}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize, config.RandomSeed = 4, 2, 1
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	scorer := core.ScorerFunc(func(_ context.Context, _ core.Example, prediction map[string]any) (float64, error) {
		if prediction["output"] == "seed instruction" {
			return -1, nil
		}
		return 0, fmt.Errorf("judge unavailable")
	})
	var train []core.Example
	for i := range 4 {
		train = append(train, core.Example{Inputs: map[string]any{"case": i}})
	}

	optimized, err := optimizer.CompileWithScorer(context.Background(), gepaInstructionProgram(), train, scorer)

	require.NoError(t, err)
	require.Equal(t, "seed instruction", optimized.Modules["alpha"].GetSignature().Instruction,
		"a child that could not be scored never replaces a scored seed")
}

func TestGEPACompileRepeatsSeededRunOnSameOptimizer(t *testing.T) {
	var log []string
	var proposals int
	llm := &gepaLoopLLM{generate: func(string) (*core.LLMResponse, error) {
		proposals++
		log = append(log, "propose")
		return &core.LLMResponse{Content: fmt.Sprintf("instruction %d", proposals)}, nil
	}}
	config := DefaultGEPAConfig()
	config.GenerationLLM, config.ReflectionLLM = llm, llm
	config.MaxGenerations, config.EvaluationBatchSize, config.ConcurrencyLevel = 12, 2, 1
	config.MutationRate, config.RandomSeed = 0.5, 7
	optimizer, err := NewGEPA(config)
	require.NoError(t, err)
	scorer := core.ScorerFunc(func(_ context.Context, example core.Example, prediction map[string]any) (float64, error) {
		log = append(log, fmt.Sprintf("%v/%v", prediction["output"], example.Inputs["case"]))
		return 0.25, nil
	})
	var train []core.Example
	for i := range 6 {
		train = append(train, core.Example{Inputs: map[string]any{"case": i}})
	}
	run := func() []string {
		log, proposals = nil, 0
		_, err := optimizer.CompileWithScorer(context.Background(), gepaInstructionProgram(), train, scorer)
		require.NoError(t, err)
		return log
	}

	first := run()
	require.NotEmpty(t, first)
	require.Equal(t, first, run(), "the seed fixes each compilation, not the optimizer's lifetime")
}
