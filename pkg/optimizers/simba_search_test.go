package optimizers

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSIMBAAppendsSuccessfulPredictorDemos(t *testing.T) {
	for _, pipeline := range []bool{false, true} {
		for _, xml := range []bool{false, true} {
			t.Run(fmt.Sprintf("pipeline=%t/xml=%t", pipeline, xml), func(t *testing.T) {
				s := newSIMBAScript(func(prompt string) string {
					if strings.HasPrefix(prompt, "Generate a variation") {
						return "BASE"
					}
					if xml {
						return "<response><rationale>checked</rationale><token>middle</token><answer>target</answer></response>"
					}
					if strings.Contains(prompt, "rationale") {
						return "rationale: checked\ntoken: middle"
					}
					return "answer: target"
				}, WithSIMBAStrategyRatio(1), WithSIMBANumCandidates(2), WithSIMBAMaxSteps(3),
					WithSamplingTemperature(0), WithPipelineProcessing(pipeline), WithLLMConcurrency(1))
				first := modules.NewChainOfThought(core.NewSignature([]core.InputField{{Name: "question"}},
					[]core.OutputField{{Name: "token", Prefix: "token:"}}).WithInstruction("BASE"))
				last := modules.NewPredict(core.NewSignature([]core.InputField{{Name: "token"}},
					[]core.OutputField{{Name: "answer", Prefix: "answer:"}}).WithInstruction("BASE"))
				if !xml {
					first.Predict.WithTextOutput()
					last.WithTextOutput()
				} else {
					last.WithXMLOutput(interceptors.DefaultXMLConfig())
				}
				for _, p := range []*modules.Predict{first.Predict, last} {
					p.SetLLM(s.primaryModel.(core.LLM))
					// A full existing demo set exercises replacement at the default cap.
					for range 4 {
						p.Demos = append(p.Demos, core.Example{Inputs: map[string]any{"old": "demo"}, Outputs: map[string]any{"old": "output"}})
					}
				}
				base := core.NewProgramWithForwardFactory(map[string]core.Module{"first": first, "last": last},
					func(ms map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
						return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
							id := inputs["id"].(int)
							// The same predictor runs twice: retain its last invocation only.
							if _, err := ms["first"].Process(ctx, map[string]any{"question": "discard"}); err != nil {
								return nil, err
							}
							middle, err := ms["first"].Process(ctx, map[string]any{"question": fmt.Sprintf("local-%d", id)})
							if err != nil {
								return nil, err
							}
							answer, err := ms["last"].Process(ctx, map[string]any{"token": middle["token"]})
							if err != nil {
								return nil, err
							}
							learned := false
							for _, demo := range ms["first"].(*modules.ChainOfThought).Predict.Demos {
								learned = learned || demo.Inputs["question"] == "local-0"
							}
							if (id == 0 || learned) && id != 3 && middle["token"] == "middle" && answer["answer"] == "target" {
								return map[string]any{"score": 1.0}, nil
							}
							return map[string]any{"score": 0.0}, nil
						}
					})
				initial, err := base.Execute(context.Background(), map[string]any{"id": 0})
				require.NoError(t, err)
				require.Equal(t, 1.0, initial["score"], "the initial program must supply a genuinely successful trajectory")
				optimized, err := s.CompileWithScorer(context.Background(), base, simbaExamples(4), core.MetricScorer(simbaTableMetric))
				require.NoError(t, err)
				assert.Equal(t, 0.75, s.GetState().BestScore, "only predictor-local demonstrations can improve this program")
				gotFirst := optimized.Modules["first"].(*modules.ChainOfThought).Predict.Demos
				gotLast := optimized.Modules["last"].(*modules.Predict).Demos
				require.Len(t, gotFirst, 4)
				require.Len(t, gotLast, 4)
				assert.Equal(t, map[string]any{"question": "local-0"}, gotFirst[3].Inputs)
				assert.Equal(t, map[string]any{"rationale": "checked", "token": "middle"}, gotFirst[3].Outputs)
				assert.Equal(t, map[string]any{"token": "middle"}, gotLast[3].Inputs)
				assert.Equal(t, map[string]any{"answer": "target"}, gotLast[3].Outputs)
				assert.Equal(t, map[string]any{"old": "demo"}, first.Predict.Demos[3].Inputs, "do not mutate the input program")
			})
		}
	}
}

func TestSIMBARevisitsScoredProgramPool(t *testing.T) {
	for _, pipeline := range []bool{false, true} {
		t.Run(fmt.Sprintf("pipeline=%t", pipeline), func(t *testing.T) {
			var parents []string
			s := newSIMBAScript(func(prompt string) string {
				if !strings.HasPrefix(prompt, "Generate a variation") {
					return "Continue."
				}
				_, parent, _ := strings.Cut(prompt, "Original: ")
				parent, _, _ = strings.Cut(parent, "\n")
				parents = append(parents, parent)
				switch len(parents) {
				case 1:
					return "BRANCH"
				case 2:
					return "DEAD-A"
				case 3:
					return "DEAD-B"
				}
				if parent == "BRANCH" {
					return "GOOD"
				}
				return "EXHAUSTED"
			}, WithSIMBAStrategyMode("instruction_only"), WithSIMBANumCandidates(4),
				WithSamplingTemperature(100), WithSIMBAMaxSteps(8), WithLLMConcurrency(1), WithPipelineProcessing(pipeline))
			base := simbaScoreProgram(func(instruction string, _ int) float64 {
				return map[string]float64{"BASE": 0.2, "BRANCH": 0.7, "DEAD-A": 0.8, "DEAD-B": 0.9, "GOOD": 1}[instruction]
			})
			_, err := s.CompileWithScorer(context.Background(), base, simbaExamples(3), core.MetricScorer(simbaTableMetric))
			require.NoError(t, err)
			assert.Equal(t, 1.0, s.GetState().BestScore, "revisit a scored proposal that did not become the incumbent")
			assert.Contains(t, parents[3:], "BRANCH")
			assert.Contains(t, parents[3:], "BASE", "the baseline remains eligible after better programs exist")
		})
	}
}

func TestSIMBAPoolSamplingUsesScores(t *testing.T) {
	var parents []string
	s := newSIMBAScript(func(prompt string) string {
		if !strings.HasPrefix(prompt, "Generate a variation") {
			return "Continue."
		}
		_, parent, _ := strings.Cut(prompt, "Original: ")
		parent, _, _ = strings.Cut(parent, "\n")
		parents = append(parents, parent)
		if len(parents) == 1 {
			return "HIGH"
		}
		if len(parents) == 2 {
			return "LOW"
		}
		return "DUD"
	}, WithSIMBAStrategyMode("instruction_only"), WithSIMBANumCandidates(4), WithSIMBAMaxSteps(64), WithLLMConcurrency(1))
	base := simbaScoreProgram(func(instruction string, _ int) float64 {
		return map[string]float64{"BASE": 0.05, "HIGH": 0.9, "LOW": 0.1, "DUD": 0.01}[instruction]
	})
	_, err := s.CompileWithScorer(context.Background(), base, simbaExamples(1), core.MetricScorer(simbaTableMetric))
	require.NoError(t, err)
	high := 0
	for _, parent := range parents[3:] {
		if parent == "HIGH" {
			high++
		}
	}
	assert.Greater(t, float64(high)/float64(len(parents)-3), 0.9, "uniform random parent selection must fail this scored-pool contract")
}

func TestSIMBARulesUseCurrentMinibatch(t *testing.T) {
	for _, pipeline := range []bool{false, true} {
		t.Run(fmt.Sprintf("pipeline=%t", pipeline), func(t *testing.T) {
			lastID, analyses := -1, 0
			s := newSIMBAScript(func(prompt string) string {
				switch {
				case strings.HasPrefix(prompt, "Analyze the following successful"):
					analyses++
					if strings.Contains(prompt, fmt.Sprintf("Input: map[id:%d]", lastID)) &&
						strings.Contains(prompt, fmt.Sprintf("Expected output: map[answer:label-%d]", lastID)) {
						return "RULE: Use current evidence."
					}
				case strings.HasPrefix(prompt, "Given the following instruction and rule"):
					return "GOOD"
				}
				return "BASE"
			}, WithSIMBAStrategyMode("rule_only"), WithSIMBANumCandidates(2), WithSIMBAMaxSteps(1),
				WithSIMBABatchSize(1), WithSamplingTemperature(0), WithLLMConcurrency(1), WithPipelineProcessing(pipeline))
			base := simbaScoreProgram(func(instruction string, id int) float64 {
				lastID = id
				if instruction == "GOOD" {
					return 1
				}
				return 0.6
			})
			examples := simbaExamples(8)
			for i := range examples {
				examples[i].Outputs["answer"] = fmt.Sprintf("label-%d", i)
			}
			_, err := s.CompileWithScorer(context.Background(), base, examples, core.MetricScorer(simbaTableMetric))
			require.NoError(t, err)
			assert.Equal(t, 1, analyses, "analyze the current minibatch even in the first step")
			assert.Equal(t, 1.0, s.GetState().BestScore)
		})
	}
}

func TestSIMBARulesCompareSameExampleBuckets(t *testing.T) {
	for _, pipeline := range []bool{false, true} {
		t.Run(fmt.Sprintf("pipeline=%t", pipeline), func(t *testing.T) {
			var analyses []string
			variations := 0
			s := newSIMBAScript(func(prompt string) string {
				switch {
				case strings.HasPrefix(prompt, "Generate a variation"):
					variations++
					if variations == 1 {
						return "A"
					}
					return "B"
				case strings.HasPrefix(prompt, "Analyze the following successful"):
					analyses = append(analyses, prompt)
					for _, id := range []int{0, 1} {
						if strings.Count(prompt, fmt.Sprintf("Input: map[id:%d]", id)) == 2 &&
							strings.Count(prompt, "Input: map[id:") == 2 &&
							strings.Contains(prompt, "Expected output: map[answer:target]") {
							return "RULE: GOOD"
						}
					}
					return "No same-example comparison."
				case strings.HasPrefix(prompt, "Given the following instruction and rule"):
					if strings.Contains(prompt, "Rule to incorporate: GOOD") {
						return "GOOD"
					}
				}
				return "BASE"
			}, WithSIMBAStrategyMode("rule_only"), WithSIMBANumCandidates(3), WithSIMBAMaxSteps(2),
				WithSamplingTemperature(0), WithLLMConcurrency(1), WithPipelineProcessing(pipeline))
			base := simbaScoreProgram(func(instruction string, id int) float64 {
				if instruction == "GOOD" || (instruction == "A" && id == 0) || (instruction == "B" && id == 1) {
					return 1
				}
				if id == 2 {
					return 0.5 // No inter-program gap; this bucket must rank last.
				}
				return 0
			})
			_, err := s.CompileWithScorer(context.Background(), base, simbaExamples(3), core.MetricScorer(simbaTableMetric))
			require.NoError(t, err)
			assert.Equal(t, 1.0, s.GetState().BestScore)
			var contrasting []string
			for _, prompt := range analyses {
				if strings.Contains(prompt, "FAILED TRAJECTORIES:") {
					contrasting = append(contrasting, prompt)
				}
			}
			require.Len(t, contrasting, 2, "one analysis for each leading same-example bucket, not one pooled analysis")
			for i, prompt := range contrasting {
				assert.Equal(t, 2, strings.Count(prompt, fmt.Sprintf("Input: map[id:%d]", i)))
				assert.NotContains(t, prompt, "Input: map[id:2]")
			}
		})
	}
}
