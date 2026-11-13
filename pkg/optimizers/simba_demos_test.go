package optimizers

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type simbaJSONModel struct{ *simbaScriptModel }

func (m *simbaJSONModel) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityJSON}
}

func (m *simbaJSONModel) GenerateWithJSON(ctx context.Context, _ string, _ ...core.GenerateOption) (map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return map[string]any{"answer": "target", "rationale": "checked", "reasoning": "checked", "provider_metadata": "not a demo field"}, nil
}

func TestSIMBAStructuredDemosHonorLimitsAndSuccessThreshold(t *testing.T) {
	for _, kind := range []string{"predict-json", "cot-json", "cot-text-fallback"} {
		for _, tc := range []struct {
			cap       int
			threshold float64
			reward    float64
			learns    bool
		}{
			{2, 1, 1, true},
			{0, 1, 1, false},
			{2, 1, 0.75, false}, // Partial credit is not a certified demonstration.
			{1, 0.75, 0.75, true},
		} {
			t.Run(fmt.Sprintf("%s/cap=%d/threshold=%g/reward=%g", kind, tc.cap, tc.threshold, tc.reward), func(t *testing.T) {
				s := newSIMBAScript(func(string) string { return "rationale: checked\nanswer: target" },
					WithSIMBAStrategyMode("demo_only"), WithSIMBANumCandidates(2), WithSIMBAMaxSteps(1),
					WithSamplingTemperature(0), WithSIMBAMaxDemos(tc.cap), WithSIMBADemoScoreThreshold(tc.threshold))
				sig := core.NewSignature([]core.InputField{{Name: "question"}}, []core.OutputField{{Name: "answer", Prefix: "answer:"}})
				var module core.Module
				var predictor *modules.Predict
				if kind == "predict-json" {
					predictor = modules.NewPredict(sig).WithStructuredOutput()
					module = predictor
				} else {
					cot := modules.NewChainOfThought(sig).WithStructuredOutput()
					module, predictor = cot, cot.Predict
				}
				if kind == "cot-text-fallback" {
					module.SetLLM(s.primaryModel.(core.LLM))
				} else {
					module.SetLLM(&simbaJSONModel{s.primaryModel.(*simbaScriptModel)})
				}
				for range 4 {
					predictor.Demos = append(predictor.Demos, core.Example{Inputs: map[string]any{"question": "old"}, Outputs: map[string]any{"answer": "old"}})
				}
				base := core.NewProgramWithForwardFactory(map[string]core.Module{"task": module},
					func(ms map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
						return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
							answer, err := ms["task"].Process(ctx, map[string]any{"question": inputs["id"]})
							if err != nil {
								return nil, err
							}
							learned := false
							for _, demo := range ms["task"].(core.DemoProvider).GetDemos() {
								learned = learned || demo.Inputs["question"] == 0
							}
							if answer["answer"] == "target" && (inputs["id"] == 0 || learned) {
								return map[string]any{"score": tc.reward}, nil
							}
							return map[string]any{"score": 0.0}, nil
						}
					})
				result, err := s.CompileWithScorer(context.Background(), base, simbaExamples(2), core.MetricScorer(simbaTableMetric))
				require.NoError(t, err)
				demos := result.Modules["task"].(core.DemoProvider).GetDemos()
				if tc.learns {
					assert.Equal(t, tc.reward, s.GetState().BestScore)
					require.Len(t, demos, tc.cap)
					demo := demos[len(demos)-1]
					assert.Equal(t, map[string]any{"question": 0}, demo.Inputs)
					wantOutputs := map[string]any{"answer": "target"}
					if kind != "predict-json" {
						wantOutputs["rationale"] = "checked"
					}
					assert.Equal(t, wantOutputs, demo.Outputs, "record parsed predictor outputs, not raw JSON or provider metadata")
				} else {
					assert.Equal(t, tc.reward/2, s.GetState().BestScore)
					assert.Equal(t, predictor.Demos, demos)
				}
				assert.Len(t, predictor.Demos, 4, "preserve the input program")
			})
		}
	}
}
