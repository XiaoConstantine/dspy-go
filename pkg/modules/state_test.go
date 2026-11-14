package modules

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stateTestSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	).WithInstruction("original")
}

// loadStateForTest loads path into program and returns the load warnings.
func loadStateForTest(t *testing.T, program *core.Program, path string) []string {
	t.Helper()
	var warnings []string
	require.NoError(t, core.LoadProgramWithOptions(program, path, core.LoadProgramOptions{
		WarningHandler: func(message string) { warnings = append(warnings, message) },
	}))
	return warnings
}

// TestSaveLoadProgramRestoresEveryPredictor optimizes each predictor in a
// module differently, then checks that a freshly built module gets back each
// predictor's own instruction, output prefix, and demos.
func TestSaveLoadProgramRestoresEveryPredictor(t *testing.T) {
	cases := []struct {
		name string
		// build returns a new module and, in a fixed order, the predictors in it.
		build func() (core.Module, []*Predict)
	}{
		{"Predict", func() (core.Module, []*Predict) {
			p := NewPredict(stateTestSignature())
			return p, []*Predict{p}
		}},
		{"ChainOfThought", func() (core.Module, []*Predict) {
			c := NewChainOfThought(stateTestSignature())
			return c, []*Predict{c.Predict}
		}},
		{"ReAct", func() (core.Module, []*Predict) {
			r := NewReAct(stateTestSignature(), tools.NewInMemoryToolRegistry(), 3)
			return r, []*Predict{r.Predict, r.Extract.Predict}
		}},
		{"Refine of ChainOfThought", func() (core.Module, []*Predict) {
			c := NewChainOfThought(stateTestSignature())
			return NewRefine(c, RefineConfig{}), []*Predict{c.Predict}
		}},
		{"MultiChainComparison", func() (core.Module, []*Predict) {
			m := NewMultiChainComparison(stateTestSignature(), 2, 0)
			return m, []*Predict{m.predict}
		}},
		{"Parallel of ReAct", func() (core.Module, []*Predict) {
			r := NewReAct(stateTestSignature(), tools.NewInMemoryToolRegistry(), 3)
			return NewParallel(r), []*Predict{r.Predict, r.Extract.Predict}
		}},
		{"ModuleChain", func() (core.Module, []*Predict) {
			first := NewPredict(stateTestSignature())
			second := NewChainOfThought(stateTestSignature())
			return core.NewModuleChain(first, second), []*Predict{first, second.Predict}
		}},
		{"OfferFeedback", func() (core.Module, []*Predict) {
			o := NewOfferFeedback()
			return o, []*Predict{o.predict}
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			module, predictors := tc.build()
			for i, predictor := range predictors {
				signature := predictor.GetSignature().WithInstruction(fmt.Sprintf("optimized %d", i))
				signature.Outputs[len(signature.Outputs)-1].Prefix = fmt.Sprintf("Final %d:", i)
				predictor.SetSignature(signature)
				predictor.SetDemos([]core.Example{{
					Inputs:  map[string]any{"question": fmt.Sprintf("q%d", i)},
					Outputs: map[string]any{"answer": fmt.Sprintf("a%d", i)},
				}})
			}
			program := core.NewProgram(map[string]core.Module{"m": module}, nil)
			path := filepath.Join(t.TempDir(), "program.json")
			require.NoError(t, core.SaveProgram(&program, path))

			freshModule, freshPredictors := tc.build()
			fresh := core.NewProgram(map[string]core.Module{"m": freshModule}, nil)
			assert.Empty(t, loadStateForTest(t, &fresh, path))

			for i, predictor := range freshPredictors {
				signature := predictor.GetSignature()
				assert.Equal(t, fmt.Sprintf("optimized %d", i), signature.Instruction, "predictor %d instruction", i)
				assert.Equal(t, fmt.Sprintf("Final %d:", i), signature.Outputs[len(signature.Outputs)-1].Prefix, "predictor %d prefix", i)
				assert.Equal(t, []core.Example{{
					Inputs:  map[string]any{"question": fmt.Sprintf("q%d", i)},
					Outputs: map[string]any{"answer": fmt.Sprintf("a%d", i)},
				}}, predictor.GetDemos(), "predictor %d demos", i)
			}
		})
	}
}

// TestLoadProgramReadsFilesWithoutInstructions loads a file in the format
// saved before instructions and nested predictors were saved: demos load and
// instructions stay as constructed, without warnings.
func TestLoadProgramReadsFilesWithoutInstructions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "program.json")
	require.NoError(t, os.WriteFile(path, []byte(`{
  "modules": {
    "cot": {
      "signature": "Inputs:\n  - question ()\nOutputs:\n  - answer ()\nInstruction: saved\n",
      "demos": [{"inputs": {"question": "q"}, "outputs": {"answer": "a"}}],
      "module_type": "ChainOfThought"
    },
    "predict": {
      "signature": "Inputs:\n  - question ()\nOutputs:\n  - answer ()\nInstruction: saved\n",
      "demos": [{"inputs": {"question": "q2"}, "outputs": {"answer": "a2"}}],
      "module_type": "Predict"
    }
  },
  "metadata": {"dspy_go_version": "`+core.Version+`"}
}`), 0o644))

	cot := NewChainOfThought(stateTestSignature())
	predict := NewPredict(stateTestSignature())
	program := core.NewProgram(map[string]core.Module{"cot": cot, "predict": predict}, nil)
	assert.Empty(t, loadStateForTest(t, &program, path))

	assert.Equal(t, "original", cot.Predict.GetSignature().Instruction)
	assert.Equal(t, "original", predict.GetSignature().Instruction)
	assert.Equal(t, []core.Example{{Inputs: map[string]any{"question": "q"}, Outputs: map[string]any{"answer": "a"}}}, cot.GetDemos())
	assert.Equal(t, []core.Example{{Inputs: map[string]any{"question": "q2"}, Outputs: map[string]any{"answer": "a2"}}}, predict.GetDemos())
}

// TestLoadProgramClearsDemosSavedEmpty checks that a predictor saved without
// demos loads without the demos its constructor gave it.
func TestLoadProgramClearsDemosSavedEmpty(t *testing.T) {
	program := core.NewProgram(map[string]core.Module{"m": NewChainOfThought(stateTestSignature())}, nil)
	path := filepath.Join(t.TempDir(), "program.json")
	require.NoError(t, core.SaveProgram(&program, path))

	fresh := NewChainOfThought(stateTestSignature())
	fresh.SetDemos([]core.Example{{Inputs: map[string]any{"question": "built-in"}, Outputs: map[string]any{"answer": "demo"}}})
	freshProgram := core.NewProgram(map[string]core.Module{"m": fresh}, nil)
	assert.Empty(t, loadStateForTest(t, &freshProgram, path))

	assert.Empty(t, fresh.Predict.GetDemos())
}
