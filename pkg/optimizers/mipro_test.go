package optimizers

import (
	"context"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestNewMIPRO(t *testing.T) {
	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 { return 0 }
	mipro := NewMIPRO(metric,
		WithNumCandidates(20),
		WithMaxBootstrappedDemos(10),
		WithMaxLabeledDemos(10),
		WithNumTrials(200),
		WithMiniBatchSize(64),
		WithFullEvalSteps(20),
		WithVerbose(true),
	)

	assert.Equal(t, 20, mipro.NumCandidates)
	assert.Equal(t, 10, mipro.MaxBootstrappedDemos)
	assert.Equal(t, 10, mipro.MaxLabeledDemos)
	assert.Equal(t, 200, mipro.NumTrials)
	assert.Equal(t, 64, mipro.MiniBatchSize)
	assert.Equal(t, 20, mipro.FullEvalSteps)
	assert.True(t, mipro.Verbose)
}

func TestMIPRO_Compile(t *testing.T) {
	ctx := core.WithExecutionState(context.Background())
	mockLLM := new(testutil.MockLLM)
	mockDataset := &testutil.MockDataset{
		Examples: []core.Example{
			{Inputs: map[string]interface{}{"input": "test1"}, Outputs: map[string]interface{}{"output": "result1"}},
			{Inputs: map[string]interface{}{"input": "test2"}, Outputs: map[string]interface{}{"output": "result2"}},
		},
	}
	// Set up expectations for the dataset
	mockDataset.On("Next").Return(core.Example{}, true).Times(len(mockDataset.Examples) * 2) // Allow for 2 trials
	mockDataset.On("Next").Return(core.Example{}, false).Maybe()
	mockDataset.On("Reset").Return().Times(3) // Expect Reset to be called for each trial

	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("Instruction", nil)
	mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(map[string]interface{}{"output": "result"}, nil)

	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 { return 1.0 }
	metricWrapper := func(expected, actual map[string]interface{}) float64 {
		return metric(expected, actual, nil)
	}

	mipro := NewMIPRO(metric,
		WithNumCandidates(2),
		WithMaxBootstrappedDemos(2),
		WithMaxLabeledDemos(0),
		WithNumTrials(2),
		WithPromptModel(mockLLM),
		WithTaskModel(mockLLM),
		WithMiniBatchSize(2),
		WithVerbose(true),
	)

	program := core.NewProgram(
		map[string]core.Module{
			"predict": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return mockLLM.GenerateWithJSON(ctx, inputs["input"].(string))
		},
	)

	compiledProgram, err := mipro.Compile(ctx, program, mockDataset, metricWrapper)

	assert.NoError(t, err)
	assert.NotNil(t, compiledProgram)

	mockLLM.AssertExpectations(t)
	mockDataset.AssertExpectations(t)

}
func TestMIPRO_CompileErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("Error in generateInstructionCandidates", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		mockDataset := &testutil.MockDataset{
			Examples: []core.Example{
				{Inputs: map[string]interface{}{"input": "test1"}, Outputs: map[string]interface{}{"output": "result1"}},
			},
		}

		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("", errors.New("generate error")).Once()
		mockDataset.On("Reset").Return().Maybe()

		mipro := setupMIPRO(mockLLM)
		program := setupProgram(mockLLM)

		_, err := mipro.Compile(ctx, program, mockDataset, metricWrapper)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to generate instruction candidates")

		mockLLM.AssertExpectations(t)
		mockDataset.AssertExpectations(t)
	})

	t.Run("Error in evaluateProgram", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		mockDataset := &testutil.MockDataset{
			Examples: []core.Example{
				{Inputs: map[string]interface{}{"input": "test1"}, Outputs: map[string]interface{}{"output": "result1"}},
			},
		}

		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("Instruction", nil)
		mockLLM.On("GenerateWithJSON", mock.Anything, mock.Anything, mock.Anything).Return(nil, errors.New("execution error")).Once()
		mockDataset.On("Next").Return(mockDataset.Examples[0], true).Once()
		mockDataset.On("Next").Return(core.Example{}, false)
		mockDataset.On("Reset").Return().Maybe()

		mipro := setupMIPRO(mockLLM)
		program := setupProgram(mockLLM)

		_, err := mipro.Compile(ctx, program, mockDataset, metricWrapper)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to evaluate program")

		mockLLM.AssertExpectations(t)
		mockDataset.AssertExpectations(t)
	})

	t.Run("No examples evaluated", func(t *testing.T) {
		mockLLM := new(testutil.MockLLM)
		emptyDataset := &testutil.MockDataset{Examples: []core.Example{}}

		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("Instruction", nil)
		emptyDataset.On("Next").Return(core.Example{}, false)
		emptyDataset.On("Reset").Return().Maybe()

		mipro := setupMIPRO(mockLLM)
		program := setupProgram(mockLLM)

		_, err := mipro.Compile(ctx, program, emptyDataset, metricWrapper)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "no examples")

		mockLLM.AssertExpectations(t)
		emptyDataset.AssertExpectations(t)
	})
}

func setupMIPRO(mockLLM *testutil.MockLLM) *MIPRO {
	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 { return 1.0 }
	return NewMIPRO(metric,
		WithNumCandidates(1),
		WithMaxBootstrappedDemos(1),
		WithMaxLabeledDemos(0),
		WithNumTrials(1),
		WithPromptModel(mockLLM),
		WithTaskModel(mockLLM),
		WithMiniBatchSize(1),
	)
}

func setupProgram(mockLLM *testutil.MockLLM) core.Program {
	return core.NewProgram(
		map[string]core.Module{
			"predict": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return mockLLM.GenerateWithJSON(ctx, inputs["input"].(string))
		},
	)
}

func metricWrapper(expected, actual map[string]interface{}) float64 {
	return 1.0
}

func TestMIPRO_generateTrial(t *testing.T) {
	mipro := NewMIPRO(nil)
	modules := []core.Module{
		modules.NewPredict(core.Signature{}),
		modules.NewPredict(core.Signature{}),
	}

	trial := mipro.generateTrial(modules, 5, 5)
	assert.Len(t, trial.Params, 4)
	assert.Contains(t, trial.Params, "instruction_0")
	assert.Contains(t, trial.Params, "instruction_1")
	assert.Contains(t, trial.Params, "demo_0")
	assert.Contains(t, trial.Params, "demo_1")
}

func TestMIPRO_constructProgram(t *testing.T) {
	mipro := NewMIPRO(nil)
	baseProgram := core.NewProgram(
		map[string]core.Module{
			"predict1": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
			"predict2": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
		},
		nil,
	)
	trial := Trial{
		Params: map[string]int{
			"instruction_0": 0,
			"demo_0":        0,
			"instruction_1": 1,
			"demo_1":        1,
		},
	}
	instructionCandidates := [][]string{
		{"Instruction1", "Instruction2"},
		{"Instruction3", "Instruction4"},
	}
	demoCandidates := [][][]core.Example{
		{
			{
				{Inputs: map[string]interface{}{"input": "test1"}, Outputs: map[string]interface{}{"output": "result1"}},
			},
			{
				{Inputs: map[string]interface{}{"input": "test2"}, Outputs: map[string]interface{}{"output": "result2"}},
			},
		},
		{
			{
				{Inputs: map[string]interface{}{"input": "test3"}, Outputs: map[string]interface{}{"output": "result3"}},
			},
			{
				{Inputs: map[string]interface{}{"input": "test4"}, Outputs: map[string]interface{}{"output": "result4"}},
			},
		},
	}
	constructedProgram := mipro.constructProgram(baseProgram, trial, instructionCandidates, demoCandidates)
	assert.NotNil(t, constructedProgram)

	cm := constructedProgram.GetModules()
	assert.Equal(t, 2, len(cm), "Expected 2 modules")

	checkModule := func(i int, expectedInstruction string, expectedDemos []core.Example) {
		t.Logf("Checking module %d", i)
		if predictor, ok := cm[i].(*modules.Predict); ok {
			t.Logf("Module %d instruction: %s", i, predictor.GetSignature().Instruction)
			t.Logf("Module %d demos: %+v", i, predictor.GetDemos())
			assert.Equal(t, expectedInstruction, predictor.GetSignature().Instruction, "Unexpected instruction for module %d", i)
			assert.Equal(t, expectedDemos, predictor.GetDemos(), "Unexpected demos for module %d", i)
		} else {
			t.Errorf("Module %d is not a *modules.Predict", i)
		}
	}

	checkModule(0, "Instruction1", demoCandidates[0][0])
	checkModule(1, "Instruction4", demoCandidates[1][1])
}

func TestMIPRO_logTrialResult(t *testing.T) {
	mipro := NewMIPRO(nil, WithVerbose(true))
	trial := Trial{
		Params: map[string]int{"instruction_0": 0, "demo_0": 0},
		Score:  0.75,
	}

	// Capture stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	mipro.logTrialResult(1, trial)

	w.Close()
	out, _ := io.ReadAll(r)
	os.Stdout = oldStdout

	assert.Contains(t, string(out), "Trial 1: Score 0.7500")
}

func TestMIPRO_generateInstructionCandidates(t *testing.T) {
	ctx := context.Background()
	mockLLM := new(testutil.MockLLM)
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return("Instruction", nil)

	mipro := NewMIPRO(nil, WithPromptModel(mockLLM), WithNumCandidates(2))
	program := core.NewProgram(
		map[string]core.Module{
			"predict": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
		},
		nil,
	)

	candidates, err := mipro.generateInstructionCandidates(ctx, program, nil)

	assert.NoError(t, err)
	assert.Len(t, candidates, 1)
	assert.Len(t, candidates[0], 2)
	assert.Equal(t, "Instruction", candidates[0][0])
	assert.Equal(t, "Instruction", candidates[0][1])

	mockLLM.AssertExpectations(t)
}

func TestMIPRO_generateDemoCandidates(t *testing.T) {
	mockDataset := &testutil.MockDataset{
		Examples: []core.Example{
			{Inputs: map[string]interface{}{"input": "test1"}, Outputs: map[string]interface{}{"output": "result1"}},
			{Inputs: map[string]interface{}{"input": "test2"}, Outputs: map[string]interface{}{"output": "result2"}},
		},
	}
	mockDataset.On("Next").Return(mockDataset.Examples[0], true).Once()
	mockDataset.On("Next").Return(mockDataset.Examples[1], true).Once()
	mockDataset.On("Next").Return(core.Example{}, false)
	mockDataset.On("Reset").Return().Maybe()
	mipro := NewMIPRO(nil, WithNumCandidates(2), WithMaxBootstrappedDemos(1), WithMaxLabeledDemos(0))
	program := core.NewProgram(
		map[string]core.Module{
			"predict": modules.NewPredict(core.Signature{
				Inputs:  []core.InputField{{Field: core.Field{Name: "input"}}},
				Outputs: []core.OutputField{{Field: core.Field{Name: "output"}}},
			}),
		},
		nil,
	)

	candidates, err := mipro.generateDemoCandidates(program, mockDataset)

	assert.NoError(t, err)
	assert.Len(t, candidates, 1)
	assert.Len(t, candidates[0], 2)
	assert.Len(t, candidates[0][0], 1)
	assert.Len(t, candidates[0][1], 1)
	assert.Equal(t, mockDataset.Examples[0], candidates[0][0][0])
	assert.Equal(t, mockDataset.Examples[1], candidates[0][1][0])
}
