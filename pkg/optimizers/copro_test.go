package optimizers

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockLLM for testing.
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) GetModelName() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLLM) Capabilities() []core.Capability {
	args := m.Called()
	if len(args) > 0 {
		return args.Get(0).([]core.Capability)
	}
	return []core.Capability{core.CapabilityCompletion}
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]any), args.Error(1)
}

func (m *MockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	args := m.Called(ctx, prompt, functions, options)
	return args.Get(0).(map[string]any), args.Error(1)
}

func (m *MockLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	args := m.Called(ctx, input, options)
	return args.Get(0).(*core.EmbeddingResult), args.Error(1)
}

func (m *MockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	args := m.Called(ctx, inputs, options)
	return args.Get(0).(*core.BatchEmbeddingResult), args.Error(1)
}

func (m *MockLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(*core.StreamResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, content, options)
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	args := m.Called(ctx, content, options)
	return args.Get(0).(*core.StreamResponse), args.Error(1)
}

func (m *MockLLM) ProviderName() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLLM) ModelID() string {
	args := m.Called()
	return args.String(0)
}

// MockModule is a mock implementation of core.Module for testing.
type MockModule struct {
	mock.Mock
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	args := m.Called(ctx, inputs)
	return args.Get(0).(map[string]any), args.Error(1)
}

func (m *MockModule) GetSignature() core.Signature {
	args := m.Called()
	return args.Get(0).(core.Signature)
}

func (m *MockModule) SetLLM(llm core.LLM) {
	m.Called(llm)
}

func (m *MockModule) Clone() core.Module {
	args := m.Called()
	return args.Get(0).(core.Module)
}

func (m *MockModule) SetSignature(signature core.Signature) {
	m.Called(signature)
}

func (m *MockModule) GetDisplayName() string {
	return "MockModule"
}

func (m *MockModule) GetModuleType() string {
	return "test"
}

func TestNewCOPRO(t *testing.T) {
	tests := []struct {
		name     string
		metric   core.Metric
		options  []COPROOption
		expected *COPRO
	}{
		{
			name:    "Default options",
			metric:  func(expected, actual map[string]any) float64 { return 1.0 },
			options: nil,
			expected: &COPRO{
				Breadth:         5,   // Updated to match Python DSPy default
				Depth:           2,   // Updated to match Python DSPy default
				InitTemperature: 1.2, // Updated to match Python DSPy default
				TrackStats:      false,
			},
		},
		{
			name:   "Custom options",
			metric: func(expected, actual map[string]any) float64 { return 0.5 },
			options: []COPROOption{
				WithBreadth(5),
				WithDepth(2),
				WithInitTemperature(1.2),
				WithTrackStats(true),
			},
			expected: &COPRO{
				Breadth:         5,
				Depth:           2,
				InitTemperature: 1.2,
				TrackStats:      true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			copro := NewCOPRO(tt.metric, tt.options...)

			assert.NotNil(t, copro)
			assert.Equal(t, tt.expected.Breadth, copro.Breadth)
			assert.Equal(t, tt.expected.Depth, copro.Depth)
			assert.Equal(t, tt.expected.InitTemperature, copro.InitTemperature)
			assert.Equal(t, tt.expected.TrackStats, copro.TrackStats)
			assert.NotNil(t, core.MetricScorer(copro.Metric))
		})
	}
}

func TestCOPROOptions(t *testing.T) {
	mockLLM := &MockLLM{}

	opts := &COPROOptions{}

	// Test WithPromptModel
	WithPromptModel(mockLLM)(opts)
	assert.Equal(t, mockLLM, opts.PromptModel)

	// Test WithBreadth
	WithBreadth(15)(opts)
	assert.Equal(t, 15, opts.Breadth)

	// Test WithDepth
	WithDepth(5)(opts)
	assert.Equal(t, 5, opts.Depth)

	// Test WithInitTemperature
	WithInitTemperature(2.0)(opts)
	assert.Equal(t, 2.0, opts.InitTemperature)

	// Test WithTrackStats
	WithTrackStats(true)(opts)
	assert.True(t, opts.TrackStats)
}

func createCOPROTestProgram() core.Program {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	return coproSinglePredictorProgram(modules.NewPredict(signature))
}

// coproSinglePredictorProgram wraps predictor as the module "predictor".
func coproSinglePredictorProgram(predictor core.Module) core.Program {
	return core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				return modules["predictor"].Process(ctx, inputs)
			}
		},
	)
}

func createCOPROTestDataset() *datasets.SimpleDataset {
	examples := []core.Example{
		{
			Inputs:  map[string]any{"question": "What is 2+2?"},
			Outputs: map[string]any{"answer": "4"},
		},
		{
			Inputs:  map[string]any{"question": "What is 3+3?"},
			Outputs: map[string]any{"answer": "6"},
		},
	}
	return datasets.NewSimpleDataset(examples)
}

func TestCOPROCompile_Success(t *testing.T) {
	// Create test program and dataset
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()

	// Create metric
	metric := func(expected, actual map[string]any) float64 {
		return 1.0 // Always return success for this test
	}

	// Create COPRO optimizer with small parameters for fast testing
	copro := NewCOPRO(metric, WithBreadth(2), WithDepth(1))

	// Mock LLM for the predictor
	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

	// Set LLM on the predictor
	predictor := program.Modules["predictor"].(*modules.Predict)
	predictor.SetLLM(mockLLM)

	ctx := core.WithExecutionState(context.Background())

	// Test compilation
	optimizedProgram, err := copro.Compile(ctx, program, dataset, metric)

	assert.NoError(t, err)
	assert.NotNil(t, optimizedProgram)
	assert.Equal(t, len(program.Modules), len(optimizedProgram.Modules))
}

func TestCOPROCompile_NoMetric(t *testing.T) {
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()

	copro := NewCOPRO(nil)
	ctx := core.WithExecutionState(context.Background())

	_, err := copro.Compile(ctx, program, dataset, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "COPRO requires a metric function")
}

func TestCOPROCompile_NoPredictModules(t *testing.T) {
	// Create program without Predict modules
	program := core.NewProgram(
		map[string]core.Module{},
		func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			return inputs, nil
		},
	)
	dataset := testutil.NewMockDataset([]core.Example{{Inputs: map[string]any{"input": "unused"}}})
	dataset.On("Reset").Return().Maybe()
	dataset.On("Next").Return().Maybe()

	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric)
	ctx := core.WithExecutionState(context.Background())

	optimizedProgram, err := copro.Compile(ctx, program, dataset, metric)

	require.ErrorContains(t, err, "no Predict module")
	assert.Equal(t, 0, len(optimizedProgram.Modules))
	dataset.AssertNotCalled(t, "Reset")
	dataset.AssertNotCalled(t, "Next")
}

func TestCOPROCompile_DoesNotConsumeCanceledDataset(t *testing.T) {
	dataset := testutil.NewMockDataset([]core.Example{{Inputs: map[string]any{"question": "unused"}}})
	dataset.On("Reset").Return().Maybe()
	dataset.On("Next").Return().Maybe()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1 })
	_, err := copro.Compile(ctx, createCOPROTestProgram(), dataset, nil)

	assert.ErrorIs(t, err, context.Canceled)
	dataset.AssertNotCalled(t, "Reset")
	dataset.AssertNotCalled(t, "Next")
}

func TestCOPROCompileExamples_PropagatesCancellationAndStopsPredictors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	).WithInstruction("Answer the question")

	firstLLM := &MockLLM{}
	firstLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
		Run(func(mock.Arguments) { cancel() }).
		Return(&core.LLMResponse{}, context.Canceled).
		Once()
	firstLLM.On("GetModelName").Return("first-model").Maybe()
	firstLLM.On("ModelID").Return("first-model").Maybe()
	firstLLM.On("ProviderName").Return("test-provider").Maybe()
	firstLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion}).Maybe()
	secondLLM := &MockLLM{}

	first := modules.NewPredict(signature)
	first.SetLLM(firstLLM)
	second := modules.NewPredict(signature)
	second.SetLLM(secondLLM)
	program := core.NewProgramWithForwardFactory(map[string]core.Module{
		"a_first":  first,
		"b_second": second,
	}, func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			return modules["a_first"].Process(ctx, inputs)
		}
	})

	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1 }, WithBreadth(1), WithDepth(0))
	_, err := copro.CompileExamples(ctx, program, []core.Example{{
		Inputs:  map[string]any{"question": "What is 2+2?"},
		Outputs: map[string]any{"answer": "4"},
	}}, nil)

	assert.ErrorIs(t, err, context.Canceled)
	firstLLM.AssertExpectations(t)
	secondLLM.AssertNotCalled(t, "Generate")
}

func TestCOPROCompileExamples_EmptyDataset(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1 })
	var compiler core.ExamplesCompiler = copro

	_, err := compiler.CompileExamples(context.Background(), createCOPROTestProgram(), nil, nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no examples in dataset for optimization")
}

func TestCOPROCompile_MaterializesLegacyDatasetOnce(t *testing.T) {
	firstSignature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("first_question")}},
		[]core.OutputField{{Field: core.NewField("first_answer")}},
	).WithInstruction("Answer the first question")
	secondSignature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("second_question")}},
		[]core.OutputField{{Field: core.NewField("second_answer")}},
	).WithInstruction("Answer the second question")

	var firstGeneratorCalls atomic.Int64
	var secondGeneratorCalls atomic.Int64
	newLLM := func(name string, generatorCalls *atomic.Int64) *MockLLM {
		llm := &MockLLM{}
		llm.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Run(func(args mock.Arguments) {
				if prompt, ok := args.Get(1).(string); ok && strings.Contains(prompt, "expert prompt engineer") {
					generatorCalls.Add(1)
				}
			}).
			Return(&core.LLMResponse{Content: "Use a concise and accurate answer"}, nil).
			Maybe()
		llm.On("GetModelName").Return(name).Maybe()
		llm.On("ModelID").Return(name).Maybe()
		llm.On("ProviderName").Return("test-provider").Maybe()
		llm.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion}).Maybe()
		return llm
	}
	firstLLM := newLLM("first-model", &firstGeneratorCalls)
	secondLLM := newLLM("second-model", &secondGeneratorCalls)

	first := modules.NewPredict(firstSignature)
	first.SetLLM(firstLLM)
	second := modules.NewPredict(secondSignature)
	second.SetLLM(secondLLM)
	program := core.NewProgramWithForwardFactory(map[string]core.Module{
		"first":  first,
		"second": second,
	}, func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			outputs, err := modules["first"].Process(ctx, inputs)
			if err != nil {
				return nil, err
			}
			secondOutputs, err := modules["second"].Process(ctx, inputs)
			if err != nil {
				return nil, err
			}
			maps.Copy(outputs, secondOutputs)
			return outputs, nil
		}
	})

	examples := []core.Example{
		{
			Inputs:  map[string]any{"first_question": "What is 2+2?", "second_question": "What is 3+3?"},
			Outputs: map[string]any{"first_answer": "4", "second_answer": "6"},
		},
		{
			Inputs:  map[string]any{"first_question": "What is 4+4?", "second_question": "What is 5+5?"},
			Outputs: map[string]any{"first_answer": "8", "second_answer": "10"},
		},
	}
	dataset := testutil.NewMockDataset(examples)
	dataset.On("Reset").Return().Once()
	dataset.On("Next").Return().Times(len(examples) + 1)

	var constructorMetricCalls atomic.Int64
	var invocationMetricCalls atomic.Int64
	copro := NewCOPRO(func(expected, actual map[string]any) float64 {
		constructorMetricCalls.Add(1)
		return -1
	}, WithBreadth(1), WithDepth(0))
	_, err := copro.Compile(context.Background(), program, dataset, func(expected, actual map[string]any) float64 {
		invocationMetricCalls.Add(1)
		return 1
	})

	require.NoError(t, err)
	dataset.AssertExpectations(t)
	assert.Zero(t, constructorMetricCalls.Load())
	assert.Positive(t, invocationMetricCalls.Load())
	assert.Positive(t, firstGeneratorCalls.Load())
	assert.Positive(t, secondGeneratorCalls.Load())
}

func TestCOPROCompile_EmptyDataset(t *testing.T) {
	program := createCOPROTestProgram()
	emptyDataset := datasets.NewSimpleDataset([]core.Example{})

	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric, WithBreadth(2), WithDepth(1))

	mockLLM := &MockLLM{}
	predictor := program.Modules["predictor"].(*modules.Predict)
	predictor.SetLLM(mockLLM)

	ctx := core.WithExecutionState(context.Background())

	_, err := copro.Compile(ctx, program, emptyDataset, metric)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no examples in dataset for optimization")
}

func TestExtractPredictors(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 })

	// Test with Predict module
	predictModule := modules.NewPredict(core.Signature{})
	program := core.NewProgram(
		map[string]core.Module{
			"predictor": predictModule,
			"other":     &MockModule{},
		},
		nil,
	)

	predictors := copro.extractPredictors(program)

	assert.Len(t, predictors, 1)
	assert.Contains(t, predictors, "predictor")
	assert.Equal(t, predictModule, predictors["predictor"])
}

func TestGenerateInitialCandidates(t *testing.T) {
	setupTestMockLLM(t) // Must be called before NewPredict to ensure it gets the mock LLM
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 }, WithBreadth(3))
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()

	candidates := copro.generateInitialCandidates(ctx, predictor, "base instruction", copro.newPromptState(ctx, predictor))

	assert.Len(t, candidates, 3)
	for _, candidate := range candidates {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 1, candidate.Generation)
	}
}

func TestRefineCandidates(t *testing.T) {
	setupTestMockLLM(t) // Must be called before NewPredict to ensure it gets the mock LLM
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 }, WithBreadth(4))
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()

	topCandidates := []PromptCandidate{
		{Instruction: "Test instruction 1", Score: 0.9, Generation: 1},
		{Instruction: "Test instruction 2", Score: 0.8, Generation: 1},
	}

	refined := copro.refineCandidates(ctx, predictor, topCandidates, 2, copro.newPromptState(ctx, predictor))

	assert.Greater(t, len(refined), 0)
	for _, candidate := range refined {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 3, candidate.Generation) // depth + 1
	}
}

func TestRefineCandidates_DeterministicFallbackStillGeneratesCandidates(t *testing.T) {
	setupTestMockLLM(t)
	copro := NewCOPRO(
		func(expected, actual map[string]any) float64 { return 1.0 },
		WithBreadth(4),
		WithInitTemperature(0),
	)
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()

	topCandidates := []PromptCandidate{
		{Instruction: "Test instruction 1", Score: 0.9, Generation: 1},
		{Instruction: "Test instruction 2", Score: 0.8, Generation: 1},
	}

	refined := copro.refineCandidates(ctx, predictor, topCandidates, 2, copro.newPromptState(ctx, predictor))

	require.NotEmpty(t, refined)
	for _, candidate := range refined {
		assert.NotEmpty(t, candidate.Instruction)
		assert.NotEqual(t, "Test instruction 1", candidate.Instruction)
		assert.NotEqual(t, "Test instruction 2", candidate.Instruction)
	}
}

func TestEvaluateCandidate(t *testing.T) {
	metric := func(expected, actual map[string]any) float64 {
		if expected["answer"] == actual["answer"] {
			return 1.0
		}
		return 0.0
	}

	copro := NewCOPRO(metric)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	candidate := PromptCandidate{
		Instruction: "Answer the question",
		Score:       0.0,
		Generation:  1,
	}

	examples := []core.Example{
		{
			Inputs:  map[string]any{"question": "What is 2+2?"},
			Outputs: map[string]any{"answer": "4"},
		},
	}

	ctx := context.Background()
	score, err := copro.evaluateCandidate(ctx, coproSinglePredictorProgram(predictor), "predictor", candidate, examples, core.MetricScorer(copro.Metric))

	require.NoError(t, err)
	assert.GreaterOrEqual(t, score, 0.0)
	assert.LessOrEqual(t, score, 1.0)
}

func TestEvaluateCandidate_LLMError(t *testing.T) {
	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{}, fmt.Errorf("LLM error"))
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("ModelID").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	candidate := PromptCandidate{Instruction: "Test", Score: 0.0, Generation: 1}
	examples := []core.Example{{
		Inputs:  map[string]any{"question": "test"},
		Outputs: map[string]any{"answer": "test"},
	}}

	ctx := context.Background()
	score, err := copro.evaluateCandidate(ctx, coproSinglePredictorProgram(predictor), "predictor", candidate, examples, core.MetricScorer(copro.Metric))

	require.NoError(t, err)
	assert.Equal(t, 0.0, score) // Should return 0 when no valid evaluations
}

func TestEvaluateCandidate_ProviderTimeoutIsInvalidScore(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1 })
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{}, context.DeadlineExceeded)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("ModelID").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	score, err := copro.evaluateCandidate(
		context.Background(),
		coproSinglePredictorProgram(predictor),
		"predictor",
		PromptCandidate{Instruction: "Test"},
		[]core.Example{{Inputs: map[string]any{"question": "test"}, Outputs: map[string]any{"answer": "test"}}},
		core.MetricScorer(copro.Metric),
	)

	require.NoError(t, err)
	assert.Zero(t, score)
}

func TestEvaluateCandidatesParallel(t *testing.T) {
	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	candidates := []PromptCandidate{
		{Instruction: "Candidate 1", Score: 0.0, Generation: 1},
		{Instruction: "Candidate 2", Score: 0.0, Generation: 1},
		{Instruction: "Candidate 3", Score: 0.0, Generation: 1},
	}

	examples := []core.Example{
		{
			Inputs:  map[string]any{"question": "What is 2+2?"},
			Outputs: map[string]any{"answer": "4"},
		},
	}

	ctx := context.Background()
	require.NoError(t, copro.evaluateCandidatesParallel(ctx, coproSinglePredictorProgram(predictor), "predictor", candidates, examples, core.MetricScorer(copro.Metric)))

	// All candidates should have been evaluated (score set)
	for _, candidate := range candidates {
		assert.GreaterOrEqual(t, candidate.Score, 0.0)
	}
}

func TestDatasetToExamples(t *testing.T) {
	dataset := createCOPROTestDataset()

	examples := core.DatasetToSlice(dataset)

	assert.Len(t, examples, 2)
	assert.Equal(t, "What is 2+2?", examples[0].Inputs["question"])
	assert.Equal(t, "4", examples[0].Outputs["answer"])
}

func TestGetEnhancedInstructionTemplates(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 })
	signature := core.Signature{}

	templates := copro.getEnhancedInstructionTemplates(signature, "test task")

	assert.Greater(t, len(templates), 0)
	for _, template := range templates {
		assert.NotEmpty(t, template)
	}
}

func TestRefineInstruction(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 })

	tests := []struct {
		name        string
		instruction string
		temperature float64
	}{
		{
			name:        "High temperature",
			instruction: "Answer the question.",
			temperature: 0.8,
		},
		{
			name:        "Low temperature",
			instruction: "Answer the question",
			temperature: 0.3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := copro.refineInstruction(tt.instruction, tt.temperature)
			assert.NotEmpty(t, result)
		})
	}
}

func TestApplyPromptToPredictor(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 })

	originalSignature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	originalSignature = originalSignature.WithInstruction("Original instruction")

	predictor := modules.NewPredict(originalSignature)
	candidate := PromptCandidate{
		Instruction: "New instruction",
		Prefix:      "",
		Score:       0.9,
		Generation:  1,
	}

	copro.applyPromptToPredictor(predictor, candidate)

	// The predictor should have the new instruction
	newSignature := predictor.GetSignature()
	assert.Equal(t, "New instruction", newSignature.Instruction)
}

func TestHelperFunctions(t *testing.T) {
	// Test built-in min function
	assert.Equal(t, 3, min(3, 5))
	assert.Equal(t, 2, min(7, 2))
	assert.Equal(t, 4, min(4, 4))

	// Test built-in max function
	assert.Equal(t, 5, max(3, 5))
	assert.Equal(t, 7, max(7, 2))
	assert.Equal(t, 4, max(4, 4))
}

func TestCOPROWithExecutionState(t *testing.T) {
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()

	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric, WithBreadth(1), WithDepth(1))

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

	predictor := program.Modules["predictor"].(*modules.Predict)
	predictor.SetLLM(mockLLM)

	// Test without execution state (should be created automatically)
	ctx := context.Background()
	optimizedProgram, err := copro.Compile(ctx, program, dataset, metric)

	assert.NoError(t, err)
	assert.NotNil(t, optimizedProgram)
}

// Benchmark tests using shared benchmark utilities.

// BenchmarkCOPRO runs comprehensive benchmarks for COPRO optimizer.
func BenchmarkCOPRO(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	configs := testutil.StandardBenchmarkConfigs()

	testCases := []struct {
		name       string
		datasetKey string
		configKey  string
	}{
		{"Fast_Tiny", "tiny", "fast"},
		{"Fast_Small", "small", "fast"},
		{"Standard_Small", "small", "standard"},
		{"Standard_Medium", "medium", "standard"},
		{"Comprehensive_Medium", "medium", "comprehensive"},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			dataset := datasets[tc.datasetKey]
			config := configs[tc.configKey]

			// Create program
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)

			// Create COPRO optimizer with config
			copro := NewCOPRO(
				testutil.BenchmarkAccuracyMetric,
				WithBreadth(config.Breadth),
				WithDepth(config.Depth),
				WithInitTemperature(config.Temperature),
			)

			// Setup mock LLM
			mockLLM := &MockLLM{}
			mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
			mockLLM.On("GetModelName").Return("test-model")
			mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := core.WithExecutionState(context.Background())
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := copro.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				require.NoError(b, err)
			}
		})
	}
}

// BenchmarkCOPRODatasetScaling tests performance across different dataset sizes.
func BenchmarkCOPRODatasetScaling(b *testing.B) {
	datasets := testutil.CreateBenchmarkDatasets()
	config := testutil.StandardBenchmarkConfigs()["fast"] // Use fast config for scaling tests

	for name, dataset := range datasets {
		b.Run(fmt.Sprintf("Dataset_%s_%d", name, dataset.Size), func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			copro := NewCOPRO(
				testutil.BenchmarkAccuracyMetric,
				WithBreadth(config.Breadth),
				WithDepth(config.Depth),
			)

			mockLLM := &MockLLM{}
			mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
			mockLLM.On("GetModelName").Return("test-model")
			mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := core.WithExecutionState(context.Background())
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := copro.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				require.NoError(b, err)
			}
		})
	}
}

// BenchmarkCOPROParameterTuning tests performance with different parameter configurations.
func BenchmarkCOPROParameterTuning(b *testing.B) {
	dataset := testutil.CreateBenchmarkDatasets()["small"] // Use small dataset for parameter testing

	parameterTests := []struct {
		name        string
		breadth     int
		depth       int
		temperature float64
	}{
		{"Breadth3_Depth1", 3, 1, 1.0},
		{"Breadth5_Depth2", 5, 2, 1.2},
		{"Breadth8_Depth3", 8, 3, 1.5},
		{"HighTemp_Breadth5", 5, 2, 2.0},
		{"LowTemp_Breadth5", 5, 2, 0.5},
	}

	for _, pt := range parameterTests {
		b.Run(pt.name, func(b *testing.B) {
			signature := core.NewSignature(
				[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
				[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
			)
			predictor := modules.NewPredict(signature)
			program := testutil.CreateBenchmarkProgram(predictor)
			copro := NewCOPRO(
				testutil.BenchmarkAccuracyMetric,
				WithBreadth(pt.breadth),
				WithDepth(pt.depth),
				WithInitTemperature(pt.temperature),
			)

			mockLLM := &MockLLM{}
			mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
			mockLLM.On("GetModelName").Return("test-model")
			mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

			predictor = program.Modules["predictor"].(*modules.Predict)
			predictor.SetLLM(mockLLM)

			ctx := core.WithExecutionState(context.Background())
			benchDataset := testutil.BenchmarkDatasetFromExamples(dataset.Examples)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := copro.Compile(ctx, program, benchDataset, testutil.BenchmarkAccuracyMetric)
				require.NoError(b, err)
			}
		})
	}
}

// BenchmarkCOPROConcurrency tests parallel evaluation performance.
func BenchmarkCOPROConcurrency(b *testing.B) {
	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	concurrencyTests := []struct {
		name          string
		numCandidates int
		numExamples   int
	}{
		{"Candidates5_Examples10", 5, 10},
		{"Candidates10_Examples20", 10, 20},
		{"Candidates20_Examples50", 20, 50},
	}

	for _, ct := range concurrencyTests {
		b.Run(ct.name, func(b *testing.B) {
			candidates := make([]PromptCandidate, ct.numCandidates)
			for i := range candidates {
				candidates[i] = PromptCandidate{
					Instruction: fmt.Sprintf("Candidate %d", i),
					Score:       0.0,
					Generation:  1,
				}
			}

			examples := make([]core.Example, ct.numExamples)
			for i := range examples {
				examples[i] = core.Example{
					Inputs:  map[string]any{"question": fmt.Sprintf("What is %d+%d?", i, i)},
					Outputs: map[string]any{"answer": fmt.Sprintf("%d", i*2)},
				}
			}

			ctx := context.Background()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Reset scores
				for j := range candidates {
					candidates[j].Score = 0.0
				}
				if err := copro.evaluateCandidatesParallel(ctx, coproSinglePredictorProgram(predictor), "predictor", candidates, examples, core.MetricScorer(copro.Metric)); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkCOPROCompile is the original benchmark for backward compatibility.
func BenchmarkCOPROCompile(b *testing.B) {
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()

	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric, WithBreadth(2), WithDepth(1))

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

	predictor := program.Modules["predictor"].(*modules.Predict)
	predictor.SetLLM(mockLLM)

	ctx := core.WithExecutionState(context.Background())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := copro.Compile(ctx, program, dataset, metric)
		require.NoError(b, err)
	}
}

func TestCOPROCompile_EmptyInstruction(t *testing.T) {
	// Test with empty instruction to cover missing branch in optimizePredictor
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	// Don't set instruction - keep it empty
	predictor := modules.NewPredict(signature)

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				return modules["predictor"].Process(ctx, inputs)
			}
		},
	)

	dataset := createCOPROTestDataset()
	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric, WithBreadth(2), WithDepth(1))

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	ctx := core.WithExecutionState(context.Background())
	optimizedProgram, err := copro.Compile(ctx, program, dataset, metric)

	assert.NoError(t, err)
	assert.NotNil(t, optimizedProgram)
}

func TestGenerateInitialCandidates_MoreThanTemplates(t *testing.T) {
	setupTestMockLLM(t) // Must be called before NewPredict to ensure it gets the mock LLM
	// Test when breadth > number of templates to cover missing branch
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 }, WithBreadth(20)) // Large breadth
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()

	candidates := copro.generateInitialCandidates(ctx, predictor, "base instruction", copro.newPromptState(ctx, predictor))

	// Should generate candidates up to breadth (may be less if LLM generation fails in tests)
	assert.GreaterOrEqual(t, len(candidates), 5, "Should generate at least 5 candidates")
	assert.LessOrEqual(t, len(candidates), 20, "Should not exceed requested breadth")
	for _, candidate := range candidates {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 1, candidate.Generation)
	}
}

func TestOptimizePredictor_EmptyDataset(t *testing.T) {
	setupTestMockLLM(t) // Must be called before NewPredict to ensure it gets the mock LLM
	// Test direct call to optimizePredictor with empty dataset
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 1.0 })
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()
	err := copro.optimizePredictors(ctx, coproSinglePredictorProgram(predictor), []string{"predictor"}, nil, core.MetricScorer(copro.Metric))

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no examples in dataset for optimization")
}

func TestOptimizePredictor_WithCurrentInstruction(t *testing.T) {
	// Test optimizePredictor with current instruction to test baseline branch
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 0.9 })

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	signature = signature.WithInstruction("Existing instruction")
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	dataset := createCOPROTestDataset()
	ctx := context.Background()

	err := copro.optimizePredictors(ctx, coproSinglePredictorProgram(predictor), []string{"predictor"}, core.MaterializeDataset(dataset), core.MetricScorer(copro.Metric))

	assert.NoError(t, err)
}

func TestOptimizePredictor_NoCurrentInstruction(t *testing.T) {
	// Test optimizePredictor without current instruction to test empty instruction branch
	copro := NewCOPRO(func(expected, actual map[string]any) float64 { return 0.9 }, WithBreadth(2), WithDepth(1))

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	// Don't set instruction - it should be empty
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	dataset := createCOPROTestDataset()
	ctx := context.Background()

	err := copro.optimizePredictors(ctx, coproSinglePredictorProgram(predictor), []string{"predictor"}, core.MaterializeDataset(dataset), core.MetricScorer(copro.Metric))

	assert.NoError(t, err)
}

func BenchmarkEvaluateCandidatesParallel(b *testing.B) {
	metric := func(expected, actual map[string]any) float64 { return 1.0 }
	copro := NewCOPRO(metric)

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)

	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "4"}, nil)
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)

	candidates := make([]PromptCandidate, 5)
	for i := range candidates {
		candidates[i] = PromptCandidate{
			Instruction: fmt.Sprintf("Candidate %d", i),
			Score:       0.0,
			Generation:  1,
		}
	}

	examples := []core.Example{
		{
			Inputs:  map[string]any{"question": "What is 2+2?"},
			Outputs: map[string]any{"answer": "4"},
		},
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset scores
		for j := range candidates {
			candidates[j].Score = 0.0
		}
		if err := copro.evaluateCandidatesParallel(ctx, coproSinglePredictorProgram(predictor), "predictor", candidates, examples, core.MetricScorer(copro.Metric)); err != nil {
			b.Fatal(err)
		}
	}
}

// scriptLLM answers prompt-generation prompts with initial and refine,
// and task prompts with behave(instruction block, prompt).
type scriptLLM struct {
	core.LLM
	mu      sync.Mutex
	prompts []string
	initial string
	refine  string
	behave  func(head, prompt string) (string, error)
}

const (
	coproInitialPrompt = "You are an expert prompt engineer specializing"
	coproRefinePrompt  = "You are an expert prompt engineer analyzing previous instruction attempts"
)

func (m *scriptLLM) ModelID() string      { return "script" }
func (m *scriptLLM) ProviderName() string { return "test" }
func (m *scriptLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *scriptLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	m.mu.Lock()
	m.prompts = append(m.prompts, prompt)
	m.mu.Unlock()
	switch {
	case strings.HasPrefix(prompt, coproInitialPrompt):
		return &core.LLMResponse{Content: m.initial}, nil
	case strings.HasPrefix(prompt, coproRefinePrompt):
		return &core.LLMResponse{Content: m.refine}, nil
	}
	out, err := m.behave(strings.Split(prompt, "---\n\n")[0], prompt)
	if err != nil {
		return nil, err
	}
	return &core.LLMResponse{Content: out}, nil
}

// sent returns the prompts that start with prefix.
func (m *scriptLLM) sent(prefix string) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var matched []string
	for _, prompt := range m.prompts {
		if strings.HasPrefix(prompt, prefix) {
			matched = append(matched, prompt)
		}
	}
	return matched
}

// promptInput returns the value of field in the prompt's final block, the one
// after any demonstrations.
func promptInput(prompt, field string) string {
	blocks := strings.Split(prompt, "---\n\n")
	for _, line := range strings.Split(blocks[len(blocks)-1], "\n") {
		if value, ok := strings.CutPrefix(line, field+": "); ok {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// useScriptLLM makes llm the default LLM, with no teacher LLM, for one test.
func useScriptLLM(t *testing.T, llm core.LLM) {
	t.Helper()
	oldDefault, oldTeacher := core.GlobalConfig.DefaultLLM, core.GlobalConfig.TeacherLLM
	t.Cleanup(func() {
		core.GlobalConfig.DefaultLLM, core.GlobalConfig.TeacherLLM = oldDefault, oldTeacher
	})
	core.GlobalConfig.DefaultLLM, core.GlobalConfig.TeacherLLM = llm, nil
}

func scriptQASignature(instruction string) core.Signature {
	return core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer", Prefix: "answer:"}}},
	).WithInstruction(instruction)
}

// scriptQAExamples returns n examples asking q<i> and expecting gold-q<i>.
func scriptQAExamples(n int) []core.Example {
	examples := make([]core.Example, n)
	for i := range examples {
		question := fmt.Sprintf("q%d", i)
		examples[i] = core.Example{
			Inputs:  map[string]any{"question": question},
			Outputs: map[string]any{"answer": "gold-" + question},
		}
	}
	return examples
}

func scriptExactMatch(expected, actual map[string]any) float64 {
	if strings.TrimSpace(fmt.Sprint(actual["answer"])) == expected["answer"] {
		return 1
	}
	return 0
}

// scriptTwoStageProgram rewrites a question into a query and answers the query.
func scriptTwoStageProgram(llm core.LLM) core.Program {
	rewrite := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.Field{Name: "query", Prefix: "query:"}}},
	).WithInstruction("Rewrite the question into a search query TAG-REWRITE."))
	answer := modules.NewPredict(core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "query"}}},
		[]core.OutputField{{Field: core.Field{Name: "answer", Prefix: "answer:"}}},
	).WithInstruction("Answer the search query in one word TAG-ANSWER."))
	rewrite.SetLLM(llm)
	answer.SetLLM(llm)
	return core.NewProgramWithForwardFactory(map[string]core.Module{"a_rewrite": rewrite, "b_answer": answer},
		func(modules map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
			return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
				rewritten, err := modules["a_rewrite"].Process(ctx, inputs)
				if err != nil {
					return nil, err
				}
				return modules["b_answer"].Process(ctx, map[string]any{"query": rewritten["query"]})
			}
		})
}

// goldWhen answers a question correctly when right(instruction block, question).
func goldWhen(right func(head, question string) bool) func(head, prompt string) (string, error) {
	return func(head, prompt string) (string, error) {
		question := promptInput(prompt, "question")
		if right(head, question) {
			return "answer: gold-" + question, nil
		}
		return "answer: wrong", nil
	}
}

func TestCOPROCompile_OptimizesChainOfThoughtPredictor(t *testing.T) {
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-GOOD one.\nCandidate instruction TAG-BAD two.",
		behave: func(head, prompt string) (string, error) {
			answer := "wrong"
			if strings.Contains(head, "TAG-GOOD") {
				answer = "gold-" + promptInput(prompt, "question")
			}
			return "<response><rationale>because</rationale><answer>" + answer + "</answer></response>", nil
		},
	}
	useScriptLLM(t, llm)
	cot := modules.NewChainOfThought(scriptQASignature("Base instruction TAG-BASE here."))
	cot.SetLLM(llm)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(0)).
		CompileExamples(context.Background(), coproSinglePredictorProgram(cot), scriptQAExamples(4), nil)

	require.NoError(t, err)
	assert.Equal(t, "Candidate instruction TAG-GOOD one.", out.Modules["predictor"].GetSignature().Instruction)
	result, err := out.Execute(context.Background(), map[string]any{"question": "new"})
	require.NoError(t, err)
	assert.Equal(t, "gold-new", result["answer"])
	assert.Contains(t, result, "rationale", "the optimized module is still a ChainOfThought")
}

func TestCOPROCompile_ScoresWholeMultiPredictorProgram(t *testing.T) {
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-GOOD one.\nCandidate instruction TAG-BAD two.",
		behave: func(head, prompt string) (string, error) {
			if question := promptInput(prompt, "question"); question != "" {
				// Only the original rewrite instruction produces a usable query.
				if strings.Contains(head, "TAG-REWRITE") {
					return "query: search-" + question, nil
				}
				return "query: broken", nil
			}
			question, ok := strings.CutPrefix(promptInput(prompt, "query"), "search-")
			if ok && strings.Contains(head, "TAG-GOOD") {
				return "answer: gold-" + question, nil
			}
			return "answer: wrong", nil
		},
	}
	useScriptLLM(t, llm)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(0)).
		CompileExamples(context.Background(), scriptTwoStageProgram(llm), scriptQAExamples(4), nil)

	require.NoError(t, err)
	assert.Contains(t, out.Modules["a_rewrite"].GetSignature().Instruction, "TAG-REWRITE", "no candidate beats the original rewrite instruction")
	assert.Equal(t, "Candidate instruction TAG-GOOD one.", out.Modules["b_answer"].GetSignature().Instruction)
	result, err := out.Execute(context.Background(), map[string]any{"question": "new"})
	require.NoError(t, err)
	assert.Equal(t, "gold-new", result["answer"])
}

func TestCOPROCompile_KeepsDemosAndPredictorSettings(t *testing.T) {
	var fewestDemoBlocks atomic.Int64
	fewestDemoBlocks.Store(-1)
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-GOOD one.\nCandidate instruction TAG-BAD two.",
		behave: func(head, prompt string) (string, error) {
			blocks := int64(len(strings.Split(prompt, "---\n\n")) - 2)
			if fewest := fewestDemoBlocks.Load(); fewest < 0 || blocks < fewest {
				fewestDemoBlocks.Store(blocks)
			}
			if strings.Contains(head, "TAG-GOOD") {
				return "answer: gold-" + promptInput(prompt, "question"), nil
			}
			return "answer: wrong", nil
		},
	}
	useScriptLLM(t, llm)
	demos := []core.Example{
		{Inputs: map[string]any{"question": "d1"}, Outputs: map[string]any{"answer": "gold-d1"}},
		{Inputs: map[string]any{"question": "d2"}, Outputs: map[string]any{"answer": "gold-d2"}},
	}
	predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here.")).
		WithName("router").
		WithDefaultOptions(core.WithGenerateOptions(core.WithTemperature(0)))
	predictor.SetLLM(llm)
	predictor.SetDemos(demos)
	// Predict refuses to export instruction artifacts while it has default options.
	_, _, optionsErr := predictor.InstructionArtifacts()
	require.Error(t, optionsErr)

	// One compile at a time, so the atomic above need not be a strict minimum.
	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(0)).
		CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(4), nil)

	require.NoError(t, err)
	optimized := out.Modules["predictor"].(*modules.Predict)
	assert.Equal(t, "Candidate instruction TAG-GOOD one.", optimized.GetSignature().Instruction)
	assert.Equal(t, demos, optimized.GetDemos())
	assert.Equal(t, "router", optimized.GetDisplayName())
	_, _, optionsErr = optimized.InstructionArtifacts()
	assert.Error(t, optionsErr, "default options are kept")
	assert.Equal(t, int64(2), fewestDemoBlocks.Load(), "every candidate is evaluated with the predictor's demos")
}

func TestCOPROCompile_FailedPredictionsCountAsZero(t *testing.T) {
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-FLAKY one.",
		behave: func(head, prompt string) (string, error) {
			question := promptInput(prompt, "question")
			if strings.Contains(head, "TAG-FLAKY") {
				// Fails on nine of ten examples and is right on the last.
				if question != "q9" {
					return "", fmt.Errorf("provider rejected prompt")
				}
				return "answer: gold-q9", nil
			}
			// The original instruction is right on nine of ten.
			if question == "q9" {
				return "answer: wrong", nil
			}
			return "answer: gold-" + question, nil
		},
	}
	useScriptLLM(t, llm)
	predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-SOLID here."))
	predictor.SetLLM(llm)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(1), WithDepth(0)).
		CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(10), nil)

	require.NoError(t, err)
	assert.Equal(t, "Base instruction TAG-SOLID here.", out.Modules["predictor"].GetSignature().Instruction)
}

func TestCOPROCompile_SelectsOnAllExamples(t *testing.T) {
	compile := func(t *testing.T, initial string, right func(head, question string) bool) string {
		llm := &scriptLLM{initial: initial, behave: goldWhen(right)}
		useScriptLLM(t, llm)
		predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here."))
		predictor.SetLLM(llm)
		out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(0)).
			CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(9), nil)
		require.NoError(t, err)
		return out.Modules["predictor"].GetSignature().Instruction
	}

	t.Run("best over all examples beats best on the last third", func(t *testing.T) {
		selected := compile(t, "Candidate instruction TAG-MOST one.\nCandidate instruction TAG-TAIL two.", func(head, question string) bool {
			switch {
			case strings.Contains(head, "TAG-MOST"): // 8 of 9: all but q8
				return question != "q8"
			case strings.Contains(head, "TAG-TAIL"): // 3 of 9: only the last third
				return question >= "q6"
			}
			return question <= "q3" // the original: 4 of 9
		})
		assert.Equal(t, "Candidate instruction TAG-MOST one.", selected)
	})

	t.Run("original instruction wins a tie", func(t *testing.T) {
		selected := compile(t, "Candidate instruction TAG-SAME one.\nCandidate instruction TAG-SAME two.", func(_, question string) bool {
			return question <= "q3"
		})
		assert.Equal(t, "Base instruction TAG-BASE here.", selected)
	})
}

func TestCOPROCompile_RefinesWithLLMAtEveryDepth(t *testing.T) {
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-I1 one.\nCandidate instruction TAG-I2 two.\nCandidate instruction TAG-I3 three.\nCandidate instruction TAG-I4 four.\nCandidate instruction TAG-I5 five.",
		refine:  "Refined instruction TAG-R1 one.\nRefined instruction TAG-R2 two.",
		// Each instruction answers a different number of questions correctly.
		behave: goldWhen(func(head, question string) bool {
			for i, tag := range []string{"TAG-BASE", "TAG-I1", "TAG-I2", "TAG-I3", "TAG-I4", "TAG-I5"} {
				if strings.Contains(head, tag) {
					return question < fmt.Sprintf("q%d", i)
				}
			}
			return false
		}),
	}
	useScriptLLM(t, llm)
	predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here."))
	predictor.SetLLM(llm)

	_, err := NewCOPRO(scriptExactMatch, WithBreadth(5), WithDepth(2)).
		CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(6), nil)
	require.NoError(t, err)

	refinements := llm.sent(coproRefinePrompt)
	require.Len(t, refinements, 2, "one refinement call per depth")
	for _, prompt := range llm.sent("Given the fields") {
		for _, suffix := range coproRefinements() {
			assert.NotContains(t, strings.Split(prompt, "---\n\n")[0], suffix, "no fixed-suffix candidate is evaluated while the LLM proposes")
		}
	}

	// The first refinement sees every evaluated instruction's score, each
	// listed once, with the best and worst in different sections.
	_, feedback, _ := strings.Cut(refinements[0], "HIGH-PERFORMING INSTRUCTIONS:")
	feedback, _, _ = strings.Cut(feedback, "\n\nAnalyze")
	high, low, found := strings.Cut(feedback, "LOW-PERFORMING INSTRUCTIONS:")
	require.True(t, found)
	for _, tag := range []string{"TAG-I5", "TAG-I4", "TAG-I3"} {
		assert.Contains(t, high, tag)
		assert.NotContains(t, low, tag)
	}
	for _, tag := range []string{"TAG-I2", "TAG-I1", "TAG-BASE"} {
		assert.Contains(t, low, tag)
		assert.NotContains(t, high, tag)
	}
	assert.Contains(t, high, "TAG-I5 five. (score: 0.833)")
	assert.Contains(t, low, "TAG-BASE here. (score: 0.000)")
}

func TestCOPROCompile_ParsesNumberedAndQuotedProposals(t *testing.T) {
	llm := &scriptLLM{
		initial: "Here are 2 instructions:\n\n1. \"Answer using only TAG-GOOD facts from the question.\"\n2) 'Reason step by step TAG-BAD before answering.'",
		refine:  "Here are the improved instructions:\n- **Give the shortest TAG-BAD answer possible.**",
		behave: goldWhen(func(head, _ string) bool {
			return strings.Contains(head, "TAG-GOOD")
		}),
	}
	useScriptLLM(t, llm)
	predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here."))
	predictor.SetLLM(llm)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(1)).
		CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(4), nil)

	require.NoError(t, err)
	assert.Equal(t, "Answer using only TAG-GOOD facts from the question.", out.Modules["predictor"].GetSignature().Instruction)
	evaluated := map[string]bool{}
	for _, prompt := range llm.sent("Given the fields") {
		evaluated[strings.Split(prompt, "---\n\n")[0]] = true
	}
	assert.Len(t, evaluated, 4, "the original and three proposals; no preamble line is a candidate")
	for head := range evaluated {
		assert.NotContains(t, head, "Here are")
		assert.NotContains(t, head, "**")
		assert.NotContains(t, head, "2)")
	}
}

func TestCOPROCompile_EvaluatesOnlyProposedInstructions(t *testing.T) {
	for _, proposals := range [][]string{
		{"Candidate instruction TAG-P1 one."},
		{"Candidate instruction TAG-P1 one.", "Candidate instruction TAG-P2 two.", "Candidate instruction TAG-P3 three."},
	} {
		t.Run(fmt.Sprintf("%d of 5 proposed", len(proposals)), func(t *testing.T) {
			llm := &scriptLLM{
				initial: strings.Join(proposals, "\n"),
				behave:  goldWhen(func(string, string) bool { return false }),
			}
			useScriptLLM(t, llm)
			predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here."))
			predictor.SetLLM(llm)

			_, err := NewCOPRO(scriptExactMatch, WithBreadth(5), WithDepth(0)).
				CompileExamples(context.Background(), coproSinglePredictorProgram(predictor), scriptQAExamples(2), nil)

			require.NoError(t, err)
			assert.Len(t, llm.sent(coproInitialPrompt), 1, "a short proposal list is accepted, not asked for again")
			evaluated := map[string]bool{}
			for _, prompt := range llm.sent("Given the fields") {
				paragraphs := strings.Split(strings.TrimSpace(strings.Split(prompt, "---\n\n")[0]), "\n\n")
				evaluated[paragraphs[len(paragraphs)-1]] = true
			}
			want := map[string]bool{"Base instruction TAG-BASE here.": true}
			for _, proposal := range proposals {
				want[proposal] = true
			}
			assert.Equal(t, want, evaluated, "only the original and the model's proposals are evaluated")
		})
	}
}

func TestCOPROCompile_ProposesAndAppliesOutputPrefix(t *testing.T) {
	llm := &scriptLLM{
		// The same instruction twice; only the second proposes a prefix.
		initial: "Reply to the question TAG-SAME tersely.\n\"Reply to the question TAG-SAME tersely.\" || \"Final answer:\"",
		refine:  "Refined instruction TAG-R1 one.",
		behave: func(head, prompt string) (string, error) {
			if strings.Contains(head, "TAG-SAME") && strings.Contains(head, "should start with 'Final answer:'") {
				return "Final answer: gold-" + promptInput(prompt, "question"), nil
			}
			return "answer: wrong", nil
		},
	}
	useScriptLLM(t, llm)
	predictor := modules.NewPredict(scriptQASignature("Base instruction TAG-BASE here."))
	predictor.SetLLM(llm)
	program := coproSinglePredictorProgram(predictor)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(1)).
		CompileExamples(context.Background(), program, scriptQAExamples(4), nil)

	require.NoError(t, err)
	optimized := out.Modules["predictor"].GetSignature()
	assert.Equal(t, "Reply to the question TAG-SAME tersely.", optimized.Instruction)
	assert.Equal(t, "Final answer:", optimized.Outputs[0].Prefix)
	assert.Equal(t, "answer:", program.Modules["predictor"].GetSignature().Outputs[0].Prefix, "the input program is untouched")
	result, err := out.Execute(context.Background(), map[string]any{"question": "new"})
	require.NoError(t, err)
	assert.Equal(t, "gold-new", result["answer"])

	initial := llm.sent(coproInitialPrompt)
	require.Len(t, initial, 1)
	assert.Contains(t, initial[0], `"answer" output field must start with (currently "answer:")`)
	refinements := llm.sent(coproRefinePrompt)
	require.Len(t, refinements, 1)
	assert.Contains(t, refinements[0], "- Reply to the question TAG-SAME tersely. (score: 1.000) [output prefix: Final answer:]\n")
	assert.Contains(t, refinements[0], "- Reply to the question TAG-SAME tersely. (score: 0.000) [output prefix: answer:]\n", "the same instruction with the original prefix is a separate attempt")
}

func TestCOPROCompile_RescoresPredictorAfterAnotherImproves(t *testing.T) {
	llm := &scriptLLM{
		initial: "Candidate instruction TAG-GOOD one.\nCandidate instruction TAG-BAD two.",
		refine:  "Refined instruction TAG-R1 one.",
		behave: func(head, prompt string) (string, error) {
			if question := promptInput(prompt, "question"); question != "" {
				if strings.Contains(head, "TAG-GOOD") {
					return "query: search-" + question, nil
				}
				return "query: vague-" + question, nil
			}
			// The original answer instruction is always wrong. The good one is
			// right on every precise query and on two of the vague ones, so the
			// better rewrite instruction pays off only once it is installed.
			if strings.Contains(head, "TAG-GOOD") {
				query := promptInput(prompt, "query")
				if question, ok := strings.CutPrefix(query, "search-"); ok {
					return "answer: gold-" + question, nil
				}
				if query == "vague-q0" || query == "vague-q1" {
					return "answer: gold-" + strings.TrimPrefix(query, "vague-"), nil
				}
			}
			return "answer: wrong", nil
		},
	}
	useScriptLLM(t, llm)

	out, err := NewCOPRO(scriptExactMatch, WithBreadth(2), WithDepth(1)).
		CompileExamples(context.Background(), scriptTwoStageProgram(llm), scriptQAExamples(4), nil)

	require.NoError(t, err)
	assert.Equal(t, "Candidate instruction TAG-GOOD one.", out.Modules["a_rewrite"].GetSignature().Instruction)
	assert.Equal(t, "Candidate instruction TAG-GOOD one.", out.Modules["b_answer"].GetSignature().Instruction)
	result, err := out.Execute(context.Background(), map[string]any{"question": "q3"})
	require.NoError(t, err)
	assert.Equal(t, "gold-q3", result["answer"])
}

func TestCOPROCompile_RescoresOriginalPrefixAfterAnotherImproves(t *testing.T) {
	// How many of the ten questions the answer predictor gets right, by the
	// kind of query it receives. Its original prompt is weak; the proposed
	// one is perfect on the original rewrite's queries.
	correct := func(question, kind string, proposed bool) bool {
		if len(question) != 2 {
			return false
		}
		index := int(question[1] - '0')
		limit := map[string]int{"O": 2, "X": 5}[kind]
		if proposed {
			limit = map[string]int{"O": 10, "X": 7}[kind]
		}
		return index < limit
	}
	llm := &scriptLLM{
		// One proposal for each predictor: an instruction with a new prefix.
		initial: "Candidate instruction TAG-X one. || new:",
		refine:  "Refined instruction TAG-R1 one.",
		behave: func(head, prompt string) (string, error) {
			proposed, newPrefix := strings.Contains(head, "TAG-X"), strings.Contains(head, "should start with 'new:'")
			if question := promptInput(prompt, "question"); question != "" {
				switch {
				case proposed && newPrefix:
					return "new: X-" + question, nil
				case strings.Contains(head, "TAG-REWRITE") && newPrefix:
					// Not a candidate: the original instruction under the
					// proposal's prefix. Useless queries.
					return "new: B-" + question, nil
				case strings.Contains(head, "TAG-REWRITE"):
					return "query: O-" + question, nil
				}
				return "query: R-" + question, nil
			}
			kind, question, _ := strings.Cut(promptInput(prompt, "query"), "-")
			switch {
			case proposed && newPrefix && correct(question, kind, true):
				return "new: gold-" + question, nil
			case strings.Contains(head, "TAG-ANSWER") && !newPrefix && correct(question, kind, false):
				return "answer: gold-" + question, nil
			}
			return "answer: wrong", nil
		},
	}
	useScriptLLM(t, llm)

	// Depth 0: the rewrite proposal wins with the weak answer prompt (0.5
	// against 0.2), then the answer proposal wins (0.7 against 0.5). Depth 1
	// scores the rewrite candidates again: with the new answer prompt the
	// original rewrite prompt, under its original prefix, scores 1.0.
	out, err := NewCOPRO(scriptExactMatch, WithBreadth(1), WithDepth(1)).
		CompileExamples(context.Background(), scriptTwoStageProgram(llm), scriptQAExamples(10), nil)

	require.NoError(t, err)
	rewrite := out.Modules["a_rewrite"].GetSignature()
	assert.Equal(t, "Rewrite the question into a search query TAG-REWRITE.", rewrite.Instruction)
	assert.Equal(t, "query:", rewrite.Outputs[0].Prefix, "the original candidate keeps the prefix it was first scored with")
	assert.Equal(t, "Candidate instruction TAG-X one.", out.Modules["b_answer"].GetSignature().Instruction)
	result, err := out.Execute(context.Background(), map[string]any{"question": "q9"})
	require.NoError(t, err)
	assert.Equal(t, "gold-q9", result["answer"])
}
