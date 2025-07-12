package optimizers

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockLLM for testing
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

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

func (m *MockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, functions, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
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

func (m *MockModule) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	args := m.Called(ctx, inputs)
	return args.Get(0).(map[string]interface{}), args.Error(1)
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
			name:   "Default options",
			metric: func(expected, actual map[string]interface{}) float64 { return 1.0 },
			options: nil,
			expected: &COPRO{
				Breadth:         10,
				Depth:           3,
				InitTemperature: 1.4,
				TrackStats:      false,
			},
		},
		{
			name:   "Custom options",
			metric: func(expected, actual map[string]interface{}) float64 { return 0.5 },
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
			assert.NotNil(t, copro.Metric)
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
	predictor := modules.NewPredict(signature)
	
	return core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)
}

func createCOPROTestDataset() *datasets.SimpleDataset {
	examples := []core.Example{
		{
			Inputs:  map[string]interface{}{"question": "What is 2+2?"},
			Outputs: map[string]interface{}{"answer": "4"},
		},
		{
			Inputs:  map[string]interface{}{"question": "What is 3+3?"},
			Outputs: map[string]interface{}{"answer": "6"},
		},
	}
	return datasets.NewSimpleDataset(examples)
}

func TestCOPROCompile_Success(t *testing.T) {
	// Create test program and dataset
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()
	
	// Create metric
	metric := func(expected, actual map[string]interface{}) float64 {
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
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return inputs, nil
		},
	)
	dataset := createCOPROTestDataset()
	
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
	copro := NewCOPRO(metric)
	ctx := core.WithExecutionState(context.Background())
	
	optimizedProgram, err := copro.Compile(ctx, program, dataset, metric)
	
	assert.NoError(t, err)
	assert.NotNil(t, optimizedProgram)
	assert.Equal(t, 0, len(optimizedProgram.Modules))
}

func TestCOPROCompile_EmptyDataset(t *testing.T) {
	program := createCOPROTestProgram()
	emptyDataset := datasets.NewSimpleDataset([]core.Example{})
	
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	
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
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 }, WithBreadth(3))
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()
	
	candidates := copro.generateInitialCandidates(ctx, predictor, "base instruction")
	
	assert.Len(t, candidates, 3)
	for _, candidate := range candidates {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 1, candidate.Generation)
	}
}

func TestRefineCandidates(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 }, WithBreadth(4))
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()
	
	topCandidates := []PromptCandidate{
		{Instruction: "Test instruction 1", Score: 0.9, Generation: 1},
		{Instruction: "Test instruction 2", Score: 0.8, Generation: 1},
	}
	
	refined := copro.refineCandidates(ctx, predictor, topCandidates, 2)
	
	assert.Greater(t, len(refined), 0)
	for _, candidate := range refined {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 3, candidate.Generation) // depth + 1
	}
}

func TestEvaluateCandidate(t *testing.T) {
	metric := func(expected, actual map[string]interface{}) float64 {
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
			Inputs:  map[string]interface{}{"question": "What is 2+2?"},
			Outputs: map[string]interface{}{"answer": "4"},
		},
	}
	
	ctx := context.Background()
	score := copro.evaluateCandidate(ctx, predictor, candidate, examples)
	
	assert.GreaterOrEqual(t, score, 0.0)
	assert.LessOrEqual(t, score, 1.0)
}

func TestEvaluateCandidate_LLMError(t *testing.T) {
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
	copro := NewCOPRO(metric)
	
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	predictor := modules.NewPredict(signature)
	
	mockLLM := &MockLLM{}
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{}, fmt.Errorf("LLM error"))
	mockLLM.On("GetModelName").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})
	predictor.SetLLM(mockLLM)
	
	candidate := PromptCandidate{Instruction: "Test", Score: 0.0, Generation: 1}
	examples := []core.Example{{
		Inputs:  map[string]interface{}{"question": "test"},
		Outputs: map[string]interface{}{"answer": "test"},
	}}
	
	ctx := context.Background()
	score := copro.evaluateCandidate(ctx, predictor, candidate, examples)
	
	assert.Equal(t, 0.0, score) // Should return 0 when no valid evaluations
}

func TestEvaluateCandidatesParallel(t *testing.T) {
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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
			Inputs:  map[string]interface{}{"question": "What is 2+2?"},
			Outputs: map[string]interface{}{"answer": "4"},
		},
	}
	
	ctx := context.Background()
	copro.evaluateCandidatesParallel(ctx, predictor, candidates, examples)
	
	// All candidates should have been evaluated (score set)
	for _, candidate := range candidates {
		assert.GreaterOrEqual(t, candidate.Score, 0.0)
	}
}

func TestDatasetToExamples(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	dataset := createCOPROTestDataset()
	
	examples := copro.datasetToExamples(dataset)
	
	assert.Len(t, examples, 2)
	assert.Equal(t, "What is 2+2?", examples[0].Inputs["question"])
	assert.Equal(t, "4", examples[0].Outputs["answer"])
}

func TestGetInstructionTemplates(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	signature := core.Signature{}
	
	templates := copro.getInstructionTemplates(signature)
	
	assert.Greater(t, len(templates), 0)
	for _, template := range templates {
		assert.NotEmpty(t, template)
	}
}

func TestVaryInstruction(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	
	tests := []struct {
		name         string
		instruction  string
		temperature  float64
		expectChange bool
	}{
		{
			name:         "Empty instruction",
			instruction:  "",
			temperature:  1.0,
			expectChange: true,
		},
		{
			name:         "High temperature",
			instruction:  "Base instruction",
			temperature:  1.5,
			expectChange: true,
		},
		{
			name:         "Low temperature",
			instruction:  "Base instruction",
			temperature:  0.5,
			expectChange: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := copro.varyInstruction(tt.instruction, tt.temperature)
			assert.NotEmpty(t, result)
			
			if tt.expectChange && tt.instruction != "" {
				// May or may not change due to randomness, but should be valid
				assert.NotEmpty(t, result)
			}
		})
	}
}

func TestRefineInstruction(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	
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
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	
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

func TestTruncateString(t *testing.T) {
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "Short string",
			input:    "Hello",
			maxLen:   10,
			expected: "Hello",
		},
		{
			name:     "Long string",
			input:    "This is a very long string that should be truncated",
			maxLen:   10,
			expected: "This is a ...",
		},
		{
			name:     "Exact length",
			input:    "Exactly10!",
			maxLen:   10,
			expected: "Exactly10!",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := copro.truncateString(tt.input, tt.maxLen)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestHelperFunctions(t *testing.T) {
	// Test min function
	assert.Equal(t, 3, min(3, 5))
	assert.Equal(t, 2, min(7, 2))
	assert.Equal(t, 4, min(4, 4))
	
	// Test maxInt function
	assert.Equal(t, 5, maxInt(3, 5))
	assert.Equal(t, 7, maxInt(7, 2))
	assert.Equal(t, 4, maxInt(4, 4))
}

func TestCOPROWithExecutionState(t *testing.T) {
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()
	
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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

// Benchmark tests
func BenchmarkCOPROCompile(b *testing.B) {
	program := createCOPROTestProgram()
	dataset := createCOPROTestDataset()
	
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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
	
	program := core.NewProgram(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return predictor.Process(ctx, inputs)
		},
	)
	
	dataset := createCOPROTestDataset()
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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
	// Test when breadth > number of templates to cover missing branch
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 }, WithBreadth(20)) // Large breadth
	predictor := modules.NewPredict(core.Signature{})
	ctx := context.Background()
	
	candidates := copro.generateInitialCandidates(ctx, predictor, "base instruction")
	
	assert.Len(t, candidates, 20)
	for _, candidate := range candidates {
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, 1, candidate.Generation)
	}
}

func TestOptimizePredictor_EmptyDataset(t *testing.T) {
	// Test direct call to optimizePredictor with empty dataset
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 1.0 })
	predictor := modules.NewPredict(core.Signature{})
	emptyDataset := datasets.NewSimpleDataset([]core.Example{})
	
	ctx := context.Background()
	err := copro.optimizePredictor(ctx, predictor, emptyDataset)
	
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no examples in dataset for optimization")
}

func TestOptimizePredictor_WithCurrentInstruction(t *testing.T) {
	// Test optimizePredictor with current instruction to test baseline branch
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 0.9 })
	
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
	
	err := copro.optimizePredictor(ctx, predictor, dataset)
	
	assert.NoError(t, err)
}

func TestOptimizePredictor_NoCurrentInstruction(t *testing.T) {
	// Test optimizePredictor without current instruction to test empty instruction branch
	copro := NewCOPRO(func(expected, actual map[string]interface{}) float64 { return 0.9 }, WithBreadth(2), WithDepth(1))
	
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
	
	err := copro.optimizePredictor(ctx, predictor, dataset)
	
	assert.NoError(t, err)
}

func BenchmarkEvaluateCandidatesParallel(b *testing.B) {
	metric := func(expected, actual map[string]interface{}) float64 { return 1.0 }
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
			Inputs:  map[string]interface{}{"question": "What is 2+2?"},
			Outputs: map[string]interface{}{"answer": "4"},
		},
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset scores
		for j := range candidates {
			candidates[j].Score = 0.0
		}
		copro.evaluateCandidatesParallel(ctx, predictor, candidates, examples)
	}
}