package optimizers

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// TestMIPRO contains all tests for the MIPRO optimizer.
func TestMIPRO(t *testing.T) {
	// We'll organize our tests into logical groups using subtests
	t.Run("Constructor and Configuration", func(t *testing.T) {
		t.Run("NewMIPRO creates instance with default values", func(t *testing.T) {
			metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
				return 1.0
			}

			mipro := NewMIPRO(metric)

			assert.NotNil(t, mipro)
			assert.Equal(t, MediumMode, mipro.config.Mode)
			assert.NotNil(t, mipro.state)
			assert.NotNil(t, mipro.metrics)
		})

		t.Run("Options properly configure MIPRO", func(t *testing.T) {
			metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
				return 1.0
			}

			mipro := NewMIPRO(metric,
				WithMode(LightMode),
				WithNumTrials(10),
				WithTeacherSettings(map[string]interface{}{
					"temperature": 0.7,
				}),
			)

			assert.Equal(t, LightMode, mipro.config.Mode)
			assert.Equal(t, 10, mipro.config.NumTrials)
			assert.Equal(t, 0.7, mipro.config.TeacherSettings["temperature"])
		})
	})

	t.Run("Teacher-Student Functionality", func(t *testing.T) {
		t.Run("Teacher generates high-quality demonstrations", func(t *testing.T) {
			ctx := context.Background()
			mockTeacher := new(testutil.MockLLM)
			mockTeacher.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
				&core.LLMResponse{Content: "High quality response"}, nil)

			mipro := createTestMIPRO(t)
			mipro.teacherStudent.Teacher = mockTeacher

			example := core.Example{
				Inputs: map[string]interface{}{
					"prompt": "Test prompt",
				},
			}

			result, err := mipro.teacherDemonstration(ctx, example)

			assert.NoError(t, err)
			assert.Equal(t, "High quality response", result.Outputs["completion"])
			mockTeacher.AssertExpectations(t)
		})

		t.Run("Teacher error is properly handled", func(t *testing.T) {
			ctx := context.Background()
			mockTeacher := new(testutil.MockLLM)
			mockTeacher.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(
				nil, errors.New(errors.LLMGenerationFailed, "teacher error"))

			mipro := createTestMIPRO(t)
			mipro.teacherStudent.Teacher = mockTeacher

			example := core.Example{
				Inputs: map[string]interface{}{
					"prompt": "Test prompt",
				},
			}

			_, err := mipro.teacherDemonstration(ctx, example)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), "teacher error")
		})
	})

	t.Run("Optimization Process", func(t *testing.T) {
		t.Run("Successfully optimizes simple program", func(t *testing.T) {
			ctx := context.Background()

			mockStrategy := new(MockSearchStrategy)
			mipro := createTestMIPRO(t)
			mipro.searchStrategy = mockStrategy

			program := createTestProgram()

			mockModule := program.Modules["test"].(*MockModule)
			// Sample example
			sampleExample := core.Example{
				Inputs:  map[string]interface{}{"input": "sample input"},
				Outputs: map[string]interface{}{"output": "expected output"},
			}

			dataset := testutil.NewMockDataset([]core.Example{sampleExample})

			// Set up instruction generator with a mock LLM
			mockLLM := new(testutil.MockLLM)
			mockLLM.On("Generate", mock.Anything, mock.Anything).Return(&core.LLMResponse{Content: "test instruction"}, nil).Maybe()
			mipro.instructionGenerator.PromptModel = mockLLM

			// Mock dependencies
			mipro.teacherStudent = &TeacherStudentOptimizer{}
			mipro.instructionGenerator = &InstructionGenerator{PromptModel: mockLLM}

			// Expectations for Compile flow
			mockStrategy.On("SuggestParams", mock.Anything).Return(map[string]interface{}{"module_0_instruction": float64(0)}, nil).Once()

			mockStrategy.On("UpdateResults", mock.Anything, mock.AnythingOfType("float64")).Return(nil).Once()
			mockStrategy.On("GetBestParams").Return(
				map[string]interface{}{"module_0_instruction": float64(0)}, 0.8).Once()
			mockModule.On("Clone").Return(mockModule).Maybe()
			mockModule.On("SetSignature", mock.Anything).Return().Maybe()
			mockModule.On("SetLLM", mock.Anything).Return().Maybe()
			mockModule.On("GetSignature").Return(program.Modules["test"].GetSignature()).Maybe()
			mockModule.On("Process", mock.Anything, mock.Anything).Return(map[string]interface{}{"output": "mocked output"}, nil).Maybe()
			dataset.On("Reset").Return()
			dataset.On("Next").Return(sampleExample, true)
			dataset.On("Next").Return(core.Example{}, false)
			// Run the test
			result, err := mipro.Compile(ctx, program, dataset, nil)

			// Verify expectations and call counts
			assert.NoError(t, err)
			assert.NotNil(t, result)
			assert.Equal(t, 0.8, mipro.state.BestScore)
			mockModule.AssertExpectations(t)
			mockStrategy.AssertExpectations(t)
			dataset.AssertNumberOfCalls(t, "Reset", 2) // Expect 2 Reset calls
			dataset.AssertNumberOfCalls(t, "Next", 4)  // Expect 4 Next calls (2 per evaluateCandidate)
		})

		t.Run("Handles optimization failures gracefully", func(t *testing.T) {
			ctx := context.Background()
			mipro := createTestMIPRO(t)

			// Create a failing search strategy
			mockStrategy := new(MockSearchStrategy)
			mockStrategy.On("SuggestParams", mock.Anything).Return(
				nil, errors.New(errors.Unknown, "search failed"))
			mipro.searchStrategy = mockStrategy

			program := createTestProgram()
			dataset := createTestDataset()

			_, err := mipro.Compile(ctx, program, dataset, nil)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), "search failed")
		})

		t.Run("Converges early with good solution", func(t *testing.T) {
			ctx := context.Background()
			mipro := createTestMIPRO(t)
			mockStrategy := new(MockSearchStrategy)
			mipro.searchStrategy = mockStrategy
			program := createTestProgram()
			dataset := createTestDataset()
			mockMod := program.Modules["test"].(*MockModule)
			mockDataset := dataset.(*testutil.MockDataset)

			sampleExample := core.Example{Inputs: map[string]interface{}{"input": "converge_test"}}

			// Mock setup for convergence test
			mipro.teacherStudent = &TeacherStudentOptimizer{}    // Simplify
			mipro.instructionGenerator = &InstructionGenerator{} // Simplify
			mockStrategy.On("SuggestParams", ctx).Return(map[string]interface{}{"module_0_instruction": float64(0)}, nil).Once()

			mockStrategy.On("UpdateResults", mock.Anything, 0.8).Return(nil).Maybe()
			mockStrategy.On("GetBestParams").Return(map[string]interface{}{"module_0_instruction": float64(0)}, 0.8).Once()

			mockMod.On("Clone").Return(mockMod).Maybe()
			mockMod.On("Clone").Return(mockMod).Maybe()
			mockMod.On("SetSignature", mock.Anything).Return().Maybe()
			mockMod.On("SetLLM", mock.Anything).Return().Maybe()
			mockMod.On("GetSignature").Return(program.Modules["test"].GetSignature()).Maybe()
			mockMod.On("Process", mock.Anything, mock.Anything).Return(map[string]interface{}{"output": "test"}, nil).Maybe()
			mockDataset.On("Reset").Return().Once()
			mockDataset.On("Next").Return(sampleExample, true).Once()   // 1st example
			mockDataset.On("Next").Return(sampleExample, true).Once()   // 2nd example
			mockDataset.On("Next").Return(core.Example{}, false).Once() // End loop

			mockStrategy.On("UpdateResults", mock.Anything, 0.8).Return(nil).Maybe()

			mockDataset.On("Reset").Return().Once()
			mockDataset.On("Next").Return(sampleExample, true).Once() // 1st example again

			mockDataset.On("Next").Return(sampleExample, true).Once()    // 2nd example
			mockDataset.On("Next").Return(core.Example{}, false).Maybe() // End loop

			start := time.Now()
			optimizedProgram, err := mipro.Compile(ctx, program, dataset, nil)

			assert.NoError(t, err)
			assert.NotNil(t, optimizedProgram)
			assert.True(t, time.Since(start) < time.Second)
			mockStrategy.AssertExpectations(t)
			mockMod.AssertExpectations(t)
			mockDataset.AssertExpectations(t)
		})
	})

	t.Run("Metrics and State Tracking", func(t *testing.T) {
		t.Run("Correctly tracks optimization metrics", func(t *testing.T) {
			mipro := createTestMIPRO(t)

			mipro.updateMetrics(0.8, &core.TokenInfo{
				PromptTokens:     100,
				CompletionTokens: 50,
			})

			assert.Len(t, mipro.metrics.OptimizationHistory, 1)
			assert.Equal(t, 100, mipro.metrics.TokenUsage.PromptTokens)
		})

		t.Run("State updates reflect optimization progress", func(t *testing.T) {
			mipro := createTestMIPRO(t)
			testProg := createTestProgram()

			mipro.updateOptimizationState(
				map[string]interface{}{"param": "test"},
				0.9,
				testProg,
			)

			assert.Equal(t, 1, mipro.state.CurrentIteration)
			assert.Equal(t, 0.9, mipro.state.BestScore)
			assert.NotEmpty(t, mipro.state.SuccessfulPatterns)
		})
	})

	t.Run("Search Strategy Integration", func(t *testing.T) {
		t.Run("Successfully uses TPE search strategy", func(t *testing.T) {
			ctx := context.Background()
			mipro := createTestMIPRO(t)

			program := createTestProgram()
			dataset := createTestDataset()

			// Add Reset expectation for the mock dataset
			mockDataset := dataset.(*testutil.MockDataset)
			mockDataset.On("Reset").Return().Times(3)
			mockDataset.On("Next").Return(core.Example{Inputs: map[string]interface{}{"input": "tpe_test"}}, true).Maybe()
			mockDataset.On("Next").Return(core.Example{Inputs: map[string]interface{}{"input": "tpe_test2"}}, true).Maybe()
			mockDataset.On("Next").Return(core.Example{}, false).Maybe()
			// Need mock expectations for the module inside the program
			mockMod := program.Modules["test"].(*MockModule)
			mockMod.On("SetSignature", mock.Anything).Maybe()
			mockMod.On("SetLLM", mock.Anything).Maybe()

			mockMod.On("Process", mock.Anything, mock.Anything).Return(map[string]any{"output": "test"}, nil).Maybe()
			optimizedProgram, err := mipro.Compile(ctx, program, dataset, nil)

			assert.NoError(t, err)
			assert.NotNil(t, optimizedProgram)
			mockDataset.AssertExpectations(t)
			mockMod.AssertExpectations(t)
		})
	})
}

// Helper functions for testing.
func createTestMIPRO(t *testing.T) *MIPRO {
	t.Helper()

	metric := func(example, prediction map[string]any, ctx context.Context) float64 {
		return 0.8 // Return a constant score for testing
	}

	mipro := NewMIPRO(metric,
		WithMode(LightMode),
		WithNumTrials(1),
		WithTeacherSettings(map[string]interface{}{
			"temperature": 0.5,
		}),
	)

	// Initialize search strategy with mock for testing
	mipro.searchStrategy = NewMockTPEOptimizer(0.25, 42)
	err := mipro.searchStrategy.Initialize(SearchConfig{
		MaxTrials: 5,
		ParamSpace: map[string][]interface{}{
			"module_0_instruction": {0, 1, 2}, // Assuming we have 3 instruction candidates
		},
	})
	if err != nil {
		t.Fatalf("Failed to initialize search strategy: %v", err)
	}

	return mipro
}

func createTestProgram() core.Program {
	mockModule := new(MockModule)
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "input"}}},
		[]core.OutputField{{Field: core.Field{Name: "output"}}},
	).WithInstruction("Test Instruction")

	mockModule.On("GetSignature").Return(signature).Maybe()
	mockModule.On("Clone").Return(mockModule).Maybe()

	// Define the modules map first
	modules := map[string]core.Module{"test": mockModule}

	program := core.NewProgram(
		modules, // Pass the map here
		// *** FIX: Capture the 'modules' map instead of 'program' ***
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			modInstance, ok := modules["test"] // Use the captured 'modules' map
			if !ok {
				return nil, fmt.Errorf("module 'test' not found in program")
			}
			// Note: We already know this is a core.Module from the map definition, so no need to check type
			return modInstance.Process(ctx, inputs)
		},
	)
	return program
}

func createTestDataset() core.Dataset {
	examples := []core.Example{
		{
			Inputs:  map[string]interface{}{"input": "test1"},
			Outputs: map[string]interface{}{"output": "result1"},
		},
		{
			Inputs:  map[string]interface{}{"input": "test2"},
			Outputs: map[string]interface{}{"output": "result2"},
		},
	}

	return testutil.NewMockDataset(examples)
}

// Test options

func TestMIPROOptions(t *testing.T) {
	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
		return 1.0
	}

	mipro := NewMIPRO(metric,
		WithMode(LightMode),
		WithNumTrials(10),
	)

	assert.Equal(t, LightMode, mipro.config.Mode)
	assert.Equal(t, 10, mipro.config.NumTrials)
}

// MockTPEOptimizer implements a simplified version of Tree-structured Parzen Estimators for testing.
type MockTPEOptimizer struct {
	gamma float64
	seed  int64
}

// NewMockTPEOptimizer creates a new mock TPE optimizer instance for testing.
func NewMockTPEOptimizer(gamma float64, seed int64) SearchStrategy {
	return &MockTPEOptimizer{
		gamma: gamma,
		seed:  seed,
	}
}

func (t *MockTPEOptimizer) SuggestParams(ctx context.Context) (map[string]interface{}, error) {
	// Simplified implementation for testing
	return map[string]interface{}{
		"module_0_instruction": float64(0), // Return float64 as expected by the code
	}, nil
}

func (t *MockTPEOptimizer) UpdateResults(params map[string]interface{}, score float64) error {
	return nil
}

func (t *MockTPEOptimizer) GetBestParams() (map[string]interface{}, float64) {
	return map[string]interface{}{"module_0_instruction": float64(0)}, 1.0
}

func (t *MockTPEOptimizer) Initialize(config SearchConfig) error {
	return nil
}

// Mock implementations for testing.
type MockSearchStrategy struct {
	mock.Mock
}

func (m *MockSearchStrategy) SuggestParams(ctx context.Context) (map[string]interface{}, error) {
	args := m.Called(ctx)
	result := args.Get(0)
	if result == nil {
		return nil, args.Error(1)
	}
	return result.(map[string]interface{}), args.Error(1)
}

func (m *MockSearchStrategy) UpdateResults(params map[string]interface{}, score float64) error {
	args := m.Called(params, score)
	return args.Error(0)
}

func (m *MockSearchStrategy) GetBestParams() (map[string]interface{}, float64) {
	args := m.Called()
	return args.Get(0).(map[string]interface{}), args.Get(1).(float64)
}

func (m *MockSearchStrategy) Initialize(config SearchConfig) error {
	args := m.Called(config)
	return args.Error(0)
}
