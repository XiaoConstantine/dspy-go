package agents

import (
	"context"
	"errors"
	"testing"
	"time"

	testutil "github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Mock implementations for testing.
type MockTaskProcessor struct {
	mock.Mock
}

func (m *MockTaskProcessor) Process(ctx context.Context, task Task, context map[string]interface{}) (interface{}, error) {
	args := m.Called(ctx, task, context)
	return args.Get(0), args.Error(1)
}

type MockTaskParser struct {
	mock.Mock
}

func (m *MockTaskParser) Parse(analyzerOutput map[string]interface{}) ([]Task, error) {
	args := m.Called(analyzerOutput)
	if tasks, ok := args.Get(0).([]Task); ok {
		return tasks, args.Error(1)
	}
	return nil, args.Error(1)
}

type MockPlanCreator struct {
	mock.Mock
}

func (m *MockPlanCreator) CreatePlan(tasks []Task) ([][]Task, error) {
	args := m.Called(tasks)
	if plan, ok := args.Get(0).([][]Task); ok {
		return plan, args.Error(1)
	}
	return nil, args.Error(1)
}

// setupTestContext creates a context with execution state for testing.
func setupTestContext() context.Context {
	ctx := context.Background()
	return core.WithExecutionState(ctx)
}

func setupTestOrchestrator() (*FlexibleOrchestrator, *MockTaskProcessor, *MockTaskParser, *MockPlanCreator, *testutil.MockLLM) {
	memory := NewInMemoryStore()
	mockProcessor := new(MockTaskProcessor)
	mockParser := new(MockTaskParser)
	mockPlanner := new(MockPlanCreator)
	mockLLM := new(testutil.MockLLM)

	// Set up default LLM for the analyzer
	core.GlobalConfig.DefaultLLM = mockLLM

	config := OrchestrationConfig{
		MaxConcurrent:  2,
		DefaultTimeout: 30 * time.Second,
		RetryConfig: &RetryConfig{
			MaxAttempts:       3,
			BackoffMultiplier: 2.0,
		},
		CustomProcessors: map[string]TaskProcessor{
			"test": mockProcessor,
		},
		TaskParser:  mockParser,
		PlanCreator: mockPlanner,
		AnalyzerConfig: AnalyzerConfig{
			FormatInstructions: "Format as XML",
		},
		Options: core.WithGenerateOptions(core.WithMaxTokens(100)),
	}

	orchestrator := NewFlexibleOrchestrator(memory, config)
	return orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM
}

func TestFlexibleOrchestrator(t *testing.T) {
	t.Run("Basic Task Processing", func(t *testing.T) {
		// Setup
		orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()

		// Test data
		tasks := []Task{
			{
				ID:            "task1",
				Type:          "test",
				ProcessorType: "test",
				Priority:      1,
				Metadata: map[string]interface{}{
					"key": "value",
				},
			},
		}

		analyzerResp := &core.LLMResponse{
			Content: `<response>
<analysis>Task has been analyzed and broken down into atomic units</analysis>
<tasks>
     <task id="task1" type="test" processor="test" priority="1">
         <description>Test task 1</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
</tasks>
</response>`,
			Usage: &core.TokenInfo{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		}

		// Setup expectations
		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(analyzerResp, nil)
		mockParser.On("Parse", mock.Anything).Return(tasks, nil)
		mockPlanner.On("CreatePlan", tasks).Return([][]Task{tasks}, nil)
		mockProcessor.On("Process", mock.Anything, tasks[0], mock.Anything).Return("result1", nil)

		// Execute with proper context
		ctx := setupTestContext()
		result, err := orchestrator.Process(ctx, "Test task", nil)

		// Verify
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Contains(t, result.CompletedTasks, "task1")
		assert.Empty(t, result.FailedTasks)

		mockLLM.AssertExpectations(t)
		mockParser.AssertExpectations(t)
		mockPlanner.AssertExpectations(t)
		mockProcessor.AssertExpectations(t)
	})

	t.Run("Parallel Task Execution", func(t *testing.T) {
		orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()

		// Test data
		tasks := []Task{
			{ID: "task1", Type: "test", ProcessorType: "test", Priority: 1},
			{ID: "task2", Type: "test", ProcessorType: "test", Priority: 1},
		}

		// Mock the analyzer response in the format expected by Predict module
		analyzerResp := &core.LLMResponse{
			Content: `<response>
<analysis>Task has been analyzed and decomposed for parallel execution</analysis>
<tasks><tasks>
     <task id="task1" type="test" processor="test" priority="1">
         <description>Parallel task 1</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
     <task id="task2" type="test" processor="test" priority="1">
         <description>Parallel task 2</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
</tasks></tasks>
</response>`,
			Usage: &core.TokenInfo{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		}
		// Setup expectations
		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(analyzerResp, nil)
		mockParser.On("Parse", mock.Anything).Return(tasks, nil)
		mockPlanner.On("CreatePlan", tasks).Return([][]Task{tasks}, nil)
		mockProcessor.On("Process", mock.Anything, mock.AnythingOfType("Task"), mock.Anything).Return("result", nil)

		// Execute with proper context
		ctx := setupTestContext()
		result, err := orchestrator.Process(ctx, "Test parallel tasks", nil)

		// Verify
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Len(t, result.CompletedTasks, 2)
		assert.Contains(t, result.CompletedTasks, "task1")
		assert.Contains(t, result.CompletedTasks, "task2")
		assert.Empty(t, result.FailedTasks)
	})

	t.Run("Retry Logic", func(t *testing.T) {
		orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()

		// Test data
		task := Task{ID: "retry-task", Type: "test", ProcessorType: "test"}
		tasks := []Task{task}

		analyzerResp := &core.LLMResponse{
			Content: `<response>
<analysis>Task has been analyzed for retry testing</analysis>
<tasks><tasks>
     <task id="retry-task" type="test" processor="test" priority="1">
         <description>Retry task</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
</tasks></tasks>
</response>`,
			Usage: &core.TokenInfo{},
		}

		// Setup expectations
		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(analyzerResp, nil)
		mockParser.On("Parse", mock.Anything).Return(tasks, nil)
		mockPlanner.On("CreatePlan", tasks).Return([][]Task{tasks}, nil)

		failureErr := errors.New("temporary failure")
		// Fail twice, succeed on third attempt
		mockProcessor.On("Process", mock.Anything, task, mock.Anything).
			Return(nil, failureErr).Once()
		mockProcessor.On("Process", mock.Anything, task, mock.Anything).
			Return(nil, failureErr).Once()
		mockProcessor.On("Process", mock.Anything, task, mock.Anything).
			Return("success", nil).Once()

		// Execute with proper context
		ctx := setupTestContext()
		result, err := orchestrator.Process(ctx, "Test retry logic", nil)

		// Verify
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Contains(t, result.CompletedTasks, "retry-task")
		assert.Empty(t, result.FailedTasks)
		mockProcessor.AssertNumberOfCalls(t, "Process", 3)
	})

	t.Run("Error Handling", func(t *testing.T) {
		testCases := []struct {
			name          string
			setupMocks    func(mockLLM *testutil.MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor)
			expectError   bool
			errorContains string
		}{
			{
				name: "Analyzer Error",
				setupMocks: func(mockLLM *testutil.MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
						Return(nil, errors.New("analyzer error"))
				},
				expectError:   true,
				errorContains: "analyzer error",
			},
			{
				name: "Parser Error",
				setupMocks: func(mockLLM *testutil.MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
						Return(&core.LLMResponse{Content: `<response><analysis>Task analyzed</analysis><tasks>tasks content</tasks></response>`, Usage: &core.TokenInfo{}}, nil)
					mockParser.On("Parse", mock.Anything).
						Return([]Task{}, errors.New("parser error"))
				},
				expectError:   true,
				errorContains: "parser error",
			},

			{
				name: "Plan Creation Error",
				setupMocks: func(mockLLM *testutil.MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					// Set up a valid analyzer response
					resp := &core.LLMResponse{
						Content: "<response><analysis>Task analyzed</analysis><tasks><tasks><task>test task</task></tasks></tasks></response>",
						Usage:   &core.TokenInfo{},
					}
					mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
						Return(resp, nil)

					// Parser returns valid tasks
					tasks := []Task{{
						ID:            "task1",
						Type:          "test",
						ProcessorType: "test",
					}}
					mockParser.On("Parse", mock.Anything).
						Return(tasks, nil)

					// Planner returns error
					mockPlanner.On("CreatePlan", tasks).
						Return([][]Task{}, errors.New("planning error")).Once()
				},
				expectError:   true,
				errorContains: "planning error",
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()
				tc.setupMocks(mockLLM, mockParser, mockPlanner, mockProcessor)

				ctx := setupTestContext()
				result, err := orchestrator.Process(ctx, "Test error handling", nil)
				if tc.expectError {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tc.errorContains,
						"error message should contain expected content")

					// If we got a result with the error, verify it contains error info
					if result != nil {
						assert.Contains(t, result.Metadata["error"], tc.errorContains)
					}
				} else {
					assert.NoError(t, err)
					assert.NotNil(t, result)
				}
				// Clean up mock expectations
				mockLLM.AssertExpectations(t)
				mockParser.AssertExpectations(t)
				mockPlanner.AssertExpectations(t)
				mockProcessor.AssertExpectations(t)
			})
		}
	})

	t.Run("Context Cancellation", func(t *testing.T) {
		orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()

		// Test data
		tasks := []Task{{ID: "task1", Type: "test", ProcessorType: "test"}}

		// Setup expectations
		mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
			Return(&core.LLMResponse{Content: "<response><analysis>Task analyzed</analysis><tasks><tasks><task>test task</task></tasks></tasks></response>", Usage: &core.TokenInfo{}}, nil)
		mockParser.On("Parse", mock.Anything).Return(tasks, nil)
		mockPlanner.On("CreatePlan", tasks).Return([][]Task{tasks}, nil)

		// Create a cancellable context
		ctx, cancel := context.WithCancel(setupTestContext())

		// Cancel after a short delay
		go func() {
			time.Sleep(100 * time.Millisecond)
			cancel()
		}()

		// Mock processor to block until context cancellation
		mockProcessor.On("Process", mock.Anything, tasks[0], mock.Anything).
			Run(func(args mock.Arguments) {
				<-ctx.Done()
			}).Return(nil, context.Canceled)

		// Execute
		_, err := orchestrator.Process(ctx, "Test context cancellation", nil)

		// Verify
		assert.Error(t, err)
		assert.True(t, errors.Is(err, context.Canceled))
	})
}
