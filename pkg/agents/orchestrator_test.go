package agents

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockLLM implements core.LLM for testing.
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	// Return nil, error if that's what was set up
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	// Otherwise return the response
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

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

func setupTestOrchestrator() (*FlexibleOrchestrator, *MockTaskProcessor, *MockTaskParser, *MockPlanCreator, *MockLLM) {
	memory := NewInMemoryStore()
	mockProcessor := new(MockTaskProcessor)
	mockParser := new(MockTaskParser)
	mockPlanner := new(MockPlanCreator)
	mockLLM := new(MockLLM)

	// Set up default LLM for the analyzer
	config.GlobalConfig.DefaultLLM = mockLLM

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
			Content: `analysis:Task has been analyzed and broken down into atomic units
tasks:<tasks>
     <task id="task1" type="test" processor="test" priority="1">
         <description>Test task 1</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
`,
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
			Content: `analysis: Task has been analyzed and decomposed for parallel execution
 
 tasks: <tasks>
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
 </tasks>
`,
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
			Content: `analysis: Task has been analyzed for retry testing
 
 tasks: <tasks>
     <task id="retry-task" type="test" processor="test" priority="1">
         <description>Retry task</description>
         <metadata>
             <item key="key">value</item>
         </metadata>
     </task>
 </tasks>
`,
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
			name string

			setupMocks func(mockLLM *MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor)
			checkErr   func(*testing.T, error)
		}{
			{
				name: "Analyzer Error",
				setupMocks: func(mockLLM *MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
						Return(nil, errors.New("analyzer error"))
				},
				checkErr: func(t *testing.T, err error) {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), "task analysis failed")
				},
			},
			{
				name: "Parser Error",
				setupMocks: func(mockLLM *MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).
						Return(&core.LLMResponse{Content: `{"analysis": "Task analyzed"}`, Usage: &core.TokenInfo{}}, nil)
					mockParser.On("Parse", mock.Anything).
						Return([]Task{}, errors.New("parser error"))
				},
				checkErr: func(t *testing.T, err error) {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), "task parsing failed")
				},
			},

			{
				name: "Plan Creation Error",
				setupMocks: func(mockLLM *MockLLM, mockParser *MockTaskParser, mockPlanner *MockPlanCreator, mockProcessor *MockTaskProcessor) {
					// Set up a valid analyzer response
					resp := &core.LLMResponse{
						Content: "analysis: Task analyzed\ntasks: <tasks><task>test task</task></tasks>",
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
				checkErr: func(t *testing.T, err error) {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), "Plan failed")
				},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				orchestrator, mockProcessor, mockParser, mockPlanner, mockLLM := setupTestOrchestrator()
				tc.setupMocks(mockLLM, mockParser, mockPlanner, mockProcessor)

				ctx := setupTestContext()
				result, err := orchestrator.Process(ctx, "Test error handling", nil)
				tc.checkErr(t, err)
				assert.Nil(t, result)
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
			Return(&core.LLMResponse{Content: "analysis: Task analyzed\ntasks: <tasks><task>test task</task></tasks>", Usage: &core.TokenInfo{}}, nil)
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
