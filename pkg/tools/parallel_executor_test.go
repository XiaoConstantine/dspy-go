package tools

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParallelExecutor_Basic(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 4)
	
	tasks := []*ParallelTask{
		{
			ID:         "task1",
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": "test1"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "task2",
			ToolName:   "validator",
			Input:      map[string]interface{}{"data": "test2"},
			Priority:   2,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, tasks, nil)
	
	require.NoError(t, err)
	assert.Len(t, results, 2)
	
	// Check that both tasks completed successfully
	for _, result := range results {
		assert.NoError(t, result.Error)
		assert.NotEmpty(t, result.TaskID)
		assert.Greater(t, result.Duration, time.Duration(0))
	}
}

func TestParallelExecutor_PriorityScheduling(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 1) // Single worker to test ordering
	
	// Create tasks with different priorities
	tasks := []*ParallelTask{
		{
			ID:         "low_priority",
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": "low"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "high_priority",
			ToolName:   "validator",
			Input:      map[string]interface{}{"data": "high"},
			Priority:   5,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "medium_priority",
			ToolName:   "transformer",
			Input:      map[string]interface{}{"data": "medium"},
			Priority:   3,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
	}
	
	ctx := context.Background()
	scheduler := &PriorityScheduler{}
	results, err := executor.ExecuteParallel(ctx, tasks, scheduler)
	
	require.NoError(t, err)
	assert.Len(t, results, 3)
	
	// Verify all tasks completed
	for _, result := range results {
		assert.NoError(t, result.Error)
	}
	
	// With single worker, execution should follow priority order
	// (though we can't easily test exact order due to goroutine scheduling)
}

func TestParallelExecutor_FairShareScheduling(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 2)
	
	scheduler := NewFairShareScheduler()
	
	// Create multiple tasks for the same tool
	tasks := []*ParallelTask{
		{
			ID:         "parser1",
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": "test1"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "parser2",
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": "test2"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "validator1",
			ToolName:   "validator",
			Input:      map[string]interface{}{"data": "test1"},
			Priority:   1,
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, tasks, scheduler)
	
	require.NoError(t, err)
	assert.Len(t, results, 3)
	
	for _, result := range results {
		assert.NoError(t, result.Error)
	}
}

func TestParallelExecutor_ErrorHandling(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 2)
	
	tasks := []*ParallelTask{
		{
			ID:         "success_task",
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": "test"},
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
		{
			ID:         "error_task",
			ToolName:   "error_tool",
			Input:      map[string]interface{}{"data": "test"},
			Context:    context.Background(),
			SubmitTime: time.Now(),
		},
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, tasks, nil)
	
	require.NoError(t, err) // ExecuteParallel itself shouldn't error
	assert.Len(t, results, 2)
	
	// Check individual results
	var successResult, errorResult ParallelResult
	for _, result := range results {
		if result.TaskID == "success_task" {
			successResult = result
		} else {
			errorResult = result
		}
	}
	
	assert.NoError(t, successResult.Error)
	assert.Error(t, errorResult.Error)
	assert.Contains(t, errorResult.Error.Error(), "simulated error")
}

func TestParallelExecutor_Retries(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 1)
	
	task := &ParallelTask{
		ID:         "retry_task",
		ToolName:   "error_tool",
		Input:      map[string]interface{}{"data": "test"},
		Retries:    3,
		Context:    context.Background(),
		SubmitTime: time.Now(),
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, []*ParallelTask{task}, nil)
	
	require.NoError(t, err)
	assert.Len(t, results, 1)
	
	result := results[0]
	assert.Error(t, result.Error) // Should still fail after retries
	assert.Equal(t, 3, result.Retries) // Should have attempted 3 retries
}

func TestParallelExecutor_Timeout(t *testing.T) {
	registry := createTestRegistry()
	// Add a slow tool
	slowTool := &mockProcessingTool{
		name:  "slow_tool",
		delay: 200 * time.Millisecond,
	}
	_ = registry.Register(slowTool)
	
	executor := NewParallelExecutor(registry, 1)
	
	task := &ParallelTask{
		ID:         "timeout_task",
		ToolName:   "slow_tool",
		Input:      map[string]interface{}{"data": "test"},
		Timeout:    5 * time.Millisecond, // Much shorter than tool delay
		Context:    context.Background(),
		SubmitTime: time.Now(),
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, []*ParallelTask{task}, nil)
	
	require.NoError(t, err)
	assert.Len(t, results, 1)
	
	result := results[0]
	assert.Error(t, result.Error)
	// Should be a timeout error
	if result.Error != nil {
		assert.Contains(t, result.Error.Error(), "context deadline exceeded")
	}
}

func TestParallelExecutor_ContextCancellation(t *testing.T) {
	registry := createTestRegistry()
	
	// Add a very slow tool that will definitely be running when we cancel
	slowTool := &mockProcessingTool{
		name:  "very_slow_tool",
		delay: 1 * time.Second,
	}
	_ = registry.Register(slowTool)
	
	executor := NewParallelExecutor(registry, 1)
	
	task := &ParallelTask{
		ID:         "cancel_task",
		ToolName:   "very_slow_tool",
		Input:      map[string]interface{}{"data": "test"},
		Context:    context.Background(),
		SubmitTime: time.Now(),
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	results, err := executor.ExecuteParallel(ctx, []*ParallelTask{task}, nil)
	
	// When context is canceled, either the main call returns an error
	// OR individual tasks have cancellation errors
	if err != nil {
		assert.Equal(t, context.Canceled, err)
	} else {
		// If no main error, check individual task results
		require.Len(t, results, 1)
		result := results[0]
		assert.Error(t, result.Error)
		assert.Equal(t, context.Canceled, result.Error)
	}
}

func TestParallelExecutor_WorkerPoolManagement(t *testing.T) {
	registry := createTestRegistry()
	maxWorkers := 3
	executor := NewParallelExecutor(registry, maxWorkers)
	
	// Check initial status
	total, active, available := executor.GetWorkerPoolStatus()
	assert.Equal(t, maxWorkers, total)
	assert.Equal(t, 0, active)
	assert.Equal(t, maxWorkers, available)
	
	// Create more tasks than workers
	tasks := make([]*ParallelTask, 5)
	for i := range tasks {
		tasks[i] = &ParallelTask{
			ID:         fmt.Sprintf("task_%d", i),
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": fmt.Sprintf("test%d", i)},
			Context:    context.Background(),
			SubmitTime: time.Now(),
		}
	}
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, tasks, nil)
	
	require.NoError(t, err)
	assert.Len(t, results, 5)
	
	// All tasks should complete successfully
	for _, result := range results {
		assert.NoError(t, result.Error)
		assert.Greater(t, result.WaitTime, time.Duration(0)) // Some tasks should have waited
	}
	
	// Check metrics
	metrics := executor.GetMetrics()
	assert.Equal(t, int64(5), metrics.TotalExecutions)
	assert.Equal(t, int64(5), metrics.ParallelTasks)
	assert.Greater(t, metrics.AverageExecTime, time.Duration(0))
}

func TestParallelExecutor_Metrics(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 2)
	
	task := &ParallelTask{
		ID:         "metrics_task",
		ToolName:   "parser",
		Input:      map[string]interface{}{"data": "test"},
		Context:    context.Background(),
		SubmitTime: time.Now(),
	}
	
	ctx := context.Background()
	_, err := executor.ExecuteParallel(ctx, []*ParallelTask{task}, nil)
	require.NoError(t, err)
	
	metrics := executor.GetMetrics()
	assert.Equal(t, int64(1), metrics.TotalExecutions)
	assert.Equal(t, int64(1), metrics.ParallelTasks)
	assert.Greater(t, metrics.AverageExecTime, time.Duration(0))
	assert.GreaterOrEqual(t, metrics.AverageWaitTime, time.Duration(0))
}

func TestBatchExecutor(t *testing.T) {
	registry := createTestRegistry()
	parallelExecutor := NewParallelExecutor(registry, 3)
	batchExecutor := NewBatchExecutor(parallelExecutor, &PriorityScheduler{})
	
	calls := []ToolCall{
		{
			ToolName: "parser",
			Input:    map[string]interface{}{"data": "test1"},
			Priority: 1,
			Timeout:  1 * time.Second,
		},
		{
			ToolName: "validator",
			Input:    map[string]interface{}{"data": "test2"},
			Priority: 2,
			Timeout:  1 * time.Second,
		},
		{
			ToolName: "transformer",
			Input:    map[string]interface{}{"data": "test3"},
			Priority: 3,
			Timeout:  1 * time.Second,
		},
	}
	
	ctx := context.Background()
	results, err := batchExecutor.ExecuteBatch(ctx, calls)
	
	require.NoError(t, err)
	assert.Len(t, results, 3)
	
	for _, result := range results {
		assert.NoError(t, result.Error)
		assert.Contains(t, result.TaskID, "batch_")
	}
}

func TestParallelPipelineExecutor(t *testing.T) {
	registry := createTestRegistry()
	
	// Create a basic pipeline
	options := PipelineOptions{
		Parallel: false, // We'll handle parallelism at executor level
	}
	pipeline := NewToolPipeline("parallel-test", registry, options)
	
	err := pipeline.AddStep(PipelineStep{ToolName: "parser"})
	require.NoError(t, err)
	err = pipeline.AddStep(PipelineStep{ToolName: "validator"})
	require.NoError(t, err)
	err = pipeline.AddStep(PipelineStep{ToolName: "transformer"})
	require.NoError(t, err)
	
	parallelExecutor := NewParallelExecutor(registry, 3)
	pipelineExecutor := NewParallelPipelineExecutor(pipeline, parallelExecutor)
	
	ctx := context.Background()
	input := map[string]interface{}{
		"data": "test input",
	}
	
	result, err := pipelineExecutor.ExecuteWithParallelSteps(ctx, input)
	require.NoError(t, err)
	assert.True(t, result.Success)
	assert.Len(t, result.Results, 3)
}

func TestParallelExecutor_ConcurrentSafety(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 5)
	
	// Run multiple concurrent executions
	var wg sync.WaitGroup
	numGoroutines := 10
	resultsChannel := make(chan []ParallelResult, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			tasks := []*ParallelTask{
				{
					ID:         fmt.Sprintf("concurrent_task_%d", id),
					ToolName:   "parser",
					Input:      map[string]interface{}{"data": fmt.Sprintf("test%d", id)},
					Context:    context.Background(),
					SubmitTime: time.Now(),
				},
			}
			
			ctx := context.Background()
			results, err := executor.ExecuteParallel(ctx, tasks, nil)
			assert.NoError(t, err)
			resultsChannel <- results
		}(i)
	}
	
	wg.Wait()
	close(resultsChannel)
	
	// Verify all executions completed successfully
	totalResults := 0
	for results := range resultsChannel {
		assert.Len(t, results, 1)
		assert.NoError(t, results[0].Error)
		totalResults++
	}
	
	assert.Equal(t, numGoroutines, totalResults)
	
	// Check final metrics
	metrics := executor.GetMetrics()
	assert.Equal(t, int64(numGoroutines), metrics.TotalExecutions)
}

func TestParallelExecutor_EmptyTaskList(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 2)
	
	ctx := context.Background()
	results, err := executor.ExecuteParallel(ctx, []*ParallelTask{}, nil)
	
	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestParallelExecutor_LargeTaskLoad(t *testing.T) {
	registry := createTestRegistry()
	executor := NewParallelExecutor(registry, 4)
	
	// Create a large number of tasks
	numTasks := 50
	tasks := make([]*ParallelTask, numTasks)
	
	for i := 0; i < numTasks; i++ {
		tasks[i] = &ParallelTask{
			ID:         fmt.Sprintf("load_task_%d", i),
			ToolName:   "parser",
			Input:      map[string]interface{}{"data": fmt.Sprintf("test%d", i)},
			Priority:   i % 5, // Vary priorities
			Context:    context.Background(),
			SubmitTime: time.Now(),
		}
	}
	
	ctx := context.Background()
	start := time.Now()
	results, err := executor.ExecuteParallel(ctx, tasks, &PriorityScheduler{})
	duration := time.Since(start)
	
	require.NoError(t, err)
	assert.Len(t, results, numTasks)
	
	// All tasks should complete successfully
	successCount := 0
	for _, result := range results {
		if result.Error == nil {
			successCount++
		}
	}
	assert.Equal(t, numTasks, successCount)
	
	// Test that parallel execution works by verifying task completion efficiency
	// With 4 workers and 50 tasks, we should see significant parallelism
	// Instead of timing, verify that all tasks completed and check metrics
	
	// Verify all tasks completed
	assert.Len(t, results, numTasks)
	
	// Verify tasks ran with reasonable parallelism by checking that 
	// duration is much less than sequential (50 * 10ms = 500ms)
	sequentialTime := time.Duration(numTasks) * 10 * time.Millisecond
	parallelismRatio := float64(duration) / float64(sequentialTime)
	
	// With 4 workers, we expect at least 3x speedup (allowing for overhead)
	assert.True(t, parallelismRatio < 0.4, 
		"Expected parallel speedup, got ratio %f (duration: %v vs sequential: %v)", 
		parallelismRatio, duration, sequentialTime)
	
	t.Logf("Executed %d tasks in %v", numTasks, duration)
}