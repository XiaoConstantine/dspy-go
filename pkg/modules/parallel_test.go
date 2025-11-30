package modules

import (
	"context"
	stderrors "errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockModule is a test module that can be configured to succeed/fail.
type MockModule struct {
	core.BaseModule
	shouldFail     bool
	processingTime time.Duration
	callCount      int
	mu             sync.Mutex
}

func NewMockModule(shouldFail bool, processingTime time.Duration) *MockModule {
	signature := core.Signature{
		Inputs: []core.InputField{
			{Field: core.Field{Name: "input"}},
		},
		Outputs: []core.OutputField{
			{Field: core.Field{Name: "output"}},
		},
	}

	return &MockModule{
		BaseModule:     *core.NewModule(signature),
		shouldFail:     shouldFail,
		processingTime: processingTime,
	}
}

func (m *MockModule) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	m.mu.Lock()
	m.callCount++
	count := m.callCount
	m.mu.Unlock()

	// Simulate processing time
	if m.processingTime > 0 {
		time.Sleep(m.processingTime)
	}

	if m.shouldFail {
		return nil, fmt.Errorf("mock module failed for call %d", count)
	}

	input, ok := inputs["input"].(string)
	if !ok {
		return nil, stderrors.New("invalid input type")
	}

	return map[string]interface{}{
		"output": fmt.Sprintf("processed_%s_call_%d", input, count),
	}, nil
}

func (m *MockModule) Clone() core.Module {
	return &MockModule{
		BaseModule:     *m.BaseModule.Clone().(*core.BaseModule),
		shouldFail:     m.shouldFail,
		processingTime: m.processingTime,
	}
}

func (m *MockModule) GetCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

func TestNewParallel(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule)

	assert.NotNil(t, parallel)
	assert.Equal(t, mockModule.GetSignature(), parallel.GetSignature())
	assert.Equal(t, mockModule, parallel.GetInnerModule())
}

func TestParallelWithOptions(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule,
		WithMaxWorkers(2),
		WithReturnFailures(true),
		WithStopOnFirstError(true),
	)

	assert.Equal(t, 2, parallel.options.MaxWorkers)
	assert.True(t, parallel.options.ReturnFailures)
	assert.True(t, parallel.options.StopOnFirstError)
}

func TestParallelSingleInput(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule)

	ctx := context.Background()
	inputs := map[string]interface{}{
		"input": "test",
	}

	result, err := parallel.Process(ctx, inputs)
	require.NoError(t, err)

	assert.Equal(t, "processed_test_call_1", result["output"])
}

func TestParallelBatchProcessing(t *testing.T) {
	mockModule := NewMockModule(false, 10*time.Millisecond)
	parallel := NewParallel(mockModule, WithMaxWorkers(3))

	ctx := context.Background()

	// Create batch inputs
	batchInputs := []map[string]interface{}{
		{"input": "item1"},
		{"input": "item2"},
		{"input": "item3"},
		{"input": "item4"},
		{"input": "item5"},
	}

	inputs := map[string]interface{}{
		"batch_inputs": batchInputs,
	}

	start := time.Now()
	result, err := parallel.Process(ctx, inputs)
	duration := time.Since(start)

	require.NoError(t, err)

	results, ok := result["results"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, results, 5)

	// Verify all results are present and processed
	processedItems := make(map[string]bool)
	for _, res := range results {
	output, ok := res["output"].(string)
	require.True(t, ok)

	// Extract the original input from the output (format: "processed_itemX_call_Y")
	parts := strings.SplitN(strings.TrimPrefix(output, "processed_"), "_", 2)
	if len(parts) > 0 {
	processedItems[parts[0]] = true
	}
	}
	assert.Len(t, processedItems, len(batchInputs), "should have processed all unique items")

	// Check that parallel processing was faster than sequential
	// With 3 workers and 5 items taking 10ms each, should be faster than 50ms
	assert.Less(t, duration, 40*time.Millisecond, "Parallel processing should be faster than sequential")

	// Verify all calls were made
	assert.Equal(t, 5, mockModule.GetCallCount())
}

func TestParallelWithFailures(t *testing.T) {
	mockModule := NewMockModule(true, 0) // Always fails
	parallel := NewParallel(mockModule, WithReturnFailures(true))

	ctx := context.Background()
	batchInputs := []map[string]interface{}{
		{"input": "item1"},
		{"input": "item2"},
	}

	inputs := map[string]interface{}{
		"batch_inputs": batchInputs,
	}

	result, err := parallel.Process(ctx, inputs)
	require.NoError(t, err)

	results, ok := result["results"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, results, 2) // Same length as input, but with nil values

	// Check that failed results are nil
	for _, res := range results {
		assert.Nil(t, res)
	}

	failures, ok := result["failures"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, failures, 2) // Two failures

	// Check failure details
	for i, failure := range failures {
		assert.Equal(t, i, failure["index"])
		assert.Contains(t, failure["error"].(string), "mock module failed")
	}
}

func TestParallelStopOnFirstError(t *testing.T) {
	mockModule := NewMockModule(true, 0) // Always fails
	parallel := NewParallel(mockModule, WithStopOnFirstError(true))

	ctx := context.Background()
	batchInputs := []map[string]interface{}{
		{"input": "item1"},
		{"input": "item2"},
		{"input": "item3"},
	}

	inputs := map[string]interface{}{
		"batch_inputs": batchInputs,
	}

	result, err := parallel.Process(ctx, inputs)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "parallel execution failed")
}

func TestParallelMixedSuccessFailure(t *testing.T) {
	// Create a module that succeeds for item1,item3 and fails for item2,item4
	mockModule := &ConditionalModule{
		BaseModule: *core.NewModule(core.Signature{
			Inputs: []core.InputField{
				{Field: core.Field{Name: "input"}},
			},
			Outputs: []core.OutputField{
				{Field: core.Field{Name: "output"}},
			},
		}),
		failInputs: map[string]bool{"item2": true, "item4": true},
	}

	parallel := NewParallel(mockModule, WithReturnFailures(true))

	ctx := context.Background()
	batchInputs := []map[string]interface{}{
		{"input": "item1"}, // Will succeed
		{"input": "item2"}, // Will fail
		{"input": "item3"}, // Will succeed
		{"input": "item4"}, // Will fail
	}

	inputs := map[string]interface{}{
		"batch_inputs": batchInputs,
	}

	result, err := parallel.Process(ctx, inputs)
	require.NoError(t, err)

	results, ok := result["results"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, results, 4) // Same length as input batch

	// Check results - should have successes for item1,item3 (indices 0,2) and nil for item2,item4 (indices 1,3)
	successCount := 0
	nilCount := 0
	for i, res := range results {
		if i == 0 || i == 2 { // item1, item3 should succeed
			assert.NotNil(t, res, "Expected success at index %d", i)
			if res != nil {
				successCount++
				assert.Contains(t, res["output"].(string), fmt.Sprintf("item%d", i+1))
			}
		} else { // item2, item4 should fail
			assert.Nil(t, res, "Expected failure at index %d", i)
			nilCount++
		}
	}
	assert.Equal(t, 2, successCount)
	assert.Equal(t, 2, nilCount)

	failures, ok := result["failures"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, failures, 2) // Two failures

	// Check that failure indices are correct
	for _, failure := range failures {
		idx := failure["index"].(int)
		assert.True(t, idx == 1 || idx == 3, "Failure index should be 1 or 3, got %d", idx)
	}
}

func TestParallelEmptyBatch(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule)

	ctx := context.Background()
	inputs := map[string]interface{}{
		"batch_inputs": []map[string]interface{}{},
	}

	result, err := parallel.Process(ctx, inputs)
	require.NoError(t, err)

	results, ok := result["results"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, results, 0)
}

func TestParallelClone(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule, WithMaxWorkers(2))

	cloned := parallel.Clone().(*Parallel)

	assert.NotSame(t, parallel, cloned)
	assert.Equal(t, parallel.options.MaxWorkers, cloned.options.MaxWorkers)
	assert.NotSame(t, parallel.innerModule, cloned.innerModule)
}

func TestParallelSetLLM(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule)

	// Create a mock LLM
	mockLLM := &MockLLM{}

	parallel.SetLLM(mockLLM)

	// Both the parallel module and inner module should have the LLM set
	assert.Equal(t, mockLLM, parallel.LLM)
	assert.Equal(t, mockLLM, parallel.innerModule.(*MockModule).LLM)
}

// ConditionalModule succeeds or fails based on the input value.
type ConditionalModule struct {
	core.BaseModule
	failInputs map[string]bool
	callCount  int
	mu         sync.Mutex
}

func (m *ConditionalModule) Process(ctx context.Context, inputs map[string]interface{}, opts ...core.Option) (map[string]interface{}, error) {
	m.mu.Lock()
	m.callCount++
	count := m.callCount
	m.mu.Unlock()

	input, ok := inputs["input"].(string)
	if !ok {
		return nil, stderrors.New("invalid input type")
	}

	// Fail if input is in the fail list
	if m.failInputs[input] {
		return nil, fmt.Errorf("conditional module failed for input %s", input)
	}

	return map[string]interface{}{
		"output": fmt.Sprintf("processed_%s_call_%d", input, count),
	}, nil
}

func (m *ConditionalModule) Clone() core.Module {
	return &ConditionalModule{
		BaseModule: *m.BaseModule.Clone().(*core.BaseModule),
		failInputs: m.failInputs,
	}
}

// MockLLM for testing.
type MockLLM struct{}

func (m *MockLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock response"}, nil
}

func (m *MockLLM) StreamGenerate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, stderrors.New("not implemented")
}

func (m *MockLLM) ProviderName() string {
	return "mock"
}

func (m *MockLLM) ModelID() string {
	return "mock-model"
}

func (m *MockLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, stderrors.New("not implemented")
}

func (m *MockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, stderrors.New("not implemented")
}

func (m *MockLLM) CreateEmbedding(ctx context.Context, input string, opts ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, stderrors.New("not implemented")
}

func (m *MockLLM) CreateEmbeddings(ctx context.Context, inputs []string, opts ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, stderrors.New("not implemented")
}

func (m *MockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	// Convert content blocks to a simple text representation for mock response
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}
	return &core.LLMResponse{Content: "mock response for content: " + textContent}, nil
}

func (m *MockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	// Create a simple mock stream response
	chunkChan := make(chan core.StreamChunk, 1)
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}

	chunkChan <- core.StreamChunk{
		Content: "mock stream response for content: " + textContent,
		Done:    true,
	}
	close(chunkChan)

	return &core.StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       func() {},
	}, nil
}

func TestParallelConcurrencyControl(t *testing.T) {
	// Test that we don't exceed the worker limit
	mockModule := NewMockModule(false, 50*time.Millisecond)
	maxWorkers := 2
	parallel := NewParallel(mockModule, WithMaxWorkers(maxWorkers))

	ctx := context.Background()
	batchInputs := make([]map[string]interface{}, 10)
	for i := 0; i < 10; i++ {
		batchInputs[i] = map[string]interface{}{
			"input": fmt.Sprintf("item%d", i),
		}
	}

	inputs := map[string]interface{}{
		"batch_inputs": batchInputs,
	}

	start := time.Now()
	result, err := parallel.Process(ctx, inputs)
	duration := time.Since(start)

	require.NoError(t, err)

	results, ok := result["results"].([]map[string]interface{})
	require.True(t, ok)
	assert.Len(t, results, 10)

	// With 2 workers and 10 items taking 50ms each, minimum time should be around 5*50ms = 250ms
	// Allow some tolerance for test execution
	minExpectedDuration := 200 * time.Millisecond
	assert.GreaterOrEqual(t, duration, minExpectedDuration,
		"Duration should respect worker limit: expected >= %v, got %v", minExpectedDuration, duration)
}

// TestEffectiveWorkers_ExplicitOverride tests that explicit MaxWorkers is respected.
func TestEffectiveWorkers_ExplicitOverride(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule, WithMaxWorkers(50))

	assert.Equal(t, 50, parallel.effectiveWorkers())
}

// TestEffectiveWorkers_Default tests default workers (100 for I/O-bound workloads).
func TestEffectiveWorkers_Default(t *testing.T) {
	mockModule := NewMockModule(false, 0)
	parallel := NewParallel(mockModule)

	// Should default to 100 workers (optimized for I/O-bound workloads)
	assert.Equal(t, 100, parallel.effectiveWorkers())
}
