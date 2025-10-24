package react

import (
	"context"
	"fmt"
	"sync"
	"testing"

	contextmgmt "github.com/XiaoConstantine/dspy-go/pkg/agents/context"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReActAgentContextIntegration tests the integration of context engineering with ReAct agents.
func TestReActAgentContextIntegration(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name                 string
		options              []Option
		expectContextEnabled bool
	}{
		{
			name:                 "default configuration - context disabled",
			options:              []Option{},
			expectContextEnabled: false,
		},
		{
			name: "basic context engineering enabled",
			options: []Option{
				WithContextEngineering(tempDir, contextmgmt.DefaultConfig()),
			},
			expectContextEnabled: true,
		},
		{
			name: "production context optimization",
			options: []Option{
				WithProductionContextOptimization(),
			},
			expectContextEnabled: true,
		},
		{
			name: "custom context optimization",
			options: []Option{
				WithContextOptimization(contextmgmt.PriorityHigh, 4096, 0.9),
			},
			expectContextEnabled: false, // Only sets optimization level, doesn't enable
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := NewReActAgent("test-agent", "Test Agent", tt.options...)
			assert.NotNil(t, agent)

			// Check if context engineering is enabled as expected
			assert.Equal(t, tt.expectContextEnabled, agent.config.EnableContextEngineering)

			if tt.expectContextEnabled {
				assert.NotNil(t, agent.contextManager, "Context manager should be initialized when enabled")
			} else {
				assert.Nil(t, agent.contextManager, "Context manager should be nil when disabled")
			}
		})
	}
}

func TestReActAgentWithContextExecution(t *testing.T) {
	tempDir := t.TempDir()

	agent := NewReActAgent(
		"context-test-agent",
		"Context Test Agent",
		WithProductionContextOptimization(),
		WithContextEngineering(tempDir, contextmgmt.ProductionConfig()),
	)

	// Initialize with mock LLM
	mockLLM := &TestMockLLM{
		responses: []string{
			"I'll help you with this task using optimized context.",
			"Task completed successfully with context engineering benefits.",
		},
	}

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("The task to complete efficiently"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The completed task result"))},
		},
	).WithInstruction("Complete the given task efficiently")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	ctx := context.Background()

	// Test execution with context optimization
	input := map[string]interface{}{
		"task": "Analyze the benefits of context engineering",
		"observations": []string{
			"Context engineering can reduce costs by 10x",
			"KV-cache optimization is crucial for cost reduction",
			"Error learning improves agent reliability",
		},
	}

	result, err := agent.Execute(ctx, input)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Verify context engineering metrics
	metrics := agent.GetContextPerformanceMetrics()
	assert.True(t, metrics["context_mgmt_enabled"].(bool))
	assert.Greater(t, metrics["total_executions"].(int64), int64(0))
	assert.Greater(t, metrics["context_version"].(int64), int64(0))
}

func TestContextPerformanceMetrics(t *testing.T) {
	tempDir := t.TempDir()

	agent := NewReActAgent(
		"metrics-test-agent",
		"Metrics Test Agent",
		WithProductionContextOptimization(),
		WithContextEngineering(tempDir, contextmgmt.DefaultConfig()),
	)

	mockLLM := &TestMockLLM{responses: []string{"Test response"}}
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("input", core.WithDescription("Test input"))},
		},
		[]core.OutputField{
			{Field: core.NewField("output", core.WithDescription("Test output"))},
		},
	).WithInstruction("Test task")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	ctx := context.Background()

	// Execute multiple tasks to generate metrics
	for i := 0; i < 3; i++ {
		input := map[string]interface{}{
			"input": "test input",
		}

		_, err := agent.Execute(ctx, input)
		assert.NoError(t, err)
	}

	// Test performance metrics
	metrics := agent.GetContextPerformanceMetrics()
	assert.NotNil(t, metrics)

	// Check required metric fields
	requiredFields := []string{
		"total_executions",
		"context_version",
		"context_savings",
		"avg_processing_time",
		"context_mgmt_enabled",
	}

	for _, field := range requiredFields {
		assert.Contains(t, metrics, field, "Missing required metric field: %s", field)
	}

	// Verify execution count
	assert.Equal(t, int64(3), metrics["total_executions"])
	assert.True(t, metrics["context_mgmt_enabled"].(bool))

	// Test health status
	health := agent.GetContextHealthStatus()
	assert.NotNil(t, health)
	assert.Contains(t, health, "overall_status")
}

func TestContextHealthStatus(t *testing.T) {
	tests := []struct {
		name           string
		enableContext  bool
		expectedStatus interface{}
	}{
		{
			name:           "context disabled",
			enableContext:  false,
			expectedStatus: "disabled",
		},
		{
			name:           "context enabled",
			enableContext:  true,
			expectedStatus: nil, // Will depend on actual health check
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var agent *ReActAgent
			if tt.enableContext {
				tempDir := t.TempDir()
				agent = NewReActAgent(
					"health-test-agent",
					"Health Test Agent",
					WithContextEngineering(tempDir, contextmgmt.DefaultConfig()),
				)
			} else {
				agent = NewReActAgent("health-test-agent", "Health Test Agent")
			}

			health := agent.GetContextHealthStatus()
			assert.NotNil(t, health)

			if tt.expectedStatus != nil {
				assert.Equal(t, tt.expectedStatus, health["status"])
			}
		})
	}
}

func TestErrorRecordingAndLearning(t *testing.T) {
	tempDir := t.TempDir()

	config := contextmgmt.DefaultConfig()
	config.EnableErrorRetention = true

	agent := NewReActAgent(
		"error-test-agent",
		"Error Test Agent",
		WithContextEngineering(tempDir, config),
		WithContextOptimization(contextmgmt.PriorityMedium, 2048, 0.8),
	)

	mockLLM := &TestMockLLM{
		responses:   []string{"Success response"},
		shouldError: false,
	}

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("The task to test"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The result"))},
		},
	).WithInstruction("Test error handling")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	ctx := context.Background()

	// Test successful execution first
	input := map[string]interface{}{
		"task": "successful task",
	}

	result, err := agent.Execute(ctx, input)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Simulate an error in the mock LLM
	mockLLM.shouldError = true
	mockLLM.errorMessage = "simulated LLM failure"

	input = map[string]interface{}{
		"task": "failing task",
	}

	_, err = agent.Execute(ctx, input)
	assert.Error(t, err) // Should fail due to mock error

	// Check that error was recorded (if context management is working)
	metrics := agent.GetContextPerformanceMetrics()
	if metrics["context_mgmt_enabled"].(bool) {
		// Error should be recorded in context manager
		assert.Greater(t, metrics["total_executions"].(int64), int64(1))
	}
}

func TestTodoManagement(t *testing.T) {
	tempDir := t.TempDir()

	config := contextmgmt.DefaultConfig()
	config.EnableTodoManagement = true

	agent := NewReActAgent(
		"todo-test-agent",
		"Todo Test Agent",
		WithContextEngineering(tempDir, config),
	)

	mockLLM := &TestMockLLM{responses: []string{"Task completed"}}
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("objective", core.WithDescription("The objective to manage"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The result"))},
		},
	).WithInstruction("Test todo management")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	ctx := context.Background()

	// Execute task with objective that should be tracked
	input := map[string]interface{}{
		"objective": "Implement feature X with proper error handling",
	}

	result, err := agent.Execute(ctx, input)
	assert.NoError(t, err)
	assert.NotNil(t, result)

	// Verify execution history contains context information
	history := agent.GetExecutionHistory()
	assert.Len(t, history, 1)

	lastExecution := history[0]
	assert.True(t, lastExecution.Success)
	assert.Greater(t, lastExecution.ContextVersion, int64(0))
}

func TestConfigurationOptions(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name   string
		option Option
		verify func(t *testing.T, config ReActAgentConfig)
	}{
		{
			name:   "WithContextEngineering",
			option: WithContextEngineering(tempDir, contextmgmt.ProductionConfig()),
			verify: func(t *testing.T, config ReActAgentConfig) {
				assert.True(t, config.EnableContextEngineering)
				assert.Equal(t, tempDir, config.ContextBaseDir)
				assert.Equal(t, contextmgmt.ProductionConfig().SessionID, config.ContextConfig.SessionID)
			},
		},
		{
			name:   "WithProductionContextOptimization",
			option: WithProductionContextOptimization(),
			verify: func(t *testing.T, config ReActAgentConfig) {
				assert.True(t, config.EnableContextEngineering)
				assert.Equal(t, contextmgmt.PriorityHigh, config.ContextOptLevel)
				assert.Equal(t, float64(0.95), config.CacheEfficiencyTarget)
				assert.True(t, config.AutoTodoManagement)
				assert.True(t, config.AutoErrorLearning)
			},
		},
		{
			name:   "WithContextOptimization",
			option: WithContextOptimization(contextmgmt.PriorityLow, 1024, 0.7),
			verify: func(t *testing.T, config ReActAgentConfig) {
				assert.Equal(t, contextmgmt.PriorityLow, config.ContextOptLevel)
				assert.Equal(t, 1024, config.MaxContextTokens)
				assert.Equal(t, float64(0.7), config.CacheEfficiencyTarget)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := DefaultReActAgentConfig()
			tt.option(&config)
			tt.verify(t, config)
		})
	}
}

func TestConcurrentExecution(t *testing.T) {
	tempDir := t.TempDir()

	agent := NewReActAgent(
		"concurrent-test-agent",
		"Concurrent Test Agent",
		WithProductionContextOptimization(),
		WithContextEngineering(tempDir, contextmgmt.DefaultConfig()),
	)

	mockLLM := &TestMockLLM{responses: []string{"Concurrent response"}}
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("The concurrent task"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The result"))},
		},
	).WithInstruction("Concurrent test")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(t, err)

	ctx := context.Background()

	// Run concurrent executions to test thread safety
	const numGoroutines = 5
	const numExecutions = 3

	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(workerID int) {
			defer func() { done <- true }()

			for j := 0; j < numExecutions; j++ {
				input := map[string]interface{}{
					"task": "concurrent task",
				}

				_, err := agent.Execute(ctx, input)
				assert.NoError(t, err)
			}
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Verify metrics are consistent
	metrics := agent.GetContextPerformanceMetrics()
	expectedExecutions := int64(numGoroutines * numExecutions)
	assert.Equal(t, expectedExecutions, metrics["total_executions"])
}

// Mock LLM for testing.
type TestMockLLM struct {
	responses    []string
	index        int
	shouldError  bool
	errorMessage string
	mu           sync.Mutex
}

func (m *TestMockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.shouldError {
		return nil, fmt.Errorf("%s", m.errorMessage)
	}

	response := "Default mock response"
	if m.index < len(m.responses) {
		response = m.responses[m.index]
		m.index++
	}

	// Format response as ReAct expects: thought, action (Finish), observation, answer
	reactResponse := fmt.Sprintf(`thought: %s
action: finish
observation: Task completed successfully
answer: %s`, response, response)

	return &core.LLMResponse{
		Content: reactResponse,
		Usage: &core.TokenInfo{
			PromptTokens:     10,
			CompletionTokens: 15,
			TotalTokens:      25,
		},
		Metadata: map[string]interface{}{
			"finish_reason": "stop",
		},
	}, nil
}

func (m *TestMockLLM) ProviderName() string {
	return "test"
}

func (m *TestMockLLM) ModelID() string {
	return "test-mock-llm"
}

func (m *TestMockLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion, core.CapabilityChat}
}

func (m *TestMockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"mock": "json response"}, nil
}

func (m *TestMockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"mock": "functions response"}, nil
}

func (m *TestMockLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return &core.EmbeddingResult{Vector: []float32{0.1, 0.2, 0.3}}, nil
}

func (m *TestMockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return &core.BatchEmbeddingResult{Embeddings: []core.EmbeddingResult{{Vector: []float32{0.1, 0.2, 0.3}}}}, nil
}

func (m *TestMockLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not supported in mock")
}

func (m *TestMockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{Content: "mock multimodal response"}, nil
}

func (m *TestMockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not supported in mock")
}

// Benchmark tests

func BenchmarkReActAgentWithContext(b *testing.B) {
	tempDir := b.TempDir()

	agent := NewReActAgent(
		"benchmark-agent",
		"Benchmark Agent",
		WithProductionContextOptimization(),
		WithContextEngineering(tempDir, contextmgmt.DefaultConfig()),
	)

	mockLLM := &TestMockLLM{responses: []string{"Benchmark response"}}
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("The benchmark task"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The result"))},
		},
	).WithInstruction("Benchmark test")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(b, err)

	ctx := context.Background()
	input := map[string]interface{}{
		"task": "benchmark task",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := agent.Execute(ctx, input)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReActAgentWithoutContext(b *testing.B) {
	agent := NewReActAgent("benchmark-agent-no-context", "Benchmark Agent No Context")

	mockLLM := &TestMockLLM{responses: []string{"Benchmark response"}}
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("The benchmark task"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("The result"))},
		},
	).WithInstruction("Benchmark test")

	err := agent.Initialize(mockLLM, signature)
	require.NoError(b, err)

	ctx := context.Background()
	input := map[string]interface{}{
		"task": "benchmark task",
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := agent.Execute(ctx, input)
		if err != nil {
			b.Fatal(err)
		}
	}
}
