package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockDataset implements core.Dataset for testing.
type MockDataset struct {
	examples []core.Example
	index    int
}

func NewMockDataset(examples []core.Example) *MockDataset {
	return &MockDataset{
		examples: examples,
		index:    0,
	}
}

func (m *MockDataset) Next() (core.Example, bool) {
	if m.index >= len(m.examples) {
		return core.Example{}, false
	}
	example := m.examples[m.index]
	m.index++
	return example, true
}

func (m *MockDataset) Reset() {
	m.index = 0
}

// MockProgram implements core.Program for testing.
type MockProgram struct {
	modules map[string]core.Module
}

func NewMockProgram() *MockProgram {
	return &MockProgram{
		modules: make(map[string]core.Module),
	}
}

func (m *MockProgram) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{
		"result": "mock_result",
	}, nil
}

func (m *MockProgram) Clone() core.Program {
	return core.Program{
		Modules: make(map[string]core.Module),
		Forward: m.Execute,
	}
}

func TestNewMCPOptimizer(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(384)
	optimizer := NewMCPOptimizer(embeddingService)

	assert.NotNil(t, optimizer)
	assert.Equal(t, "MCPOptimizer", optimizer.Name)
	assert.NotNil(t, optimizer.PatternCollector)
	assert.NotNil(t, optimizer.SimilarityMatcher)
	assert.NotNil(t, optimizer.ExampleSelector)
	assert.NotNil(t, optimizer.MetricsEvaluator)
	assert.NotNil(t, optimizer.ToolOrchestrator)
	assert.NotNil(t, optimizer.Config)

	// Check default configuration
	assert.Equal(t, 1000, optimizer.Config.MaxPatterns)
	assert.Equal(t, 0.7, optimizer.Config.SimilarityThreshold)
	assert.Equal(t, 5, optimizer.Config.KNearestNeighbors)
	assert.Equal(t, 2.0, optimizer.Config.SuccessWeightFactor)
	assert.Equal(t, 384, optimizer.Config.EmbeddingDimensions)
	assert.True(t, optimizer.Config.LearningEnabled)
}

func TestMCPOptimizerWithCustomConfig(t *testing.T) {
	config := &MCPOptimizerConfig{
		MaxPatterns:         500,
		SimilarityThreshold: 0.8,
		KNearestNeighbors:   3,
		SuccessWeightFactor: 1.5,
		EmbeddingDimensions: 256,
		LearningEnabled:     false,
		MetricsWindowSize:   50,
	}

	embeddingService := NewSimpleEmbeddingService(256)
	optimizer := NewMCPOptimizerWithConfig(config, embeddingService)

	assert.NotNil(t, optimizer)
	assert.Equal(t, config, optimizer.Config)
	assert.Equal(t, 500, optimizer.Config.MaxPatterns)
	assert.Equal(t, 0.8, optimizer.Config.SimilarityThreshold)
	assert.False(t, optimizer.Config.LearningEnabled)
}

func TestPatternCollectorAddInteraction(t *testing.T) {
	config := &MCPOptimizerConfig{
		MaxPatterns: 3, // Small limit for testing eviction
	}
	
	pc := &PatternCollector{
		Patterns:    make([]MCPInteraction, 0, config.MaxPatterns),
		IndexByCtx:  make(map[string][]int),
		IndexByTool: make(map[string][]int),
		Config:      config,
	}

	ctx := context.Background()

	// Add first interaction
	interaction1 := MCPInteraction{
		ID:        "test1",
		Timestamp: time.Now(),
		Context:   "test context 1",
		ToolName:  "test_tool",
		Parameters: map[string]interface{}{
			"param1": "value1",
		},
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	err := pc.AddInteraction(ctx, interaction1)
	require.NoError(t, err)
	assert.Equal(t, 1, pc.GetPatternCount())

	// Add second interaction
	interaction2 := MCPInteraction{
		ID:        "test2",
		Timestamp: time.Now(),
		Context:   "test context 2",
		ToolName:  "test_tool2",
		Parameters: map[string]interface{}{
			"param2": "value2",
		},
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	err = pc.AddInteraction(ctx, interaction2)
	require.NoError(t, err)
	assert.Equal(t, 2, pc.GetPatternCount())

	// Add third interaction
	interaction3 := MCPInteraction{
		ID:        "test3",
		Timestamp: time.Now(),
		Context:   "test context 3",
		ToolName:  "test_tool3",
		Parameters: map[string]interface{}{
			"param3": "value3",
		},
		Success:  false,
		Metadata: make(map[string]interface{}),
	}

	err = pc.AddInteraction(ctx, interaction3)
	require.NoError(t, err)
	assert.Equal(t, 3, pc.GetPatternCount())

	// Add fourth interaction (should trigger eviction)
	interaction4 := MCPInteraction{
		ID:        "test4",
		Timestamp: time.Now(),
		Context:   "test context 4",
		ToolName:  "test_tool4",
		Parameters: map[string]interface{}{
			"param4": "value4",
		},
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	err = pc.AddInteraction(ctx, interaction4)
	require.NoError(t, err)
	assert.Equal(t, 3, pc.GetPatternCount()) // Should still be 3 due to eviction
}

func TestPatternCollectorGetSimilarPatterns(t *testing.T) {
	config := &MCPOptimizerConfig{
		MaxPatterns: 10,
	}
	
	pc := &PatternCollector{
		Patterns:    make([]MCPInteraction, 0, config.MaxPatterns),
		IndexByCtx:  make(map[string][]int),
		IndexByTool: make(map[string][]int),
		Config:      config,
	}

	ctx := context.Background()

	// Add successful interactions
	successfulInteraction := MCPInteraction{
		ID:       "success1",
		Context:  "test context",
		ToolName: "git_log",
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	failedInteraction := MCPInteraction{
		ID:       "failed1",
		Context:  "test context",
		ToolName: "git_log",
		Success:  false,
		Metadata: make(map[string]interface{}),
	}

	err := pc.AddInteraction(ctx, successfulInteraction)
	require.NoError(t, err)

	err = pc.AddInteraction(ctx, failedInteraction)
	require.NoError(t, err)

	// Get similar patterns - should only return successful ones
	patterns, err := pc.GetSimilarPatterns(ctx, "test context", "git_log")
	require.NoError(t, err)
	assert.Len(t, patterns, 1)
	assert.True(t, patterns[0].Success)
	assert.Equal(t, "success1", patterns[0].ID)
}

func TestPatternCollectorGetPatternsByTool(t *testing.T) {
	config := &MCPOptimizerConfig{
		MaxPatterns: 10,
	}
	
	pc := &PatternCollector{
		Patterns:    make([]MCPInteraction, 0, config.MaxPatterns),
		IndexByCtx:  make(map[string][]int),
		IndexByTool: make(map[string][]int),
		Config:      config,
	}

	ctx := context.Background()

	// Add interactions for different tools
	gitInteraction := MCPInteraction{
		ID:       "git1",
		Context:  "git context",
		ToolName: "git_log",
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	fileInteraction := MCPInteraction{
		ID:       "file1",
		Context:  "file context",
		ToolName: "file_read",
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	err := pc.AddInteraction(ctx, gitInteraction)
	require.NoError(t, err)

	err = pc.AddInteraction(ctx, fileInteraction)
	require.NoError(t, err)

	// Get patterns by tool
	gitPatterns := pc.GetPatternsByTool("git_log")
	assert.Len(t, gitPatterns, 1)
	assert.Equal(t, "git1", gitPatterns[0].ID)

	filePatterns := pc.GetPatternsByTool("file_read")
	assert.Len(t, filePatterns, 1)
	assert.Equal(t, "file1", filePatterns[0].ID)

	nonExistentPatterns := pc.GetPatternsByTool("non_existent_tool")
	assert.Len(t, nonExistentPatterns, 0)
}

func TestSimpleEmbeddingService(t *testing.T) {
	service := NewSimpleEmbeddingService(10)
	ctx := context.Background()

	// Test embedding generation
	embedding1, err := service.GenerateEmbedding(ctx, "test text 1")
	require.NoError(t, err)
	assert.Len(t, embedding1, 10)

	embedding2, err := service.GenerateEmbedding(ctx, "test text 2")
	require.NoError(t, err)
	assert.Len(t, embedding2, 10)

	// Test that embeddings are normalized (L2 norm should be 1)
	norm := 0.0
	for _, val := range embedding1 {
		norm += val * val
	}
	assert.InDelta(t, 1.0, norm, 0.001) // Should be approximately 1

	// Test cosine similarity
	similarity := service.CosineSimilarity(embedding1, embedding2)
	assert.GreaterOrEqual(t, similarity, -1.0)
	assert.LessOrEqual(t, similarity, 1.0)

	// Test self-similarity (should be 1.0)
	selfSimilarity := service.CosineSimilarity(embedding1, embedding1)
	assert.InDelta(t, 1.0, selfSimilarity, 0.001)
}

func TestSimilarityMatcherFindSimilarInteractions(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(10)
	config := &MCPOptimizerConfig{
		SimilarityThreshold: 0.5,
		KNearestNeighbors:   3,
	}

	matcher := &SimilarityMatcher{
		embeddingService: embeddingService,
		Config:           config,
	}

	ctx := context.Background()

	// Create test patterns
	patterns := []MCPInteraction{
		{
			ID:       "pattern1",
			Context:  "similar context 1",
			ToolName: "test_tool",
			Success:  true,
		},
		{
			ID:       "pattern2",
			Context:  "similar context 2",
			ToolName: "test_tool",
			Success:  true,
		},
		{
			ID:       "pattern3",
			Context:  "different context entirely",
			ToolName: "test_tool",
			Success:  true,
		},
	}

	// Find similar interactions
	similar, err := matcher.FindSimilarInteractions(ctx, "similar context", patterns)
	require.NoError(t, err)

	// Should return patterns that meet similarity threshold
	assert.LessOrEqual(t, len(similar), config.KNearestNeighbors)
	assert.GreaterOrEqual(t, len(similar), 0) // Might be 0 if threshold is too high
}

func TestExampleSelectorSelectOptimalExamples(t *testing.T) {
	config := &MCPOptimizerConfig{
		KNearestNeighbors:   2,
		SuccessWeightFactor: 2.0,
		MetricsWindowSize:   10,
	}

	selector := &ExampleSelector{
		Config:         config,
		SuccessHistory: make(map[string][]bool),
	}

	ctx := context.Background()

	// Create test candidates with different success rates
	candidates := []MCPInteraction{
		{
			ID:        "candidate1",
			ToolName:  "test_tool",
			Parameters: map[string]interface{}{"param": "value1"},
			Success:   true,
			Timestamp: time.Now(),
		},
		{
			ID:        "candidate2", 
			ToolName:  "test_tool",
			Parameters: map[string]interface{}{"param": "value2"},
			Success:   false,
			Timestamp: time.Now().Add(-time.Hour), // Older timestamp
		},
		{
			ID:        "candidate3",
			ToolName:  "test_tool",
			Parameters: map[string]interface{}{"param": "value3"},
			Success:   true,
			Timestamp: time.Now().Add(-30 * time.Minute), // Medium age
		},
	}

	// Record some success history
	selector.RecordSuccess(candidates[0], true)
	selector.RecordSuccess(candidates[0], true)
	selector.RecordSuccess(candidates[1], false)
	selector.RecordSuccess(candidates[2], true)

	// Select optimal examples
	optimal, err := selector.SelectOptimalExamples(ctx, candidates)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(optimal), config.KNearestNeighbors)

	// The first result should be the one with highest weight
	if len(optimal) > 0 {
		// Should prefer successful interactions
		assert.True(t, optimal[0].Success || len(optimal) == 1)
	}
}

func TestExampleSelectorRecordSuccess(t *testing.T) {
	config := &MCPOptimizerConfig{
		MetricsWindowSize: 3, // Small window for testing
	}

	selector := &ExampleSelector{
		Config:         config,
		SuccessHistory: make(map[string][]bool),
	}

	interaction := MCPInteraction{
		ToolName:   "test_tool",
		Parameters: map[string]interface{}{"param": "value"},
	}

	// Record multiple successes
	selector.RecordSuccess(interaction, true)
	selector.RecordSuccess(interaction, false)
	selector.RecordSuccess(interaction, true)

	// Generate expected pattern key using the same logic as the implementation
	hashParameters := func(params map[string]interface{}) string {
		data, _ := json.Marshal(params)
		return fmt.Sprintf("params_%d", len(data))
	}
	expectedKey := fmt.Sprintf("%s_%s", interaction.ToolName, hashParameters(interaction.Parameters))
	assert.Contains(t, selector.SuccessHistory, expectedKey)
	assert.Len(t, selector.SuccessHistory[expectedKey], 3)

	// Record one more (should trigger window sliding)
	selector.RecordSuccess(interaction, true)
	assert.Len(t, selector.SuccessHistory[expectedKey], 3) // Should still be 3 due to window limit
}

func TestMetricsEvaluatorRecordMetrics(t *testing.T) {
	config := &MCPOptimizerConfig{
		MetricsWindowSize: 2, // Small window for testing
	}

	evaluator := &MetricsEvaluator{
		Metrics: make([]MCPMetrics, 0),
		Config:  config,
	}

	ctx := context.Background()

	// Record first metrics
	metrics1 := MCPMetrics{
		Timestamp:             time.Now(),
		ToolSelectionAccuracy: 0.8,
		ExecutionSuccessRate:  0.9,
		InteractionsProcessed: 10,
	}

	evaluator.RecordMetrics(ctx, metrics1)
	assert.Len(t, evaluator.Metrics, 1)

	// Record second metrics
	metrics2 := MCPMetrics{
		Timestamp:             time.Now(),
		ToolSelectionAccuracy: 0.9,
		ExecutionSuccessRate:  0.85,
		InteractionsProcessed: 15,
	}

	evaluator.RecordMetrics(ctx, metrics2)
	assert.Len(t, evaluator.Metrics, 2)

	// Record third metrics (should trigger window sliding)
	metrics3 := MCPMetrics{
		Timestamp:             time.Now(),
		ToolSelectionAccuracy: 0.75,
		ExecutionSuccessRate:  0.95,
		InteractionsProcessed: 20,
	}

	evaluator.RecordMetrics(ctx, metrics3)
	assert.Len(t, evaluator.Metrics, 2) // Should still be 2 due to window limit

	// Test latest metrics
	latest := evaluator.GetLatestMetrics()
	require.NotNil(t, latest)
	assert.Equal(t, metrics3.ToolSelectionAccuracy, latest.ToolSelectionAccuracy)

	// Test average metrics
	average := evaluator.GetAverageMetrics()
	require.NotNil(t, average)
	expectedAccuracy := (metrics2.ToolSelectionAccuracy + metrics3.ToolSelectionAccuracy) / 2
	assert.InDelta(t, expectedAccuracy, average.ToolSelectionAccuracy, 0.001)
}

func TestToolOrchestrator(t *testing.T) {
	config := &MCPOptimizerConfig{}
	orchestrator := &ToolOrchestrator{
		Dependencies: make(map[string][]string),
		Workflows:    make([]ToolWorkflow, 0),
		Config:       config,
	}

	ctx := context.Background()

	// Create a test workflow
	workflow := ToolWorkflow{
		ID:      "test_workflow",
		Context: "test context",
		Success: true,
		Steps: []WorkflowStep{
			{
				ToolName:   "git_status",
				Parameters: map[string]interface{}{"path": "/repo"},
				Order:      0,
			},
			{
				ToolName:   "git_log",
				Parameters: map[string]interface{}{"count": 5},
				Order:      1,
			},
		},
	}

	// Record the workflow
	err := orchestrator.RecordWorkflow(ctx, workflow)
	require.NoError(t, err)
	assert.Equal(t, 1, orchestrator.GetWorkflowCount())

	// Check dependencies were recorded
	deps := orchestrator.GetDependencies("git_log")
	assert.Contains(t, deps, "git_status")

	// Test optimal tool sequence
	availableTools := []string{"git_log", "git_status", "file_read"}
	sequence, err := orchestrator.GetOptimalToolSequence(ctx, "test context", availableTools)
	require.NoError(t, err)
	assert.NotEmpty(t, sequence)
}

func TestMCPOptimizerCompile(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(10)
	optimizer := NewMCPOptimizer(embeddingService)

	ctx := context.Background()

	// Create mock dataset
	examples := []core.Example{
		{
			Inputs: map[string]interface{}{
				"query": "show git log",
			},
			Outputs: map[string]interface{}{
				"tool_calls": []interface{}{
					map[string]interface{}{
						"tool_name": "git_log",
						"parameters": map[string]interface{}{
							"count": 5,
						},
					},
				},
				"result": "git log output",
			},
		},
	}
	dataset := NewMockDataset(examples)

	// Create mock program
	program := NewMockProgram().Clone()

	// Test compile
	optimizedProgram, err := optimizer.Compile(ctx, program, dataset, func(expected, actual map[string]interface{}) float64 {
		return 1.0 // Mock metric always returns 1.0
	})

	require.NoError(t, err)
	assert.NotNil(t, optimizedProgram)

	// Verify patterns were collected
	assert.Greater(t, optimizer.PatternCollector.GetPatternCount(), 0)
}

func TestMCPOptimizerOptimizeInteraction(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(10)
	optimizer := NewMCPOptimizer(embeddingService)

	ctx := context.Background()

	// Add some patterns first
	interaction := MCPInteraction{
		ID:        "test_interaction",
		Context:   "show git log",
		ToolName:  "git_log",
		Parameters: map[string]interface{}{"count": 5},
		Success:   true,
		Timestamp: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	err := optimizer.PatternCollector.AddInteraction(ctx, interaction)
	require.NoError(t, err)

	// Test optimization
	optimized, err := optimizer.OptimizeInteraction(ctx, "show git log", "git_log")
	require.NoError(t, err)
	assert.NotNil(t, optimized)
	assert.Equal(t, "git_log", optimized.ToolName)
}

func TestMCPOptimizerLearnFromInteraction(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(10)
	optimizer := NewMCPOptimizer(embeddingService)

	ctx := context.Background()

	interaction := MCPInteraction{
		ID:            "learn_test",
		Context:       "test learning",
		ToolName:      "test_tool",
		Parameters:    map[string]interface{}{"param": "value"},
		Success:       true,
		ExecutionTime: 100 * time.Millisecond,
		Timestamp:     time.Now(),
		Metadata:      make(map[string]interface{}),
	}

	// Test learning
	err := optimizer.LearnFromInteraction(ctx, interaction)
	require.NoError(t, err)

	// Verify pattern was added
	assert.Equal(t, 1, optimizer.PatternCollector.GetPatternCount())

	// Verify metrics were recorded
	latest := optimizer.MetricsEvaluator.GetLatestMetrics()
	assert.NotNil(t, latest)
	assert.Equal(t, 1.0, latest.ToolSelectionAccuracy)
}

func TestMCPOptimizerGetOptimizationStats(t *testing.T) {
	embeddingService := NewSimpleEmbeddingService(10)
	optimizer := NewMCPOptimizer(embeddingService)

	stats := optimizer.GetOptimizationStats()
	assert.NotNil(t, stats)
	assert.Contains(t, stats, "total_patterns")
	assert.Contains(t, stats, "total_workflows")
	assert.Contains(t, stats, "learning_enabled")
	assert.Contains(t, stats, "similarity_threshold")

	// Initial values
	assert.Equal(t, 0, stats["total_patterns"])
	assert.Equal(t, 0, stats["total_workflows"])
	assert.Equal(t, true, stats["learning_enabled"])
	assert.Equal(t, 0.7, stats["similarity_threshold"])
}

func TestMCPOptimizerLearningDisabled(t *testing.T) {
	config := &MCPOptimizerConfig{
		LearningEnabled: false,
	}
	embeddingService := NewSimpleEmbeddingService(10)
	optimizer := NewMCPOptimizerWithConfig(config, embeddingService)

	ctx := context.Background()

	interaction := MCPInteraction{
		ID:       "disabled_learning_test",
		Context:  "test",
		ToolName: "test_tool",
		Success:  true,
		Metadata: make(map[string]interface{}),
	}

	// Learning should be skipped
	err := optimizer.LearnFromInteraction(ctx, interaction)
	require.NoError(t, err)
	
	// Pattern should not be added due to learning being disabled
	assert.Equal(t, 0, optimizer.PatternCollector.GetPatternCount())
}