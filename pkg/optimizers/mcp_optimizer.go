package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
)

// MCPOptimizer implements an optimizer specifically designed for MCP (Model Context Protocol) workflows.
// It follows the KNNFewShot + Statistical Weighting methodology to learn from successful MCP tool interactions.
type MCPOptimizer struct {
	core.BaseOptimizer
	PatternCollector  *PatternCollector
	SimilarityMatcher *SimilarityMatcher
	ExampleSelector   *ExampleSelector
	MetricsEvaluator  *MetricsEvaluator
	ToolOrchestrator  *ToolOrchestrator
	Config            *MCPOptimizerConfig
	mu                sync.RWMutex
}

// MCPOptimizerConfig holds configuration parameters for the MCP optimizer.
type MCPOptimizerConfig struct {
	MaxPatterns          int     `json:"max_patterns"`          // Maximum number of patterns to store
	SimilarityThreshold  float64 `json:"similarity_threshold"`  // Minimum similarity for pattern matching
	KNearestNeighbors    int     `json:"k_nearest_neighbors"`   // Number of neighbors for KNN matching
	SuccessWeightFactor  float64 `json:"success_weight_factor"` // Weight factor for successful patterns
	EmbeddingDimensions  int     `json:"embedding_dimensions"`  // Dimensions for context embeddings
	LearningEnabled      bool    `json:"learning_enabled"`      // Whether to learn from new interactions
	MetricsWindowSize    int     `json:"metrics_window_size"`   // Window size for performance metrics
	OptimizationInterval int     `json:"optimization_interval"` // Interval for optimization cycles (in interactions)
}

// MCPInteraction represents a single MCP tool interaction with all relevant context.
type MCPInteraction struct {
	ID            string                 `json:"id"`
	Timestamp     time.Time              `json:"timestamp"`
	Context       string                 `json:"context"`                  // The user query or context that triggered this interaction
	ToolName      string                 `json:"tool_name"`                // Name of the MCP tool used
	Parameters    map[string]interface{} `json:"parameters"`               // Parameters passed to the tool
	Result        core.ToolResult        `json:"result"`                   // Result returned by the tool
	Success       bool                   `json:"success"`                  // Whether the interaction was successful
	ExecutionTime time.Duration          `json:"execution_time"`           // Time taken to execute
	ErrorMessage  string                 `json:"error_message,omitempty"`  // Error message if failed
	ContextVector []float64              `json:"context_vector,omitempty"` // Embedding vector for the context
	Metadata      map[string]interface{} `json:"metadata"`                 // Additional metadata
}

// PatternCollector logs and stores successful MCP tool interactions with context.
type PatternCollector struct {
	Patterns    []MCPInteraction    `json:"patterns"`
	IndexByCtx  map[string][]int    `json:"index_by_ctx"`  // Index patterns by context hash
	IndexByTool map[string][]int    `json:"index_by_tool"` // Index patterns by tool name
	Config      *MCPOptimizerConfig `json:"config"`
	mu          sync.RWMutex        `json:"-"`
}

// SimilarityMatcher performs KNN-based context matching using embeddings.
type SimilarityMatcher struct {
	embeddingService EmbeddingService    `json:"-"`
	Config           *MCPOptimizerConfig `json:"config"`
}

// ExampleSelector implements statistical weighting system for optimal example selection.
type ExampleSelector struct {
	Config         *MCPOptimizerConfig `json:"config"`
	SuccessHistory map[string][]bool   `json:"success_history"` // Track success history by pattern hash
	mu             sync.RWMutex        `json:"-"`
}

// ToolOrchestrator optimizes multi-tool workflows and dependencies.
type ToolOrchestrator struct {
	Dependencies map[string][]string `json:"dependencies"` // Tool dependency mapping
	Workflows    []ToolWorkflow      `json:"workflows"`    // Recorded successful workflows
	Config       *MCPOptimizerConfig `json:"config"`
	mu           sync.RWMutex        `json:"-"`
}

// MetricsEvaluator provides MCP-specific performance metrics.
type MetricsEvaluator struct {
	Metrics []MCPMetrics        `json:"metrics"`
	Config  *MCPOptimizerConfig `json:"config"`
	mu      sync.RWMutex        `json:"-"`
}

// MCPMetrics represents performance metrics for MCP tool interactions.
type MCPMetrics struct {
	Timestamp             time.Time `json:"timestamp"`
	ToolSelectionAccuracy float64   `json:"tool_selection_accuracy"`
	ParameterOptimality   float64   `json:"parameter_optimality"`
	ExecutionSuccessRate  float64   `json:"execution_success_rate"`
	AverageExecutionTime  float64   `json:"average_execution_time"`
	InteractionsProcessed int       `json:"interactions_processed"`
}

// ToolWorkflow represents a sequence of tool calls that achieved a successful outcome.
type ToolWorkflow struct {
	ID        string                 `json:"id"`
	Steps     []WorkflowStep         `json:"steps"`
	Context   string                 `json:"context"`
	Success   bool                   `json:"success"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// WorkflowStep represents a single step in a tool workflow.
type WorkflowStep struct {
	ToolName   string                 `json:"tool_name"`
	Parameters map[string]interface{} `json:"parameters"`
	Result     core.ToolResult        `json:"result"`
	Order      int                    `json:"order"`
	Duration   time.Duration          `json:"duration"`
}

// EmbeddingService defines the interface for generating context embeddings.
type EmbeddingService interface {
	GenerateEmbedding(ctx context.Context, text string) ([]float64, error)
	CosineSimilarity(vec1, vec2 []float64) float64
}

// NewMCPOptimizer creates a new MCP optimizer with default configuration.
func NewMCPOptimizer(embeddingService EmbeddingService) *MCPOptimizer {
	config := &MCPOptimizerConfig{
		MaxPatterns:          1000,
		SimilarityThreshold:  0.7,
		KNearestNeighbors:    5,
		SuccessWeightFactor:  2.0,
		EmbeddingDimensions:  384, // Standard embedding dimension
		LearningEnabled:      true,
		MetricsWindowSize:    100,
		OptimizationInterval: 50,
	}

	return NewMCPOptimizerWithConfig(config, embeddingService)
}

// NewMCPOptimizerWithConfig creates a new MCP optimizer with custom configuration.
func NewMCPOptimizerWithConfig(config *MCPOptimizerConfig, embeddingService EmbeddingService) *MCPOptimizer {
	patternCollector := &PatternCollector{
		Patterns:    make([]MCPInteraction, 0, config.MaxPatterns),
		IndexByCtx:  make(map[string][]int),
		IndexByTool: make(map[string][]int),
		Config:      config,
	}

	similarityMatcher := &SimilarityMatcher{
		embeddingService: embeddingService,
		Config:           config,
	}

	exampleSelector := &ExampleSelector{
		Config:         config,
		SuccessHistory: make(map[string][]bool),
	}

	toolOrchestrator := &ToolOrchestrator{
		Dependencies: make(map[string][]string),
		Workflows:    make([]ToolWorkflow, 0),
		Config:       config,
	}

	metricsEvaluator := &MetricsEvaluator{
		Metrics: make([]MCPMetrics, 0),
		Config:  config,
	}

	return &MCPOptimizer{
		BaseOptimizer:     core.BaseOptimizer{Name: "MCPOptimizer"},
		PatternCollector:  patternCollector,
		SimilarityMatcher: similarityMatcher,
		ExampleSelector:   exampleSelector,
		MetricsEvaluator:  metricsEvaluator,
		ToolOrchestrator:  toolOrchestrator,
		Config:            config,
	}
}

// Compile implements the core.Optimizer interface for MCP-specific optimization.
func (m *MCPOptimizer) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	logger := logging.GetLogger()
	logger.Info(ctx, "Starting MCP optimization compilation")

	optimizedProgram := program.Clone()

	// Phase 1: Collect patterns from dataset examples
	if err := m.collectPatternsFromDataset(ctx, dataset); err != nil {
		return optimizedProgram, fmt.Errorf("failed to collect patterns from dataset: %w", err)
	}

	// Phase 2: Optimize program modules based on collected patterns
	if err := m.optimizeProgram(ctx, &optimizedProgram); err != nil {
		return optimizedProgram, fmt.Errorf("failed to optimize program: %w", err)
	}

	// Phase 3: Evaluate optimization effectiveness
	if err := m.evaluateOptimization(ctx, program, optimizedProgram, dataset, metric); err != nil {
		logger.Warn(ctx, "Failed to evaluate optimization: %v", err)
		// Don't fail compilation if evaluation fails
	}

	logger.Info(ctx, "MCP optimization compilation completed successfully")
	return optimizedProgram, nil
}

// collectPatternsFromDataset processes the dataset to collect MCP interaction patterns.
func (m *MCPOptimizer) collectPatternsFromDataset(ctx context.Context, dataset core.Dataset) error {
	logger := logging.GetLogger()
	dataset.Reset()

	count := 0
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		// Extract MCP interactions from the example
		if interactions, err := m.extractMCPInteractions(ctx, example); err == nil {
			for _, interaction := range interactions {
				if err := m.PatternCollector.AddInteraction(ctx, interaction); err != nil {
					logger.Warn(ctx, "Failed to add interaction to pattern collector: %v", err)
				}
			}
		}
		count++
	}

	logger.Info(ctx, "Collected patterns from %d examples", count)
	return nil
}

// extractMCPInteractions extracts MCP tool interactions from a training example.
func (m *MCPOptimizer) extractMCPInteractions(ctx context.Context, example core.Example) ([]MCPInteraction, error) {
	var interactions []MCPInteraction

	// Look for MCP tool calls in the example data
	// This is a simplified implementation - in practice, you'd parse execution traces
	if toolCalls, exists := example.Outputs["tool_calls"]; exists {
		if callsArray, ok := toolCalls.([]interface{}); ok {
			for i, call := range callsArray {
				if callMap, ok := call.(map[string]interface{}); ok {
					interaction := MCPInteraction{
						ID:        fmt.Sprintf("example_%d_call_%d", len(interactions), i),
						Timestamp: time.Now(),
						Context:   fmt.Sprintf("%v", example.Inputs),
						ToolName:  fmt.Sprintf("%v", callMap["tool_name"]),
						Parameters: func() map[string]interface{} {
							if params, ok := callMap["parameters"].(map[string]interface{}); ok {
								return params
							}
							return make(map[string]interface{})
						}(),
						Success:  true, // Assume success if in training data
						Metadata: make(map[string]interface{}),
					}
					interactions = append(interactions, interaction)
				}
			}
		}
	}

	return interactions, nil
}

// optimizeProgram applies learned patterns to optimize the program.
func (m *MCPOptimizer) optimizeProgram(ctx context.Context, program *core.Program) error {
	logger := logging.GetLogger()
	logger.Info(ctx, "Optimizing program with %d collected patterns", len(m.PatternCollector.Patterns))

	// Optimize each module in the program
	for moduleName, module := range program.Modules {
		if err := m.optimizeModule(ctx, moduleName, module); err != nil {
			logger.Warn(ctx, "Failed to optimize module %s: %v", moduleName, err)
		}
	}

	return nil
}

// optimizeModule optimizes a specific module based on collected patterns.
func (m *MCPOptimizer) optimizeModule(ctx context.Context, moduleName string, module core.Module) error {
	// For modules that use tools, optimize tool selection and parameters
	if toolUser, ok := module.(interface {
		GetToolRegistry() *tools.InMemoryToolRegistry
	}); ok {
		registry := toolUser.GetToolRegistry()
		if registry != nil {
			return m.optimizeToolUsage(ctx, registry)
		}
	}

	return nil
}

// optimizeToolUsage optimizes tool selection and parameter usage based on patterns.
func (m *MCPOptimizer) optimizeToolUsage(ctx context.Context, registry *tools.InMemoryToolRegistry) error {
	// This would implement tool selection optimization
	// For now, this is a placeholder for the optimization logic
	return nil
}

// evaluateOptimization evaluates the effectiveness of the optimization.
func (m *MCPOptimizer) evaluateOptimization(ctx context.Context, original, optimized core.Program, dataset core.Dataset, metric core.Metric) error {
	logger := logging.GetLogger()

	// Run both programs on a subset of the dataset and compare performance
	dataset.Reset()
	var originalScores, optimizedScores []float64

	count := 0
	maxEvalExamples := 10 // Limit evaluation to avoid long compile times

	for count < maxEvalExamples {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		// Test original program
		if originalResult, err := original.Execute(ctx, example.Inputs); err == nil {
			score := metric(example.Outputs, originalResult)
			originalScores = append(originalScores, score)
		}

		// Test optimized program
		if optimizedResult, err := optimized.Execute(ctx, example.Inputs); err == nil {
			score := metric(example.Outputs, optimizedResult)
			optimizedScores = append(optimizedScores, score)
		}

		count++
	}

	// Calculate average scores
	if len(originalScores) > 0 && len(optimizedScores) > 0 {
		originalAvg := average(originalScores)
		optimizedAvg := average(optimizedScores)
		improvement := optimizedAvg - originalAvg

		logger.Info(ctx, "Original program average score: %.3f", originalAvg)
		logger.Info(ctx, "Optimized program average score: %.3f", optimizedAvg)
		logger.Info(ctx, "Improvement: %.3f", improvement)

		// Record metrics
		metrics := MCPMetrics{
			Timestamp:             time.Now(),
			ToolSelectionAccuracy: optimizedAvg,
			ParameterOptimality:   improvement,
			ExecutionSuccessRate:  float64(len(optimizedScores)) / float64(count),
			InteractionsProcessed: count,
		}
		m.MetricsEvaluator.RecordMetrics(ctx, metrics)
	}

	return nil
}

// PatternCollector methods

// AddInteraction adds a new MCP interaction to the pattern collection.
func (pc *PatternCollector) AddInteraction(ctx context.Context, interaction MCPInteraction) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	logger := logging.GetLogger()

	// Generate context hash for indexing
	contextHash := pc.hashContext(interaction.Context)

	// Check if we need to evict old patterns
	if len(pc.Patterns) >= pc.Config.MaxPatterns {
		pc.evictOldestPattern()
	}

	// Add to patterns array
	patternIndex := len(pc.Patterns)
	pc.Patterns = append(pc.Patterns, interaction)

	// Update indices
	pc.IndexByCtx[contextHash] = append(pc.IndexByCtx[contextHash], patternIndex)
	pc.IndexByTool[interaction.ToolName] = append(pc.IndexByTool[interaction.ToolName], patternIndex)

	logger.Debug(ctx, "Added MCP interaction pattern: tool=%s, success=%t", interaction.ToolName, interaction.Success)
	return nil
}

// GetSimilarPatterns retrieves patterns similar to the given context.
func (pc *PatternCollector) GetSimilarPatterns(ctx context.Context, context string, toolName string) ([]MCPInteraction, error) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	var candidates []MCPInteraction

	// If tool name is specified, filter by tool first
	if toolName != "" {
		if indices, exists := pc.IndexByTool[toolName]; exists {
			for _, idx := range indices {
				if idx < len(pc.Patterns) {
					candidates = append(candidates, pc.Patterns[idx])
				}
			}
		}
	} else {
		// Otherwise, use all patterns
		candidates = append(candidates, pc.Patterns...)
	}

	// Filter successful patterns only
	var successfulPatterns []MCPInteraction
	for _, pattern := range candidates {
		if pattern.Success {
			successfulPatterns = append(successfulPatterns, pattern)
		}
	}

	return successfulPatterns, nil
}

// GetPatternsByTool retrieves all patterns for a specific tool.
func (pc *PatternCollector) GetPatternsByTool(toolName string) []MCPInteraction {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	var patterns []MCPInteraction
	if indices, exists := pc.IndexByTool[toolName]; exists {
		for _, idx := range indices {
			if idx < len(pc.Patterns) {
				patterns = append(patterns, pc.Patterns[idx])
			}
		}
	}
	return patterns
}

// evictOldestPattern removes the oldest pattern to make space for new ones.
func (pc *PatternCollector) evictOldestPattern() {
	if len(pc.Patterns) == 0 {
		return
	}

	// Remove the first (oldest) pattern
	evicted := pc.Patterns[0]
	pc.Patterns = pc.Patterns[1:]

	// Update indices by removing references to index 0 and decrementing all others
	pc.updateIndicesAfterEviction(evicted)
}

// updateIndicesAfterEviction updates all indices after a pattern eviction.
func (pc *PatternCollector) updateIndicesAfterEviction(evicted MCPInteraction) {
	contextHash := pc.hashContext(evicted.Context)

	// Update context index
	if indices, exists := pc.IndexByCtx[contextHash]; exists {
		newIndices := make([]int, 0, len(indices))
		for _, idx := range indices {
			if idx == 0 {
				continue // Skip the evicted index
			}
			newIndices = append(newIndices, idx-1) // Decrement by 1
		}
		if len(newIndices) > 0 {
			pc.IndexByCtx[contextHash] = newIndices
		} else {
			delete(pc.IndexByCtx, contextHash)
		}
	}

	// Update tool index
	if indices, exists := pc.IndexByTool[evicted.ToolName]; exists {
		newIndices := make([]int, 0, len(indices))
		for _, idx := range indices {
			if idx == 0 {
				continue // Skip the evicted index
			}
			newIndices = append(newIndices, idx-1) // Decrement by 1
		}
		if len(newIndices) > 0 {
			pc.IndexByTool[evicted.ToolName] = newIndices
		} else {
			delete(pc.IndexByTool, evicted.ToolName)
		}
	}
}

// hashContext creates a hash of the context string for indexing.
func (pc *PatternCollector) hashContext(context string) string {
	// Simple hash implementation - in production, use a proper hash function
	return fmt.Sprintf("ctx_%d", len(context))
}

// GetPatternCount returns the total number of stored patterns.
func (pc *PatternCollector) GetPatternCount() int {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return len(pc.Patterns)
}

// SimilarityMatcher methods

// FindSimilarInteractions finds the K most similar interactions to the given context.
func (sm *SimilarityMatcher) FindSimilarInteractions(ctx context.Context, targetContext string, patterns []MCPInteraction) ([]MCPInteraction, error) {
	if sm.embeddingService == nil {
		return patterns, fmt.Errorf("embedding service not configured")
	}

	// Generate embedding for target context
	targetEmbedding, err := sm.embeddingService.GenerateEmbedding(ctx, targetContext)
	if err != nil {
		return patterns, fmt.Errorf("failed to generate target embedding: %w", err)
	}

	// Calculate similarities and sort
	type PatternSimilarity struct {
		Pattern    MCPInteraction
		Similarity float64
	}

	var similarities []PatternSimilarity
	for _, pattern := range patterns {
		// Generate embedding for pattern context if not cached
		if len(pattern.ContextVector) == 0 {
			embedding, err := sm.embeddingService.GenerateEmbedding(ctx, pattern.Context)
			if err != nil {
				continue // Skip patterns we can't embed
			}
			pattern.ContextVector = embedding
		}

		similarity := sm.embeddingService.CosineSimilarity(targetEmbedding, pattern.ContextVector)
		if similarity >= sm.Config.SimilarityThreshold {
			similarities = append(similarities, PatternSimilarity{
				Pattern:    pattern,
				Similarity: similarity,
			})
		}
	}

	// Sort by similarity (descending)
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Similarity > similarities[j].Similarity
	})

	// Return top K results
	k := sm.Config.KNearestNeighbors
	if k > len(similarities) {
		k = len(similarities)
	}

	result := make([]MCPInteraction, k)
	for i := 0; i < k; i++ {
		result[i] = similarities[i].Pattern
	}

	return result, nil
}

// ExampleSelector methods

// SelectOptimalExamples selects the best examples based on statistical weighting.
func (es *ExampleSelector) SelectOptimalExamples(ctx context.Context, candidates []MCPInteraction) ([]MCPInteraction, error) {
	if len(candidates) == 0 {
		return candidates, nil
	}

	type WeightedExample struct {
		Interaction MCPInteraction
		Weight      float64
	}

	var weighted []WeightedExample
	for _, candidate := range candidates {
		weight := es.calculateWeight(candidate)
		weighted = append(weighted, WeightedExample{
			Interaction: candidate,
			Weight:      weight,
		})
	}

	// Sort by weight (descending)
	sort.Slice(weighted, func(i, j int) bool {
		return weighted[i].Weight > weighted[j].Weight
	})

	// Select top examples (up to K)
	maxExamples := es.Config.KNearestNeighbors
	if maxExamples > len(weighted) {
		maxExamples = len(weighted)
	}

	result := make([]MCPInteraction, maxExamples)
	for i := 0; i < maxExamples; i++ {
		result[i] = weighted[i].Interaction
	}

	return result, nil
}

// calculateWeight calculates the statistical weight for an example.
func (es *ExampleSelector) calculateWeight(interaction MCPInteraction) float64 {
	es.mu.RLock()
	defer es.mu.RUnlock()

	baseWeight := 1.0

	// Success factor
	if interaction.Success {
		baseWeight *= es.Config.SuccessWeightFactor
	}

	// Recency factor (more recent interactions get higher weight)
	timeFactor := math.Exp(-time.Since(interaction.Timestamp).Hours() / 24.0) // Decay over days
	baseWeight *= timeFactor

	// Tool-specific success rate
	patternKey := fmt.Sprintf("%s_%s", interaction.ToolName, es.hashParameters(interaction.Parameters))
	if history, exists := es.SuccessHistory[patternKey]; exists {
		successRate := es.calculateSuccessRate(history)
		baseWeight *= (1.0 + successRate) // Boost weight based on historical success
	}

	return baseWeight
}

// hashParameters creates a simple hash of parameters for tracking success patterns.
func (es *ExampleSelector) hashParameters(params map[string]interface{}) string {
	// Simple hash - in production, use proper JSON marshaling and hashing
	data, _ := json.Marshal(params)
	return fmt.Sprintf("params_%d", len(data))
}

// calculateSuccessRate calculates the success rate from a boolean history.
func (es *ExampleSelector) calculateSuccessRate(history []bool) float64 {
	if len(history) == 0 {
		return 0.0
	}

	successes := 0
	for _, success := range history {
		if success {
			successes++
		}
	}
	return float64(successes) / float64(len(history))
}

// RecordSuccess records the success/failure of an interaction pattern.
func (es *ExampleSelector) RecordSuccess(interaction MCPInteraction, success bool) {
	es.mu.Lock()
	defer es.mu.Unlock()

	patternKey := fmt.Sprintf("%s_%s", interaction.ToolName, es.hashParameters(interaction.Parameters))

	// Add to history
	if history, exists := es.SuccessHistory[patternKey]; exists {
		es.SuccessHistory[patternKey] = append(history, success)

		// Keep only recent history (sliding window)
		if len(es.SuccessHistory[patternKey]) > es.Config.MetricsWindowSize {
			es.SuccessHistory[patternKey] = es.SuccessHistory[patternKey][1:]
		}
	} else {
		es.SuccessHistory[patternKey] = []bool{success}
	}
}

// MetricsEvaluator methods

// RecordMetrics records new performance metrics.
func (me *MetricsEvaluator) RecordMetrics(ctx context.Context, metrics MCPMetrics) {
	me.mu.Lock()
	defer me.mu.Unlock()

	me.Metrics = append(me.Metrics, metrics)

	// Keep only recent metrics (sliding window)
	if len(me.Metrics) > me.Config.MetricsWindowSize {
		me.Metrics = me.Metrics[1:]
	}

	logger := logging.GetLogger()
	logger.Info(ctx, "Recorded MCP metrics: accuracy=%.3f, success_rate=%.3f", metrics.ToolSelectionAccuracy, metrics.ExecutionSuccessRate)
}

// GetLatestMetrics returns the most recent metrics.
func (me *MetricsEvaluator) GetLatestMetrics() *MCPMetrics {
	me.mu.RLock()
	defer me.mu.RUnlock()

	if len(me.Metrics) == 0 {
		return nil
	}
	return &me.Metrics[len(me.Metrics)-1]
}

// GetAverageMetrics calculates average metrics over the window.
func (me *MetricsEvaluator) GetAverageMetrics() *MCPMetrics {
	me.mu.RLock()
	defer me.mu.RUnlock()

	if len(me.Metrics) == 0 {
		return nil
	}

	var totalAccuracy, totalOptimality, totalSuccessRate, totalTime float64
	var totalInteractions int

	for _, m := range me.Metrics {
		totalAccuracy += m.ToolSelectionAccuracy
		totalOptimality += m.ParameterOptimality
		totalSuccessRate += m.ExecutionSuccessRate
		totalTime += m.AverageExecutionTime
		totalInteractions += m.InteractionsProcessed
	}

	count := float64(len(me.Metrics))
	return &MCPMetrics{
		Timestamp:             time.Now(),
		ToolSelectionAccuracy: totalAccuracy / count,
		ParameterOptimality:   totalOptimality / count,
		ExecutionSuccessRate:  totalSuccessRate / count,
		AverageExecutionTime:  totalTime / count,
		InteractionsProcessed: totalInteractions,
	}
}

// ToolOrchestrator methods

// RecordWorkflow records a successful multi-tool workflow.
func (to *ToolOrchestrator) RecordWorkflow(ctx context.Context, workflow ToolWorkflow) error {
	to.mu.Lock()
	defer to.mu.Unlock()

	logger := logging.GetLogger()

	// Add to workflows
	to.Workflows = append(to.Workflows, workflow)

	// Update dependencies based on workflow steps
	for i := 1; i < len(workflow.Steps); i++ {
		prevTool := workflow.Steps[i-1].ToolName
		currentTool := workflow.Steps[i].ToolName

		// Record dependency: currentTool depends on prevTool
		if to.Dependencies[currentTool] == nil {
			to.Dependencies[currentTool] = make([]string, 0)
		}

		// Check if dependency already exists
		exists := false
		for _, dep := range to.Dependencies[currentTool] {
			if dep == prevTool {
				exists = true
				break
			}
		}

		if !exists {
			to.Dependencies[currentTool] = append(to.Dependencies[currentTool], prevTool)
		}
	}

	logger.Debug(ctx, "Recorded workflow with %d steps", len(workflow.Steps))
	return nil
}

// GetOptimalToolSequence suggests the optimal sequence of tools for a given context.
func (to *ToolOrchestrator) GetOptimalToolSequence(ctx context.Context, context string, availableTools []string) ([]string, error) {
	to.mu.RLock()
	defer to.mu.RUnlock()

	// Find workflows that match the context
	var matchedWorkflows []ToolWorkflow
	for _, workflow := range to.Workflows {
		if to.contextMatches(workflow.Context, context) && workflow.Success {
			matchedWorkflows = append(matchedWorkflows, workflow)
		}
	}

	if len(matchedWorkflows) == 0 {
		// No matched workflows, return available tools in dependency order
		return to.sortByDependencies(availableTools), nil
	}

	// Find the most common successful sequence
	sequenceFreq := make(map[string]int)
	for _, workflow := range matchedWorkflows {
		sequence := to.extractToolSequence(workflow)
		key := to.sequenceToKey(sequence)
		sequenceFreq[key]++
	}

	// Get the most frequent sequence
	var bestSequence string
	maxFreq := 0
	for seq, freq := range sequenceFreq {
		if freq > maxFreq {
			maxFreq = freq
			bestSequence = seq
		}
	}

	return to.keyToSequence(bestSequence), nil
}

// contextMatches checks if two contexts are similar (simple implementation).
func (to *ToolOrchestrator) contextMatches(workflow, target string) bool {
	// Simple string similarity - in production, use embedding similarity
	return len(workflow) > 0 && len(target) > 0 // Placeholder logic
}

// extractToolSequence extracts the tool sequence from a workflow.
func (to *ToolOrchestrator) extractToolSequence(workflow ToolWorkflow) []string {
	sequence := make([]string, len(workflow.Steps))
	for i, step := range workflow.Steps {
		sequence[i] = step.ToolName
	}
	return sequence
}

// sequenceToKey converts a tool sequence to a string key.
func (to *ToolOrchestrator) sequenceToKey(sequence []string) string {
	return fmt.Sprintf("%v", sequence)
}

// keyToSequence converts a string key back to a tool sequence.
func (to *ToolOrchestrator) keyToSequence(key string) []string {
	// Simple implementation - in production, use proper serialization
	return []string{key} // Placeholder
}

// sortByDependencies sorts tools based on their dependencies.
func (to *ToolOrchestrator) sortByDependencies(tools []string) []string {
	// Simple topological sort implementation
	sorted := make([]string, 0, len(tools))
	remaining := make(map[string]bool)

	for _, tool := range tools {
		remaining[tool] = true
	}

	for len(remaining) > 0 {
		// Find a tool with no unresolved dependencies
		for tool := range remaining {
			hasUnresolvedDeps := false
			if deps, exists := to.Dependencies[tool]; exists {
				for _, dep := range deps {
					if remaining[dep] {
						hasUnresolvedDeps = true
						break
					}
				}
			}

			if !hasUnresolvedDeps {
				sorted = append(sorted, tool)
				delete(remaining, tool)
				break
			}
		}

		// Prevent infinite loop
		if len(sorted) == 0 && len(remaining) > 0 {
			// Add remaining tools in arbitrary order
			for tool := range remaining {
				sorted = append(sorted, tool)
				delete(remaining, tool)
			}
		}
	}

	return sorted
}

// GetDependencies returns the dependencies for a given tool.
func (to *ToolOrchestrator) GetDependencies(toolName string) []string {
	to.mu.RLock()
	defer to.mu.RUnlock()

	if deps, exists := to.Dependencies[toolName]; exists {
		return append([]string(nil), deps...) // Return a copy
	}
	return []string{}
}

// GetWorkflowCount returns the total number of recorded workflows.
func (to *ToolOrchestrator) GetWorkflowCount() int {
	to.mu.RLock()
	defer to.mu.RUnlock()
	return len(to.Workflows)
}

// SimpleEmbeddingService provides a basic embedding service implementation.
type SimpleEmbeddingService struct {
	dimensions int
}

// NewSimpleEmbeddingService creates a new simple embedding service.
func NewSimpleEmbeddingService(dimensions int) *SimpleEmbeddingService {
	return &SimpleEmbeddingService{
		dimensions: dimensions,
	}
}

// GenerateEmbedding generates a simple embedding based on text characteristics.
// This is a placeholder implementation - in production, use a proper embedding model.
func (s *SimpleEmbeddingService) GenerateEmbedding(ctx context.Context, text string) ([]float64, error) {
	embedding := make([]float64, s.dimensions)

	// Simple hash-based embedding (not suitable for production)
	for i := 0; i < s.dimensions; i++ {
		hash := 0
		for j, char := range text {
			hash = hash*31 + int(char) + i + j
		}
		embedding[i] = math.Sin(float64(hash)) // Normalize to [-1, 1]
	}

	// L2 normalize
	norm := 0.0
	for _, val := range embedding {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding, nil
}

// CosineSimilarity calculates cosine similarity between two vectors.
func (s *SimpleEmbeddingService) CosineSimilarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0.0
	}

	var dotProduct, norm1, norm2 float64
	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// OptimizeInteraction optimizes a single MCP interaction using learned patterns.
func (m *MCPOptimizer) OptimizeInteraction(ctx context.Context, context string, toolName string) (*MCPInteraction, error) {
	logger := logging.GetLogger()

	// Get similar patterns
	patterns, err := m.PatternCollector.GetSimilarPatterns(ctx, context, toolName)
	if err != nil {
		return nil, fmt.Errorf("failed to get similar patterns: %w", err)
	}

	if len(patterns) == 0 {
		logger.Debug(ctx, "No similar patterns found for context and tool: %s", toolName)
		return nil, fmt.Errorf("no similar patterns available")
	}

	// Find similar interactions using embeddings
	similarInteractions, err := m.SimilarityMatcher.FindSimilarInteractions(ctx, context, patterns)
	if err != nil {
		return nil, fmt.Errorf("failed to find similar interactions: %w", err)
	}

	// Select optimal examples
	optimalExamples, err := m.ExampleSelector.SelectOptimalExamples(ctx, similarInteractions)
	if err != nil {
		return nil, fmt.Errorf("failed to select optimal examples: %w", err)
	}

	if len(optimalExamples) == 0 {
		return nil, fmt.Errorf("no optimal examples found")
	}

	// Return the best example (highest weighted)
	bestExample := optimalExamples[0]
	logger.Info(ctx, "Selected optimal MCP interaction for tool %s", toolName)

	return &bestExample, nil
}

// LearnFromInteraction learns from a new MCP interaction.
func (m *MCPOptimizer) LearnFromInteraction(ctx context.Context, interaction MCPInteraction) error {
	if !m.Config.LearningEnabled {
		return nil
	}

	logger := logging.GetLogger()

	// Add to pattern collection
	if err := m.PatternCollector.AddInteraction(ctx, interaction); err != nil {
		return fmt.Errorf("failed to add interaction to pattern collector: %w", err)
	}

	// Record success/failure in example selector
	m.ExampleSelector.RecordSuccess(interaction, interaction.Success)

	// Update metrics
	metrics := MCPMetrics{
		Timestamp: time.Now(),
		ToolSelectionAccuracy: func() float64 {
			if interaction.Success {
				return 1.0
			}
			return 0.0
		}(),
		ParameterOptimality: func() float64 {
			if interaction.Success {
				return 1.0
			}
			return 0.5
		}(),
		ExecutionSuccessRate: func() float64 {
			if interaction.Success {
				return 1.0
			}
			return 0.0
		}(),
		AverageExecutionTime:  interaction.ExecutionTime.Seconds(),
		InteractionsProcessed: 1,
	}
	m.MetricsEvaluator.RecordMetrics(ctx, metrics)

	logger.Debug(ctx, "Learned from MCP interaction: tool=%s, success=%t", interaction.ToolName, interaction.Success)
	return nil
}

// GetOptimizationStats returns current optimization statistics.
func (m *MCPOptimizer) GetOptimizationStats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"total_patterns":       m.PatternCollector.GetPatternCount(),
		"total_workflows":      m.ToolOrchestrator.GetWorkflowCount(),
		"latest_metrics":       m.MetricsEvaluator.GetLatestMetrics(),
		"average_metrics":      m.MetricsEvaluator.GetAverageMetrics(),
		"learning_enabled":     m.Config.LearningEnabled,
		"similarity_threshold": m.Config.SimilarityThreshold,
	}
}

// Helper function to calculate average of float64 slice.
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
