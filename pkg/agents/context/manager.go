package context

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// Manager orchestrates all Manus-inspired context engineering patterns.
// This is the unified interface that provides 10x cost reduction through
// sophisticated context management, KV-cache optimization, and attention manipulation.
type Manager struct {
	mu sync.RWMutex

	// Core components implementing Manus patterns
	memory      *FileSystemMemory
	cacheOpt    *CacheOptimizer
	todoMgr     *TodoManager
	errorRetain *ErrorRetainer
	compressor  *Compressor
	diversifier *ContextDiversifier

	// Manager state
	sessionID   string
	agentID     string
	config      Config
	isEnabled   bool

	// Performance metrics
	totalRequests      int64
	compressionSavings int64
	errorsPrevented   int64
	diversityRotations int64

	// Context building state
	contextCache      map[string]CachedContext
	lastContextUpdate time.Time
	contextVersion    int64
}

// CachedContext represents a processed context ready for agent consumption.
type CachedContext struct {
	Content         string                 `json:"content"`
	Checksum        string                 `json:"checksum"`
	Timestamp       time.Time              `json:"timestamp"`
	CacheOptimized  bool                   `json:"cache_optimized"`
	Compressed      bool                   `json:"compressed"`
	Diversified     bool                   `json:"diversified"`
	TokenCount      int                    `json:"token_count"`
	ProcessingTime  time.Duration          `json:"processing_time"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ContextRequest specifies how context should be built and optimized.
type ContextRequest struct {
	// Content inputs
	Observations    []string               `json:"observations"`
	CurrentTask     string                 `json:"current_task"`
	AdditionalData  map[string]interface{} `json:"additional_data"`

	// Optimization preferences
	PrioritizeCache     bool                `json:"prioritize_cache"`
	CompressionPriority CompressionPriority `json:"compression_priority"`
	AllowDiversification bool               `json:"allow_diversification"`
	IncludeErrors       bool                `json:"include_errors"`
	IncludeTodos        bool                `json:"include_todos"`

	// Context constraints
	MaxTokens       int     `json:"max_tokens"`
	MinCacheEfficiency float64 `json:"min_cache_efficiency"`
}

// ContextResponse provides the optimized context plus detailed metrics.
type ContextResponse struct {
	// Optimized context ready for agent
	Context         string                 `json:"context"`

	// Performance metrics
	ProcessingTime  time.Duration          `json:"processing_time"`
	TokenCount      int                    `json:"token_count"`
	CacheHitRate    float64                `json:"cache_hit_rate"`
	CompressionRatio float64               `json:"compression_ratio"`
	DiversityScore  float64                `json:"diversity_score"`

	// Applied optimizations
	OptimizationsApplied []string           `json:"optimizations_applied"`
	CostSavings         float64            `json:"cost_savings"`

	// Metadata
	ContextVersion      int64              `json:"context_version"`
	Metadata            map[string]interface{} `json:"metadata"`
}

// NewManager creates a new context manager with all Manus patterns enabled.
func NewManager(sessionID, agentID, baseDir string, config Config) (*Manager, error) {
	// Validate required parameters
	if sessionID == "" {
		return nil, fmt.Errorf("sessionID cannot be empty")
	}
	if agentID == "" {
		return nil, fmt.Errorf("agentID cannot be empty")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Initialize filesystem memory
	memory, err := NewFileSystemMemory(baseDir, sessionID, agentID, config.Memory)
	if err != nil {
		return nil, fmt.Errorf("failed to create filesystem memory: %w", err)
	}

	// Initialize all components
	cacheOpt := NewCacheOptimizer(config.Cache)
	todoMgr := NewTodoManager(memory, config.Todo)
	errorRetain := NewErrorRetainer(memory, config)
	compressor := NewCompressor(memory, config)
	diversifier := NewContextDiversifier(memory, config)

	manager := &Manager{
		memory:      memory,
		cacheOpt:    cacheOpt,
		todoMgr:     todoMgr,
		errorRetain: errorRetain,
		compressor:  compressor,
		diversifier: diversifier,
		sessionID:   sessionID,
		agentID:     agentID,
		config:      config,
		isEnabled:   true,
		contextCache: make(map[string]CachedContext),
	}

	return manager, nil
}

// BuildOptimizedContext is the CORE method that applies all Manus patterns
// to create highly optimized context for 10x cost reduction.
func (m *Manager) BuildOptimizedContext(ctx context.Context, request ContextRequest) (*ContextResponse, error) {
	if !m.isEnabled {
		return m.buildBasicContext(request)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	startTime := time.Now()
	logger := logging.GetLogger()

	m.totalRequests++
	m.contextVersion++

	// Step 1: Build base context from inputs
	baseContext := m.buildBaseContext(request)

	// Step 2: Apply KV-cache optimization (CRITICAL for cost savings)
	var optimizedContext string
	if request.PrioritizeCache && m.config.EnableCacheOptimization {
		optimizedContext = m.cacheOpt.OptimizePrompt(baseContext, time.Now())
		logger.Debug(ctx, "Applied KV-cache optimization")
	} else {
		optimizedContext = baseContext
	}

	// Step 3: Include todo.md for attention manipulation
	if request.IncludeTodos && m.config.EnableTodoManagement {
		todoContent := m.todoMgr.GetTodoContent()
		if todoContent != "" {
			optimizedContext = fmt.Sprintf("%s\n\n%s", optimizedContext, todoContent)
			logger.Debug(ctx, "Included todo.md for attention manipulation")
		}
	}

	// Step 4: Include error context for learning
	if request.IncludeErrors && m.config.EnableErrorRetention {
		errorContext := m.errorRetain.GetErrorContext()
		if errorContext != "" {
			optimizedContext = fmt.Sprintf("%s\n\n%s", optimizedContext, errorContext)
			logger.Debug(ctx, "Included error context for learning")
		}
	}

	// Step 5: Apply compression if content is large
	compressionResult := CompressionResult{}
	if m.config.EnableCompression && len(optimizedContext) > int(m.config.CompressionThreshold) {
		compressibleContent := CompressibleContent{
			Content:     optimizedContext,
			ContentType: "text",
			Priority:    request.CompressionPriority,
		}

		compressed, result, err := m.compressor.CompressContent(ctx, compressibleContent)
		if err != nil {
			logger.Warn(ctx, "Compression failed: %v", err)
		} else if result.CompressionRatio < 0.9 { // Only use if meaningful compression
			optimizedContext = compressed
			compressionResult = result
			m.compressionSavings += (result.OriginalSize - result.CompressedSize)
			logger.Debug(ctx, "Applied content compression: %.1f%% reduction",
				(1.0-result.CompressionRatio)*100)
		}
	}

	// Step 6: Apply context diversification to prevent few-shot traps
	diversificationResult := DiversificationResult{}
	if request.AllowDiversification && m.config.EnableDiversification {
		diversified, result, err := m.diversifier.DiversifyContext(ctx, optimizedContext, "agent_context")
		if err != nil {
			logger.Warn(ctx, "Diversification failed: %v", err)
		} else if result.Transformation != "none" {
			optimizedContext = diversified
			diversificationResult = result
			m.diversityRotations++
			logger.Debug(ctx, "Applied context diversification: %s", result.Transformation)
		}
	}

	// Step 7: Validate token limits and trim if necessary
	tokenCount := m.cacheOpt.EstimateTokens(optimizedContext)
	if request.MaxTokens > 0 && tokenCount > request.MaxTokens {
		optimizedContext = m.trimToTokenLimit(optimizedContext, request.MaxTokens)
		tokenCount = request.MaxTokens
		logger.Debug(ctx, "Trimmed context to %d tokens", tokenCount)
	}

	// Step 8: Cache the result for potential reuse
	cachedContext := CachedContext{
		Content:        optimizedContext,
		Checksum:       fmt.Sprintf("%x", sha256.Sum256([]byte(optimizedContext))), // Content checksum
		Timestamp:      time.Now(),
		CacheOptimized: request.PrioritizeCache,
		Compressed:     compressionResult.Method != "",
		Diversified:    diversificationResult.Transformation != "none",
		TokenCount:     tokenCount,
		ProcessingTime: time.Since(startTime),
	}

	cacheKey := m.generateCacheKey(request)
	m.contextCache[cacheKey] = cachedContext

	// Build response with metrics
	response := &ContextResponse{
		Context:              optimizedContext,
		ProcessingTime:       time.Since(startTime),
		TokenCount:           tokenCount,
		CacheHitRate:         m.cacheOpt.GetHitRate(),
		CompressionRatio:     compressionResult.CompressionRatio,
		DiversityScore:       m.diversifier.GetDiversityHealth().DiversityScore,
		OptimizationsApplied: m.buildOptimizationsList(request, compressionResult, diversificationResult),
		CostSavings:          m.calculateCostSavings(),
		ContextVersion:       m.contextVersion,
		Metadata: map[string]interface{}{
			"session_id":           m.sessionID,
			"agent_id":             m.agentID,
			"compression_method":   compressionResult.Method,
			"diversity_transform":  diversificationResult.Transformation,
			"cache_optimized":      request.PrioritizeCache,
		},
	}

	logger.Info(ctx, "Built optimized context: %d tokens, %.2f cache hit rate, $%.6f estimated savings",
		tokenCount, response.CacheHitRate, response.CostSavings)

	return response, nil
}

// RecordError integrates error recording with the error retention system.
func (m *Manager) RecordError(ctx context.Context, errorType, message string, severity ErrorSeverity, errorCtx map[string]interface{}) {
	if !m.config.EnableErrorRetention {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	reference := m.errorRetain.RecordError(ctx, errorType, message, severity, errorCtx)

	// Check if we should prevent retries
	if m.errorRetain.ShouldPreventRetry(errorType, message, 1) {
		m.errorsPrevented++
	}

	logger := logging.GetLogger()
	logger.Debug(ctx, "Recorded error for learning: %s", reference)
}

// RecordSuccess records successful operations for pattern reinforcement.
func (m *Manager) RecordSuccess(ctx context.Context, successType, description string, successCtx map[string]interface{}) {
	if !m.config.EnableErrorRetention {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	reference := m.errorRetain.RecordSuccess(ctx, successType, description, successCtx)

	logger := logging.GetLogger()
	logger.Debug(ctx, "Recorded success for pattern reinforcement: %s", reference)
}

// UpdateTodos manages the todo.md attention manipulation system.
func (m *Manager) UpdateTodos(ctx context.Context, todos []TodoItem) error {
	if !m.config.EnableTodoManagement {
		return nil
	}

	return m.todoMgr.UpdateTodos(ctx, todos)
}

// AddTodo adds a new task to the attention management system.
func (m *Manager) AddTodo(ctx context.Context, description string, priority int) error {
	if !m.config.EnableTodoManagement {
		return nil
	}

	return m.todoMgr.AddTodo(ctx, description, priority)
}

// SetActiveTodo marks a specific todo as currently active.
func (m *Manager) SetActiveTodo(ctx context.Context, todoID string) error {
	if !m.config.EnableTodoManagement {
		return nil
	}

	return m.todoMgr.SetActive(ctx, todoID)
}

// CompleteTodo marks a todo as completed and moves it to completed list.
func (m *Manager) CompleteTodo(ctx context.Context, todoID string) error {
	if !m.config.EnableTodoManagement {
		return nil
	}

	return m.todoMgr.CompleteTodo(ctx, todoID)
}

// GetPerformanceMetrics returns comprehensive performance and cost metrics.
func (m *Manager) GetPerformanceMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	cacheMetrics := m.cacheOpt.GetMetrics()
	diversityHealth := m.diversifier.GetDiversityHealth()
	errorMetrics := m.errorRetain.GetLearningMetrics()
	compressionStats := m.compressor.GetCompressionStats()
	todoMetrics := m.todoMgr.GetMetrics()

	return map[string]interface{}{
		// Overall manager metrics
		"total_requests":       m.totalRequests,
		"context_version":      m.contextVersion,
		"last_context_update":  m.lastContextUpdate,
		"enabled":              m.isEnabled,

		// Cost savings metrics
		"cache_hit_rate":       cacheMetrics.HitRate,
		"cache_cost_savings":   cacheMetrics.CostSavings,
		"compression_savings":  m.compressionSavings,
		"total_cost_savings":   m.calculateCostSavings(),

		// Performance metrics
		"errors_prevented":     m.errorsPrevented,
		"diversity_rotations":  m.diversityRotations,
		"compression_ratio":    compressionStats["average_compression_ratio"],

		// Component-specific metrics
		"cache":       cacheMetrics,
		"diversity":   diversityHealth,
		"errors":      errorMetrics,
		"compression": compressionStats,
		"todos":       todoMetrics,
	}
}

// GetHealthStatus returns overall health of the context management system.
func (m *Manager) GetHealthStatus() map[string]interface{} {
	metrics := m.GetPerformanceMetrics()

	health := map[string]interface{}{
		"overall_status": "healthy",
		"components":     make(map[string]string),
		"warnings":       make([]string, 0),
		"recommendations": make([]string, 0),
	}

	// Check cache health
	cacheHitRate := metrics["cache_hit_rate"].(float64)
	if cacheHitRate >= 0.8 {
		health["components"].(map[string]string)["cache"] = "excellent"
	} else if cacheHitRate >= 0.6 {
		health["components"].(map[string]string)["cache"] = "good"
	} else {
		health["components"].(map[string]string)["cache"] = "needs_attention"
		health["warnings"] = append(health["warnings"].([]string), "Low cache hit rate detected")
	}

	// Check diversity health
	diversityScore := metrics["diversity"].(DiversityMetrics).DiversityScore
	if diversityScore >= 0.7 {
		health["components"].(map[string]string)["diversity"] = "healthy"
	} else {
		health["components"].(map[string]string)["diversity"] = "stagnating"
		health["recommendations"] = append(health["recommendations"].([]string), "Consider forcing template rotation")
	}

	// Check error learning
	learningRate := metrics["errors"].(map[string]interface{})["learning_rate"].(float64)
	if learningRate >= 0.5 {
		health["components"].(map[string]string)["error_learning"] = "effective"
	} else {
		health["components"].(map[string]string)["error_learning"] = "limited"
	}

	return health
}

// Enable/Disable allows toggling context management features.
func (m *Manager) SetEnabled(enabled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.isEnabled = enabled
}

// Private helper methods

func (m *Manager) buildBaseContext(request ContextRequest) string {
	var builder strings.Builder

	// Add current task if provided
	if request.CurrentTask != "" {
		builder.WriteString(fmt.Sprintf("## Current Task\n%s\n\n", request.CurrentTask))
	}

	// Add observations (with filesystem storage for large content)
	if len(request.Observations) > 0 {
		builder.WriteString("## Observations\n")
		for i, obs := range request.Observations {
			// Store large observations in filesystem and use references
			if m.config.EnableFileSystemMemory && len(obs) > int(m.config.CompressionThreshold) {
				ref, err := m.memory.StoreLargeObservation(context.TODO(), fmt.Sprintf("obs_%d_%d", i, time.Now().UnixNano()), []byte(obs), map[string]interface{}{
					"observation_index": i,
					"size": len(obs),
				})
				if err != nil {
					// Fallback to direct inclusion if storage fails
					builder.WriteString(fmt.Sprintf("%d. %s\n", i+1, obs))
				} else {
					builder.WriteString(fmt.Sprintf("%d. %s\n", i+1, ref))
				}
			} else {
				builder.WriteString(fmt.Sprintf("%d. %s\n", i+1, obs))
			}
		}
		builder.WriteString("\n")
	}

	// Add additional data
	if len(request.AdditionalData) > 0 {
		builder.WriteString("## Additional Context\n")
		for key, value := range request.AdditionalData {
			builder.WriteString(fmt.Sprintf("- %s: %v\n", key, value))
		}
		builder.WriteString("\n")
	}

	return builder.String()
}

func (m *Manager) buildBasicContext(request ContextRequest) (*ContextResponse, error) {
	// Fallback for when context management is disabled
	baseContext := m.buildBaseContext(request)
	tokenCount := m.cacheOpt.EstimateTokens(baseContext)

	return &ContextResponse{
		Context:              baseContext,
		ProcessingTime:       time.Millisecond,
		TokenCount:           tokenCount,
		CacheHitRate:         0.0,
		CompressionRatio:     1.0,
		DiversityScore:       0.0,
		OptimizationsApplied: []string{"none"},
		CostSavings:          0.0,
		ContextVersion:       m.contextVersion,
		Metadata: map[string]interface{}{
			"context_management_enabled": false,
		},
	}, nil
}

func (m *Manager) buildOptimizationsList(request ContextRequest, compression CompressionResult, diversity DiversificationResult) []string {
	var optimizations []string

	if request.PrioritizeCache && m.config.EnableCacheOptimization {
		optimizations = append(optimizations, "kv_cache_optimization")
	}

	if compression.Method != "" {
		optimizations = append(optimizations, fmt.Sprintf("compression_%s", compression.Method))
	}

	if diversity.Transformation != "none" {
		optimizations = append(optimizations, fmt.Sprintf("diversification_%s", diversity.Transformation))
	}

	if request.IncludeTodos && m.config.EnableTodoManagement {
		optimizations = append(optimizations, "attention_manipulation")
	}

	if request.IncludeErrors && m.config.EnableErrorRetention {
		optimizations = append(optimizations, "error_learning")
	}

	if len(optimizations) == 0 {
		optimizations = append(optimizations, "none")
	}

	return optimizations
}

func (m *Manager) calculateCostSavings() float64 {
	cacheMetrics := m.cacheOpt.GetMetrics()
	compressionStats := m.compressor.GetCompressionStats()

	cacheSavings := cacheMetrics.CostSavings
	compressionSavingsFloat := float64(compressionStats["total_bytes_saved"].(int64)) * (3.0 - 0.30) / 1000000.0

	return cacheSavings + compressionSavingsFloat
}

func (m *Manager) generateCacheKey(request ContextRequest) string {
	// Generate a cache key based on request parameters
	keyData := map[string]interface{}{
		"task":         request.CurrentTask,
		"observations": len(request.Observations),
		"cache":        request.PrioritizeCache,
		"compression":  request.CompressionPriority,
		"diversity":    request.AllowDiversification,
	}

	keyBytes, _ := json.Marshal(keyData)
	return fmt.Sprintf("context_%x", keyBytes)
}

func (m *Manager) trimToTokenLimit(content string, maxTokens int) string {
	// Simple token-based trimming - estimate 4 chars per token
	maxChars := maxTokens * 4
	if len(content) <= maxChars {
		return content
	}

	// Trim from the middle to preserve beginning and end
	keepStart := maxChars / 3
	keepEnd := maxChars / 3

	start := content[:keepStart]
	end := content[len(content)-keepEnd:]
	trimmed := fmt.Sprintf("%s\n\n[... content trimmed for token limit ...]\n\n%s", start, end)

	return trimmed
}
