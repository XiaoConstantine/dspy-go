package tools

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// PerformanceMetrics tracks tool execution statistics.
type PerformanceMetrics struct {
	ExecutionCount    int64         `json:"execution_count"`
	SuccessCount      int64         `json:"success_count"`
	FailureCount      int64         `json:"failure_count"`
	AverageLatency    time.Duration `json:"average_latency"`
	LastExecutionTime time.Time     `json:"last_execution_time"`
	SuccessRate       float64       `json:"success_rate"`
	ReliabilityScore  float64       `json:"reliability_score"` // 0.0 to 1.0
}

// ToolCapability represents a specific capability a tool can handle.
type ToolCapability struct {
	Name        string  `json:"name"`
	Confidence  float64 `json:"confidence"` // 0.0 to 1.0
	Description string  `json:"description"`
}

// ToolScore represents the computed score for tool selection.
type ToolScore struct {
	Tool             core.Tool `json:"-"`
	MatchScore       float64   `json:"match_score"`       // How well it matches the intent
	PerformanceScore float64   `json:"performance_score"` // Based on historical performance
	CapabilityScore  float64   `json:"capability_score"`  // Based on capability match
	FinalScore       float64   `json:"final_score"`       // Weighted combination
}

// ToolSelector defines the interface for tool selection algorithms.
type ToolSelector interface {
	SelectBest(ctx context.Context, intent string, candidates []ToolScore) (core.Tool, error)
	ScoreTools(ctx context.Context, intent string, tools []core.Tool) ([]ToolScore, error)
}

// SmartToolRegistry provides intelligent tool selection and management.
type SmartToolRegistry struct {
	mu           sync.RWMutex
	tools        map[string]core.Tool
	capabilities map[string][]ToolCapability
	performance  map[string]*PerformanceMetrics
	selector     ToolSelector
	fallbacks    map[string][]string // intent -> fallback tool names
	mcpDiscovery MCPDiscoveryService
}

// MCPDiscoveryService handles automatic tool discovery from MCP servers.
type MCPDiscoveryService interface {
	DiscoverTools(ctx context.Context) ([]core.Tool, error)
	Subscribe(callback func(tools []core.Tool)) error
}

// SmartToolRegistryConfig configures the smart registry.
type SmartToolRegistryConfig struct {
	Selector                   ToolSelector
	MCPDiscovery               MCPDiscoveryService
	AutoDiscoveryEnabled       bool
	PerformanceTrackingEnabled bool
	FallbackEnabled            bool
}

// NewSmartToolRegistry creates a new smart tool registry.
func NewSmartToolRegistry(config *SmartToolRegistryConfig) *SmartToolRegistry {
	if config.Selector == nil {
		config.Selector = NewBayesianToolSelector()
	}

	registry := &SmartToolRegistry{
		tools:        make(map[string]core.Tool),
		capabilities: make(map[string][]ToolCapability),
		performance:  make(map[string]*PerformanceMetrics),
		selector:     config.Selector,
		fallbacks:    make(map[string][]string),
		mcpDiscovery: config.MCPDiscovery,
	}

	// Start auto-discovery if enabled
	if config.AutoDiscoveryEnabled && config.MCPDiscovery != nil {
		go registry.startAutoDiscovery()
	}

	return registry
}

// Register adds a tool to the registry with capability analysis.
func (r *SmartToolRegistry) Register(tool core.Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if tool == nil {
		return errors.New(errors.InvalidInput, "cannot register a nil tool")
	}

	name := tool.Name()
	if _, exists := r.tools[name]; exists {
		return errors.WithFields(errors.New(errors.InvalidInput, "tool already registered"), errors.Fields{
			"tool_name": name,
		})
	}

	r.tools[name] = tool

	// Initialize performance metrics
	r.performance[name] = &PerformanceMetrics{
		ReliabilityScore: 0.5, // Start with neutral score
	}

	// Extract capabilities from tool metadata
	r.extractCapabilities(tool)

	return nil
}

// Get retrieves a tool by name.
func (r *SmartToolRegistry) Get(name string) (core.Tool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tool, exists := r.tools[name]
	if !exists {
		return nil, errors.WithFields(errors.New(errors.ResourceNotFound, "tool not found"), errors.Fields{
			"tool_name": name,
		})
	}
	return tool, nil
}

// List returns all registered tools.
func (r *SmartToolRegistry) List() []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	list := make([]core.Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		list = append(list, tool)
	}
	return list
}

// Match finds and ranks tools for a given intent using intelligent selection.
func (r *SmartToolRegistry) Match(intent string) []core.Tool {
	ctx := context.Background()
	scores, err := r.selector.ScoreTools(ctx, intent, r.List())
	if err != nil {
		// Fallback to simple matching
		return r.simpleMatch(intent)
	}

	// Sort by final score descending
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].FinalScore > scores[j].FinalScore
	})

	// Extract tools from scores
	tools := make([]core.Tool, len(scores))
	for i, score := range scores {
		tools[i] = score.Tool
	}

	return tools
}

// SelectBest selects the best tool for a given intent.
func (r *SmartToolRegistry) SelectBest(ctx context.Context, intent string) (core.Tool, error) {
	tools := r.List()
	if len(tools) == 0 {
		return nil, errors.New(errors.ResourceNotFound, "no tools available")
	}

	scores, err := r.selector.ScoreTools(ctx, intent, tools)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "failed to score tools")
	}

	if len(scores) == 0 {
		return nil, errors.New(errors.ResourceNotFound, "no suitable tools found")
	}

	// Sort by final score and select the best
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].FinalScore > scores[j].FinalScore
	})

	bestTool := scores[0].Tool

	// Try fallbacks if the best tool has low reliability
	if r.performance[bestTool.Name()].ReliabilityScore < 0.3 {
		if fallbackTool := r.getFallback(intent, bestTool.Name()); fallbackTool != nil {
			return fallbackTool, nil
		}
	}

	return bestTool, nil
}

// ExecuteWithTracking executes a tool and tracks performance metrics.
func (r *SmartToolRegistry) ExecuteWithTracking(ctx context.Context, toolName string, params map[string]interface{}) (core.ToolResult, error) {
	tool, err := r.Get(toolName)
	if err != nil {
		return core.ToolResult{}, err
	}

	startTime := time.Now()
	result, execErr := tool.Execute(ctx, params)
	latency := time.Since(startTime)

	// Update performance metrics
	r.updatePerformanceMetrics(toolName, latency, execErr == nil)

	return result, execErr
}

// AddFallback adds a fallback tool for a specific intent.
func (r *SmartToolRegistry) AddFallback(intent string, fallbackToolName string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[fallbackToolName]; !exists {
		return errors.WithFields(errors.New(errors.ResourceNotFound, "fallback tool not found"), errors.Fields{
			"tool_name": fallbackToolName,
		})
	}

	r.fallbacks[intent] = append(r.fallbacks[intent], fallbackToolName)
	return nil
}

// GetPerformanceMetrics returns performance metrics for a tool.
func (r *SmartToolRegistry) GetPerformanceMetrics(toolName string) (*PerformanceMetrics, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics, exists := r.performance[toolName]
	if !exists {
		return nil, errors.WithFields(errors.New(errors.ResourceNotFound, "metrics not found"), errors.Fields{
			"tool_name": toolName,
		})
	}

	// Return a copy to prevent external modification
	metricsCopy := *metrics
	return &metricsCopy, nil
}

// GetCapabilities returns the capabilities for a tool.
func (r *SmartToolRegistry) GetCapabilities(toolName string) ([]ToolCapability, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	capabilities, exists := r.capabilities[toolName]
	if !exists {
		return nil, errors.WithFields(errors.New(errors.ResourceNotFound, "capabilities not found"), errors.Fields{
			"tool_name": toolName,
		})
	}

	return capabilities, nil
}

// Private methods

func (r *SmartToolRegistry) extractCapabilities(tool core.Tool) {
	metadata := tool.Metadata()
	if metadata == nil {
		return
	}

	capabilities := make([]ToolCapability, 0, len(metadata.Capabilities))
	for _, cap := range metadata.Capabilities {
		capabilities = append(capabilities, ToolCapability{
			Name:        cap,
			Confidence:  1.0, // Default confidence
			Description: fmt.Sprintf("Capability: %s", cap),
		})
	}

	// Add capability based on tool description analysis
	if metadata.Description != "" {
		inferredCaps := r.inferCapabilitiesFromDescription(metadata.Description)
		capabilities = append(capabilities, inferredCaps...)
	}

	r.capabilities[tool.Name()] = capabilities
}

func (r *SmartToolRegistry) inferCapabilitiesFromDescription(description string) []ToolCapability {
	var capabilities []ToolCapability
	desc := strings.ToLower(description)

	// Simple keyword-based capability inference
	keywordMap := map[string]string{
		"search":    "search",
		"find":      "search",
		"query":     "search",
		"read":      "data_access",
		"write":     "data_modification",
		"create":    "creation",
		"generate":  "generation",
		"analyze":   "analysis",
		"process":   "processing",
		"transform": "transformation",
		"validate":  "validation",
	}

	for keyword, capability := range keywordMap {
		if strings.Contains(desc, keyword) {
			capabilities = append(capabilities, ToolCapability{
				Name:        capability,
				Confidence:  0.7, // Lower confidence for inferred capabilities
				Description: fmt.Sprintf("Inferred from keyword: %s", keyword),
			})
		}
	}

	return capabilities
}

func (r *SmartToolRegistry) updatePerformanceMetrics(toolName string, latency time.Duration, success bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	metrics, exists := r.performance[toolName]
	if !exists {
		return
	}

	metrics.ExecutionCount++
	metrics.LastExecutionTime = time.Now()

	if success {
		metrics.SuccessCount++
	} else {
		metrics.FailureCount++
	}

	// Update success rate
	metrics.SuccessRate = float64(metrics.SuccessCount) / float64(metrics.ExecutionCount)

	// Update average latency (exponential moving average)
	if metrics.AverageLatency == 0 {
		metrics.AverageLatency = latency
	} else {
		alpha := 0.1 // Smoothing factor
		metrics.AverageLatency = time.Duration(float64(metrics.AverageLatency)*(1-alpha) + float64(latency)*alpha)
	}

	// Update reliability score (composite of success rate and consistency)
	latencyMs := float64(metrics.AverageLatency.Milliseconds())
	latencyScore := 1.0 / (1.0 + latencyMs/1000.0) // Penalize high latency
	metrics.ReliabilityScore = (metrics.SuccessRate*0.7 + latencyScore*0.3)
}

func (r *SmartToolRegistry) getFallback(intent string, excludeToolName string) core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	fallbacks, exists := r.fallbacks[intent]
	if !exists {
		return nil
	}

	for _, fallbackName := range fallbacks {
		if fallbackName != excludeToolName {
			if tool, exists := r.tools[fallbackName]; exists {
				return tool
			}
		}
	}

	return nil
}

func (r *SmartToolRegistry) simpleMatch(intent string) []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var matches []core.Tool
	lowerIntent := strings.ToLower(intent)

	for name, tool := range r.tools {
		if strings.Contains(lowerIntent, strings.ToLower(name)) {
			matches = append(matches, tool)
			continue
		}

		// Check tool description
		if metadata := tool.Metadata(); metadata != nil {
			if strings.Contains(strings.ToLower(metadata.Description), lowerIntent) {
				matches = append(matches, tool)
				continue
			}
		}
	}

	return matches
}

func (r *SmartToolRegistry) startAutoDiscovery() {
	if r.mcpDiscovery == nil {
		return
	}

	// Subscribe to tool discovery updates
	err := r.mcpDiscovery.Subscribe(func(tools []core.Tool) {
		for _, tool := range tools {
			// Only register if not already present
			if _, err := r.Get(tool.Name()); err != nil {
				if regErr := r.Register(tool); regErr != nil {
					// Log error but continue with other tools
					continue
				}
			}
		}
	})

	if err != nil {
		// Log error but don't fail
		return
	}

	// Perform initial discovery
	ctx := context.Background()
	tools, err := r.mcpDiscovery.DiscoverTools(ctx)
	if err != nil {
		return
	}

	for _, tool := range tools {
		if regErr := r.Register(tool); regErr != nil {
			// Log error but continue
			continue
		}
	}
}

// Ensure SmartToolRegistry implements the ToolRegistry interface.
var _ core.ToolRegistry = (*SmartToolRegistry)(nil)
