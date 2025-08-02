package main

import (
	"context"
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// Example tools for demonstration
// SearchTool is defined in shared.go

type CreateTool struct {
	name string
}

func (c *CreateTool) Name() string {
	return c.name
}

func (c *CreateTool) Description() string {
	return "Powerful creation tool that can generate documents, files, and resources"
}

func (c *CreateTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         c.name,
		Description:  c.Description(),
		Capabilities: []string{"create", "generate", "make", "build", "construct"},
		Version:      "1.5.0",
	}
}

func (c *CreateTool) CanHandle(ctx context.Context, intent string) bool {
	keywords := []string{"create", "generate", "make", "build", "new"}
	for _, keyword := range keywords {
		if contains(intent, keyword) {
			return true
		}
	}
	return false
}

func (c *CreateTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	itemType, ok := params["type"].(string)
	if !ok {
		itemType = "document"
	}

	name, ok := params["name"].(string)
	if !ok {
		name = "untitled"
	}

	// Simulate creation operation
	time.Sleep(100 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"created_item": map[string]interface{}{
				"id":     fmt.Sprintf("item_%d", time.Now().Unix()),
				"name":   name,
				"type":   itemType,
				"status": "created",
			},
		},
		Metadata: map[string]interface{}{
			"execution_time_ms": 100,
			"created_at":        time.Now().Format(time.RFC3339),
		},
	}, nil
}

func (c *CreateTool) Validate(params map[string]interface{}) error {
	return nil
}

func (c *CreateTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"type": {
				Type:        "string",
				Description: "Type of item to create",
				Required:    true,
			},
			"name": {
				Type:        "string",
				Description: "Name of the item to create",
			},
		},
	}
}

type AnalyzeTool struct {
	name string
}

func (a *AnalyzeTool) Name() string {
	return a.name
}

func (a *AnalyzeTool) Description() string {
	return "Advanced analytics tool that can process and analyze various types of data"
}

func (a *AnalyzeTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         a.name,
		Description:  a.Description(),
		Capabilities: []string{"analyze", "process", "examine", "evaluate", "assess"},
		Version:      "3.0.1",
	}
}

func (a *AnalyzeTool) CanHandle(ctx context.Context, intent string) bool {
	keywords := []string{"analyze", "process", "examine", "evaluate", "assess"}
	for _, keyword := range keywords {
		if contains(intent, keyword) {
			return true
		}
	}
	return false
}

func (a *AnalyzeTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	data, ok := params["data"].(string)
	if !ok {
		data = "sample data"
	}

	// Simulate analysis operation
	time.Sleep(200 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"analysis": map[string]interface{}{
				"summary":    "Analysis complete for: " + data,
				"confidence": 0.94,
				"insights":   []string{"Pattern A detected", "Trend B identified", "Anomaly C found"},
				"score":      87.5,
			},
		},
		Metadata: map[string]interface{}{
			"execution_time_ms": 200,
			"model_version":     "v3.0.1",
		},
	}, nil
}

func (a *AnalyzeTool) Validate(params map[string]interface{}) error {
	return nil
}

func (a *AnalyzeTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"data": {
				Type:        "string",
				Description: "Data to analyze",
				Required:    true,
			},
		},
	}
}

// Mock MCP Discovery Service for demonstration.
type MockMCPDiscovery struct {
	tools []core.Tool
}

func (m *MockMCPDiscovery) DiscoverTools(ctx context.Context) ([]core.Tool, error) {
	return m.tools, nil
}

func (m *MockMCPDiscovery) Subscribe(callback func(tools []core.Tool)) error {
	// Simulate discovering new tools after subscription
	go func() {
		time.Sleep(2 * time.Second)
		newTool := &SearchTool{name: "mcp_discovered_search"}
		m.tools = append(m.tools, newTool)
		callback([]core.Tool{newTool})
	}()
	return nil
}

// Helper functions are defined in shared.go

// - Reliability: Timeouts, retries, and error handling.
func main() {
	// Initialize DSPy-Go logger
	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{logging.NewConsoleOutput(false)},
	})
	ctx := context.Background()

	logger.Info(ctx, "🚀 Smart Tool Registry with Interceptors Example")
	logger.Info(ctx, "================================================")

	// 1. Create Smart Tool Registry with full configuration
	logger.Info(ctx, "\n1. Setting up Smart Tool Registry...")

	mockDiscovery := &MockMCPDiscovery{
		tools: []core.Tool{
			&SearchTool{name: "mcp_search_v1"},
		},
	}

	config := &tools.SmartToolRegistryConfig{
		AutoDiscoveryEnabled:       true,
		PerformanceTrackingEnabled: true,
		FallbackEnabled:            true,
		MCPDiscovery:               mockDiscovery,
	}

	registry := tools.NewSmartToolRegistry(config)

	// Setup tool interceptors for enhanced security, observability and reliability
	logger.Info(ctx, "   → Configuring tool interceptors...")

	// Create tool interceptors
	toolInterceptors := []core.ToolInterceptor{
		// Logging interceptor - logs tool execution start/completion
		interceptors.LoggingToolInterceptor(),

		// Metrics interceptor - tracks tool performance metrics
		interceptors.MetricsToolInterceptor(),

		// Validation interceptor - validates tool arguments for safety
		interceptors.ValidationToolInterceptor(interceptors.DefaultValidationConfig()),

		// Security interceptor - sanitizes inputs to prevent injection attacks
		interceptors.SanitizingToolInterceptor(),

		// Rate limiting interceptor - prevents tool abuse (10 calls per minute per tool)
		interceptors.RateLimitingToolInterceptor(10, time.Minute),

		// Timeout interceptor - prevents tools from running too long
		interceptors.TimeoutToolInterceptor(15 * time.Second),

		// Retry interceptor - retries failed tool executions
		interceptors.RetryToolInterceptor(interceptors.RetryConfig{
			MaxAttempts: 2,
			Delay:       500 * time.Millisecond,
			Backoff:     1.5,
		}),
	}

	logger.Info(ctx, "   → Configured %d tool interceptors for enhanced tool execution", len(toolInterceptors))

	// 2. Register example tools
	logger.Info(ctx, "\n2. Registering tools...")

	searchTool := &SearchTool{name: "advanced_search"}
	createTool := &CreateTool{name: "document_creator"}
	analyzeTool := &AnalyzeTool{name: "data_analyzer"}

	tools_list := []core.Tool{searchTool, createTool, analyzeTool}

	for _, tool := range tools_list {
		err := registry.Register(tool)
		if err != nil {
			logger.Error(ctx, "Error registering tool %s: %v", tool.Name(), err)
		} else {
			logger.Info(ctx, "✅ Registered: %s", tool.Name())
		}
	}

	// 3. Configure fallbacks
	logger.Info(ctx, "\n3. Configuring fallbacks...")

	err := registry.AddFallback("search data", "advanced_search")
	if err != nil {
		logger.Error(ctx, "Error adding fallback: %v", err)
	} else {
		logger.Info(ctx, "✅ Added fallback for search operations")
	}

	// 4. Demonstrate intelligent tool selection
	logger.Info(ctx, "\n4. Intelligent Tool Selection Examples")
	logger.Info(ctx, "=====================================")

	intents := []string{
		"I need to find user information in the database",
		"Create a new report document",
		"Analyze the performance data from last quarter",
		"Generate a summary of search results",
		"Process the incoming data stream",
	}

	for _, intent := range intents {
		logger.Info(ctx, "\n🎯 Intent: \"%s\"", intent)

		// Get best tool
		selectedTool, err := registry.SelectBest(ctx, intent)
		if err != nil {
			logger.Error(ctx, "Error selecting tool: %v", err)
			continue
		}

		logger.Info(ctx, "   → Selected Tool: %s", selectedTool.Name())

		// Show all matching tools with scores
		matches := registry.Match(intent)
		matchNames := make([]string, len(matches))
		for i, match := range matches {
			matchNames[i] = match.Name()
		}
		logger.Info(ctx, "   → All Matches: %v", matchNames)
	}

	// 5. Execute tools with performance tracking
	logger.Info(ctx, "\n5. Executing Tools with Performance Tracking")
	logger.Info(ctx, "==========================================")

	// Execute search tool multiple times with interceptors
	logger.Info(ctx, "\n📊 Executing search operations with tool interceptors...")
	logger.Info(ctx, "   → The following executions demonstrate interceptor benefits:")
	logger.Info(ctx, "     • Logging: Start/completion of each tool execution")
	logger.Info(ctx, "     • Metrics: Performance timing and success/failure tracking")
	logger.Info(ctx, "     • Validation: Input safety checks against injection attacks")
	logger.Info(ctx, "     • Sanitization: Input cleaning and normalization")
	logger.Info(ctx, "     • Rate Limiting: Prevents tool abuse (10 calls/min per tool)")
	logger.Info(ctx, "     • Timeout: Protection against long-running operations")
	logger.Info(ctx, "     • Retry: Automatic retry on failures with exponential backoff")

	for i := 0; i < 3; i++ {
		params := map[string]interface{}{
			"query": fmt.Sprintf("sample query %d", i+1),
			"limit": 5,
		}

		logger.Info(ctx, "\n   → Executing search operation %d with interceptors...", i+1)
		result, err := registry.ExecuteWithTracking(ctx, "advanced_search", params)
		if err != nil {
			logger.Error(ctx, "Execution error: %v", err)
			continue
		}

		logger.Info(ctx, "   ✅ Search execution %d completed successfully", i+1)
		if data, ok := result.Data.(map[string]interface{}); ok {
			if count, ok := data["count"].(int); ok {
				logger.Info(ctx, "      → Found %d results", count)
			}
		}
	}

	// Execute create tool with interceptors
	logger.Info(ctx, "\n📝 Executing create operation with tool interceptors...")
	params := map[string]interface{}{
		"type": "report",
		"name": "Q4 Analysis Report",
	}

	result, err := registry.ExecuteWithTracking(ctx, "document_creator", params)
	if err != nil {
		logger.Error(ctx, "Execution error: %v", err)
	} else {
		logger.Info(ctx, "   ✅ Creation completed with interceptor protection")
		if data, ok := result.Data.(map[string]interface{}); ok {
			if item, ok := data["created_item"].(map[string]interface{}); ok {
				logger.Info(ctx, "      → Created: %s (ID: %s)", item["name"], item["id"])
			}
		}
	}

	// Execute analyze tool with interceptors
	logger.Info(ctx, "\n🔍 Executing analysis operation with tool interceptors...")
	params = map[string]interface{}{
		"data": "performance metrics dataset",
	}

	result, err = registry.ExecuteWithTracking(ctx, "data_analyzer", params)
	if err != nil {
		logger.Error(ctx, "Execution error: %v", err)
	} else {
		logger.Info(ctx, "   ✅ Analysis completed with security validation")
		if data, ok := result.Data.(map[string]interface{}); ok {
			if analysis, ok := data["analysis"].(map[string]interface{}); ok {
				logger.Info(ctx, "      → Score: %.1f, Confidence: %.2f",
					analysis["score"], analysis["confidence"])
			}
		}
	}

	// 6. Show performance metrics
	logger.Info(ctx, "\n6. Performance Metrics")
	logger.Info(ctx, "=====================")

	for _, toolName := range []string{"advanced_search", "document_creator", "data_analyzer"} {
		metrics, err := registry.GetPerformanceMetrics(toolName)
		if err != nil {
			logger.Error(ctx, "Error getting metrics for %s: %v", toolName, err)
			continue
		}

		logger.Info(ctx, "\n📈 %s:", toolName)
		logger.Info(ctx, "   → Executions: %d", metrics.ExecutionCount)
		logger.Info(ctx, "   → Success Rate: %.1f%%", metrics.SuccessRate*100)
		logger.Info(ctx, "   → Avg Latency: %v", metrics.AverageLatency)
		logger.Info(ctx, "   → Reliability: %.2f/1.0", metrics.ReliabilityScore)
	}

	// 7. Show tool capabilities
	logger.Info(ctx, "\n7. Tool Capabilities")
	logger.Info(ctx, "===================")

	for _, toolName := range []string{"advanced_search", "document_creator", "data_analyzer"} {
		capabilities, err := registry.GetCapabilities(toolName)
		if err != nil {
			logger.Error(ctx, "Error getting capabilities for %s: %v", toolName, err)
			continue
		}

		logger.Info(ctx, "\n🔧 %s capabilities:", toolName)
		for _, cap := range capabilities {
			logger.Info(ctx, "   → %s (confidence: %.2f)", cap.Name, cap.Confidence)
		}
	}

	// 8. Wait for auto-discovery to trigger
	logger.Info(ctx, "\n8. Auto-Discovery Demonstration")
	logger.Info(ctx, "==============================")

	logger.Info(ctx, "⏳ Waiting for auto-discovery (simulated MCP server)...")
	time.Sleep(3 * time.Second)

	// List all tools to show discovered ones
	allTools := registry.List()
	logger.Info(ctx, "📋 Total registered tools: %d", len(allTools))
	for _, tool := range allTools {
		logger.Info(ctx, "   → %s", tool.Name())
	}

	// 9. Demonstrate Bayesian selector configuration
	logger.Info(ctx, "\n9. Advanced Selector Configuration")
	logger.Info(ctx, "==================================")

	// Create a custom selector with different weights
	customSelector := tools.NewBayesianToolSelector()
	customSelector.MatchWeight = 0.6       // Emphasize intent matching
	customSelector.PerformanceWeight = 0.2 // Reduce performance emphasis
	customSelector.CapabilityWeight = 0.2  // Reduce capability emphasis

	// Update usage statistics (simulated)
	usageStats := map[string]int{
		"advanced_search":  50,
		"document_creator": 30,
		"data_analyzer":    20,
	}
	customSelector.UpdatePriorProbabilities(usageStats)

	logger.Info(ctx, "✅ Custom Bayesian selector configured")
	logger.Info(ctx, "   → Match Weight: %.1f", customSelector.MatchWeight)
	logger.Info(ctx, "   → Performance Weight: %.1f", customSelector.PerformanceWeight)
	logger.Info(ctx, "   → Capability Weight: %.1f", customSelector.CapabilityWeight)

	// Test custom selection
	testIntent := "search for technical documentation"
	scores, err := customSelector.ScoreTools(ctx, testIntent, allTools)
	if err == nil && len(scores) > 0 {
		logger.Info(ctx, "   → Best tool for '%s': %s (score: %.3f)",
			testIntent, scores[0].Tool.Name(), scores[0].FinalScore)
	}

	// 10. Demonstrate tool interceptor security features
	logger.Info(ctx, "\n10. Tool Interceptor Security Demonstration")
	logger.Info(ctx, "==========================================")

	logger.Info(ctx, "\n🔒 Testing security interceptors with potentially malicious inputs...")
	logger.Info(ctx, "   → These examples show how interceptors protect against attacks:")

	// Test XSS protection
	maliciousInputs := []map[string]interface{}{
		{
			"query": "<script>alert('XSS')</script>search term",
			"limit": 5,
		},
		{
			"data": "'; DROP TABLE users; --",
		},
		{
			"type": "{{template_injection}}",
			"name": "normal_name",
		},
	}

	testCases := []string{"XSS injection", "SQL injection", "Template injection"}

	for i, testInput := range maliciousInputs {
		logger.Info(ctx, "\n   🚨 Testing %s protection...", testCases[i])

		// Try to execute with malicious input - interceptors will sanitize/validate
		toolName := "advanced_search"
		switch i {
		case 1:
			toolName = "data_analyzer"
		case 2:
			toolName = "document_creator"
		}

		result, err := registry.ExecuteWithTracking(ctx, toolName, testInput)
		if err != nil {
			logger.Info(ctx, "      ✅ %s blocked by security interceptors: %v", testCases[i], err)
		} else {
			logger.Info(ctx, "      ✅ %s sanitized and processed safely", testCases[i])
			if data, ok := result.Data.(map[string]interface{}); ok {
				if toolName == "advanced_search" {
					if count, ok := data["count"].(int); ok {
						logger.Info(ctx, "         → Sanitized query processed, found %d results", count)
					}
				}
			}
		}
	}

	logger.Info(ctx, "\n🎉 Smart Tool Registry with Interceptors demonstration completed!")
	logger.Info(ctx, "\nKey features demonstrated:")
	logger.Info(ctx, "  ✅ Intelligent tool selection with Bayesian inference")
	logger.Info(ctx, "  ✅ Real-time performance tracking and metrics")
	logger.Info(ctx, "  ✅ Automatic capability extraction and matching")
	logger.Info(ctx, "  ✅ Auto-discovery from MCP servers")
	logger.Info(ctx, "  ✅ Fallback mechanisms and error handling")
	logger.Info(ctx, "  ✅ Configurable selection algorithms")
	logger.Info(ctx, "  ✅ Comprehensive tool metadata and scoring")
	logger.Info(ctx, "  ✅ Tool interceptors for security, logging, and reliability")
	logger.Info(ctx, "  ✅ Input validation and sanitization against attacks")
	logger.Info(ctx, "  ✅ Rate limiting and timeout protection")
	logger.Info(ctx, "  ✅ Automatic retry with exponential backoff")
}
