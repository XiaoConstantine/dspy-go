package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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

// main demonstrates basic Smart Tool Registry usage.
func main() {
	fmt.Println("🚀 Smart Tool Registry Example")
	fmt.Println("==============================")

	// 1. Create Smart Tool Registry with full configuration
	fmt.Println("\n1. Setting up Smart Tool Registry...")

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

	// 2. Register example tools
	fmt.Println("\n2. Registering tools...")

	searchTool := &SearchTool{name: "advanced_search"}
	createTool := &CreateTool{name: "document_creator"}
	analyzeTool := &AnalyzeTool{name: "data_analyzer"}

	tools_list := []core.Tool{searchTool, createTool, analyzeTool}

	for _, tool := range tools_list {
		err := registry.Register(tool)
		if err != nil {
			log.Printf("Error registering tool %s: %v", tool.Name(), err)
		} else {
			fmt.Printf("✅ Registered: %s\n", tool.Name())
		}
	}

	// 3. Configure fallbacks
	fmt.Println("\n3. Configuring fallbacks...")

	err := registry.AddFallback("search data", "advanced_search")
	if err != nil {
		log.Printf("Error adding fallback: %v", err)
	} else {
		fmt.Println("✅ Added fallback for search operations")
	}

	// 4. Demonstrate intelligent tool selection
	fmt.Println("\n4. Intelligent Tool Selection Examples")
	fmt.Println("=====================================")

	intents := []string{
		"I need to find user information in the database",
		"Create a new report document",
		"Analyze the performance data from last quarter",
		"Generate a summary of search results",
		"Process the incoming data stream",
	}

	ctx := context.Background()

	for _, intent := range intents {
		fmt.Printf("\n🎯 Intent: \"%s\"\n", intent)

		// Get best tool
		selectedTool, err := registry.SelectBest(ctx, intent)
		if err != nil {
			log.Printf("Error selecting tool: %v", err)
			continue
		}

		fmt.Printf("   → Selected Tool: %s\n", selectedTool.Name())

		// Show all matching tools with scores
		matches := registry.Match(intent)
		fmt.Printf("   → All Matches: ")
		for i, match := range matches {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Print(match.Name())
		}
		fmt.Println()
	}

	// 5. Execute tools with performance tracking
	fmt.Println("\n5. Executing Tools with Performance Tracking")
	fmt.Println("==========================================")

	// Execute search tool multiple times
	fmt.Println("\n📊 Executing search operations...")
	for i := 0; i < 3; i++ {
		params := map[string]interface{}{
			"query": fmt.Sprintf("sample query %d", i+1),
			"limit": 5,
		}

		result, err := registry.ExecuteWithTracking(ctx, "advanced_search", params)
		if err != nil {
			log.Printf("Execution error: %v", err)
			continue
		}

		fmt.Printf("   ✅ Execution %d completed\n", i+1)
		if data, ok := result.Data.(map[string]interface{}); ok {
			if count, ok := data["count"].(int); ok {
				fmt.Printf("      → Found %d results\n", count)
			}
		}
	}

	// Execute create tool
	fmt.Println("\n📝 Executing create operation...")
	params := map[string]interface{}{
		"type": "report",
		"name": "Q4 Analysis Report",
	}

	result, err := registry.ExecuteWithTracking(ctx, "document_creator", params)
	if err != nil {
		log.Printf("Execution error: %v", err)
	} else {
		fmt.Println("   ✅ Creation completed")
		if data, ok := result.Data.(map[string]interface{}); ok {
			if item, ok := data["created_item"].(map[string]interface{}); ok {
				fmt.Printf("      → Created: %s (ID: %s)\n", item["name"], item["id"])
			}
		}
	}

	// Execute analyze tool
	fmt.Println("\n🔍 Executing analysis operation...")
	params = map[string]interface{}{
		"data": "performance metrics dataset",
	}

	result, err = registry.ExecuteWithTracking(ctx, "data_analyzer", params)
	if err != nil {
		log.Printf("Execution error: %v", err)
	} else {
		fmt.Println("   ✅ Analysis completed")
		if data, ok := result.Data.(map[string]interface{}); ok {
			if analysis, ok := data["analysis"].(map[string]interface{}); ok {
				fmt.Printf("      → Score: %.1f, Confidence: %.2f\n",
					analysis["score"], analysis["confidence"])
			}
		}
	}

	// 6. Show performance metrics
	fmt.Println("\n6. Performance Metrics")
	fmt.Println("=====================")

	for _, toolName := range []string{"advanced_search", "document_creator", "data_analyzer"} {
		metrics, err := registry.GetPerformanceMetrics(toolName)
		if err != nil {
			log.Printf("Error getting metrics for %s: %v", toolName, err)
			continue
		}

		fmt.Printf("\n📈 %s:\n", toolName)
		fmt.Printf("   → Executions: %d\n", metrics.ExecutionCount)
		fmt.Printf("   → Success Rate: %.1f%%\n", metrics.SuccessRate*100)
		fmt.Printf("   → Avg Latency: %v\n", metrics.AverageLatency)
		fmt.Printf("   → Reliability: %.2f/1.0\n", metrics.ReliabilityScore)
	}

	// 7. Show tool capabilities
	fmt.Println("\n7. Tool Capabilities")
	fmt.Println("===================")

	for _, toolName := range []string{"advanced_search", "document_creator", "data_analyzer"} {
		capabilities, err := registry.GetCapabilities(toolName)
		if err != nil {
			log.Printf("Error getting capabilities for %s: %v", toolName, err)
			continue
		}

		fmt.Printf("\n🔧 %s capabilities:\n", toolName)
		for _, cap := range capabilities {
			fmt.Printf("   → %s (confidence: %.2f)\n", cap.Name, cap.Confidence)
		}
	}

	// 8. Wait for auto-discovery to trigger
	fmt.Println("\n8. Auto-Discovery Demonstration")
	fmt.Println("==============================")

	fmt.Println("⏳ Waiting for auto-discovery (simulated MCP server)...")
	time.Sleep(3 * time.Second)

	// List all tools to show discovered ones
	allTools := registry.List()
	fmt.Printf("📋 Total registered tools: %d\n", len(allTools))
	for _, tool := range allTools {
		fmt.Printf("   → %s\n", tool.Name())
	}

	// 9. Demonstrate Bayesian selector configuration
	fmt.Println("\n9. Advanced Selector Configuration")
	fmt.Println("=================================")

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

	fmt.Println("✅ Custom Bayesian selector configured")
	fmt.Printf("   → Match Weight: %.1f\n", customSelector.MatchWeight)
	fmt.Printf("   → Performance Weight: %.1f\n", customSelector.PerformanceWeight)
	fmt.Printf("   → Capability Weight: %.1f\n", customSelector.CapabilityWeight)

	// Test custom selection
	testIntent := "search for technical documentation"
	scores, err := customSelector.ScoreTools(ctx, testIntent, allTools)
	if err == nil && len(scores) > 0 {
		fmt.Printf("   → Best tool for '%s': %s (score: %.3f)\n",
			testIntent, scores[0].Tool.Name(), scores[0].FinalScore)
	}

	fmt.Println("\n🎉 Smart Tool Registry demonstration completed!")
	fmt.Println("\nKey features demonstrated:")
	fmt.Println("  ✅ Intelligent tool selection with Bayesian inference")
	fmt.Println("  ✅ Real-time performance tracking and metrics")
	fmt.Println("  ✅ Automatic capability extraction and matching")
	fmt.Println("  ✅ Auto-discovery from MCP servers")
	fmt.Println("  ✅ Fallback mechanisms and error handling")
	fmt.Println("  ✅ Configurable selection algorithms")
	fmt.Println("  ✅ Comprehensive tool metadata and scoring")
}
