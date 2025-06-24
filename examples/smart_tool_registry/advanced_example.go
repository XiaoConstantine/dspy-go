package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// DatabaseTool - A tool that sometimes fails to demonstrate error handling.
type DatabaseTool struct {
	name        string
	failureRate float64
	execCount   int
}

func (d *DatabaseTool) Name() string {
	return d.name
}

func (d *DatabaseTool) Description() string {
	return "Database query tool that can read from multiple database systems"
}

func (d *DatabaseTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         d.name,
		Description:  d.Description(),
		Capabilities: []string{"database", "query", "read", "sql", "data_access"},
		Version:      "1.0.0",
	}
}

func (d *DatabaseTool) CanHandle(ctx context.Context, intent string) bool {
	keywords := []string{"database", "query", "sql", "table", "record"}
	for _, keyword := range keywords {
		if contains(intent, keyword) {
			return true
		}
	}
	return false
}

func (d *DatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	d.execCount++

	// Simulate random failures based on failure rate
	if rand.Float64() < d.failureRate {
		return core.ToolResult{}, errors.New("database connection timeout")
	}

	query, ok := params["query"].(string)
	if !ok {
		query = "SELECT * FROM users"
	}

	// Simulate database operation
	time.Sleep(time.Duration(50+rand.IntN(100)) * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"rows": []map[string]interface{}{
				{"id": 1, "name": "Alice", "email": "alice@example.com"},
				{"id": 2, "name": "Bob", "email": "bob@example.com"},
			},
			"query":    query,
			"duration": fmt.Sprintf("%dms", 50+rand.IntN(100)),
		},
		Metadata: map[string]interface{}{
			"execution_count": d.execCount,
			"database":        "postgresql",
		},
	}, nil
}

func (d *DatabaseTool) Validate(params map[string]interface{}) error {
	return nil
}

func (d *DatabaseTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "SQL query to execute",
				Required:    true,
			},
		},
	}
}

// BackupDatabaseTool - A more reliable fallback tool.
type BackupDatabaseTool struct {
	name string
}

func (b *BackupDatabaseTool) Name() string {
	return b.name
}

func (b *BackupDatabaseTool) Description() string {
	return "Backup database tool with cached responses for high reliability"
}

func (b *BackupDatabaseTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         b.name,
		Description:  b.Description(),
		Capabilities: []string{"database", "query", "backup", "cache", "reliable"},
		Version:      "2.0.0",
	}
}

func (b *BackupDatabaseTool) CanHandle(ctx context.Context, intent string) bool {
	return contains(intent, "database") || contains(intent, "query")
}

func (b *BackupDatabaseTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	// This tool is highly reliable (backup/cache)
	time.Sleep(20 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"rows": []map[string]interface{}{
				{"id": 1, "name": "Cached Alice", "email": "alice@cache.com"},
			},
			"source": "backup_cache",
			"note":   "Served from backup cache for reliability",
		},
		Metadata: map[string]interface{}{
			"source":      "backup",
			"reliability": "high",
		},
	}, nil
}

func (b *BackupDatabaseTool) Validate(params map[string]interface{}) error {
	return nil
}

func (b *BackupDatabaseTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "Query to execute against backup cache",
			},
		},
	}
}

// FileProcessorTool - Tool with rich metadata.
type FileProcessorTool struct {
	name string
}

func (f *FileProcessorTool) Name() string {
	return f.name
}

func (f *FileProcessorTool) Description() string {
	return "Advanced file processor that can read, write, transform, and analyze various file formats including JSON, CSV, XML, and text files"
}

func (f *FileProcessorTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:        f.name,
		Description: f.Description(),
		Capabilities: []string{
			"file_processing", "read", "write", "transform", "analyze",
			"json", "csv", "xml", "text", "parsing", "validation",
		},
		Version: "3.2.1",
	}
}

func (f *FileProcessorTool) CanHandle(ctx context.Context, intent string) bool {
	keywords := []string{"file", "read", "write", "process", "parse", "json", "csv", "xml"}
	for _, keyword := range keywords {
		if contains(intent, keyword) {
			return true
		}
	}
	return false
}

func (f *FileProcessorTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	operation, ok := params["operation"].(string)
	if !ok {
		operation = "read"
	}

	filename, ok := params["filename"].(string)
	if !ok {
		filename = "sample.json"
	}

	// Simulate file operation
	time.Sleep(30 * time.Millisecond)

	return core.ToolResult{
		Data: map[string]interface{}{
			"operation": operation,
			"filename":  filename,
			"result":    fmt.Sprintf("Successfully %sed file: %s", operation, filename),
			"size":      "1.2MB",
			"format":    "JSON",
		},
		Metadata: map[string]interface{}{
			"processor_version": "3.2.1",
			"timestamp":         time.Now().Format(time.RFC3339),
		},
	}, nil
}

func (f *FileProcessorTool) Validate(params map[string]interface{}) error {
	return nil
}

func (f *FileProcessorTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"operation": {
				Type:        "string",
				Description: "Operation to perform (read, write, transform, analyze)",
				Required:    true,
			},
			"filename": {
				Type:        "string",
				Description: "File to process",
				Required:    true,
			},
		},
	}
}

// Helper functions and SearchTool are defined in shared.go

// Demo function for performance comparison.
//nolint:unused // Used when main() is uncommented
func demonstratePerformanceTracking(registry *tools.SmartToolRegistry) {
	fmt.Println("\n🏃‍♂️ Performance Tracking Demonstration")
	fmt.Println("======================================")

	ctx := context.Background()

	// Execute database tools multiple times to build performance history
	fmt.Println("Executing tools to build performance history...")

	for i := 0; i < 10; i++ {
		params := map[string]interface{}{
			"query": fmt.Sprintf("SELECT * FROM table_%d", i),
		}

		// Execute primary database tool (may fail sometimes)
		_, err := registry.ExecuteWithTracking(ctx, "primary_database", params)
		if err != nil {
			fmt.Printf("   ⚠️  Primary database execution %d failed: %v\n", i+1, err)
		} else {
			fmt.Printf("   ✅ Primary database execution %d succeeded\n", i+1)
		}

		// Execute backup database tool (should always succeed)
		_, err = registry.ExecuteWithTracking(ctx, "backup_database", params)
		if err != nil {
			fmt.Printf("   ❌ Backup database execution %d failed: %v\n", i+1, err)
		} else {
			fmt.Printf("   ✅ Backup database execution %d succeeded\n", i+1)
		}

		time.Sleep(100 * time.Millisecond)
	}

	// Show comparative metrics
	fmt.Println("\n📊 Performance Comparison:")
	for _, toolName := range []string{"primary_database", "backup_database"} {
		metrics, err := registry.GetPerformanceMetrics(toolName)
		if err != nil {
			continue
		}

		fmt.Printf("\n🔧 %s:\n", toolName)
		fmt.Printf("   → Success Rate: %.1f%% (%d/%d)\n",
			metrics.SuccessRate*100, metrics.SuccessCount, metrics.ExecutionCount)
		fmt.Printf("   → Reliability Score: %.3f\n", metrics.ReliabilityScore)
		fmt.Printf("   → Average Latency: %v\n", metrics.AverageLatency)
	}
}

// Demo function for intelligent selection with fallbacks.
//nolint:unused // Used when main() is uncommented
func demonstrateIntelligentSelection(registry *tools.SmartToolRegistry) {
	fmt.Println("\n🧠 Intelligent Selection with Fallbacks")
	fmt.Println("=======================================")

	ctx := context.Background()

	// Test various intents to show intelligent selection
	intents := []struct {
		description string
		intent      string
		expectTool  string
	}{
		{
			description: "Database query intent",
			intent:      "I need to query the user database for active accounts",
			expectTool:  "primary_database", // Should select based on capability match
		},
		{
			description: "File processing intent",
			intent:      "Process the CSV file and convert it to JSON format",
			expectTool:  "file_processor", // Should select file processor
		},
		{
			description: "Mixed intent (database + file)",
			intent:      "Read data from database and save to file",
			expectTool:  "", // Registry will decide based on scoring
		},
		{
			description: "Ambiguous intent",
			intent:      "I need to work with some data",
			expectTool:  "", // Will select based on scoring algorithm
		},
	}

	for _, test := range intents {
		fmt.Printf("\n🎯 %s\n", test.description)
		fmt.Printf("   Intent: \"%s\"\n", test.intent)

		// Get best tool selection
		selectedTool, err := registry.SelectBest(ctx, test.intent)
		if err != nil {
			fmt.Printf("   ❌ Selection failed: %v\n", err)
			continue
		}

		fmt.Printf("   → Selected: %s\n", selectedTool.Name())

		// Show all matches with their relative ranking
		matches := registry.Match(test.intent)
		fmt.Printf("   → All matches: ")
		for i, match := range matches {
			if i > 0 {
				fmt.Print(" > ")
			}
			fmt.Print(match.Name())
		}
		fmt.Println()

		if test.expectTool != "" && selectedTool.Name() != test.expectTool {
			fmt.Printf("   ℹ️  Note: Expected %s but got %s (algorithm decided)\n",
				test.expectTool, selectedTool.Name())
		}
	}
}

// Demo function for capability analysis.
//nolint:unused // Used when main() is uncommented
func demonstrateCapabilityAnalysis(registry *tools.SmartToolRegistry) {
	fmt.Println("\n🔍 Capability Analysis")
	fmt.Println("======================")

	allTools := registry.List()

	for _, tool := range allTools {
		fmt.Printf("\n🔧 %s:\n", tool.Name())
		fmt.Printf("   Description: %s\n", tool.Description())

		capabilities, err := registry.GetCapabilities(tool.Name())
		if err != nil {
			fmt.Printf("   ❌ Could not get capabilities: %v\n", err)
			continue
		}

		fmt.Printf("   Capabilities:\n")
		for _, cap := range capabilities {
			confidenceBar := ""
			barLength := int(cap.Confidence * 10)
			for i := 0; i < barLength; i++ {
				confidenceBar += "█"
			}
			for i := barLength; i < 10; i++ {
				confidenceBar += "░"
			}

			fmt.Printf("     → %-15s [%s] %.2f\n", cap.Name, confidenceBar, cap.Confidence)
		}
	}
}

// Demo function for custom selector configuration.
//nolint:unused // Used when main() is uncommented
func demonstrateCustomSelector() {
	fmt.Println("\n⚙️  Custom Selector Configuration")
	fmt.Println("================================")

	// Create selectors with different weight configurations
	selectors := map[string]*tools.BayesianToolSelector{
		"match_focused": {
			MatchWeight:       0.7,
			PerformanceWeight: 0.2,
			CapabilityWeight:  0.1,
		},
		"performance_focused": {
			MatchWeight:       0.2,
			PerformanceWeight: 0.6,
			CapabilityWeight:  0.2,
		},
		"capability_focused": {
			MatchWeight:       0.2,
			PerformanceWeight: 0.2,
			CapabilityWeight:  0.6,
		},
	}

	// Create test tools for comparison
	testTools := []core.Tool{
		&SearchTool{name: "fast_search"},
		&DatabaseTool{name: "slow_database", failureRate: 0.1},
		&FileProcessorTool{name: "file_tool"},
	}

	testIntent := "search through database files"
	ctx := context.Background()

	fmt.Printf("Testing intent: \"%s\"\n", testIntent)

	for name, selector := range selectors {
		fmt.Printf("\n🎛️  %s selector:\n", name)
		fmt.Printf("   Weights - Match: %.1f, Performance: %.1f, Capability: %.1f\n",
			selector.MatchWeight, selector.PerformanceWeight, selector.CapabilityWeight)

		scores, err := selector.ScoreTools(ctx, testIntent, testTools)
		if err != nil {
			fmt.Printf("   ❌ Scoring failed: %v\n", err)
			continue
		}

		// Sort by final score
		for i := 0; i < len(scores)-1; i++ {
			for j := i + 1; j < len(scores); j++ {
				if scores[i].FinalScore < scores[j].FinalScore {
					scores[i], scores[j] = scores[j], scores[i]
				}
			}
		}

		fmt.Printf("   Results:\n")
		for i, score := range scores {
			fmt.Printf("     %d. %-15s (Final: %.3f, Match: %.3f, Perf: %.3f, Cap: %.3f)\n",
				i+1, score.Tool.Name(), score.FinalScore, score.MatchScore,
				score.PerformanceScore, score.CapabilityScore)
		}
	}
}

//nolint:unused // Used when main() is uncommented
func runAdvancedExample() {
	fmt.Println("🚀 Advanced Smart Tool Registry Features")
	fmt.Println("========================================")

	// Setup registry
	config := &tools.SmartToolRegistryConfig{
		AutoDiscoveryEnabled:       false, // Disable for focused demo
		PerformanceTrackingEnabled: true,
		FallbackEnabled:            true,
	}

	registry := tools.NewSmartToolRegistry(config)

	// Register tools with different reliability characteristics
	primaryDB := &DatabaseTool{name: "primary_database", failureRate: 0.3} // 30% failure rate
	backupDB := &BackupDatabaseTool{name: "backup_database"}               // High reliability
	fileProcessor := &FileProcessorTool{name: "file_processor"}

	toolsList := []core.Tool{primaryDB, backupDB, fileProcessor}

	for _, tool := range toolsList {
		err := registry.Register(tool)
		if err != nil {
			log.Printf("Error registering %s: %v", tool.Name(), err)
		} else {
			fmt.Printf("✅ Registered: %s\n", tool.Name())
		}
	}

	// Configure fallbacks
	err := registry.AddFallback("database query", "backup_database")
	if err != nil {
		log.Printf("Error adding fallback: %v", err)
	} else {
		fmt.Println("✅ Configured database fallback")
	}

	// Run demonstrations
	demonstratePerformanceTracking(registry)
	demonstrateIntelligentSelection(registry)
	demonstrateCapabilityAnalysis(registry)
	demonstrateCustomSelector()

	fmt.Println("\n🎉 Advanced demonstration completed!")
	fmt.Println("\nAdvanced features demonstrated:")
	fmt.Println("  ✅ Performance tracking with failure handling")
	fmt.Println("  ✅ Intelligent fallback selection")
	fmt.Println("  ✅ Comparative tool analysis")
	fmt.Println("  ✅ Detailed capability inspection")
	fmt.Println("  ✅ Custom selector weight configuration")
	fmt.Println("  ✅ Real-world reliability scenarios")
}

// Random generator is now automatically seeded in Go 1.20+

// Uncomment to run this example instead of the basic one.
// func main() {
// 	runAdvancedExample()
// }
