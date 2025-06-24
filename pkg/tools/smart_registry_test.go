package tools

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Mock tool for testing.
type mockTool struct {
	name         string
	description  string
	capabilities []string
	canHandle    func(string) bool
}

func (m *mockTool) Name() string {
	return m.name
}

func (m *mockTool) Description() string {
	return m.description
}

func (m *mockTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         m.name,
		Description:  m.description,
		Capabilities: m.capabilities,
		Version:      "1.0.0",
	}
}

func (m *mockTool) CanHandle(ctx context.Context, intent string) bool {
	if m.canHandle != nil {
		return m.canHandle(intent)
	}
	return strings.Contains(strings.ToLower(intent), strings.ToLower(m.name))
}

func (m *mockTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{
		Data: map[string]interface{}{"result": "success"},
	}, nil
}

func (m *mockTool) Validate(params map[string]interface{}) error {
	return nil
}

func (m *mockTool) InputSchema() models.InputSchema {
	return models.InputSchema{}
}

func newMockTool(name, description string, capabilities []string) *mockTool {
	return &mockTool{
		name:         name,
		description:  description,
		capabilities: capabilities,
	}
}

// Mock tool with nil metadata for testing.
type mockToolWithNilMetadata struct {
	name string
}

func (m *mockToolWithNilMetadata) Name() string {
	return m.name
}

func (m *mockToolWithNilMetadata) Description() string {
	return ""
}

func (m *mockToolWithNilMetadata) Metadata() *core.ToolMetadata {
	return nil
}

func (m *mockToolWithNilMetadata) CanHandle(ctx context.Context, intent string) bool {
	return false
}

func (m *mockToolWithNilMetadata) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{}, nil
}

func (m *mockToolWithNilMetadata) Validate(params map[string]interface{}) error {
	return nil
}

func (m *mockToolWithNilMetadata) InputSchema() models.InputSchema {
	return models.InputSchema{}
}

// Mock MCP Discovery Service.
type mockMCPDiscovery struct {
	tools     []core.Tool
	callbacks []func(tools []core.Tool)
}

func (m *mockMCPDiscovery) DiscoverTools(ctx context.Context) ([]core.Tool, error) {
	return m.tools, nil
}

func (m *mockMCPDiscovery) Subscribe(callback func(tools []core.Tool)) error {
	m.callbacks = append(m.callbacks, callback)
	return nil
}

func (m *mockMCPDiscovery) TriggerUpdate() {
	for _, callback := range m.callbacks {
		callback(m.tools)
	}
}

func TestSmartToolRegistry_Register(t *testing.T) {
	config := &SmartToolRegistryConfig{
		AutoDiscoveryEnabled:       false,
		PerformanceTrackingEnabled: true,
		FallbackEnabled:            true,
	}
	registry := NewSmartToolRegistry(config)

	tests := []struct {
		name        string
		tool        core.Tool
		expectError bool
		errorMsg    string
	}{
		{
			name:        "register valid tool",
			tool:        newMockTool("search", "Search for information", []string{"search", "query"}),
			expectError: false,
		},
		{
			name:        "register nil tool",
			tool:        nil,
			expectError: true,
			errorMsg:    "cannot register a nil tool",
		},
		{
			name:        "register duplicate tool",
			tool:        newMockTool("search", "Another search tool", []string{"search"}),
			expectError: true,
			errorMsg:    "tool already registered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := registry.Register(tt.tool)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
			} else {
				assert.NoError(t, err)

				// Verify tool was registered
				retrieved, getErr := registry.Get(tt.tool.Name())
				assert.NoError(t, getErr)
				assert.Equal(t, tt.tool, retrieved)

				// Verify performance metrics were initialized
				metrics, metricsErr := registry.GetPerformanceMetrics(tt.tool.Name())
				assert.NoError(t, metricsErr)
				assert.Equal(t, int64(0), metrics.ExecutionCount)
				assert.Equal(t, 0.5, metrics.ReliabilityScore)
			}
		})
	}
}

func TestSmartToolRegistry_SelectBest(t *testing.T) {
	config := &SmartToolRegistryConfig{
		AutoDiscoveryEnabled:       false,
		PerformanceTrackingEnabled: true,
	}
	registry := NewSmartToolRegistry(config)

	// Register test tools
	searchTool := newMockTool("search", "Search for information", []string{"search", "query"})
	createTool := newMockTool("create", "Create new resources", []string{"create", "generate"})
	analyzeTool := newMockTool("analyze", "Analyze data and provide insights", []string{"analyze", "process"})

	require.NoError(t, registry.Register(searchTool))
	require.NoError(t, registry.Register(createTool))
	require.NoError(t, registry.Register(analyzeTool))

	tests := []struct {
		name         string
		intent       string
		expectedTool string
		expectError  bool
	}{
		{
			name:         "search intent",
			intent:       "I need to search for user information",
			expectedTool: "search",
			expectError:  false,
		},
		{
			name:         "create intent",
			intent:       "Create a new user account",
			expectedTool: "create",
			expectError:  false,
		},
		{
			name:         "analyze intent",
			intent:       "Analyze the performance data",
			expectedTool: "analyze",
			expectError:  false,
		},
		{
			name:         "ambiguous intent",
			intent:       "Help me with data",
			expectedTool: "", // Will select based on scoring
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			selectedTool, err := registry.SelectBest(ctx, tt.intent)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, selectedTool)

				if tt.expectedTool != "" {
					assert.Equal(t, tt.expectedTool, selectedTool.Name())
				}
			}
		})
	}
}

func TestSmartToolRegistry_ExecuteWithTracking(t *testing.T) {
	config := &SmartToolRegistryConfig{
		PerformanceTrackingEnabled: true,
	}
	registry := NewSmartToolRegistry(config)

	tool := newMockTool("test", "Test tool", []string{"test"})
	require.NoError(t, registry.Register(tool))

	// Execute the tool multiple times
	ctx := context.Background()
	params := map[string]interface{}{"input": "test"}

	for i := 0; i < 5; i++ {
		result, err := registry.ExecuteWithTracking(ctx, "test", params)
		assert.NoError(t, err)
		assert.NotNil(t, result)
	}

	// Check performance metrics
	metrics, err := registry.GetPerformanceMetrics("test")
	assert.NoError(t, err)
	assert.Equal(t, int64(5), metrics.ExecutionCount)
	assert.Equal(t, int64(5), metrics.SuccessCount)
	assert.Equal(t, int64(0), metrics.FailureCount)
	assert.Equal(t, 1.0, metrics.SuccessRate)
	assert.True(t, metrics.ReliabilityScore > 0.5)
}

func TestSmartToolRegistry_Fallbacks(t *testing.T) {
	config := &SmartToolRegistryConfig{
		FallbackEnabled: true,
	}
	registry := NewSmartToolRegistry(config)

	primaryTool := newMockTool("primary", "Primary tool", []string{"primary"})
	fallbackTool := newMockTool("fallback", "Fallback tool", []string{"fallback"})

	require.NoError(t, registry.Register(primaryTool))
	require.NoError(t, registry.Register(fallbackTool))

	// Add fallback configuration
	err := registry.AddFallback("search data", "fallback")
	assert.NoError(t, err)

	// Test with non-existent fallback tool
	err = registry.AddFallback("search data", "nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fallback tool not found")
}

func TestSmartToolRegistry_CapabilityExtraction(t *testing.T) {
	config := &SmartToolRegistryConfig{}
	registry := NewSmartToolRegistry(config)

	tool := newMockTool("processor", "This tool can search, create, and analyze data", []string{"explicit_capability"})
	require.NoError(t, registry.Register(tool))

	capabilities, err := registry.GetCapabilities("processor")
	assert.NoError(t, err)
	assert.NotEmpty(t, capabilities)

	// Should have explicit capability
	hasExplicit := false
	for _, cap := range capabilities {
		if cap.Name == "explicit_capability" {
			hasExplicit = true
			assert.Equal(t, 1.0, cap.Confidence)
		}
	}
	assert.True(t, hasExplicit)

	// Should have inferred capabilities from description
	inferredCaps := make(map[string]bool)
	for _, cap := range capabilities {
		if cap.Confidence < 1.0 { // Inferred capabilities have lower confidence
			inferredCaps[cap.Name] = true
		}
	}

	// Description contains "search", "create", "analyze" - should be inferred
	expectedInferred := []string{"search", "creation", "analysis"}
	for _, expected := range expectedInferred {
		assert.True(t, inferredCaps[expected], "Expected inferred capability: %s", expected)
	}
}

func TestSmartToolRegistry_AutoDiscovery(t *testing.T) {
	mockDiscovery := &mockMCPDiscovery{
		tools: []core.Tool{
			newMockTool("discovered1", "Auto-discovered tool 1", []string{"auto"}),
			newMockTool("discovered2", "Auto-discovered tool 2", []string{"auto"}),
		},
	}

	config := &SmartToolRegistryConfig{
		AutoDiscoveryEnabled: true,
		MCPDiscovery:         mockDiscovery,
	}
	registry := NewSmartToolRegistry(config)

	// Give some time for auto-discovery to run
	time.Sleep(100 * time.Millisecond)

	// Check that discovered tools were registered
	tools := registry.List()
	assert.Len(t, tools, 2)

	toolNames := make(map[string]bool)
	for _, tool := range tools {
		toolNames[tool.Name()] = true
	}

	assert.True(t, toolNames["discovered1"])
	assert.True(t, toolNames["discovered2"])

	// Test dynamic discovery
	mockDiscovery.tools = append(mockDiscovery.tools,
		newMockTool("discovered3", "Auto-discovered tool 3", []string{"auto"}))

	mockDiscovery.TriggerUpdate()
	time.Sleep(50 * time.Millisecond)

	// Should now have 3 tools
	tools = registry.List()
	assert.Len(t, tools, 3)
}

func TestSmartToolRegistry_Match(t *testing.T) {
	config := &SmartToolRegistryConfig{}
	registry := NewSmartToolRegistry(config)

	// Register tools with different capabilities
	searchTool := newMockTool("search", "Search for information in databases", []string{"search", "query", "find"})
	createTool := newMockTool("create", "Create new resources and files", []string{"create", "generate", "make"})
	analyzeTool := newMockTool("analyze", "Analyze and process data", []string{"analyze", "process", "examine"})

	require.NoError(t, registry.Register(searchTool))
	require.NoError(t, registry.Register(createTool))
	require.NoError(t, registry.Register(analyzeTool))

	tests := []struct {
		name          string
		intent        string
		expectedFirst string
		minResults    int
	}{
		{
			name:          "specific search intent",
			intent:        "find user information",
			expectedFirst: "search",
			minResults:    1,
		},
		{
			name:          "specific create intent",
			intent:        "generate new report",
			expectedFirst: "create",
			minResults:    1,
		},
		{
			name:          "specific analyze intent",
			intent:        "process the data",
			expectedFirst: "analyze",
			minResults:    1,
		},
		{
			name:       "broad intent",
			intent:     "help with data",
			minResults: 1, // Should match at least one tool
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matches := registry.Match(tt.intent)
			assert.GreaterOrEqual(t, len(matches), tt.minResults)

			if tt.expectedFirst != "" && len(matches) > 0 {
				assert.Equal(t, tt.expectedFirst, matches[0].Name())
			}
		})
	}
}

func TestSmartToolRegistry_PerformanceMetricsUpdate(t *testing.T) {
	config := &SmartToolRegistryConfig{
		PerformanceTrackingEnabled: true,
	}
	registry := NewSmartToolRegistry(config)

	tool := newMockTool("perf_test", "Performance test tool", []string{"test"})
	require.NoError(t, registry.Register(tool))

	ctx := context.Background()
	params := map[string]interface{}{"input": "test"}

	// Execute successful operations
	for i := 0; i < 3; i++ {
		_, err := registry.ExecuteWithTracking(ctx, "perf_test", params)
		assert.NoError(t, err)
	}

	metrics, err := registry.GetPerformanceMetrics("perf_test")
	assert.NoError(t, err)
	assert.Equal(t, int64(3), metrics.ExecutionCount)
	assert.Equal(t, int64(3), metrics.SuccessCount)
	assert.Equal(t, 1.0, metrics.SuccessRate)
	assert.True(t, metrics.ReliabilityScore >= 0.5)
	assert.True(t, metrics.AverageLatency > 0)
}

func TestSmartToolRegistry_EmptyRegistry(t *testing.T) {
	config := &SmartToolRegistryConfig{}
	registry := NewSmartToolRegistry(config)

	// Test operations on empty registry
	ctx := context.Background()

	// SelectBest should return error
	_, err := registry.SelectBest(ctx, "any intent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no tools available")

	// Match should return empty slice
	matches := registry.Match("any intent")
	assert.Empty(t, matches)

	// List should return empty slice
	tools := registry.List()
	assert.Empty(t, tools)

	// Get should return error
	_, err = registry.Get("nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tool not found")
}

func BenchmarkSmartToolRegistry_SelectBest(b *testing.B) {
	config := &SmartToolRegistryConfig{}
	registry := NewSmartToolRegistry(config)

	// Register multiple tools
	for i := 0; i < 50; i++ {
		tool := newMockTool(
			fmt.Sprintf("tool_%d", i),
			fmt.Sprintf("Tool number %d for testing", i),
			[]string{fmt.Sprintf("capability_%d", i%5)},
		)
		_ = registry.Register(tool)
	}

	ctx := context.Background()
	intent := "find data using capability_2"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := registry.SelectBest(ctx, intent)
		if err != nil {
			b.Fatalf("SelectBest failed: %v", err)
		}
	}
}

func BenchmarkSmartToolRegistry_Match(b *testing.B) {
	config := &SmartToolRegistryConfig{}
	registry := NewSmartToolRegistry(config)

	// Register multiple tools
	for i := 0; i < 100; i++ {
		tool := newMockTool(
			fmt.Sprintf("tool_%d", i),
			fmt.Sprintf("Tool number %d for testing", i),
			[]string{fmt.Sprintf("capability_%d", i%10)},
		)
		_ = registry.Register(tool)
	}

	intent := "search for information"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matches := registry.Match(intent)
		_ = matches // Prevent optimization
	}
}
