package context

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewManager(t *testing.T) {
	tests := []struct {
		name        string
		sessionID   string
		agentID     string
		baseDir     string
		config      Config
		expectError bool
	}{
		{
			name:      "valid configuration",
			sessionID: "test-session",
			agentID:   "test-agent",
			baseDir:   t.TempDir(),
			config:    DefaultConfig(),
		},
		{
			name:        "invalid configuration - empty session ID",
			sessionID:   "",
			agentID:     "test-agent",
			baseDir:     t.TempDir(),
			config:      DefaultConfig(),
			expectError: true,
		},
		{
			name:        "invalid configuration - empty agent ID",
			sessionID:   "test-session",
			agentID:     "",
			baseDir:     t.TempDir(),
			config:      DefaultConfig(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager, err := NewManager(tt.sessionID, tt.agentID, tt.baseDir, tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, manager)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, manager)
				assert.Equal(t, tt.sessionID, manager.sessionID)
				assert.Equal(t, tt.agentID, manager.agentID)
			}
		})
	}
}

func TestBuildOptimizedContext(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	tests := []struct {
		name     string
		request  ContextRequest
		validate func(t *testing.T, response *ContextResponse)
	}{
		{
			name: "basic context request",
			request: ContextRequest{
				Observations:         []string{"observation 1", "observation 2"},
				CurrentTask:          "test task",
				PrioritizeCache:      true,
				CompressionPriority:  PriorityMedium,
				AllowDiversification: true,
				IncludeErrors:        true,
				IncludeTodos:         true,
				MaxTokens:            1000,
			},
			validate: func(t *testing.T, response *ContextResponse) {
				assert.NotEmpty(t, response.Context)
				assert.Greater(t, response.TokenCount, 0)
				assert.GreaterOrEqual(t, response.CacheHitRate, 0.0)
				assert.LessOrEqual(t, response.CacheHitRate, 1.0)
				assert.NotEmpty(t, response.OptimizationsApplied)
				assert.Greater(t, response.ContextVersion, int64(0))
			},
		},
		{
			name: "large content compression",
			request: ContextRequest{
				Observations: []string{
					generateLargeString(60000), // Trigger compression
				},
				CurrentTask:         "large content task",
				CompressionPriority: PriorityHigh,
				MaxTokens:           4096,
			},
			validate: func(t *testing.T, response *ContextResponse) {
				assert.NotEmpty(t, response.Context)
				assert.LessOrEqual(t, response.CompressionRatio, 1.0)  // Ratio <= 1.0 (less means compression occurred)
				assert.GreaterOrEqual(t, response.CompressionRatio, 0.0) // Ratio >= 0.0 (sanity check)
				assert.LessOrEqual(t, response.TokenCount, 4096)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := manager.BuildOptimizedContext(ctx, tt.request)
			assert.NoError(t, err)
			assert.NotNil(t, response)
			tt.validate(t, response)
		})
	}
}

func TestErrorRecording(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.EnableErrorRetention = true

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Record various types of errors
	testCases := []struct {
		errorType string
		message   string
		severity  ErrorSeverity
	}{
		{"timeout", "Request timed out after 30s", SeverityHigh},
		{"auth_failure", "Authentication failed", SeverityCritical},
		{"tool_error", "Tool execution failed", SeverityMedium},
		{"general_error", "Something went wrong", SeverityLow},
	}

	for _, tc := range testCases {
		manager.RecordError(ctx, tc.errorType, tc.message, tc.severity, map[string]interface{}{
			"test_context": "unit_test",
		})
	}

	// Test that errors are recorded and can be retrieved
	request := ContextRequest{
		IncludeErrors: true,
		CurrentTask:   "test error retrieval",
	}

	response, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Context should include error information
	assert.Contains(t, response.Context, "Recent Learning History")
}

func TestTodoManagement(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.EnableTodoManagement = true

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Add todos
	err = manager.AddTodo(ctx, "Implement feature A", 8)
	assert.NoError(t, err)

	err = manager.AddTodo(ctx, "Fix bug B", 9)
	assert.NoError(t, err)

	err = manager.AddTodo(ctx, "Write tests", 6)
	assert.NoError(t, err)

	// Test context includes todos
	request := ContextRequest{
		IncludeTodos: true,
		CurrentTask:  "test todo inclusion",
	}

	response, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Context should include todo information
	assert.Contains(t, response.Context, "Current Objectives")
}

func TestSuccessRecording(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.EnableErrorRetention = true

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Record successes
	manager.RecordSuccess(ctx, "feature_implementation", "Successfully implemented authentication", map[string]interface{}{
		"duration": "2 hours",
		"tests":    "all passing",
	})

	manager.RecordSuccess(ctx, "bug_fix", "Fixed memory leak in cache", map[string]interface{}{
		"impact": "20% performance improvement",
	})

	// Test that successes influence context
	request := ContextRequest{
		IncludeErrors: true, // Success info is included with error context
		CurrentTask:   "test success recording",
	}

	response, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)
	assert.NotNil(t, response)
}

func TestPerformanceMetrics(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Execute several context builds to generate metrics
	for i := 0; i < 5; i++ {
		request := ContextRequest{
			CurrentTask:         "performance test task",
			PrioritizeCache:     true,
			IncludeErrors:       true,
			IncludeTodos:        true,
		}

		_, err := manager.BuildOptimizedContext(ctx, request)
		assert.NoError(t, err)
	}

	// Check metrics
	metrics := manager.GetPerformanceMetrics()
	assert.NotNil(t, metrics)

	// Verify expected metric keys
	expectedKeys := []string{
		"total_requests",
		"context_version",
		"enabled",
		"cache",
		"diversity",
		"errors",
		"compression",
		"todos",
	}

	for _, key := range expectedKeys {
		assert.Contains(t, metrics, key, "Missing metric key: %s", key)
	}

	// Check that request count is tracked
	assert.Equal(t, int64(5), metrics["total_requests"])
}

func TestHealthStatus(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	// Test health status
	health := manager.GetHealthStatus()
	assert.NotNil(t, health)

	// Should have overall status
	assert.Contains(t, health, "overall_status")
	assert.Contains(t, health, "components")

	// Components should be tracked
	components := health["components"].(map[string]string)
	assert.NotEmpty(t, components)
}

func TestEnableDisable(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Initially enabled
	request := ContextRequest{
		CurrentTask: "test enable/disable",
	}

	response1, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)

	// Disable
	manager.SetEnabled(false)

	response2, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)

	// Should fall back to basic context when disabled
	assert.NotEqual(t, response1.OptimizationsApplied, response2.OptimizationsApplied)

	// Re-enable
	manager.SetEnabled(true)

	response3, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)

	// Should work normally again
	assert.NotEqual(t, response2.OptimizationsApplied, response3.OptimizationsApplied)
}

func TestConcurrentAccess(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Run concurrent operations to test thread safety
	const numGoroutines = 10
	const numOperations = 5

	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(workerID int) {
			defer func() { done <- true }()

			for j := 0; j < numOperations; j++ {
				// Mix different operations
				switch j % 4 {
				case 0:
					request := ContextRequest{
						CurrentTask: "concurrent test",
					}
					_, err := manager.BuildOptimizedContext(ctx, request)
					assert.NoError(t, err)

				case 1:
					manager.RecordError(ctx, "test_error", "concurrent error test", SeverityLow, nil)

				case 2:
					manager.RecordSuccess(ctx, "test_success", "concurrent success test", nil)

				case 3:
					err := manager.AddTodo(ctx, "concurrent todo", 5)
					assert.NoError(t, err)
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Verify system is still functional
	metrics := manager.GetPerformanceMetrics()
	assert.NotNil(t, metrics)

	totalRequests := metrics["total_requests"].(int64)
	assert.Greater(t, totalRequests, int64(0))
}

func TestFileSystemIntegration(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.EnableFileSystemMemory = true

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Generate large content that should be stored in filesystem
	largeObservations := []string{
		generateLargeString(100000), // 100KB string
		generateLargeString(150000), // 150KB string
	}

	request := ContextRequest{
		Observations:        largeObservations,
		CurrentTask:         "filesystem test",
		CompressionPriority: PriorityMedium,
	}

	response, err := manager.BuildOptimizedContext(ctx, request)
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Verify that files were created in the memory directory
	memoryDir := filepath.Join(tempDir, "memory", "test-session", "test-agent")
	_, err = os.Stat(memoryDir)
	assert.NoError(t, err, "Memory directory should exist")

	// Check that some files were created
	files, err := os.ReadDir(memoryDir)
	assert.NoError(t, err)
	assert.NotEmpty(t, files, "Memory directory should contain files")
}

func TestConfigValidation(t *testing.T) {
	tempDir := t.TempDir()

	// Test invalid configurations
	invalidConfigs := []Config{
		{
			SessionID: "", // Empty session ID
			AgentID:   "test-agent",
			BaseDir:   tempDir,
		},
		{
			SessionID: "test-session",
			AgentID:   "", // Empty agent ID
			BaseDir:   tempDir,
		},
		{
			SessionID: "test-session",
			AgentID:   "test-agent",
			BaseDir:   "", // Empty base dir
		},
		{
			SessionID:        "test-session",
			AgentID:          "test-agent",
			BaseDir:          tempDir,
			CacheHitTarget:   -0.1, // Invalid cache target
		},
		{
			SessionID:        "test-session",
			AgentID:          "test-agent",
			BaseDir:          tempDir,
			CacheHitTarget:   1.1, // Invalid cache target
		},
		{
			SessionID:        "test-session",
			AgentID:          "test-agent",
			BaseDir:          tempDir,
			Cache: CacheConfig{
				TimestampGranularity: "second", // Invalid granularity
			},
		},
	}

	for i, config := range invalidConfigs {
		t.Run(fmt.Sprintf("invalid_config_%d", i), func(t *testing.T) {
			_, err := NewManager("test-session", "test-agent", tempDir, config)
			assert.Error(t, err, "Should reject invalid configuration")
		})
	}
}

// Helper functions

func generateLargeString(size int) string {
	content := make([]byte, size)
	for i := range content {
		content[i] = byte('A' + (i % 26))
	}
	return string(content)
}

// Benchmark tests

func BenchmarkBuildOptimizedContext(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()

	manager, err := NewManager("bench-session", "bench-agent", tempDir, config)
	require.NoError(b, err)

	ctx := context.Background()
	request := ContextRequest{
		Observations:        []string{"benchmark observation 1", "benchmark observation 2"},
		CurrentTask:         "benchmark task",
		PrioritizeCache:     true,
		CompressionPriority: PriorityMedium,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := manager.BuildOptimizedContext(ctx, request)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkErrorRecording(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	config.EnableErrorRetention = true

	manager, err := NewManager("bench-session", "bench-agent", tempDir, config)
	require.NoError(b, err)

	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		manager.RecordError(ctx, "benchmark_error", "benchmark error message", SeverityMedium, nil)
	}
}

func BenchmarkTodoManagement(b *testing.B) {
	tempDir := b.TempDir()
	config := DefaultConfig()
	config.EnableTodoManagement = true

	manager, err := NewManager("bench-session", "bench-agent", tempDir, config)
	require.NoError(b, err)

	ctx := context.Background()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := manager.AddTodo(ctx, "benchmark todo", 5)
		if err != nil {
			b.Fatal(err)
		}
	}
}
// Additional tests for improved coverage

func TestManager_TodoManagement(t *testing.T) {
	tempDir := t.TempDir()
	config := DefaultConfig()
	config.EnableTodoManagement = true

	manager, err := NewManager("test-session", "test-agent", tempDir, config)
	require.NoError(t, err)

	ctx := context.Background()

	// Test UpdateTodos
	todos := []TodoItem{
		{
			ID:          "task-1",
			Description: "Test task",
			Status:      TodoPending,
			Priority:    5,
		},
	}

	err = manager.UpdateTodos(ctx, todos)
	assert.NoError(t, err)

	// Test SetActiveTodo
	err = manager.SetActiveTodo(ctx, "task-1")
	assert.NoError(t, err)

	// Test CompleteTodo
	err = manager.CompleteTodo(ctx, "task-1")
	assert.NoError(t, err)
}

func TestConfig_FactoryFunctions(t *testing.T) {
	// Test DevelopmentConfig
	devConfig := DevelopmentConfig()
	assert.NotNil(t, devConfig)
	assert.True(t, devConfig.EnableTodoManagement)

	// Test ProductionConfig
	prodConfig := ProductionConfig()
	assert.NotNil(t, prodConfig)
	assert.Equal(t, 0.95, prodConfig.CacheHitTarget)

	// Test GetTimestampFormat
	config := DefaultConfig()
	format := config.GetTimestampFormat()
	assert.NotEmpty(t, format)
}
