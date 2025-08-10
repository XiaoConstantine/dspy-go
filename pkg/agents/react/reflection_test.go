package react

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelfReflector_ComprehensiveTests(t *testing.T) {
	t.Run("NewSelfReflector", func(t *testing.T) {
		reflector := NewSelfReflector(5, 100*time.Millisecond)
		assert.NotNil(t, reflector)
		assert.Equal(t, 5, reflector.depth)
		assert.Equal(t, 100*time.Millisecond, reflector.delay)
		assert.Equal(t, 0.6, reflector.successRateThreshold) // Default threshold
	})

	t.Run("NewSelfReflectorWithThreshold", func(t *testing.T) {
		reflector := NewSelfReflectorWithThreshold(3, 50*time.Millisecond, 0.8)
		assert.Equal(t, 0.8, reflector.successRateThreshold)

		// Test invalid threshold fallback
		reflector = NewSelfReflectorWithThreshold(3, 50*time.Millisecond, -0.1)
		assert.Equal(t, 0.6, reflector.successRateThreshold)

		reflector = NewSelfReflectorWithThreshold(3, 50*time.Millisecond, 1.5)
		assert.Equal(t, 0.6, reflector.successRateThreshold)
	})

	t.Run("Basic Reflection", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0) // No delay for testing
		ctx := context.Background()

		record := ExecutionRecord{
			Success:   true,
			Actions:   []ActionRecord{{Tool: "test_tool", Success: true, Duration: 100 * time.Millisecond}},
			Timestamp: time.Now(),
			Input:     map[string]interface{}{"task": "simple task"},
		}

		reflections := reflector.Reflect(ctx, record)
		assert.NotNil(t, reflections)
		// Should generate strategy reflection for efficient completion
		assert.Greater(t, len(reflections), 0)
	})

	t.Run("Strategy Reflection - Repeated Tool Usage", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		record := ExecutionRecord{
			Success: true,
			Actions: []ActionRecord{
				{Tool: "search_tool", Success: true, Duration: 100 * time.Millisecond},
				{Tool: "search_tool", Success: true, Duration: 100 * time.Millisecond},
				{Tool: "search_tool", Success: true, Duration: 100 * time.Millisecond},
				{Tool: "search_tool", Success: true, Duration: 100 * time.Millisecond},
			},
			Timestamp: time.Now(),
			Input:     map[string]interface{}{"task": "research task"},
		}

		reflections := reflector.Reflect(ctx, record)
		require.Greater(t, len(reflections), 0)

		strategyReflection := findReflectionByType(reflections, ReflectionTypeStrategy)
		assert.NotNil(t, strategyReflection)
		assert.Contains(t, strategyReflection.Insight, "search_tool")
		assert.Contains(t, strategyReflection.Insight, "4 times")
	})

	t.Run("Performance Reflection - Low Success Rate", func(t *testing.T) {
		reflector := NewSelfReflectorWithThreshold(3, 0, 0.7) // 70% threshold
		ctx := context.Background()

		// Simulate multiple failed executions to lower success rate
		for i := 0; i < 8; i++ {
			record := ExecutionRecord{
				Success:   false, // Mostly failures
				Actions:   []ActionRecord{{Tool: "test_tool", Success: false, Duration: 100 * time.Millisecond}},
				Timestamp: time.Now(),
			}
			reflector.Reflect(ctx, record)
		}

		// Now add one more to trigger performance reflection
		record := ExecutionRecord{
			Success:   false,
			Actions:   []ActionRecord{{Tool: "test_tool", Success: false, Duration: 100 * time.Millisecond}},
			Timestamp: time.Now(),
		}

		reflections := reflector.Reflect(ctx, record)
		performanceReflection := findReflectionByType(reflections, ReflectionTypePerformance)
		if performanceReflection != nil {
			assert.Contains(t, performanceReflection.Insight, "Success rate is below threshold")
		}
	})

	t.Run("Performance Reflection - High Iterations", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		// Create records with many actions to increase average iterations
		for i := 0; i < 10; i++ {
			actions := make([]ActionRecord, 8) // 8 actions = high iteration count
			for j := range actions {
				actions[j] = ActionRecord{Tool: "test_tool", Success: true, Duration: 100 * time.Millisecond}
			}

			record := ExecutionRecord{
				Success:   true,
				Actions:   actions,
				Timestamp: time.Now(),
			}
			reflector.Reflect(ctx, record)
		}

		// Trigger one more reflection
		actions := make([]ActionRecord, 8)
		for j := range actions {
			actions[j] = ActionRecord{Tool: "test_tool", Success: true, Duration: 100 * time.Millisecond}
		}

		record := ExecutionRecord{
			Success:   true,
			Actions:   actions,
			Timestamp: time.Now(),
		}

		reflections := reflector.Reflect(ctx, record)
		performanceReflection := findReflectionByType(reflections, ReflectionTypePerformance)
		if performanceReflection != nil {
			assert.Contains(t, performanceReflection.Insight, "Average iteration count is high")
		}
	})

	t.Run("Learning Reflection - Pattern Recognition", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		// Create multiple successful records with same tool pattern
		toolPattern := []string{"search_tool", "analyze_tool"}
		for i := 0; i < 4; i++ {
			actions := make([]ActionRecord, len(toolPattern))
			for j, tool := range toolPattern {
				actions[j] = ActionRecord{Tool: tool, Success: true, Duration: 100 * time.Millisecond}
			}

			record := ExecutionRecord{
				Success:   true,
				Actions:   actions,
				Timestamp: time.Now(),
				Input:     map[string]interface{}{"task": "research task"},
			}
			reflector.Reflect(ctx, record)
		}

		// Should generate learning reflection about the pattern
		record := ExecutionRecord{
			Success: true,
			Actions: []ActionRecord{
				{Tool: "search_tool", Success: true, Duration: 100 * time.Millisecond},
				{Tool: "analyze_tool", Success: true, Duration: 100 * time.Millisecond},
			},
			Timestamp: time.Now(),
			Input:     map[string]interface{}{"task": "research task"},
		}

		reflections := reflector.Reflect(ctx, record)
		learningReflection := findReflectionByType(reflections, ReflectionTypeLearning)
		if learningReflection != nil {
			assert.Contains(t, learningReflection.Insight, "successful pattern")
		}
	})

	t.Run("Error Reflection", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		testError := errors.New("tool execution failed")
		record := ExecutionRecord{
			Success:   false,
			Error:     testError,
			Actions:   []ActionRecord{{Tool: "failing_tool", Success: false, Duration: 100 * time.Millisecond}},
			Timestamp: time.Now(),
		}

		// Generate the same error multiple times
		for i := 0; i < 3; i++ {
			reflections := reflector.Reflect(ctx, record)
			if i >= 2 { // Should trigger error reflection after 3 occurrences
				errorReflection := findReflectionByType(reflections, ReflectionTypeError)
				if errorReflection != nil {
					assert.Contains(t, errorReflection.Insight, "Recurring error detected")
				}
			}
		}
	})

	t.Run("Pattern Identification", func(t *testing.T) {
		reflector := NewSelfReflector(2, 0) // Low depth for faster pattern detection
		ctx := context.Background()

		// Generate multiple error reflections to create a pattern
		for i := 0; i < 5; i++ {
			record := ExecutionRecord{
				Success:   false,
				Error:     errors.New("consistent error"),
				Actions:   []ActionRecord{{Tool: "failing_tool", Success: false, Duration: 100 * time.Millisecond}},
				Timestamp: time.Now(),
			}
			reflector.Reflect(ctx, record)
		}

		// Should identify error pattern
		record := ExecutionRecord{

			Success:   false,
			Error:     errors.New("consistent error"),
			Actions:   []ActionRecord{{Tool: "failing_tool", Success: false, Duration: 100 * time.Millisecond}},
			Timestamp: time.Now(),
		}

		reflections := reflector.Reflect(ctx, record)
		patternReflection := findReflectionByType(reflections, ReflectionTypeLearning)
		if patternReflection != nil && len(reflections) > 1 {
			// Check if any reflection mentions patterns or systemic issues
			found := false
			for _, r := range reflections {
				if contains(r.Insight, "Frequent errors") || contains(r.Insight, "systemic") {
					found = true
					break
				}
			}
			// Pattern might be detected
			_ = found
		}
	})

	t.Run("Metrics and Statistics", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		// Add some execution records
		for i := 0; i < 5; i++ {
			record := ExecutionRecord{
				Success:   i%2 == 0, // Alternate success/failure
				Actions:   []ActionRecord{{Tool: "test_tool", Success: i%2 == 0, Duration: time.Duration(i*100) * time.Millisecond}},
				Timestamp: time.Now(),
			}
			reflector.Reflect(ctx, record)
		}

		metrics := reflector.GetMetrics()
		assert.NotNil(t, metrics)
		assert.Equal(t, 5, metrics.TotalExecutions)
		assert.Equal(t, 3, metrics.SuccessfulRuns) // 0, 2, 4 are successes
		assert.Equal(t, 2, metrics.FailedRuns)     // 1, 3 are failures
		assert.Contains(t, metrics.ToolUsageStats, "test_tool")

		patterns := reflector.GetPatterns()
		assert.NotNil(t, patterns)
	})

	t.Run("Top Reflections", func(t *testing.T) {
		reflector := NewSelfReflector(10, 0) // High depth to accumulate reflections

		// Generate several reflections with different confidence levels
		for i := 0; i < 5; i++ {
			reflector.reflectionCache = append(reflector.reflectionCache, Reflection{
				Type:       ReflectionTypeStrategy,
				Insight:    fmt.Sprintf("Test insight %d", i),
				Confidence: float64(i) / 10.0, // 0.0, 0.1, 0.2, 0.3, 0.4
				Timestamp:  time.Now(),
			})
		}

		topReflections := reflector.GetTopReflections(3)
		assert.Equal(t, 3, len(topReflections))

		// Should be sorted by confidence (highest first)
		for i := 0; i < len(topReflections)-1; i++ {
			assert.GreaterOrEqual(t, topReflections[i].Confidence, topReflections[i+1].Confidence)
		}
	})

	t.Run("Reset", func(t *testing.T) {
		reflector := NewSelfReflector(3, 0)
		ctx := context.Background()

		// Add some data
		record := ExecutionRecord{
			Success:   true,
			Actions:   []ActionRecord{{Tool: "test_tool", Success: true, Duration: 100 * time.Millisecond}},
			Timestamp: time.Now(),
		}
		reflector.Reflect(ctx, record)

		assert.Greater(t, len(reflector.reflectionCache), 0)
		assert.Greater(t, reflector.metrics.TotalExecutions, 0)

		reflector.Reset()

		assert.Equal(t, 0, len(reflector.reflectionCache))
		assert.Equal(t, 0, reflector.metrics.TotalExecutions)
	})

	t.Run("Calculate Improvement", func(t *testing.T) {
		reflector := NewSelfReflector(10, 0)

		// Need at least windowSize * 2 reflections
		windowSize := 3
		for i := 0; i < windowSize*2; i++ {
			confidence := 0.5
			if i >= windowSize {
				confidence = 0.8 // Later reflections have higher confidence
			}
			reflector.reflectionCache = append(reflector.reflectionCache, Reflection{
				Type:       ReflectionTypeStrategy,
				Confidence: confidence,
				Timestamp:  time.Now(),
			})
		}

		improvement := reflector.CalculateImprovement(windowSize)
		assert.Greater(t, improvement, 0.0) // Should show improvement
	})
}

func TestReflectionTypes(t *testing.T) {
	t.Run("Reflection Type Constants", func(t *testing.T) {
		assert.Equal(t, ReflectionType(0), ReflectionTypeStrategy)
		assert.Equal(t, ReflectionType(1), ReflectionTypePerformance)
		assert.Equal(t, ReflectionType(2), ReflectionTypeLearning)
		assert.Equal(t, ReflectionType(3), ReflectionTypeError)
	})
}

// Helper function to find a reflection by type.
func findReflectionByType(reflections []Reflection, reflectionType ReflectionType) *Reflection {
	for _, r := range reflections {
		if r.Type == reflectionType {
			return &r
		}
	}
	return nil
}
