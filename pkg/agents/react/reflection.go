package react

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// ReflectionType categorizes different types of reflections.
type ReflectionType int

const (
	// ReflectionTypeStrategy reflects on overall strategy effectiveness.
	ReflectionTypeStrategy ReflectionType = iota
	// ReflectionTypePerformance reflects on performance metrics.
	ReflectionTypePerformance
	// ReflectionTypeLearning reflects on learned patterns.
	ReflectionTypeLearning
	// ReflectionTypeError reflects on errors and failures.
	ReflectionTypeError
)

// Reflection represents an insight gained from analyzing past executions.
type Reflection struct {
	Type           ReflectionType
	Insight        string
	Confidence     float64
	Recommendation string
	Evidence       []string
	Timestamp      time.Time
}

// SelfReflector implements self-reflection capabilities for agents.
type SelfReflector struct {
	depth                int
	delay                time.Duration
	successRateThreshold float64
	reflectionCache      []Reflection
	patterns             map[string]*Pattern
	metrics              *PerformanceMetrics
	mu                   sync.RWMutex // Protects concurrent access to internal state
}

// Pattern represents a recurring pattern in agent behavior.
type Pattern struct {
	Name        string
	Occurrences int
	SuccessRate float64
	LastSeen    time.Time
	Context     map[string]interface{}
}

// PerformanceMetrics tracks agent performance over time.
type PerformanceMetrics struct {
	TotalExecutions   int
	SuccessfulRuns    int
	FailedRuns        int
	AverageIterations float64
	AverageDuration   time.Duration
	ToolUsageStats    map[string]*ToolStats
	ErrorPatterns     map[string]int
}

// ToolStats tracks statistics for individual tool usage.
type ToolStats struct {
	UsageCount  int
	SuccessRate float64
	AvgDuration time.Duration
	LastUsed    time.Time
}

// NewSelfReflector creates a new self-reflection module.
func NewSelfReflector(depth int, delay time.Duration) *SelfReflector {
	return NewSelfReflectorWithThreshold(depth, delay, 0.6)
}

// NewSelfReflectorWithThreshold creates a new self-reflection module with configurable success rate threshold.
func NewSelfReflectorWithThreshold(depth int, delay time.Duration, successRateThreshold float64) *SelfReflector {
	if successRateThreshold <= 0 || successRateThreshold > 1 {
		successRateThreshold = 0.6 // Default fallback
	}
	return &SelfReflector{
		depth:                depth,
		delay:                delay,
		successRateThreshold: successRateThreshold,
		reflectionCache:      make([]Reflection, 0),
		patterns:             make(map[string]*Pattern),
		metrics: &PerformanceMetrics{
			ToolUsageStats: make(map[string]*ToolStats),
			ErrorPatterns:  make(map[string]int),
		},
	}
}

// Reflect analyzes an execution record and generates reflections.
func (sr *SelfReflector) Reflect(ctx context.Context, record ExecutionRecord) []Reflection {
	logger := logging.GetLogger()
	logger.Debug(ctx, "Starting reflection on execution record")

	// Wait for delay to allow for processing
	if sr.delay > 0 {
		time.Sleep(sr.delay)
	}

	reflections := make([]Reflection, 0)

	// Update metrics first
	sr.updateMetrics(record)

	// Generate different types of reflections
	if strategyReflection := sr.reflectOnStrategy(ctx, record); strategyReflection != nil {
		reflections = append(reflections, *strategyReflection)
	}

	if performanceReflection := sr.reflectOnPerformance(ctx, record); performanceReflection != nil {
		reflections = append(reflections, *performanceReflection)
	}

	if learningReflection := sr.reflectOnLearning(ctx, record); learningReflection != nil {
		reflections = append(reflections, *learningReflection)
	}

	if record.Error != nil {
		if errorReflection := sr.reflectOnError(ctx, record); errorReflection != nil {
			reflections = append(reflections, *errorReflection)
		}
	}

	// Identify patterns across multiple executions (thread-safe)
	sr.mu.RLock()
	cacheLen := len(sr.reflectionCache)
	sr.mu.RUnlock()

	if cacheLen >= sr.depth {
		if patternReflection := sr.identifyPatterns(ctx); patternReflection != nil {
			reflections = append(reflections, *patternReflection)
		}
	}

	// Cache reflections for pattern analysis (thread-safe)
	sr.mu.Lock()
	sr.reflectionCache = append(sr.reflectionCache, reflections...)
	if len(sr.reflectionCache) > sr.depth*10 {
		// Keep only recent reflections
		sr.reflectionCache = sr.reflectionCache[len(sr.reflectionCache)-sr.depth*10:]
	}
	sr.mu.Unlock()

	logger.Info(ctx, "Generated %d reflections", len(reflections))
	return reflections
}

// reflectOnStrategy analyzes strategic decisions.
func (sr *SelfReflector) reflectOnStrategy(ctx context.Context, record ExecutionRecord) *Reflection {
	logger := logging.GetLogger()

	// Analyze action sequence for strategic insights
	if len(record.Actions) == 0 {
		return nil
	}

	// Check for inefficient patterns
	toolSequence := make([]string, len(record.Actions))
	for i, action := range record.Actions {
		toolSequence[i] = action.Tool
	}

	// Detect repeated tool usage
	toolCounts := make(map[string]int)
	for _, tool := range toolSequence {
		toolCounts[tool]++
	}

	for tool, count := range toolCounts {
		if count > 3 {
			logger.Debug(ctx, "Detected repeated use of tool %s (%d times)", tool, count)
			return &Reflection{
				Type:       ReflectionTypeStrategy,
				Insight:    fmt.Sprintf("Tool '%s' was used %d times, suggesting potential inefficiency", tool, count),
				Confidence: 0.7,
				Recommendation: "Consider caching results or using a more powerful tool for complex queries",
				Evidence: []string{
					fmt.Sprintf("Tool used %d times in single execution", count),
					fmt.Sprintf("Total actions: %d", len(record.Actions)),
				},
				Timestamp: time.Now(),
			}
		}
	}

	// Analyze success patterns
	if record.Success && len(record.Actions) < 3 {
		return &Reflection{
			Type:       ReflectionTypeStrategy,
			Insight:    "Task completed efficiently with minimal tool usage",
			Confidence: 0.9,
			Recommendation: "Current strategy is effective for this task type",
			Evidence: []string{
				fmt.Sprintf("Completed in %d actions", len(record.Actions)),
			},
			Timestamp: time.Now(),
		}
	}

	return nil
}

// reflectOnPerformance analyzes performance metrics.
func (sr *SelfReflector) reflectOnPerformance(ctx context.Context, record ExecutionRecord) *Reflection {
	// Thread-safe access to metrics
	sr.mu.RLock()
	totalExecutions := sr.metrics.TotalExecutions
	successfulRuns := sr.metrics.SuccessfulRuns
	failedRuns := sr.metrics.FailedRuns
	averageIterations := sr.metrics.AverageIterations
	sr.mu.RUnlock()

	if totalExecutions < 5 {
		// Not enough data for performance reflection
		return nil
	}

	successRate := float64(successfulRuns) / float64(totalExecutions)

	// Check if performance is degrading
	if successRate < sr.successRateThreshold {
		return &Reflection{
			Type:       ReflectionTypePerformance,
			Insight:    fmt.Sprintf("Success rate is below threshold: %.2f%%", successRate*100),
			Confidence: 0.85,
			Recommendation: "increase_max_iterations",
			Evidence: []string{
				fmt.Sprintf("Success rate: %.2f%%", successRate*100),
				fmt.Sprintf("Failed runs: %d/%d", failedRuns, totalExecutions),
			},
			Timestamp: time.Now(),
		}
	}

	// Check iteration efficiency
	if averageIterations > 7 {
		return &Reflection{
			Type:       ReflectionTypePerformance,
			Insight:    "Average iteration count is high, suggesting complex problem solving",
			Confidence: 0.75,
			Recommendation: "use_rewoo_for_structured_tasks",
			Evidence: []string{
				fmt.Sprintf("Average iterations: %.2f", averageIterations),
			},
			Timestamp: time.Now(),
		}
	}

	return nil
}

// reflectOnLearning identifies learning opportunities.
func (sr *SelfReflector) reflectOnLearning(ctx context.Context, record ExecutionRecord) *Reflection {
	if !record.Success {
		return nil
	}

	// Look for successful patterns to learn from
	successfulTools := make([]string, 0)
	for _, action := range record.Actions {
		if action.Success {
			successfulTools = append(successfulTools, action.Tool)
		}
	}

	if len(successfulTools) > 0 {
		patternKey := fmt.Sprintf("%v", successfulTools)

		// Thread-safe access to patterns map
		sr.mu.Lock()
		if pattern, exists := sr.patterns[patternKey]; exists {
			pattern.Occurrences++
			pattern.LastSeen = time.Now()
			pattern.SuccessRate = (pattern.SuccessRate*float64(pattern.Occurrences-1) + 1.0) / float64(pattern.Occurrences)
		} else {
			sr.patterns[patternKey] = &Pattern{
				Name:        patternKey,
				Occurrences: 1,
				SuccessRate: 1.0,
				LastSeen:    time.Now(),
				Context:     record.Input,
			}
		}

		occurrences := sr.patterns[patternKey].Occurrences
		sr.mu.Unlock()

		if occurrences > 2 {
			return &Reflection{
				Type:       ReflectionTypeLearning,
				Insight:    fmt.Sprintf("Identified successful pattern: %s", patternKey),
				Confidence: 0.8,
				Recommendation: "Cache this pattern for similar tasks",
				Evidence: []string{
					fmt.Sprintf("Pattern seen %d times", sr.patterns[patternKey].Occurrences),
					fmt.Sprintf("Success rate: %.2f%%", sr.patterns[patternKey].SuccessRate*100),
				},
				Timestamp: time.Now(),
			}
		}
	}

	return nil
}

// reflectOnError analyzes errors and failures.
func (sr *SelfReflector) reflectOnError(ctx context.Context, record ExecutionRecord) *Reflection {
	if record.Error == nil {
		return nil
	}

	errorStr := record.Error.Error()

	// Thread-safe access to error patterns
	sr.mu.Lock()
	sr.metrics.ErrorPatterns[errorStr]++
	errorCount := sr.metrics.ErrorPatterns[errorStr]
	sr.mu.Unlock()

	// Check if this is a recurring error
	if errorCount > 2 {
		return &Reflection{
			Type:       ReflectionTypeError,
			Insight:    fmt.Sprintf("Recurring error detected: %s", errorStr),
			Confidence: 0.9,
			Recommendation: "Add specific error handling or retry logic",
			Evidence: []string{
				fmt.Sprintf("Error occurred %d times", sr.metrics.ErrorPatterns[errorStr]),
				fmt.Sprintf("Last occurrence: %v", record.Timestamp),
			},
			Timestamp: time.Now(),
		}
	}

	// Analyze error context
	failedTools := make([]string, 0)
	for _, action := range record.Actions {
		if !action.Success {
			failedTools = append(failedTools, action.Tool)
		}
	}

	if len(failedTools) > 0 {
		return &Reflection{
			Type:       ReflectionTypeError,
			Insight:    fmt.Sprintf("Tools failed during execution: %v", failedTools),
			Confidence: 0.75,
			Recommendation: "Consider alternative tools or improve error handling",
			Evidence: []string{
				fmt.Sprintf("Failed tools: %v", failedTools),
				fmt.Sprintf("Error: %s", errorStr),
			},
			Timestamp: time.Now(),
		}
	}

	return nil
}

// identifyPatterns looks for patterns across multiple reflections.
func (sr *SelfReflector) identifyPatterns(ctx context.Context) *Reflection {
	logger := logging.GetLogger()

	// Analyze reflection cache for meta-patterns (thread-safe)
	sr.mu.RLock()
	typeCount := make(map[ReflectionType]int)
	for _, ref := range sr.reflectionCache {
		typeCount[ref.Type]++
	}
	cacheLen := len(sr.reflectionCache)
	sr.mu.RUnlock()

	// Find dominant reflection type
	var dominantType ReflectionType
	maxCount := 0
	for rType, count := range typeCount {
		if count > maxCount {
			dominantType = rType
			maxCount = count
		}
	}

	if float64(maxCount) > float64(cacheLen)*0.5 {
		logger.Debug(ctx, "Identified dominant reflection pattern: %v", dominantType)

		insight := ""
		recommendation := ""

		switch dominantType {
		case ReflectionTypeError:
			insight = "Frequent errors suggest systemic issues"
			recommendation = "Review tool configuration and error handling"
		case ReflectionTypePerformance:
			insight = "Performance concerns are dominant"
			recommendation = "Consider optimization strategies"
		case ReflectionTypeStrategy:
			insight = "Strategy adjustments are frequently needed"
			recommendation = "Implement adaptive strategy selection"
		}

		if insight != "" {
			return &Reflection{
				Type:           ReflectionTypeLearning,
				Insight:        insight,
				Confidence:     0.7,
				Recommendation: recommendation,
				Evidence: []string{
					fmt.Sprintf("Pattern type %v appeared %d times", dominantType, maxCount),
					fmt.Sprintf("Total reflections analyzed: %d", cacheLen),
				},
				Timestamp: time.Now(),
			}
		}
	}

	return nil
}

// updateMetrics updates performance metrics based on execution record.
func (sr *SelfReflector) updateMetrics(record ExecutionRecord) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.metrics.TotalExecutions++

	if record.Success {
		sr.metrics.SuccessfulRuns++
	} else {
		sr.metrics.FailedRuns++
	}

	// Update average iterations
	actionCount := float64(len(record.Actions))
	sr.metrics.AverageIterations = (sr.metrics.AverageIterations*float64(sr.metrics.TotalExecutions-1) + actionCount) / float64(sr.metrics.TotalExecutions)

	// Update tool usage stats
	for _, action := range record.Actions {
		if stats, exists := sr.metrics.ToolUsageStats[action.Tool]; exists {
			stats.UsageCount++
			if action.Success {
				stats.SuccessRate = (stats.SuccessRate*float64(stats.UsageCount-1) + 1.0) / float64(stats.UsageCount)
			} else {
				stats.SuccessRate = (stats.SuccessRate * float64(stats.UsageCount-1)) / float64(stats.UsageCount)
			}
			stats.AvgDuration += (action.Duration - stats.AvgDuration) / time.Duration(stats.UsageCount)
			stats.LastUsed = time.Now()
		} else {
			successRate := 0.0
			if action.Success {
				successRate = 1.0
			}
			sr.metrics.ToolUsageStats[action.Tool] = &ToolStats{
				UsageCount:  1,
				SuccessRate: successRate,
				AvgDuration: action.Duration,
				LastUsed:    time.Now(),
			}
		}
	}
}

// GetTopReflections returns the most confident reflections (thread-safe).
func (sr *SelfReflector) GetTopReflections(n int) []Reflection {
	sr.mu.RLock()
	// Sort reflections by confidence
	sorted := make([]Reflection, len(sr.reflectionCache))
	copy(sorted, sr.reflectionCache)
	sr.mu.RUnlock()

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Confidence > sorted[j].Confidence
	})

	if n > len(sorted) {
		n = len(sorted)
	}

	return sorted[:n]
}

// GetMetrics returns current performance metrics.
func (sr *SelfReflector) GetMetrics() *PerformanceMetrics {
	return sr.metrics
}

// GetPatterns returns identified patterns (thread-safe).
func (sr *SelfReflector) GetPatterns() map[string]*Pattern {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	// Return a copy of the patterns map to avoid concurrent access
	patterns := make(map[string]*Pattern)
	for k, v := range sr.patterns {
		patterns[k] = v
	}
	return patterns
}

// Reset clears all reflection data (thread-safe).
func (sr *SelfReflector) Reset() {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	sr.reflectionCache = make([]Reflection, 0)
	sr.patterns = make(map[string]*Pattern)
	sr.metrics = &PerformanceMetrics{
		ToolUsageStats: make(map[string]*ToolStats),
		ErrorPatterns:  make(map[string]int),
	}
}

// CalculateImprovement compares metrics over time windows (thread-safe).
func (sr *SelfReflector) CalculateImprovement(windowSize int) float64 {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	if len(sr.reflectionCache) < windowSize*2 {
		return 0.0
	}

	// Compare recent window with previous window
	recentStart := len(sr.reflectionCache) - windowSize
	previousStart := recentStart - windowSize

	recentConfidence := 0.0
	previousConfidence := 0.0

	for i := previousStart; i < recentStart; i++ {
		previousConfidence += sr.reflectionCache[i].Confidence
	}

	for i := recentStart; i < len(sr.reflectionCache); i++ {
		recentConfidence += sr.reflectionCache[i].Confidence
	}

	previousAvg := previousConfidence / float64(windowSize)
	recentAvg := recentConfidence / float64(windowSize)

	improvement := (recentAvg - previousAvg) / math.Max(previousAvg, 0.01)
	return improvement
}
