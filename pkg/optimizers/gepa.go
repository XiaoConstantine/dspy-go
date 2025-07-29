// Package optimizers implements the GEPA (Generative Evolutionary Prompt Adaptation) algorithm.
//
// This file contains both the core functional GEPA implementation and advanced research-grade
// features including multi-level reflection systems, LLM-based self-critique, and sophisticated
// pattern analysis capabilities. Some advanced functions are currently unused but preserved for
// future integration and research purposes.
//
//nolint:unused // Advanced reflection and analysis functions are research implementations
package optimizers

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
	"github.com/sourcegraph/conc/pool"
)

// Context keys for GEPA state management.
type contextKey string

const (
	gepaStateKey      contextKey = "gepa_state"
	candidateIDKey    contextKey = "current_candidate_id"
	executionPhaseKey contextKey = "execution_phase"
)

// GEPAConfig contains configuration options for GEPA optimizer.
type GEPAConfig struct {
	// Evolutionary parameters
	PopulationSize int     `json:"population_size"` // Default: 20
	MaxGenerations int     `json:"max_generations"` // Default: 10
	MutationRate   float64 `json:"mutation_rate"`   // Default: 0.3
	CrossoverRate  float64 `json:"crossover_rate"`  // Default: 0.7
	ElitismRate    float64 `json:"elitism_rate"`    // Default: 0.1

	// Reflection parameters
	ReflectionFreq   int     `json:"reflection_frequency"` // Default: 2
	ReflectionDepth  int     `json:"reflection_depth"`     // Default: 3
	SelfCritiqueTemp float64 `json:"self_critique_temp"`   // Default: 0.7

	// Selection parameters
	TournamentSize    int    `json:"tournament_size"`    // Default: 3
	SelectionStrategy string `json:"selection_strategy"` // Default: "tournament" | "roulette" | "pareto" | "adaptive_pareto"

	// Convergence parameters
	ConvergenceThreshold float64 `json:"convergence_threshold"` // Default: 0.01
	StagnationLimit      int     `json:"stagnation_limit"`      // Default: 3

	// Performance parameters
	EvaluationBatchSize int `json:"evaluation_batch_size"` // Default: 5
	ConcurrencyLevel    int `json:"concurrency_level"`     // Default: 3

	// LLM parameters
	GenerationModel string  `json:"generation_model"` // Default: uses core.GetDefaultLLM()
	ReflectionModel string  `json:"reflection_model"` // Default: uses core.GetTeacherLLM()
	Temperature     float64 `json:"temperature"`      // Default: 0.8
	MaxTokens       int     `json:"max_tokens"`       // Default: 500
}

// DefaultGEPAConfig returns the default configuration for GEPA.
func DefaultGEPAConfig() *GEPAConfig {
	return &GEPAConfig{
		PopulationSize:       20,
		MaxGenerations:       10,
		MutationRate:         0.3,
		CrossoverRate:        0.7,
		ElitismRate:          0.1,
		ReflectionFreq:       2,
		ReflectionDepth:      3,
		SelfCritiqueTemp:     0.7,
		TournamentSize:       3,
		SelectionStrategy:    "adaptive_pareto", // Use adaptive Pareto as default for sophisticated multi-objective optimization
		ConvergenceThreshold: 0.01,
		StagnationLimit:      3,
		EvaluationBatchSize:  5,
		ConcurrencyLevel:     3,
		Temperature:          0.8,
		MaxTokens:            500,
	}
}

// GEPACandidate represents a single prompt candidate in the GEPA population.
type GEPACandidate struct {
	ID             string                 `json:"id"`
	ModuleName     string                 `json:"module_name"`
	Instruction    string                 `json:"instruction"`
	Demonstrations []core.Example         `json:"demonstrations"`
	Generation     int                    `json:"generation"`
	Fitness        float64                `json:"fitness"`
	ParentIDs      []string               `json:"parent_ids"`
	CreatedAt      time.Time              `json:"created_at"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// Population represents a generation of prompt candidates.
type Population struct {
	Candidates    []*GEPACandidate `json:"candidates"`
	Generation    int              `json:"generation"`
	BestFitness   float64          `json:"best_fitness"`
	BestCandidate *GEPACandidate   `json:"best_candidate"`
	Size          int              `json:"size"`
}

// ExecutionTrace tracks the execution of a module for reflection analysis.
type ExecutionTrace struct {
	CandidateID string                 `json:"candidate_id"`
	ModuleName  string                 `json:"module_name"`
	Inputs      map[string]any         `json:"inputs"`
	Outputs     map[string]any         `json:"outputs"`
	Error       error                  `json:"error"`
	Duration    time.Duration          `json:"duration"`
	Success     bool                   `json:"success"`
	Timestamp   time.Time              `json:"timestamp"`
	ContextData map[string]interface{} `json:"context_data"`
}

// ExecutionPatterns represents analyzed patterns from execution traces.
type ExecutionPatterns struct {
	SuccessRate         float64        `json:"success_rate"`
	SuccessCount        int            `json:"success_count"`
	TotalExecutions     int            `json:"total_executions"`
	AverageResponseTime time.Duration  `json:"average_response_time"`
	CommonFailures      []string       `json:"common_failures"`
	QualityIndicators   []string       `json:"quality_indicators"`
	ErrorDistribution   map[string]int `json:"error_distribution"`
	PerformanceTrends   []float64      `json:"performance_trends"`
}

// ReflectionResult contains the results of reflecting on a prompt candidate.
type ReflectionResult struct {
	CandidateID     string    `json:"candidate_id"`
	Strengths       []string  `json:"strengths"`
	Weaknesses      []string  `json:"weaknesses"`
	Suggestions     []string  `json:"suggestions"`
	ConfidenceScore float64   `json:"confidence_score"`
	Timestamp       time.Time `json:"timestamp"`
	ReflectionDepth int       `json:"reflection_depth"`
}

// ConvergenceStatus tracks convergence indicators for the optimization.
type ConvergenceStatus struct {
	StagnationCount                int     `json:"stagnation_count"`
	PrematureConvergenceRisk       string  `json:"premature_convergence_risk"`
	ExplorationExploitationBalance string  `json:"exploration_exploitation_balance"`
	DiversityIndex                 float64 `json:"diversity_index"`
	IsConverged                    bool    `json:"is_converged"`
}

// PopulationInsights contains analysis of the current population.
type PopulationInsights struct {
	DiversityIndex         float64  `json:"diversity_index"`
	AverageFitness         float64  `json:"average_fitness"`
	BestFitness            float64  `json:"best_fitness"`
	WorstFitness           float64  `json:"worst_fitness"`
	FitnessVariance        float64  `json:"fitness_variance"`
	HighPerformingPatterns []string `json:"high_performing_patterns"`
	CommonWeaknesses       []string `json:"common_weaknesses"`
}

// CandidateMetrics tracks detailed metrics for each candidate.
type CandidateMetrics struct {
	TotalEvaluations int                    `json:"total_evaluations"`
	SuccessCount     int                    `json:"success_count"`
	AverageFitness   float64                `json:"average_fitness"`
	BestFitness      float64                `json:"best_fitness"`
	ExecutionTimes   []time.Duration        `json:"execution_times"`
	ErrorCounts      map[string]int         `json:"error_counts"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// GEPAState tracks the complete state of GEPA optimization.
type GEPAState struct {
	CurrentGeneration        int                               `json:"current_generation"`
	BestCandidate            *GEPACandidate                    `json:"best_candidate"`
	BestFitness              float64                           `json:"best_fitness"`
	PopulationHistory        []*Population                     `json:"population_history"`
	ReflectionHistory        []*ReflectionResult               `json:"reflection_history"`
	ConvergenceStatus        *ConvergenceStatus                `json:"convergence_status"`
	StartTime                time.Time                         `json:"start_time"`
	LastImprovement          time.Time                         `json:"last_improvement"`
	ExecutionTraces          map[string][]ExecutionTrace       `json:"execution_traces"`
	CandidateMetrics         map[string]*CandidateMetrics      `json:"candidate_metrics"`
	MultiObjectiveFitnessMap map[string]*MultiObjectiveFitness `json:"multi_objective_fitness_map"`
	mu                       sync.RWMutex
}

// NewGEPAState creates a new GEPA optimization state.
func NewGEPAState() *GEPAState {
	return &GEPAState{
		CurrentGeneration:        0,
		BestFitness:              0.0,
		PopulationHistory:        make([]*Population, 0),
		ReflectionHistory:        make([]*ReflectionResult, 0),
		StartTime:                time.Now(),
		LastImprovement:          time.Now(),
		ExecutionTraces:          make(map[string][]ExecutionTrace),
		CandidateMetrics:         make(map[string]*CandidateMetrics),
		MultiObjectiveFitnessMap: make(map[string]*MultiObjectiveFitness),
		ConvergenceStatus: &ConvergenceStatus{
			StagnationCount:                0,
			PrematureConvergenceRisk:       "low",
			ExplorationExploitationBalance: "balanced",
			DiversityIndex:                 1.0,
			IsConverged:                    false,
		},
	}
}

// AddTrace adds an execution trace to the state.
func (s *GEPAState) AddTrace(trace *ExecutionTrace) {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := fmt.Sprintf("%s-%s", trace.CandidateID, trace.ModuleName)
	s.ExecutionTraces[key] = append(s.ExecutionTraces[key], *trace)

	// Update candidate metrics
	if _, exists := s.CandidateMetrics[trace.CandidateID]; !exists {
		s.CandidateMetrics[trace.CandidateID] = &CandidateMetrics{
			TotalEvaluations: 0,
			SuccessCount:     0,
			AverageFitness:   0.0,
			BestFitness:      0.0,
			ExecutionTimes:   make([]time.Duration, 0),
			ErrorCounts:      make(map[string]int),
			Metadata:         make(map[string]interface{}),
		}
	}

	metrics := s.CandidateMetrics[trace.CandidateID]
	metrics.TotalEvaluations++
	if trace.Success {
		metrics.SuccessCount++
	}
	metrics.ExecutionTimes = append(metrics.ExecutionTimes, trace.Duration)

	if trace.Error != nil {
		errorType := trace.Error.Error()
		metrics.ErrorCounts[errorType]++
	}
}

// GetTracesForCandidate returns all execution traces for a specific candidate.
func (s *GEPAState) GetTracesForCandidate(candidateID string) []ExecutionTrace {
	s.mu.RLock()
	defer s.mu.RUnlock()

	traces := make([]ExecutionTrace, 0)
	for key, traceList := range s.ExecutionTraces {
		if strings.HasPrefix(key, candidateID+"-") {
			traces = append(traces, traceList...)
		}
	}

	return traces
}

// PerformanceLogger tracks performance metrics for GEPA optimization.
type PerformanceLogger struct {
	metrics        map[string]*CandidateMetrics
	contextMetrics map[string]*ContextPerformanceMetrics
	mu             sync.RWMutex
}

// NewPerformanceLogger creates a new performance logger.
func NewPerformanceLogger() *PerformanceLogger {
	return &PerformanceLogger{
		metrics:        make(map[string]*CandidateMetrics),
		contextMetrics: make(map[string]*ContextPerformanceMetrics),
	}
}

// LogCandidateMetrics logs metrics for a candidate.
func (pl *PerformanceLogger) LogCandidateMetrics(candidateID string, metrics *CandidateMetrics) {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	pl.metrics[candidateID] = metrics
}

// GetCandidateMetrics retrieves metrics for a candidate.
func (pl *PerformanceLogger) GetCandidateMetrics(candidateID string) *CandidateMetrics {
	pl.mu.RLock()
	defer pl.mu.RUnlock()
	return pl.metrics[candidateID]
}

// GEPA represents the main GEPA optimizer.
type GEPA struct {
	config *GEPAConfig
	state  *GEPAState

	// LLMs
	generationLLM core.LLM
	reflectionLLM core.LLM

	// Interceptor integration
	interceptorChain *core.InterceptorChain

	// Performance monitoring
	progressReporter  core.ProgressReporter
	performanceLogger *PerformanceLogger

	// Random number generator
	rng *rand.Rand
}

// MultiObjectiveFitness represents fitness across multiple objectives for Pareto-based selection.
type MultiObjectiveFitness struct {
	// Core objectives
	SuccessRate    float64 `json:"success_rate"`   // Objective 1: Basic success rate
	OutputQuality  float64 `json:"output_quality"` // Objective 2: Quality of outputs
	Efficiency     float64 `json:"efficiency"`     // Objective 3: Execution efficiency
	Robustness     float64 `json:"robustness"`     // Objective 4: Error handling capability
	Generalization float64 `json:"generalization"` // Objective 5: Cross-context performance

	// Meta-objectives
	Diversity  float64 `json:"diversity"`  // Population diversity contribution
	Innovation float64 `json:"innovation"` // Novel solution characteristics

	// Aggregated score for backward compatibility
	WeightedScore float64 `json:"weighted_score"`
}

// ParetoFront represents a Pareto front for multi-objective optimization.
type ParetoFront struct {
	Candidates []*GEPACandidate `json:"candidates"`
	Rank       int              `json:"rank"` // Pareto rank (1 = best front)
	Size       int              `json:"size"`
}

// ContextAwarePerformanceTracker tracks performance metrics with context awareness.
type ContextAwarePerformanceTracker struct {
	// Performance metrics by context type
	contextMetrics map[string]*ContextPerformanceMetrics

	// Historical context patterns
	contextPatterns map[string]*ContextPattern

	// Context similarity analysis
	contextSimilarity map[string]map[string]float64
}

// ContextPerformanceMetrics tracks performance within a specific context.
type ContextPerformanceMetrics struct {
	ContextType      string                 `json:"context_type"`
	ExecutionCount   int                    `json:"execution_count"`
	SuccessRate      float64                `json:"success_rate"`
	AverageLatency   time.Duration          `json:"average_latency"`
	QualityScore     float64                `json:"quality_score"`
	ErrorPatterns    map[string]int         `json:"error_patterns"`
	PerformanceTrend []float64              `json:"performance_trend"`
	LastUpdated      time.Time              `json:"last_updated"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// ContextPattern represents learned patterns within execution contexts.
type ContextPattern struct {
	PatternID         string                 `json:"pattern_id"`
	ContextFeatures   map[string]interface{} `json:"context_features"`
	SuccessFactors    []string               `json:"success_factors"`
	FailureFactors    []string               `json:"failure_factors"`
	OptimalCandidates []string               `json:"optimal_candidates"`
	Confidence        float64                `json:"confidence"`
	LastUpdated       time.Time              `json:"last_updated"`
}

// PerformanceContext represents the context of a performance measurement.
type PerformanceContext struct {
	// Execution context
	ExecutionID string `json:"execution_id"`
	CandidateID string `json:"candidate_id"`
	ModuleName  string `json:"module_name"`

	// Input characteristics
	InputTypes      []string `json:"input_types"`
	InputSize       int      `json:"input_size"`
	InputComplexity float64  `json:"input_complexity"`

	// System context
	SystemLoad      float64 `json:"system_load"`
	MemoryUsage     float64 `json:"memory_usage"`
	ConcurrentTasks int     `json:"concurrent_tasks"`

	// Generation context
	Generation        int     `json:"generation"`
	PopulationSize    int     `json:"population_size"`
	SelectionPressure float64 `json:"selection_pressure"`

	// Temporal context
	TimeOfDay      int    `json:"time_of_day"`     // Hour of day
	ExecutionPhase string `json:"execution_phase"` // init, evolution, reflection

	// Custom context data
	CustomData map[string]interface{} `json:"custom_data"`
}

// calculateMultiObjectiveFitness calculates fitness across multiple objectives.
func (g *GEPA) calculateMultiObjectiveFitness(candidateID string, inputs, outputs map[string]any, err error, context map[string]interface{}) *MultiObjectiveFitness {
	fitness := &MultiObjectiveFitness{}

	// Objective 1: Success Rate (0-1)
	if err == nil && outputs != nil && len(outputs) > 0 {
		fitness.SuccessRate = 1.0
	} else {
		fitness.SuccessRate = 0.0
	}

	// Objective 2: Output Quality (0-1)
	if outputs != nil {
		fitness.OutputQuality = g.assessOutputQuality(outputs)
	}

	// Objective 3: Efficiency (0-1, higher is better) - Context-aware
	fitness.Efficiency = g.assessContextAwareEfficiency(candidateID, inputs, outputs, err, context)

	// Objective 4: Robustness (0-1, based on error handling and edge cases)
	fitness.Robustness = g.assessRobustness(inputs, outputs, err, candidateID)

	// Objective 5: Generalization (0-1, based on performance across different contexts)
	fitness.Generalization = g.assessGeneralization(candidateID, inputs, outputs)

	// Objective 6: Diversity (0-1, contribution to population diversity)
	fitness.Diversity = g.assessDiversityContribution(candidateID)

	// Objective 7: Innovation (0-1, novelty of the solution)
	fitness.Innovation = g.assessInnovation(candidateID, outputs)

	// Calculate weighted score for backward compatibility
	weights := map[string]float64{
		"success":        0.25,
		"quality":        0.20,
		"efficiency":     0.15,
		"robustness":     0.15,
		"generalization": 0.15,
		"diversity":      0.05,
		"innovation":     0.05,
	}

	fitness.WeightedScore =
		fitness.SuccessRate*weights["success"] +
			fitness.OutputQuality*weights["quality"] +
			fitness.Efficiency*weights["efficiency"] +
			fitness.Robustness*weights["robustness"] +
			fitness.Generalization*weights["generalization"] +
			fitness.Diversity*weights["diversity"] +
			fitness.Innovation*weights["innovation"]

	return fitness
}

// assessRobustness evaluates how well a candidate handles edge cases and errors.
func (g *GEPA) assessRobustness(inputs, outputs map[string]any, err error, candidateID string) float64 {
	// Get historical performance for this candidate
	metrics := g.state.CandidateMetrics[candidateID]
	if metrics == nil {
		return 0.5 // Default for new candidates
	}

	robustness := 0.0

	// Factor 1: Error handling capability (40%)
	if metrics.TotalEvaluations > 0 {
		errorRate := float64(metrics.TotalEvaluations-metrics.SuccessCount) / float64(metrics.TotalEvaluations)
		robustness += 0.4 * (1.0 - errorRate)
	}

	// Factor 2: Consistency across different inputs (30%)
	if len(metrics.ExecutionTimes) > 1 {
		// Calculate coefficient of variation for execution times
		var sum, mean, variance float64
		for _, duration := range metrics.ExecutionTimes {
			sum += float64(duration.Nanoseconds())
		}
		mean = sum / float64(len(metrics.ExecutionTimes))

		for _, duration := range metrics.ExecutionTimes {
			diff := float64(duration.Nanoseconds()) - mean
			variance += diff * diff
		}
		variance /= float64(len(metrics.ExecutionTimes))

		stdDev := variance
		if mean > 0 {
			cv := stdDev / mean
			consistency := 1.0 / (1.0 + cv) // Lower CV = higher consistency
			robustness += 0.3 * consistency
		}
	} else {
		robustness += 0.15 // Partial credit for single execution
	}

	// Factor 3: Graceful degradation (30%)
	if err != nil {
		// Check if error is informative rather than catastrophic
		errorMsg := err.Error()
		if strings.Contains(errorMsg, "validation") ||
			strings.Contains(errorMsg, "format") ||
			strings.Contains(errorMsg, "input") {
			robustness += 0.1 // Partial credit for graceful errors
		}
	} else if outputs != nil {
		robustness += 0.3 // Full credit for successful execution
	}

	return robustness
}

// assessGeneralization evaluates performance across different contexts.
func (g *GEPA) assessGeneralization(candidateID string, inputs, outputs map[string]any) float64 {
	// Get execution history for this candidate
	traces := g.state.GetTracesForCandidate(candidateID)
	if len(traces) < 2 {
		return 0.5 // Default for candidates with limited history
	}

	// Analyze performance across different input patterns
	inputPatterns := make(map[string][]float64)
	for _, trace := range traces {
		// Create a simple pattern key based on input types and sizes
		pattern := g.createInputPattern(trace.Inputs)

		// Calculate success score for this trace
		score := 0.0
		if trace.Success && trace.Outputs != nil {
			score = 1.0
		}

		inputPatterns[pattern] = append(inputPatterns[pattern], score)
	}

	// Calculate generalization as consistency across patterns
	if len(inputPatterns) < 2 {
		return 0.7 // Good but not excellent for single pattern
	}

	totalVariance := 0.0
	patternCount := 0

	for _, scores := range inputPatterns {
		if len(scores) > 1 {
			// Calculate variance for this pattern
			var sum, mean, variance float64
			for _, score := range scores {
				sum += score
			}
			mean = sum / float64(len(scores))

			for _, score := range scores {
				diff := score - mean
				variance += diff * diff
			}
			variance /= float64(len(scores))

			totalVariance += variance
			patternCount++
		}
	}

	if patternCount == 0 {
		return 0.6
	}

	avgVariance := totalVariance / float64(patternCount)
	generalization := 1.0 - avgVariance // Lower variance = better generalization

	if generalization < 0 {
		generalization = 0
	}
	if generalization > 1 {
		generalization = 1
	}

	return generalization
}

// assessDiversityContribution evaluates how much a candidate contributes to population diversity.
func (g *GEPA) assessDiversityContribution(candidateID string) float64 {
	// Get current population
	if len(g.state.PopulationHistory) == 0 {
		return 1.0 // Maximum diversity for first generation
	}

	currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
	if currentPop == nil || len(currentPop.Candidates) == 0 {
		return 1.0
	}

	// Find the candidate
	var targetCandidate *GEPACandidate
	for _, candidate := range currentPop.Candidates {
		if candidate.ID == candidateID {
			targetCandidate = candidate
			break
		}
	}

	if targetCandidate == nil {
		return 0.5
	}

	// Calculate diversity based on instruction uniqueness
	uniqueness := 0.0
	similarities := make([]float64, 0)

	for _, other := range currentPop.Candidates {
		if other.ID != candidateID {
			similarity := g.calculateInstructionSimilarity(targetCandidate.Instruction, other.Instruction)
			similarities = append(similarities, similarity)
		}
	}

	if len(similarities) > 0 {
		// Calculate average similarity
		var sumSimilarity float64
		for _, sim := range similarities {
			sumSimilarity += sim
		}
		avgSimilarity := sumSimilarity / float64(len(similarities))
		uniqueness = 1.0 - avgSimilarity // Higher uniqueness = lower average similarity
	} else {
		uniqueness = 1.0
	}

	return uniqueness
}

// assessInnovation evaluates the novelty of a candidate's solution.
func (g *GEPA) assessInnovation(candidateID string, outputs map[string]any) float64 {
	// Innovation based on novel output patterns and solution approaches
	innovation := 0.0

	// Factor 1: Output pattern novelty (50%)
	if outputs != nil {
		patternNovelty := g.assessOutputPatternNovelty(outputs, candidateID)
		innovation += 0.5 * patternNovelty
	}

	// Factor 2: Solution approach uniqueness (30%)
	approachUniqueness := g.assessApproachUniqueness(candidateID)
	innovation += 0.3 * approachUniqueness

	// Factor 3: Problem-solving creativity (20%)
	creativity := g.assessCreativity(candidateID, outputs)
	innovation += 0.2 * creativity

	return innovation
}

// Helper methods for multi-objective fitness

// createInputPattern creates a pattern key from input data.
func (g *GEPA) createInputPattern(inputs map[string]any) string {
	if inputs == nil {
		return "empty"
	}

	pattern := fmt.Sprintf("count:%d", len(inputs))

	// Add type information
	types := make([]string, 0)
	for _, value := range inputs {
		if value == nil {
			types = append(types, "nil")
		} else {
			types = append(types, fmt.Sprintf("%T", value))
		}
	}

	if len(types) > 0 {
		pattern += fmt.Sprintf("_types:%v", types)
	}

	return pattern
}

// calculateInstructionSimilarity calculates similarity between two instructions.
func (g *GEPA) calculateInstructionSimilarity(inst1, inst2 string) float64 {
	if inst1 == inst2 {
		return 1.0
	}

	// Simple similarity based on common words
	words1 := strings.Fields(strings.ToLower(inst1))
	words2 := strings.Fields(strings.ToLower(inst2))

	if len(words1) == 0 && len(words2) == 0 {
		return 1.0
	}
	if len(words1) == 0 || len(words2) == 0 {
		return 0.0
	}

	// Create word frequency maps
	freq1 := make(map[string]int)
	freq2 := make(map[string]int)

	for _, word := range words1 {
		freq1[word]++
	}
	for _, word := range words2 {
		freq2[word]++
	}

	// Calculate Jaccard similarity
	intersection := 0
	union := 0

	allWords := make(map[string]bool)
	for word := range freq1 {
		allWords[word] = true
	}
	for word := range freq2 {
		allWords[word] = true
	}

	for word := range allWords {
		count1 := freq1[word]
		count2 := freq2[word]

		if count1 > 0 && count2 > 0 {
			intersection++
		}
		if count1 > 0 || count2 > 0 {
			union++
		}
	}

	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// assessOutputPatternNovelty evaluates the novelty of output patterns.
func (g *GEPA) assessOutputPatternNovelty(outputs map[string]any, candidateID string) float64 {
	// Compare with historical output patterns
	allTraces := make([]ExecutionTrace, 0)
	for _, traces := range g.state.ExecutionTraces {
		allTraces = append(allTraces, traces...)
	}

	if len(allTraces) < 2 {
		return 1.0 // High novelty for early candidates
	}

	// Create pattern signature for current outputs
	currentPattern := g.createOutputPattern(outputs)

	// Compare with historical patterns
	similarities := make([]float64, 0)
	for _, trace := range allTraces {
		if trace.CandidateID != candidateID && trace.Outputs != nil {
			historicalPattern := g.createOutputPattern(trace.Outputs)
			similarity := g.calculatePatternSimilarity(currentPattern, historicalPattern)
			similarities = append(similarities, similarity)
		}
	}

	if len(similarities) == 0 {
		return 1.0
	}

	// Calculate average similarity and return novelty as 1 - similarity
	var avgSimilarity float64
	for _, sim := range similarities {
		avgSimilarity += sim
	}
	avgSimilarity /= float64(len(similarities))

	novelty := 1.0 - avgSimilarity
	if novelty < 0 {
		novelty = 0
	}

	return novelty
}

// assessApproachUniqueness evaluates the uniqueness of the problem-solving approach.
func (g *GEPA) assessApproachUniqueness(candidateID string) float64 {
	// Get the candidate
	var candidate *GEPACandidate
	if len(g.state.PopulationHistory) > 0 {
		currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
		for _, c := range currentPop.Candidates {
			if c.ID == candidateID {
				candidate = c
				break
			}
		}
	}

	if candidate == nil {
		return 0.5
	}

	// Analyze instruction patterns for uniqueness
	instruction := candidate.Instruction

	// Look for unique keywords or patterns
	uniqueKeywords := []string{
		"innovative", "creative", "novel", "unique", "original",
		"alternative", "unconventional", "breakthrough", "revolutionary",
	}

	uniqueness := 0.0
	for _, keyword := range uniqueKeywords {
		if strings.Contains(strings.ToLower(instruction), keyword) {
			uniqueness += 0.1
		}
	}

	// Add uniqueness based on instruction length and complexity
	words := strings.Fields(instruction)
	if len(words) > 20 {
		uniqueness += 0.2 // Bonus for detailed instructions
	}

	// Check for specific methodological approaches
	methodKeywords := []string{
		"step-by-step", "systematic", "analytical", "holistic",
		"iterative", "recursive", "meta", "multi-level",
	}

	for _, keyword := range methodKeywords {
		if strings.Contains(strings.ToLower(instruction), keyword) {
			uniqueness += 0.15
		}
	}

	if uniqueness > 1.0 {
		uniqueness = 1.0
	}

	return uniqueness
}

// assessCreativity evaluates the creativity of the solution.
func (g *GEPA) assessCreativity(candidateID string, outputs map[string]any) float64 {
	if outputs == nil {
		return 0.0
	}

	creativity := 0.0

	// Factor 1: Output diversity within single execution (40%)
	outputTypes := make(map[string]int)
	for _, value := range outputs {
		if value != nil {
			outputTypes[fmt.Sprintf("%T", value)]++
		}
	}

	if len(outputTypes) > 1 {
		creativity += 0.4 * (float64(len(outputTypes)) / float64(len(outputs)))
	}

	// Factor 2: Creative language patterns in string outputs (40%)
	for _, value := range outputs {
		if str, ok := value.(string); ok {
			if g.containsCreativePatterns(str) {
				creativity += 0.4
				break // Only award once per execution
			}
		}
	}

	// Factor 3: Solution completeness and depth (20%)
	if len(outputs) > 0 {
		avgLength := 0
		validOutputs := 0

		for _, value := range outputs {
			if str, ok := value.(string); ok && str != "" {
				avgLength += len(str)
				validOutputs++
			}
		}

		if validOutputs > 0 {
			avgLength /= validOutputs
			if avgLength > 100 { // Substantial outputs
				creativity += 0.2
			} else if avgLength > 50 {
				creativity += 0.1
			}
		}
	}

	return creativity
}

// createOutputPattern creates a pattern signature from outputs.
func (g *GEPA) createOutputPattern(outputs map[string]any) string {
	if outputs == nil {
		return "empty"
	}

	pattern := fmt.Sprintf("count:%d", len(outputs))

	// Add type and size information
	for key, value := range outputs {
		if value == nil {
			pattern += fmt.Sprintf("_%s:nil", key)
		} else {
			valueType := fmt.Sprintf("%T", value)
			if str, ok := value.(string); ok {
				pattern += fmt.Sprintf("_%s:%s:len%d", key, valueType, len(str))
			} else {
				pattern += fmt.Sprintf("_%s:%s", key, valueType)
			}
		}
	}

	return pattern
}

// calculatePatternSimilarity calculates similarity between two patterns.
func (g *GEPA) calculatePatternSimilarity(pattern1, pattern2 string) float64 {
	if pattern1 == pattern2 {
		return 1.0
	}

	// Simple substring similarity
	shorter, longer := pattern1, pattern2
	if len(pattern1) > len(pattern2) {
		shorter, longer = pattern2, pattern1
	}

	if len(shorter) == 0 {
		if len(longer) == 0 {
			return 1.0
		}
		return 0.0
	}

	// Calculate longest common subsequence ratio
	commonChars := 0
	for i := 0; i < len(shorter); i++ {
		if i < len(longer) && shorter[i] == longer[i] {
			commonChars++
		}
	}

	return float64(commonChars) / float64(len(longer))
}

// containsCreativePatterns checks for creative language patterns.
func (g *GEPA) containsCreativePatterns(text string) bool {
	creativePatterns := []string{
		"metaphor", "analogy", "creative", "innovative", "imagine",
		"envision", "conceptual", "abstract", "artistic", "elegant",
		"sophisticated", "nuanced", "multifaceted", "dynamic",
	}

	lowerText := strings.ToLower(text)
	for _, pattern := range creativePatterns {
		if strings.Contains(lowerText, pattern) {
			return true
		}
	}

	return false
}

// Context-Aware Performance Tracking Methods

// NewContextAwarePerformanceTracker creates a new context-aware performance tracker.
func NewContextAwarePerformanceTracker() *ContextAwarePerformanceTracker {
	return &ContextAwarePerformanceTracker{
		contextMetrics:    make(map[string]*ContextPerformanceMetrics),
		contextPatterns:   make(map[string]*ContextPattern),
		contextSimilarity: make(map[string]map[string]float64),
	}
}

// assessContextAwareEfficiency evaluates efficiency with context awareness.
func (g *GEPA) assessContextAwareEfficiency(candidateID string, inputs, outputs map[string]any, err error, context map[string]interface{}) float64 {
	if g.performanceLogger == nil {
		return g.fallbackEfficiencyAssessment(context)
	}

	// Create performance context
	perfContext := g.createPerformanceContext(candidateID, inputs, outputs, context)

	// Get historical performance for similar contexts
	contextualBaseline := g.getContextualBaseline(perfContext)

	// Calculate raw efficiency
	rawEfficiency := g.calculateRawEfficiency(context, contextualBaseline)

	// Apply context-specific adjustments
	contextAdjustedEfficiency := g.applyContextAdjustments(rawEfficiency, perfContext)

	// Update context tracking
	g.updateContextMetrics(perfContext, outputs, err, contextAdjustedEfficiency)

	return contextAdjustedEfficiency
}

// createPerformanceContext creates a performance context from execution data.
func (g *GEPA) createPerformanceContext(candidateID string, inputs, outputs map[string]any, context map[string]interface{}) *PerformanceContext {
	perfContext := &PerformanceContext{
		ExecutionID: fmt.Sprintf("%s-%d", candidateID, time.Now().UnixNano()),
		CandidateID: candidateID,
		Generation:  g.state.CurrentGeneration,
		CustomData:  make(map[string]interface{}),
	}

	// Extract input characteristics
	if inputs != nil {
		perfContext.InputSize = len(inputs)
		perfContext.InputTypes = make([]string, 0)
		for _, value := range inputs {
			if value != nil {
				perfContext.InputTypes = append(perfContext.InputTypes, fmt.Sprintf("%T", value))
			}
		}
		perfContext.InputComplexity = g.calculateInputComplexity(inputs)
	}

	// Extract system context from provided context
	if context != nil {
		if moduleName, ok := context["module_name"].(string); ok {
			perfContext.ModuleName = moduleName
		}
		if systemLoad, ok := context["system_load"].(float64); ok {
			perfContext.SystemLoad = systemLoad
		}
		if memUsage, ok := context["memory_usage"].(float64); ok {
			perfContext.MemoryUsage = memUsage
		}
		if concurrentTasks, ok := context["concurrent_tasks"].(int); ok {
			perfContext.ConcurrentTasks = concurrentTasks
		}
		if phase, ok := context["execution_phase"].(string); ok {
			perfContext.ExecutionPhase = phase
		}

		// Copy custom data
		for key, value := range context {
			if !g.isReservedContextKey(key) {
				perfContext.CustomData[key] = value
			}
		}
	}

	// Set temporal context
	perfContext.TimeOfDay = time.Now().Hour()

	// Set generation context
	if len(g.state.PopulationHistory) > 0 {
		currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
		if currentPop != nil {
			perfContext.PopulationSize = currentPop.Size
			// Calculate selection pressure based on population diversity
			perfContext.SelectionPressure = g.calculateSelectionPressure(currentPop)
		}
	}

	return perfContext
}

// getContextualBaseline gets performance baseline for similar contexts.
func (g *GEPA) getContextualBaseline(perfContext *PerformanceContext) time.Duration {
	// Find similar contexts
	similarContexts := g.findSimilarContexts(perfContext)

	if len(similarContexts) == 0 {
		return time.Second // Default baseline
	}

	// Calculate weighted average of similar contexts
	totalLatency := time.Duration(0)
	totalWeight := 0.0

	for contextType, similarity := range similarContexts {
		if g.performanceLogger != nil {
			g.performanceLogger.mu.RLock()
			if metrics, exists := g.performanceLogger.contextMetrics[contextType]; exists {
				weight := similarity * float64(metrics.ExecutionCount)
				totalLatency += time.Duration(float64(metrics.AverageLatency) * weight)
				totalWeight += weight
			}
			g.performanceLogger.mu.RUnlock()
		}
	}

	if totalWeight > 0 {
		return time.Duration(float64(totalLatency) / totalWeight)
	}

	return time.Second
}

// calculateRawEfficiency calculates basic efficiency from execution time.
func (g *GEPA) calculateRawEfficiency(context map[string]interface{}, baseline time.Duration) float64 {
	if context == nil {
		return 0.5
	}

	duration, ok := context["execution_time"].(time.Duration)
	if !ok {
		return 0.5
	}

	// Normalize against contextual baseline
	if duration <= baseline {
		return 1.0 - (float64(duration.Nanoseconds()) / float64(baseline.Nanoseconds()) * 0.3)
	} else {
		return 0.7 * (float64(baseline.Nanoseconds()) / float64(duration.Nanoseconds()))
	}
}

// applyContextAdjustments applies context-specific efficiency adjustments.
func (g *GEPA) applyContextAdjustments(rawEfficiency float64, perfContext *PerformanceContext) float64 {
	adjustedEfficiency := rawEfficiency

	// System load adjustment
	if perfContext.SystemLoad > 0.8 {
		adjustedEfficiency *= 1.1 // Bonus for performing well under high load
	} else if perfContext.SystemLoad < 0.2 {
		adjustedEfficiency *= 0.95 // Slight penalty for not utilizing available resources
	}

	// Memory usage adjustment
	if perfContext.MemoryUsage > 0.9 {
		adjustedEfficiency *= 0.9 // Penalty for high memory usage
	}

	// Concurrent tasks adjustment
	if perfContext.ConcurrentTasks > 5 {
		adjustedEfficiency *= 1.05 // Bonus for good performance under concurrency
	}

	// Input complexity adjustment
	if perfContext.InputComplexity > 0.8 {
		adjustedEfficiency *= 1.1 // Bonus for handling complex inputs efficiently
	}

	// Generation-based adjustment (favor efficiency improvements over time)
	generationBonus := float64(perfContext.Generation) * 0.01
	adjustedEfficiency *= (1.0 + generationBonus)

	// Time-of-day adjustment (some hours might be more resource-constrained)
	if perfContext.TimeOfDay >= 9 && perfContext.TimeOfDay <= 17 {
		adjustedEfficiency *= 0.98 // Slight penalty during business hours
	}

	// Clamp to [0, 1]
	if adjustedEfficiency > 1.0 {
		adjustedEfficiency = 1.0
	}
	if adjustedEfficiency < 0.0 {
		adjustedEfficiency = 0.0
	}

	return adjustedEfficiency
}

// updateContextMetrics updates performance metrics for the given context.
func (g *GEPA) updateContextMetrics(perfContext *PerformanceContext, outputs map[string]any, err error, efficiency float64) {
	if g.performanceLogger == nil {
		return
	}

	contextType := g.deriveContextType(perfContext)

	g.performanceLogger.mu.Lock()
	defer g.performanceLogger.mu.Unlock()

	if g.performanceLogger.contextMetrics == nil {
		g.performanceLogger.contextMetrics = make(map[string]*ContextPerformanceMetrics)
	}

	metrics, exists := g.performanceLogger.contextMetrics[contextType]
	if !exists {
		metrics = &ContextPerformanceMetrics{
			ContextType:      contextType,
			ExecutionCount:   0,
			SuccessRate:      0.0,
			AverageLatency:   0,
			QualityScore:     0.0,
			ErrorPatterns:    make(map[string]int),
			PerformanceTrend: make([]float64, 0),
			Metadata:         make(map[string]interface{}),
		}
		g.performanceLogger.contextMetrics[contextType] = metrics
	}

	// Update metrics
	metrics.ExecutionCount++

	// Update success rate
	if err == nil && outputs != nil && len(outputs) > 0 {
		metrics.SuccessRate = (metrics.SuccessRate*float64(metrics.ExecutionCount-1) + 1.0) / float64(metrics.ExecutionCount)
	} else {
		metrics.SuccessRate = (metrics.SuccessRate * float64(metrics.ExecutionCount-1)) / float64(metrics.ExecutionCount)

		// Track error patterns
		if err != nil {
			errorType := g.categorizeError(err)
			metrics.ErrorPatterns[errorType]++
		}
	}

	// Update latency (if available)
	if perfContext.CustomData != nil {
		if duration, ok := perfContext.CustomData["execution_time"].(time.Duration); ok {
			if metrics.ExecutionCount == 1 {
				metrics.AverageLatency = duration
			} else {
				// Exponential moving average
				alpha := 0.1
				metrics.AverageLatency = time.Duration(float64(metrics.AverageLatency)*(1-alpha) + float64(duration)*alpha)
			}
		}
	}

	// Update quality score
	if outputs != nil {
		qualityScore := g.assessOutputQuality(outputs)
		if metrics.ExecutionCount == 1 {
			metrics.QualityScore = qualityScore
		} else {
			alpha := 0.1
			metrics.QualityScore = metrics.QualityScore*(1-alpha) + qualityScore*alpha
		}
	}

	// Update performance trend
	metrics.PerformanceTrend = append(metrics.PerformanceTrend, efficiency)
	if len(metrics.PerformanceTrend) > 100 {
		// Keep only last 100 measurements
		metrics.PerformanceTrend = metrics.PerformanceTrend[1:]
	}

	metrics.LastUpdated = time.Now()
}

// Helper methods for context-aware performance tracking

// calculateInputComplexity estimates the complexity of input data.
func (g *GEPA) calculateInputComplexity(inputs map[string]any) float64 {
	if inputs == nil {
		return 0.0
	}

	complexity := 0.0

	for _, value := range inputs {
		if value == nil {
			continue
		}

		switch v := value.(type) {
		case string:
			// String complexity based on length and content variety
			length := float64(len(v))
			complexity += (length / 1000.0) * 0.3 // Length component

			// Content variety (unique characters)
			uniqueChars := make(map[rune]bool)
			for _, char := range v {
				uniqueChars[char] = true
			}
			complexity += (float64(len(uniqueChars)) / 256.0) * 0.2 // Character variety

		case []interface{}:
			// Array complexity based on size and nesting
			complexity += (float64(len(v)) / 100.0) * 0.4

		case map[string]interface{}:
			// Map complexity based on key count and nesting
			complexity += (float64(len(v)) / 50.0) * 0.5

		default:
			// Basic types get minimal complexity
			complexity += 0.1
		}
	}

	// Normalize complexity to [0, 1]
	if complexity > 1.0 {
		complexity = 1.0
	}

	return complexity
}

// isReservedContextKey checks if a context key is reserved for system use.
func (g *GEPA) isReservedContextKey(key string) bool {
	reservedKeys := []string{
		"execution_time", "module_name", "system_load", "memory_usage",
		"concurrent_tasks", "execution_phase",
	}

	for _, reserved := range reservedKeys {
		if key == reserved {
			return true
		}
	}

	return false
}

// calculateSelectionPressure calculates selection pressure from population diversity.
func (g *GEPA) calculateSelectionPressure(population *Population) float64 {
	if population == nil || len(population.Candidates) < 2 {
		return 0.5
	}

	// Calculate fitness variance as a proxy for selection pressure
	var fitnessSum, fitnessSquareSum float64
	count := 0

	for _, candidate := range population.Candidates {
		fitness := candidate.Fitness
		fitnessSum += fitness
		fitnessSquareSum += fitness * fitness
		count++
	}

	if count == 0 {
		return 0.5
	}

	mean := fitnessSum / float64(count)
	variance := (fitnessSquareSum / float64(count)) - (mean * mean)

	// High variance = low selection pressure, low variance = high selection pressure
	selectionPressure := 1.0 / (1.0 + variance)

	return selectionPressure
}

// findSimilarContexts finds contexts similar to the given context.
func (g *GEPA) findSimilarContexts(perfContext *PerformanceContext) map[string]float64 {
	if g.performanceLogger == nil {
		return make(map[string]float64)
	}

	g.performanceLogger.mu.RLock()
	defer g.performanceLogger.mu.RUnlock()

	similarContexts := make(map[string]float64)

	for contextType, metrics := range g.performanceLogger.contextMetrics {
		// Calculate similarity based on context features
		similarity := g.calculateContextSimilarity(perfContext, contextType, metrics)

		if similarity > 0.5 { // Only consider contexts with > 50% similarity
			similarContexts[contextType] = similarity
		}
	}

	return similarContexts
}

// calculateContextSimilarity calculates similarity between contexts.
func (g *GEPA) calculateContextSimilarity(perfContext *PerformanceContext, contextType string, metrics *ContextPerformanceMetrics) float64 {
	// Parse context type to extract features
	contextFeatures := g.parseContextType(contextType)

	similarity := 0.0
	factors := 0

	// Input size similarity
	if contextFeatures["input_size"] != nil {
		if refSize, ok := contextFeatures["input_size"].(int); ok {
			sizeDiff := float64(abs(perfContext.InputSize - refSize))
			sizeSim := 1.0 / (1.0 + sizeDiff/10.0)
			similarity += sizeSim
			factors++
		}
	}

	// Generation similarity
	if contextFeatures["generation"] != nil {
		if refGen, ok := contextFeatures["generation"].(int); ok {
			genDiff := float64(abs(perfContext.Generation - refGen))
			genSim := 1.0 / (1.0 + genDiff/5.0)
			similarity += genSim
			factors++
		}
	}

	// Time of day similarity
	if contextFeatures["time_of_day"] != nil {
		if refTime, ok := contextFeatures["time_of_day"].(int); ok {
			timeDiff := float64(abs(perfContext.TimeOfDay - refTime))
			timeSim := 1.0 / (1.0 + timeDiff/6.0)
			similarity += timeSim
			factors++
		}
	}

	// Module name similarity
	if contextFeatures["module_name"] != nil {
		if refModule, ok := contextFeatures["module_name"].(string); ok {
			if perfContext.ModuleName == refModule {
				similarity += 1.0
			}
			factors++
		}
	}

	if factors == 0 {
		return 0.0
	}

	return similarity / float64(factors)
}

// deriveContextType creates a context type string from performance context.
func (g *GEPA) deriveContextType(perfContext *PerformanceContext) string {
	// Create a context type based on key features
	features := []string{}

	if perfContext.ModuleName != "" {
		features = append(features, fmt.Sprintf("module:%s", perfContext.ModuleName))
	}

	features = append(features, fmt.Sprintf("input_size:%d", perfContext.InputSize))
	features = append(features, fmt.Sprintf("generation:%d", perfContext.Generation))
	features = append(features, fmt.Sprintf("time_of_day:%d", perfContext.TimeOfDay))

	if perfContext.ExecutionPhase != "" {
		features = append(features, fmt.Sprintf("phase:%s", perfContext.ExecutionPhase))
	}

	// Discretize continuous values
	if perfContext.InputComplexity > 0.7 {
		features = append(features, "complexity:high")
	} else if perfContext.InputComplexity > 0.3 {
		features = append(features, "complexity:medium")
	} else {
		features = append(features, "complexity:low")
	}

	return strings.Join(features, "_")
}

// parseContextType parses a context type string back into features.
func (g *GEPA) parseContextType(contextType string) map[string]interface{} {
	features := make(map[string]interface{})
	parts := strings.Split(contextType, "_")

	for _, part := range parts {
		if colonIdx := strings.Index(part, ":"); colonIdx != -1 {
			key := part[:colonIdx]
			value := part[colonIdx+1:]

			// Try to parse as int first
			if intVal, err := fmt.Sscanf(value, "%d", new(int)); err == nil && intVal == 1 {
				var parsedInt int
				if _, err := fmt.Sscanf(value, "%d", &parsedInt); err == nil {
					features[key] = parsedInt
				} else {
					features[key] = value
				}
			} else {
				features[key] = value
			}
		}
	}

	return features
}

// fallbackEfficiencyAssessment provides fallback when context tracking is unavailable.
func (g *GEPA) fallbackEfficiencyAssessment(context map[string]interface{}) float64 {
	if context == nil {
		return 0.5
	}

	duration, ok := context["execution_time"].(time.Duration)
	if !ok {
		return 0.5
	}

	// Simple baseline approach
	baseline := time.Second
	if duration <= baseline {
		return 1.0 - (float64(duration.Nanoseconds()) / float64(baseline.Nanoseconds()) * 0.5)
	} else {
		return 0.5 * (float64(baseline.Nanoseconds()) / float64(duration.Nanoseconds()))
	}
}

// abs returns the absolute value of an integer.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Pareto-based Selection Methods

// calculateParetoFronts calculates Pareto fronts for multi-objective optimization.
func (g *GEPA) calculateParetoFronts(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness) []*ParetoFront {
	if len(candidates) == 0 {
		return []*ParetoFront{}
	}

	fronts := make([]*ParetoFront, 0)
	remaining := make([]*GEPACandidate, len(candidates))
	copy(remaining, candidates)

	rank := 1

	for len(remaining) > 0 {
		currentFront := &ParetoFront{
			Candidates: make([]*GEPACandidate, 0),
			Rank:       rank,
		}

		// Find non-dominated candidates
		nonDominated := make([]*GEPACandidate, 0)

		for i, candidate := range remaining {
			isDominated := false

			for j, other := range remaining {
				if i == j {
					continue
				}

				if g.dominates(fitnessMap[other.ID], fitnessMap[candidate.ID]) {
					isDominated = true
					break
				}
			}

			if !isDominated {
				nonDominated = append(nonDominated, candidate)
			}
		}

		// Add non-dominated candidates to current front
		currentFront.Candidates = nonDominated
		currentFront.Size = len(nonDominated)
		fronts = append(fronts, currentFront)

		// Remove non-dominated candidates from remaining
		newRemaining := make([]*GEPACandidate, 0)
		for _, candidate := range remaining {
			isDominated := false
			for _, nonDom := range nonDominated {
				if candidate.ID == nonDom.ID {
					isDominated = true
					break
				}
			}
			if !isDominated {
				newRemaining = append(newRemaining, candidate)
			}
		}
		remaining = newRemaining
		rank++
	}

	return fronts
}

// dominates checks if fitness1 dominates fitness2 in multi-objective space.
func (g *GEPA) dominates(fitness1, fitness2 *MultiObjectiveFitness) bool {
	if fitness1 == nil || fitness2 == nil {
		return false
	}

	// Extract objective values
	objectives1 := []float64{
		fitness1.SuccessRate,
		fitness1.OutputQuality,
		fitness1.Efficiency,
		fitness1.Robustness,
		fitness1.Generalization,
		fitness1.Diversity,
		fitness1.Innovation,
	}

	objectives2 := []float64{
		fitness2.SuccessRate,
		fitness2.OutputQuality,
		fitness2.Efficiency,
		fitness2.Robustness,
		fitness2.Generalization,
		fitness2.Diversity,
		fitness2.Innovation,
	}

	// Check if fitness1 dominates fitness2
	// fitness1 dominates if it's >= in all objectives and > in at least one
	allGreaterOrEqual := true
	atLeastOneGreater := false

	for i := 0; i < len(objectives1); i++ {
		if objectives1[i] < objectives2[i] {
			allGreaterOrEqual = false
			break
		}
		if objectives1[i] > objectives2[i] {
			atLeastOneGreater = true
		}
	}

	return allGreaterOrEqual && atLeastOneGreater
}

// selectWithParetoRanking selects candidates using Pareto ranking and crowding distance.
func (g *GEPA) selectWithParetoRanking(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness, selectionSize int) []*GEPACandidate {
	if len(candidates) <= selectionSize {
		return candidates
	}

	// Calculate Pareto fronts
	fronts := g.calculateParetoFronts(candidates, fitnessMap)

	selected := make([]*GEPACandidate, 0, selectionSize)

	// Add candidates from fronts in order
	for _, front := range fronts {
		if len(selected)+len(front.Candidates) <= selectionSize {
			// Add entire front
			selected = append(selected, front.Candidates...)
		} else {
			// Need to select subset based on crowding distance
			remaining := selectionSize - len(selected)
			crowdingDistances := g.calculateCrowdingDistance(front.Candidates, fitnessMap)

			// Sort by crowding distance (descending)
			frontCandidates := make([]*GEPACandidate, len(front.Candidates))
			copy(frontCandidates, front.Candidates)

			for i := 0; i < len(frontCandidates)-1; i++ {
				for j := 0; j < len(frontCandidates)-1-i; j++ {
					if crowdingDistances[frontCandidates[j].ID] < crowdingDistances[frontCandidates[j+1].ID] {
						frontCandidates[j], frontCandidates[j+1] = frontCandidates[j+1], frontCandidates[j]
					}
				}
			}

			// Add candidates with highest crowding distance
			selected = append(selected, frontCandidates[:remaining]...)
			break
		}

		if len(selected) >= selectionSize {
			break
		}
	}

	return selected
}

// calculateCrowdingDistance calculates crowding distance for diversity preservation.
func (g *GEPA) calculateCrowdingDistance(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness) map[string]float64 {
	distances := make(map[string]float64)

	if len(candidates) <= 2 {
		// Assign infinite distance to boundary points
		for _, candidate := range candidates {
			distances[candidate.ID] = 1e9
		}
		return distances
	}

	// Initialize distances to 0
	for _, candidate := range candidates {
		distances[candidate.ID] = 0.0
	}

	// Objective names for easier processing
	objectiveNames := []string{
		"SuccessRate", "OutputQuality", "Efficiency", "Robustness",
		"Generalization", "Diversity", "Innovation",
	}

	// Calculate crowding distance for each objective
	for _, objectiveName := range objectiveNames {
		// Sort candidates by this objective
		sortedCandidates := make([]*GEPACandidate, len(candidates))
		copy(sortedCandidates, candidates)

		// Sort by objective value
		for i := 0; i < len(sortedCandidates)-1; i++ {
			for j := 0; j < len(sortedCandidates)-1-i; j++ {
				val1 := g.getObjectiveValue(fitnessMap[sortedCandidates[j].ID], objectiveName)
				val2 := g.getObjectiveValue(fitnessMap[sortedCandidates[j+1].ID], objectiveName)
				if val1 > val2 {
					sortedCandidates[j], sortedCandidates[j+1] = sortedCandidates[j+1], sortedCandidates[j]
				}
			}
		}

		// Boundary points get infinite distance
		distances[sortedCandidates[0].ID] = 1e9
		distances[sortedCandidates[len(sortedCandidates)-1].ID] = 1e9

		// Calculate range for normalization
		minVal := g.getObjectiveValue(fitnessMap[sortedCandidates[0].ID], objectiveName)
		maxVal := g.getObjectiveValue(fitnessMap[sortedCandidates[len(sortedCandidates)-1].ID], objectiveName)
		objRange := maxVal - minVal

		if objRange > 0 {
			// Calculate distance for intermediate points
			for i := 1; i < len(sortedCandidates)-1; i++ {
				prevVal := g.getObjectiveValue(fitnessMap[sortedCandidates[i-1].ID], objectiveName)
				nextVal := g.getObjectiveValue(fitnessMap[sortedCandidates[i+1].ID], objectiveName)

				distances[sortedCandidates[i].ID] += (nextVal - prevVal) / objRange
			}
		}
	}

	return distances
}

// buildMultiObjectiveFitnessMap creates a fitness map from candidate metrics for Pareto operations.
func (g *GEPA) buildMultiObjectiveFitnessMap(candidates []*GEPACandidate) map[string]*MultiObjectiveFitness {
	fitnessMap := make(map[string]*MultiObjectiveFitness)

	for _, candidate := range candidates {
		// Try to get multi-objective fitness from candidate metrics
		if metrics, exists := g.state.CandidateMetrics[candidate.ID]; exists {
			if fitness, ok := metrics.Metadata["multi_objective_fitness"].(*MultiObjectiveFitness); ok {
				fitnessMap[candidate.ID] = fitness
			}
		}

		// Fallback: create fitness from single objective score if multi-objective not available
		if _, exists := fitnessMap[candidate.ID]; !exists {
			// Convert single fitness to multi-objective fitness for compatibility
			fallbackFitness := &MultiObjectiveFitness{
				SuccessRate:    candidate.Fitness,
				OutputQuality:  candidate.Fitness,
				Efficiency:     0.5, // Default neutral value
				Robustness:     0.5, // Default neutral value
				Generalization: 0.5, // Default neutral value
				Diversity:      0.5, // Default neutral value
				Innovation:     0.5, // Default neutral value
				WeightedScore:  candidate.Fitness,
			}
			fitnessMap[candidate.ID] = fallbackFitness
		}
	}

	return fitnessMap
}

// getObjectiveValue extracts objective value by name from fitness struct.
func (g *GEPA) getObjectiveValue(fitness *MultiObjectiveFitness, objectiveName string) float64 {
	if fitness == nil {
		return 0.0
	}

	switch objectiveName {
	case "SuccessRate":
		return fitness.SuccessRate
	case "OutputQuality":
		return fitness.OutputQuality
	case "Efficiency":
		return fitness.Efficiency
	case "Robustness":
		return fitness.Robustness
	case "Generalization":
		return fitness.Generalization
	case "Diversity":
		return fitness.Diversity
	case "Innovation":
		return fitness.Innovation
	default:
		return 0.0
	}
}

// adaptiveWeightedSelection provides fallback selection when Pareto optimization is not suitable.
func (g *GEPA) adaptiveWeightedSelection(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness, selectionSize int, generation int) []*GEPACandidate {
	if len(candidates) <= selectionSize {
		return candidates
	}

	// Adapt weights based on generation and population characteristics
	weights := g.calculateAdaptiveWeights(generation, candidates, fitnessMap)

	// Calculate weighted scores
	weightedScores := make(map[string]float64)
	for _, candidate := range candidates {
		fitness := fitnessMap[candidate.ID]
		if fitness != nil {
			score :=
				fitness.SuccessRate*weights["success"] +
					fitness.OutputQuality*weights["quality"] +
					fitness.Efficiency*weights["efficiency"] +
					fitness.Robustness*weights["robustness"] +
					fitness.Generalization*weights["generalization"] +
					fitness.Diversity*weights["diversity"] +
					fitness.Innovation*weights["innovation"]

			weightedScores[candidate.ID] = score
		}
	}

	// Sort candidates by weighted score
	sortedCandidates := make([]*GEPACandidate, len(candidates))
	copy(sortedCandidates, candidates)

	for i := 0; i < len(sortedCandidates)-1; i++ {
		for j := 0; j < len(sortedCandidates)-1-i; j++ {
			if weightedScores[sortedCandidates[j].ID] < weightedScores[sortedCandidates[j+1].ID] {
				sortedCandidates[j], sortedCandidates[j+1] = sortedCandidates[j+1], sortedCandidates[j]
			}
		}
	}

	return sortedCandidates[:selectionSize]
}

// paretoBasedSelection implements pure Pareto-based selection using non-dominated sorting and crowding distance.
func (g *GEPA) paretoBasedSelection(population *Population, selectionSize int) []*GEPACandidate {
	logger := logging.GetLogger()

	if len(population.Candidates) <= selectionSize {
		logger.Debug(context.Background(), "Population size (%d) <= selection size (%d), returning all candidates",
			len(population.Candidates), selectionSize)
		return population.Candidates
	}

	// Use stored multi-objective fitness map from current evaluation
	fitnessMap := g.getCurrentMultiObjectiveFitnessMap()

	// Fallback to building from metrics if not available
	if len(fitnessMap) == 0 {
		logger.Debug(context.Background(), "No stored multi-objective fitness map, building from candidate metrics")
		fitnessMap = g.buildMultiObjectiveFitnessMap(population.Candidates)
	}

	// Check if we have valid multi-objective fitness data
	hasValidFitness := false
	for _, fitness := range fitnessMap {
		if fitness != nil {
			hasValidFitness = true
			break
		}
	}

	if !hasValidFitness {
		logger.Warn(context.Background(), "No valid multi-objective fitness data found, fallback to tournament selection")
		return g.tournamentSelection(population, selectionSize)
	}

	logger.Debug(context.Background(), "Performing Pareto-based selection: population=%d, target=%d",
		len(population.Candidates), selectionSize)

	// Use existing Pareto ranking method
	selected := g.selectWithParetoRanking(population.Candidates, fitnessMap, selectionSize)

	// Log detailed selection statistics for debugging
	g.logParetoSelectionStats(context.Background(), population.Candidates, fitnessMap, selected, "Pure Pareto")

	logger.Debug(context.Background(), "Pareto selection completed: selected=%d candidates", len(selected))
	return selected
}

// adaptiveParetoSelection combines Pareto selection with adaptive strategies based on population diversity and generation.
func (g *GEPA) adaptiveParetoSelection(population *Population, selectionSize int) []*GEPACandidate {
	logger := logging.GetLogger()

	if len(population.Candidates) <= selectionSize {
		return population.Candidates
	}

	// Use stored multi-objective fitness map from current evaluation
	fitnessMap := g.getCurrentMultiObjectiveFitnessMap()

	// Fallback to building from metrics if not available
	if len(fitnessMap) == 0 {
		logger.Debug(context.Background(), "No stored multi-objective fitness map, building from candidate metrics")
		fitnessMap = g.buildMultiObjectiveFitnessMap(population.Candidates)
	}

	// Assess population diversity and convergence state
	diversityScore := g.assessPopulationDiversity(population.Candidates, fitnessMap)
	convergenceProgress := float64(g.state.CurrentGeneration) / float64(g.config.MaxGenerations)

	logger.Debug(context.Background(), "Adaptive Pareto selection: diversity=%.3f, progress=%.3f",
		diversityScore, convergenceProgress)

	// Choose selection strategy based on population state
	if diversityScore < 0.3 && convergenceProgress > 0.5 {
		// Low diversity, late stage: use weighted selection to encourage exploration
		logger.Debug(context.Background(), "Low diversity detected, using adaptive weighted selection")
		selected := g.adaptiveWeightedSelection(population.Candidates, fitnessMap, selectionSize, g.state.CurrentGeneration)
		g.logParetoSelectionStats(context.Background(), population.Candidates, fitnessMap, selected, "Adaptive Pareto - Weighted Fallback")
		return selected
	} else if convergenceProgress < 0.3 {
		// Early stage: pure Pareto with emphasis on diversity
		logger.Debug(context.Background(), "Early stage, using diversity-enhanced Pareto selection")
		selected := g.diversityEnhancedParetoSelection(population.Candidates, fitnessMap, selectionSize)
		return selected
	} else {
		// Standard Pareto selection
		logger.Debug(context.Background(), "Using standard Pareto selection")
		selected := g.selectWithParetoRanking(population.Candidates, fitnessMap, selectionSize)
		g.logParetoSelectionStats(context.Background(), population.Candidates, fitnessMap, selected, "Adaptive Pareto - Standard")
		return selected
	}
}

// diversityEnhancedParetoSelection performs Pareto selection with additional diversity pressure.
func (g *GEPA) diversityEnhancedParetoSelection(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness, selectionSize int) []*GEPACandidate {
	if len(candidates) <= selectionSize {
		return candidates
	}

	// First, get Pareto fronts
	fronts := g.calculateParetoFronts(candidates, fitnessMap)
	if len(fronts) == 0 {
		return candidates[:selectionSize]
	}

	selected := make([]*GEPACandidate, 0, selectionSize)

	// Add candidates from fronts with enhanced diversity consideration
	for _, front := range fronts {
		if len(selected)+len(front.Candidates) <= selectionSize {
			// Add entire front
			selected = append(selected, front.Candidates...)
		} else {
			// Need to select subset with diversity enhancement
			remaining := selectionSize - len(selected)
			crowdingDistances := g.calculateCrowdingDistance(front.Candidates, fitnessMap)

			// Enhance crowding distances with instruction diversity
			for _, candidate := range front.Candidates {
				instructionDiversity := g.calculateInstructionDiversityScore(candidate, selected)
				crowdingDistances[candidate.ID] += instructionDiversity * 0.5 // Boost diversity
			}

			// Sort by enhanced crowding distance
			frontCandidates := make([]*GEPACandidate, len(front.Candidates))
			copy(frontCandidates, front.Candidates)

			for i := 0; i < len(frontCandidates)-1; i++ {
				for j := 0; j < len(frontCandidates)-1-i; j++ {
					if crowdingDistances[frontCandidates[j].ID] < crowdingDistances[frontCandidates[j+1].ID] {
						frontCandidates[j], frontCandidates[j+1] = frontCandidates[j+1], frontCandidates[j]
					}
				}
			}

			selected = append(selected, frontCandidates[:remaining]...)
			break
		}

		if len(selected) >= selectionSize {
			break
		}
	}

	// Log diversity-enhanced selection statistics
	g.logParetoSelectionStats(context.Background(), candidates, fitnessMap, selected, "Diversity Enhanced Pareto")

	return selected
}

// assessPopulationDiversity calculates diversity score of current population.
func (g *GEPA) assessPopulationDiversity(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness) float64 {
	if len(candidates) <= 1 {
		return 0.0
	}

	// Calculate diversity in objective space
	objectiveDiversity := g.calculateObjectiveSpaceDiversity(fitnessMap)

	// Calculate diversity in instruction space
	instructionDiversity := g.calculateInstructionSpaceDiversity(candidates)

	// Combine both measures
	return (objectiveDiversity + instructionDiversity) / 2.0
}

// calculateObjectiveSpaceDiversity measures diversity in the multi-objective fitness space.
func (g *GEPA) calculateObjectiveSpaceDiversity(fitnessMap map[string]*MultiObjectiveFitness) float64 {
	if len(fitnessMap) <= 1 {
		return 0.0
	}

	objectives := [][]float64{}
	for _, fitness := range fitnessMap {
		if fitness != nil {
			objectives = append(objectives, []float64{
				fitness.SuccessRate,
				fitness.OutputQuality,
				fitness.Efficiency,
				fitness.Robustness,
				fitness.Generalization,
				fitness.Diversity,
				fitness.Innovation,
			})
		}
	}

	if len(objectives) <= 1 {
		return 0.0
	}

	// Calculate average pairwise distance in objective space
	totalDistance := 0.0
	pairCount := 0

	for i := 0; i < len(objectives); i++ {
		for j := i + 1; j < len(objectives); j++ {
			distance := 0.0
			for k := 0; k < len(objectives[i]); k++ {
				diff := objectives[i][k] - objectives[j][k]
				distance += diff * diff
			}
			totalDistance += distance
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0.0
	}

	avgDistance := totalDistance / float64(pairCount)
	// Normalize to [0, 1] range (sqrt of sum of squares can be at most sqrt(7) for 7 objectives)
	return avgDistance / 7.0
}

// calculateInstructionSpaceDiversity measures diversity in instruction content.
func (g *GEPA) calculateInstructionSpaceDiversity(candidates []*GEPACandidate) float64 {
	if len(candidates) <= 1 {
		return 0.0
	}

	totalSimilarity := 0.0
	pairCount := 0

	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			similarity := g.calculateInstructionSimilarity(candidates[i].Instruction, candidates[j].Instruction)
			totalSimilarity += similarity
			pairCount++
		}
	}

	if pairCount == 0 {
		return 0.0
	}

	avgSimilarity := totalSimilarity / float64(pairCount)
	return 1.0 - avgSimilarity // Convert similarity to diversity
}

// calculateInstructionDiversityScore calculates how diverse a candidate's instruction is compared to a set.
func (g *GEPA) calculateInstructionDiversityScore(candidate *GEPACandidate, compareSet []*GEPACandidate) float64 {
	if len(compareSet) == 0 {
		return 1.0 // Maximum diversity if nothing to compare to
	}

	totalSimilarity := 0.0
	for _, other := range compareSet {
		similarity := g.calculateInstructionSimilarity(candidate.Instruction, other.Instruction)
		totalSimilarity += similarity
	}

	avgSimilarity := totalSimilarity / float64(len(compareSet))
	return 1.0 - avgSimilarity // Convert to diversity score
}

// logParetoSelectionStats logs detailed statistics about Pareto-based selection for debugging.
func (g *GEPA) logParetoSelectionStats(ctx context.Context, candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness, selected []*GEPACandidate, selectionType string) {
	logger := logging.GetLogger()

	if len(candidates) == 0 {
		return
	}

	// Calculate Pareto fronts for statistics
	fronts := g.calculateParetoFronts(candidates, fitnessMap)

	// Log basic selection statistics
	logger.Info(ctx, "Pareto Selection Stats [%s]: candidates=%d, selected=%d, fronts=%d",
		selectionType, len(candidates), len(selected), len(fronts))

	// Log front distribution
	for i, front := range fronts {
		logger.Debug(ctx, "Pareto Front %d: size=%d, rank=%d", i+1, front.Size, front.Rank)
	}

	// Calculate and log objective statistics
	if len(fitnessMap) > 0 {
		objStats := g.calculateObjectiveStatistics(fitnessMap)
		logger.Debug(ctx, "Objective Statistics: %+v", objStats)
	}

	// Log diversity metrics
	diversityScore := g.assessPopulationDiversity(candidates, fitnessMap)
	selectedDiversity := g.assessPopulationDiversity(selected, g.buildMultiObjectiveFitnessMap(selected))

	logger.Debug(ctx, "Diversity: population=%.3f, selected=%.3f", diversityScore, selectedDiversity)

	// Log selection pressure metrics
	convergenceProgress := float64(g.state.CurrentGeneration) / float64(g.config.MaxGenerations)
	logger.Debug(ctx, "Selection Context: generation=%d/%d, progress=%.3f",
		g.state.CurrentGeneration, g.config.MaxGenerations, convergenceProgress)
}

// calculateObjectiveStatistics computes statistics for each objective across the population.
func (g *GEPA) calculateObjectiveStatistics(fitnessMap map[string]*MultiObjectiveFitness) map[string]ObjectiveStats {
	if len(fitnessMap) == 0 {
		return make(map[string]ObjectiveStats)
	}

	stats := make(map[string]ObjectiveStats)
	objectiveNames := []string{"SuccessRate", "OutputQuality", "Efficiency", "Robustness", "Generalization", "Diversity", "Innovation"}

	for _, objName := range objectiveNames {
		values := make([]float64, 0, len(fitnessMap))
		for _, fitness := range fitnessMap {
			if fitness != nil {
				values = append(values, g.getObjectiveValue(fitness, objName))
			}
		}

		if len(values) > 0 {
			stats[objName] = ObjectiveStats{
				Mean:   g.calculateMean(values),
				StdDev: g.calculateStdDev(values),
				Min:    g.calculateMin(values),
				Max:    g.calculateMax(values),
				Range:  g.calculateMax(values) - g.calculateMin(values),
			}
		}
	}

	return stats
}

// ObjectiveStats represents statistical information for a single objective.
type ObjectiveStats struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Range  float64 `json:"range"`
}

// Helper statistical functions.
func (g *GEPA) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (g *GEPA) calculateStdDev(values []float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}
	mean := g.calculateMean(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	return sumSquares / float64(len(values)-1)
}

func (g *GEPA) calculateMin(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func (g *GEPA) calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// calculateAdaptiveWeights calculates adaptive weights based on generation and population state.
func (g *GEPA) calculateAdaptiveWeights(generation int, candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness) map[string]float64 {
	weights := map[string]float64{
		"success":        0.25,
		"quality":        0.20,
		"efficiency":     0.15,
		"robustness":     0.15,
		"generalization": 0.15,
		"diversity":      0.05,
		"innovation":     0.05,
	}

	maxGenerations := float64(g.config.MaxGenerations)
	progress := float64(generation) / maxGenerations

	// Early generations: prioritize exploration (diversity + innovation)
	if progress < 0.3 {
		weights["diversity"] = 0.15
		weights["innovation"] = 0.15
		weights["success"] = 0.20
		weights["quality"] = 0.15
		weights["efficiency"] = 0.10
		weights["robustness"] = 0.15
		weights["generalization"] = 0.10
	} else if progress < 0.7 {
		// Middle generations: balanced approach
		weights["success"] = 0.25
		weights["quality"] = 0.20
		weights["efficiency"] = 0.15
		weights["robustness"] = 0.15
		weights["generalization"] = 0.15
		weights["diversity"] = 0.05
		weights["innovation"] = 0.05
	} else {
		// Late generations: prioritize exploitation (success + quality)
		weights["success"] = 0.35
		weights["quality"] = 0.25
		weights["efficiency"] = 0.15
		weights["robustness"] = 0.15
		weights["generalization"] = 0.10
		weights["diversity"] = 0.00
		weights["innovation"] = 0.00
	}

	// Adjust based on population diversity
	if len(candidates) > 1 {
		avgDiversity := 0.0
		count := 0

		for _, candidate := range candidates {
			if fitness := fitnessMap[candidate.ID]; fitness != nil {
				avgDiversity += fitness.Diversity
				count++
			}
		}

		if count > 0 {
			avgDiversity /= float64(count)

			// If diversity is low, increase diversity weight
			if avgDiversity < 0.3 {
				diversityBoost := 0.1
				weights["diversity"] += diversityBoost
				// Reduce other weights proportionally
				reduction := diversityBoost / 6.0
				weights["success"] -= reduction
				weights["quality"] -= reduction
				weights["efficiency"] -= reduction
				weights["robustness"] -= reduction
				weights["generalization"] -= reduction
				weights["innovation"] -= reduction
			}
		}
	}

	return weights
}

// NewGEPA creates a new GEPA optimizer with the given configuration.
func NewGEPA(config *GEPAConfig) (*GEPA, error) {
	if config == nil {
		config = DefaultGEPAConfig()
	}

	// Merge with defaults for any missing fields
	defaults := DefaultGEPAConfig()
	if config.ConcurrencyLevel <= 0 {
		config.ConcurrencyLevel = defaults.ConcurrencyLevel
	}
	if config.EvaluationBatchSize <= 0 {
		config.EvaluationBatchSize = defaults.EvaluationBatchSize
	}
	if config.TournamentSize <= 0 {
		config.TournamentSize = defaults.TournamentSize
	}
	if config.SelectionStrategy == "" {
		config.SelectionStrategy = defaults.SelectionStrategy
	}

	// Initialize LLMs
	generationLLM := core.GetDefaultLLM()
	if generationLLM == nil {
		return nil, fmt.Errorf("no default LLM available for generation")
	}

	reflectionLLM := core.GetTeacherLLM()
	if reflectionLLM == nil {
		reflectionLLM = generationLLM
	}

	// Initialize random number generator
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	gepa := &GEPA{
		config:            config,
		state:             NewGEPAState(),
		generationLLM:     generationLLM,
		reflectionLLM:     reflectionLLM,
		interceptorChain:  core.NewInterceptorChain(),
		performanceLogger: NewPerformanceLogger(),
		rng:               rng,
	}

	// Set up interceptors
	gepa.setupInterceptors()

	return gepa, nil
}

// SetProgressReporter sets a progress reporter for the optimizer.
func (g *GEPA) SetProgressReporter(reporter core.ProgressReporter) {
	g.progressReporter = reporter
}

// GetOptimizationState returns the current optimization state.
func (g *GEPA) GetOptimizationState() *GEPAState {
	return g.state
}

// Compile implements the core.Optimizer interface for GEPA.
func (g *GEPA) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	ctx, span := core.StartSpan(ctx, "GEPA.Compile")
	defer core.EndSpan(ctx)
	_ = span // Suppress unused variable warning

	logger := logging.GetLogger()
	logger.Info(ctx, "Starting GEPA optimization with population_size=%d, max_generations=%d",
		g.config.PopulationSize,
		g.config.MaxGenerations)

	// Initialize GEPA state in context
	ctx = g.withGEPAState(ctx)

	// Install interceptors on the program
	optimizedProgram := g.installInterceptors(program)

	// Initialize population
	err := g.initializePopulation(ctx, optimizedProgram)
	if err != nil {
		return program, fmt.Errorf("failed to initialize population: %w", err)
	}

	// Main evolutionary loop
	for generation := 0; generation < g.config.MaxGenerations; generation++ {
		g.state.CurrentGeneration = generation

		logger.Info(ctx, "Starting generation %d", generation)

		// Evaluate current population and get multi-objective fitness map
		multiObjFitnessMap, err := g.evaluatePopulation(ctx, optimizedProgram, dataset, metric)
		if err != nil {
			return program, fmt.Errorf("evaluation failed at generation %d: %w", generation, err)
		}

		// Store multi-objective fitness map for this generation
		g.setCurrentMultiObjectiveFitnessMap(multiObjFitnessMap)

		// Periodic reflection
		if generation%g.config.ReflectionFreq == 0 && generation > 0 {
			err = g.performReflection(ctx, generation)
			if err != nil {
				logger.Error(ctx, "Reflection failed at generation %d: %v", generation, err)
			}
		}

		// Check convergence
		if g.hasConverged() {
			logger.Info(ctx, "Convergence achieved at generation %d", generation)
			break
		}

		// Create next generation (skip for last generation)
		if generation < g.config.MaxGenerations-1 {
			err = g.evolvePopulation(ctx)
			if err != nil {
				return program, fmt.Errorf("evolution failed at generation %d: %w", generation, err)
			}
		}

		// Report progress
		if g.progressReporter != nil {
			g.progressReporter.Report("GEPA Evolution", generation+1, g.config.MaxGenerations)
		}
	}

	// Apply best candidate to program
	finalProgram := g.applyBestCandidate(optimizedProgram)

	// Log optimization results
	g.logOptimizationResults(ctx)

	return finalProgram, nil
}

// withGEPAState adds GEPA state to the context.
func (g *GEPA) withGEPAState(ctx context.Context) context.Context {
	return context.WithValue(ctx, gepaStateKey, g.state)
}

// GetGEPAState retrieves GEPA state from context.
func GetGEPAState(ctx context.Context) *GEPAState {
	if state, ok := ctx.Value(gepaStateKey).(*GEPAState); ok {
		return state
	}
	return nil
}

// setupInterceptors configures the interceptor chain for GEPA.
func (g *GEPA) setupInterceptors() {
	g.interceptorChain.
		AddModuleInterceptor(g.gepaExecutionTracker).
		AddModuleInterceptor(g.gepaPerformanceCollector).
		AddModuleInterceptor(g.gepaReflectionLogger)
}

// installInterceptors applies GEPA interceptors to the program.
func (g *GEPA) installInterceptors(program core.Program) core.Program {
	// Clone the program to avoid modifying the original
	clonedProgram := program.Clone()

	// Install GEPA tracking interceptors using the interceptor chain
	chain := core.NewInterceptorChain().
		AddModuleInterceptor(g.gepaExecutionTracker).
		AddModuleInterceptor(g.gepaPerformanceCollector).
		AddModuleInterceptor(g.gepaReflectionLogger)

	// Apply interceptors to all modules that support them
	for name, module := range clonedProgram.Modules {
		if interceptable, ok := module.(interface{ SetInterceptors(*core.InterceptorChain) }); ok {
			interceptable.SetInterceptors(chain)
		} else {
			// Log warning if module doesn't support interceptors
			logging.GetLogger().Warn(context.Background(),
				"Module %s does not support interceptors, GEPA tracking will be limited", name)
		}
	}

	return clonedProgram
}

// gepaExecutionTracker tracks module executions for analysis.
func (g *GEPA) gepaExecutionTracker(ctx context.Context, inputs map[string]any,
	info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {

	// Extract GEPA state from context
	gepaState := GetGEPAState(ctx)
	if gepaState == nil {
		return handler(ctx, inputs, opts...)
	}

	// Track execution
	start := time.Now()
	outputs, err := handler(ctx, inputs, opts...)
	duration := time.Since(start)

	// Store execution trace
	trace := &ExecutionTrace{
		CandidateID: g.getCurrentCandidateID(ctx),
		ModuleName:  info.ModuleName,
		Inputs:      inputs,
		Outputs:     outputs,
		Error:       err,
		Duration:    duration,
		Success:     (err == nil),
		Timestamp:   start,
		ContextData: make(map[string]interface{}),
	}

	gepaState.AddTrace(trace)
	return outputs, err
}

// gepaPerformanceCollector collects performance metrics for fitness calculation.
func (g *GEPA) gepaPerformanceCollector(ctx context.Context, inputs map[string]any,
	info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {

	// Real-time performance tracking start
	start := time.Now()
	candidateID := g.getCurrentCandidateID(ctx)

	// Execute with real-time monitoring
	outputs, err := handler(ctx, inputs, opts...)
	duration := time.Since(start)

	// Create enhanced context for multi-objective fitness calculation
	perfContext := map[string]interface{}{
		"execution_time":   duration,
		"module_name":      info.ModuleName,
		"system_load":      g.getCurrentSystemLoad(),
		"memory_usage":     g.getCurrentMemoryUsage(),
		"concurrent_tasks": g.getCurrentConcurrentTasks(),
		"execution_phase":  g.getCurrentExecutionPhase(ctx),
		"generation":       g.state.CurrentGeneration,
	}

	// Calculate multi-objective fitness with context awareness
	fitness := g.calculateFitness(inputs, outputs, err)
	multiObjFitness := g.calculateMultiObjectiveFitness(candidateID, inputs, outputs, err, perfContext)

	// Real-time fitness broadcasting for live monitoring
	g.broadcastFitnessUpdate(candidateID, fitness, multiObjFitness, perfContext)

	// Store fitness information in context or state
	if gepaState := GetGEPAState(ctx); gepaState != nil {
		if metrics, exists := gepaState.CandidateMetrics[candidateID]; exists {
			// Update running fitness averages
			totalEvals := float64(metrics.TotalEvaluations)
			metrics.AverageFitness = (metrics.AverageFitness*totalEvals + fitness) / (totalEvals + 1)
			if fitness > metrics.BestFitness {
				metrics.BestFitness = fitness
			}

			// Store multi-objective fitness in metadata
			if metrics.Metadata == nil {
				metrics.Metadata = make(map[string]interface{})
			}
			metrics.Metadata["multi_objective_fitness"] = multiObjFitness
			metrics.Metadata["last_execution_context"] = perfContext

			// Log to performance logger with enhanced metrics
			g.performanceLogger.LogCandidateMetrics(candidateID, metrics)
		}

		// Update real-time population statistics
		g.updateRealTimePopulationStats(candidateID, fitness, multiObjFitness)
	}

	return outputs, err
}

// gepaReflectionLogger logs data needed for reflection analysis.
func (g *GEPA) gepaReflectionLogger(ctx context.Context, inputs map[string]any,
	info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {

	// Extract GEPA state from context
	gepaState := GetGEPAState(ctx)
	if gepaState == nil {
		return handler(ctx, inputs, opts...)
	}

	// Execute the handler
	start := time.Now()
	outputs, err := handler(ctx, inputs, opts...)
	duration := time.Since(start)

	// Log reflection data
	candidateID := g.getCurrentCandidateID(ctx)
	reflectionData := map[string]interface{}{
		"module_name":    info.ModuleName,
		"execution_time": duration,
		"success":        err == nil,
		"input_types":    g.getInputTypes(inputs),
		"output_types":   g.getOutputTypes(outputs),
		"error_type":     g.getErrorType(err),
	}

	// Store in candidate metrics for later reflection
	if metrics, exists := gepaState.CandidateMetrics[candidateID]; exists {
		if metrics.Metadata == nil {
			metrics.Metadata = make(map[string]interface{})
		}
		if _, exists := metrics.Metadata["reflection_data"]; !exists {
			metrics.Metadata["reflection_data"] = make([]map[string]interface{}, 0)
		}
		reflectionDataSlice := metrics.Metadata["reflection_data"].([]map[string]interface{})
		metrics.Metadata["reflection_data"] = append(reflectionDataSlice, reflectionData)
	}

	return outputs, err
}

// getCurrentCandidateID extracts the current candidate ID from context.
func (g *GEPA) getCurrentCandidateID(ctx context.Context) string {
	if candidateID, ok := ctx.Value(candidateIDKey).(string); ok {
		return candidateID
	}
	return "unknown"
}

// Real-time tracking support methods

// getCurrentSystemLoad gets current system load for context-aware tracking.
func (g *GEPA) getCurrentSystemLoad() float64 {
	// In a real implementation, this would get actual system metrics
	// For now, return a simulated value
	return 0.5 // TODO: Replace with actual system load monitoring
}

// getCurrentMemoryUsage gets current memory usage for context-aware tracking.
func (g *GEPA) getCurrentMemoryUsage() float64 {
	// In a real implementation, this would get actual memory metrics
	// For now, return a simulated value
	return 0.3 // TODO: Replace with actual memory usage monitoring
}

// getCurrentConcurrentTasks gets current number of concurrent tasks.
func (g *GEPA) getCurrentConcurrentTasks() int {
	// In a real implementation, this would count active goroutines or tasks
	// For now, return a simulated value
	return 1 // TODO: Replace with actual concurrency monitoring
}

// getCurrentExecutionPhase determines the current phase of execution.
func (g *GEPA) getCurrentExecutionPhase(ctx context.Context) string {
	if phase, ok := ctx.Value(executionPhaseKey).(string); ok {
		return phase
	}

	// Infer phase from generation state
	if g.state.CurrentGeneration == 0 {
		return "initialization"
	} else if g.state.CurrentGeneration%g.config.ReflectionFreq == 0 {
		return "reflection"
	} else {
		return "evolution"
	}
}

// broadcastFitnessUpdate broadcasts fitness updates for real-time monitoring.
func (g *GEPA) broadcastFitnessUpdate(candidateID string, fitness float64, multiObjFitness *MultiObjectiveFitness, perfContext map[string]interface{}) {
	// Real-time fitness update notification
	_ = map[string]interface{}{
		"candidate_id":            candidateID,
		"fitness":                 fitness,
		"multi_objective_fitness": multiObjFitness,
		"context":                 perfContext,
		"timestamp":               time.Now(),
		"generation":              g.state.CurrentGeneration,
	}

	// In a real implementation, this would broadcast to monitoring systems
	// For now, log the update
	logger := logging.GetLogger()
	logger.Debug(context.Background(),
		"Real-time fitness update: candidate=%s, fitness=%.3f, success_rate=%.3f, quality=%.3f",
		candidateID, fitness, multiObjFitness.SuccessRate, multiObjFitness.OutputQuality)
}

// updateRealTimePopulationStats updates population statistics in real-time.
func (g *GEPA) updateRealTimePopulationStats(candidateID string, fitness float64, multiObjFitness *MultiObjectiveFitness) {
	g.state.mu.Lock()
	defer g.state.mu.Unlock()

	// Update best fitness if necessary
	if fitness > g.state.BestFitness {
		g.state.BestFitness = fitness
		g.state.LastImprovement = time.Now()

		// Find and update best candidate
		if len(g.state.PopulationHistory) > 0 {
			currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
			for _, candidate := range currentPop.Candidates {
				if candidate.ID == candidateID {
					g.state.BestCandidate = candidate
					candidate.Fitness = fitness
					break
				}
			}
		}
	}

	// Update convergence monitoring
	g.updateConvergenceMonitoring(fitness, multiObjFitness)
}

// updateConvergenceMonitoring updates convergence status based on real-time fitness data.
func (g *GEPA) updateConvergenceMonitoring(fitness float64, multiObjFitness *MultiObjectiveFitness) {
	if g.state.ConvergenceStatus == nil {
		return
	}

	// Check for stagnation
	if fitness <= g.state.BestFitness {
		g.state.ConvergenceStatus.StagnationCount++
	} else {
		g.state.ConvergenceStatus.StagnationCount = 0
	}

	// Update diversity index based on current fitness distribution
	if len(g.state.PopulationHistory) > 0 {
		currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
		diversity := g.calculatePopulationDiversity(currentPop)
		g.state.ConvergenceStatus.DiversityIndex = diversity

		// Update exploration-exploitation balance
		if diversity > 0.7 {
			g.state.ConvergenceStatus.ExplorationExploitationBalance = "exploration"
		} else if diversity < 0.3 {
			g.state.ConvergenceStatus.ExplorationExploitationBalance = "exploitation"
		} else {
			g.state.ConvergenceStatus.ExplorationExploitationBalance = "balanced"
		}

		// Update premature convergence risk
		if g.state.ConvergenceStatus.StagnationCount > g.config.StagnationLimit/2 && diversity < 0.2 {
			g.state.ConvergenceStatus.PrematureConvergenceRisk = "high"
		} else if g.state.ConvergenceStatus.StagnationCount > g.config.StagnationLimit/4 && diversity < 0.4 {
			g.state.ConvergenceStatus.PrematureConvergenceRisk = "medium"
		} else {
			g.state.ConvergenceStatus.PrematureConvergenceRisk = "low"
		}
	}

	// Check for convergence
	if g.state.ConvergenceStatus.StagnationCount >= g.config.StagnationLimit {
		g.state.ConvergenceStatus.IsConverged = true
	}
}

// calculatePopulationDiversity calculates diversity index for a population.
func (g *GEPA) calculatePopulationDiversity(population *Population) float64 {
	if population == nil || len(population.Candidates) < 2 {
		return 1.0
	}

	// Calculate fitness variance as diversity metric
	var sum, sumSquared float64
	count := len(population.Candidates)

	for _, candidate := range population.Candidates {
		sum += candidate.Fitness
		sumSquared += candidate.Fitness * candidate.Fitness
	}

	mean := sum / float64(count)
	variance := (sumSquared / float64(count)) - (mean * mean)

	// Normalize variance to [0, 1] range
	maxVariance := 0.25 // Theoretical maximum for fitness in [0, 1]
	diversity := variance / maxVariance

	if diversity > 1.0 {
		diversity = 1.0
	}
	if diversity < 0.0 {
		diversity = 0.0
	}

	return diversity
}

// Enhanced interceptor chain monitoring

// getRealTimeInterceptorStats provides real-time statistics about interceptor performance.
//
//nolint:unused // This is an advanced feature not yet integrated
func (g *GEPA) getRealTimeInterceptorStats() map[string]interface{} {
	stats := map[string]interface{}{
		"total_executions":      0,
		"successful_executions": 0,
		"failed_executions":     0,
		"average_latency":       time.Duration(0),
		"interceptor_overhead":  time.Duration(0),
		"active_candidates":     0,
	}

	g.state.mu.RLock()
	defer g.state.mu.RUnlock()

	totalExecutions := 0
	successfulExecutions := 0
	var totalLatency time.Duration
	activeCandidates := make(map[string]bool)

	// Aggregate stats from all execution traces
	for _, traces := range g.state.ExecutionTraces {
		for _, trace := range traces {
			totalExecutions++
			if trace.Success {
				successfulExecutions++
			}
			totalLatency += trace.Duration
			activeCandidates[trace.CandidateID] = true
		}
	}

	stats["total_executions"] = totalExecutions
	stats["successful_executions"] = successfulExecutions
	stats["failed_executions"] = totalExecutions - successfulExecutions
	stats["active_candidates"] = len(activeCandidates)

	if totalExecutions > 0 {
		stats["average_latency"] = totalLatency / time.Duration(totalExecutions)
		stats["success_rate"] = float64(successfulExecutions) / float64(totalExecutions)
	}

	return stats
}

// monitorInterceptorHealth monitors the health of the interceptor chain.
//
//nolint:unused // This is an advanced feature not yet integrated
func (g *GEPA) monitorInterceptorHealth() {
	stats := g.getRealTimeInterceptorStats()

	// Log health metrics
	logger := logging.GetLogger()
	logger.Debug(context.Background(),
		"Interceptor health: executions=%d, success_rate=%.2f, avg_latency=%v, active_candidates=%d",
		stats["total_executions"],
		stats["success_rate"],
		stats["average_latency"],
		stats["active_candidates"])

	// Alert on potential issues
	if successRate, ok := stats["success_rate"].(float64); ok && successRate < 0.5 {
		logger.Warn(context.Background(),
			"Low interceptor success rate detected: %.2f", successRate)
	}

	if avgLatency, ok := stats["average_latency"].(time.Duration); ok && avgLatency > 5*time.Second {
		logger.Warn(context.Background(),
			"High interceptor latency detected: %v", avgLatency)
	}
}

// calculateFitness calculates fitness score for a given execution using enhanced metrics.
func (g *GEPA) calculateFitness(inputs, outputs map[string]any, err error) float64 {
	if err != nil {
		return 0.0
	}

	// Enhanced fitness calculation based on multiple factors
	fitness := 0.0

	// Base success score (40% weight)
	if len(outputs) > 0 {
		fitness += 0.4
	}

	// Output completeness score (30% weight)
	if outputs != nil {
		nonNilOutputs := 0
		for _, value := range outputs {
			if value != nil {
				nonNilOutputs++
			}
		}
		if len(outputs) > 0 {
			completeness := float64(nonNilOutputs) / float64(len(outputs))
			fitness += 0.3 * completeness
		}
	}

	// Output quality indicators (20% weight)
	if outputs != nil {
		qualityScore := g.assessOutputQuality(outputs)
		fitness += 0.2 * qualityScore
	}

	// Input utilization score (10% weight)
	if inputs != nil && outputs != nil {
		utilizationScore := g.assessInputUtilization(inputs, outputs)
		fitness += 0.1 * utilizationScore
	}

	return fitness
}

// Helper methods for enhanced fitness calculation

// assessOutputQuality evaluates the quality of outputs based on various indicators.
func (g *GEPA) assessOutputQuality(outputs map[string]any) float64 {
	qualityScore := 0.0
	qualityFactors := 0

	for _, value := range outputs {
		if value == nil {
			continue
		}

		qualityFactors++

		// Check for string outputs
		if str, ok := value.(string); ok {
			if len(str) > 10 && len(str) < 1000 { // Reasonable length
				qualityScore += 0.5
			}
			if !containsErrorKeywords(str) { // No error indicators
				qualityScore += 0.3
			}
			if containsStructuredContent(str) { // Has structure
				qualityScore += 0.2
			}
		} else {
			// Non-string outputs get base quality score
			qualityScore += 0.5
		}
	}

	if qualityFactors == 0 {
		return 0.0
	}

	return qualityScore / float64(qualityFactors)
}

// assessInputUtilization evaluates how well inputs were utilized in generating outputs.
func (g *GEPA) assessInputUtilization(inputs, outputs map[string]any) float64 {
	if len(inputs) == 0 {
		return 1.0 // No inputs to utilize
	}

	utilizationScore := 0.0

	for _, inputValue := range inputs {
		if inputValue == nil {
			continue
		}

		// Check if input appears to be reflected in outputs
		if inputStr, ok := inputValue.(string); ok && inputStr != "" {
			for _, outputValue := range outputs {
				if outputStr, ok := outputValue.(string); ok {
					// Simple heuristic: check if input concepts appear in output
					if containsInputConcepts(inputStr, outputStr) {
						utilizationScore += 1.0
						break
					}
				}
			}
		} else {
			// Non-string inputs: assume utilized if there are outputs
			if len(outputs) > 0 {
				utilizationScore += 1.0
			}
		}
	}

	return utilizationScore / float64(len(inputs))
}

// Helper functions for quality assessment

func containsErrorKeywords(text string) bool {
	errorKeywords := []string{"error", "failed", "unable", "cannot", "invalid", "undefined"}
	lowerText := strings.ToLower(text)
	for _, keyword := range errorKeywords {
		if strings.Contains(lowerText, keyword) {
			return true
		}
	}
	return false
}

func containsStructuredContent(text string) bool {
	// Simple heuristics for structured content
	return strings.Contains(text, ":") ||
		strings.Contains(text, "-") ||
		strings.Contains(text, ".") ||
		strings.Contains(text, "\n")
}

func containsInputConcepts(input, output string) bool {
	// Simple heuristic: check if key words from input appear in output
	inputWords := strings.Fields(strings.ToLower(input))
	outputLower := strings.ToLower(output)

	matchedWords := 0
	for _, word := range inputWords {
		if len(word) > 3 && strings.Contains(outputLower, word) { // Only count significant words
			matchedWords++
		}
	}

	// Consider utilized if at least 30% of significant words appear
	return len(inputWords) > 0 && float64(matchedWords)/float64(len(inputWords)) >= 0.3
}

// getInputTypes extracts type information from inputs for reflection.
func (g *GEPA) getInputTypes(inputs map[string]any) map[string]string {
	types := make(map[string]string)
	for key, value := range inputs {
		if value == nil {
			types[key] = "nil"
		} else {
			types[key] = fmt.Sprintf("%T", value)
		}
	}
	return types
}

// getOutputTypes extracts type information from outputs for reflection.
func (g *GEPA) getOutputTypes(outputs map[string]any) map[string]string {
	types := make(map[string]string)
	for key, value := range outputs {
		if value == nil {
			types[key] = "nil"
		} else {
			types[key] = fmt.Sprintf("%T", value)
		}
	}
	return types
}

// getErrorType categorizes error types for reflection.
func (g *GEPA) getErrorType(err error) string {
	if err == nil {
		return "none"
	}
	return g.categorizeError(err)
}

// logOptimizationResults logs the final optimization results.
func (g *GEPA) logOptimizationResults(ctx context.Context) {
	logger := logging.GetLogger()

	duration := time.Since(g.state.StartTime)

	logger.Info(ctx, "GEPA optimization completed: duration=%v, generations=%d, best_fitness=%.3f, population_size=%d",
		duration,
		g.state.CurrentGeneration,
		g.state.BestFitness,
		g.config.PopulationSize)

	if g.state.BestCandidate != nil {
		logger.Info(ctx, "Best candidate found: id=%s, generation=%d, fitness=%.3f, instruction=%s",
			g.state.BestCandidate.ID,
			g.state.BestCandidate.Generation,
			g.state.BestCandidate.Fitness,
			g.state.BestCandidate.Instruction)
	}
}

// Population Management Methods

// initializePopulation creates the initial population of prompt candidates.
func (g *GEPA) initializePopulation(ctx context.Context, program core.Program) error {
	logger := logging.GetLogger()
	logger.Info(ctx, "Initializing GEPA population with size %d", g.config.PopulationSize)

	candidates := make([]*GEPACandidate, 0, g.config.PopulationSize)

	// Create diverse initial population for each module
	for moduleName, module := range program.Modules {
		baseInstruction := g.extractInstructionFromModule(module)

		// Calculate how many candidates per module
		candidatesPerModule := g.config.PopulationSize / len(program.Modules)
		if candidatesPerModule == 0 {
			candidatesPerModule = 1
		}

		// Generate variations for this module
		variations, err := g.generateInitialVariations(ctx, baseInstruction, moduleName, candidatesPerModule)
		if err != nil {
			logger.Error(ctx, "Failed to generate variations for module %s: %v", moduleName, err)
			// Fallback to base instruction
			variations = []string{baseInstruction}
		}

		// Create candidates from variations
		for i, variation := range variations {
			if len(candidates) >= g.config.PopulationSize {
				break
			}

			candidate := &GEPACandidate{
				ID:          g.generateCandidateID(),
				ModuleName:  moduleName,
				Instruction: variation,
				Generation:  0,
				Fitness:     0.0,
				ParentIDs:   []string{},
				CreatedAt:   time.Now(),
				Metadata: map[string]interface{}{
					"variation_index":  i,
					"base_instruction": baseInstruction,
				},
			}
			candidates = append(candidates, candidate)

			// Initialize candidate metrics
			g.state.CandidateMetrics[candidate.ID] = &CandidateMetrics{
				TotalEvaluations: 0,
				SuccessCount:     0,
				AverageFitness:   0.0,
				BestFitness:      0.0,
				ExecutionTimes:   make([]time.Duration, 0),
				ErrorCounts:      make(map[string]int),
				Metadata:         make(map[string]interface{}),
			}
		}
	}

	// Fill remaining slots if needed
	for len(candidates) < g.config.PopulationSize {
		// Duplicate and mutate existing candidates
		if len(candidates) > 0 {
			original := candidates[g.rng.Intn(len(candidates))]
			mutated := g.createMutatedCandidate(original)
			candidates = append(candidates, mutated)
		} else {
			break
		}
	}

	// Create initial population
	population := &Population{
		Candidates:    candidates,
		Generation:    0,
		BestFitness:   0.0,
		BestCandidate: nil,
		Size:          len(candidates),
	}

	g.state.mu.Lock()
	g.state.PopulationHistory = []*Population{population}
	g.state.mu.Unlock()

	logger.Info(ctx, "Population initialized with %d candidates across %d modules",
		len(candidates),
		len(program.Modules))

	return nil
}

// extractInstructionFromModule extracts the current instruction from a module.
func (g *GEPA) extractInstructionFromModule(module core.Module) string {
	signature := module.GetSignature()
	if signature.Instruction != "" {
		return signature.Instruction
	}

	// Fallback to a generic instruction based on signature
	if len(signature.Inputs) > 0 && len(signature.Outputs) > 0 {
		inputNames := make([]string, 0, len(signature.Inputs))
		for _, input := range signature.Inputs {
			inputNames = append(inputNames, input.Name)
		}
		outputNames := make([]string, 0, len(signature.Outputs))
		for _, output := range signature.Outputs {
			outputNames = append(outputNames, output.Name)
		}

		return fmt.Sprintf("Given %s, provide %s.",
			strings.Join(inputNames, ", "),
			strings.Join(outputNames, ", "))
	}

	return "Process the input and provide appropriate output."
}

// generateInitialVariations creates diverse variations of the base instruction.
func (g *GEPA) generateInitialVariations(ctx context.Context, baseInstruction, moduleName string, count int) ([]string, error) {
	if count <= 1 {
		return []string{baseInstruction}, nil
	}

	prompt := fmt.Sprintf(`Generate %d diverse variations of this instruction for a %s module:
Original: "%s"

Requirements:
- Maintain the core intent and functionality
- Vary specificity, tone, and approach
- Include different levels of detail
- Use different phrasing styles
- Each variation should be on a separate line
- Number each variation (1., 2., etc.)

Variations:`, count-1, moduleName, baseInstruction)

	response, err := g.generationLLM.Generate(ctx, prompt)
	if err != nil {
		return []string{baseInstruction}, fmt.Errorf("failed to generate variations: %w", err)
	}

	variations := g.parseVariations(response.Content)

	// Always include the original as the first variation
	result := []string{baseInstruction}
	result = append(result, variations...)

	// Ensure we don't exceed the requested count
	if len(result) > count {
		result = result[:count]
	}

	return result, nil
}

// parseVariations extracts variations from LLM response.
func (g *GEPA) parseVariations(content string) []string {
	lines := strings.Split(content, "\n")
	variations := make([]string, 0)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Remove numbering (1., 2., etc.) and leading/trailing quotes
		line = strings.TrimPrefix(line, "1.")
		line = strings.TrimPrefix(line, "2.")
		line = strings.TrimPrefix(line, "3.")
		line = strings.TrimPrefix(line, "4.")
		line = strings.TrimPrefix(line, "5.")
		line = strings.TrimPrefix(line, "6.")
		line = strings.TrimPrefix(line, "7.")
		line = strings.TrimPrefix(line, "8.")
		line = strings.TrimPrefix(line, "9.")
		line = strings.TrimSpace(line)
		line = strings.Trim(line, "\"'")

		if line != "" && len(line) > 10 { // Minimum reasonable instruction length
			variations = append(variations, line)
		}
	}

	return variations
}

// generateCandidateID generates a unique ID for candidates.
func (g *GEPA) generateCandidateID() string {
	return fmt.Sprintf("gepa_%d_%d", time.Now().UnixNano(), g.rng.Intn(10000))
}

// createMutatedCandidate creates a mutated version of an existing candidate.
func (g *GEPA) createMutatedCandidate(original *GEPACandidate) *GEPACandidate {
	// Simple mutation for initialization - just add a variation prefix
	mutationPrefixes := []string{
		"Carefully ",
		"Thoroughly ",
		"Precisely ",
		"Systematically ",
		"Effectively ",
	}

	prefix := mutationPrefixes[g.rng.Intn(len(mutationPrefixes))]
	mutatedInstruction := prefix + strings.ToLower(string(original.Instruction[0])) + original.Instruction[1:]

	return &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  original.ModuleName,
		Instruction: mutatedInstruction,
		Generation:  0,
		Fitness:     0.0,
		ParentIDs:   []string{original.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"mutation_type": "prefix_addition",
			"parent_id":     original.ID,
		},
	}
}

// getCurrentPopulation returns the current population.
func (g *GEPA) getCurrentPopulation() *Population {
	g.state.mu.RLock()
	defer g.state.mu.RUnlock()

	if len(g.state.PopulationHistory) > 0 {
		return g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
	}

	return nil
}

// setCurrentMultiObjectiveFitnessMap stores the multi-objective fitness map for the current generation.
func (g *GEPA) setCurrentMultiObjectiveFitnessMap(fitnessMap map[string]*MultiObjectiveFitness) {
	g.state.mu.Lock()
	defer g.state.mu.Unlock()
	g.state.MultiObjectiveFitnessMap = fitnessMap
}

// getCurrentMultiObjectiveFitnessMap returns the current multi-objective fitness map.
func (g *GEPA) getCurrentMultiObjectiveFitnessMap() map[string]*MultiObjectiveFitness {
	g.state.mu.RLock()
	defer g.state.mu.RUnlock()
	return g.state.MultiObjectiveFitnessMap
}

// updateBestCandidate updates the best candidate if a better one is found.
func (g *GEPA) updateBestCandidate(candidate *GEPACandidate) {
	g.state.mu.Lock()
	defer g.state.mu.Unlock()

	if g.state.BestCandidate == nil || candidate.Fitness > g.state.BestFitness {
		g.state.BestCandidate = candidate
		g.state.BestFitness = candidate.Fitness
		g.state.LastImprovement = time.Now()

		logger := logging.GetLogger()
		logger.Info(context.Background(), "New best candidate found: id=%s, fitness=%.3f, generation=%d, instruction=%s",
			candidate.ID,
			candidate.Fitness,
			candidate.Generation,
			candidate.Instruction)
	}
}

// Evolutionary Operators

// evolvePopulation creates the next generation using evolutionary operators.
func (g *GEPA) evolvePopulation(ctx context.Context) error {
	logger := logging.GetLogger()
	currentPop := g.getCurrentPopulation()
	if currentPop == nil {
		return fmt.Errorf("no current population found")
	}

	logger.Info(ctx, "Evolving population for generation %d", currentPop.Generation)

	// Select parents for reproduction
	parents := g.selectParents(currentPop)

	// Create offspring through crossover and mutation
	offspring := make([]*GEPACandidate, 0, g.config.PopulationSize)

	// Elitism: keep best candidates
	eliteCount := int(float64(g.config.PopulationSize) * g.config.ElitismRate)
	elite := g.selectElite(currentPop, eliteCount)
	offspring = append(offspring, elite...)

	// Generate remaining offspring
	for len(offspring) < g.config.PopulationSize {
		// Select two parents
		parent1 := parents[g.rng.Intn(len(parents))]
		parent2 := parents[g.rng.Intn(len(parents))]

		// Ensure parents are different (if possible)
		for parent2.ID == parent1.ID && len(parents) > 1 {
			parent2 = parents[g.rng.Intn(len(parents))]
		}

		// Apply crossover
		child1, child2 := g.crossover(parent1, parent2)

		// Apply mutation
		child1 = g.mutate(ctx, child1)
		if len(offspring) < g.config.PopulationSize-1 {
			child2 = g.mutate(ctx, child2)
		}

		offspring = append(offspring, child1)
		if len(offspring) < g.config.PopulationSize {
			offspring = append(offspring, child2)
		}
	}

	// Ensure we don't exceed population size
	if len(offspring) > g.config.PopulationSize {
		offspring = offspring[:g.config.PopulationSize]
	}

	// Create new population
	newPopulation := &Population{
		Candidates:    offspring,
		Generation:    currentPop.Generation + 1,
		BestFitness:   0.0,
		BestCandidate: nil,
		Size:          len(offspring),
	}

	// Add to population history
	g.state.mu.Lock()
	g.state.PopulationHistory = append(g.state.PopulationHistory, newPopulation)
	g.state.mu.Unlock()

	// Initialize metrics for new candidates
	for _, candidate := range offspring {
		if _, exists := g.state.CandidateMetrics[candidate.ID]; !exists {
			g.state.CandidateMetrics[candidate.ID] = &CandidateMetrics{
				TotalEvaluations: 0,
				SuccessCount:     0,
				AverageFitness:   0.0,
				BestFitness:      0.0,
				ExecutionTimes:   make([]time.Duration, 0),
				ErrorCounts:      make(map[string]int),
				Metadata:         make(map[string]interface{}),
			}
		}
	}

	logger.Info(ctx, "Population evolved: generation=%d, offspring_count=%d, elite_count=%d",
		newPopulation.Generation,
		len(offspring),
		eliteCount)

	return nil
}

// selectParents selects parents for reproduction based on the selection strategy.
func (g *GEPA) selectParents(population *Population) []*GEPACandidate {
	logger := logging.GetLogger()
	selectionSize := g.config.PopulationSize / 2

	switch g.config.SelectionStrategy {
	case "tournament":
		logger.Debug(context.Background(), "Using tournament selection for parent selection")
		return g.tournamentSelection(population, selectionSize)
	case "roulette":
		logger.Debug(context.Background(), "Using roulette selection for parent selection")
		return g.rouletteSelection(population, selectionSize)
	case "pareto":
		logger.Debug(context.Background(), "Using Pareto-based selection for parent selection")
		return g.paretoBasedSelection(population, selectionSize)
	case "adaptive_pareto":
		logger.Debug(context.Background(), "Using adaptive Pareto-based selection for parent selection")
		return g.adaptiveParetoSelection(population, selectionSize)
	default:
		logger.Debug(context.Background(), "Using default tournament selection for parent selection")
		return g.tournamentSelection(population, selectionSize)
	}
}

// tournamentSelection implements tournament selection.
func (g *GEPA) tournamentSelection(population *Population, count int) []*GEPACandidate {
	selected := make([]*GEPACandidate, 0, count)

	for i := 0; i < count; i++ {
		// Run tournament
		tournament := make([]*GEPACandidate, g.config.TournamentSize)
		for j := 0; j < g.config.TournamentSize; j++ {
			idx := g.rng.Intn(len(population.Candidates))
			tournament[j] = population.Candidates[idx]
		}

		// Select best from tournament
		best := tournament[0]
		for _, candidate := range tournament[1:] {
			if candidate.Fitness > best.Fitness {
				best = candidate
			}
		}

		selected = append(selected, best)
	}

	return selected
}

// rouletteSelection implements roulette wheel selection.
func (g *GEPA) rouletteSelection(population *Population, count int) []*GEPACandidate {
	// Calculate fitness sum
	totalFitness := 0.0
	for _, candidate := range population.Candidates {
		totalFitness += candidate.Fitness
	}

	if totalFitness == 0 {
		// Fallback to random selection if no fitness
		selected := make([]*GEPACandidate, 0, count)
		for i := 0; i < count; i++ {
			idx := g.rng.Intn(len(population.Candidates))
			selected = append(selected, population.Candidates[idx])
		}
		return selected
	}

	selected := make([]*GEPACandidate, 0, count)
	for i := 0; i < count; i++ {
		spin := g.rng.Float64() * totalFitness
		cumulative := 0.0

		for _, candidate := range population.Candidates {
			cumulative += candidate.Fitness
			if cumulative >= spin {
				selected = append(selected, candidate)
				break
			}
		}
	}

	return selected
}

// selectElite selects the best candidates for elitism using multi-objective Pareto-based or single-objective methods.
func (g *GEPA) selectElite(population *Population, count int) []*GEPACandidate {
	logger := logging.GetLogger()

	if count <= 0 {
		return []*GEPACandidate{}
	}

	// Try to use Pareto-based elite selection for multi-objective optimization
	if g.config.SelectionStrategy == "pareto" || g.config.SelectionStrategy == "adaptive_pareto" {
		fitnessMap := g.buildMultiObjectiveFitnessMap(population.Candidates)

		// Check if we have valid multi-objective fitness data
		hasValidFitness := false
		for _, fitness := range fitnessMap {
			if fitness != nil {
				hasValidFitness = true
				break
			}
		}

		if hasValidFitness {
			logger.Debug(context.Background(), "Using Pareto-based elite selection: count=%d", count)
			paretoElite := g.selectParetoElite(population.Candidates, fitnessMap, count)

			// Create copies for next generation
			elite := make([]*GEPACandidate, len(paretoElite))
			for i, candidate := range paretoElite {
				elite[i] = g.copyCandidate(candidate)
				elite[i].Generation = population.Generation + 1
			}

			logger.Debug(context.Background(), "Pareto elite selection completed: selected=%d candidates", len(elite))
			return elite
		}
	}

	// Fallback to traditional single-objective elite selection
	logger.Debug(context.Background(), "Using single-objective elite selection: count=%d", count)

	// Sort candidates by fitness (descending)
	candidates := make([]*GEPACandidate, len(population.Candidates))
	copy(candidates, population.Candidates)

	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].Fitness > candidates[i].Fitness {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Take top candidates
	if count > len(candidates) {
		count = len(candidates)
	}

	elite := make([]*GEPACandidate, count)
	for i := 0; i < count; i++ {
		// Create copies of elite candidates for next generation
		elite[i] = g.copyCandidate(candidates[i])
		elite[i].Generation = population.Generation + 1
	}

	return elite
}

// selectParetoElite selects elite candidates using Pareto dominance and crowding distance.
func (g *GEPA) selectParetoElite(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness, count int) []*GEPACandidate {
	if len(candidates) <= count {
		return candidates
	}

	// Calculate Pareto fronts
	fronts := g.calculateParetoFronts(candidates, fitnessMap)
	if len(fronts) == 0 {
		// Fallback to first N candidates if no fronts
		if count > len(candidates) {
			count = len(candidates)
		}
		return candidates[:count]
	}

	selected := make([]*GEPACandidate, 0, count)

	// Add candidates from Pareto fronts in order of rank
	for _, front := range fronts {
		if len(selected)+len(front.Candidates) <= count {
			// Add entire front
			selected = append(selected, front.Candidates...)
		} else {
			// Need to select subset from this front using crowding distance
			remaining := count - len(selected)
			crowdingDistances := g.calculateCrowdingDistance(front.Candidates, fitnessMap)

			// Sort front candidates by crowding distance (descending)
			frontCandidates := make([]*GEPACandidate, len(front.Candidates))
			copy(frontCandidates, front.Candidates)

			for i := 0; i < len(frontCandidates)-1; i++ {
				for j := 0; j < len(frontCandidates)-1-i; j++ {
					if crowdingDistances[frontCandidates[j].ID] < crowdingDistances[frontCandidates[j+1].ID] {
						frontCandidates[j], frontCandidates[j+1] = frontCandidates[j+1], frontCandidates[j]
					}
				}
			}

			// Add the most diverse candidates from this front
			selected = append(selected, frontCandidates[:remaining]...)
			break
		}

		if len(selected) >= count {
			break
		}
	}

	return selected
}

// crossover applies crossover between two parents.
func (g *GEPA) crossover(parent1, parent2 *GEPACandidate) (*GEPACandidate, *GEPACandidate) {
	if g.rng.Float64() > g.config.CrossoverRate {
		// No crossover, return copies of parents
		return g.copyCandidate(parent1), g.copyCandidate(parent2)
	}

	return g.semanticCrossover(parent1, parent2)
}

// semanticCrossover performs LLM-based semantic crossover.
func (g *GEPA) semanticCrossover(parent1, parent2 *GEPACandidate) (*GEPACandidate, *GEPACandidate) {
	prompt := fmt.Sprintf(`Create two new instruction variations by combining the best aspects of these parent instructions:

Parent 1 (fitness: %.3f): "%s"
Parent 2 (fitness: %.3f): "%s"

Generate two offspring that:
1. Combine semantic elements from both parents  
2. Maintain clarity and effectiveness
3. Create novel but coherent instructions
4. Each offspring should be on a separate line
5. Number each offspring (1., 2.)

Offspring:`,
		parent1.Fitness, parent1.Instruction,
		parent2.Fitness, parent2.Instruction)

	response, err := g.generationLLM.Generate(context.Background(), prompt)
	if err != nil {
		// Fallback: simple text mixing
		return g.fallbackCrossover(parent1, parent2)
	}

	offspring := g.parseOffspring(response.Content)
	if len(offspring) < 2 {
		return g.fallbackCrossover(parent1, parent2)
	}

	child1 := &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  parent1.ModuleName,
		Instruction: offspring[0],
		Generation:  utils.Max(parent1.Generation, parent2.Generation) + 1,
		ParentIDs:   []string{parent1.ID, parent2.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"crossover_type":  "semantic",
			"parent1_fitness": parent1.Fitness,
			"parent2_fitness": parent2.Fitness,
		},
	}

	child2 := &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  parent1.ModuleName,
		Instruction: offspring[1],
		Generation:  utils.Max(parent1.Generation, parent2.Generation) + 1,
		ParentIDs:   []string{parent1.ID, parent2.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"crossover_type":  "semantic",
			"parent1_fitness": parent1.Fitness,
			"parent2_fitness": parent2.Fitness,
		},
	}

	return child1, child2
}

// fallbackCrossover provides simple fallback crossover when LLM fails.
func (g *GEPA) fallbackCrossover(parent1, parent2 *GEPACandidate) (*GEPACandidate, *GEPACandidate) {
	// Simple structural mixing - split instructions and recombine
	words1 := strings.Fields(parent1.Instruction)
	words2 := strings.Fields(parent2.Instruction)

	// Create child1: first half of parent1 + second half of parent2
	split1 := len(words1) / 2
	split2 := len(words2) / 2

	child1Words := append(words1[:split1], words2[split2:]...)
	child2Words := append(words2[:split2], words1[split1:]...)

	child1 := &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  parent1.ModuleName,
		Instruction: strings.Join(child1Words, " "),
		Generation:  utils.Max(parent1.Generation, parent2.Generation) + 1,
		ParentIDs:   []string{parent1.ID, parent2.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"crossover_type": "structural_fallback",
		},
	}

	child2 := &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  parent1.ModuleName,
		Instruction: strings.Join(child2Words, " "),
		Generation:  utils.Max(parent1.Generation, parent2.Generation) + 1,
		ParentIDs:   []string{parent1.ID, parent2.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"crossover_type": "structural_fallback",
		},
	}

	return child1, child2
}

// parseOffspring extracts offspring from crossover LLM response.
func (g *GEPA) parseOffspring(content string) []string {
	lines := strings.Split(content, "\n")
	offspring := make([]string, 0, 2)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Remove numbering and quotes
		line = strings.TrimPrefix(line, "1.")
		line = strings.TrimPrefix(line, "2.")
		line = strings.TrimSpace(line)
		line = strings.Trim(line, "\"'")

		if line != "" && len(line) > 10 {
			offspring = append(offspring, line)
			if len(offspring) >= 2 {
				break
			}
		}
	}

	return offspring
}

// mutate applies mutation to a candidate.
func (g *GEPA) mutate(ctx context.Context, candidate *GEPACandidate) *GEPACandidate {
	if g.rng.Float64() > g.config.MutationRate {
		return candidate // No mutation
	}

	return g.semanticMutation(ctx, candidate)
}

// semanticMutation performs LLM-based semantic mutation.
func (g *GEPA) semanticMutation(ctx context.Context, candidate *GEPACandidate) *GEPACandidate {
	mutationTypes := []string{
		"enhance_specificity",
		"simplify_language",
		"add_examples",
		"change_tone",
		"restructure_logic",
		"add_constraints",
		"modify_emphasis",
	}

	mutationType := mutationTypes[g.rng.Intn(len(mutationTypes))]

	prompt := fmt.Sprintf(`Apply a "%s" mutation to this instruction:
Original: "%s"

Mutation requirements for "%s":
%s

Generate a mutated version that maintains the core intent while applying the specified change:`,
		mutationType,
		candidate.Instruction,
		mutationType,
		g.getMutationDescription(mutationType))

	response, err := g.generationLLM.Generate(ctx, prompt)
	if err != nil {
		return g.fallbackMutation(candidate)
	}

	mutatedInstruction := g.extractMutatedInstruction(response.Content)
	if mutatedInstruction == "" || mutatedInstruction == candidate.Instruction {
		return g.fallbackMutation(candidate)
	}

	return &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  candidate.ModuleName,
		Instruction: mutatedInstruction,
		Generation:  candidate.Generation + 1,
		ParentIDs:   []string{candidate.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"mutation_type":  mutationType,
			"parent_fitness": candidate.Fitness,
		},
	}
}

// getMutationDescription returns description for mutation type.
func (g *GEPA) getMutationDescription(mutationType string) string {
	descriptions := map[string]string{
		"enhance_specificity": "Make the instruction more specific and detailed",
		"simplify_language":   "Use simpler, clearer language while maintaining meaning",
		"add_examples":        "Include concrete examples or scenarios",
		"change_tone":         "Modify the tone (formal/casual, direct/gentle, etc.)",
		"restructure_logic":   "Reorganize the logical flow of the instruction",
		"add_constraints":     "Add helpful constraints or guidelines",
		"modify_emphasis":     "Change what aspects are emphasized or prioritized",
	}
	return descriptions[mutationType]
}

// extractMutatedInstruction extracts the mutated instruction from LLM response.
func (g *GEPA) extractMutatedInstruction(content string) string {
	lines := strings.Split(content, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "Mutation") || strings.HasPrefix(line, "Original") {
			continue
		}

		// Remove quotes and clean up
		line = strings.Trim(line, "\"'")
		if len(line) > 10 {
			return line
		}
	}

	return ""
}

// fallbackMutation provides simple fallback mutation when LLM fails.
func (g *GEPA) fallbackMutation(candidate *GEPACandidate) *GEPACandidate {
	// Simple word-level mutations
	words := strings.Fields(candidate.Instruction)
	if len(words) == 0 {
		return candidate
	}

	// Random mutations
	mutationType := g.rng.Intn(3)
	var mutatedInstruction string

	switch mutationType {
	case 0: // Add adjective
		adjectives := []string{"carefully", "thoroughly", "precisely", "clearly", "effectively"}
		adj := adjectives[g.rng.Intn(len(adjectives))]
		mutatedInstruction = adj + " " + candidate.Instruction
	case 1: // Replace a word with synonym
		synonyms := map[string][]string{
			"provide": {"give", "supply", "deliver", "present"},
			"analyze": {"examine", "study", "evaluate", "assess"},
			"create":  {"generate", "produce", "develop", "build"},
			"explain": {"describe", "clarify", "elaborate", "detail"},
		}

		mutatedWords := make([]string, len(words))
		copy(mutatedWords, words)

		for i, word := range words {
			if syns, exists := synonyms[strings.ToLower(word)]; exists && g.rng.Float64() < 0.3 {
				mutatedWords[i] = syns[g.rng.Intn(len(syns))]
				break
			}
		}
		mutatedInstruction = strings.Join(mutatedWords, " ")
	default: // Add qualifying phrase
		phrases := []string{" with attention to detail", " ensuring accuracy", " step by step", " comprehensively"}
		phrase := phrases[g.rng.Intn(len(phrases))]
		mutatedInstruction = candidate.Instruction + phrase
	}

	return &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  candidate.ModuleName,
		Instruction: mutatedInstruction,
		Generation:  candidate.Generation + 1,
		ParentIDs:   []string{candidate.ID},
		CreatedAt:   time.Now(),
		Metadata: map[string]interface{}{
			"mutation_type":  "fallback",
			"parent_fitness": candidate.Fitness,
		},
	}
}

// copyCandidate creates a deep copy of a candidate.
func (g *GEPA) copyCandidate(original *GEPACandidate) *GEPACandidate {
	metadata := make(map[string]interface{})
	for k, v := range original.Metadata {
		metadata[k] = v
	}

	parentIDs := make([]string, len(original.ParentIDs))
	copy(parentIDs, original.ParentIDs)

	demonstrations := make([]core.Example, len(original.Demonstrations))
	copy(demonstrations, original.Demonstrations)

	return &GEPACandidate{
		ID:             original.ID,
		ModuleName:     original.ModuleName,
		Instruction:    original.Instruction,
		Demonstrations: demonstrations,
		Generation:     original.Generation,
		Fitness:        original.Fitness,
		ParentIDs:      parentIDs,
		CreatedAt:      original.CreatedAt,
		Metadata:       metadata,
	}
}

// Population Evaluation Methods

// evaluatePopulation evaluates all candidates in the current population and returns multi-objective fitness map.
func (g *GEPA) evaluatePopulation(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (map[string]*MultiObjectiveFitness, error) {
	logger := logging.GetLogger()
	population := g.getCurrentPopulation()
	if population == nil {
		return nil, fmt.Errorf("no current population to evaluate")
	}

	logger.Info(ctx, "Evaluating population: generation=%d, candidates=%d",
		population.Generation,
		len(population.Candidates))

	// Use concurrent evaluation with controlled concurrency
	p := pool.New().WithMaxGoroutines(g.config.ConcurrencyLevel)

	var mu sync.Mutex
	evaluatedCount := 0
	multiObjFitnessMap := make(map[string]*MultiObjectiveFitness)

	for _, candidate := range population.Candidates {
		candidate := candidate // Capture loop variable
		p.Go(func() {
			fitness := g.evaluateCandidate(ctx, candidate, program, dataset, metric)

			mu.Lock()
			candidate.Fitness = fitness
			evaluatedCount++

			// Get multi-objective fitness from candidate metrics
			if metrics := g.performanceLogger.GetCandidateMetrics(candidate.ID); metrics != nil {
				if multiObjFitness, ok := metrics.Metadata["multi_objective_fitness"].(*MultiObjectiveFitness); ok {
					multiObjFitnessMap[candidate.ID] = multiObjFitness
				}
			}

			// Update best candidate
			g.updateBestCandidate(candidate)

			// Update population stats
			if fitness > population.BestFitness {
				population.BestFitness = fitness
				population.BestCandidate = candidate
			}

			// Check for progress logging while still holding the mutex
			if evaluatedCount%5 == 0 {
				logger.Info(ctx, "Evaluation progress: evaluated=%d, total=%d, current_fitness=%.3f",
					evaluatedCount,
					len(population.Candidates),
					fitness)
			}
			mu.Unlock()
		})
	}

	p.Wait()

	logger.Info(ctx, "Population evaluation completed: generation=%d, best_fitness=%.3f, evaluated_candidates=%d, multi_objective_entries=%d",
		population.Generation,
		population.BestFitness,
		evaluatedCount,
		len(multiObjFitnessMap))

	return multiObjFitnessMap, nil
}

// evaluateCandidate evaluates a single candidate's fitness.
func (g *GEPA) evaluateCandidate(ctx context.Context, candidate *GEPACandidate,
	program core.Program, dataset core.Dataset, metric core.Metric) float64 {

	// Create a copy of the program with the candidate's instruction
	modifiedProgram := g.applyCandidate(program, candidate)

	// Add candidate ID to context for tracking
	candidateCtx := context.WithValue(ctx, candidateIDKey, candidate.ID)

	// Evaluate on dataset
	totalScore := 0.0
	evaluationCount := 0

	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		// Execute program with candidate's instruction
		outputs, err := modifiedProgram.Execute(candidateCtx, example.Inputs)
		if err != nil {
			// Penalize errors
			continue
		}

		// Calculate score using metric
		score := metric(example.Outputs, outputs)
		totalScore += score
		evaluationCount++

		// Limit evaluations for efficiency
		if evaluationCount >= g.config.EvaluationBatchSize {
			break
		}
	}

	if evaluationCount == 0 {
		return 0.0
	}

	return totalScore / float64(evaluationCount)
}

// applyCandidate applies a candidate's instruction to a program.
func (g *GEPA) applyCandidate(program core.Program, candidate *GEPACandidate) core.Program {
	// Clone the program
	modified := program.Clone()

	// Find the module that matches this candidate and update its instruction
	for moduleName, module := range modified.Modules {
		if moduleName == candidate.ModuleName {
			// Update the module's signature with the new instruction
			sig := module.GetSignature()
			newSig := core.Signature{
				Instruction: candidate.Instruction,
				Inputs:      sig.Inputs,
				Outputs:     sig.Outputs,
			}

			module.SetSignature(newSig)
			break
		}
	}

	return modified
}

// Reflection Engine

// performReflection conducts reflection analysis on the current generation.
func (g *GEPA) performReflection(ctx context.Context, generation int) error {
	logger := logging.GetLogger()
	logger.Info(ctx, "Performing reflection analysis for generation %d", generation)

	population := g.getCurrentPopulation()
	if population == nil {
		return fmt.Errorf("no current population for reflection")
	}

	// Individual candidate reflections
	reflections := make([]*ReflectionResult, 0, len(population.Candidates))

	for _, candidate := range population.Candidates {
		traces := g.state.GetTracesForCandidate(candidate.ID)
		if len(traces) == 0 {
			continue // Skip candidates with no execution traces
		}

		reflection, err := g.reflectOnCandidate(ctx, candidate, traces)
		if err != nil {
			logger.Error(ctx, "Failed to reflect on candidate %s: %v",
				candidate.ID, err)
			continue
		}

		reflections = append(reflections, reflection)
	}

	// Store reflection results
	g.state.mu.Lock()
	g.state.ReflectionHistory = append(g.state.ReflectionHistory, reflections...)
	g.state.mu.Unlock()

	logger.Info(ctx, "Reflection analysis completed: generation=%d, reflections_generated=%d",
		generation,
		len(reflections))

	return nil
}

// reflectOnCandidate performs reflection analysis on a single candidate.
func (g *GEPA) reflectOnCandidate(ctx context.Context, candidate *GEPACandidate,
	traces []ExecutionTrace) (*ReflectionResult, error) {

	// Analyze execution patterns
	patterns := g.analyzeExecutionPatterns(traces)

	// Generate reflection prompt
	prompt := g.buildReflectionPrompt(candidate, patterns)

	// Get reflection from LLM
	response, err := g.reflectionLLM.Generate(ctx, prompt)
	if err != nil {
		return g.createFallbackReflection(candidate, patterns), nil
	}

	// Parse reflection response
	reflection := g.parseReflectionResponse(response.Content, candidate.ID)

	return reflection, nil
}

// analyzeExecutionPatterns analyzes execution traces to identify patterns.
func (g *GEPA) analyzeExecutionPatterns(traces []ExecutionTrace) *ExecutionPatterns {
	if len(traces) == 0 {
		return &ExecutionPatterns{
			ErrorDistribution: make(map[string]int),
			PerformanceTrends: make([]float64, 0),
		}
	}

	patterns := &ExecutionPatterns{
		ErrorDistribution: make(map[string]int),
		PerformanceTrends: make([]float64, 0),
	}

	totalDuration := time.Duration(0)
	successCount := 0

	for _, trace := range traces {
		totalDuration += trace.Duration
		if trace.Success {
			successCount++
			patterns.PerformanceTrends = append(patterns.PerformanceTrends, 1.0)
		} else {
			patterns.PerformanceTrends = append(patterns.PerformanceTrends, 0.0)
			if trace.Error != nil {
				errorType := g.categorizeError(trace.Error)
				patterns.ErrorDistribution[errorType]++
			}
		}
	}

	patterns.TotalExecutions = len(traces)
	patterns.SuccessCount = successCount
	patterns.SuccessRate = float64(successCount) / float64(len(traces))
	patterns.AverageResponseTime = totalDuration / time.Duration(len(traces))
	patterns.CommonFailures = g.identifyCommonFailures(patterns.ErrorDistribution)
	patterns.QualityIndicators = g.identifyQualityIndicators(patterns.PerformanceTrends)

	return patterns
}

// buildReflectionPrompt creates a reflection prompt for a candidate.
func (g *GEPA) buildReflectionPrompt(candidate *GEPACandidate, patterns *ExecutionPatterns) string {
	return fmt.Sprintf(`As an expert prompt engineer, critically analyze this instruction and its performance:

INSTRUCTION: "%s"
MODULE: %s
GENERATION: %d
FITNESS: %.3f

EXECUTION PATTERNS:
- Success Rate: %.1f%% (%d/%d executions)
- Average Response Time: %v
- Common Failure Modes: %s
- Quality Indicators: %s

Please provide a detailed self-reflection covering:

1. STRENGTHS: What aspects of this instruction work well?
2. WEAKNESSES: What specific issues limit its effectiveness?
3. IMPROVEMENT SUGGESTIONS: Concrete ways to enhance this instruction
4. CONFIDENCE: Rate your confidence in this analysis (0.0-1.0)

Format your response as:
STRENGTHS:
- [strength 1]
- [strength 2]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

CONFIDENCE: [0.0-1.0]`,
		candidate.Instruction,
		candidate.ModuleName,
		candidate.Generation,
		candidate.Fitness,
		patterns.SuccessRate*100,
		patterns.SuccessCount,
		patterns.TotalExecutions,
		patterns.AverageResponseTime,
		strings.Join(patterns.CommonFailures, ", "),
		strings.Join(patterns.QualityIndicators, ", "))
}

// parseReflectionResponse parses the LLM reflection response.
func (g *GEPA) parseReflectionResponse(content, candidateID string) *ReflectionResult {
	reflection := &ReflectionResult{
		CandidateID:     candidateID,
		Strengths:       make([]string, 0),
		Weaknesses:      make([]string, 0),
		Suggestions:     make([]string, 0),
		ConfidenceScore: 0.5, // Default confidence
		Timestamp:       time.Now(),
		ReflectionDepth: g.config.ReflectionDepth,
	}

	lines := strings.Split(content, "\n")
	currentSection := ""

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Identify sections
		if strings.HasPrefix(line, "STRENGTHS:") {
			currentSection = "strengths"
			continue
		} else if strings.HasPrefix(line, "WEAKNESSES:") {
			currentSection = "weaknesses"
			continue
		} else if strings.HasPrefix(line, "SUGGESTIONS:") {
			currentSection = "suggestions"
			continue
		} else if strings.HasPrefix(line, "CONFIDENCE:") {
			// Extract confidence score
			parts := strings.Split(line, ":")
			if len(parts) > 1 {
				confStr := strings.TrimSpace(parts[1])
				if _, err := fmt.Sscanf(confStr, "%f", &reflection.ConfidenceScore); err != nil {
					// Failed to parse confidence, use default
					reflection.ConfidenceScore = 0.5
				}
			}
			continue
		}

		// Add to appropriate section
		if strings.HasPrefix(line, "- ") {
			item := strings.TrimPrefix(line, "- ")
			switch currentSection {
			case "strengths":
				reflection.Strengths = append(reflection.Strengths, item)
			case "weaknesses":
				reflection.Weaknesses = append(reflection.Weaknesses, item)
			case "suggestions":
				reflection.Suggestions = append(reflection.Suggestions, item)
			}
		}
	}

	return reflection
}

// createFallbackReflection creates a basic reflection when LLM fails.
func (g *GEPA) createFallbackReflection(candidate *GEPACandidate, patterns *ExecutionPatterns) *ReflectionResult {
	strengths := []string{}
	weaknesses := []string{}
	suggestions := []string{}

	// Basic analysis based on patterns
	if patterns.SuccessRate > 0.7 {
		strengths = append(strengths, "High success rate indicates effective instruction")
	} else {
		weaknesses = append(weaknesses, "Low success rate suggests instruction needs improvement")
		suggestions = append(suggestions, "Consider making the instruction more specific and clear")
	}

	if patterns.AverageResponseTime < time.Second {
		strengths = append(strengths, "Fast execution time")
	} else {
		weaknesses = append(weaknesses, "Slow execution time")
		suggestions = append(suggestions, "Simplify the instruction to reduce processing time")
	}

	if len(patterns.CommonFailures) > 0 {
		weaknesses = append(weaknesses, fmt.Sprintf("Common failures: %s", strings.Join(patterns.CommonFailures, ", ")))
		suggestions = append(suggestions, "Address the common failure modes identified")
	}

	return &ReflectionResult{
		CandidateID:     candidate.ID,
		Strengths:       strengths,
		Weaknesses:      weaknesses,
		Suggestions:     suggestions,
		ConfidenceScore: 0.3, // Low confidence for fallback
		Timestamp:       time.Now(),
		ReflectionDepth: 1,
	}
}

// Helper methods for reflection analysis

// categorizeError categorizes an error for analysis.
func (g *GEPA) categorizeError(err error) string {
	errStr := err.Error()
	switch {
	case strings.Contains(errStr, "timeout"):
		return "timeout"
	case strings.Contains(errStr, "parse"):
		return "parsing_error"
	case strings.Contains(errStr, "format"):
		return "format_error"
	case strings.Contains(errStr, "invalid"):
		return "invalid_input"
	default:
		return "other"
	}
}

// identifyCommonFailures identifies the most common failure modes.
func (g *GEPA) identifyCommonFailures(errorDist map[string]int) []string {
	if len(errorDist) == 0 {
		return []string{}
	}

	// Sort by frequency
	type errorCount struct {
		errorType string
		count     int
	}

	errors := make([]errorCount, 0, len(errorDist))
	for errorType, count := range errorDist {
		errors = append(errors, errorCount{errorType, count})
	}

	// Simple sort by count (descending)
	for i := 0; i < len(errors); i++ {
		for j := i + 1; j < len(errors); j++ {
			if errors[j].count > errors[i].count {
				errors[i], errors[j] = errors[j], errors[i]
			}
		}
	}

	// Return top 3 most common failures
	result := make([]string, 0, 3)
	for i, err := range errors {
		if i >= 3 {
			break
		}
		result = append(result, err.errorType)
	}

	return result
}

// identifyQualityIndicators identifies quality indicators from performance trends.
func (g *GEPA) identifyQualityIndicators(trends []float64) []string {
	if len(trends) == 0 {
		return []string{}
	}

	indicators := make([]string, 0)

	// Calculate average performance
	sum := 0.0
	for _, score := range trends {
		sum += score
	}
	avg := sum / float64(len(trends))

	if avg > 0.8 {
		indicators = append(indicators, "consistently_high_performance")
	} else if avg < 0.3 {
		indicators = append(indicators, "consistently_low_performance")
	} else {
		indicators = append(indicators, "variable_performance")
	}

	// Check for improvement trend
	if len(trends) > 3 {
		firstHalf := trends[:len(trends)/2]
		secondHalf := trends[len(trends)/2:]

		firstAvg := 0.0
		for _, score := range firstHalf {
			firstAvg += score
		}
		firstAvg /= float64(len(firstHalf))

		secondAvg := 0.0
		for _, score := range secondHalf {
			secondAvg += score
		}
		secondAvg /= float64(len(secondHalf))

		if secondAvg > firstAvg+0.1 {
			indicators = append(indicators, "improving_over_time")
		} else if secondAvg < firstAvg-0.1 {
			indicators = append(indicators, "degrading_over_time")
		}
	}

	return indicators
}

// Convergence and Final Application Methods

// hasConverged checks if the optimization has converged.
func (g *GEPA) hasConverged() bool {
	g.state.mu.RLock()
	defer g.state.mu.RUnlock()

	if g.state.ConvergenceStatus.IsConverged {
		return true
	}

	// Check stagnation
	timeSinceImprovement := time.Since(g.state.LastImprovement)
	if timeSinceImprovement > time.Duration(g.config.StagnationLimit)*time.Minute {
		g.state.ConvergenceStatus.IsConverged = true
		g.state.ConvergenceStatus.StagnationCount = g.config.StagnationLimit
		return true
	}

	// Check if fitness improvement is below threshold
	if len(g.state.PopulationHistory) >= 2 {
		current := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
		previous := g.state.PopulationHistory[len(g.state.PopulationHistory)-2]

		improvement := current.BestFitness - previous.BestFitness
		if improvement < g.config.ConvergenceThreshold {
			g.state.ConvergenceStatus.StagnationCount++
			if g.state.ConvergenceStatus.StagnationCount >= g.config.StagnationLimit {
				g.state.ConvergenceStatus.IsConverged = true
				return true
			}
		} else {
			g.state.ConvergenceStatus.StagnationCount = 0
		}
	}

	return false
}

// applyBestCandidate applies the best candidate to the final program.
func (g *GEPA) applyBestCandidate(program core.Program) core.Program {
	if g.state.BestCandidate == nil {
		logging.GetLogger().Warn(context.Background(), "No best candidate found, returning original program")
		return program
	}

	logger := logging.GetLogger()
	logger.Info(context.Background(), "Applying best candidate to program: id=%s, fitness=%.3f, instruction=%s",
		g.state.BestCandidate.ID,
		g.state.BestCandidate.Fitness,
		g.state.BestCandidate.Instruction)

	return g.applyCandidate(program, g.state.BestCandidate)
}

// LLM-based Self-Critique Implementation

// selfCritiqueCandidate performs LLM-based self-critique on a prompt candidate.
func (g *GEPA) selfCritiqueCandidate(ctx context.Context, candidate *GEPACandidate) (*ReflectionResult, error) {
	// Gather execution data for the candidate
	traces := g.state.GetTracesForCandidate(candidate.ID)
	if len(traces) == 0 {
		return nil, fmt.Errorf("no execution traces found for candidate %s", candidate.ID)
	}

	// Build comprehensive critique prompt
	critiquePrompt := g.buildCritiquePrompt(candidate, traces)

	// Use reflection LLM for critique
	critiqueResponse, err := g.reflectionLLM.Generate(ctx, critiquePrompt, core.WithTemperature(g.config.SelfCritiqueTemp))
	if err != nil {
		return nil, fmt.Errorf("failed to generate self-critique: %w", err)
	}

	// Parse critique response into structured reflection result
	reflectionResult := g.parseCritiqueResponse(candidate.ID, critiqueResponse.Content)

	// Store reflection result
	g.state.ReflectionHistory = append(g.state.ReflectionHistory, reflectionResult)

	return reflectionResult, nil
}

// buildCritiquePrompt constructs a comprehensive prompt for LLM-based self-critique.
func (g *GEPA) buildCritiquePrompt(candidate *GEPACandidate, traces []ExecutionTrace) string {
	var promptBuilder strings.Builder

	// Header
	promptBuilder.WriteString("# Prompt Candidate Analysis and Critique\n\n")

	// Candidate information
	promptBuilder.WriteString("## Candidate Information\n")
	promptBuilder.WriteString(fmt.Sprintf("- **ID**: %s\n", candidate.ID))
	promptBuilder.WriteString(fmt.Sprintf("- **Generation**: %d\n", candidate.Generation))
	promptBuilder.WriteString(fmt.Sprintf("- **Current Fitness**: %.3f\n", candidate.Fitness))
	promptBuilder.WriteString(fmt.Sprintf("- **Module**: %s\n", candidate.ModuleName))
	promptBuilder.WriteString(fmt.Sprintf("- **Created**: %s\n\n", candidate.CreatedAt.Format("2006-01-02 15:04:05")))

	// Current instruction
	promptBuilder.WriteString("## Current Instruction\n")
	promptBuilder.WriteString("```\n")
	promptBuilder.WriteString(candidate.Instruction)
	promptBuilder.WriteString("\n```\n\n")

	// Execution performance analysis
	promptBuilder.WriteString("## Execution Performance Analysis\n")
	g.addPerformanceAnalysis(&promptBuilder, traces)

	// Pattern analysis
	promptBuilder.WriteString("## Observed Patterns\n")
	g.addPatternAnalysis(&promptBuilder, traces)

	// Comparative analysis (if parent candidates exist)
	if len(candidate.ParentIDs) > 0 {
		promptBuilder.WriteString("## Comparative Analysis\n")
		g.addComparativeAnalysis(&promptBuilder, candidate)
	}

	// Critique request
	promptBuilder.WriteString("## Critique Request\n")
	promptBuilder.WriteString("Based on the above analysis, please provide a comprehensive critique of this prompt candidate. ")
	promptBuilder.WriteString("Focus on the following aspects:\n\n")
	promptBuilder.WriteString("1. **Strengths**: What does this prompt do well? What patterns lead to success?\n")
	promptBuilder.WriteString("2. **Weaknesses**: What are the main limitations? Where does it fail?\n")
	promptBuilder.WriteString("3. **Specific Improvements**: Concrete suggestions for enhancing the prompt\n")
	promptBuilder.WriteString("4. **Confidence Assessment**: How confident are you in this analysis? (0-1 scale)\n")
	promptBuilder.WriteString("5. **Success Predictions**: What types of inputs/contexts will this prompt handle well or poorly?\n\n")

	promptBuilder.WriteString("Please structure your response as follows:\n")
	promptBuilder.WriteString("**STRENGTHS:**\n[List 2-4 key strengths]\n\n")
	promptBuilder.WriteString("**WEAKNESSES:**\n[List 2-4 key weaknesses]\n\n")
	promptBuilder.WriteString("**IMPROVEMENTS:**\n[List 3-5 specific improvement suggestions]\n\n")
	promptBuilder.WriteString("**CONFIDENCE:** [0.0-1.0]\n\n")
	promptBuilder.WriteString("**PREDICTION:** [Description of expected performance patterns]\n")

	return promptBuilder.String()
}

// addPerformanceAnalysis adds performance metrics to the critique prompt.
func (g *GEPA) addPerformanceAnalysis(builder *strings.Builder, traces []ExecutionTrace) {
	if len(traces) == 0 {
		builder.WriteString("No execution data available.\n\n")
		return
	}

	// Calculate performance metrics
	successCount := 0
	totalDuration := time.Duration(0)
	errorTypes := make(map[string]int)

	for _, trace := range traces {
		if trace.Success {
			successCount++
		} else if trace.Error != nil {
			errorType := g.categorizeError(trace.Error)
			errorTypes[errorType]++
		}
		totalDuration += trace.Duration
	}

	successRate := float64(successCount) / float64(len(traces))
	avgDuration := totalDuration / time.Duration(len(traces))

	fmt.Fprintf(builder, "- **Total Executions**: %d\n", len(traces))
	fmt.Fprintf(builder, "- **Success Rate**: %.2f%% (%d/%d)\n", successRate*100, successCount, len(traces))
	fmt.Fprintf(builder, "- **Average Execution Time**: %v\n", avgDuration)

	if len(errorTypes) > 0 {
		builder.WriteString("- **Error Distribution**:\n")
		for errorType, count := range errorTypes {
			percentage := float64(count) / float64(len(traces)) * 100
			fmt.Fprintf(builder, "  - %s: %d occurrences (%.1f%%)\n", errorType, count, percentage)
		}
	}

	builder.WriteString("\n")
}

// addPatternAnalysis adds pattern analysis to the critique prompt.
func (g *GEPA) addPatternAnalysis(builder *strings.Builder, traces []ExecutionTrace) {
	if len(traces) == 0 {
		return
	}

	// Analyze input patterns
	inputPatterns := g.analyzeInputPatterns(traces)
	if len(inputPatterns) > 0 {
		builder.WriteString("### Input Patterns\n")
		for pattern, info := range inputPatterns {
			fmt.Fprintf(builder, "- **%s**: %s\n", pattern, info)
		}
		builder.WriteString("\n")
	}

	// Analyze output patterns
	outputPatterns := g.analyzeOutputPatterns(traces)
	if len(outputPatterns) > 0 {
		builder.WriteString("### Output Patterns\n")
		for pattern, info := range outputPatterns {
			fmt.Fprintf(builder, "- **%s**: %s\n", pattern, info)
		}
		builder.WriteString("\n")
	}

	// Analyze failure patterns
	failurePatterns := g.analyzeFailurePatterns(traces)
	if len(failurePatterns) > 0 {
		builder.WriteString("### Failure Patterns\n")
		for pattern, info := range failurePatterns {
			fmt.Fprintf(builder, "- **%s**: %s\n", pattern, info)
		}
		builder.WriteString("\n")
	}
}

// addComparativeAnalysis adds comparative analysis with parent candidates.
func (g *GEPA) addComparativeAnalysis(builder *strings.Builder, candidate *GEPACandidate) {
	fmt.Fprintf(builder, "This candidate evolved from %d parent(s):\n", len(candidate.ParentIDs))

	for i, parentID := range candidate.ParentIDs {
		// Find parent candidate in population history
		parentCandidate := g.findCandidateInHistory(parentID)
		if parentCandidate != nil {
			fmt.Fprintf(builder, "### Parent %d (ID: %s)\n", i+1, parentID)
			fmt.Fprintf(builder, "- **Fitness**: %.3f (vs current: %.3f)\n",
				parentCandidate.Fitness, candidate.Fitness)
			fmt.Fprintf(builder, "- **Generation**: %d (vs current: %d)\n",
				parentCandidate.Generation, candidate.Generation)

			// Show key differences in instruction
			differences := g.compareInstructions(parentCandidate.Instruction, candidate.Instruction)
			if differences != "" {
				fmt.Fprintf(builder, "- **Key Changes**: %s\n", differences)
			}
		} else {
			fmt.Fprintf(builder, "### Parent %d (ID: %s) - Data not available\n", i+1, parentID)
		}
	}

	builder.WriteString("\n")
}

// parseCritiqueResponse parses the LLM critique response into a structured ReflectionResult.
func (g *GEPA) parseCritiqueResponse(candidateID, response string) *ReflectionResult {
	result := &ReflectionResult{
		CandidateID:     candidateID,
		Strengths:       make([]string, 0),
		Weaknesses:      make([]string, 0),
		Suggestions:     make([]string, 0),
		ConfidenceScore: 0.5, // Default confidence
		Timestamp:       time.Now(),
		ReflectionDepth: 1,
	}

	lines := strings.Split(response, "\n")
	currentSection := ""

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Detect section headers
		upperLine := strings.ToUpper(line)
		if strings.Contains(upperLine, "STRENGTHS:") {
			currentSection = "strengths"
			continue
		} else if strings.Contains(upperLine, "WEAKNESSES:") {
			currentSection = "weaknesses"
			continue
		} else if strings.Contains(upperLine, "IMPROVEMENTS:") {
			currentSection = "suggestions"
			continue
		} else if strings.Contains(upperLine, "CONFIDENCE:") {
			currentSection = "confidence"
			// Try to extract confidence value
			if idx := strings.Index(upperLine, "CONFIDENCE:"); idx != -1 {
				confStr := strings.TrimSpace(line[idx+11:])
				if conf, err := strconv.ParseFloat(confStr, 64); err == nil {
					result.ConfidenceScore = conf
				}
			}
			continue
		} else if strings.Contains(upperLine, "PREDICTION:") {
			currentSection = "prediction"
			continue
		}

		// Process content based on current section
		switch currentSection {
		case "strengths":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
				strength := strings.TrimSpace(line[1:])
				if strength != "" {
					result.Strengths = append(result.Strengths, strength)
				}
			}
		case "weaknesses":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
				weakness := strings.TrimSpace(line[1:])
				if weakness != "" {
					result.Weaknesses = append(result.Weaknesses, weakness)
				}
			}
		case "suggestions":
			if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
				suggestion := strings.TrimSpace(line[1:])
				if suggestion != "" {
					result.Suggestions = append(result.Suggestions, suggestion)
				}
			}
		}
	}

	// Ensure minimum content
	if len(result.Strengths) == 0 {
		result.Strengths = []string{"Unable to identify specific strengths from analysis"}
	}
	if len(result.Weaknesses) == 0 {
		result.Weaknesses = []string{"Unable to identify specific weaknesses from analysis"}
	}
	if len(result.Suggestions) == 0 {
		result.Suggestions = []string{"Insufficient data for specific improvement suggestions"}
	}

	return result
}

// Helper methods for pattern analysis

// analyzeInputPatterns analyzes patterns in input data across executions.
func (g *GEPA) analyzeInputPatterns(traces []ExecutionTrace) map[string]string {
	patterns := make(map[string]string)

	if len(traces) == 0 {
		return patterns
	}

	// Analyze input types and sizes
	inputTypeCounts := make(map[string]int)
	inputSizes := make([]int, 0)

	for _, trace := range traces {
		if trace.Inputs != nil {
			inputSizes = append(inputSizes, len(trace.Inputs))
			for _, value := range trace.Inputs {
				if value != nil {
					inputTypeCounts[fmt.Sprintf("%T", value)]++
				}
			}
		}
	}

	// Most common input types
	if len(inputTypeCounts) > 0 {
		maxCount := 0
		mostCommonType := ""
		for inputType, count := range inputTypeCounts {
			if count > maxCount {
				maxCount = count
				mostCommonType = inputType
			}
		}
		patterns["Most Common Input Type"] = fmt.Sprintf("%s (%d/%d executions)",
			mostCommonType, maxCount, len(traces))
	}

	// Input size patterns
	if len(inputSizes) > 0 {
		sum := 0
		min, max := inputSizes[0], inputSizes[0]
		for _, size := range inputSizes {
			sum += size
			if size < min {
				min = size
			}
			if size > max {
				max = size
			}
		}
		avg := float64(sum) / float64(len(inputSizes))
		patterns["Input Size Range"] = fmt.Sprintf("Min: %d, Max: %d, Avg: %.1f", min, max, avg)
	}

	return patterns
}

// analyzeOutputPatterns analyzes patterns in output data across executions.
func (g *GEPA) analyzeOutputPatterns(traces []ExecutionTrace) map[string]string {
	patterns := make(map[string]string)

	successfulTraces := make([]ExecutionTrace, 0)
	for _, trace := range traces {
		if trace.Success && trace.Outputs != nil {
			successfulTraces = append(successfulTraces, trace)
		}
	}

	if len(successfulTraces) == 0 {
		patterns["Success Pattern"] = "No successful executions to analyze"
		return patterns
	}

	// Analyze output types and sizes
	outputTypeCounts := make(map[string]int)
	outputSizes := make([]int, 0)

	for _, trace := range successfulTraces {
		outputSizes = append(outputSizes, len(trace.Outputs))
		for _, value := range trace.Outputs {
			if value != nil {
				outputTypeCounts[fmt.Sprintf("%T", value)]++
			}
		}
	}

	// Most common output types
	if len(outputTypeCounts) > 0 {
		maxCount := 0
		mostCommonType := ""
		for outputType, count := range outputTypeCounts {
			if count > maxCount {
				maxCount = count
				mostCommonType = outputType
			}
		}
		patterns["Most Common Output Type"] = fmt.Sprintf("%s (%d/%d successful executions)",
			mostCommonType, maxCount, len(successfulTraces))
	}

	// Output consistency
	if len(outputSizes) > 1 {
		consistent := true
		firstSize := outputSizes[0]
		for _, size := range outputSizes[1:] {
			if size != firstSize {
				consistent = false
				break
			}
		}
		if consistent {
			patterns["Output Consistency"] = fmt.Sprintf("Consistent output size: %d fields", firstSize)
		} else {
			sum := 0
			for _, size := range outputSizes {
				sum += size
			}
			avg := float64(sum) / float64(len(outputSizes))
			patterns["Output Consistency"] = fmt.Sprintf("Variable output size, avg: %.1f fields", avg)
		}
	}

	return patterns
}

// analyzeFailurePatterns analyzes patterns in failed executions.
func (g *GEPA) analyzeFailurePatterns(traces []ExecutionTrace) map[string]string {
	patterns := make(map[string]string)

	failedTraces := make([]ExecutionTrace, 0)
	for _, trace := range traces {
		if !trace.Success {
			failedTraces = append(failedTraces, trace)
		}
	}

	if len(failedTraces) == 0 {
		return patterns
	}

	// Analyze error types
	errorTypeCounts := make(map[string]int)
	for _, trace := range failedTraces {
		if trace.Error != nil {
			errorType := g.categorizeError(trace.Error)
			errorTypeCounts[errorType]++
		}
	}

	if len(errorTypeCounts) > 0 {
		maxCount := 0
		mostCommonError := ""
		for errorType, count := range errorTypeCounts {
			if count > maxCount {
				maxCount = count
				mostCommonError = errorType
			}
		}
		patterns["Most Common Failure"] = fmt.Sprintf("%s (%d/%d failures)",
			mostCommonError, maxCount, len(failedTraces))
	}

	// Failure rate analysis
	failureRate := float64(len(failedTraces)) / float64(len(traces)) * 100
	patterns["Failure Rate"] = fmt.Sprintf("%.1f%% (%d/%d executions)",
		failureRate, len(failedTraces), len(traces))

	return patterns
}

// findCandidateInHistory finds a candidate by ID in population history.
func (g *GEPA) findCandidateInHistory(candidateID string) *GEPACandidate {
	for _, population := range g.state.PopulationHistory {
		for _, candidate := range population.Candidates {
			if candidate.ID == candidateID {
				return candidate
			}
		}
	}
	return nil
}

// compareInstructions compares two instructions and returns key differences.
func (g *GEPA) compareInstructions(oldInstr, newInstr string) string {
	if oldInstr == newInstr {
		return "No changes"
	}

	oldWords := strings.Fields(strings.ToLower(oldInstr))
	newWords := strings.Fields(strings.ToLower(newInstr))

	// Simple difference detection
	if len(newWords) > len(oldWords) {
		return fmt.Sprintf("Expanded by %d words", len(newWords)-len(oldWords))
	} else if len(newWords) < len(oldWords) {
		return fmt.Sprintf("Shortened by %d words", len(oldWords)-len(newWords))
	} else {
		return "Modified with similar length"
	}
}

// batchSelfCritique performs self-critique on multiple candidates.
func (g *GEPA) batchSelfCritique(ctx context.Context, candidates []*GEPACandidate) ([]*ReflectionResult, error) {
	results := make([]*ReflectionResult, 0, len(candidates))

	// Use concurrent processing for efficiency
	p := pool.New().WithMaxGoroutines(g.config.ConcurrencyLevel)
	resultChan := make(chan *ReflectionResult, len(candidates))
	errorChan := make(chan error, len(candidates))

	for _, candidate := range candidates {
		candidate := candidate // Capture loop variable
		p.Go(func() {
			result, err := g.selfCritiqueCandidate(ctx, candidate)
			if err != nil {
				errorChan <- err
				return
			}
			resultChan <- result
		})
	}

	p.Wait()
	close(resultChan)
	close(errorChan)

	// Collect results
	for result := range resultChan {
		results = append(results, result)
	}

	// Check for errors
	var errs []error
	for err := range errorChan {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return results, fmt.Errorf("encountered %d errors during batch critique", len(errs))
	}

	return results, nil
}

// Multi-Level Reflection Implementation

// performMultiLevelReflection performs reflection at both individual and population levels.
func (g *GEPA) performMultiLevelReflection(ctx context.Context) error {
	logger := logging.GetLogger()
	logger.Info(ctx, "Performing multi-level reflection for generation %d", g.state.CurrentGeneration)

	// Level 1: Individual Candidate Reflection
	if err := g.performIndividualReflection(ctx); err != nil {
		logger.Warn(ctx, "Individual reflection failed: %v", err)
	}

	// Level 2: Population-Level Reflection
	if err := g.performPopulationReflection(ctx); err != nil {
		logger.Warn(ctx, "Population reflection failed: %v", err)
	}

	// Level 3: Cross-Generation Reflection
	if err := g.performCrossGenerationReflection(ctx); err != nil {
		logger.Warn(ctx, "Cross-generation reflection failed: %v", err)
	}

	// Level 4: Meta-Reflection (reflection on reflection)
	if g.state.CurrentGeneration > 0 && len(g.state.ReflectionHistory) > 5 {
		if err := g.performMetaReflection(ctx); err != nil {
			logger.Warn(ctx, "Meta-reflection failed: %v", err)
		}
	}

	logger.Info(ctx, "Multi-level reflection completed")
	return nil
}

// performIndividualReflection performs reflection on individual candidates.
func (g *GEPA) performIndividualReflection(ctx context.Context) error {
	if len(g.state.PopulationHistory) == 0 {
		return fmt.Errorf("no population history available for reflection")
	}

	currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]

	// Select candidates for individual reflection based on performance and diversity
	candidatesForReflection := g.selectCandidatesForReflection(currentPop)

	// Perform self-critique on selected candidates
	reflectionResults, err := g.batchSelfCritique(ctx, candidatesForReflection)
	if err != nil {
		return fmt.Errorf("failed to perform individual reflection: %w", err)
	}

	// Analyze reflection results for insights
	insights := g.analyzeIndividualReflectionResults(reflectionResults)

	// Store insights in state metadata
	if g.state.BestCandidate != nil && g.state.BestCandidate.Metadata == nil {
		g.state.BestCandidate.Metadata = make(map[string]interface{})
	}
	if g.state.BestCandidate != nil {
		g.state.BestCandidate.Metadata["individual_reflection_insights"] = insights
	}

	logger := logging.GetLogger()
	insightsCount := len(insights.CommonStrengths) + len(insights.CommonWeaknesses) +
		len(insights.SuccessPatterns) + len(insights.FailurePatterns) + len(insights.ImprovementThemes)
	logger.Debug(ctx, "Individual reflection completed: analyzed %d candidates, extracted %d insights",
		len(candidatesForReflection), insightsCount)

	return nil
}

// performPopulationReflection performs reflection on the entire population.
func (g *GEPA) performPopulationReflection(ctx context.Context) error {
	if len(g.state.PopulationHistory) == 0 {
		return fmt.Errorf("no population history available for reflection")
	}

	// Analyze population-level patterns
	populationInsights := g.analyzePopulationPatterns()

	// Generate population-level critique using LLM
	populationCritique, err := g.generatePopulationCritique(ctx, populationInsights)
	if err != nil {
		return fmt.Errorf("failed to generate population critique: %w", err)
	}

	// Extract actionable insights from the critique
	actionableInsights := g.extractActionableInsights(populationCritique)

	// Store population reflection results
	populationReflection := &ReflectionResult{
		CandidateID:     "population",
		Strengths:       actionableInsights.Strengths,
		Weaknesses:      actionableInsights.Weaknesses,
		Suggestions:     actionableInsights.Suggestions,
		ConfidenceScore: actionableInsights.Confidence,
		Timestamp:       time.Now(),
		ReflectionDepth: 2, // Population level
	}

	g.state.ReflectionHistory = append(g.state.ReflectionHistory, populationReflection)

	logger := logging.GetLogger()
	logger.Debug(ctx, "Population reflection completed: identified %d strengths, %d weaknesses, %d suggestions",
		len(actionableInsights.Strengths),
		len(actionableInsights.Weaknesses),
		len(actionableInsights.Suggestions))

	return nil
}

// performCrossGenerationReflection performs reflection across multiple generations.
func (g *GEPA) performCrossGenerationReflection(ctx context.Context) error {
	if len(g.state.PopulationHistory) < 2 {
		return fmt.Errorf("insufficient generation history for cross-generation reflection")
	}

	// Analyze evolutionary trends across generations
	evolutionaryTrends := g.analyzeEvolutionaryTrends()

	// Generate cross-generation insights
	crossGenInsights, err := g.generateCrossGenerationInsights(ctx, evolutionaryTrends)
	if err != nil {
		return fmt.Errorf("failed to generate cross-generation insights: %w", err)
	}

	// Update convergence status based on cross-generation analysis
	g.updateConvergenceStatusFromReflection(crossGenInsights)

	// Store cross-generation reflection
	crossGenReflection := &ReflectionResult{
		CandidateID:     "cross_generation",
		Strengths:       crossGenInsights.PositiveTrends,
		Weaknesses:      crossGenInsights.NegativeTrends,
		Suggestions:     crossGenInsights.Recommendations,
		ConfidenceScore: crossGenInsights.TrendConfidence,
		Timestamp:       time.Now(),
		ReflectionDepth: 3, // Cross-generation level
	}

	g.state.ReflectionHistory = append(g.state.ReflectionHistory, crossGenReflection)

	logger := logging.GetLogger()
	logger.Debug(ctx, "Cross-generation reflection completed: analyzed %d generations, confidence: %.2f",
		len(g.state.PopulationHistory), crossGenInsights.TrendConfidence)

	return nil
}

// performMetaReflection performs reflection on the reflection process itself.
func (g *GEPA) performMetaReflection(ctx context.Context) error {
	if len(g.state.ReflectionHistory) < 3 {
		return fmt.Errorf("insufficient reflection history for meta-reflection")
	}

	// Analyze the quality and accuracy of previous reflections
	reflectionQuality := g.analyzeReflectionQuality()

	// Generate meta-insights about the reflection process
	metaInsights, err := g.generateMetaReflectionInsights(ctx, reflectionQuality)
	if err != nil {
		return fmt.Errorf("failed to generate meta-reflection insights: %w", err)
	}

	// Adjust reflection parameters based on meta-insights
	g.adjustReflectionParameters(metaInsights)

	// Store meta-reflection results
	metaReflection := &ReflectionResult{
		CandidateID:     "meta_reflection",
		Strengths:       metaInsights.ReflectionStrengths,
		Weaknesses:      metaInsights.ReflectionWeaknesses,
		Suggestions:     metaInsights.ProcessImprovements,
		ConfidenceScore: metaInsights.MetaConfidence,
		Timestamp:       time.Now(),
		ReflectionDepth: 4, // Meta-reflection level
	}

	g.state.ReflectionHistory = append(g.state.ReflectionHistory, metaReflection)

	logger := logging.GetLogger()
	logger.Debug(ctx, "Meta-reflection completed: reflection quality score: %.2f, adjustments made: %d",
		reflectionQuality.OverallScore, len(metaInsights.ProcessImprovements))

	return nil
}

// Supporting structures for multi-level reflection

type IndividualReflectionInsights struct {
	CommonStrengths        []string       `json:"common_strengths"`
	CommonWeaknesses       []string       `json:"common_weaknesses"`
	SuccessPatterns        []string       `json:"success_patterns"`
	FailurePatterns        []string       `json:"failure_patterns"`
	ImprovementThemes      []string       `json:"improvement_themes"`
	ConfidenceDistribution map[string]int `json:"confidence_distribution"`
}

type PopulationCritique struct {
	OverallHealth       string   `json:"overall_health"`
	DiversityAssessment string   `json:"diversity_assessment"`
	PerformanceTrends   []string `json:"performance_trends"`
	RecommendedActions  []string `json:"recommended_actions"`
	CritiqueText        string   `json:"critique_text"`
}

type ActionableInsights struct {
	Strengths   []string `json:"strengths"`
	Weaknesses  []string `json:"weaknesses"`
	Suggestions []string `json:"suggestions"`
	Confidence  float64  `json:"confidence"`
}

type EvolutionaryTrends struct {
	FitnessProgression    []float64 `json:"fitness_progression"`
	DiversityTrend        []float64 `json:"diversity_trend"`
	ConvergenceIndicators []string  `json:"convergence_indicators"`
	StagnationPeriods     []int     `json:"stagnation_periods"`
	BreakthroughMoments   []int     `json:"breakthrough_moments"`
}

type CrossGenerationInsights struct {
	PositiveTrends   []string `json:"positive_trends"`
	NegativeTrends   []string `json:"negative_trends"`
	Recommendations  []string `json:"recommendations"`
	TrendConfidence  float64  `json:"trend_confidence"`
	PredictedOutcome string   `json:"predicted_outcome"`
}

type ReflectionQuality struct {
	OverallScore           float64            `json:"overall_score"`
	AccuracyScores         map[string]float64 `json:"accuracy_scores"`
	Usefulness             float64            `json:"usefulness"`
	ConsistencyScore       float64            `json:"consistency_score"`
	ImprovementSuggestions []string           `json:"improvement_suggestions"`
}

type MetaReflectionInsights struct {
	ReflectionStrengths  []string           `json:"reflection_strengths"`
	ReflectionWeaknesses []string           `json:"reflection_weaknesses"`
	ProcessImprovements  []string           `json:"process_improvements"`
	MetaConfidence       float64            `json:"meta_confidence"`
	ParameterAdjustments map[string]float64 `json:"parameter_adjustments"`
}

// Implementation of helper methods for multi-level reflection

// selectCandidatesForReflection selects candidates for individual reflection.
func (g *GEPA) selectCandidatesForReflection(population *Population) []*GEPACandidate {
	if population == nil || len(population.Candidates) == 0 {
		return []*GEPACandidate{}
	}

	selected := make([]*GEPACandidate, 0)

	// Always include the best candidate
	if population.BestCandidate != nil {
		selected = append(selected, population.BestCandidate)
	}

	// Include diverse candidates for comprehensive reflection
	diverseCandidates := g.selectDiverseCandidates(population.Candidates, 3)
	for _, candidate := range diverseCandidates {
		// Avoid duplicates
		if population.BestCandidate == nil || candidate.ID != population.BestCandidate.ID {
			selected = append(selected, candidate)
		}
	}

	// Include candidates with interesting performance patterns
	interestingCandidates := g.selectInterestingCandidates(population.Candidates, 2)
	for _, candidate := range interestingCandidates {
		// Check for duplicates
		isDuplicate := false
		for _, existing := range selected {
			if existing.ID == candidate.ID {
				isDuplicate = true
				break
			}
		}
		if !isDuplicate {
			selected = append(selected, candidate)
		}
	}

	return selected
}

// selectDiverseCandidates selects candidates with diverse characteristics.
func (g *GEPA) selectDiverseCandidates(candidates []*GEPACandidate, count int) []*GEPACandidate {
	if len(candidates) <= count {
		return candidates
	}

	selected := make([]*GEPACandidate, 0, count)
	remaining := make([]*GEPACandidate, len(candidates))
	copy(remaining, candidates)

	// Select first candidate randomly
	if len(remaining) > 0 {
		idx := g.rng.Intn(len(remaining))
		selected = append(selected, remaining[idx])
		remaining = append(remaining[:idx], remaining[idx+1:]...)
	}

	// Select remaining candidates based on maximum diversity
	for len(selected) < count && len(remaining) > 0 {
		maxDiversity := -1.0
		var mostDiverse *GEPACandidate
		var mostDiverseIdx int

		for i, candidate := range remaining {
			minSimilarity := 1.0
			for _, selectedCandidate := range selected {
				similarity := g.calculateInstructionSimilarity(candidate.Instruction, selectedCandidate.Instruction)
				if similarity < minSimilarity {
					minSimilarity = similarity
				}
			}
			diversity := 1.0 - minSimilarity
			if diversity > maxDiversity {
				maxDiversity = diversity
				mostDiverse = candidate
				mostDiverseIdx = i
			}
		}

		if mostDiverse != nil {
			selected = append(selected, mostDiverse)
			remaining = append(remaining[:mostDiverseIdx], remaining[mostDiverseIdx+1:]...)
		} else {
			break
		}
	}

	return selected
}

// selectInterestingCandidates selects candidates with interesting performance patterns.
func (g *GEPA) selectInterestingCandidates(candidates []*GEPACandidate, count int) []*GEPACandidate {
	if len(candidates) <= count {
		return candidates
	}

	// Score candidates based on "interestingness"
	type candidateScore struct {
		candidate *GEPACandidate
		score     float64
	}

	scores := make([]candidateScore, 0, len(candidates))

	for _, candidate := range candidates {
		score := 0.0

		// Factors that make a candidate interesting:

		// 1. Significant fitness improvement over parents
		if len(candidate.ParentIDs) > 0 {
			avgParentFitness := 0.0
			parentCount := 0
			for _, parentID := range candidate.ParentIDs {
				if parent := g.findCandidateInHistory(parentID); parent != nil {
					avgParentFitness += parent.Fitness
					parentCount++
				}
			}
			if parentCount > 0 {
				avgParentFitness /= float64(parentCount)
				if candidate.Fitness > avgParentFitness {
					score += (candidate.Fitness - avgParentFitness) * 2.0
				}
			}
		}

		// 2. Unique instruction patterns
		uniqueness := g.assessDiversityContribution(candidate.ID)
		score += uniqueness

		// 3. High or low performance (outliers are interesting)
		if candidate.Fitness > 0.8 || candidate.Fitness < 0.2 {
			score += 0.5
		}

		// 4. Rich execution history
		traces := g.state.GetTracesForCandidate(candidate.ID)
		if len(traces) > 5 {
			score += 0.3
		}

		scores = append(scores, candidateScore{candidate: candidate, score: score})
	}

	// Sort by score (descending)
	for i := 0; i < len(scores)-1; i++ {
		for j := 0; j < len(scores)-1-i; j++ {
			if scores[j].score < scores[j+1].score {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	// Return top candidates
	result := make([]*GEPACandidate, 0, count)
	for i := 0; i < count && i < len(scores); i++ {
		result = append(result, scores[i].candidate)
	}

	return result
}

// analyzeIndividualReflectionResults analyzes results from individual candidate reflections.
func (g *GEPA) analyzeIndividualReflectionResults(results []*ReflectionResult) *IndividualReflectionInsights {
	insights := &IndividualReflectionInsights{
		CommonStrengths:        make([]string, 0),
		CommonWeaknesses:       make([]string, 0),
		SuccessPatterns:        make([]string, 0),
		FailurePatterns:        make([]string, 0),
		ImprovementThemes:      make([]string, 0),
		ConfidenceDistribution: make(map[string]int),
	}

	if len(results) == 0 {
		return insights
	}

	// Aggregate strengths and weaknesses
	strengthCounts := make(map[string]int)
	weaknessCounts := make(map[string]int)
	suggestionCounts := make(map[string]int)

	for _, result := range results {
		// Count confidence distribution
		confBucket := "medium"
		if result.ConfidenceScore >= 0.8 {
			confBucket = "high"
		} else if result.ConfidenceScore <= 0.3 {
			confBucket = "low"
		}
		insights.ConfidenceDistribution[confBucket]++

		// Aggregate strengths
		for _, strength := range result.Strengths {
			strengthCounts[strength]++
		}

		// Aggregate weaknesses
		for _, weakness := range result.Weaknesses {
			weaknessCounts[weakness]++
		}

		// Aggregate suggestions
		for _, suggestion := range result.Suggestions {
			suggestionCounts[suggestion]++
		}
	}

	// Extract common patterns (appearing in > 25% of results)
	threshold := int(float64(len(results)) * 0.25)

	for strength, count := range strengthCounts {
		if count > threshold {
			insights.CommonStrengths = append(insights.CommonStrengths, strength)
		}
	}

	for weakness, count := range weaknessCounts {
		if count > threshold {
			insights.CommonWeaknesses = append(insights.CommonWeaknesses, weakness)
		}
	}

	for suggestion, count := range suggestionCounts {
		if count > threshold {
			insights.ImprovementThemes = append(insights.ImprovementThemes, suggestion)
		}
	}

	return insights
}

// analyzePopulationPatterns analyzes patterns across the entire population.
func (g *GEPA) analyzePopulationPatterns() *PopulationInsights {
	if len(g.state.PopulationHistory) == 0 {
		return &PopulationInsights{}
	}

	currentPop := g.state.PopulationHistory[len(g.state.PopulationHistory)-1]
	insights := &PopulationInsights{
		HighPerformingPatterns: make([]string, 0),
		CommonWeaknesses:       make([]string, 0),
	}

	if currentPop == nil || len(currentPop.Candidates) == 0 {
		return insights
	}

	// Calculate basic statistics
	var fitnessSum, minFitness, maxFitness float64
	minFitness = currentPop.Candidates[0].Fitness
	maxFitness = currentPop.Candidates[0].Fitness

	for _, candidate := range currentPop.Candidates {
		fitnessSum += candidate.Fitness
		if candidate.Fitness < minFitness {
			minFitness = candidate.Fitness
		}
		if candidate.Fitness > maxFitness {
			maxFitness = candidate.Fitness
		}
	}

	insights.AverageFitness = fitnessSum / float64(len(currentPop.Candidates))
	insights.BestFitness = maxFitness
	insights.WorstFitness = minFitness

	// Calculate fitness variance
	var varianceSum float64
	for _, candidate := range currentPop.Candidates {
		diff := candidate.Fitness - insights.AverageFitness
		varianceSum += diff * diff
	}
	insights.FitnessVariance = varianceSum / float64(len(currentPop.Candidates))

	// Calculate diversity index
	insights.DiversityIndex = g.calculatePopulationDiversity(currentPop)

	// Analyze high-performing patterns
	topPerformers := make([]*GEPACandidate, 0)
	performanceThreshold := insights.AverageFitness + (insights.FitnessVariance * 0.5)

	for _, candidate := range currentPop.Candidates {
		if candidate.Fitness >= performanceThreshold {
			topPerformers = append(topPerformers, candidate)
		}
	}

	if len(topPerformers) > 0 {
		commonPatterns := g.extractCommonPatterns(topPerformers)
		insights.HighPerformingPatterns = commonPatterns
	}

	return insights
}

// generatePopulationCritique generates LLM-based critique of the population.
func (g *GEPA) generatePopulationCritique(ctx context.Context, insights *PopulationInsights) (*PopulationCritique, error) {
	// Build population critique prompt
	prompt := g.buildPopulationCritiquePrompt(insights)

	// Generate critique using reflection LLM
	response, err := g.reflectionLLM.Generate(ctx, prompt, core.WithTemperature(g.config.SelfCritiqueTemp))
	if err != nil {
		return nil, fmt.Errorf("failed to generate population critique: %w", err)
	}

	// Parse critique response
	critique := g.parsePopulationCritique(response.Content)

	return critique, nil
}

// buildPopulationCritiquePrompt builds a prompt for population-level critique.
func (g *GEPA) buildPopulationCritiquePrompt(insights *PopulationInsights) string {
	var builder strings.Builder

	builder.WriteString("# Population-Level Analysis and Critique\n\n")
	builder.WriteString("## Population Statistics\n")
	builder.WriteString(fmt.Sprintf("- **Generation**: %d\n", g.state.CurrentGeneration))
	builder.WriteString(fmt.Sprintf("- **Population Size**: %d\n", len(g.state.PopulationHistory[len(g.state.PopulationHistory)-1].Candidates)))
	builder.WriteString(fmt.Sprintf("- **Average Fitness**: %.3f\n", insights.AverageFitness))
	builder.WriteString(fmt.Sprintf("- **Best Fitness**: %.3f\n", insights.BestFitness))
	builder.WriteString(fmt.Sprintf("- **Worst Fitness**: %.3f\n", insights.WorstFitness))
	builder.WriteString(fmt.Sprintf("- **Fitness Variance**: %.3f\n", insights.FitnessVariance))
	builder.WriteString(fmt.Sprintf("- **Diversity Index**: %.3f\n\n", insights.DiversityIndex))

	if len(insights.HighPerformingPatterns) > 0 {
		builder.WriteString("## High-Performing Patterns\n")
		for _, pattern := range insights.HighPerformingPatterns {
			builder.WriteString(fmt.Sprintf("- %s\n", pattern))
		}
		builder.WriteString("\n")
	}

	builder.WriteString("## Critique Request\n")
	builder.WriteString("Based on the population statistics and patterns above, provide a comprehensive critique of the current population. Address:\n\n")
	builder.WriteString("1. **Population Health**: Overall assessment of population fitness and diversity\n")
	builder.WriteString("2. **Evolutionary Progress**: Is the population making good progress toward optimization goals?\n")
	builder.WriteString("3. **Diversity Assessment**: Is there sufficient diversity for continued evolution?\n")
	builder.WriteString("4. **Potential Issues**: Any signs of premature convergence, stagnation, or other problems?\n")
	builder.WriteString("5. **Recommended Actions**: Specific actions to improve population performance\n\n")

	builder.WriteString("Please structure your response with clear sections and actionable insights.\n")

	return builder.String()
}

// parsePopulationCritique parses the LLM response into structured critique.
func (g *GEPA) parsePopulationCritique(response string) *PopulationCritique {
	critique := &PopulationCritique{
		PerformanceTrends:  make([]string, 0),
		RecommendedActions: make([]string, 0),
		CritiqueText:       response,
	}

	// Simple parsing - in practice, this would be more sophisticated
	lines := strings.Split(response, "\n")
	currentSection := ""

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Detect sections
		lowerLine := strings.ToLower(line)
		if strings.Contains(lowerLine, "population health") {
			currentSection = "health"
		} else if strings.Contains(lowerLine, "diversity") {
			currentSection = "diversity"
		} else if strings.Contains(lowerLine, "recommended") || strings.Contains(lowerLine, "actions") {
			currentSection = "actions"
		}

		// Extract content
		if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") {
			content := strings.TrimSpace(line[1:])
			if content != "" {
				switch currentSection {
				case "actions":
					critique.RecommendedActions = append(critique.RecommendedActions, content)
				default:
					critique.PerformanceTrends = append(critique.PerformanceTrends, content)
				}
			}
		}

		// Extract key assessments
		if strings.Contains(lowerLine, "healthy") || strings.Contains(lowerLine, "good") {
			critique.OverallHealth = "good"
		} else if strings.Contains(lowerLine, "poor") || strings.Contains(lowerLine, "concerning") {
			critique.OverallHealth = "poor"
		} else if critique.OverallHealth == "" {
			critique.OverallHealth = "moderate"
		}

		if strings.Contains(lowerLine, "diverse") || strings.Contains(lowerLine, "variety") {
			critique.DiversityAssessment = "good"
		} else if strings.Contains(lowerLine, "convergence") || strings.Contains(lowerLine, "similar") {
			critique.DiversityAssessment = "low"
		}
	}

	// Set defaults
	if critique.OverallHealth == "" {
		critique.OverallHealth = "moderate"
	}
	if critique.DiversityAssessment == "" {
		critique.DiversityAssessment = "moderate"
	}

	return critique
}

// extractActionableInsights extracts actionable insights from population critique.
func (g *GEPA) extractActionableInsights(critique *PopulationCritique) *ActionableInsights {
	insights := &ActionableInsights{
		Strengths:   make([]string, 0),
		Weaknesses:  make([]string, 0),
		Suggestions: critique.RecommendedActions,
		Confidence:  0.7, // Default confidence
	}

	// Extract strengths and weaknesses from performance trends
	for _, trend := range critique.PerformanceTrends {
		lowerTrend := strings.ToLower(trend)
		if strings.Contains(lowerTrend, "good") || strings.Contains(lowerTrend, "strong") ||
			strings.Contains(lowerTrend, "effective") || strings.Contains(lowerTrend, "successful") {
			insights.Strengths = append(insights.Strengths, trend)
		} else if strings.Contains(lowerTrend, "poor") || strings.Contains(lowerTrend, "weak") ||
			strings.Contains(lowerTrend, "problematic") || strings.Contains(lowerTrend, "concerning") {
			insights.Weaknesses = append(insights.Weaknesses, trend)
		}
	}

	// Adjust confidence based on assessment quality
	if critique.OverallHealth == "good" && critique.DiversityAssessment == "good" {
		insights.Confidence = 0.9
	} else if critique.OverallHealth == "poor" || critique.DiversityAssessment == "low" {
		insights.Confidence = 0.5
	}

	return insights
}

// Placeholder implementations for remaining methods (to be expanded as needed)

func (g *GEPA) analyzeEvolutionaryTrends() *EvolutionaryTrends {
	trends := &EvolutionaryTrends{
		FitnessProgression:    make([]float64, 0),
		DiversityTrend:        make([]float64, 0),
		ConvergenceIndicators: make([]string, 0),
		StagnationPeriods:     make([]int, 0),
		BreakthroughMoments:   make([]int, 0),
	}

	// Analyze fitness progression across generations
	for i, pop := range g.state.PopulationHistory {
		if pop.BestCandidate != nil {
			trends.FitnessProgression = append(trends.FitnessProgression, pop.BestCandidate.Fitness)
		}

		// Calculate diversity for this generation
		diversity := g.calculatePopulationDiversity(pop)
		trends.DiversityTrend = append(trends.DiversityTrend, diversity)

		// Detect breakthrough moments (significant fitness improvements)
		if i > 0 && len(trends.FitnessProgression) >= 2 {
			currentFitness := trends.FitnessProgression[len(trends.FitnessProgression)-1]
			prevFitness := trends.FitnessProgression[len(trends.FitnessProgression)-2]
			if currentFitness-prevFitness > 0.1 { // Significant improvement threshold
				trends.BreakthroughMoments = append(trends.BreakthroughMoments, i)
			}
		}
	}

	return trends
}

func (g *GEPA) generateCrossGenerationInsights(ctx context.Context, trends *EvolutionaryTrends) (*CrossGenerationInsights, error) {
	insights := &CrossGenerationInsights{
		PositiveTrends:   make([]string, 0),
		NegativeTrends:   make([]string, 0),
		Recommendations:  make([]string, 0),
		TrendConfidence:  0.7,
		PredictedOutcome: "continued_progress",
	}

	// Analyze fitness progression
	if len(trends.FitnessProgression) >= 2 {
		recentImprovement := trends.FitnessProgression[len(trends.FitnessProgression)-1] -
			trends.FitnessProgression[len(trends.FitnessProgression)-2]
		if recentImprovement > 0.05 {
			insights.PositiveTrends = append(insights.PositiveTrends, "Recent fitness improvement detected")
		} else if recentImprovement < -0.05 {
			insights.NegativeTrends = append(insights.NegativeTrends, "Recent fitness decline detected")
		}
	}

	// Analyze diversity trends
	if len(trends.DiversityTrend) >= 3 {
		recentDiversity := trends.DiversityTrend[len(trends.DiversityTrend)-1]
		if recentDiversity < 0.2 {
			insights.NegativeTrends = append(insights.NegativeTrends, "Low population diversity detected")
			insights.Recommendations = append(insights.Recommendations, "Increase mutation rate to maintain diversity")
		}
	}

	return insights, nil
}

func (g *GEPA) updateConvergenceStatusFromReflection(insights *CrossGenerationInsights) {
	// Update convergence status based on cross-generation insights
	// This is a simplified implementation
	for _, trend := range insights.NegativeTrends {
		if strings.Contains(strings.ToLower(trend), "diversity") {
			g.state.ConvergenceStatus.PrematureConvergenceRisk = "high"
		}
	}
}

func (g *GEPA) analyzeReflectionQuality() *ReflectionQuality {
	quality := &ReflectionQuality{
		OverallScore:           0.7,
		AccuracyScores:         make(map[string]float64),
		Usefulness:             0.7,
		ConsistencyScore:       0.8,
		ImprovementSuggestions: make([]string, 0),
	}

	// Analyze reflection history for quality metrics
	if len(g.state.ReflectionHistory) > 0 {
		// Calculate average confidence as a proxy for quality
		var totalConfidence float64
		count := 0
		for _, reflection := range g.state.ReflectionHistory {
			totalConfidence += reflection.ConfidenceScore
			count++
		}
		quality.OverallScore = totalConfidence / float64(count)
	}

	return quality
}

func (g *GEPA) generateMetaReflectionInsights(ctx context.Context, quality *ReflectionQuality) (*MetaReflectionInsights, error) {
	insights := &MetaReflectionInsights{
		ReflectionStrengths:  make([]string, 0),
		ReflectionWeaknesses: make([]string, 0),
		ProcessImprovements:  make([]string, 0),
		MetaConfidence:       quality.OverallScore,
		ParameterAdjustments: make(map[string]float64),
	}

	if quality.OverallScore > 0.8 {
		insights.ReflectionStrengths = append(insights.ReflectionStrengths, "High-quality reflection analysis")
	} else if quality.OverallScore < 0.5 {
		insights.ReflectionWeaknesses = append(insights.ReflectionWeaknesses, "Low reflection quality detected")
		insights.ProcessImprovements = append(insights.ProcessImprovements, "Improve reflection prompt quality")
	}

	return insights, nil
}

func (g *GEPA) adjustReflectionParameters(insights *MetaReflectionInsights) {
	// Adjust reflection parameters based on meta-insights
	for param, adjustment := range insights.ParameterAdjustments {
		switch param {
		case "self_critique_temp":
			newTemp := g.config.SelfCritiqueTemp + adjustment
			if newTemp >= 0.1 && newTemp <= 1.0 {
				g.config.SelfCritiqueTemp = newTemp
			}
		case "reflection_freq":
			newFreq := int(float64(g.config.ReflectionFreq) + adjustment)
			if newFreq >= 1 && newFreq <= 10 {
				g.config.ReflectionFreq = newFreq
			}
		}
	}
}

func (g *GEPA) extractCommonPatterns(candidates []*GEPACandidate) []string {
	patterns := make([]string, 0)

	if len(candidates) == 0 {
		return patterns
	}

	// Simple pattern extraction based on common words in instructions
	wordCounts := make(map[string]int)
	totalCandidates := len(candidates)

	for _, candidate := range candidates {
		words := strings.Fields(strings.ToLower(candidate.Instruction))
		seenWords := make(map[string]bool)

		for _, word := range words {
			if len(word) > 3 && !seenWords[word] { // Only count unique words per candidate
				wordCounts[word]++
				seenWords[word] = true
			}
		}
	}

	// Extract patterns that appear in majority of high performers
	threshold := int(float64(totalCandidates) * 0.6)
	for word, count := range wordCounts {
		if count >= threshold {
			patterns = append(patterns, fmt.Sprintf("Common keyword: '%s' (appears in %d/%d top performers)",
				word, count, totalCandidates))
		}
	}

	return patterns
}
