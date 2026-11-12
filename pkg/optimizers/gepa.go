// Package optimizers implements the GEPA (Generative Evolutionary Prompt Adaptation) algorithm.
package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
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
	gepaStateKey   contextKey = "gepa_state"
	candidateIDKey contextKey = "current_candidate_id"
	caseTracesKey  contextKey = "current_case_traces"
)

const (
	objectiveSuccess        = "success"
	objectiveQuality        = "quality"
	objectiveEfficiency     = "efficiency"
	objectiveRobustness     = "robustness"
	objectiveGeneralization = "generalization"
	objectiveDiversity      = "diversity"
	objectiveInnovation     = "innovation"
)

const (
	componentSelectionRandom     = "random"
	componentSelectionRoundRobin = "round_robin"
	componentSelectionAll        = "all"
)

const gepaComponentSelectionCursorMetadataKey = "component_selection_cursor"

// DefaultMultiObjectiveWeights returns baseline weights for external diagnostic
// objectives. GEPA's parent selection does not use these weights.
func DefaultMultiObjectiveWeights() map[string]float64 {
	return map[string]float64{
		objectiveSuccess:        0.25,
		objectiveQuality:        0.20,
		objectiveEfficiency:     0.15,
		objectiveRobustness:     0.15,
		objectiveGeneralization: 0.15,
		objectiveDiversity:      0.05,
		objectiveInnovation:     0.05,
	}
}

// GEPAConfig contains configuration options for GEPA optimizer.
type GEPAConfig struct {
	// Evolutionary parameters
	MaxGenerations int     `json:"max_generations"` // Safety cap including seed validation; default: 1000; 0 disables.
	MutationRate   float64 `json:"mutation_rate"`   // Per-selected-component proposal probability; default: 1.0

	// Reflection parameters
	ReflectionDepth int `json:"reflection_depth"` // Default: 3

	// Selection parameters
	TournamentSize     int    `json:"tournament_size"`     // Default: 3
	SelectionStrategy  string `json:"selection_strategy"`  // Default: "adaptive_pareto"; both Pareto strategies sample per-example metric winners
	ComponentSelection string `json:"component_selection"` // Default: "round_robin" | "random" | "all"

	// Convergence parameters
	ConvergenceThreshold float64 `json:"convergence_threshold"` // Default: 0.01
	StagnationLimit      int     `json:"stagnation_limit"`      // Optional attempted updates without improvement; default: 0 (disabled).

	// Performance parameters. Compile uses epoch-shuffled training minibatches
	// (the last batch may be smaller). Validation uses the explicit examples or
	// holdout below; with neither, it scores the full training set, not a batch.
	EvaluationBatchSize        int                   `json:"evaluation_batch_size"` // Default: 5
	ConcurrencyLevel           int                   `json:"concurrency_level"`     // Program executions run at once per evaluation; default: 3. Above 1 the program must be safe for concurrent use; Compile serializes metric and FeedbackEvaluator calls; CompileWithScorer runs them at this level too.
	ValidationSplit            float64               `json:"validation_split"`      // Default: 0.0 (no holdout)
	ValidationExamples         []core.Example        `json:"-"`
	RandomSeed                 int64                 `json:"random_seed"`           // Default: 0 (time-based)
	MaxMergeInvocations        int                   `json:"max_merge_invocations"` // Default: 5
	MaxMetricCalls             int                   `json:"max_metric_calls"`      // Default: 1000; 0 disables. Complete evaluations must fit the remaining budget.
	SkipPerfectScore           bool                  `json:"skip_perfect_score"`    // Default: true; skip proposals when every minibatch case reaches PerfectScore.
	PerfectScore               float64               `json:"perfect_score"`         // Default: 1.0; configure for the supplied metric's scale.
	ScoreThreshold             float64               `json:"score_threshold"`       // Default: 0.0 (disabled)
	MaxRuntime                 time.Duration         `json:"max_runtime"`           // Default: 0 (disabled)
	RunDir                     string                `json:"run_dir,omitempty"`
	StopCallbacks              []GEPAStopper         `json:"-"`
	FeedbackEvaluator          GEPAFeedbackEvaluator `json:"-"`
	AddFormatFailureAsFeedback bool                  `json:"add_format_failure_as_feedback"`

	// LLM parameters
	GenerationModel string   `json:"generation_model"` // Default: uses core.GetDefaultLLM()
	ReflectionModel string   `json:"reflection_model"` // Default: uses core.GetTeacherLLM()
	GenerationLLM   core.LLM `json:"-"`
	ReflectionLLM   core.LLM `json:"-"`
}

// DefaultGEPAConfig returns the default configuration for GEPA.
func DefaultGEPAConfig() *GEPAConfig {
	return &GEPAConfig{
		MaxGenerations:       1000,
		MutationRate:         1.0, // The single-candidate loop should attempt an update on every iteration
		ReflectionDepth:      3,
		TournamentSize:       3,
		SelectionStrategy:    "adaptive_pareto", // Sample per-example metric winners by coverage
		ComponentSelection:   componentSelectionRoundRobin,
		ConvergenceThreshold: 0.01,
		StagnationLimit:      0,
		EvaluationBatchSize:  5,
		ConcurrencyLevel:     3,
		ValidationSplit:      0.0,
		MaxMergeInvocations:  5,
		MaxMetricCalls:       1000,
		SkipPerfectScore:     true,
		PerfectScore:         1.0,
		ScoreThreshold:       0.0,
		MaxRuntime:           0,
	}
}

// GEPACandidate represents a single prompt candidate in the GEPA population.
type GEPACandidate struct {
	ID             string            `json:"id"`
	ModuleName     string            `json:"module_name"`
	Instruction    string            `json:"instruction"`
	ComponentTexts map[string]string `json:"component_texts,omitempty"`
	Demonstrations []core.Example    `json:"demonstrations"`
	Generation     int               `json:"generation"`
	Fitness        float64           `json:"fitness"`
	ParentIDs      []string          `json:"parent_ids"`
	CreatedAt      time.Time         `json:"created_at"`
	Metadata       map[string]any    `json:"metadata"`
}

func cloneComponentTexts(source map[string]string) map[string]string {
	if len(source) == 0 {
		return nil
	}

	cloned := make(map[string]string, len(source))
	for key, value := range source {
		cloned[key] = value
	}

	return cloned
}

func cloneCandidateComponentTexts(candidate *GEPACandidate) map[string]string {
	if candidate == nil {
		return nil
	}

	componentTexts := cloneComponentTexts(candidate.ComponentTexts)
	if moduleName := strings.TrimSpace(candidate.ModuleName); moduleName != "" {
		if componentTexts == nil {
			componentTexts = make(map[string]string, 1)
		}
		if _, exists := componentTexts[moduleName]; !exists {
			componentTexts[moduleName] = candidate.Instruction
		}
	}

	return componentTexts
}

func deriveBaseComponentTexts(base map[string]string, moduleName, instruction string) map[string]string {
	componentTexts := cloneComponentTexts(base)
	moduleName = strings.TrimSpace(moduleName)
	if moduleName == "" {
		return componentTexts
	}

	if componentTexts == nil {
		componentTexts = make(map[string]string, 1)
	}
	// GEPA candidates carry the full program instruction map. Mutations update
	// one selected component while preserving the rest of the candidate state.
	componentTexts[moduleName] = instruction
	return componentTexts
}

func deriveCandidateComponentTexts(candidate *GEPACandidate, moduleName, instruction string) map[string]string {
	return deriveBaseComponentTexts(cloneCandidateComponentTexts(candidate), moduleName, instruction)
}

func candidateInstructionForModule(candidate *GEPACandidate, moduleName string) string {
	if candidate == nil {
		return ""
	}

	moduleName = strings.TrimSpace(moduleName)
	if moduleName == "" {
		return ""
	}

	if instruction, exists := candidate.ComponentTexts[moduleName]; exists {
		return instruction
	}

	// Backward compatibility for callers that still construct candidates without
	// a populated ComponentTexts map.
	if candidate.ModuleName == moduleName {
		return candidate.Instruction
	}

	return ""
}

func componentTextsFromProgram(program core.Program) map[string]string {
	if len(program.Modules) == 0 {
		return nil
	}

	componentTexts := make(map[string]string, len(program.Modules))
	for moduleName, module := range program.Modules {
		componentTexts[moduleName] = module.GetSignature().Instruction
	}

	return componentTexts
}

func componentModuleNames(componentTexts map[string]string) []string {
	if len(componentTexts) == 0 {
		return nil
	}

	moduleNames := make([]string, 0, len(componentTexts))
	for moduleName := range componentTexts {
		moduleName = strings.TrimSpace(moduleName)
		if moduleName == "" {
			continue
		}
		moduleNames = append(moduleNames, moduleName)
	}
	sort.Strings(moduleNames)
	return moduleNames
}

// Population represents a generation of prompt candidates.
type Population struct {
	Candidates    []*GEPACandidate `json:"candidates"`
	Generation    int              `json:"generation"`
	BestFitness   float64          `json:"best_fitness"`
	BestCandidate *GEPACandidate   `json:"best_candidate"`
}

// ExecutionTrace tracks the execution of a module for reflection analysis.
type ExecutionTrace struct {
	CandidateID string         `json:"candidate_id"`
	ModuleName  string         `json:"module_name"`
	Inputs      map[string]any `json:"inputs"`
	Outputs     map[string]any `json:"outputs"`
	Error       error          `json:"error"`
	Duration    time.Duration  `json:"duration"`
	Success     bool           `json:"success"`
	Timestamp   time.Time      `json:"timestamp"`
	ContextData map[string]any `json:"context_data"`
}

// ExecutionPatterns represents analyzed patterns from execution traces.
type ExecutionPatterns struct {
	SuccessRate         float64        `json:"success_rate"`
	SuccessCount        int            `json:"success_count"`
	TotalExecutions     int            `json:"total_executions"`
	AverageResponseTime time.Duration  `json:"average_response_time"`
	CommonFailures      []string       `json:"common_failures"`
	QualityIndicators   []string       `json:"quality_indicators"`
	RichTraceEvidence   []string       `json:"rich_trace_evidence"`
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

// CandidateMetrics tracks detailed metrics for each candidate.
type CandidateMetrics struct {
	TotalEvaluations int             `json:"total_evaluations"`
	SuccessCount     int             `json:"success_count"`
	AverageFitness   float64         `json:"average_fitness"`
	BestFitness      float64         `json:"best_fitness"`
	ExecutionTimes   []time.Duration `json:"execution_times"`
	ErrorCounts      map[string]int  `json:"error_counts"`
	Metadata         map[string]any  `json:"metadata"`
}

// gepaValidationFrontierEntry records the tied-best validation-frontier
// programs for one validation example. CandidateID remains as a canonical
// representative for compatibility, while CandidateIDs preserves the full
// pruned tied-best set used for coverage-based selection.
type gepaValidationFrontierEntry struct {
	CaseIndex    int          `json:"case_index"`
	CandidateID  string       `json:"candidate_id,omitempty"`
	CandidateIDs []string     `json:"candidate_ids,omitempty"`
	Score        float64      `json:"score"`
	Example      core.Example `json:"example"`
}

// GEPAState tracks the complete state of GEPA optimization.
type GEPAState struct {
	CurrentGeneration         int                                  `json:"current_generation"`
	BestCandidate             *GEPACandidate                       `json:"best_candidate"`
	BestFitness               float64                              `json:"best_fitness"`
	BestValidationCandidate   *GEPACandidate                       `json:"best_validation_candidate,omitempty"`
	BestValidationFitness     float64                              `json:"best_validation_fitness"`
	PopulationHistory         []*Population                        `json:"population_history"`
	ReflectionHistory         []*ReflectionResult                  `json:"reflection_history"`
	ConvergenceStatus         *ConvergenceStatus                   `json:"convergence_status"`
	StartTime                 time.Time                            `json:"start_time"`
	LastImprovement           time.Time                            `json:"last_improvement"`
	LastValidatedGeneration   int                                  `json:"last_validated_generation"`
	ExecutionTraces           map[string][]ExecutionTrace          `json:"execution_traces"`
	CandidateMetrics          map[string]*CandidateMetrics         `json:"candidate_metrics"`
	MultiObjectiveFitnessMap  map[string]*MultiObjectiveFitness    `json:"multi_objective_fitness_map"`
	ValidationFrontier        map[int]*gepaValidationFrontierEntry `json:"validation_frontier,omitempty"`
	ValidationCoverage        map[string]int                       `json:"validation_coverage,omitempty"`
	MergeInvocations          int                                  `json:"merge_invocations,omitempty"`
	PerformedMerges           map[string]bool                      `json:"performed_merges,omitempty"`
	MetricCalls               int                                  `json:"metric_calls,omitempty"`
	StopReason                string                               `json:"stop_reason,omitempty"`
	StopMetadata              map[string]any                       `json:"stop_metadata,omitempty"`
	candidateReflections      map[string]*ReflectionResult
	candidateEvaluations      map[string]*gepaCandidateEvaluation
	candidateValidationEvals  map[string]*gepaCandidateEvaluation
	evaluationCaseCache       map[string]*gepaCachedEvaluationCase
	candidateCaseObservations map[string][]CandidateCaseObservation

	// Pareto archive for elite solution management
	ParetoArchive     []*GEPACandidate                  `json:"pareto_archive"`
	ArchiveFitnessMap map[string]*MultiObjectiveFitness `json:"archive_fitness_map"`
	MaxArchiveSize    int                               `json:"max_archive_size"`

	mu sync.RWMutex
}

// NewGEPAState creates a new GEPA optimization state.
func NewGEPAState() *GEPAState {
	return &GEPAState{
		CurrentGeneration:         0,
		BestFitness:               0.0,
		BestValidationFitness:     math.Inf(-1),
		PopulationHistory:         make([]*Population, 0),
		ReflectionHistory:         make([]*ReflectionResult, 0),
		StartTime:                 time.Now(),
		LastImprovement:           time.Now(),
		LastValidatedGeneration:   -1,
		ExecutionTraces:           make(map[string][]ExecutionTrace),
		CandidateMetrics:          make(map[string]*CandidateMetrics),
		MultiObjectiveFitnessMap:  make(map[string]*MultiObjectiveFitness),
		ValidationFrontier:        make(map[int]*gepaValidationFrontierEntry),
		ValidationCoverage:        make(map[string]int),
		PerformedMerges:           make(map[string]bool),
		candidateReflections:      make(map[string]*ReflectionResult),
		candidateEvaluations:      make(map[string]*gepaCandidateEvaluation),
		candidateValidationEvals:  make(map[string]*gepaCandidateEvaluation),
		evaluationCaseCache:       make(map[string]*gepaCachedEvaluationCase),
		candidateCaseObservations: make(map[string][]CandidateCaseObservation),
		ParetoArchive:             make([]*GEPACandidate, 0),
		ArchiveFitnessMap:         make(map[string]*MultiObjectiveFitness),
		MaxArchiveSize:            50, // Configurable archive size
		ConvergenceStatus: &ConvergenceStatus{
			StagnationCount:                0,
			PrematureConvergenceRisk:       "low",
			ExplorationExploitationBalance: "balanced",
			DiversityIndex:                 1.0,
			IsConverged:                    false,
		},
	}
}

// AddTrace adds an execution trace to the state. It does not attribute the
// trace to an example; use AddTraceContext for per-example reflection evidence.
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
			Metadata:         make(map[string]any),
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

// AddTraceContext adds an execution trace and attributes it to the example
// being evaluated in ctx, so concurrently evaluated examples keep separate
// reflection evidence. Adapters that record their own traces should use it.
func (s *GEPAState) AddTraceContext(ctx context.Context, trace *ExecutionTrace) {
	s.AddTrace(trace)
	if traces, ok := ctx.Value(caseTracesKey).(*gepaCaseTraces); ok {
		traces.add(trace)
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

// SetCandidateReflections replaces the latest candidate reflection cache for the
// current generation. Proposal generation consumes this to turn reflection into
// concrete instruction updates without re-running reflection.
func (s *GEPAState) SetCandidateReflections(reflections map[string]*ReflectionResult) {
	if s == nil {
		return
	}

	cloned := make(map[string]*ReflectionResult, len(reflections))
	for candidateID, reflection := range reflections {
		cloned[candidateID] = cloneReflectionResult(reflection)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.candidateReflections = cloned
}

// GetCandidateReflection returns the cached reflection for a candidate from the
// latest reflected generation.
func (s *GEPAState) GetCandidateReflection(candidateID string) *ReflectionResult {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneReflectionResult(s.candidateReflections[candidateID])
}

// SetCandidateEvaluations replaces the latest candidate evaluation cache for the
// current generation. Reflection consumes this to build example-grounded
// prompts without re-running candidate evaluation.
func (s *GEPAState) SetCandidateEvaluations(evaluations map[string]*gepaCandidateEvaluation) {
	if s == nil {
		return
	}

	cloned := make(map[string]*gepaCandidateEvaluation, len(evaluations))
	for candidateID, evaluation := range evaluations {
		cloned[candidateID] = cloneGEPACandidateEvaluation(evaluation)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.candidateEvaluations = cloned
}

// GetCandidateEvaluation returns the cached evaluation for a candidate from the
// latest evaluated generation.
func (s *GEPAState) GetCandidateEvaluation(candidateID string) *gepaCandidateEvaluation {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneGEPACandidateEvaluation(s.candidateEvaluations[candidateID])
}

// SetCandidateValidationEvaluations replaces the current validation-evaluation
// cache. Validation selection and final candidate choice consult this
// separately from minibatch/training evaluations.
func (s *GEPAState) SetCandidateValidationEvaluations(evaluations map[string]*gepaCandidateEvaluation) {
	if s == nil {
		return
	}

	var cloned map[string]*gepaCandidateEvaluation
	if len(evaluations) > 0 {
		cloned = make(map[string]*gepaCandidateEvaluation, len(evaluations))
		for candidateID, evaluation := range evaluations {
			cloned[candidateID] = cloneGEPACandidateEvaluation(evaluation)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.candidateValidationEvals = cloned
}

// GetCandidateValidationEvaluation returns the cached validation evaluation for
// a candidate from the latest validation pass.
func (s *GEPAState) GetCandidateValidationEvaluation(candidateID string) *gepaCandidateEvaluation {
	if s == nil {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneGEPACandidateEvaluation(s.candidateValidationEvals[candidateID])
}

// BestCandidateForApplication returns the final candidate that should be
// surfaced to callers, preferring the best validation candidate when one
// exists and otherwise falling back to the best training candidate.
func (s *GEPAState) BestCandidateForApplication() (*GEPACandidate, float64, bool) {
	if s == nil {
		return nil, 0, false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	switch {
	case s.BestValidationCandidate != nil:
		return CloneCandidate(s.BestValidationCandidate), s.BestValidationFitness, true
	case s.BestCandidate != nil:
		return CloneCandidate(s.BestCandidate), s.BestFitness, false
	default:
		return nil, 0, false
	}
}

func (s *GEPAState) ResetEvaluationCaseCache() {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.evaluationCaseCache = make(map[string]*gepaCachedEvaluationCase)
}

func (s *GEPAState) GetEvaluationCaseCache(key string) *gepaCachedEvaluationCase {
	if s == nil || strings.TrimSpace(key) == "" {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneCachedEvaluationCase(s.evaluationCaseCache[key])
}

func (s *GEPAState) UpsertEvaluationCaseCache(key string, cached *gepaCachedEvaluationCase) {
	if s == nil || strings.TrimSpace(key) == "" || cached == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.evaluationCaseCache == nil {
		s.evaluationCaseCache = make(map[string]*gepaCachedEvaluationCase)
	}
	s.evaluationCaseCache[key] = cloneCachedEvaluationCase(cached)
}

// SetValidationFrontier replaces the current validation frontier and candidate
// coverage counts derived from the latest validation pass.
func (s *GEPAState) SetValidationFrontier(frontier map[int]*gepaValidationFrontierEntry, coverage map[string]int) {
	if s == nil {
		return
	}

	var clonedFrontier map[int]*gepaValidationFrontierEntry
	if len(frontier) > 0 {
		clonedFrontier = make(map[int]*gepaValidationFrontierEntry, len(frontier))
		for caseIndex, entry := range frontier {
			clonedFrontier[caseIndex] = cloneValidationFrontierEntry(entry)
		}
	}

	var clonedCoverage map[string]int
	if len(coverage) > 0 {
		clonedCoverage = make(map[string]int, len(coverage))
		for candidateID, count := range coverage {
			clonedCoverage[candidateID] = count
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.ValidationFrontier = clonedFrontier
	s.ValidationCoverage = clonedCoverage
}

// ValidationFrontierSnapshot returns cloned validation-frontier state for
// selection and tests without exposing shared mutable maps.
func (s *GEPAState) ValidationFrontierSnapshot() (map[int]*gepaValidationFrontierEntry, map[string]int) {
	if s == nil {
		return nil, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var frontier map[int]*gepaValidationFrontierEntry
	if len(s.ValidationFrontier) > 0 {
		frontier = make(map[int]*gepaValidationFrontierEntry, len(s.ValidationFrontier))
		for caseIndex, entry := range s.ValidationFrontier {
			frontier[caseIndex] = cloneValidationFrontierEntry(entry)
		}
	}

	var coverage map[string]int
	if len(s.ValidationCoverage) > 0 {
		coverage = make(map[string]int, len(s.ValidationCoverage))
		for candidateID, count := range s.ValidationCoverage {
			coverage[candidateID] = count
		}
	}

	return frontier, coverage
}

func (s *GEPAState) HasRecordedMerge(key string) bool {
	if s == nil || strings.TrimSpace(key) == "" {
		return false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.PerformedMerges[key]
}

func (s *GEPAState) MergeBudgetAvailable(maxInvocations int) bool {
	if s == nil {
		return false
	}
	if maxInvocations <= 0 {
		return true
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.MergeInvocations < maxInvocations
}

func (s *GEPAState) RecordMergeInvocation(key string, maxInvocations int) bool {
	if s == nil || strings.TrimSpace(key) == "" {
		return false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if maxInvocations > 0 && s.MergeInvocations >= maxInvocations {
		return false
	}
	if s.PerformedMerges == nil {
		s.PerformedMerges = make(map[string]bool)
	}
	if s.PerformedMerges[key] {
		return false
	}

	s.PerformedMerges[key] = true
	s.MergeInvocations++
	return true
}

func cloneValidationFrontierEntry(entry *gepaValidationFrontierEntry) *gepaValidationFrontierEntry {
	if entry == nil {
		return nil
	}

	return &gepaValidationFrontierEntry{
		CaseIndex:    entry.CaseIndex,
		CandidateID:  entry.CandidateID,
		CandidateIDs: append([]string(nil), entry.CandidateIDs...),
		Score:        entry.Score,
		Example:      cloneEvaluationExample(entry.Example),
	}
}

// UpsertCandidateEvaluation updates the cached evaluation for a single
// candidate without replacing the rest of the current evaluation cache.
func (s *GEPAState) UpsertCandidateEvaluation(candidateID string, evaluation *gepaCandidateEvaluation) {
	if s == nil || strings.TrimSpace(candidateID) == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.candidateEvaluations == nil {
		s.candidateEvaluations = make(map[string]*gepaCandidateEvaluation)
	}
	s.candidateEvaluations[candidateID] = cloneGEPACandidateEvaluation(evaluation)
}

// UpdateParetoArchive retains per-example metric winners across generations.
// fitnessMap is retained for compatibility as diagnostic data only; archive
// membership requires cached candidate evaluations and never uses these vectors.
func (s *GEPAState) UpdateParetoArchive(candidates []*GEPACandidate, fitnessMap map[string]*MultiObjectiveFitness) {
	s.mu.Lock()
	defer s.mu.Unlock()

	combined := make(map[string]*GEPACandidate, len(s.ParetoArchive)+len(candidates))
	combinedFitness := make(map[string]*MultiObjectiveFitness)
	evaluations := make(map[string]*gepaCandidateEvaluation)
	for _, group := range [][]*GEPACandidate{s.ParetoArchive, candidates} {
		for _, candidate := range group {
			if candidate == nil || s.candidateEvaluations[candidate.ID] == nil {
				continue
			}
			combined[candidate.ID] = candidate
			evaluations[candidate.ID] = s.candidateEvaluations[candidate.ID]
			if fitness := fitnessMap[candidate.ID]; fitness != nil {
				combinedFitness[candidate.ID] = fitness
			} else if fitness := s.ArchiveFitnessMap[candidate.ID]; fitness != nil {
				combinedFitness[candidate.ID] = fitness
			}
		}
	}
	_, coverage := buildValidationFrontier(evaluations)
	archive := make([]*GEPACandidate, 0, len(coverage))
	for id := range coverage {
		archive = append(archive, combined[id])
	}
	sort.Slice(archive, func(i, j int) bool {
		left, right := archive[i].ID, archive[j].ID
		if coverage[left] != coverage[right] {
			return coverage[left] > coverage[right]
		}
		if evaluations[left].AverageScore != evaluations[right].AverageScore {
			return evaluations[left].AverageScore > evaluations[right].AverageScore
		}
		return left < right
	})
	if len(archive) > s.MaxArchiveSize {
		archive = archive[:max(0, s.MaxArchiveSize)]
	}
	s.ParetoArchive = archive
	s.ArchiveFitnessMap = make(map[string]*MultiObjectiveFitness)
	for _, candidate := range archive {
		if fitness := combinedFitness[candidate.ID]; fitness != nil {
			s.ArchiveFitnessMap[candidate.ID] = fitness
		}
	}
}

// GetParetoArchive returns the current Pareto archive of elite solutions.
func (s *GEPAState) GetParetoArchive() []*GEPACandidate {
	s.mu.RLock()
	defer s.mu.RUnlock()

	archive := make([]*GEPACandidate, len(s.ParetoArchive))
	copy(archive, s.ParetoArchive)
	return archive
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
	generationLLM core.PromptModel
	reflectionLLM core.PromptModel

	// Latest stable minibatch adapter from the current generation. Mutation
	// proposals can use this to do candidate-level accept/reject before they
	// enter the next generation.
	evaluationAdapterMu     sync.RWMutex
	latestEvaluationAdapter *gepaEvaluationAdapter

	// Interceptor integration
	interceptorChain *core.InterceptorChain

	// Progress reporting
	progressReporter core.ProgressReporter

	// Random number generator
	rng *rand.Rand
}

// MultiObjectiveFitness stores diagnostic objectives for external evaluators.
// GEPA selection uses per-example metric scores, not this vector.
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

// ObjectiveValue returns the named objective value from the fitness vector.
func (f *MultiObjectiveFitness) ObjectiveValue(objective string) float64 {
	if f == nil {
		return 0.0
	}

	switch objective {
	case objectiveSuccess:
		return f.SuccessRate
	case objectiveQuality:
		return f.OutputQuality
	case objectiveEfficiency:
		return f.Efficiency
	case objectiveRobustness:
		return f.Robustness
	case objectiveGeneralization:
		return f.Generalization
	case objectiveDiversity:
		return f.Diversity
	case objectiveInnovation:
		return f.Innovation
	default:
		return 0.0
	}
}

// ComputeWeightedScore aggregates the objective vector using the supplied weights.
func (f *MultiObjectiveFitness) ComputeWeightedScore(weights map[string]float64) float64 {
	if weights == nil {
		weights = DefaultMultiObjectiveWeights()
	}

	return f.SuccessRate*weights[objectiveSuccess] +
		f.OutputQuality*weights[objectiveQuality] +
		f.Efficiency*weights[objectiveEfficiency] +
		f.Robustness*weights[objectiveRobustness] +
		f.Generalization*weights[objectiveGeneralization] +
		f.Diversity*weights[objectiveDiversity] +
		f.Innovation*weights[objectiveInnovation]
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
	ContextType      string         `json:"context_type"`
	ExecutionCount   int            `json:"execution_count"`
	SuccessRate      float64        `json:"success_rate"`
	AverageLatency   time.Duration  `json:"average_latency"`
	QualityScore     float64        `json:"quality_score"`
	ErrorPatterns    map[string]int `json:"error_patterns"`
	PerformanceTrend []float64      `json:"performance_trend"`
	LastUpdated      time.Time      `json:"last_updated"`
	Metadata         map[string]any `json:"metadata"`
}

// ContextPattern represents learned patterns within execution contexts.
type ContextPattern struct {
	PatternID         string         `json:"pattern_id"`
	ContextFeatures   map[string]any `json:"context_features"`
	SuccessFactors    []string       `json:"success_factors"`
	FailureFactors    []string       `json:"failure_factors"`
	OptimalCandidates []string       `json:"optimal_candidates"`
	Confidence        float64        `json:"confidence"`
	LastUpdated       time.Time      `json:"last_updated"`
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
	CustomData map[string]any `json:"custom_data"`
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
			similarity := g.calculateHotPathInstructionSimilarity(targetCandidate.Instruction, other.Instruction)
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

// calculateHotPathInstructionSimilarity keeps pairwise diversity loops off the LLM path.
func (g *GEPA) calculateHotPathInstructionSimilarity(inst1, inst2 string) float64 {
	return g.calculateLexicalSimilarity(inst1, inst2)
}

// calculateLexicalSimilarity provides enhanced word-based similarity as fallback.
func (g *GEPA) calculateLexicalSimilarity(inst1, inst2 string) float64 {
	if inst1 == inst2 {
		return 1.0
	}

	words1 := strings.Fields(strings.ToLower(inst1))
	words2 := strings.Fields(strings.ToLower(inst2))

	if len(words1) == 0 && len(words2) == 0 {
		return 1.0
	}

	if len(words1) == 0 || len(words2) == 0 {
		return 0.0
	}

	// Enhanced similarity combining multiple metrics
	jaccardSim := g.calculateJaccardSimilarity(words1, words2)
	cosineSim := g.calculateCosineSimilarity(words1, words2)

	// Weighted combination favoring cosine similarity for better semantic approximation
	return 0.3*jaccardSim + 0.7*cosineSim
}

// calculateJaccardSimilarity computes Jaccard similarity coefficient.
func (g *GEPA) calculateJaccardSimilarity(words1, words2 []string) float64 {
	set1 := make(map[string]bool)
	set2 := make(map[string]bool)

	for _, word := range words1 {
		set1[word] = true
	}
	for _, word := range words2 {
		set2[word] = true
	}

	intersection := 0
	for word := range set1 {
		if set2[word] {
			intersection++
		}
	}

	union := len(set1) + len(set2) - intersection
	if union == 0 {
		return 1.0
	}

	return float64(intersection) / float64(union)
}

// calculateCosineSimilarity computes cosine similarity using word frequency vectors.
func (g *GEPA) calculateCosineSimilarity(words1, words2 []string) float64 {
	// Build vocabulary
	vocab := make(map[string]int)
	for _, word := range words1 {
		if _, exists := vocab[word]; !exists {
			vocab[word] = len(vocab)
		}
	}
	for _, word := range words2 {
		if _, exists := vocab[word]; !exists {
			vocab[word] = len(vocab)
		}
	}

	if len(vocab) == 0 {
		return 1.0
	}

	// Create frequency vectors
	vec1 := make([]float64, len(vocab))
	vec2 := make([]float64, len(vocab))

	for _, word := range words1 {
		vec1[vocab[word]]++
	}
	for _, word := range words2 {
		vec2[vocab[word]]++
	}

	// Compute cosine similarity
	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0.0 || norm2 == 0.0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
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

// ObjectiveStats represents statistical information for a single objective.
type ObjectiveStats struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Range  float64 `json:"range"`
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
	if config.ValidationSplit < 0 {
		config.ValidationSplit = defaults.ValidationSplit
	}
	if config.ValidationSplit > 1 {
		config.ValidationSplit = 0.9
	}
	if config.TournamentSize <= 0 {
		config.TournamentSize = defaults.TournamentSize
	}
	if config.MaxMergeInvocations <= 0 {
		config.MaxMergeInvocations = defaults.MaxMergeInvocations
	}
	if config.SelectionStrategy == "" {
		config.SelectionStrategy = defaults.SelectionStrategy
	}
	switch config.ComponentSelection {
	case "", componentSelectionRandom, componentSelectionRoundRobin, componentSelectionAll:
		if config.ComponentSelection == "" {
			config.ComponentSelection = defaults.ComponentSelection
		}
	default:
		config.ComponentSelection = defaults.ComponentSelection
	}

	// Initialize LLMs
	generationLLM := config.GenerationLLM
	if generationLLM == nil {
		generationLLM = core.GetDefaultLLM()
	}

	reflectionLLM := config.ReflectionLLM
	if reflectionLLM == nil {
		reflectionLLM = core.GetTeacherLLM()
	}
	if reflectionLLM == nil {
		reflectionLLM = generationLLM
	}
	if reflectionLLM == nil {
		return nil, fmt.Errorf("no LLM available for GEPA reflection")
	}
	if generationLLM == nil {
		generationLLM = reflectionLLM
	}

	// Initialize random number generator
	seed := config.RandomSeed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	gepa := &GEPA{
		config:           config,
		state:            NewGEPAState(),
		generationLLM:    generationLLM,
		reflectionLLM:    reflectionLLM,
		interceptorChain: core.NewInterceptorChain(),
		rng:              rng,
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

// Compile implements the core.Optimizer interface for GEPA. Metric and
// FeedbackEvaluator calls are serialized.
func (g *GEPA) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	return g.compile(ctx, program, dataset, core.MetricScorer(metric), false)
}

// CompileWithScorer optimizes a program from materialized, read-only examples. It
// calls scorer and FeedbackEvaluator from up to ConcurrencyLevel goroutines at
// once, so both must be safe for concurrent use. An example the scorer cannot
// score counts as zero and is scored again the next time its candidate is
// evaluated.
func (g *GEPA) CompileWithScorer(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	return g.compile(ctx, program, newGEPAExampleDataset(examples), scorer, true)
}

func (g *GEPA) compile(ctx context.Context, program core.Program, dataset core.Dataset, scorer core.Scorer, concurrentScoring bool) (core.Program, error) {
	ctx, span := core.StartSpan(ctx, "GEPA.Compile")
	defer core.EndSpan(ctx)
	_ = span // Suppress unused variable warning

	logger := logging.GetLogger()
	logger.Info(ctx, "Starting GEPA optimization with max_metric_calls=%d, max_generations=%d",
		g.config.MaxMetricCalls,
		g.config.MaxGenerations)
	resumed, err := g.prepareRunState()
	if err != nil {
		return program, err
	}
	if !resumed {
		g.state.ResetRunControls(time.Now())
	}
	g.state.ResetEvaluationCaseCache()

	// Initialize GEPA state in context
	ctx = g.withGEPAState(ctx)

	// Install interceptors on the program
	optimizedProgram := g.installInterceptors(program)
	err = g.runIterativeOptimizationLoop(ctx, optimizedProgram, dataset, scorer, concurrentScoring)
	if err != nil {
		return program, err
	}
	if err := ctx.Err(); err != nil {
		return program, err
	}

	// Apply best candidate to program
	finalProgram := g.applyBestCandidate(optimizedProgram)

	// Log optimization results
	g.logOptimizationResults(ctx)

	return finalProgram, nil
}

func (g *GEPA) runIterativeOptimizationLoop(ctx context.Context, program core.Program, dataset core.Dataset, scorer core.Scorer, concurrentScoring bool) error {
	if g.config.MaxGenerations <= 0 && g.config.MaxMetricCalls <= 0 {
		return fmt.Errorf("optimizers: a metric budget or iteration limit is required")
	}
	if scorer == nil {
		return fmt.Errorf("optimizers: metric is required")
	}

	trainDataset, validationExamples, err := g.prepareOptimizationDatasets(dataset)
	if err != nil {
		return err
	}
	if trainDataset == nil {
		return fmt.Errorf("optimizers: training dataset is required")
	}
	trainExamples, err := core.MaterializeDatasetContext(ctx, trainDataset)
	if err != nil {
		return err
	}
	if len(trainExamples) == 0 {
		return fmt.Errorf("optimizers: training dataset must not be empty")
	}
	// A rotating minibatch is not a comparable all-time score. Without a
	// holdout, use the complete training set for validation/final selection.
	if len(validationExamples) == 0 {
		validationExamples = cloneEvaluationExamples(trainExamples)
	}
	if g.getCurrentPopulation() == nil {
		g.resetValidationState()
		if err := g.initializePopulation(ctx, program); err != nil {
			return fmt.Errorf("failed to initialize population: %w", err)
		}
		g.state.CurrentGeneration = 0
	}

	validationAdapter := newGEPAScorerAdapter(program, validationExamples, scorer, concurrentScoring)
	// Fresh runs validate only the seed. Checkpoints omit transient evaluations,
	// so resumed runs rehydrate each retained candidate once, charging the budget.
	for _, candidate := range g.getCurrentPopulation().Candidates {
		if !g.validateRetainedCandidate(ctx, candidate, validationAdapter) {
			return g.saveRunState()
		}
	}
	if err := g.saveRunState(); err != nil {
		return err
	}

	for generation := g.state.CurrentGeneration + 1; g.config.MaxGenerations <= 0 || generation < g.config.MaxGenerations; generation++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if decision := g.evaluateStopDecision(ctx); decision != nil {
			g.recordStopDecision(decision)
			return g.saveRunState()
		}
		if g.validationIsPerfect() {
			g.recordStopDecision(&GEPAStopDecision{Reason: "perfect_validation"})
			return g.saveRunState()
		}
		adapter := newGEPAScorerAdapter(program, g.evaluationBatchForGeneration(trainExamples, generation-1), scorer, concurrentScoring)
		g.setLatestEvaluationAdapter(adapter)
		g.state.CurrentGeneration = generation
		if !g.runCandidateIteration(ctx, adapter, validationAdapter) {
			return g.saveRunState()
		}
		if err := g.saveRunState(); err != nil {
			return err
		}
		if g.config.MaxMetricCalls > 0 {
			g.reportOptimizationProgress("GEPA Metric Calls", g.state.MetricCallCount(), g.config.MaxMetricCalls)
		} else {
			g.reportOptimizationProgress("GEPA Iterations", generation+1, g.config.MaxGenerations)
		}
	}
	g.recordStopDecision(&GEPAStopDecision{Reason: "iteration_limit"})
	return g.saveRunState()
}

func (g *GEPA) prepareOptimizationDatasets(dataset core.Dataset) (core.Dataset, []core.Example, error) {
	if dataset == nil {
		return dataset, nil, nil
	}
	if len(g.config.ValidationExamples) > 0 {
		trainExamples := core.DatasetToSlice(dataset)
		if len(trainExamples) == 0 {
			return nil, nil, fmt.Errorf("optimizers: explicit validation examples require at least one training example")
		}
		return newGEPAExampleDataset(cloneEvaluationExamples(trainExamples)), cloneEvaluationExamples(g.config.ValidationExamples), nil
	}
	if g.config.ValidationSplit <= 0 {
		return dataset, nil, nil
	}

	allExamples := core.DatasetToSlice(dataset)
	if len(allExamples) <= 1 {
		return newGEPAExampleDataset(allExamples), nil, nil
	}

	allExamples = cloneEvaluationExamples(allExamples)
	rng := g.rng
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}
	// Validation uses a shuffled holdout slice so ordered datasets do not bias the
	// tail examples into becoming the validation set by construction.
	rng.Shuffle(len(allExamples), func(i, j int) {
		allExamples[i], allExamples[j] = allExamples[j], allExamples[i]
	})

	validationCount := int(math.Ceil(float64(len(allExamples)) * g.config.ValidationSplit))
	if validationCount >= len(allExamples) {
		validationCount = len(allExamples) - 1
	}
	if validationCount <= 0 {
		return newGEPAExampleDataset(allExamples), nil, nil
	}

	split := len(allExamples) - validationCount
	trainExamples := allExamples[:split]
	validationExamples := allExamples[split:]
	if len(trainExamples) == 0 {
		return nil, nil, fmt.Errorf("optimizers: validation split consumed all training examples")
	}

	return newGEPAExampleDataset(trainExamples), validationExamples, nil
}

func (g *GEPA) resetValidationState() {
	if g == nil || g.state == nil {
		return
	}

	g.state.mu.Lock()
	defer g.state.mu.Unlock()
	g.state.candidateValidationEvals = make(map[string]*gepaCandidateEvaluation)
	g.state.ValidationFrontier = make(map[int]*gepaValidationFrontierEntry)
	g.state.ValidationCoverage = make(map[string]int)
	g.state.BestValidationCandidate = nil
	g.state.BestValidationFitness = math.Inf(-1)
	g.state.LastValidatedGeneration = -1
}

func (g *GEPA) evaluateValidationPopulation(ctx context.Context, program core.Program, validationExamples []core.Example, metric core.Metric) error {
	if len(validationExamples) == 0 {
		g.resetValidationState()
		return nil
	}

	population := g.getCurrentPopulation()
	if population == nil {
		return fmt.Errorf("no current population to validate")
	}

	logger := logging.GetLogger()
	logger.Info(ctx, "Validating population: generation=%d, candidates=%d, examples=%d",
		population.Generation,
		len(population.Candidates),
		len(validationExamples))

	adapter := g.newEvaluationAdapterForExamples(program, validationExamples, metric)
	p := pool.New().WithMaxGoroutines(g.config.ConcurrencyLevel)

	var mu sync.Mutex
	bestFitness := math.Inf(-1)
	var bestCandidate *GEPACandidate
	candidateEvaluations := make(map[string]*gepaCandidateEvaluation, len(population.Candidates))
	recordValidationEvaluation := func(candidate *GEPACandidate, evaluation *gepaCandidateEvaluation) {
		if candidate == nil || evaluation == nil {
			return
		}

		score := evaluation.AverageScore

		mu.Lock()
		defer mu.Unlock()

		candidateEvaluations[candidate.ID] = evaluation
		if bestCandidate == nil || score > bestFitness || (score == bestFitness && candidate.ID < bestCandidate.ID) {
			bestCandidate = g.copyCandidate(candidate)
			bestCandidate.Fitness = score
			bestFitness = score
			if bestCandidate.Metadata == nil {
				bestCandidate.Metadata = make(map[string]any)
			}
			bestCandidate.Metadata["validation_score"] = score
		}
	}

	for _, candidate := range population.Candidates {
		if candidate == nil {
			continue
		}

		if cachedEvaluation := g.state.GetCandidateValidationEvaluation(candidate.ID); cachedEvaluation != nil {
			recordValidationEvaluation(candidate, cachedEvaluation)
			continue
		}

		candidate := candidate
		p.Go(func() {
			// Validation must not overwrite training fitness on the working
			// population or on the all-time best candidate it may reference.
			evaluation := g.evaluateCandidateWithAdapter(ctx, g.copyCandidate(candidate), adapter)
			recordValidationEvaluation(candidate, evaluation)
		})
	}

	p.Wait()
	frontier, coverage := buildValidationFrontier(candidateEvaluations)
	g.state.SetCandidateValidationEvaluations(candidateEvaluations)
	g.state.SetValidationFrontier(frontier, coverage)
	g.setBestValidationCandidate(bestCandidate, bestFitness)

	logger.Info(ctx, "Validation completed: generation=%d, evaluated_candidates=%d, best_validation_fitness=%.3f",
		population.Generation,
		len(candidateEvaluations),
		bestFitness)

	return nil
}

func (g *GEPA) setBestValidationCandidate(candidate *GEPACandidate, fitness float64) {
	if g == nil || g.state == nil || candidate == nil || math.IsInf(fitness, -1) {
		return
	}

	g.state.mu.Lock()
	defer g.state.mu.Unlock()

	if g.state.BestValidationCandidate == nil || fitness > g.state.BestValidationFitness ||
		(fitness == g.state.BestValidationFitness && candidate.ID < g.state.BestValidationCandidate.ID) {
		g.state.BestValidationCandidate = CloneCandidate(candidate)
		g.state.BestValidationFitness = fitness
	}
}

func buildValidationFrontier(evaluations map[string]*gepaCandidateEvaluation) (map[int]*gepaValidationFrontierEntry, map[string]int) {
	if len(evaluations) == 0 {
		return nil, nil
	}

	frontier := make(map[int]*gepaValidationFrontierEntry)
	candidateIDs := make([]string, 0, len(evaluations))
	candidateScores := make(map[string]float64, len(evaluations))
	for candidateID := range evaluations {
		candidateIDs = append(candidateIDs, candidateID)
	}
	sort.Strings(candidateIDs)

	for _, candidateID := range candidateIDs {
		evaluation := evaluations[candidateID]
		if evaluation == nil {
			continue
		}
		candidateScores[candidateID] = evaluation.AverageScore

		for caseIndex, evalCase := range evaluation.Cases {
			score := evalCase.Score
			if evalCase.Err != nil || math.IsNaN(score) || math.IsInf(score, 0) {
				continue
			}
			current := frontier[caseIndex]
			switch {
			case current == nil || score > current.Score:
				frontier[caseIndex] = &gepaValidationFrontierEntry{
					CaseIndex:    caseIndex,
					CandidateID:  candidateID,
					CandidateIDs: []string{candidateID},
					Score:        score,
					Example:      cloneEvaluationExample(evalCase.Example),
				}
			case score == current.Score:
				current.CandidateIDs = append(current.CandidateIDs, candidateID)
				if current.CandidateID == "" || candidateID < current.CandidateID {
					current.CandidateID = candidateID
				}
			}
		}
	}

	prunedFrontier := pruneValidationFrontier(frontier, candidateScores)
	coverage := make(map[string]int)
	for _, entry := range prunedFrontier {
		if entry == nil {
			continue
		}
		for _, candidateID := range entry.CandidateIDs {
			coverage[candidateID]++
		}
	}

	return prunedFrontier, coverage
}

func pruneValidationFrontier(frontier map[int]*gepaValidationFrontierEntry, candidateScores map[string]float64) map[int]*gepaValidationFrontierEntry {
	if len(frontier) == 0 {
		return nil
	}

	orderedCandidateIDs := validationFrontierCandidateIDs(frontier)
	if len(orderedCandidateIDs) <= 1 {
		return frontier
	}

	sort.SliceStable(orderedCandidateIDs, func(i, j int) bool {
		left := orderedCandidateIDs[i]
		right := orderedCandidateIDs[j]
		leftScore := candidateScores[left]
		rightScore := candidateScores[right]
		if leftScore == rightScore {
			return left < right
		}
		return leftScore < rightScore
	})

	remaining := make(map[string]struct{}, len(orderedCandidateIDs))
	for _, candidateID := range orderedCandidateIDs {
		remaining[candidateID] = struct{}{}
	}

	for _, candidateID := range orderedCandidateIDs {
		if _, ok := remaining[candidateID]; !ok {
			continue
		}
		if isValidationFrontierCandidateDominated(candidateID, remaining, frontier) {
			delete(remaining, candidateID)
		}
	}

	prunedFrontier := make(map[int]*gepaValidationFrontierEntry, len(frontier))
	for caseIndex, entry := range frontier {
		if entry == nil {
			continue
		}

		candidateIDs := make([]string, 0, len(entry.CandidateIDs))
		for _, candidateID := range entry.CandidateIDs {
			if _, ok := remaining[candidateID]; ok {
				candidateIDs = append(candidateIDs, candidateID)
			}
		}
		if len(candidateIDs) == 0 {
			continue
		}

		prunedFrontier[caseIndex] = &gepaValidationFrontierEntry{
			CaseIndex:    entry.CaseIndex,
			CandidateID:  candidateIDs[0],
			CandidateIDs: candidateIDs,
			Score:        entry.Score,
			Example:      cloneEvaluationExample(entry.Example),
		}
	}

	return prunedFrontier
}

func validationFrontierCandidateIDs(frontier map[int]*gepaValidationFrontierEntry) []string {
	seen := make(map[string]struct{})
	candidateIDs := make([]string, 0)
	for _, entry := range frontier {
		if entry == nil {
			continue
		}
		for _, candidateID := range entry.CandidateIDs {
			if candidateID == "" {
				continue
			}
			if _, ok := seen[candidateID]; ok {
				continue
			}
			seen[candidateID] = struct{}{}
			candidateIDs = append(candidateIDs, candidateID)
		}
	}
	return candidateIDs
}

func isValidationFrontierCandidateDominated(candidateID string, remaining map[string]struct{}, frontier map[int]*gepaValidationFrontierEntry) bool {
	if candidateID == "" || len(remaining) <= 1 {
		return false
	}

	for _, entry := range frontier {
		if entry == nil || !validationFrontierEntryContainsCandidate(entry, candidateID) {
			continue
		}

		coveredByOther := false
		for _, otherCandidateID := range entry.CandidateIDs {
			if otherCandidateID == candidateID {
				continue
			}
			if _, ok := remaining[otherCandidateID]; ok {
				coveredByOther = true
				break
			}
		}
		if !coveredByOther {
			return false
		}
	}

	return true
}

func validationFrontierEntryContainsCandidate(entry *gepaValidationFrontierEntry, candidateID string) bool {
	if entry == nil || candidateID == "" {
		return false
	}
	for _, currentCandidateID := range entry.CandidateIDs {
		if currentCandidateID == candidateID {
			return true
		}
	}
	return false
}

func (g *GEPA) reportOptimizationProgress(phase string, current, total int) {
	if g.progressReporter == nil {
		return
	}
	g.progressReporter.Report(phase, current, total)
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
		AddModuleInterceptor(g.gepaReflectionLogger)
}

// installInterceptors applies GEPA interceptors to the program.
func (g *GEPA) installInterceptors(program core.Program) core.Program {
	// Clone the program to avoid modifying the original
	clonedProgram := program.Clone()

	// Install GEPA tracking interceptors using the interceptor chain
	chain := core.NewInterceptorChain().
		AddModuleInterceptor(g.gepaExecutionTracker).
		AddModuleInterceptor(g.gepaReflectionLogger)

	// Apply interceptors to all modules that support them
	for name, module := range clonedProgram.Modules {
		if interceptable, ok := module.(core.InterceptableModule); ok {
			interceptable.SetInterceptors(chain.GetModuleInterceptors())
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
		ContextData: make(map[string]any),
	}

	gepaState.AddTraceContext(ctx, trace)
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
	reflectionData := map[string]any{
		"module_name":    info.ModuleName,
		"execution_time": duration,
		"success":        err == nil,
		"input_types":    g.getInputTypes(inputs),
		"output_types":   g.getOutputTypes(outputs),
		"error_type":     g.getErrorType(err),
	}

	// Store in candidate metrics for later reflection
	gepaState.mu.Lock()
	defer gepaState.mu.Unlock()
	if metrics, exists := gepaState.CandidateMetrics[candidateID]; exists {
		if metrics.Metadata == nil {
			metrics.Metadata = make(map[string]any)
		}
		switch previous := metrics.Metadata["reflection_data"].(type) {
		case []map[string]any:
			metrics.Metadata["reflection_data"] = append(previous, reflectionData)
		case []any: // JSON-decoded checkpoint metadata.
			metrics.Metadata["reflection_data"] = append(previous, reflectionData)
		default:
			metrics.Metadata["reflection_data"] = []map[string]any{reflectionData}
		}
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

// CurrentCandidateID returns the GEPA candidate ID attached to the context, or
// "unknown" when the context is outside an active GEPA evaluation.
func CurrentCandidateID(ctx context.Context) string {
	if candidateID, ok := ctx.Value(candidateIDKey).(string); ok {
		return candidateID
	}
	return "unknown"
}

// Enhanced interceptor chain monitoring

// getRealTimeInterceptorStats provides real-time statistics about interceptor performance.
//
//nolint:unused // This is an advanced feature not yet integrated
func (g *GEPA) getRealTimeInterceptorStats() map[string]any {
	stats := map[string]any{
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

	logger.Info(ctx, "GEPA optimization completed: duration=%v, generations=%d, best_fitness=%.3f",
		duration,
		g.state.CurrentGeneration,
		g.state.BestFitness)
	logger.Info(ctx, "GEPA run control summary: metric_calls=%d, stop_reason=%s",
		g.state.MetricCalls,
		g.state.StopReason)

	if g.state.BestCandidate != nil {
		logger.Info(ctx, "Best candidate found: id=%s, generation=%d, fitness=%.3f, instruction=%s",
			g.state.BestCandidate.ID,
			g.state.BestCandidate.Generation,
			g.state.BestCandidate.Fitness,
			g.state.BestCandidate.Instruction)
	}
	if g.state.BestValidationCandidate != nil {
		logger.Info(ctx, "Best validation candidate found: id=%s, generation=%d, validation_fitness=%.3f, instruction=%s",
			g.state.BestValidationCandidate.ID,
			g.state.BestValidationCandidate.Generation,
			g.state.BestValidationFitness,
			g.state.BestValidationCandidate.Instruction)
	}
}

// Population Management Methods

// initializePopulation creates the initial population of prompt candidates.
func (g *GEPA) initializePopulation(ctx context.Context, program core.Program) error {
	baseComponentTexts := componentTextsFromProgram(program)
	moduleNames := componentModuleNames(baseComponentTexts)
	if len(moduleNames) == 0 {
		return fmt.Errorf("optimizers: seed program must contain at least one module")
	}
	moduleName := moduleNames[0]
	seed := &GEPACandidate{
		ID: g.generateCandidateID(), ModuleName: moduleName,
		Instruction: baseComponentTexts[moduleName], ComponentTexts: baseComponentTexts,
		CreatedAt: time.Now(),
	}
	g.ensureCandidateMetrics(seed.ID)
	population := &Population{
		Candidates: []*GEPACandidate{seed},
	}
	g.state.mu.Lock()
	g.state.PopulationHistory = []*Population{population}
	g.state.mu.Unlock()
	logging.GetLogger().Info(ctx, "Initialized GEPA with the seed program (%d modules)", len(moduleNames))
	return nil
}

// generateCandidateID generates a unique ID for candidates.
func (g *GEPA) generateCandidateID() string {
	return fmt.Sprintf("gepa_%d_%d", time.Now().UnixNano(), g.rng.Intn(10000))
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

// evolvePopulation advances the working population by updating one selected
// candidate at a time. The latest population snapshot remains available for
// history/debugging, but the operational step is now a single proposal/accept
// cycle rather than rebuilding an entire offspring population.
func (g *GEPA) evolvePopulation(ctx context.Context) error {
	logger := logging.GetLogger()
	currentPop := g.getCurrentPopulation()
	if currentPop == nil {
		return fmt.Errorf("no current population found")
	}

	nextGeneration := currentPop.Generation + 1
	logger.Info(ctx, "Updating population for generation %d", nextGeneration)

	source := g.selectCandidateForUpdate(currentPop)
	if source == nil {
		return fmt.Errorf("no candidates available for proposal evolution")
	}

	nextCandidate, attempted := g.proposeNextGenerationCandidate(ctx, currentPop, source, nextGeneration)
	g.ensureCandidateMetrics(nextCandidate.ID)
	g.syncCandidateUpdateState(ctx, nextCandidate)

	newPopulation, replaced := g.populationSnapshotWithReplacement(currentPop, source.ID, nextCandidate, nextGeneration)
	if !replaced {
		return fmt.Errorf("candidate %s not found in current population", source.ID)
	}
	g.recomputePopulationBest(newPopulation)

	// Add to population history
	g.state.mu.Lock()
	g.state.PopulationHistory = append(g.state.PopulationHistory, newPopulation)
	// A skipped mutation provides no evidence of stagnation. Count an update
	// only when a proposal was attempted, including proposals rejected by the
	// metric. Advance the counter here, not each time HasConverged is queried.
	if attempted {
		if newPopulation.BestFitness-currentPop.BestFitness < g.config.ConvergenceThreshold {
			g.state.ConvergenceStatus.StagnationCount++
		} else {
			g.state.ConvergenceStatus.StagnationCount = 0
		}
	}
	g.state.mu.Unlock()
	if newPopulation.BestCandidate != nil {
		g.updateBestCandidate(newPopulation.BestCandidate)
	}

	logger.Info(ctx, "Population updated: generation=%d, source_candidate=%s, replacement_candidate=%s, population_size=%d",
		newPopulation.Generation,
		source.ID,
		nextCandidate.ID,
		len(newPopulation.Candidates))

	return nil
}

func (g *GEPA) proposeNextGenerationCandidate(ctx context.Context, population *Population, source *GEPACandidate, nextGeneration int) (*GEPACandidate, bool) {
	if population == nil || source == nil {
		return nil, false
	}

	componentSelection := g.selectComponentsForUpdate(source)

	merged, mergeAttempted := g.tryAncestorMergeProposal(ctx, population, source, nextGeneration)
	if merged != nil {
		return g.applyComponentSelectionState(merged, componentSelection.nextCursor), true
	}

	var proposed *GEPACandidate
	var mutationAttempted bool
	switch len(componentSelection.modules) {
	case 0:
		proposed = nil
	case 1:
		focusedSource := g.prepareCandidateForComponentUpdate(source, componentSelection.modules[0])
		proposed, mutationAttempted = g.mutate(ctx, focusedSource)
		if proposed == focusedSource {
			proposed = source
		}
	default:
		proposed, mutationAttempted = g.proposeMultiComponentCandidate(ctx, source, componentSelection.modules, nextGeneration)
	}

	if proposed == nil || proposed == source {
		carried := g.copyCandidate(source)
		carried.Generation = nextGeneration
		return g.applyComponentSelectionState(carried, componentSelection.nextCursor), mergeAttempted || mutationAttempted
	}

	if proposed.Generation < nextGeneration {
		proposed.Generation = nextGeneration
	}

	return g.applyComponentSelectionState(proposed, componentSelection.nextCursor), true
}

func (g *GEPA) prepareCandidateForComponentUpdate(source *GEPACandidate, moduleName string) *GEPACandidate {
	if source == nil {
		return nil
	}

	focused := g.copyCandidate(source)
	focused.ModuleName = moduleName
	focused.Instruction = candidateInstructionForModule(source, moduleName)
	return focused
}

type gepaComponentSelectionPlan struct {
	modules    []string
	nextCursor int
}

func (g *GEPA) selectComponentsForUpdate(candidate *GEPACandidate) gepaComponentSelectionPlan {
	if candidate == nil {
		return gepaComponentSelectionPlan{}
	}

	moduleNames := componentModuleNames(cloneCandidateComponentTexts(candidate))
	if len(moduleNames) == 0 {
		moduleName := strings.TrimSpace(candidate.ModuleName)
		if moduleName == "" {
			return gepaComponentSelectionPlan{}
		}
		return gepaComponentSelectionPlan{
			modules:    []string{moduleName},
			nextCursor: 0,
		}
	}

	switch g.config.ComponentSelection {
	case componentSelectionAll:
		return gepaComponentSelectionPlan{
			modules:    moduleNames,
			nextCursor: componentSelectionCursor(candidate),
		}
	case componentSelectionRandom:
		return gepaComponentSelectionPlan{
			modules:    []string{moduleNames[g.rng.Intn(len(moduleNames))]},
			nextCursor: componentSelectionCursor(candidate),
		}
	default:
		cursor := componentSelectionCursor(candidate)
		index := cursor % len(moduleNames)
		return gepaComponentSelectionPlan{
			modules:    []string{moduleNames[index]},
			nextCursor: (index + 1) % len(moduleNames),
		}
	}
}

func componentSelectionCursor(candidate *GEPACandidate) int {
	if candidate == nil || candidate.Metadata == nil {
		return 0
	}

	switch value := candidate.Metadata[gepaComponentSelectionCursorMetadataKey].(type) {
	case int:
		if value >= 0 {
			return value
		}
	case float64:
		if value >= 0 {
			return int(value)
		}
	}

	return 0
}

func (g *GEPA) applyComponentSelectionState(candidate *GEPACandidate, nextCursor int) *GEPACandidate {
	if candidate == nil {
		return nil
	}
	if candidate.Metadata == nil {
		candidate.Metadata = make(map[string]any)
	}
	candidate.Metadata[gepaComponentSelectionCursorMetadataKey] = nextCursor
	return candidate
}

func (g *GEPA) proposeMultiComponentCandidate(ctx context.Context, source *GEPACandidate, modules []string, nextGeneration int) (*GEPACandidate, bool) {
	if source == nil || len(modules) == 0 {
		return nil, false
	}

	working := g.copyCandidate(source)
	componentTexts := cloneCandidateComponentTexts(source)
	if len(componentTexts) == 0 {
		return nil, false
	}

	appliedModules := make([]string, 0, len(modules))
	metadataSources := []map[string]any{source.Metadata}
	focusedModule := strings.TrimSpace(source.ModuleName)

	for _, moduleName := range modules {
		focused := g.copyCandidate(working)
		focused.ComponentTexts = cloneComponentTexts(componentTexts)
		focused.ModuleName = moduleName
		focused.Instruction = candidateInstructionForModule(working, moduleName)

		proposed := g.proposeMutationCandidate(ctx, focused)
		if proposed == nil || proposed == focused {
			continue
		}

		componentTexts = cloneCandidateComponentTexts(proposed)
		working.ComponentTexts = cloneComponentTexts(componentTexts)
		working.ModuleName = moduleName
		working.Instruction = candidateInstructionForModule(proposed, moduleName)
		metadataSources = append(metadataSources, proposed.Metadata)
		appliedModules = append(appliedModules, moduleName)
		focusedModule = moduleName
	}

	if len(appliedModules) == 0 {
		return nil, false
	}

	proposed := &GEPACandidate{
		ID:             g.generateCandidateID(),
		ModuleName:     focusedModule,
		Instruction:    candidateInstructionForModule(working, focusedModule),
		ComponentTexts: cloneComponentTexts(componentTexts),
		Generation:     nextGeneration,
		Fitness:        source.Fitness,
		ParentIDs:      []string{source.ID},
		CreatedAt:      time.Now(),
		Metadata: mergeCandidateMetadata(map[string]any{
			"proposal_type":        "multi_component_update",
			"updated_components":   append([]string(nil), appliedModules...),
			"component_update_all": true,
		}, metadataSources...),
	}

	accepted := g.acceptCandidateProposal(ctx, source, proposed)
	if accepted == source {
		return nil, true
	}
	return accepted, true
}

// selectCandidates returns candidates sampled according to the configured
// selection strategy.
func (g *GEPA) selectCandidates(population *Population, count int) []*GEPACandidate {
	if population == nil || len(population.Candidates) == 0 {
		return nil
	}
	if count <= 0 {
		count = 1
	}

	// Never mix training and holdout scores. A partially validated population
	// falls back to the stable training batch until the next validation pass.
	selectionPopulation, hasCoverage := g.metricSelectionPopulation(population, true)
	if !hasCoverage {
		selectionPopulation = population
	} else if g.config.SelectionStrategy != "pareto" && g.config.SelectionStrategy != "adaptive_pareto" {
		// Coverage is a Pareto sampling weight, not a scalar fitness. In
		// particular, pruning tied winners must not penalize equally accurate
		// candidates in tournament or metric-weighted roulette selection.
		for _, candidate := range selectionPopulation.Candidates {
			candidate.Fitness = g.state.GetCandidateValidationEvaluation(candidate.ID).AverageScore
		}
		g.recomputePopulationBest(selectionPopulation)
	}

	var selected []*GEPACandidate
	switch g.config.SelectionStrategy {
	case "roulette":
		selected = g.rouletteSelection(selectionPopulation, count)
	case "pareto", "adaptive_pareto":
		// Both names use the paper's per-example frontier, sampled in proportion
		// to coverage after pruning redundant tied winners. No heuristic fallback.
		if !hasCoverage {
			selectionPopulation, hasCoverage = g.metricSelectionPopulation(population, false)
		}
		if hasCoverage {
			selected = g.rouletteSelection(selectionPopulation, count)
		} else {
			selected = g.tournamentSelection(population, count)
		}
	default:
		selected = g.tournamentSelection(selectionPopulation, count)
	}

	// Coverage lives only on selection copies; candidate fitness remains the
	// user's average metric for acceptance, convergence, and final selection.
	for i, candidate := range selected {
		selected[i] = g.findPopulationCandidateByID(population, candidate.ID)
	}
	return selected
}

func (g *GEPA) selectCandidateForUpdate(population *Population) *GEPACandidate {
	selected := g.selectCandidates(population, 1)
	if len(selected) > 0 && selected[0] != nil {
		return selected[0]
	}
	if population == nil || len(population.Candidates) == 0 {
		return nil
	}
	return population.Candidates[g.rng.Intn(len(population.Candidates))]
}

// metricSelectionPopulation derives coverage from a complete, aligned batch of
// metric evaluations. It is shared by training and validation selection.
func (g *GEPA) metricSelectionPopulation(population *Population, validation bool) (*Population, bool) {
	if population == nil || len(population.Candidates) == 0 || g.state == nil {
		return nil, false
	}
	evaluations := make(map[string]*gepaCandidateEvaluation, len(population.Candidates))
	var batchKeys []string
	for _, candidate := range population.Candidates {
		if candidate == nil {
			continue
		}
		var evaluation *gepaCandidateEvaluation
		if validation {
			evaluation = g.state.GetCandidateValidationEvaluation(candidate.ID)
		} else {
			evaluation = g.state.GetCandidateEvaluation(candidate.ID)
		}
		if evaluation == nil || len(evaluation.Cases) == 0 {
			return nil, false
		}
		keys := make([]string, len(evaluation.Cases))
		for i, evalCase := range evaluation.Cases {
			keys[i] = exampleCacheKey(evalCase.Example)
		}
		if batchKeys == nil {
			batchKeys = keys
		} else if !slices.Equal(batchKeys, keys) {
			return nil, false
		}
		evaluations[candidate.ID] = evaluation
	}

	frontier, coverage := buildValidationFrontier(evaluations)
	if len(coverage) == 0 {
		return nil, false
	}
	selectionPopulation := &Population{Generation: population.Generation}
	for _, candidate := range population.Candidates {
		if candidate == nil {
			continue
		}
		cloned := g.copyCandidate(candidate)
		cloned.Fitness = float64(coverage[candidate.ID]) / float64(len(frontier))
		selectionPopulation.Candidates = append(selectionPopulation.Candidates, cloned)
	}
	g.recomputePopulationBest(selectionPopulation)
	return selectionPopulation, true
}

func (g *GEPA) findPopulationCandidateByID(population *Population, candidateID string) *GEPACandidate {
	if population == nil || strings.TrimSpace(candidateID) == "" {
		return nil
	}

	for _, candidate := range population.Candidates {
		if candidate != nil && candidate.ID == candidateID {
			return candidate
		}
	}

	return nil
}

func (g *GEPA) populationSnapshotWithReplacement(current *Population, sourceID string, replacement *GEPACandidate, generation int) (*Population, bool) {
	if current == nil {
		return nil, false
	}

	snapshot := &Population{
		Candidates:  make([]*GEPACandidate, len(current.Candidates)),
		Generation:  generation,
		BestFitness: current.BestFitness,
	}

	replaced := false
	for i, candidate := range current.Candidates {
		switch {
		case !replaced && candidate != nil && candidate.ID == sourceID:
			snapshot.Candidates[i] = g.copyCandidate(replacement)
			replaced = true
		default:
			snapshot.Candidates[i] = g.copyCandidate(candidate)
		}
	}

	return snapshot, replaced
}

func (g *GEPA) recomputePopulationBest(population *Population) {
	if population == nil {
		return
	}

	population.BestCandidate = nil
	population.BestFitness = math.Inf(-1)
	for _, candidate := range population.Candidates {
		if candidate == nil {
			continue
		}
		if population.BestCandidate == nil || candidate.Fitness > population.BestFitness {
			population.BestCandidate = candidate
			population.BestFitness = candidate.Fitness
		}
	}
	if population.BestCandidate == nil {
		population.BestFitness = 0.0
	}
}

func (g *GEPA) syncCandidateUpdateState(ctx context.Context, updated *GEPACandidate) {
	if updated == nil {
		return
	}

	evaluation := g.state.GetCandidateEvaluation(updated.ID)
	if evaluation == nil {
		adapter := g.getLatestEvaluationAdapter()
		if adapter != nil {
			evaluation = g.evaluateCandidateWithAdapter(ctx, updated, adapter)
			g.state.UpsertCandidateEvaluation(updated.ID, evaluation)
		}
	}
	if evaluation != nil {
		updated.Fitness = evaluation.AverageScore
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
			if cumulative > spin {
				selected = append(selected, candidate)
				break
			}
		}
	}

	return selected
}

// mutate applies mutation to a candidate and reports whether a proposal was attempted.
func (g *GEPA) mutate(ctx context.Context, candidate *GEPACandidate) (*GEPACandidate, bool) {
	proposed := g.proposeMutationCandidate(ctx, candidate)
	if proposed == candidate {
		return candidate, false
	}

	return g.acceptMutationProposal(ctx, candidate, proposed), true
}

func (g *GEPA) proposeMutationCandidate(ctx context.Context, candidate *GEPACandidate) *GEPACandidate {
	if g.rng.Float64() > g.config.MutationRate {
		return candidate // No mutation
	}

	if proposed := g.reflectionGuidedMutation(ctx, candidate); proposed != nil {
		return proposed
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

	var promptBuilder strings.Builder
	promptBuilder.WriteString(fmt.Sprintf(`Apply a "%s" mutation to this instruction:
Original: "%s"

Mutation requirements for "%s":
%s`,
		mutationType,
		candidate.Instruction,
		mutationType,
		g.getMutationDescription(mutationType)))

	if reflection := g.latestReflectionForCandidate(candidate.ID); reflection != nil {
		if len(reflection.Weaknesses) > 0 {
			promptBuilder.WriteString("\n\nRecent weaknesses to address:\n")
			for _, weakness := range reflection.Weaknesses {
				if strings.TrimSpace(weakness) == "" {
					continue
				}
				promptBuilder.WriteString("- ")
				promptBuilder.WriteString(strings.TrimSpace(weakness))
				promptBuilder.WriteString("\n")
			}
		}
		if len(reflection.Suggestions) > 0 {
			promptBuilder.WriteString("\nRecent improvement suggestions to incorporate:\n")
			for _, suggestion := range reflection.Suggestions {
				if strings.TrimSpace(suggestion) == "" {
					continue
				}
				promptBuilder.WriteString("- ")
				promptBuilder.WriteString(strings.TrimSpace(suggestion))
				promptBuilder.WriteString("\n")
			}
		}
	}

	promptBuilder.WriteString("\nGenerate a mutated version that keeps the core intent but explicitly addresses the recent weaknesses and suggestions when they are relevant.")
	prompt := promptBuilder.String()

	core.RecordModelCall(ctx, g.generationLLM)
	response, err := g.generationLLM.Generate(ctx, prompt)
	if err != nil {
		return g.fallbackMutation(candidate)
	}

	mutatedInstruction := g.extractMutatedInstruction(response.Content)
	if mutatedInstruction == "" || mutatedInstruction == candidate.Instruction {
		return g.fallbackMutation(candidate)
	}

	return &GEPACandidate{
		ID:             g.generateCandidateID(),
		ModuleName:     candidate.ModuleName,
		Instruction:    mutatedInstruction,
		ComponentTexts: deriveCandidateComponentTexts(candidate, candidate.ModuleName, mutatedInstruction),
		Generation:     candidate.Generation + 1,
		ParentIDs:      []string{candidate.ID},
		CreatedAt:      time.Now(),
		Metadata: mergeCandidateMetadata(map[string]any{
			"mutation_type":  mutationType,
			"parent_fitness": candidate.Fitness,
		}, candidate.Metadata),
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

func (g *GEPA) latestReflectionForCandidate(candidateID string) *ReflectionResult {
	if g == nil || candidateID == "" {
		return nil
	}

	g.state.mu.RLock()
	defer g.state.mu.RUnlock()

	for _, reflection := range slices.Backward(g.state.ReflectionHistory) {
		if reflection == nil || reflection.CandidateID != candidateID {
			continue
		}
		cloned := *reflection
		cloned.Strengths = append([]string(nil), reflection.Strengths...)
		cloned.Weaknesses = append([]string(nil), reflection.Weaknesses...)
		cloned.Suggestions = append([]string(nil), reflection.Suggestions...)
		return &cloned
	}
	return nil
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
		ID:             g.generateCandidateID(),
		ModuleName:     candidate.ModuleName,
		Instruction:    mutatedInstruction,
		ComponentTexts: deriveCandidateComponentTexts(candidate, candidate.ModuleName, mutatedInstruction),
		Generation:     candidate.Generation + 1,
		ParentIDs:      []string{candidate.ID},
		CreatedAt:      time.Now(),
		Metadata: mergeCandidateMetadata(map[string]any{
			"mutation_type":  "fallback",
			"parent_fitness": candidate.Fitness,
		}, candidate.Metadata),
	}
}

// copyCandidate creates a deep copy of a candidate.
func (g *GEPA) copyCandidate(original *GEPACandidate) *GEPACandidate {
	metadata := cloneCandidateMetadata(original.Metadata)

	parentIDs := make([]string, len(original.ParentIDs))
	copy(parentIDs, original.ParentIDs)

	demonstrations := make([]core.Example, len(original.Demonstrations))
	copy(demonstrations, original.Demonstrations)

	return &GEPACandidate{
		ID:             original.ID,
		ModuleName:     original.ModuleName,
		Instruction:    original.Instruction,
		ComponentTexts: cloneComponentTexts(original.ComponentTexts),
		Demonstrations: demonstrations,
		Generation:     original.Generation,
		Fitness:        original.Fitness,
		ParentIDs:      parentIDs,
		CreatedAt:      original.CreatedAt,
		Metadata:       metadata,
	}
}

// Population Evaluation Methods

// evaluatePopulation evaluates all candidates using the user's metric.
func (g *GEPA) evaluatePopulation(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) error {
	logger := logging.GetLogger()
	population := g.getCurrentPopulation()
	if population == nil {
		return fmt.Errorf("no current population to evaluate")
	}

	logger.Info(ctx, "Evaluating population: generation=%d, candidates=%d",
		population.Generation,
		len(population.Candidates))

	adapter := g.newEvaluationAdapter(program, dataset, metric)
	g.setLatestEvaluationAdapter(adapter)

	// Use concurrent evaluation with controlled concurrency
	p := pool.New().WithMaxGoroutines(g.config.ConcurrencyLevel)

	var mu sync.Mutex
	evaluatedCount := 0
	candidateEvaluations := make(map[string]*gepaCandidateEvaluation, len(population.Candidates))

	for _, candidate := range population.Candidates {
		candidate := candidate // Capture loop variable
		p.Go(func() {
			evaluation := g.evaluateCandidateWithAdapter(ctx, candidate, adapter)
			fitness := evaluation.AverageScore

			mu.Lock()
			candidate.Fitness = fitness
			evaluatedCount++
			candidateEvaluations[candidate.ID] = evaluation

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
	g.state.SetCandidateEvaluations(candidateEvaluations)
	g.recomputePopulationBest(population)
	// Publish aggregate fitness only after all examples and module calls finish.
	for _, candidate := range population.Candidates {
		g.updateBestCandidate(candidate)
	}

	logger.Info(ctx, "Population evaluation completed: generation=%d, best_fitness=%.3f, evaluated_candidates=%d",
		population.Generation,
		population.BestFitness,
		evaluatedCount)

	return nil
}

func (g *GEPA) applyComponentTexts(program core.Program, componentTexts map[string]string) core.Program {
	modified := program.Clone()
	if len(componentTexts) == 0 {
		return modified
	}

	for moduleName, instruction := range componentTexts {
		module, exists := modified.Modules[moduleName]
		if !exists {
			continue
		}

		sig := module.GetSignature()
		module.SetSignature(core.Signature{
			Instruction: instruction,
			Inputs:      sig.Inputs,
			Outputs:     sig.Outputs,
		})
	}

	return modified
}

// applyCandidate applies the candidate's whole-program instruction state during
// evaluation. ModuleName/Instruction remain the focused component for the
// current update step, while ComponentTexts is the authoritative program state.
func (g *GEPA) applyCandidate(program core.Program, candidate *GEPACandidate) core.Program {
	if candidate == nil {
		return program.Clone()
	}

	componentTexts := cloneCandidateComponentTexts(candidate)
	if len(componentTexts) > 0 {
		return g.applyComponentTexts(program, componentTexts)
	}

	moduleName := strings.TrimSpace(candidate.ModuleName)
	if moduleName == "" {
		return program.Clone()
	}

	instruction := candidateInstructionForModule(candidate, moduleName)
	if instruction == "" {
		return program.Clone()
	}

	return g.applyComponentTexts(program, map[string]string{moduleName: instruction})
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
	latestReflections := make(map[string]*ReflectionResult, len(population.Candidates))

	for _, candidate := range population.Candidates {
		traces := g.state.GetTracesForCandidate(candidate.ID)
		evaluation := g.state.GetCandidateEvaluation(candidate.ID)
		if len(traces) == 0 && evaluation == nil {
			continue // Skip candidates with no reflection evidence
		}

		reflection, err := g.reflectOnCandidate(ctx, candidate, evaluation, traces)
		if err != nil {
			logger.Error(ctx, "Failed to reflect on candidate %s: %v",
				candidate.ID, err)
			continue
		}

		reflections = append(reflections, reflection)
		latestReflections[candidate.ID] = reflection
	}

	// Store reflection results
	g.state.mu.Lock()
	g.state.ReflectionHistory = append(g.state.ReflectionHistory, reflections...)
	g.state.mu.Unlock()
	g.state.SetCandidateReflections(latestReflections)

	logger.Info(ctx, "Reflection analysis completed: generation=%d, reflections_generated=%d",
		generation,
		len(reflections))

	return nil
}

// reflectOnCandidate performs reflection analysis on a single candidate.
func (g *GEPA) reflectOnCandidate(ctx context.Context, candidate *GEPACandidate,
	evaluation *gepaCandidateEvaluation, traces []ExecutionTrace) (*ReflectionResult, error) {

	// Analyze execution patterns
	patterns := g.analyzeExecutionPatterns(traces)
	reflectionInput := g.buildReflectionInput(evaluation)

	// Generate reflection prompt
	prompt := g.buildReflectionPrompt(candidate, patterns, reflectionInput)

	// Get reflection from LLM
	core.RecordModelCall(ctx, g.reflectionLLM)
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
			RichTraceEvidence: make([]string, 0),
		}
	}

	patterns := &ExecutionPatterns{
		ErrorDistribution: make(map[string]int),
		PerformanceTrends: make([]float64, 0),
		RichTraceEvidence: make([]string, 0),
	}

	totalDuration := time.Duration(0)
	successCount := 0
	evidenceCounts := make(map[string]int)

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

		for _, evidence := range extractRichTraceEvidence(trace) {
			evidenceCounts[evidence]++
		}
	}

	patterns.TotalExecutions = len(traces)
	patterns.SuccessCount = successCount
	patterns.SuccessRate = float64(successCount) / float64(len(traces))
	patterns.AverageResponseTime = totalDuration / time.Duration(len(traces))
	patterns.CommonFailures = g.identifyCommonFailures(patterns.ErrorDistribution)
	patterns.QualityIndicators = g.identifyQualityIndicators(patterns.PerformanceTrends)
	patterns.RichTraceEvidence = topRichTraceEvidence(evidenceCounts, 6)

	return patterns
}

// buildReflectionPrompt creates a reflection prompt for a candidate.
func (g *GEPA) buildReflectionPrompt(candidate *GEPACandidate, patterns *ExecutionPatterns, reflectionInput *gepaReflectionInput) string {
	richTraceEvidence := "none recorded"
	if len(patterns.RichTraceEvidence) > 0 {
		richTraceEvidence = "- " + strings.Join(patterns.RichTraceEvidence, "\n- ")
	}
	caseEvidence := g.formatReflectionCaseEvidence(reflectionInput)

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

RICH TRACE EVIDENCE:
%s

EXAMPLE-LEVEL EVIDENCE:
%s

Please provide a detailed self-reflection covering:

1. STRENGTHS: What aspects of this instruction work well?
2. WEAKNESSES: What specific issues limit its effectiveness? Use the rich trace evidence and example-level evidence sections to ground concrete failure modes, loops, mismatches, bad outputs, and error cases.
3. IMPROVEMENT SUGGESTIONS: Concrete ways to enhance this instruction based on the execution patterns, rich trace evidence, and example-level evidence.
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
		strings.Join(patterns.QualityIndicators, ", "),
		richTraceEvidence,
		caseEvidence)
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
	if len(patterns.RichTraceEvidence) > 0 {
		weaknesses = append(weaknesses, fmt.Sprintf("Trace evidence: %s", strings.Join(patterns.RichTraceEvidence, "; ")))
		suggestions = append(suggestions, "Use the trace evidence to target repeated failure loops, mismatches, and termination issues")
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

func extractRichTraceEvidence(trace ExecutionTrace) []string {
	if trace.ContextData == nil {
		return nil
	}

	if raw, ok := trace.ContextData["rich_trace_evidence"]; ok {
		if evidence := utils.DedupeStrings(evidenceFromValue(raw), 0); len(evidence) > 0 {
			return evidence
		}
	}

	evidence := make([]string, 0, 8)
	if termination, ok := trace.ContextData["termination_cause"].(string); ok && strings.TrimSpace(termination) != "" {
		evidence = append(evidence, "termination="+termination)
	}
	if status, ok := trace.ContextData["trace_status"].(string); ok && strings.TrimSpace(status) != "" {
		evidence = append(evidence, "trace_status="+status)
	}
	if failedTests, ok := trace.ContextData["failed_tests"]; ok {
		for _, item := range evidenceFromValue(failedTests) {
			evidence = append(evidence, "failed_test="+item)
		}
	}
	if diagnostics, ok := trace.ContextData["diagnostics"].(map[string]any); ok {
		for _, key := range []string{"evaluation_error", "execution_error", "comparison_error"} {
			if raw, exists := diagnostics[key]; exists {
				if message, ok := raw.(string); ok && strings.TrimSpace(message) != "" {
					evidence = append(evidence, fmt.Sprintf("%s=%s", key, truncateEvidence(message, 100)))
				}
			}
		}
	}
	if len(evidence) == 0 {
		if summary, ok := trace.ContextData["rich_trace_summary"].(string); ok && strings.TrimSpace(summary) != "" {
			evidence = append(evidence, truncateEvidence(summary, 160))
		}
	}

	return utils.DedupeStrings(evidence, 0)
}

func evidenceFromValue(raw any) []string {
	switch value := raw.(type) {
	case []string:
		return append([]string(nil), value...)
	case []any:
		result := make([]string, 0, len(value))
		for _, item := range value {
			if str, ok := item.(string); ok && strings.TrimSpace(str) != "" {
				result = append(result, str)
			}
		}
		return result
	default:
		return nil
	}
}

func topRichTraceEvidence(counts map[string]int, limit int) []string {
	if len(counts) == 0 {
		return []string{}
	}

	type countedEvidence struct {
		evidence string
		count    int
	}

	items := make([]countedEvidence, 0, len(counts))
	for evidence, count := range counts {
		if strings.TrimSpace(evidence) == "" {
			continue
		}
		items = append(items, countedEvidence{evidence: evidence, count: count})
	}

	sort.Slice(items, func(i, j int) bool {
		if items[i].count == items[j].count {
			return items[i].evidence < items[j].evidence
		}
		return items[i].count > items[j].count
	})

	if limit > 0 && len(items) > limit {
		items = items[:limit]
	}

	result := make([]string, 0, len(items))
	for _, item := range items {
		result = append(result, fmt.Sprintf("%s (seen %dx)", item.evidence, item.count))
	}
	return result
}

func truncateEvidence(value string, limit int) string {
	if limit <= 0 || len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
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
	g.state.mu.Lock()
	defer g.state.mu.Unlock()

	if g.config.StagnationLimit <= 0 {
		return false
	}
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

	if g.state.ConvergenceStatus.StagnationCount >= g.config.StagnationLimit {
		g.state.ConvergenceStatus.IsConverged = true
		return true
	}

	return false
}

// applyBestCandidate applies the best whole-program candidate to the final
// program. This intentionally prefers a coherent candidate state over the older
// "best module from anywhere in history" composition, matching whole-program
// GEPA semantics even when that means a module-specific local winner from a
// different candidate is not stitched into the final result.
func (g *GEPA) applyBestCandidate(program core.Program) core.Program {
	bestCandidate, bestFitness, usingValidation := g.bestCandidateForApplication()
	if bestCandidate == nil {
		logging.GetLogger().Warn(context.Background(), "No best candidate found, returning original program")
		return program
	}

	logger := logging.GetLogger()
	if usingValidation {
		logger.Info(context.Background(), "Applying best validation candidate to program: id=%s, validation_fitness=%.3f, training_fitness=%.3f, instruction=%s",
			bestCandidate.ID,
			bestFitness,
			bestCandidate.Fitness,
			bestCandidate.Instruction)
	} else {
		logger.Info(context.Background(), "Applying best candidate to program: id=%s, fitness=%.3f, instruction=%s",
			bestCandidate.ID,
			bestCandidate.Fitness,
			bestCandidate.Instruction)
	}

	return g.applyCandidate(program, bestCandidate)
}

func (g *GEPA) bestCandidateForApplication() (*GEPACandidate, float64, bool) {
	if g == nil {
		return nil, 0, false
	}

	return g.state.BestCandidateForApplication()
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
