package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/sourcegraph/conc/pool"
)

// StrategyType defines the optimization strategy type.
type StrategyType string

const (
	// InstructionPerturbation is the original strategy that modifies instructions.
	InstructionPerturbation StrategyType = "instruction_perturbation"
	// RuleGeneration is the new strategy that generates rules from trajectories.
	RuleGeneration StrategyType = "rule_generation"
)

// SIMBAConfig contains configuration options for SIMBA optimizer.
type SIMBAConfig struct {
	// Mini-batch configuration
	BatchSize     int `json:"batch_size"`     // Default: 32
	MaxSteps      int `json:"max_steps"`      // Default: 8
	NumCandidates int `json:"num_candidates"` // Default: 6

	// Temperature controls
	SamplingTemperature float64 `json:"sampling_temperature"` // Default: 0.2

	// Introspective learning; zero disables it. Advice feeds the next proposals.
	IntrospectionFrequency int `json:"introspection_frequency"` // Default: 2

	// Deprecated: ConvergenceThreshold no longer stops the run. Use EarlyStoppingPatience.
	ConvergenceThreshold float64 `json:"convergence_threshold"`
	MinImprovementRatio  float64 `json:"min_improvement_ratio"` // Default: 0.05

	// Concurrency and resources
	MaxGoroutines  int `json:"max_goroutines"`  // Default: 10 (for non-LLM operations)
	LLMConcurrency int `json:"llm_concurrency"` // Default: 0 (unlimited for LLM calls)

	// Strategy configuration
	StrategyMode       string  `json:"strategy_mode"`        // Default: "both" (both, instruction_only, rule_only, demo_only)
	StrategyRatio      float64 `json:"strategy_ratio"`       // Default: 0.5 (probability of a demo proposal in both mode)
	MaxDemos           int     `json:"max_demos"`            // Default: 4 per predictor; zero disables demo proposals
	DemoScoreThreshold float64 `json:"demo_score_threshold"` // Default: 1; minimum successful program score for demos

	// Bucket sorting configuration
	UseBucketSorting      bool      `json:"use_bucket_sorting"`      // Default: false
	BucketSortingCriteria []string  `json:"bucket_sorting_criteria"` // Default: ["max_to_min_gap", "max_score", "max_to_avg_gap"]
	BucketSortingWeights  []float64 `json:"bucket_sorting_weights"`  // Default: [0.4, 0.4, 0.2]

	// Pipeline processing configuration
	UsePipelineProcessing bool `json:"use_pipeline_processing"` // Default: false
	PipelineBufferSize    int  `json:"pipeline_buffer_size"`    // Default: 2

	// Early stopping configuration
	EarlyStoppingPatience  int     `json:"early_stopping_patience"`  // Default: 0 (disabled)
	EarlyStoppingThreshold float64 `json:"early_stopping_threshold"` // Default: 0.01

	// Fast mode configuration for Python compatibility
	FastMode                       bool `json:"fast_mode"`                        // Default: false
	DisableTrajectoryTracking      bool `json:"disable_trajectory_tracking"`      // Default: false
	DisableRuleGeneration          bool `json:"disable_rule_generation"`          // Default: false
	DisableInstructionPerturbation bool `json:"disable_instruction_perturbation"` // Default: false
}

// Trajectory represents an execution trajectory for rule extraction.
type Trajectory struct {
	Example         core.Example
	Prediction      map[string]any
	Score           float64
	Success         bool
	ProgramID       string // To track which program generated this
	ExecutionTime   time.Duration
	PredictorTraces []SIMBAPredictorTrace // Predictor-local inputs and parsed outputs, in execution order.
}

// SIMBAState tracks optimization progress and history.
type SIMBAState struct {
	CurrentStep      int
	BestScore        float64
	BestProgram      core.Program
	CandidateHistory []CandidateResult
	PerformanceLog   []StepResult
	IntrospectionLog []string
	StartTime        time.Time
	Trajectories     []Trajectory // Track execution trajectories for rule extraction
}

// CandidateResult represents a candidate program and its performance.
type CandidateResult struct {
	Program     core.Program       `json:"-"`
	Score       float64            `json:"score"`
	Step        int                `json:"step"`
	Temperature float64            `json:"temperature"`
	CreatedAt   time.Time          `json:"created_at"`
	Metadata    *CandidateMetadata `json:"metadata,omitempty"`
}

// CandidateMetadata contains detailed performance metrics for a candidate.
type CandidateMetadata struct {
	// Individual performance metrics
	IndividualScores []float64 `json:"individual_scores"`
	DiversityScore   float64   `json:"diversity_score"`
	ImprovementDelta float64   `json:"improvement_delta"`

	// Multi-criteria scores
	MaxToMinGap float64 `json:"max_to_min_gap"`
	MaxScore    float64 `json:"max_score"`
	MaxToAvgGap float64 `json:"max_to_avg_gap"`

	// Selection tracking
	SelectionRank    int     `json:"selection_rank"`
	BucketAssignment int     `json:"bucket_assignment"`
	CompositeScore   float64 `json:"composite_score"`
}

// candidatePair holds a candidate program with its metadata for bucket sorting.
type candidatePair struct {
	candidate core.Program
	score     float64
	metadata  CandidateMetadata
	index     int
}

// StepResult captures metrics for each optimization step.
type StepResult struct {
	Step            int           `json:"step"`
	BestScore       float64       `json:"best_score"`
	CandidateScores []float64     `json:"candidate_scores"`
	Temperature     float64       `json:"temperature"`
	BatchSize       int           `json:"batch_size"`
	Introspection   string        `json:"introspection,omitempty"`
	Duration        time.Duration `json:"duration"`
	Improvement     float64       `json:"improvement"`
}

// IntrospectionResult contains self-analysis and advice.
type IntrospectionResult struct {
	Analysis             string   `json:"analysis"`
	Recommendations      []string `json:"recommendations"`
	Confidence           float64  `json:"confidence"`
	IdentifiedPatterns   []string `json:"identified_patterns"`
	SuggestedAdjustments []string `json:"suggested_adjustments"`
}

// Pipeline processing structures for overlapping execution

// PipelineStage represents a pipeline stage with candidates and associated data.
type PipelineStage struct {
	StepIndex  int
	Candidates []core.Program
	Batch      []core.Example
	Scores     []float64
	Timestamp  time.Time
	Error      error
}

// PipelineChannels contains channels for pipeline communication.
type PipelineChannels struct {
	CandidateGeneration chan *PipelineStage
	BatchSampling       chan *PipelineStage
	CandidateEvaluation chan *PipelineStage
	Results             chan *PipelineStage
	Errors              chan error
	Done                chan struct{}
}

// PipelineResult represents the result of a pipeline stage.
type PipelineResult struct {
	StepIndex      int
	BestProgram    core.Program
	BestScore      float64
	AllScores      []float64
	ProcessingTime time.Duration
	StageTimings   map[string]time.Duration
}

// SIMBA implements Stochastic Introspective Mini-Batch Ascent. It retains scored
// proposals, softmax-samples parents from the top programs plus the baseline,
// and builds demos or predictor-specific rules from current, same-example traces.
// Each evidence-enabled step runs up to NumCandidates additional programs on the
// minibatch. Only the step winner is validated on the full dataset; Compile
// returns the best full-dataset result, never merely the best minibatch estimate.
type SIMBA struct {
	// Core configuration
	config SIMBAConfig

	// Language models
	primaryModel  core.PromptModel // Primary optimization model
	analyzerModel core.PromptModel // Introspective analysis model

	// Internal state
	state  *SIMBAState
	logger *logging.Logger
	rng    *rand.Rand
	search *simbaSearch // Owned by the current compile; pipeline stages exchange ownership.

	// Pipeline processing
	pipelineChannels *PipelineChannels

	// Set per compile, under compileGate: whether the full-dataset pass may
	// run concurrently when LLMConcurrency is unset.
	concurrentFullEvaluation bool

	// Thread safety
	compileGate chan struct{}
	mu          sync.RWMutex
}

var (
	_ core.Optimizer        = (*SIMBA)(nil)
	_ core.ExamplesCompiler = (*SIMBA)(nil)
	_ core.ScorerCompiler   = (*SIMBA)(nil)
)

// SIMBAOption defines functional options for SIMBA configuration.
type SIMBAOption func(*SIMBA)

// WithSIMBABatchSize sets the mini-batch size.
func WithSIMBABatchSize(size int) SIMBAOption {
	return func(s *SIMBA) {
		s.config.BatchSize = size
	}
}

// WithSIMBAMaxSteps sets the maximum optimization steps.
func WithSIMBAMaxSteps(steps int) SIMBAOption {
	return func(s *SIMBA) {
		s.config.MaxSteps = steps
	}
}

// WithSIMBANumCandidates sets the number of candidate programs per iteration,
// including the incumbent. Compile rejects values below 2.
func WithSIMBANumCandidates(num int) SIMBAOption {
	return func(s *SIMBA) {
		s.config.NumCandidates = num
	}
}

// WithSamplingTemperature sets the decaying softmax temperature for source
// programs and exploratory step-winner selection. Zero chooses the highest score.
func WithSamplingTemperature(temperature float64) SIMBAOption {
	return func(s *SIMBA) {
		s.config.SamplingTemperature = temperature
	}
}

// WithSIMBAStrategyMode selects both (demos and rules), instruction_only,
// rule_only, or demo_only. Missing usable evidence falls back to instruction
// perturbation, except in demo_only mode, which leaves that proposal unchanged.
func WithSIMBAStrategyMode(mode string) SIMBAOption {
	return func(s *SIMBA) {
		s.config.StrategyMode = mode
	}
}

// WithSIMBADemoScoreThreshold sets the minimum program score that certifies a
// successful demonstration. The default is 1 for accuracy metrics; set this for
// metrics with another scale. Failed executions never supply demonstrations.
func WithSIMBADemoScoreThreshold(score float64) SIMBAOption {
	return func(s *SIMBA) { s.config.DemoScoreThreshold = score }
}

// WithSIMBAMaxDemos sets the hard cap on demonstrations per predictor in new
// proposals. Zero disables appending demos; the input program is never modified.
func WithSIMBAMaxDemos(count int) SIMBAOption {
	return func(s *SIMBA) { s.config.MaxDemos = max(0, count) }
}

// WithSIMBAStrategyRatio sets the probability of trying a demo rather than a rule
// in both mode. Previously it allocated instruction-perturbation proposals.
func WithSIMBAStrategyRatio(ratio float64) SIMBAOption {
	return func(s *SIMBA) {
		// Ensure ratio is between 0 and 1
		if ratio < 0 {
			ratio = 0
		} else if ratio > 1 {
			ratio = 1
		}
		s.config.StrategyRatio = ratio
	}
}

// WithBucketSorting controls the legacy weighted step-winner selection. It does
// not disable the same-example evidence buckets, which always use DSPy's ordering.
func WithBucketSorting(enabled bool) SIMBAOption {
	return func(s *SIMBA) {
		s.config.UseBucketSorting = enabled
	}
}

// WithBucketSortingCriteria sets the criteria for bucket sorting.
func WithBucketSortingCriteria(criteria []string) SIMBAOption {
	return func(s *SIMBA) {
		if len(criteria) > 0 {
			s.config.BucketSortingCriteria = make([]string, len(criteria))
			copy(s.config.BucketSortingCriteria, criteria)
		}
	}
}

// WithBucketSortingWeights sets the weights for bucket sorting criteria.
func WithBucketSortingWeights(weights []float64) SIMBAOption {
	return func(s *SIMBA) {
		if len(weights) > 0 {
			// Normalize weights to sum to 1
			total := 0.0
			for _, w := range weights {
				total += w
			}
			if total > 0 {
				s.config.BucketSortingWeights = make([]float64, len(weights))
				for i, w := range weights {
					s.config.BucketSortingWeights[i] = w / total
				}
			}
		}
	}
}

// WithLLMConcurrency sets the concurrency limit for LLM calls.
func WithLLMConcurrency(concurrency int) SIMBAOption {
	return func(s *SIMBA) {
		s.config.LLMConcurrency = concurrency
	}
}

// WithPipelineProcessing enables or disables staged pipeline processing.
// Each step waits for full-dataset acceptance before generating the next step.
// Candidates and examples still run concurrently within a stage; dependent steps
// do not overlap, so they cannot use a stale parent or incomplete trajectories.
func WithPipelineProcessing(enabled bool) SIMBAOption {
	return func(s *SIMBA) {
		s.config.UsePipelineProcessing = enabled
	}
}

// WithPipelineBufferSize sets the buffer size for pipeline channels.
func WithPipelineBufferSize(size int) SIMBAOption {
	return func(s *SIMBA) {
		if size > 0 {
			s.config.PipelineBufferSize = size
		}
	}
}

// WithFastMode configures SIMBA for optimal speed with minimal features.
func WithFastMode(enabled bool) SIMBAOption {
	return func(s *SIMBA) {
		s.config.FastMode = enabled
		if enabled {
			// Disable pipeline processing for small datasets (reduces overhead)
			s.config.UsePipelineProcessing = false
			// Use instruction-only strategy (faster than dual strategy)
			s.config.StrategyMode = "instruction_only"
			// Disable bucket sorting (simpler selection)
			s.config.UseBucketSorting = false
			// Disable introspection completely for speed
			s.config.IntrospectionFrequency = 0
			// Use maximum LLM concurrency
			s.config.LLMConcurrency = 0
			// Ultra-small batch size for speed
			s.config.BatchSize = 2
			// Minimal candidates for speed
			s.config.NumCandidates = 2
			// Very aggressive early stopping
			s.config.EarlyStoppingPatience = 1
			s.config.EarlyStoppingThreshold = 0.001
			// Reduce max steps
			s.config.MaxSteps = 3
			// Disable expensive features for Python compatibility
			s.config.DisableTrajectoryTracking = true
			s.config.DisableRuleGeneration = true
			s.config.DisableInstructionPerturbation = false
		}
	}
}

// NewSIMBA creates a new SIMBA optimizer.
func NewSIMBA(opts ...SIMBAOption) *SIMBA {
	s := &SIMBA{
		config: SIMBAConfig{
			BatchSize:              32,
			MaxSteps:               8,
			NumCandidates:          6,
			SamplingTemperature:    0.2,
			IntrospectionFrequency: 2,
			ConvergenceThreshold:   0.001,
			MinImprovementRatio:    0.05,
			MaxGoroutines:          10,
			LLMConcurrency:         0, // 0 = unlimited for LLM calls
			StrategyMode:           "both",
			StrategyRatio:          0.5,
			MaxDemos:               4,
			DemoScoreThreshold:     1,
			UseBucketSorting:       false,
			BucketSortingCriteria:  []string{"max_to_min_gap", "max_score", "max_to_avg_gap"},
			BucketSortingWeights:   []float64{0.4, 0.4, 0.2},
			UsePipelineProcessing:  false,
			PipelineBufferSize:     2,
			EarlyStoppingPatience:  0, // Disabled by default
			EarlyStoppingThreshold: 0.01,
		},
		state: &SIMBAState{
			CurrentStep:      0,
			BestScore:        0.0,
			CandidateHistory: make([]CandidateResult, 0),
			PerformanceLog:   make([]StepResult, 0),
			IntrospectionLog: make([]string, 0),
			StartTime:        time.Now(),
			Trajectories:     make([]Trajectory, 0),
		},
		logger:      logging.GetLogger(),
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
		compileGate: make(chan struct{}, 1),
	}

	// Apply options
	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Compile implements the legacy cursor-based Optimizer API. It materializes
// the dataset once before entering the optimizer.
func (s *SIMBA) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	ctx = normalizeSIMBAContext(ctx)
	if err := s.acquireCompile(ctx); err != nil {
		return program, err
	}
	defer s.releaseCompile()

	scorer := core.MetricScorer(metric)
	ctx, err := s.prepareCompile(ctx, scorer)
	if err != nil {
		return program, err
	}
	s.concurrentFullEvaluation = false
	examples, err := core.MaterializeDatasetContext(ctx, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to materialize dataset: %w", err)
	}
	return s.compileExamples(ctx, program, examples, scorer)
}

// CompileExamples optimizes a program from materialized, read-only examples.
func (s *SIMBA) CompileExamples(ctx context.Context, program core.Program, examples []core.Example, metric core.Metric) (core.Program, error) {
	return s.compileScored(ctx, program, examples, core.MetricScorer(metric), false)
}

// CompileWithScorer optimizes a program from materialized, read-only examples. It
// calls scorer concurrently, bounded by LLMConcurrency. Unlike Compile, it also
// scores the full dataset concurrently when LLMConcurrency is unset, with up to
// BatchSize*NumCandidates program calls in flight (192 with the defaults). An example
// the scorer cannot score rejects the candidate, like a failed execution.
func (s *SIMBA) CompileWithScorer(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	return s.compileScored(ctx, program, examples, scorer, true)
}

func (s *SIMBA) compileScored(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer, concurrentFullEvaluation bool) (core.Program, error) {
	ctx = normalizeSIMBAContext(ctx)
	if err := s.acquireCompile(ctx); err != nil {
		return program, err
	}
	defer s.releaseCompile()

	ctx, err := s.prepareCompile(ctx, scorer)
	if err != nil {
		return program, err
	}
	s.concurrentFullEvaluation = concurrentFullEvaluation
	return s.compileExamples(ctx, program, examples, scorer)
}

func normalizeSIMBAContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func (s *SIMBA) acquireCompile(ctx context.Context) error {
	select {
	case s.compileGate <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *SIMBA) releaseCompile() {
	<-s.compileGate
}

func (s *SIMBA) prepareCompile(ctx context.Context, scorer core.Scorer) (context.Context, error) {
	if err := ctx.Err(); err != nil {
		return ctx, err
	}
	if scorer == nil {
		return ctx, errors.New(errors.InvalidInput, "SIMBA requires a metric function")
	}
	if s.config.NumCandidates < 2 {
		return ctx, errors.New(errors.InvalidInput, "SIMBA requires at least 2 candidates (the incumbent and a proposal)")
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}
	s.resetState()
	return ctx, nil
}

func (s *SIMBA) compileExamples(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	if err := requireRebindableProgram(program); err != nil {
		return program, err
	}
	ctx, span := core.StartSpan(ctx, "SIMBA.Compile")
	defer core.EndSpan(ctx)

	s.logger.Info(ctx, "Starting SIMBA optimization with config: batch_size=%d, max_steps=%d, num_candidates=%d, strategy_mode=%s, strategy_ratio=%.2f, llm_concurrency=%d",
		s.config.BatchSize, s.config.MaxSteps, s.config.NumCandidates, s.config.StrategyMode, s.config.StrategyRatio, s.config.LLMConcurrency)

	// Initialize models if not set
	if s.primaryModel == nil {
		s.primaryModel = core.GetDefaultLLM()
	}
	if s.analyzerModel == nil {
		s.analyzerModel = core.GetTeacherLLM()
		if s.analyzerModel == nil {
			s.analyzerModel = s.primaryModel
		}
	}

	if len(examples) == 0 {
		return program, errors.New(errors.ValidationFailed, "dataset is empty")
	}

	// Initialize with current program as best candidate
	bestProgram := program.Clone()
	initialScore, err := s.evaluateProgramOnExamples(ctx, bestProgram, examples, scorer)
	if err != nil {
		return program, err
	}
	s.state.BestScore = initialScore
	s.state.BestProgram = bestProgram
	s.search = &simbaSearch{pool: []simbaPoolEntry{{program: bestProgram, fullScore: &initialScore}}}

	s.logger.Info(ctx, "Initial program score: %.4f", initialScore)

	// Choose optimization mode: pipeline processing or sequential processing
	if s.config.UsePipelineProcessing {
		s.logger.Info(ctx, "Using pipeline processing mode with buffer size %d", s.config.PipelineBufferSize)
		return s.runPipelineProcessing(ctx, bestProgram, examples, scorer)
	}

	s.logger.Info(ctx, "Using sequential processing mode")
	// The incumbent is advanced using minibatch comparisons, but BestProgram is
	// tracked only with full-dataset scores. This avoids comparing estimates from
	// different sample populations while requiring just one full evaluation per
	// step (rather than one per generated candidate).
	incumbentProgram := bestProgram

	// Main optimization loop (sequential processing)
	for step := 0; step < s.config.MaxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return program, err
		}
		stepCtx, stepSpan := core.StartSpan(ctx, fmt.Sprintf("SIMBA.Step.%d", step))
		stepStart := time.Now()

		s.mu.Lock()
		s.state.CurrentStep = step
		s.mu.Unlock()

		// Evidence and proposals use the same examples in this step.
		batch, err := s.prepareStep(stepCtx, examples, scorer)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, err
		}
		candidates, err := s.generateCandidates(stepCtx, incumbentProgram)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, errors.WithFields(
				errors.Wrap(err, errors.ValidationFailed, "failed to generate candidates"),
				errors.Fields{"step": step, "error": err.Error()},
			)
		}

		// generateCandidates includes the incumbent as its first candidate, so it
		// is scored on the exact same sampled batch as every proposed replacement.
		scores, err := s.evaluateCandidates(stepCtx, candidates, batch, scorer)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, err
		}

		// Select best candidate using temperature-controlled sampling
		selectedProgram, _ := s.selectBestCandidate(candidates, scores)
		incumbentProgram = selectedProgram.Clone()

		// Full-evaluate only the minibatch winner so globally returned programs are
		// ranked using comparable full-dataset measurements.
		selectedFullScore, err := s.evaluateProgramOnExamples(stepCtx, selectedProgram, examples, scorer)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, err
		}
		s.rememberCandidates(candidates, scores, selectedProgram, selectedFullScore)
		improvement := selectedFullScore - s.state.BestScore
		if selectedFullScore > s.state.BestScore {
			s.mu.Lock()
			s.state.BestScore = selectedFullScore
			s.state.BestProgram = selectedProgram.Clone()
			s.mu.Unlock()
			s.logger.Info(stepCtx, "New best full-dataset score: %.4f (improvement: +%.4f)", selectedFullScore, improvement)
		}

		// A minibatch win must not move the search to a worse full-set lineage.
		// Keep tied proposals available for exploration.
		if selectedFullScore < s.state.BestScore || math.IsNaN(selectedFullScore) {
			incumbentProgram = s.state.BestProgram
		}

		// Record step metrics
		stepDuration := time.Since(stepStart)
		stepResult := StepResult{
			Step:            step,
			BestScore:       s.state.BestScore,
			CandidateScores: scores,
			Temperature:     s.getCurrentTemperature(step),
			BatchSize:       len(batch),
			Duration:        stepDuration,
			Improvement:     improvement,
		}

		s.recordStep(stepCtx, stepResult)

		// Check for early stopping
		if s.shouldStopEarly() {
			s.logger.Info(stepCtx, "Early stopping triggered at step %d", step)
			stepSpan.WithAnnotation("early_stopped", true)
			core.EndSpan(stepCtx)
			break
		}

		stepSpan.WithAnnotation("best_score", s.state.BestScore)
		stepSpan.WithAnnotation("improvement", improvement)
		core.EndSpan(stepCtx)
	}

	totalDuration := time.Since(s.state.StartTime)
	s.logger.Info(ctx, "SIMBA optimization completed: final_score=%.4f, steps=%d, duration=%v",
		s.state.BestScore, s.state.CurrentStep+1, totalDuration)

	span.WithAnnotation("final_score", s.state.BestScore)
	span.WithAnnotation("total_steps", s.state.CurrentStep+1)
	span.WithAnnotation("duration", totalDuration.String())

	return s.state.BestProgram, nil
}

// Helper methods for SIMBA implementation

// createLLMPool creates a pool for LLM operations with appropriate concurrency limits.
func (s *SIMBA) createLLMPool() *pool.Pool {
	if s.config.LLMConcurrency <= 0 {
		// Unlimited concurrency for LLM calls
		return pool.New()
	}
	return pool.New().WithMaxGoroutines(s.config.LLMConcurrency)
}

// resetState initializes the optimization state.
func (s *SIMBA) resetState() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.search = nil
	s.state.CurrentStep = 0
	s.state.BestScore = 0.0
	s.state.BestProgram = core.Program{}
	s.state.CandidateHistory = make([]CandidateResult, 0)
	s.state.PerformanceLog = make([]StepResult, 0)
	s.state.IntrospectionLog = make([]string, 0)
	s.state.StartTime = time.Now()
	s.state.Trajectories = make([]Trajectory, 0)
}

func (s *SIMBA) evaluateProgramOnExamples(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (float64, error) {
	// Honor LLMConcurrency like batch scoring does. Without one the pass stays
	// sequential, as it always was, unless the caller compiled with a Scorer;
	// then it is bounded by the burst one optimization step already issues.
	workers := s.config.LLMConcurrency
	if workers <= 0 && s.concurrentFullEvaluation {
		workers = s.config.BatchSize * s.config.NumCandidates
	}
	results := scoreExamples(ctx, workers, false, examples, func(example core.Example) (float64, error) {
		prediction, err := program.Forward(ctx, example.Inputs)
		if err != nil {
			return 0, err
		}
		return scorer.Score(ctx, example, prediction)
	})
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	var totalScore float64
	failed := false
	for _, result := range results {
		if result.Err != nil {
			s.logger.Debug(ctx, "Program evaluation failed for example: %v", result.Err)
			failed = true
			continue
		}
		totalScore += result.Score
	}

	if len(results) == 0 {
		return 0, nil
	}
	if failed {
		return math.Inf(-1), nil
	}
	return totalScore / float64(len(results)), nil
}

// generateCandidates creates candidate programs for the current step.
func (s *SIMBA) generateCandidates(ctx context.Context, baseProgram core.Program) ([]core.Program, error) {
	candidates := []core.Program{baseProgram.Clone()}
	results := make([]core.Program, s.config.NumCandidates-1)
	type proposal struct {
		parent core.Program
		mode   string
		bucket *simbaBucket
		rng    *rand.Rand
	}
	plans := make([]proposal, len(results))
	for i := range plans {
		// Sample parents and strategies in candidate order, not goroutine order.
		parent := s.sampleSource(baseProgram)
		mode := s.config.StrategyMode
		if mode == "both" {
			s.mu.Lock()
			demo := s.config.MaxDemos > 0 && s.rng.Float64() < s.config.StrategyRatio
			s.mu.Unlock()
			mode = "rule_only"
			if demo {
				mode = "demo"
			}
		}
		if (s.config.DisableRuleGeneration && mode == "rule_only") ||
			(s.config.DisableTrajectoryTracking && mode != "demo_only") {
			mode = "instruction_only"
		}
		var bucket *simbaBucket
		if s.search != nil && len(s.search.buckets) > 0 {
			bucket = &s.search.buckets[i%len(s.search.buckets)]
		}
		s.mu.Lock()
		seed := s.rng.Int63()
		s.mu.Unlock()
		plans[i] = proposal{parent: parent, mode: mode, bucket: bucket, rng: rand.New(rand.NewSource(seed))}
	}
	p := s.createLLMPool()
	for i, plan := range plans {
		p.Go(func() {
			candidate, err := s.proposeFromBucket(context.WithValue(ctx, simbaProposalRNGKey{}, plan.rng), plan.parent, plan.mode, plan.bucket)
			if err != nil {
				s.logger.Debug(ctx, "Failed to generate candidate %d: %v", i, err)
				return
			}
			results[i] = candidate
		})
	}
	p.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, candidate := range results {
		if candidate.Modules != nil {
			s.limitProposalDemos(candidate)
			candidates = append(candidates, candidate)
		}
	}
	return candidates, nil
}

type simbaProposalRNGKey struct{}

// proposalIntn draws the random choices of one proposal. Proposals of a step
// are generated concurrently, so each draws from its own generator, seeded in
// candidate order; sharing s.rng would make a seeded run depend on goroutine
// scheduling. Outside candidate generation it falls back to s.rng.
func (s *SIMBA) proposalIntn(ctx context.Context, n int) int {
	if rng, ok := ctx.Value(simbaProposalRNGKey{}).(*rand.Rand); ok {
		return rng.Intn(n)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rng.Intn(n)
}

func (s *SIMBA) proposeFromBucket(ctx context.Context, parent core.Program, mode string, bucket *simbaBucket) (core.Program, error) {
	if mode == "demo" || mode == "demo_only" {
		if bucket != nil {
			if candidate, ok := s.appendDemo(parent, *bucket); ok {
				return candidate, nil
			}
		}
		if mode == "demo_only" {
			return parent.Clone(), nil
		}
	}
	if s.config.DisableInstructionPerturbation {
		return parent.Clone(), nil
	}
	if mode == "rule_only" {
		var rules []string
		var err error
		if bucket != nil {
			return s.generateBucketRuleCandidate(ctx, parent, *bucket)
		} else if s.search == nil {
			// Direct helper use outside Compile, without a current minibatch.
			rules, err = s.extractRulesFromTrajectories(ctx)
		}
		if err != nil {
			s.logger.Debug(ctx, "Rule analysis failed: %v", err)
		}
		return s.generateRuleBasedCandidate(ctx, parent, rules)
	}
	return s.perturbProgram(ctx, parent)
}

// simbaInstructionModules finds nested predictors without setting signatures on their
// wrappers (ChainOfThought.SetSignature, for example, adds a rationale field).
// Custom modules without Predict children are optimized through core.Module.
func simbaInstructionModules(program core.Program) []core.Module {
	var targets []core.Module
	for _, module := range program.GetModules() {
		predictors := demoPredictors(module)
		if len(predictors) == 0 {
			targets = append(targets, module)
			continue
		}
		for _, predictor := range predictors {
			targets = append(targets, predictor)
		}
	}
	return targets
}

// perturbProgram creates a variation of the base program by modifying instructions.
func (s *SIMBA) perturbProgram(ctx context.Context, baseProgram core.Program) (core.Program, error) {
	variant := baseProgram.Clone()
	modules := simbaInstructionModules(variant)
	if len(modules) == 0 {
		return variant, nil
	}

	module := modules[s.proposalIntn(ctx, len(modules))]
	signature := module.GetSignature()
	instruction, err := s.generateInstructionVariation(ctx, signature.Instruction)
	if err != nil {
		return variant, err
	}
	signature.Instruction = instruction
	module.SetSignature(signature)
	return variant, nil
}

// generateInstructionVariation creates a variation of an instruction using LLM.
func (s *SIMBA) generateInstructionVariation(ctx context.Context, originalInstruction string) (string, error) {
	if s.primaryModel == nil {
		return originalInstruction, nil
	}

	prompt := fmt.Sprintf(`Generate a variation of the following instruction that maintains the same intent but uses different wording:

Original: %s

Variation:`, originalInstruction)

	core.RecordModelCall(ctx, s.primaryModel)
	response, err := s.primaryModel.Generate(ctx, s.withIntrospection(prompt))
	if err != nil {
		// Handle LLM failures gracefully by returning original instruction
		return originalInstruction, nil
	}

	if response.Content == "" {
		return originalInstruction, nil
	}

	return response.Content, nil
}

// generateRuleBasedCandidate creates a candidate program using rule generation from trajectories.
func (s *SIMBA) generateRuleBasedCandidate(ctx context.Context, baseProgram core.Program, rules []string) (core.Program, error) {
	if len(rules) == 0 {
		return s.perturbProgram(ctx, baseProgram)
	}
	variant := baseProgram.Clone()
	modules := simbaInstructionModules(variant)
	if len(modules) == 0 {
		return variant, nil
	}

	module := modules[s.proposalIntn(ctx, len(modules))]
	signature := module.GetSignature()
	instruction, err := s.appendRuleToInstruction(ctx, signature.Instruction, rules)
	if err != nil {
		return variant, err
	}
	signature.Instruction = instruction
	module.SetSignature(signature)
	return variant, nil
}

// extractRulesFromTrajectories analyzes recent trajectories to extract patterns and rules.
func (s *SIMBA) extractRulesFromTrajectories(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	trajectories := s.state.Trajectories
	s.mu.RUnlock()

	if len(trajectories) < 3 { // Lowered threshold for faster rule extraction
		return []string{}, nil
	}

	// Group trajectories by success/failure
	successfulTrajectories := []Trajectory{}
	failedTrajectories := []Trajectory{}

	for _, t := range trajectories {
		if t.Success {
			successfulTrajectories = append(successfulTrajectories, t)
		} else {
			failedTrajectories = append(failedTrajectories, t)
		}
	}

	// Enhanced trajectory analysis - can work with just successful examples
	if len(successfulTrajectories) == 0 {
		return []string{}, nil
	}

	// Use pattern-based rule extraction for better performance
	if len(failedTrajectories) > 0 {
		// Full comparative analysis when we have both types
		return s.extractComparativeRules(ctx, successfulTrajectories, failedTrajectories)
	} else {
		// Success-pattern analysis when we only have successful examples
		return s.extractSuccessPatternRules(ctx, successfulTrajectories)
	}
}

// buildTrajectoryAnalysisPrompt creates a prompt for analyzing trajectories.
func (s *SIMBA) buildTrajectoryAnalysisPrompt(successful, failed []Trajectory) string {
	prompt := `Analyze the following successful and failed execution trajectories to identify patterns and extract rules that distinguish successful predictions from failures.

SUCCESSFUL TRAJECTORIES:
`
	// Include a sample of successful trajectories
	sampleSize := 3
	if len(successful) < sampleSize {
		sampleSize = len(successful)
	}
	for i := 0; i < sampleSize; i++ {
		t := successful[i]
		prompt += fmt.Sprintf("\nExample %d:\n", i+1)
		prompt += fmt.Sprintf("Input: %v\n", t.Example.Inputs)
		prompt += fmt.Sprintf("Output: %v\n", t.Prediction)
		prompt += fmt.Sprintf("Expected output: %v\n", t.Example.Outputs)
		prompt += fmt.Sprintf("Score: %.3f\n", t.Score)
		prompt += simbaFormatPredictorTraces(t.PredictorTraces)
	}

	prompt += `
FAILED TRAJECTORIES:
`
	// Include a sample of failed trajectories
	if len(failed) < sampleSize {
		sampleSize = len(failed)
	}
	for i := 0; i < sampleSize; i++ {
		t := failed[i]
		prompt += fmt.Sprintf("\nExample %d:\n", i+1)
		prompt += fmt.Sprintf("Input: %v\n", t.Example.Inputs)
		prompt += fmt.Sprintf("Output: %v\n", t.Prediction)
		prompt += fmt.Sprintf("Expected output: %v\n", t.Example.Outputs)
		prompt += fmt.Sprintf("Score: %.3f\n", t.Score)
		prompt += simbaFormatPredictorTraces(t.PredictorTraces)
	}

	prompt += `

Based on these trajectories, extract 1-3 specific rules that could improve performance. Rules should be:
1. Concrete and actionable
2. Based on patterns you observe
3. Focused on what makes predictions successful

Format each rule on a new line starting with "RULE: "

Analysis:`

	return prompt
}

// parseRulesFromAnalysis extracts rules from LLM analysis.
func (s *SIMBA) parseRulesFromAnalysis(analysis string) []string {
	rules := []string{}
	lines := strings.Split(analysis, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "RULE: ") {
			rule := strings.TrimPrefix(line, "RULE: ")
			if rule != "" {
				rules = append(rules, rule)
			}
		}
	}

	// Limit to top 3 rules
	if len(rules) > 3 {
		rules = rules[:3]
	}

	return rules
}

// extractComparativeRules extracts rules by comparing successful and failed trajectories.
func (s *SIMBA) extractComparativeRules(ctx context.Context, successful, failed []Trajectory) ([]string, error) {
	// Use LLM to analyze patterns
	analysisPrompt := s.buildTrajectoryAnalysisPrompt(successful, failed)

	if s.analyzerModel == nil {
		return []string{}, nil
	}

	core.RecordModelCall(ctx, s.analyzerModel)
	response, err := s.analyzerModel.Generate(ctx, analysisPrompt)
	if err != nil {
		return []string{}, err
	}

	return s.parseRulesFromAnalysis(response.Content), nil
}

// extractSuccessPatternRules extracts rules from successful trajectory patterns.
func (s *SIMBA) extractSuccessPatternRules(ctx context.Context, successful []Trajectory) ([]string, error) {
	if len(successful) == 0 {
		return []string{}, nil
	}

	// Build success pattern analysis prompt
	prompt := s.buildSuccessPatternAnalysisPrompt(successful)

	if s.analyzerModel == nil {
		return []string{}, nil
	}

	core.RecordModelCall(ctx, s.analyzerModel)
	response, err := s.analyzerModel.Generate(ctx, prompt)
	if err != nil {
		return []string{}, err
	}

	return s.parseRulesFromAnalysis(response.Content), nil
}

// buildSuccessPatternAnalysisPrompt creates a prompt for analyzing successful patterns.
func (s *SIMBA) buildSuccessPatternAnalysisPrompt(successful []Trajectory) string {
	prompt := `Analyze the following successful execution trajectories to identify patterns and extract rules that lead to successful predictions.

SUCCESSFUL TRAJECTORIES:
`
	// Include a sample of successful trajectories
	sampleSize := 3
	if len(successful) < sampleSize {
		sampleSize = len(successful)
	}
	for i := 0; i < sampleSize; i++ {
		t := successful[i]
		prompt += fmt.Sprintf("\nExample %d:\n", i+1)
		prompt += fmt.Sprintf("Input: %v\n", t.Example.Inputs)
		prompt += fmt.Sprintf("Output: %v\n", t.Prediction)
		prompt += fmt.Sprintf("Expected output: %v\n", t.Example.Outputs)
		prompt += fmt.Sprintf("Score: %.3f\n", t.Score)
		prompt += simbaFormatPredictorTraces(t.PredictorTraces)
	}

	prompt += `

Based on these successful trajectories, extract 1-3 specific rules that could improve performance. Rules should be:
1. Concrete and actionable
2. Based on patterns you observe in successful cases
3. Focused on what makes predictions successful

Format each rule on a new line starting with "RULE: "

Analysis:`

	return prompt
}

// appendRuleToInstruction creates an enhanced instruction by appending extracted rules.
func (s *SIMBA) appendRuleToInstruction(ctx context.Context, originalInstruction string, rules []string) (string, error) {
	if len(rules) == 0 || s.primaryModel == nil {
		return originalInstruction, nil
	}

	// Select a random rule to append (thread-safe)
	selectedRule := rules[s.proposalIntn(ctx, len(rules))]

	prompt := fmt.Sprintf(`Given the following instruction and rule, create an enhanced instruction that incorporates the rule naturally:

Original Instruction: %s

Rule to incorporate: %s

Create an enhanced instruction that:
1. Maintains the original intent
2. Naturally incorporates the rule
3. Is clear and concise

Enhanced Instruction:`, originalInstruction, selectedRule)

	core.RecordModelCall(ctx, s.primaryModel)
	response, err := s.primaryModel.Generate(ctx, s.withIntrospection(prompt))
	if err != nil {
		// Fall back to simple concatenation
		return originalInstruction + " " + selectedRule, nil
	}

	if response.Content == "" {
		return originalInstruction + " " + selectedRule, nil
	}

	return response.Content, nil
}

func (s *SIMBA) sampleMiniBatchFromExamples(ctx context.Context, examples []core.Example) ([]core.Example, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(examples) == 0 {
		return nil, errors.New(errors.ValidationFailed, "dataset is empty")
	}

	// Sample random batch
	batchSize := s.config.BatchSize
	if batchSize > len(examples) {
		batchSize = len(examples)
	}

	s.mu.Lock()
	indices := s.rng.Perm(len(examples))
	s.mu.Unlock()
	batch := make([]core.Example, 0, batchSize)
	for _, idx := range indices[:batchSize] {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		batch = append(batch, examples[idx])
	}

	return batch, nil
}

// evaluateCandidates evaluates all candidate programs on the mini-batch.
func (s *SIMBA) evaluateCandidates(ctx context.Context, candidates []core.Program, batch []core.Example, scorer core.Scorer) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Keep a bounded diagnostic window of this step's candidate evaluations.
	// Generation already consumed the separate, complete current-batch buckets.
	s.mu.Lock()
	s.state.Trajectories = nil
	s.mu.Unlock()

	scores := make([]float64, len(candidates))
	errs := make([]error, len(candidates))
	p := s.createLLMPool()

	for i, candidate := range candidates {
		i, candidate := i, candidate
		p.Go(func() {
			var trajectories []Trajectory
			scores[i], trajectories, errs[i] = s.evaluateBatch(ctx, candidate, batch, scorer)
			if errs[i] == nil {
				for j := range trajectories {
					trajectories[j].ProgramID = fmt.Sprintf("step_%d_candidate_%d", s.state.CurrentStep, i)
				}
				s.storeTrajectories(trajectories)
			}
		})
	}
	p.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return scores, nil
}

// evaluateCandidateOnBatch evaluates a single candidate on the mini-batch with parallel processing.
func (s *SIMBA) evaluateCandidateOnBatch(ctx context.Context, candidate core.Program, batch []core.Example, scorer core.Scorer) (float64, error) {
	score, trajectories, err := s.evaluateBatch(ctx, candidate, batch, scorer)
	if err == nil {
		s.storeTrajectories(trajectories)
	}
	return score, err
}

func (s *SIMBA) storeTrajectories(trajectories []Trajectory) {
	if len(trajectories) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Trajectories = append(s.state.Trajectories, trajectories...)
	if len(s.state.Trajectories) > 100 {
		s.state.Trajectories = s.state.Trajectories[len(s.state.Trajectories)-100:]
	}
}

// evaluateBatch returns evidence in example order, independently of completion
// order. Callers decide whether it belongs in the search buckets or diagnostics.
func (s *SIMBA) evaluateBatch(ctx context.Context, candidate core.Program, batch []core.Example, scorer core.Scorer) (float64, []Trajectory, error) {
	if err := ctx.Err(); err != nil {
		return 0, nil, err
	}
	if len(batch) == 0 {
		return 0, nil, nil
	}
	scores := make([]float64, len(batch))
	errs := make([]error, len(batch))
	var trajectories []Trajectory
	if !s.config.DisableTrajectoryTracking {
		trajectories = make([]Trajectory, len(batch))
	}
	p := s.createLLMPool()
	for i, example := range batch {
		p.Go(func() {
			if err := ctx.Err(); err != nil {
				errs[i] = err
				return
			}
			startTime := time.Now()
			var prediction map[string]any
			var trace []SIMBAPredictorTrace
			var err error
			if s.config.DisableTrajectoryTracking {
				prediction, err = candidate.Forward(ctx, example.Inputs)
			} else {
				prediction, trace, err = simbaTraceForward(ctx, candidate, example.Inputs)
			}
			if err == nil {
				scores[i], err = scorer.Score(ctx, example, prediction)
			}
			errs[i] = err
			if err != nil {
				s.logger.Debug(ctx, "Candidate evaluation failed for example: %v", err)
				prediction, trace = nil, nil
				scores[i] = 0
			}
			if trajectories != nil {
				trajectories[i] = Trajectory{
					Example: example, Prediction: prediction, Score: scores[i],
					Success:       err == nil && scores[i] > 0.5,
					ProgramID:     fmt.Sprintf("prog_%d", s.state.CurrentStep),
					ExecutionTime: time.Since(startTime), PredictorTraces: trace,
				}
			}
		})
	}
	p.Wait()
	if err := ctx.Err(); err != nil {
		return 0, nil, err
	}
	var total float64
	for i, err := range errs {
		if err != nil {
			return math.Inf(-1), trajectories, nil
		}
		total += scores[i]
	}
	return total / float64(len(batch)), trajectories, nil
}

// selectBestCandidate uses bucket sorting or temperature-controlled sampling to select program.
func (s *SIMBA) selectBestCandidate(candidates []core.Program, scores []float64) (core.Program, float64) {
	if len(candidates) == 0 || len(scores) == 0 {
		return core.Program{}, math.Inf(-1)
	}

	// Failed evaluations use -Inf so they cannot outrank valid negative metric
	// values. Remove them before bucket normalization or temperature sampling.
	limit := min(len(candidates), len(scores))
	validCandidates := make([]core.Program, 0, limit)
	validScores := make([]float64, 0, limit)
	for i := 0; i < limit; i++ {
		score := scores[i]
		if math.IsInf(score, 1) {
			return candidates[i], score
		}
		if math.IsInf(score, -1) || math.IsNaN(score) {
			continue
		}
		validCandidates = append(validCandidates, candidates[i])
		validScores = append(validScores, score)
	}
	if len(validCandidates) == 0 {
		return candidates[0], math.Inf(-1)
	}
	candidates, scores = validCandidates, validScores

	// Use bucket sorting if enabled
	if s.config.UseBucketSorting {
		selectedProgram, selectedScore, metadata := s.selectCandidateWithBucketSorting(candidates, scores)

		// Record candidate metadata if available
		if metadata != nil {
			s.recordCandidateMetadata(selectedProgram, selectedScore, metadata)
		}

		return selectedProgram, selectedScore
	}

	// Fall back to original temperature-controlled sampling
	return s.selectBestCandidateWithTemperature(candidates, scores)
}

// selectBestCandidateWithTemperature uses temperature-controlled sampling to select program (original implementation).
func (s *SIMBA) selectBestCandidateWithTemperature(candidates []core.Program, scores []float64) (core.Program, float64) {
	if len(candidates) == 0 || len(scores) == 0 {
		return candidates[0], scores[0]
	}

	// Find best performing candidate
	bestIdx := 0
	bestScore := scores[0]
	for i, score := range scores {
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	// Use temperature-controlled sampling for exploration (thread-safe)
	temperature := s.getCurrentTemperature(s.state.CurrentStep)
	s.mu.Lock()
	shouldExplore := temperature > 0 && s.rng.Float64() < 0.3 // 30% chance for exploration
	s.mu.Unlock()

	if shouldExplore {
		selectedIdx := s.temperatureSample(scores, temperature)
		return candidates[selectedIdx], scores[selectedIdx]
	}

	return candidates[bestIdx], bestScore
}

// temperatureSample performs temperature-controlled sampling from scores.
func (s *SIMBA) temperatureSample(scores []float64, temperature float64) int {
	if temperature <= 0 || len(scores) <= 1 {
		// Return best scoring index
		bestIdx := 0
		for i, score := range scores {
			if score > scores[bestIdx] {
				bestIdx = i
			}
		}
		return bestIdx
	}

	// Convert scores to probabilities using softmax with temperature
	probs := make([]float64, len(scores))
	maxScore := scores[0]
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}

	var sum float64
	for i, score := range scores {
		probs[i] = math.Exp((score - maxScore) / temperature)
		sum += probs[i]
	}

	// Normalize probabilities
	for i := range probs {
		probs[i] /= sum
	}

	// Sample from probability distribution (thread-safe)
	s.mu.Lock()
	r := s.rng.Float64()
	s.mu.Unlock()

	var cumulative float64
	for i, prob := range probs {
		cumulative += prob
		if r <= cumulative {
			return i
		}
	}

	return len(scores) - 1
}

// getCurrentTemperature returns current temperature for the optimization step.
func (s *SIMBA) getCurrentTemperature(step int) float64 {
	// Simple temperature decay schedule
	decayRate := 0.95
	return s.config.SamplingTemperature * math.Pow(decayRate, float64(step))
}

// calculateMultiCriteriaScore computes multi-criteria scores for bucket sorting.
func (s *SIMBA) calculateMultiCriteriaScore(candidates []core.Program, scores []float64) []CandidateMetadata {
	if len(candidates) == 0 || len(scores) == 0 {
		return []CandidateMetadata{}
	}

	metadata := make([]CandidateMetadata, len(candidates))

	// Calculate basic statistics
	minScore := scores[0]
	maxScore := scores[0]
	totalScore := 0.0

	for _, score := range scores {
		if score < minScore {
			minScore = score
		}
		if score > maxScore {
			maxScore = score
		}
		totalScore += score
	}

	avgScore := totalScore / float64(len(scores))

	// Calculate multi-criteria scores for each candidate
	for i, score := range scores {
		// Core metrics
		maxToMinGap := maxScore - minScore
		maxToAvgGap := maxScore - avgScore

		// Diversity score based on score distance from average
		diversityScore := math.Abs(score - avgScore)

		// Improvement delta (relative to best score)
		improvementDelta := score - maxScore

		// Composite score calculation using configured criteria and weights
		compositeScore := s.calculateCompositeScore(score, maxScore, minScore, avgScore)

		metadata[i] = CandidateMetadata{
			IndividualScores: []float64{score},
			DiversityScore:   diversityScore,
			ImprovementDelta: improvementDelta,
			MaxToMinGap:      maxToMinGap,
			MaxScore:         maxScore,
			MaxToAvgGap:      maxToAvgGap,
			CompositeScore:   compositeScore,
		}
	}

	return metadata
}

// calculateCompositeScore computes weighted composite score based on configured criteria.
func (s *SIMBA) calculateCompositeScore(score, maxScore, minScore, avgScore float64) float64 {
	if len(s.config.BucketSortingCriteria) == 0 {
		return score
	}

	compositeScore := 0.0

	for i, criterion := range s.config.BucketSortingCriteria {
		weight := 1.0 / float64(len(s.config.BucketSortingCriteria))
		if i < len(s.config.BucketSortingWeights) {
			weight = s.config.BucketSortingWeights[i]
		}

		var criterionScore float64
		switch criterion {
		case "max_to_min_gap":
			// Favor candidates with larger gaps (more discriminative)
			gap := maxScore - minScore
			if gap > 0 {
				criterionScore = (score - minScore) / gap
			} else {
				criterionScore = 1.0
			}
		case "max_score":
			// Favor candidates with higher scores
			// Min/max normalization preserves ordering for negative and mixed
			// scores; dividing by max either collapsed non-positive inputs or
			// inverted ordering when max was negative.
			if maxScore > minScore {
				criterionScore = (score - minScore) / (maxScore - minScore)
			} else {
				criterionScore = 1.0
			}
		case "max_to_avg_gap":
			// Favor candidates above average
			gap := maxScore - avgScore
			if gap > 0 {
				criterionScore = math.Max(0, (score-avgScore)/gap)
			} else {
				if score >= avgScore {
					criterionScore = 1.0
				} else {
					criterionScore = 0.0
				}
			}
		case "diversity":
			// Favor candidates with unique scores
			diversity := math.Abs(score - avgScore)
			maxDiversity := math.Max(math.Abs(maxScore-avgScore), math.Abs(minScore-avgScore))
			if maxDiversity > 0 {
				criterionScore = diversity / maxDiversity
			} else {
				criterionScore = 0.0
			}
		case "improvement_potential":
			// Favor candidates with high potential for improvement
			if score > avgScore {
				criterionScore = (score - avgScore) / (maxScore - avgScore + 1e-10)
			} else {
				criterionScore = 0.0
			}
		default:
			criterionScore = score
		}

		compositeScore += weight * criterionScore
	}

	return compositeScore
}

// selectCandidateWithBucketSorting implements bucket sorting candidate selection.
func (s *SIMBA) selectCandidateWithBucketSorting(candidates []core.Program, scores []float64) (core.Program, float64, *CandidateMetadata) {
	if len(candidates) == 0 || len(scores) == 0 {
		return candidates[0], scores[0], nil
	}

	// Calculate multi-criteria scores and metadata
	metadata := s.calculateMultiCriteriaScore(candidates, scores)

	// Create candidate-metadata pairs for sorting
	pairs := make([]candidatePair, len(candidates))
	for i := 0; i < len(candidates); i++ {
		pairs[i] = candidatePair{
			candidate: candidates[i],
			score:     scores[i],
			metadata:  metadata[i],
			index:     i,
		}
	}

	// Sort by composite score (descending)
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].metadata.CompositeScore < pairs[j].metadata.CompositeScore {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Assign bucket rankings
	for i, pair := range pairs {
		pair.metadata.SelectionRank = i + 1
		pair.metadata.BucketAssignment = s.assignBucket(pair.metadata.CompositeScore, i, len(pairs))
		pairs[i] = pair
	}

	// Select from top bucket with optional temperature sampling
	topBucket := s.getTopBucket(pairs)
	if len(topBucket) == 0 {
		// Fallback to best candidate
		selected := pairs[0]
		return selected.candidate, selected.score, &selected.metadata
	}

	// Apply temperature sampling within top bucket
	temperature := s.getCurrentTemperature(s.state.CurrentStep)
	if temperature > 0 && len(topBucket) > 1 {
		// Extract scores from top bucket for temperature sampling
		bucketScores := make([]float64, len(topBucket))
		for i, pair := range topBucket {
			bucketScores[i] = pair.metadata.CompositeScore
		}

		// Sample from top bucket using temperature
		selectedIdx := s.temperatureSample(bucketScores, temperature)
		if selectedIdx < len(topBucket) {
			selected := topBucket[selectedIdx]
			return selected.candidate, selected.score, &selected.metadata
		}
	}

	// Default to best candidate from top bucket
	selected := topBucket[0]
	return selected.candidate, selected.score, &selected.metadata
}

// assignBucket assigns a bucket number based on composite score and ranking.
func (s *SIMBA) assignBucket(compositeScore float64, rank int, totalCandidates int) int {
	// Simple bucket assignment: top 30% = bucket 1, next 40% = bucket 2, rest = bucket 3
	if rank < int(0.3*float64(totalCandidates))+1 {
		return 1 // Top bucket
	} else if rank < int(0.7*float64(totalCandidates))+1 {
		return 2 // Middle bucket
	} else {
		return 3 // Bottom bucket
	}
}

// getTopBucket returns candidates from the top-performing bucket.
func (s *SIMBA) getTopBucket(pairs []candidatePair) []candidatePair {
	if len(pairs) == 0 {
		return []candidatePair{}
	}

	// Get all candidates from bucket 1 (top bucket)
	topBucket := []candidatePair{}
	for _, pair := range pairs {
		if pair.metadata.BucketAssignment == 1 {
			topBucket = append(topBucket, pair)
		}
	}

	// If no candidates in top bucket, return best candidate
	if len(topBucket) == 0 {
		return []candidatePair{pairs[0]}
	}

	return topBucket
}

// recordCandidateMetadata records metadata for the selected candidate.
func (s *SIMBA) recordCandidateMetadata(program core.Program, score float64, metadata *CandidateMetadata) {
	if metadata == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Create candidate result with metadata
	result := CandidateResult{
		Program:     program,
		Score:       score,
		Step:        s.state.CurrentStep,
		Temperature: s.getCurrentTemperature(s.state.CurrentStep),
		CreatedAt:   time.Now(),
		Metadata:    metadata,
	}

	// Add to candidate history
	s.state.CandidateHistory = append(s.state.CandidateHistory, result)

	// Keep sliding window of recent candidates (max 50)
	if len(s.state.CandidateHistory) > 50 {
		s.state.CandidateHistory = s.state.CandidateHistory[len(s.state.CandidateHistory)-50:]
	}
}

// performIntrospection analyzes current optimization progress and provides advice.
func (s *SIMBA) performIntrospection(ctx context.Context) *IntrospectionResult {
	if s.analyzerModel == nil || len(s.state.PerformanceLog) < 2 {
		return &IntrospectionResult{
			Analysis:             "Insufficient data for introspection",
			Recommendations:      []string{},
			Confidence:           0.0,
			IdentifiedPatterns:   []string{},
			SuggestedAdjustments: []string{},
		}
	}

	// Analyze recent performance trends
	recentSteps := s.getRecentSteps(5)
	analysisPrompt := s.buildIntrospectionPrompt(recentSteps)

	core.RecordModelCall(ctx, s.analyzerModel)
	response, err := s.analyzerModel.Generate(ctx, analysisPrompt)
	if err != nil {
		s.logger.Debug(ctx, "Introspection failed: %v", err)
		return &IntrospectionResult{
			Analysis:             fmt.Sprintf("Introspection failed: %v", err),
			Recommendations:      []string{},
			Confidence:           0.0,
			IdentifiedPatterns:   []string{},
			SuggestedAdjustments: []string{},
		}
	}

	return &IntrospectionResult{
		Analysis:             response.Content,
		Recommendations:      s.extractRecommendations(response.Content),
		Confidence:           0.7, // Simple confidence score
		IdentifiedPatterns:   s.identifyPatterns(recentSteps),
		SuggestedAdjustments: s.suggestAdjustments(recentSteps),
	}
}

// getRecentSteps returns the most recent optimization steps.
func (s *SIMBA) getRecentSteps(n int) []StepResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	log := s.state.PerformanceLog
	if len(log) <= n {
		return log
	}
	return log[len(log)-n:]
}

// buildIntrospectionPrompt creates a prompt for self-analysis.
func (s *SIMBA) buildIntrospectionPrompt(steps []StepResult) string {
	prompt := "Analyze the following optimization progress and provide insights:\n\n"
	for _, step := range steps {
		prompt += fmt.Sprintf("Step %d: Score=%.4f, Improvement=%.4f, Temperature=%.4f\n",
			step.Step, step.BestScore, step.Improvement, step.Temperature)
	}

	prompt += "\nProvide analysis of:\n"
	prompt += "1. Performance trends\n"
	prompt += "2. Optimization effectiveness\n"
	prompt += "3. Recommended adjustments\n"
	prompt += "\nAnalysis:"

	return prompt
}

// extractRecommendations extracts actionable recommendations from analysis.
func (s *SIMBA) extractRecommendations(analysis string) []string {
	recommendations := []string{}
	analysisLower := strings.ToLower(analysis)

	// Look for specific recommendation keywords in the LLM analysis
	if strings.Contains(analysisLower, "increase temperature") || strings.Contains(analysisLower, "more exploration") {
		recommendations = append(recommendations, "Increase sampling temperature for more exploration")
	}

	if strings.Contains(analysisLower, "decrease temperature") || strings.Contains(analysisLower, "less exploration") {
		recommendations = append(recommendations, "Decrease sampling temperature for more exploitation")
	}

	if strings.Contains(analysisLower, "batch size") && strings.Contains(analysisLower, "increase") {
		recommendations = append(recommendations, "Consider increasing batch size for more stable gradients")
	}

	if strings.Contains(analysisLower, "batch size") && strings.Contains(analysisLower, "decrease") {
		recommendations = append(recommendations, "Consider decreasing batch size for faster iterations")
	}

	if strings.Contains(analysisLower, "stagnant") || strings.Contains(analysisLower, "plateau") {
		recommendations = append(recommendations, "Try different instruction variations to escape local optimum")
		recommendations = append(recommendations, "Consider restarting with different random seed")
	}

	if strings.Contains(analysisLower, "overfitting") || strings.Contains(analysisLower, "memorizing") {
		recommendations = append(recommendations, "Increase diversity in candidate generation")
		recommendations = append(recommendations, "Consider validation-based stopping criteria")
	}

	if strings.Contains(analysisLower, "slow progress") || strings.Contains(analysisLower, "small improvement") {
		recommendations = append(recommendations, "Increase number of candidates per iteration")
		recommendations = append(recommendations, "Try more aggressive instruction perturbations")
	}

	if strings.Contains(analysisLower, "unstable") || strings.Contains(analysisLower, "volatile") {
		recommendations = append(recommendations, "Increase batch size for more stable evaluation")
		recommendations = append(recommendations, "Consider smoothing or averaging recent scores")
	}

	// If no specific patterns found but analysis exists, provide generic advice
	if len(recommendations) == 0 && len(strings.TrimSpace(analysis)) > 20 {
		recommendations = append(recommendations, "Continue current optimization approach with monitoring")
	}

	return recommendations
}

// identifyPatterns identifies optimization patterns from recent steps.
func (s *SIMBA) identifyPatterns(steps []StepResult) []string {
	patterns := []string{}

	if len(steps) < 2 {
		return patterns
	}

	// Analyze score improvement trends
	improvements := make([]float64, 0, len(steps)-1)
	for i := 1; i < len(steps); i++ {
		improvements = append(improvements, steps[i].BestScore-steps[i-1].BestScore)
	}

	// Count consecutive non-improving steps
	consecutiveNoImprovement := 0
	maxConsecutiveNoImprovement := 0
	for _, improvement := range improvements {
		if improvement <= 0.0001 { // Consider very small improvements as no improvement
			consecutiveNoImprovement++
			if consecutiveNoImprovement > maxConsecutiveNoImprovement {
				maxConsecutiveNoImprovement = consecutiveNoImprovement
			}
		} else {
			consecutiveNoImprovement = 0
		}
	}

	// Analyze improvement magnitude trends
	var totalImprovement, avgImprovement float64
	positiveImprovements := 0
	for _, improvement := range improvements {
		totalImprovement += improvement
		if improvement > 0 {
			positiveImprovements++
		}
	}
	avgImprovement = totalImprovement / float64(len(improvements))

	// Calculate improvement variance to detect volatility
	var variance float64
	for _, improvement := range improvements {
		variance += (improvement - avgImprovement) * (improvement - avgImprovement)
	}
	variance /= float64(len(improvements))
	stdDev := math.Sqrt(variance)

	// Identify specific patterns
	improvementRatio := float64(positiveImprovements) / float64(len(improvements))

	if improvementRatio >= 0.7 {
		patterns = append(patterns, "Strong upward trend detected")
	} else if improvementRatio >= 0.4 {
		patterns = append(patterns, "Moderate improvement trend")
	} else if improvementRatio <= 0.2 {
		patterns = append(patterns, "Consistent stagnation or decline")
	}

	if maxConsecutiveNoImprovement >= 3 {
		patterns = append(patterns, fmt.Sprintf("Extended plateau detected (%d consecutive non-improving steps)", maxConsecutiveNoImprovement))
	}

	if avgImprovement > 0.01 {
		patterns = append(patterns, "High magnitude improvements")
	} else if avgImprovement > 0.001 {
		patterns = append(patterns, "Small but steady improvements")
	} else if avgImprovement <= 0 {
		patterns = append(patterns, "No net improvement or decline")
	}

	// Detect volatility patterns
	if stdDev > math.Abs(avgImprovement)*2 && len(improvements) >= 3 {
		patterns = append(patterns, "High volatility in performance")
	} else if stdDev < math.Abs(avgImprovement)*0.5 {
		patterns = append(patterns, "Stable, consistent performance")
	}

	// Analyze recent vs early performance (if we have enough data)
	if len(steps) >= 6 {
		earlyAvg := (steps[0].BestScore + steps[1].BestScore + steps[2].BestScore) / 3
		recentAvg := (steps[len(steps)-3].BestScore + steps[len(steps)-2].BestScore + steps[len(steps)-1].BestScore) / 3

		if recentAvg > earlyAvg*1.1 {
			patterns = append(patterns, "Accelerating improvement in recent steps")
		} else if recentAvg < earlyAvg*0.95 {
			patterns = append(patterns, "Performance degradation detected")
		}
	}

	// Simple trend line analysis (linear regression slope)
	if len(steps) >= 4 {
		slope := s.calculateTrendSlope(steps)
		if slope > 0.01 {
			patterns = append(patterns, "Strong positive trend (linear fit)")
		} else if slope > 0.001 {
			patterns = append(patterns, "Weak positive trend (linear fit)")
		} else if slope < -0.001 {
			patterns = append(patterns, "Negative trend detected (linear fit)")
		} else {
			patterns = append(patterns, "Flat trend (linear fit)")
		}
	}

	return patterns
}

// calculateTrendSlope performs simple linear regression to calculate trend slope.
func (s *SIMBA) calculateTrendSlope(steps []StepResult) float64 {
	n := float64(len(steps))
	if n < 2 {
		return 0
	}

	// Calculate means
	var sumX, sumY float64
	for i, step := range steps {
		sumX += float64(i)
		sumY += step.BestScore
	}
	meanX := sumX / n
	meanY := sumY / n

	// Calculate slope using least squares formula
	var numerator, denominator float64
	for i, step := range steps {
		x := float64(i)
		y := step.BestScore
		numerator += (x - meanX) * (y - meanY)
		denominator += (x - meanX) * (x - meanX)
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// suggestAdjustments suggests parameter adjustments based on performance.
func (s *SIMBA) suggestAdjustments(steps []StepResult) []string {
	adjustments := []string{}

	if len(steps) < 2 {
		return adjustments
	}

	// Analyze recent improvements
	recentImprovement := steps[len(steps)-1].Improvement
	if recentImprovement < s.config.MinImprovementRatio {
		adjustments = append(adjustments, "Consider increasing exploration temperature")
		adjustments = append(adjustments, "Try different instruction variations")
	}

	return adjustments
}

// recordStep shares progress and useful introspection between both execution modes.
func (s *SIMBA) recordStep(ctx context.Context, step StepResult) {
	s.mu.Lock()
	s.state.PerformanceLog = append(s.state.PerformanceLog, step)
	s.mu.Unlock()
	frequency := s.config.IntrospectionFrequency
	if frequency <= 0 || step.Step == 0 || step.Step%frequency != 0 ||
		step.Step+1 >= s.config.MaxSteps || s.shouldStopEarly() {
		return
	}
	introspection := s.performIntrospection(ctx)
	s.mu.Lock()
	s.state.PerformanceLog[len(s.state.PerformanceLog)-1].Introspection = introspection.Analysis
	s.state.IntrospectionLog = append(s.state.IntrospectionLog, introspection.Analysis)
	s.mu.Unlock()
}

// withIntrospection gives the next proposals the advice that previously was only logged.
func (s *SIMBA) withIntrospection(prompt string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if n := len(s.state.IntrospectionLog); n > 0 {
		prompt += "\n\nUse this optimization feedback when revising the instruction:\n" + s.state.IntrospectionLog[n-1]
	}
	return prompt
}

// GetState returns the current optimization state (thread-safe).
func (s *SIMBA) GetState() SIMBAState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Compile keeps appending to these slices and fills in the last step's
	// introspection later, so the caller gets its own copies.
	state := *s.state
	state.CandidateHistory = slices.Clone(state.CandidateHistory)
	state.PerformanceLog = slices.Clone(state.PerformanceLog)
	state.IntrospectionLog = slices.Clone(state.IntrospectionLog)
	state.Trajectories = slices.Clone(state.Trajectories)
	return state
}

// GetConfig returns the current configuration.
func (s *SIMBA) GetConfig() SIMBAConfig {
	return s.config
}

// shouldStopEarly determines if optimization should stop early due to lack of improvement.
func (s *SIMBA) shouldStopEarly() bool {
	if s.config.EarlyStoppingPatience <= 0 {
		return false // Early stopping disabled
	}

	log := s.state.PerformanceLog
	if len(log) < s.config.EarlyStoppingPatience+1 {
		return false // Not enough steps to evaluate
	}

	// Get the best score from patience steps ago
	patienceStepsAgo := len(log) - s.config.EarlyStoppingPatience - 1
	if patienceStepsAgo < 0 {
		return false
	}

	pastBestScore := log[patienceStepsAgo].BestScore
	currentBestScore := s.state.BestScore

	// Check if improvement is below threshold
	improvement := currentBestScore - pastBestScore
	return improvement < s.config.EarlyStoppingThreshold
}

// Pipeline Processing Implementation

// initializePipelineChannels creates and initializes the pipeline channels.
func (s *SIMBA) initializePipelineChannels() {
	bufferSize := s.config.PipelineBufferSize
	s.pipelineChannels = &PipelineChannels{
		CandidateGeneration: make(chan *PipelineStage, bufferSize),
		BatchSampling:       make(chan *PipelineStage, bufferSize),
		CandidateEvaluation: make(chan *PipelineStage, bufferSize),
		Results:             make(chan *PipelineStage, bufferSize),
		Errors:              make(chan error, bufferSize*3),
		Done:                make(chan struct{}),
	}
}

// safeSendStage applies backpressure until the consumer is ready or the run stops.
func (s *SIMBA) safeSendStage(ctx context.Context, ch chan<- *PipelineStage, stage *PipelineStage) bool {
	select {
	case ch <- stage:
		return true
	case <-ctx.Done():
		return false
	case <-s.pipelineChannels.Done:
		return false
	}
}

// runPipelineProcessing executes the optimization using pipeline processing.
func (s *SIMBA) runPipelineProcessing(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	s.initializePipelineChannels()
	// Don't use defer for channel closing since we manage it manually

	// Create a cancellable context for pipeline workers
	pipelineCtx, pipelineCancel := context.WithCancel(ctx)
	defer pipelineCancel()

	// Start pipeline workers using Go 1.25's WaitGroup.Go() for cleaner goroutine management
	var wg sync.WaitGroup

	// Evidence must precede candidate generation, which precedes evaluation.
	wg.Go(func() { s.batchSamplingWorker(pipelineCtx, program, examples, scorer) })
	wg.Go(func() { s.candidateGenerationWorker(pipelineCtx) })

	// Start candidate evaluation worker
	wg.Go(func() { s.candidateEvaluationWorker(pipelineCtx, scorer) })

	// Start pipeline coordinator
	return s.pipelineCoordinator(ctx, program, examples, scorer, &wg, pipelineCancel)
}

// batchSamplingWorker waits for full-set acceptance before collecting evidence
// for the next step. No worker reads a pool that another stage is still updating.
func (s *SIMBA) batchSamplingWorker(ctx context.Context, baseProgram core.Program, examples []core.Example, scorer core.Scorer) {
	defer close(s.pipelineChannels.BatchSampling)
	currentProgram := baseProgram
	for step := 0; step < s.config.MaxSteps; step++ {
		if ctx.Err() != nil {
			return
		}
		s.mu.Lock()
		s.state.CurrentStep = step
		s.mu.Unlock()
		startTime := time.Now()
		batch, err := s.prepareStep(ctx, examples, scorer)
		stage := &PipelineStage{StepIndex: step, Candidates: []core.Program{currentProgram}, Batch: batch, Timestamp: startTime, Error: err}
		if !s.safeSendStage(ctx, s.pipelineChannels.BatchSampling, stage) {
			return
		}
		if step+1 == s.config.MaxSteps {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-s.pipelineChannels.Done:
			return
		case result := <-s.pipelineChannels.Results:
			currentProgram = result.Candidates[0]
		}
	}
}

// candidateGenerationWorker builds proposals from the current minibatch evidence.
func (s *SIMBA) candidateGenerationWorker(ctx context.Context) {
	defer close(s.pipelineChannels.CandidateGeneration)

	for {
		select {
		case <-ctx.Done():
			return
		case <-s.pipelineChannels.Done:
			return
		case stage, ok := <-s.pipelineChannels.BatchSampling:
			if !ok {
				return
			}

			if stage.Error != nil {
				select {
				case s.pipelineChannels.Errors <- stage.Error:
				case <-ctx.Done():
					return
				case <-s.pipelineChannels.Done:
					return
				}
				continue
			}

			startTime := time.Now()
			candidates, err := s.generateCandidates(ctx, stage.Candidates[0])
			stage.Candidates = candidates
			stage.Error = err

			if err != nil {
				select {
				case s.pipelineChannels.Errors <- err:
				case <-ctx.Done():
					return
				case <-s.pipelineChannels.Done:
					return
				}
				continue
			}

			s.logger.Debug(ctx, "Pipeline: Candidates generated for step %d in %v",
				stage.StepIndex, time.Since(startTime))

			select {
			case s.pipelineChannels.CandidateGeneration <- stage:
			case <-ctx.Done():
				return
			case <-s.pipelineChannels.Done:
				return
			}
		}
	}
}

// candidateEvaluationWorker evaluates candidates in parallel.
// Note: This worker is launched via WaitGroup.Go() which handles Done() automatically.
func (s *SIMBA) candidateEvaluationWorker(ctx context.Context, scorer core.Scorer) {
	defer close(s.pipelineChannels.CandidateEvaluation)

	for {
		select {
		case <-ctx.Done():
			return
		case <-s.pipelineChannels.Done:
			return
		case stage, ok := <-s.pipelineChannels.CandidateGeneration:
			if !ok {
				return
			}

			if stage.Error != nil {
				select {
				case s.pipelineChannels.Errors <- stage.Error:
				case <-ctx.Done():
					return
				case <-s.pipelineChannels.Done:
					return
				}
				continue
			}

			startTime := time.Now()
			scores, err := s.evaluateCandidates(ctx, stage.Candidates, stage.Batch, scorer)
			if err != nil {
				stage.Error = err
				select {
				case s.pipelineChannels.Errors <- err:
				case <-ctx.Done():
				case <-s.pipelineChannels.Done:
				}
				return
			}
			stage.Scores = scores

			s.logger.Debug(ctx, "Pipeline: Candidates evaluated for step %d in %v",
				stage.StepIndex, time.Since(startTime))

			select {
			case s.pipelineChannels.CandidateEvaluation <- stage:
			case <-ctx.Done():
				return
			case <-s.pipelineChannels.Done:
				return
			}
		}
	}
}

// pipelineCoordinator coordinates the pipeline and collects results.
func (s *SIMBA) pipelineCoordinator(ctx context.Context, initialProgram core.Program, examples []core.Example, scorer core.Scorer, wg *sync.WaitGroup, pipelineCancel context.CancelFunc) (core.Program, error) {
	bestProgram := initialProgram.Clone()
	bestScore := s.state.BestScore
	processedSteps := 0

	// The coordinator owns pipeline cancellation. Stage producers own and close
	// their output channels, so normal upstream completion propagates through
	// the pipeline without depending on this shutdown signal.
	defer func() {
		pipelineCancel()
		close(s.pipelineChannels.Done)
		wg.Wait()
	}()

	stageTimings := make(map[string]time.Duration)

	for processedSteps < s.config.MaxSteps {
		select {
		case <-ctx.Done():
			return bestProgram, ctx.Err()
		case err := <-s.pipelineChannels.Errors:
			s.logger.Error(ctx, "Pipeline error: %v", err)
			return bestProgram, err
		case stage, ok := <-s.pipelineChannels.CandidateEvaluation:
			if !ok {
				if err := ctx.Err(); err != nil {
					return bestProgram, err
				}
				// Workers report an error before closing their output channel. Both
				// receives can therefore become ready together, and select does not
				// prioritize the error case. Preserve the worker error before treating
				// the closed stage channel as successful completion.
				select {
				case err := <-s.pipelineChannels.Errors:
					s.logger.Error(ctx, "Pipeline error: %v", err)
					return bestProgram, err
				default:
				}
				return bestProgram, nil
			}

			processingTime := time.Since(stage.Timestamp)
			stageTimings[fmt.Sprintf("step_%d", stage.StepIndex)] = processingTime

			// Select best candidate from this step
			selectedProgram, _ := s.selectBestCandidate(stage.Candidates, stage.Scores)

			// As in sequential mode, only compare full-dataset scores globally.
			// Candidate/incumbent step selection above remains batch-local.
			selectedFullScore, err := s.evaluateProgramOnExamples(ctx, selectedProgram, examples, scorer)
			if err != nil {
				return bestProgram, err
			}
			s.rememberCandidates(stage.Candidates, stage.Scores, selectedProgram, selectedFullScore)
			improvement := selectedFullScore - bestScore
			if selectedFullScore > bestScore {
				s.mu.Lock()
				s.state.BestScore = selectedFullScore
				s.state.BestProgram = selectedProgram.Clone()
				bestProgram = s.state.BestProgram
				bestScore = selectedFullScore
				s.mu.Unlock()

				s.logger.Info(ctx, "Pipeline: New best score: %.4f (improvement: +%.4f) at step %d",
					selectedFullScore, improvement, stage.StepIndex)
			}

			// Record step metrics
			stepResult := StepResult{
				Step:            stage.StepIndex,
				BestScore:       bestScore,
				CandidateScores: stage.Scores,
				Temperature:     s.getCurrentTemperature(stage.StepIndex),
				BatchSize:       len(stage.Batch),
				Duration:        processingTime,
				Improvement:     improvement,
			}
			s.recordStep(ctx, stepResult)
			processedSteps++
			if s.shouldStopEarly() {
				s.logger.Info(ctx, "Pipeline: Early stopping at step %d", stage.StepIndex)
				return bestProgram, nil
			}

			// Feed back the accepted full-set parent, not the minibatch winner.
			if selectedFullScore < bestScore || math.IsNaN(selectedFullScore) {
				selectedProgram = bestProgram
			}
			if processedSteps < s.config.MaxSteps && !s.safeSendStage(ctx, s.pipelineChannels.Results,
				&PipelineStage{Candidates: []core.Program{selectedProgram.Clone()}}) {
				return bestProgram, ctx.Err()
			}
		}
	}

	return bestProgram, nil
}
