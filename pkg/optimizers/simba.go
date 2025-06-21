package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	dspyModules "github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/sourcegraph/conc/pool"
)

// SIMBAConfig contains configuration options for SIMBA optimizer.
type SIMBAConfig struct {
	// Mini-batch configuration
	BatchSize     int `json:"batch_size"`     // Default: 32
	MaxSteps      int `json:"max_steps"`      // Default: 8
	NumCandidates int `json:"num_candidates"` // Default: 6

	// Temperature controls
	SamplingTemperature float64 `json:"sampling_temperature"` // Default: 0.2

	// Introspective learning
	IntrospectionFrequency int `json:"introspection_frequency"` // Default: 2

	// Performance thresholds
	ConvergenceThreshold float64 `json:"convergence_threshold"` // Default: 0.001
	MinImprovementRatio  float64 `json:"min_improvement_ratio"` // Default: 0.05

	// Concurrency and resources
	MaxGoroutines int `json:"max_goroutines"` // Default: 10
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
}

// CandidateResult represents a candidate program and its performance.
type CandidateResult struct {
	Program     core.Program `json:"-"`
	Score       float64      `json:"score"`
	Step        int          `json:"step"`
	Temperature float64      `json:"temperature"`
	CreatedAt   time.Time    `json:"created_at"`
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

// SIMBA implements Stochastic Introspective Mini-Batch Ascent optimizer.
type SIMBA struct {
	// Core configuration
	config SIMBAConfig

	// Evaluation metric (set during Compile)
	metric func(expected, actual map[string]interface{}) float64

	// Language models
	primaryModel  core.LLM // Primary optimization model
	analyzerModel core.LLM // Introspective analysis model

	// Internal state
	state  *SIMBAState
	logger *logging.Logger
	rng    *rand.Rand

	// Thread safety
	mu sync.RWMutex
}

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

// WithSIMBANumCandidates sets the number of candidate programs per iteration.
func WithSIMBANumCandidates(num int) SIMBAOption {
	return func(s *SIMBA) {
		s.config.NumCandidates = num
	}
}

// WithSamplingTemperature sets the sampling temperature.
func WithSamplingTemperature(temperature float64) SIMBAOption {
	return func(s *SIMBA) {
		s.config.SamplingTemperature = temperature
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
		},
		metric: nil, // Will be set during Compile
		state: &SIMBAState{
			CurrentStep:      0,
			BestScore:        0.0,
			CandidateHistory: make([]CandidateResult, 0),
			PerformanceLog:   make([]StepResult, 0),
			IntrospectionLog: make([]string, 0),
			StartTime:        time.Now(),
		},
		logger: logging.GetLogger(),
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Apply options
	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Compile implements the core.Optimizer interface for SIMBA.
func (s *SIMBA) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}

	ctx, span := core.StartSpan(ctx, "SIMBA.Compile")
	defer core.EndSpan(ctx)

	s.logger.Info(ctx, "Starting SIMBA optimization with config: batch_size=%d, max_steps=%d, num_candidates=%d",
		s.config.BatchSize, s.config.MaxSteps, s.config.NumCandidates)

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

	// Convert core.Metric to SIMBA's metric format
	simbaMetric := func(expected, actual map[string]interface{}) float64 {
		return metric(expected, actual)
	}
	s.metric = simbaMetric

	// Reset state for new compilation
	s.resetState()

	// Initialize with current program as best candidate
	bestProgram := program.Clone()
	initialScore := s.evaluateProgram(ctx, bestProgram, dataset)
	s.state.BestScore = initialScore
	s.state.BestProgram = bestProgram

	s.logger.Info(ctx, "Initial program score: %.4f", initialScore)

	// Main optimization loop
	for step := 0; step < s.config.MaxSteps; step++ {
		stepCtx, stepSpan := core.StartSpan(ctx, fmt.Sprintf("SIMBA.Step.%d", step))
		stepStart := time.Now()

		s.mu.Lock()
		s.state.CurrentStep = step
		s.mu.Unlock()

		// Generate candidate programs
		candidates, err := s.generateCandidates(stepCtx, bestProgram)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, errors.WithFields(
				errors.New(errors.ValidationFailed, "failed to generate candidates"),
				errors.Fields{"step": step, "error": err.Error()},
			)
		}

		// Process mini-batch and evaluate candidates
		batch, err := s.sampleMiniBatch(stepCtx, dataset)
		if err != nil {
			stepSpan.WithError(err)
			core.EndSpan(stepCtx)
			return program, errors.WithFields(
				errors.New(errors.ValidationFailed, "failed to sample mini-batch"),
				errors.Fields{"step": step, "error": err.Error()},
			)
		}

		scores := s.evaluateCandidates(stepCtx, candidates, batch)

		// Select best candidate using temperature-controlled sampling
		selectedProgram, selectedScore := s.selectBestCandidate(candidates, scores)

		// Update best program if improvement found
		improvement := selectedScore - s.state.BestScore
		if selectedScore > s.state.BestScore {
			s.mu.Lock()
			s.state.BestScore = selectedScore
			s.state.BestProgram = selectedProgram.Clone()
			bestProgram = s.state.BestProgram
			s.mu.Unlock()
			s.logger.Info(stepCtx, "New best score: %.4f (improvement: +%.4f)", selectedScore, improvement)
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

		// Perform introspective analysis periodically
		if step%s.config.IntrospectionFrequency == 0 && step > 0 {
			introspection := s.performIntrospection(stepCtx)
			stepResult.Introspection = introspection.Analysis
			s.state.IntrospectionLog = append(s.state.IntrospectionLog, introspection.Analysis)
		}

		s.state.PerformanceLog = append(s.state.PerformanceLog, stepResult)

		// Check for convergence
		if s.hasConverged() {
			s.logger.Info(stepCtx, "Optimization converged at step %d", step)
			stepSpan.WithAnnotation("converged", true)
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

// resetState initializes the optimization state.
func (s *SIMBA) resetState() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.state.CurrentStep = 0
	s.state.BestScore = 0.0
	s.state.CandidateHistory = make([]CandidateResult, 0)
	s.state.PerformanceLog = make([]StepResult, 0)
	s.state.IntrospectionLog = make([]string, 0)
	s.state.StartTime = time.Now()
}

// evaluateProgram evaluates a program against the full dataset.
func (s *SIMBA) evaluateProgram(ctx context.Context, program core.Program, dataset core.Dataset) float64 {
	var totalScore float64
	var count int

	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}

		prediction, err := program.Forward(ctx, example.Inputs)
		if err != nil {
			s.logger.Debug(ctx, "Program evaluation failed for example: %v", err)
			// Assign penalty score for failed predictions
			totalScore += 0.0
			count++
			continue
		}

		score := s.metric(example.Outputs, prediction)
		totalScore += score
		count++
	}

	if count == 0 {
		return 0.0
	}
	return totalScore / float64(count)
}

// generateCandidates creates candidate programs for the current step.
func (s *SIMBA) generateCandidates(ctx context.Context, baseProgram core.Program) ([]core.Program, error) {
	candidates := make([]core.Program, 0, s.config.NumCandidates)

	// Include the base program as first candidate
	candidates = append(candidates, baseProgram.Clone())

	// Generate variations using instruction perturbation
	for i := 1; i < s.config.NumCandidates; i++ {
		candidate, err := s.perturbProgram(ctx, baseProgram)
		if err != nil {
			s.logger.Debug(ctx, "Failed to generate candidate %d: %v", i, err)
			continue
		}
		candidates = append(candidates, candidate)
	}

	if len(candidates) == 0 {
		return nil, errors.New(errors.ValidationFailed, "failed to generate any candidates")
	}

	return candidates, nil
}

// perturbProgram creates a variation of the base program by modifying instructions.
func (s *SIMBA) perturbProgram(ctx context.Context, baseProgram core.Program) (core.Program, error) {
	variant := baseProgram.Clone()
	modules := variant.GetModules()

	if len(modules) == 0 {
		return variant, nil
	}

	// Randomly select a module to modify
	moduleIdx := s.rng.Intn(len(modules))
	module := modules[moduleIdx]

	// Generate instruction variation if it's a Predict module
	if predictor, ok := module.(*dspyModules.Predict); ok {
		newInstruction, err := s.generateInstructionVariation(ctx, predictor.GetSignature().Instruction)
		if err != nil {
			return variant, err
		}

		// Update the module's signature
		signature := predictor.GetSignature()
		signature.Instruction = newInstruction
		predictor.SetSignature(signature)
	}

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

	response, err := s.primaryModel.Generate(ctx, prompt)
	if err != nil {
		// Handle LLM failures gracefully by returning original instruction
		return originalInstruction, nil
	}

	if response.Content == "" {
		return originalInstruction, nil
	}

	return response.Content, nil
}

// sampleMiniBatch creates a random mini-batch from the dataset.
func (s *SIMBA) sampleMiniBatch(ctx context.Context, dataset core.Dataset) ([]core.Example, error) {
	// Collect all examples first (simple implementation)
	allExamples := make([]core.Example, 0)
	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}
		allExamples = append(allExamples, example)
	}

	if len(allExamples) == 0 {
		return nil, errors.New(errors.ValidationFailed, "dataset is empty")
	}

	// Sample random batch
	batchSize := s.config.BatchSize
	if batchSize > len(allExamples) {
		batchSize = len(allExamples)
	}

	batch := make([]core.Example, 0, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := s.rng.Intn(len(allExamples))
		batch = append(batch, allExamples[idx])
	}

	return batch, nil
}

// evaluateCandidates evaluates all candidate programs on the mini-batch.
func (s *SIMBA) evaluateCandidates(ctx context.Context, candidates []core.Program, batch []core.Example) []float64 {
	scores := make([]float64, len(candidates))

	p := pool.New().WithMaxGoroutines(s.config.MaxGoroutines)
	defer p.Wait()

	for i, candidate := range candidates {
		i, candidate := i, candidate
		p.Go(func() {
			scores[i] = s.evaluateCandidateOnBatch(ctx, candidate, batch)
		})
	}

	return scores
}

// evaluateCandidateOnBatch evaluates a single candidate on the mini-batch.
func (s *SIMBA) evaluateCandidateOnBatch(ctx context.Context, candidate core.Program, batch []core.Example) float64 {
	var totalScore float64
	var count int

	for _, example := range batch {
		prediction, err := candidate.Forward(ctx, example.Inputs)
		if err != nil {
			s.logger.Debug(ctx, "Candidate evaluation failed for example: %v", err)
			// Assign penalty score for failed predictions
			totalScore += 0.0
			count++
			continue
		}

		score := s.metric(example.Outputs, prediction)
		totalScore += score
		count++
	}

	if count == 0 {
		return 0.0
	}
	return totalScore / float64(count)
}

// selectBestCandidate uses temperature-controlled sampling to select program.
func (s *SIMBA) selectBestCandidate(candidates []core.Program, scores []float64) (core.Program, float64) {
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

	// Use temperature-controlled sampling for exploration
	temperature := s.getCurrentTemperature(s.state.CurrentStep)
	if temperature > 0 && s.rng.Float64() < 0.3 { // 30% chance for exploration
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

	// Sample from probability distribution
	r := s.rng.Float64()
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

// hasConverged checks if optimization has converged.
func (s *SIMBA) hasConverged() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	log := s.state.PerformanceLog
	if len(log) < 3 {
		return false
	}

	// Check if recent improvements are below threshold
	recentSteps := 3
	if len(log) < recentSteps {
		recentSteps = len(log)
	}

	var totalImprovement float64
	for i := len(log) - recentSteps; i < len(log); i++ {
		totalImprovement += math.Abs(log[i].Improvement)
	}

	avgImprovement := totalImprovement / float64(recentSteps)
	return avgImprovement < s.config.ConvergenceThreshold
}

// GetState returns the current optimization state (thread-safe).
func (s *SIMBA) GetState() SIMBAState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return *s.state
}

// GetConfig returns the current configuration.
func (s *SIMBA) GetConfig() SIMBAConfig {
	return s.config
}
