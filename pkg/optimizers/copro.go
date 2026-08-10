package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// COPRO implements the Chain-of-Processing optimizer for prompt instruction and prefix optimization.
type COPRO struct {
	PromptModel     core.LLM // Optional model for generating prompts (if nil, uses default)
	Metric          core.Metric
	Breadth         int     // Number of prompt candidates to generate
	Depth           int     // Iterations of prompt refinement
	InitTemperature float64 // Randomness in prompt generation
	TrackStats      bool    // Optional performance tracking

	// LLM-assisted prompt generation seeds. Each predictor receives independent
	// operation-local state derived from these values.
	PromptGenerator  *LLMPromptGenerator
	CandidateHistory []PromptCandidate
}

var (
	_ core.Optimizer        = (*COPRO)(nil)
	_ core.ExamplesCompiler = (*COPRO)(nil)
)

// PromptCandidate represents a candidate prompt configuration.
type PromptCandidate struct {
	Instruction     string
	Prefix          string
	Score           float64 // Training score
	ValidationScore float64 // Validation score to prevent overfitting
	Generation      int     // Which depth iteration this was generated in
	Diversity       float64 // Semantic diversity score
	Rank            int     // Performance ranking
	AttemptID       string  // Unique identifier for tracking
}

type coproPromptState struct {
	generator *LLMPromptGenerator
	history   []PromptCandidate
}

// COPROOptions provides configuration options for COPRO.
type COPROOptions struct {
	PromptModel     core.LLM
	Breadth         int
	Depth           int
	InitTemperature float64
	TrackStats      bool
}

// COPROOption is a functional option for configuring COPRO.
type COPROOption func(*COPROOptions)

// WithPromptModel sets the model used for generating prompts.
func WithPromptModel(model core.LLM) COPROOption {
	return func(opts *COPROOptions) {
		opts.PromptModel = model
	}
}

// WithBreadth sets the number of prompt candidates to generate.
func WithBreadth(breadth int) COPROOption {
	return func(opts *COPROOptions) {
		opts.Breadth = breadth
	}
}

// WithDepth sets the number of refinement iterations.
func WithDepth(depth int) COPROOption {
	return func(opts *COPROOptions) {
		opts.Depth = depth
	}
}

// WithInitTemperature sets the randomness in prompt generation.
func WithInitTemperature(temp float64) COPROOption {
	return func(opts *COPROOptions) {
		opts.InitTemperature = temp
	}
}

// WithTrackStats enables performance tracking.
func WithTrackStats(track bool) COPROOption {
	return func(opts *COPROOptions) {
		opts.TrackStats = track
	}
}

// NewCOPRO creates a new COPRO optimizer with enhanced LLM-assisted prompt generation.
func NewCOPRO(metric core.Metric, options ...COPROOption) *COPRO {
	opts := &COPROOptions{
		Breadth:         5,   // Reduced for higher quality candidates
		Depth:           2,   // Match Python DSPy default
		InitTemperature: 1.2, // Match Python DSPy default
		TrackStats:      false,
	}

	for _, option := range options {
		option(opts)
	}

	return &COPRO{
		PromptModel:      opts.PromptModel,
		Metric:           metric,
		Breadth:          opts.Breadth,
		Depth:            opts.Depth,
		InitTemperature:  opts.InitTemperature,
		TrackStats:       opts.TrackStats,
		CandidateHistory: make([]PromptCandidate, 0),
	}
}

// Compile implements the legacy cursor-based Optimizer API. It materializes
// the dataset once before optimizing any predictors.
func (c *COPRO) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	ctx, compileMetric, err := c.prepareCompile(ctx, metric)
	if err != nil {
		return program, err
	}

	// Preserve the legacy behavior of returning immediately when there is
	// nothing to optimize without consuming the dataset.
	if len(c.extractPredictors(program)) == 0 {
		return c.compileExamples(ctx, program, nil, compileMetric)
	}

	examples, err := core.MaterializeDatasetContext(ctx, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to materialize dataset: %w", err)
	}
	return c.compileExamples(ctx, program, examples, compileMetric)
}

// CompileExamples optimizes a program from materialized, read-only examples.
func (c *COPRO) CompileExamples(ctx context.Context, program core.Program, examples []core.Example, metric core.Metric) (core.Program, error) {
	ctx, compileMetric, err := c.prepareCompile(ctx, metric)
	if err != nil {
		return program, err
	}
	return c.compileExamples(ctx, program, examples, compileMetric)
}

func (c *COPRO) prepareCompile(ctx context.Context, metric core.Metric) (context.Context, core.Metric, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return ctx, nil, err
	}
	if metric == nil {
		metric = c.Metric
	}
	if metric == nil {
		return ctx, nil, fmt.Errorf("COPRO requires a metric function")
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}
	return ctx, metric, nil
}

func (c *COPRO) compileExamples(ctx context.Context, program core.Program, examples []core.Example, metric core.Metric) (core.Program, error) {
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, "COPROCompilation")
	defer core.EndSpan(ctx)

	// Clone the program for optimization
	optimizedProgram := program.Clone()

	// Extract all Predict modules that need optimization
	predictors := c.extractPredictors(optimizedProgram)
	if len(predictors) == 0 {
		logger.Info(ctx, "COPRO: No Predict modules found to optimize")
		return optimizedProgram, nil
	}

	logger.Info(ctx, "COPRO: Found %d Predict modules to optimize", len(predictors))

	moduleNames := make([]string, 0, len(predictors))
	for moduleName := range predictors {
		moduleNames = append(moduleNames, moduleName)
	}
	sort.Strings(moduleNames)

	// Optimize each predictor's prompts in deterministic order.
	for _, moduleName := range moduleNames {
		if err := ctx.Err(); err != nil {
			return optimizedProgram, err
		}
		predictor := predictors[moduleName]
		moduleCtx, moduleSpan := core.StartSpan(ctx, fmt.Sprintf("OptimizePredictor_%s", moduleName))

		err := c.optimizePredictor(moduleCtx, predictor, examples, metric)
		if err != nil {
			moduleSpan.WithError(err)
			core.EndSpan(moduleCtx)
			return optimizedProgram, fmt.Errorf("error optimizing predictor %s: %w", moduleName, err)
		}

		core.EndSpan(moduleCtx)
	}

	span.WithAnnotation("optimized_predictors", len(predictors))
	return optimizedProgram, nil
}

// extractPredictors finds all Predict modules in the program.
func (c *COPRO) extractPredictors(program core.Program) map[string]*modules.Predict {
	predictors := make(map[string]*modules.Predict)

	for moduleName, module := range program.Modules {
		if predictor, ok := module.(*modules.Predict); ok {
			predictors[moduleName] = predictor
		}
	}

	return predictors
}

// optimizePredictor optimizes prompts for a single Predict module.
func (c *COPRO) optimizePredictor(ctx context.Context, predictor *modules.Predict, examples []core.Example, metric core.Metric) error {
	ctx, span := core.StartSpan(ctx, "OptimizePredictorPrompts")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()

	if err := ctx.Err(); err != nil {
		return err
	}
	if len(examples) == 0 {
		return fmt.Errorf("no examples in dataset for optimization")
	}

	// Split examples into train/validation to prevent overfitting
	trainSize := len(examples) * 2 / 3 // Use 2/3 for training
	if trainSize < 1 {
		trainSize = 1
	}
	trainExamples := examples[:trainSize]
	validationExamples := examples[trainSize:]
	logger.Info(ctx, "COPRO: Using %d examples for training, %d for validation", len(trainExamples), len(validationExamples))

	promptState := c.newPromptState(ctx, predictor)

	// Get current instruction and prefix as baseline
	signature := predictor.GetSignature()
	currentInstruction := signature.Instruction
	currentPrefix := "" // Extract from signature if available

	// Initialize candidates with current configuration
	var candidates []PromptCandidate

	// Add current configuration as baseline
	if currentInstruction != "" {
		baseline := PromptCandidate{
			Instruction: currentInstruction,
			Prefix:      currentPrefix,
			Generation:  0,
		}
		var err error
		baseline.Score, err = c.evaluateCandidate(ctx, predictor, baseline, trainExamples, metric)
		if err != nil {
			return err
		}
		candidates = append(candidates, baseline)
		logger.Info(ctx, "COPRO: Baseline score: %.3f", baseline.Score)
	}

	// Generate initial candidates
	initialCandidates := c.generateInitialCandidates(ctx, predictor, currentInstruction, promptState)
	if err := ctx.Err(); err != nil {
		return err
	}

	// Evaluate initial candidates in parallel on training data
	if err := c.evaluateCandidatesParallel(ctx, predictor, initialCandidates, trainExamples, metric); err != nil {
		return err
	}

	candidates = append(candidates, initialCandidates...)

	// Iterative refinement across depth levels
	for depth := 1; depth <= c.Depth; depth++ {
		depthCtx, depthSpan := core.StartSpan(ctx, fmt.Sprintf("Depth_%d", depth))

		// Select top candidates from previous iteration
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Score > candidates[j].Score
		})

		// Keep top candidates for refinement
		topK := min(c.Breadth/2, len(candidates))
		topCandidates := candidates[:topK]

		// Generate refined candidates
		refinedCandidates := c.refineCandidates(depthCtx, predictor, topCandidates, depth, promptState)
		if err := depthCtx.Err(); err != nil {
			core.EndSpan(depthCtx)
			return err
		}

		// Evaluate refined candidates in parallel on training data
		if err := c.evaluateCandidatesParallel(depthCtx, predictor, refinedCandidates, trainExamples, metric); err != nil {
			core.EndSpan(depthCtx)
			return err
		}

		// Add refined candidates to pool
		candidates = append(candidates, refinedCandidates...)

		// Log progress
		bestScore := candidates[0].Score
		if len(refinedCandidates) > 0 {
			sort.Slice(refinedCandidates, func(i, j int) bool {
				return refinedCandidates[i].Score > refinedCandidates[j].Score
			})
			logger.Info(depthCtx, "COPRO: Depth %d - Best refined score: %.3f (overall best: %.3f)",
				depth, refinedCandidates[0].Score, bestScore)
		}

		depthSpan.WithAnnotation("refined_candidates", len(refinedCandidates))
		depthSpan.WithAnnotation("best_score", bestScore)
		core.EndSpan(depthCtx)
	}

	// Validate all candidates on validation set to prevent overfitting
	logger.Info(ctx, "COPRO: Validating %d candidates on %d validation examples", len(candidates), len(validationExamples))
	for i := range candidates {
		var err error
		candidates[i].ValidationScore, err = c.evaluateCandidate(ctx, predictor, candidates[i], validationExamples, metric)
		if err != nil {
			return err
		}
	}

	// Select candidate with best validation score (not training score)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].ValidationScore > candidates[j].ValidationScore
	})

	if len(candidates) > 0 {
		bestCandidate := candidates[0]
		logger.Info(ctx, "COPRO: Selected best candidate - Training: %.3f, Validation: %.3f, Instruction: %s",
			bestCandidate.Score, bestCandidate.ValidationScore, c.truncateString(bestCandidate.Instruction, 100))

		// Log overfitting warning if training score much higher than validation
		overfitGap := bestCandidate.Score - bestCandidate.ValidationScore
		if overfitGap > 0.2 {
			logger.Warn(ctx, "COPRO: Potential overfitting detected - gap: %.3f (training: %.3f vs validation: %.3f)",
				overfitGap, bestCandidate.Score, bestCandidate.ValidationScore)
		}

		// Apply the best prompt to the predictor
		c.applyPromptToPredictor(predictor, bestCandidate)

		span.WithAnnotation("best_training_score", bestCandidate.Score)
		span.WithAnnotation("best_validation_score", bestCandidate.ValidationScore)
		span.WithAnnotation("total_candidates", len(candidates))
	}

	return nil
}

// generateInitialCandidates generates sophisticated initial prompt candidates using LLM assistance.
func (c *COPRO) generateInitialCandidates(ctx context.Context, predictor *modules.Predict, baseInstruction string, state *coproPromptState) []PromptCandidate {
	// Get the signature to understand the task
	signature := predictor.GetSignature()
	logger := logging.GetLogger()

	taskDescription := c.getTaskDescription(signature, baseInstruction)
	instructions := []string{}
	if state.generator != nil {
		// Generate sophisticated instructions using LLM assistance with retry logic
		generated, err := state.generator.generateBasicInstructionsWithRetry(ctx, taskDescription, c.Breadth, c.InitTemperature)
		logger.Info(ctx, "COPRO: Generated %d initial candidates", len(generated))
		if err != nil {
			logger.Error(ctx, "COPRO: Failed to generate LLM-assisted instructions after retries, falling back to enhanced templates: %v", err)
		} else {
			instructions = generated
		}
	}
	if len(instructions) == 0 {
		if state.generator == nil {
			logger.Info(ctx, "COPRO: No prompt generation model resolved, falling back to enhanced templates")
		}
		instructions = c.getEnhancedInstructionTemplates(signature, baseInstruction)
	}

	// Create candidates with diversity scoring
	var candidates []PromptCandidate
	for i, instruction := range instructions {
		if i >= c.Breadth {
			break
		}

		candidate := PromptCandidate{
			Instruction: instruction,
			Prefix:      "",
			Generation:  1,
			AttemptID:   fmt.Sprintf("init_%d", i),
			Rank:        i,
		}
		candidates = append(candidates, candidate)
	}

	// Calculate diversity scores
	c.calculatePromptDiversity(candidates)

	return candidates
}

func (c *COPRO) resolvePromptLLM(ctx context.Context, predictor *modules.Predict) core.LLM {
	if c.PromptModel != nil {
		return c.PromptModel
	}
	info := core.NewModuleInfo(predictor.GetDisplayName(), predictor.GetModuleType(), predictor.GetSignature()).WithLLM(predictor.LLM)
	return core.ResolveDefaultLLM(ctx, info)
}

func (c *COPRO) newPromptState(ctx context.Context, predictor *modules.Predict) *coproPromptState {
	state := &coproPromptState{
		history: append([]PromptCandidate(nil), c.CandidateHistory...),
	}

	var promptLLM core.PromptModel
	if c.PromptGenerator != nil && c.PromptGenerator.llm != nil {
		promptLLM = c.PromptGenerator.llm
	} else {
		promptLLM = c.resolvePromptLLM(ctx, predictor)
	}
	if promptLLM != nil {
		state.generator = NewLLMPromptGenerator(promptLLM, predictor.GetSignature())
		if c.PromptGenerator != nil {
			state.generator.diversityThreshold = c.PromptGenerator.diversityThreshold
		}
	}
	return state
}

// refineCandidates generates sophisticated refined candidates using performance feedback.
func (c *COPRO) refineCandidates(ctx context.Context, predictor *modules.Predict, topCandidates []PromptCandidate, depth int, state *coproPromptState) []PromptCandidate {
	// Add current candidates to this predictor's history for learning.
	state.history = append(state.history, topCandidates...)

	// Try LLM-assisted refinement first
	if state.generator != nil && len(state.history) >= 3 {
		// Use LLM to generate refined instructions based on performance history
		refinedInstructions, err := state.generator.generateRefinedInstructions(ctx, state.history, c.Breadth, c.InitTemperature*math.Pow(0.8, float64(depth)))
		if err == nil && len(refinedInstructions) > 0 {
			var refined []PromptCandidate
			for i, instruction := range refinedInstructions {
				if i >= c.Breadth {
					break
				}

				refinedCandidate := PromptCandidate{
					Instruction: instruction,
					Prefix:      "",
					Generation:  depth + 1,
					AttemptID:   fmt.Sprintf("refined_%d_%d", depth, i),
				}
				refined = append(refined, refinedCandidate)
			}

			// Calculate diversity for refined candidates
			c.calculatePromptDiversity(refined)
			return refined
		}
	}

	// Fallback to enhanced refinement strategy
	var refined []PromptCandidate
	for _, candidate := range topCandidates {
		numRefinements := max(1, c.Breadth/len(topCandidates))

		for i := 0; i < numRefinements; i++ {
			temp := c.InitTemperature * math.Pow(0.8, float64(depth))
			refinedInstruction := c.refineInstruction(candidate.Instruction, temp)

			// The fallback strategy must always generate at least some candidate
			// variation; otherwise COPRO can stall on a generation and tests become flaky.
			if refinedInstruction == candidate.Instruction {
				refinedInstruction = c.forceRefineInstruction(candidate.Instruction, depth+i)
			}

			refinedCandidate := PromptCandidate{
				Instruction: refinedInstruction,
				Prefix:      candidate.Prefix,
				Generation:  depth + 1,
				AttemptID:   fmt.Sprintf("basic_refined_%d_%d", depth, i),
			}
			refined = append(refined, refinedCandidate)
		}
	}

	// Calculate diversity scores
	c.calculatePromptDiversity(refined)
	return refined
}

// evaluateCandidate evaluates a prompt candidate using the metric.
func (c *COPRO) evaluateCandidate(ctx context.Context, predictor *modules.Predict, candidate PromptCandidate, examples []core.Example, metric core.Metric) (float64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Temporarily apply the candidate prompt
	originalSignature := predictor.GetSignature()
	tempSignature := originalSignature.WithInstruction(candidate.Instruction)

	// Create a temporary predictor with the candidate prompt
	tempPredictor := modules.NewPredict(tempSignature)
	tempPredictor.SetLLM(predictor.LLM)

	// Evaluate all examples in parallel for better performance
	scores := make([]float64, len(examples))
	valid := make([]bool, len(examples))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errOnce sync.Once
	var evaluationErr error
	recordError := func(err error) {
		errOnce.Do(func() {
			evaluationErr = err
		})
	}

	// Use a semaphore to limit concurrent LLM calls per candidate
	semaphore := make(chan struct{}, 10) // Allow 10 concurrent evaluations per candidate

	for i, example := range examples {
		idx, ex := i, example // Capture loop variables for Go 1.25 closure
		wg.Go(func() {
			select {
			case semaphore <- struct{}{}: // Acquire semaphore
				defer func() { <-semaphore }() // Release semaphore
			case <-ctx.Done():
				recordError(ctx.Err())
				return
			}

			// Get prediction with candidate prompt
			prediction, err := tempPredictor.Process(ctx, ex.Inputs)
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					recordError(ctxErr)
					return
				}
				logger := logging.GetLogger()
				logger.Error(ctx, "COPRO: Error evaluating candidate: %v", err)
				return
			}
			if ctxErr := ctx.Err(); ctxErr != nil {
				recordError(ctxErr)
				return
			}

			// Evaluate using metric
			score := metric(ex.Outputs, prediction)
			mu.Lock()
			scores[idx] = score
			valid[idx] = true
			mu.Unlock()
		})
	}

	wg.Wait()
	if evaluationErr != nil {
		return 0, evaluationErr
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Calculate average score
	var totalScore float64
	validEvaluations := 0
	for i, score := range scores {
		if valid[i] {
			totalScore += score
			validEvaluations++
		}
	}

	if validEvaluations == 0 {
		return 0, nil
	}

	return totalScore / float64(validEvaluations), nil
}

// evaluateCandidatesParallel evaluates multiple candidates in parallel for better performance.
func (c *COPRO) evaluateCandidatesParallel(ctx context.Context, predictor *modules.Predict, candidates []PromptCandidate, examples []core.Example, metric core.Metric) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	const maxGoroutines = 20 // Increased concurrency for better performance
	semaphore := make(chan struct{}, maxGoroutines)
	var wg sync.WaitGroup
	var errOnce sync.Once
	var evaluationErr error

	for i := range candidates {
		wg.Go(func() {
			select {
			case semaphore <- struct{}{}: // Acquire semaphore
				defer func() { <-semaphore }() // Release semaphore
			case <-ctx.Done():
				errOnce.Do(func() { evaluationErr = ctx.Err() })
				return
			}

			score, err := c.evaluateCandidate(ctx, predictor, candidates[i], examples, metric)
			if err != nil {
				errOnce.Do(func() { evaluationErr = err })
				return
			}
			candidates[i].Score = score
		})
	}

	wg.Wait()
	if evaluationErr != nil {
		return evaluationErr
	}
	return ctx.Err()
}

// LLMPromptGenerator handles sophisticated prompt generation using LLM assistance.
type LLMPromptGenerator struct {
	llm                core.PromptModel
	signature          core.Signature
	candidateCache     map[string]PromptCandidate
	diversityThreshold float64
}

// NewLLMPromptGenerator creates a new LLM-assisted prompt generator.
func NewLLMPromptGenerator(llm core.PromptModel, signature core.Signature) *LLMPromptGenerator {
	return &LLMPromptGenerator{
		llm:                llm,
		signature:          signature,
		candidateCache:     make(map[string]PromptCandidate),
		diversityThreshold: 0.7,
	}
}

// generateBasicInstructionsWithRetry creates initial high-quality instruction candidates using LLM with retry logic.
func (lpg *LLMPromptGenerator) generateBasicInstructionsWithRetry(ctx context.Context, taskDescription string, breadth int, temperature float64) ([]string, error) {
	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		instructions, err := lpg.generateBasicInstructions(ctx, taskDescription, breadth, temperature)
		if err == nil && len(instructions) >= breadth/2 { // Accept if we get at least half the requested instructions
			return instructions, nil
		}
		// Increase temperature for retry to get more diverse results
		temperature = temperature * 1.2
	}
	return nil, fmt.Errorf("failed to generate instructions after %d retries", maxRetries)
}

// generateBasicInstructions creates initial high-quality instruction candidates using LLM.
func (lpg *LLMPromptGenerator) generateBasicInstructions(ctx context.Context, taskDescription string, breadth int, temperature float64) ([]string, error) {
	// Create a sophisticated prompt for LLM-assisted instruction generation
	inputFields := strings.Join(getFieldNames(lpg.signature.Inputs), ", ")
	outputFields := strings.Join(getFieldNames(lpg.signature.Outputs), ", ")

	generatorPrompt := fmt.Sprintf(`You are an expert prompt engineer specializing in question-answering tasks. Generate %d high-quality, diverse instruction variations that will help a language model answer questions accurately.

Current Task: Convert "%s" into "%s"
Task Context: %s

Create %d DIFFERENT instruction approaches, each using a unique strategy:

Strategy Examples:
- Direct factual approach: "Answer the question with accurate, factual information"
- Analytical approach: "Analyze the question carefully and provide a well-reasoned answer"
- Step-by-step approach: "Break down the question step-by-step before answering"
- Comprehensive approach: "Provide a thorough and complete answer to the question"
- Precise approach: "Give a precise, specific answer to the question asked"

Requirements for each instruction:
1. Must be 10-25 words long
2. Should use different verbs and approaches
3. Must guide the model to produce accurate answers
4. Should NOT be generic or vague
5. Each must be clearly distinct from the others

Return EXACTLY %d instructions, one per line, no numbering:`,
		breadth, inputFields, outputFields, taskDescription, breadth, breadth)

	// Use LLM to generate sophisticated instructions
	core.RecordModelCall(ctx, lpg.llm)
	output, err := lpg.llm.Generate(ctx, generatorPrompt,
		core.WithTemperature(temperature),
		core.WithMaxTokens(1024))
	if err != nil {
		return nil, fmt.Errorf("failed to generate instructions: %w", err)
	}

	// Parse the generated instructions
	response := output.Content

	instructions := strings.Split(strings.TrimSpace(response), "\n")

	// Clean and validate instructions
	var validInstructions []string
	for _, inst := range instructions {
		cleaned := strings.TrimSpace(inst)
		if len(cleaned) > 10 && !strings.HasPrefix(cleaned, "#") {
			validInstructions = append(validInstructions, cleaned)
		}
	}

	// If we didn't get enough valid instructions, add some fallbacks
	if len(validInstructions) < breadth {
		fallbacks := lpg.getFallbackInstructions()
		for i := len(validInstructions); i < breadth && i < len(fallbacks); i++ {
			validInstructions = append(validInstructions, fallbacks[i])
		}
	}

	return validInstructions[:min(len(validInstructions), breadth)], nil
}

// generateRefinedInstructions creates improved instructions based on previous attempts.
func (lpg *LLMPromptGenerator) generateRefinedInstructions(ctx context.Context, history []PromptCandidate, breadth int, temperature float64) ([]string, error) {
	if len(history) == 0 {
		return nil, fmt.Errorf("no history available for refinement")
	}

	// Sort history by score to get best and worst performers
	sort.Slice(history, func(i, j int) bool {
		return history[i].Score > history[j].Score
	})

	// Create refinement prompt with performance feedback
	bestInstructions := ""
	worstInstructions := ""

	for i := 0; i < min(3, len(history)); i++ {
		bestInstructions += fmt.Sprintf("- %s (score: %.3f)\n", history[i].Instruction, history[i].Score)
	}

	for i := max(0, len(history)-3); i < len(history); i++ {
		worstInstructions += fmt.Sprintf("- %s (score: %.3f)\n", history[i].Instruction, history[i].Score)
	}

	refinementPrompt := fmt.Sprintf(`You are an expert prompt engineer analyzing previous instruction attempts. Based on the performance data below, generate %d improved instruction variations.

Task: %s

HIGH-PERFORMING INSTRUCTIONS:
%s
LOW-PERFORMING INSTRUCTIONS:
%s

Analyze what made the high-performing instructions successful and what made the low-performing ones less effective. Generate %d new, improved instructions that:
1. Build on the successful patterns from high-performers
2. Avoid the weaknesses of low-performers
3. Introduce new effective approaches
4. Are diverse and semantically distinct
5. Provide clear, actionable guidance

Return ONLY the improved instructions, one per line.`,
		breadth, lpg.getTaskDescription(),
		bestInstructions, worstInstructions, breadth)

	core.RecordModelCall(ctx, lpg.llm)
	output, err := lpg.llm.Generate(ctx, refinementPrompt,
		core.WithTemperature(temperature),
		core.WithMaxTokens(1024))
	if err != nil {
		return nil, fmt.Errorf("failed to generate refined instructions: %w", err)
	}

	response := output.Content

	instructions := strings.Split(strings.TrimSpace(response), "\n")
	var validInstructions []string
	for _, inst := range instructions {
		cleaned := strings.TrimSpace(inst)
		if len(cleaned) > 10 {
			validInstructions = append(validInstructions, cleaned)
		}
	}

	return validInstructions[:min(len(validInstructions), breadth)], nil
}

// getFallbackInstructions provides sophisticated fallback instructions.
func (lpg *LLMPromptGenerator) getFallbackInstructions() []string {
	return []string{
		"Answer the question directly with accurate, factual information.",
		"Provide a clear, concise response based on the given question.",
		"Think carefully about the question and give a precise answer.",
		"Analyze the question and respond with relevant, correct information.",
		"Give a straightforward answer to the specific question asked.",
		"Consider the question carefully and provide an accurate response.",
		"Respond to the question with clear, factual information.",
		"Answer the question using your knowledge to provide correct information.",
		"Read the question carefully and give an appropriate, accurate answer.",
		"Provide a helpful, correct answer to the question presented.",
	}
}

// getTaskDescription creates a description of the current task.
func (lpg *LLMPromptGenerator) getTaskDescription() string {
	if lpg.signature.Instruction != "" {
		return lpg.signature.Instruction
	}
	return fmt.Sprintf("Process %s to generate %s",
		strings.Join(getFieldNames(lpg.signature.Inputs), ", "),
		strings.Join(getFieldNames(lpg.signature.Outputs), ", "))
}

// calculatePromptDiversity computes semantic diversity between prompt candidates.
func (c *COPRO) calculatePromptDiversity(candidates []PromptCandidate) {
	for i := range candidates {
		diversitySum := 0.0
		comparisons := 0

		for j := range candidates {
			if i != j {
				similarity := c.computeTextSimilarity(candidates[i].Instruction, candidates[j].Instruction)
				diversitySum += (1.0 - similarity)
				comparisons++
			}
		}

		if comparisons > 0 {
			candidates[i].Diversity = diversitySum / float64(comparisons)
		}
	}
}

// computeTextSimilarity computes similarity between two text strings.
func (c *COPRO) computeTextSimilarity(text1, text2 string) float64 {
	// Simple word-based similarity (can be enhanced with embedding similarity)
	words1 := strings.Fields(strings.ToLower(text1))
	words2 := strings.Fields(strings.ToLower(text2))

	wordSet1 := make(map[string]bool)
	wordSet2 := make(map[string]bool)

	for _, word := range words1 {
		wordSet1[word] = true
	}
	for _, word := range words2 {
		wordSet2[word] = true
	}

	intersection := 0
	for word := range wordSet1 {
		if wordSet2[word] {
			intersection++
		}
	}

	union := len(wordSet1) + len(wordSet2) - intersection
	if union == 0 {
		return 1.0
	}

	return float64(intersection) / float64(union)
}

// refineInstruction applies sophisticated refinement strategies.
func (c *COPRO) refineInstruction(instruction string, temperature float64) string {
	refinements := coproRefinements()

	if temperature > 0.4 && rand.Float64() < temperature {
		return applyInstructionRefinement(instruction, refinements[rand.Intn(len(refinements))])
	}

	return instruction
}

func (c *COPRO) forceRefineInstruction(instruction string, seed int) string {
	refinements := coproRefinements()
	return applyInstructionRefinement(instruction, refinements[seed%len(refinements)])
}

func coproRefinements() []string {
	return []string{
		"with methodical analysis",
		"using step-by-step reasoning",
		"with comprehensive evaluation",
		"through careful consideration",
		"with detailed justification",
		"using systematic approach",
		"with thorough examination",
		"through logical analysis",
	}
}

func applyInstructionRefinement(instruction, refinement string) string {
	if strings.Contains(instruction, ".") {
		lastDot := strings.LastIndex(instruction, ".")
		return instruction[:lastDot] + " " + refinement + instruction[lastDot:]
	}
	if strings.Contains(instruction, ",") {
		firstComma := strings.Index(instruction, ",")
		return instruction[:firstComma+1] + " " + refinement + "," + instruction[firstComma+1:]
	}
	return instruction + " " + refinement + "."
}

func (c *COPRO) applyPromptToPredictor(predictor *modules.Predict, candidate PromptCandidate) {
	originalSignature := predictor.GetSignature()
	newSignature := originalSignature.WithInstruction(candidate.Instruction)
	// Update the predictor with the new signature
	// Note: This may require recreating the predictor since signatures are immutable
	newPredictor := modules.NewPredict(newSignature)
	newPredictor.SetLLM(predictor.LLM)
	// Copy the new predictor's state back (this is a limitation of the current API)
	*predictor = *newPredictor
}

func (c *COPRO) truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// getFieldNames extracts field names from InputField or OutputField slices.
func getFieldNames(fields any) []string {
	switch f := fields.(type) {
	case []core.InputField:
		names := make([]string, len(f))
		for i, field := range f {
			names[i] = field.Name
		}
		return names
	case []core.OutputField:
		names := make([]string, len(f))
		for i, field := range f {
			names[i] = field.Name
		}
		return names
	default:
		return []string{}
	}
}

// getTaskDescription creates a comprehensive task description for LLM prompt generation.
func (c *COPRO) getTaskDescription(signature core.Signature, baseInstruction string) string {
	if baseInstruction != "" {
		return baseInstruction
	}
	if signature.Instruction != "" {
		return signature.Instruction
	}
	return fmt.Sprintf("Process %s to generate %s",
		strings.Join(getFieldNames(signature.Inputs), ", "),
		strings.Join(getFieldNames(signature.Outputs), ", "))
}

// getEnhancedInstructionTemplates provides sophisticated fallback instruction templates.
func (c *COPRO) getEnhancedInstructionTemplates(signature core.Signature, baseInstruction string) []string {
	templates := []string{
		"Answer the question with accurate, factual information based on your knowledge.",
		"Provide a clear, direct answer to the specific question being asked.",
		"Think through the question carefully and give a precise, correct response.",
		"Read the question and respond with relevant, accurate information.",
		"Give a straightforward answer using factual knowledge about the topic.",
		"Consider the question and provide a helpful, correct answer.",
		"Respond to the question with clear, accurate information.",
		"Answer the question directly with appropriate factual details.",
		"Provide a correct answer based on the specific question asked.",
		"Give an accurate response that directly addresses the question.",
	}

	// Ensure we have enough templates
	for len(templates) < c.Breadth {
		templates = append(templates, "Answer the question with accurate, factual information.")
	}

	return templates[:c.Breadth]
}
