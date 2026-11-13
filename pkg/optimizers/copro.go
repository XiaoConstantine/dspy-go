package optimizers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"regexp"
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
	_ core.ScorerCompiler   = (*COPRO)(nil)
)

// PromptCandidate represents a candidate prompt configuration.
type PromptCandidate struct {
	Instruction     string
	Prefix          string
	Score           float64 // Average score over all examples
	ValidationScore float64 // Unused: candidates are selected on Score
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
	ctx, scorer, err := c.prepareCompile(ctx, core.MetricScorer(metric))
	if err != nil {
		return program, err
	}

	// A program with nothing to optimize is rejected without consuming the
	// dataset.
	if len(c.extractPredictors(program)) == 0 {
		return c.compileExamples(ctx, program, nil, scorer)
	}

	examples, err := core.MaterializeDatasetContext(ctx, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to materialize dataset: %w", err)
	}
	return c.compileExamples(ctx, program, examples, scorer)
}

// CompileExamples optimizes a program from materialized, read-only examples.
func (c *COPRO) CompileExamples(ctx context.Context, program core.Program, examples []core.Example, metric core.Metric) (core.Program, error) {
	return c.CompileWithScorer(ctx, program, examples, core.MetricScorer(metric))
}

// CompileWithScorer optimizes a program from materialized, read-only examples. It
// calls scorer concurrently. An example the program fails on or the scorer
// cannot score counts as zero in a candidate's average.
func (c *COPRO) CompileWithScorer(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	ctx, scorer, err := c.prepareCompile(ctx, scorer)
	if err != nil {
		return program, err
	}
	return c.compileExamples(ctx, program, examples, scorer)
}

// prepareCompile falls back to the constructor metric when scorer is nil.
func (c *COPRO) prepareCompile(ctx context.Context, scorer core.Scorer) (context.Context, core.Scorer, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return ctx, nil, err
	}
	if scorer == nil {
		scorer = core.MetricScorer(c.Metric)
	}
	if scorer == nil {
		return ctx, nil, fmt.Errorf("COPRO requires a metric function")
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}
	return ctx, scorer, nil
}

func (c *COPRO) compileExamples(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	if err := requireRebindableProgram(program); err != nil {
		return program, err
	}
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, "COPROCompilation")
	defer core.EndSpan(ctx)

	// Clone the program for optimization
	optimizedProgram := program.Clone()

	// Extract all Predict modules that need optimization
	predictors := c.extractPredictors(optimizedProgram)
	if len(predictors) == 0 {
		return program, fmt.Errorf("COPRO: program has no Predict module to optimize")
	}
	if program.Forward == nil {
		return program, fmt.Errorf("COPRO: program has no forward function, so its candidates cannot be scored")
	}

	logger.Info(ctx, "COPRO: Found %d Predict modules to optimize", len(predictors))

	predictorNames := make([]string, 0, len(predictors))
	for name := range predictors {
		predictorNames = append(predictorNames, name)
	}
	sort.Strings(predictorNames)

	// Optimize each predictor's prompts in deterministic order. Later
	// predictors are optimized with the earlier winners already installed.
	for _, name := range predictorNames {
		if err := ctx.Err(); err != nil {
			return optimizedProgram, err
		}
		moduleCtx, moduleSpan := core.StartSpan(ctx, fmt.Sprintf("OptimizePredictor_%s", name))

		err := c.optimizePredictor(moduleCtx, optimizedProgram, name, examples, scorer)
		if err != nil {
			moduleSpan.WithError(err)
			core.EndSpan(moduleCtx)
			return optimizedProgram, fmt.Errorf("error optimizing predictor %s: %w", name, err)
		}

		core.EndSpan(moduleCtx)
	}

	span.WithAnnotation("optimized_predictors", len(predictors))
	return optimizedProgram, nil
}

// extractPredictors finds all Predict modules in the program, including those
// nested in composites such as ChainOfThought. A module's first predictor is
// keyed by the module name and any further ones by "name[i]".
func (c *COPRO) extractPredictors(program core.Program) map[string]*modules.Predict {
	predictors := make(map[string]*modules.Predict)

	for moduleName, module := range program.Modules {
		for i, predictor := range demoPredictors(module) {
			name := moduleName
			if i > 0 {
				name = fmt.Sprintf("%s[%d]", moduleName, i)
			}
			predictors[name] = predictor
		}
	}

	return predictors
}

// optimizePredictor optimizes the instruction of the predictor called name
// (see extractPredictors) and installs the winner in program. Candidates are
// scored by running the whole program on all examples.
func (c *COPRO) optimizePredictor(ctx context.Context, program core.Program, name string, examples []core.Example, scorer core.Scorer) error {
	ctx, span := core.StartSpan(ctx, "OptimizePredictorPrompts")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()

	if err := ctx.Err(); err != nil {
		return err
	}
	if len(examples) == 0 {
		return fmt.Errorf("no examples in dataset for optimization")
	}
	predictor, ok := c.extractPredictors(program)[name]
	if !ok {
		return fmt.Errorf("predictor %s not found in program", name)
	}

	promptState := c.newPromptState(ctx, predictor)
	currentInstruction := predictor.GetSignature().Instruction

	// The current instruction is always the first candidate, so it wins ties
	// and optimization never returns a lower-scoring program.
	baseline := PromptCandidate{Instruction: currentInstruction}
	var err error
	baseline.Score, err = c.evaluateCandidate(ctx, program, name, baseline, examples, scorer)
	if err != nil {
		return err
	}
	logger.Info(ctx, "COPRO: Baseline score: %.3f", baseline.Score)

	// candidates holds every evaluated candidate once, in evaluation order.
	candidates := []PromptCandidate{baseline}
	seen := map[string]bool{currentInstruction: true}
	evaluate := func(ctx context.Context, proposals []PromptCandidate) error {
		var fresh []PromptCandidate
		for _, proposal := range proposals {
			if !seen[proposal.Instruction] {
				seen[proposal.Instruction] = true
				fresh = append(fresh, proposal)
			}
		}
		if err := c.evaluateCandidatesParallel(ctx, program, name, fresh, examples, scorer); err != nil {
			return err
		}
		candidates = append(candidates, fresh...)
		return nil
	}

	// Generate and evaluate initial candidates
	initialCandidates := c.generateInitialCandidates(ctx, predictor, currentInstruction, promptState)
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := evaluate(ctx, initialCandidates); err != nil {
		return err
	}

	// Iterative refinement across depth levels
	for depth := 1; depth <= c.Depth; depth++ {
		depthCtx, depthSpan := core.StartSpan(ctx, fmt.Sprintf("Depth_%d", depth))

		// Propose new candidates from everything evaluated so far
		refinedCandidates := c.refineCandidates(depthCtx, predictor, candidates, depth, promptState)
		if err := depthCtx.Err(); err != nil {
			core.EndSpan(depthCtx)
			return err
		}
		if err := evaluate(depthCtx, refinedCandidates); err != nil {
			core.EndSpan(depthCtx)
			return err
		}

		bestScore := bestPromptCandidate(candidates).Score
		logger.Info(depthCtx, "COPRO: Depth %d - %d candidates evaluated, best score: %.3f", depth, len(candidates), bestScore)
		depthSpan.WithAnnotation("refined_candidates", len(refinedCandidates))
		depthSpan.WithAnnotation("best_score", bestScore)
		core.EndSpan(depthCtx)
	}

	bestCandidate := bestPromptCandidate(candidates)
	logger.Info(ctx, "COPRO: Selected best candidate - Score: %.3f, Instruction: %s",
		bestCandidate.Score, c.truncateString(bestCandidate.Instruction, 100))
	c.applyPromptToPredictor(predictor, bestCandidate)

	span.WithAnnotation("best_score", bestCandidate.Score)
	span.WithAnnotation("total_candidates", len(candidates))
	return nil
}

// bestPromptCandidate returns the highest-scoring candidate; the earliest one
// wins a tie. candidates must not be empty.
func bestPromptCandidate(candidates []PromptCandidate) PromptCandidate {
	best := candidates[0]
	for _, candidate := range candidates[1:] {
		if candidate.Score > best.Score {
			best = candidate
		}
	}
	return best
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

// refineCandidates proposes new candidates from every candidate evaluated so
// far. It asks the prompt model at every depth and falls back to fixed
// refinements of the best candidates only when the model proposes nothing.
func (c *COPRO) refineCandidates(ctx context.Context, predictor *modules.Predict, evaluated []PromptCandidate, depth int, state *coproPromptState) []PromptCandidate {
	if len(evaluated) == 0 {
		return nil
	}

	// The history lists each instruction once: seeded attempts, then evaluated ones.
	var history []PromptCandidate
	seen := make(map[string]bool)
	for _, candidate := range append(append([]PromptCandidate(nil), state.history...), evaluated...) {
		if !seen[candidate.Instruction] {
			seen[candidate.Instruction] = true
			history = append(history, candidate)
		}
	}

	if state.generator != nil {
		// Use LLM to generate refined instructions based on performance history
		refinedInstructions, err := state.generator.generateRefinedInstructions(ctx, history, c.Breadth, c.InitTemperature*math.Pow(0.8, float64(depth)))
		if err != nil {
			logging.GetLogger().Error(ctx, "COPRO: Failed to generate refined instructions: %v", err)
		}
		var refined []PromptCandidate
		for i, instruction := range refinedInstructions {
			if len(refined) >= c.Breadth {
				break
			}
			if seen[instruction] {
				continue
			}
			seen[instruction] = true
			refined = append(refined, PromptCandidate{
				Instruction: instruction,
				Prefix:      "",
				Generation:  depth + 1,
				AttemptID:   fmt.Sprintf("refined_%d_%d", depth, i),
			})
		}
		// Proposals that only repeat evaluated instructions add no candidates;
		// the fixed refinements below are for a model that proposes nothing.
		if len(refinedInstructions) > 0 {
			c.calculatePromptDiversity(refined)
			return refined
		}
	}

	// Fall back to fixed refinements of the best candidates.
	topCandidates := append([]PromptCandidate(nil), evaluated...)
	sort.SliceStable(topCandidates, func(i, j int) bool {
		return topCandidates[i].Score > topCandidates[j].Score
	})
	topCandidates = topCandidates[:max(1, min(c.Breadth/2, len(topCandidates)))]

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

// evaluateCandidate scores program with candidate's instruction installed on
// the predictor called name. It runs a clone of the whole program, so the
// other modules, the forward logic, and the predictor's demos and options all
// take part. The score is the average over all examples; an example whose
// prediction fails or cannot be scored counts as zero.
func (c *COPRO) evaluateCandidate(ctx context.Context, program core.Program, name string, candidate PromptCandidate, examples []core.Example, scorer core.Scorer) (float64, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if len(examples) == 0 {
		return 0, nil
	}

	candidateProgram := program.Clone()
	predictor, ok := c.extractPredictors(candidateProgram)[name]
	if !ok {
		return 0, fmt.Errorf("predictor %s not found in program", name)
	}
	c.applyPromptToPredictor(predictor, candidate)

	// Evaluate all examples in parallel for better performance
	scores := make([]float64, len(examples))
	var wg sync.WaitGroup
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
			prediction, err := candidateProgram.Execute(ctx, ex.Inputs)
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					recordError(ctxErr)
					return
				}
				// A failed prediction counts as zero, so it cannot raise the average.
				logging.GetLogger().Error(ctx, "COPRO: Error evaluating candidate: %v", err)
				return
			}
			if ctxErr := ctx.Err(); ctxErr != nil {
				recordError(ctxErr)
				return
			}

			score, err := scorer.Score(ctx, ex, prediction)
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					recordError(ctxErr)
					return
				}
				// An unscorable example counts as zero, so it cannot raise the average.
				logging.GetLogger().Error(ctx, "COPRO: Error scoring candidate: %v", err)
				score = 0
			}
			scores[idx] = score
		})
	}

	wg.Wait()
	if evaluationErr != nil {
		return 0, evaluationErr
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Average over all examples, including the failed ones.
	var totalScore float64
	for _, score := range scores {
		totalScore += score
	}
	return totalScore / float64(len(examples)), nil
}

// evaluateCandidatesParallel evaluates multiple candidates in parallel for better performance.
func (c *COPRO) evaluateCandidatesParallel(ctx context.Context, program core.Program, name string, candidates []PromptCandidate, examples []core.Example, scorer core.Scorer) error {
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

			score, err := c.evaluateCandidate(ctx, program, name, candidates[i], examples, scorer)
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

	validInstructions := parseInstructionLines(output.Content)

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

	// Sort a copy by score, keeping evaluation order among equals, to list the
	// best performers and, separately, the worst.
	history = append([]PromptCandidate(nil), history...)
	sort.SliceStable(history, func(i, j int) bool {
		return history[i].Score > history[j].Score
	})

	bestCount := min(3, len(history))
	bestInstructions := ""
	worstInstructions := ""

	for _, candidate := range history[:bestCount] {
		bestInstructions += fmt.Sprintf("- %s (score: %.3f)\n", candidate.Instruction, candidate.Score)
	}

	for _, candidate := range history[max(bestCount, len(history)-3):] {
		worstInstructions += fmt.Sprintf("- %s (score: %.3f)\n", candidate.Instruction, candidate.Score)
	}
	if worstInstructions == "" {
		worstInstructions = "- (none yet)\n"
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

	validInstructions := parseInstructionLines(output.Content)
	return validInstructions[:min(len(validInstructions), breadth)], nil
}

var coproListMarker = regexp.MustCompile(`^(?:\d+[.):]|[-*•])\s+`)

// parseInstructionLines reads one instruction per line from a proposal
// response. It strips list numbering, bullets, and surrounding quotes, and
// skips headings, preamble lines, and fragments too short to be instructions.
func parseInstructionLines(response string) []string {
	var instructions []string
	for _, line := range strings.Split(response, "\n") {
		cleaned := coproListMarker.ReplaceAllString(strings.TrimSpace(line), "")
		cleaned = strings.Trim(cleaned, "\"'`“”* ")
		lower := strings.ToLower(cleaned)
		if len(cleaned) <= 10 || strings.HasPrefix(cleaned, "#") || strings.HasSuffix(cleaned, ":") ||
			strings.HasPrefix(lower, "here are ") || strings.HasPrefix(lower, "here is ") {
			continue
		}
		instructions = append(instructions, cleaned)
	}
	return instructions
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

// applyPromptToPredictor changes only the instruction; the predictor keeps
// its demos, options, name, and LLM.
func (c *COPRO) applyPromptToPredictor(predictor *modules.Predict, candidate PromptCandidate) {
	predictor.SetSignature(predictor.GetSignature().WithInstruction(candidate.Instruction))
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
