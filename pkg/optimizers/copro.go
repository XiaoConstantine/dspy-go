package optimizers

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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
}

// PromptCandidate represents a candidate prompt configuration.
type PromptCandidate struct {
	Instruction string
	Prefix      string
	Score       float64
	Generation  int // Which depth iteration this was generated in
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

// NewCOPRO creates a new COPRO optimizer.
func NewCOPRO(metric core.Metric, options ...COPROOption) *COPRO {
	opts := &COPROOptions{
		Breadth:         10,
		Depth:           3,
		InitTemperature: 1.4,
		TrackStats:      false,
	}

	for _, option := range options {
		option(opts)
	}

	return &COPRO{
		PromptModel:     opts.PromptModel,
		Metric:          metric,
		Breadth:         opts.Breadth,
		Depth:           opts.Depth,
		InitTemperature: opts.InitTemperature,
		TrackStats:      opts.TrackStats,
	}
}

// Compile implements the core.Optimizer interface.
func (c *COPRO) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	if metric != nil {
		c.Metric = metric
	}

	if c.Metric == nil {
		return program, fmt.Errorf("COPRO requires a metric function")
	}

	// Ensure execution state exists
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}

	ctx, span := core.StartSpan(ctx, "COPROCompilation")
	defer core.EndSpan(ctx)

	// Clone the program for optimization
	optimizedProgram := program.Clone()

	// Extract all Predict modules that need optimization
	predictors := c.extractPredictors(optimizedProgram)
	if len(predictors) == 0 {
		log.Println("COPRO: No Predict modules found to optimize")
		return optimizedProgram, nil
	}

	log.Printf("COPRO: Found %d Predict modules to optimize", len(predictors))

	// Optimize each predictor's prompts
	for moduleName, predictor := range predictors {
		moduleCtx, moduleSpan := core.StartSpan(ctx, fmt.Sprintf("OptimizePredictor_%s", moduleName))

		err := c.optimizePredictor(moduleCtx, predictor, dataset)
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
func (c *COPRO) optimizePredictor(ctx context.Context, predictor *modules.Predict, dataset core.Dataset) error {
	ctx, span := core.StartSpan(ctx, "OptimizePredictorPrompts")
	defer core.EndSpan(ctx)

	// Convert dataset to examples for evaluation
	examples := c.datasetToExamples(dataset)
	if len(examples) == 0 {
		return fmt.Errorf("no examples in dataset for optimization")
	}

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
		baseline.Score = c.evaluateCandidate(ctx, predictor, baseline, examples)
		candidates = append(candidates, baseline)
		log.Printf("COPRO: Baseline score: %.3f", baseline.Score)
	}

	// Generate initial candidates
	initialCandidates := c.generateInitialCandidates(ctx, predictor, currentInstruction)

	// Evaluate initial candidates in parallel
	c.evaluateCandidatesParallel(ctx, predictor, initialCandidates, examples)

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
		refinedCandidates := c.refineCandidates(depthCtx, predictor, topCandidates, depth)

		// Evaluate refined candidates in parallel
		c.evaluateCandidatesParallel(depthCtx, predictor, refinedCandidates, examples)

		// Add refined candidates to pool
		candidates = append(candidates, refinedCandidates...)

		// Log progress
		bestScore := candidates[0].Score
		if len(refinedCandidates) > 0 {
			sort.Slice(refinedCandidates, func(i, j int) bool {
				return refinedCandidates[i].Score > refinedCandidates[j].Score
			})
			log.Printf("COPRO: Depth %d - Best refined score: %.3f (overall best: %.3f)",
				depth, refinedCandidates[0].Score, bestScore)
		}

		depthSpan.WithAnnotation("refined_candidates", len(refinedCandidates))
		depthSpan.WithAnnotation("best_score", bestScore)
		core.EndSpan(depthCtx)
	}

	// Select the best candidate overall
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	if len(candidates) > 0 {
		bestCandidate := candidates[0]
		log.Printf("COPRO: Selected best candidate - Score: %.3f, Instruction: %s",
			bestCandidate.Score, c.truncateString(bestCandidate.Instruction, 100))

		// Apply the best prompt to the predictor
		c.applyPromptToPredictor(predictor, bestCandidate)

		span.WithAnnotation("best_score", bestCandidate.Score)
		span.WithAnnotation("total_candidates", len(candidates))
	}

	return nil
}

// generateInitialCandidates generates the initial set of prompt candidates.
func (c *COPRO) generateInitialCandidates(ctx context.Context, predictor *modules.Predict, baseInstruction string) []PromptCandidate {
	var candidates []PromptCandidate

	// Get the signature to understand the task
	signature := predictor.GetSignature()

	// Generate diverse instruction variations
	instructionTemplates := c.getInstructionTemplates(signature)

	for i := 0; i < c.Breadth; i++ {
		// Apply temperature-based variation
		temp := c.InitTemperature * (1.0 - float64(i)/float64(c.Breadth))

		var instruction string
		if i < len(instructionTemplates) {
			instruction = instructionTemplates[i]
		} else {
			// Generate variations of base instruction or templates
			instruction = c.varyInstruction(baseInstruction, temp)
		}

		candidate := PromptCandidate{
			Instruction: instruction,
			Prefix:      "", // Could add prefix variations here
			Generation:  1,
		}

		candidates = append(candidates, candidate)
	}

	return candidates
}

// refineCandidates generates refined candidates based on top performers.
func (c *COPRO) refineCandidates(ctx context.Context, predictor *modules.Predict, topCandidates []PromptCandidate, depth int) []PromptCandidate {
	var refined []PromptCandidate

	// Generate refinements for each top candidate
	for _, candidate := range topCandidates {
		// Generate multiple refinements per candidate
		numRefinements := maxInt(1, c.Breadth/len(topCandidates))

		for i := 0; i < numRefinements; i++ {
			// Apply temperature decay with depth
			temp := c.InitTemperature * math.Pow(0.8, float64(depth))

			refinedInstruction := c.refineInstruction(candidate.Instruction, temp)

			refinedCandidate := PromptCandidate{
				Instruction: refinedInstruction,
				Prefix:      candidate.Prefix,
				Generation:  depth + 1,
			}

			refined = append(refined, refinedCandidate)
		}
	}

	return refined
}

// evaluateCandidate evaluates a prompt candidate using the metric.
func (c *COPRO) evaluateCandidate(ctx context.Context, predictor *modules.Predict, candidate PromptCandidate, examples []core.Example) float64 {
	// Temporarily apply the candidate prompt
	originalSignature := predictor.GetSignature()
	tempSignature := originalSignature.WithInstruction(candidate.Instruction)

	// Create a temporary predictor with the candidate prompt
	tempPredictor := modules.NewPredict(tempSignature)
	tempPredictor.SetLLM(predictor.LLM)

	var totalScore float64
	validEvaluations := 0

	// Evaluate on a subset of examples for efficiency - reduced for speed
	maxEval := min(len(examples), 5) // Reduced from 20 to 5 for faster evaluation

	for i := 0; i < maxEval; i++ {
		example := examples[i]

		// Get prediction with candidate prompt
		prediction, err := tempPredictor.Process(ctx, example.Inputs)
		if err != nil {
			log.Printf("COPRO: Error evaluating candidate: %v", err)
			continue
		}

		// Evaluate using metric
		score := c.Metric(example.Outputs, prediction)
		totalScore += score
		validEvaluations++
	}

	if validEvaluations == 0 {
		return 0.0
	}

	return totalScore / float64(validEvaluations)
}

// evaluateCandidatesParallel evaluates multiple candidates in parallel for better performance.
func (c *COPRO) evaluateCandidatesParallel(ctx context.Context, predictor *modules.Predict, candidates []PromptCandidate, examples []core.Example) {
	const maxGoroutines = 3 // Limit concurrent evaluations to avoid API rate limits
	semaphore := make(chan struct{}, maxGoroutines)
	var wg sync.WaitGroup

	for i := range candidates {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			candidates[idx].Score = c.evaluateCandidate(ctx, predictor, candidates[idx], examples)
		}(i)
	}

	wg.Wait()
}

// Helper functions

func (c *COPRO) datasetToExamples(dataset core.Dataset) []core.Example {
	var examples []core.Example
	dataset.Reset()

	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}
		examples = append(examples, example)
	}

	return examples
}

func (c *COPRO) getInstructionTemplates(signature core.Signature) []string {
	// Generate diverse instruction templates based on the signature
	templates := []string{
		"Given the following information, provide a clear and accurate response.",
		"Please analyze the input and generate an appropriate output.",
		"Based on the provided context, answer the question thoroughly.",
		"Process the given input and provide a well-reasoned response.",
		"Review the information provided and give a detailed answer.",
		"Examine the input carefully and provide a precise response.",
		"Consider the given information and generate an appropriate answer.",
		"Analyze the provided data and respond accurately.",
		"Think through the problem step by step and provide your answer.",
		"Based on your understanding, provide a comprehensive response.",
	}

	return templates
}

func (c *COPRO) varyInstruction(baseInstruction string, temperature float64) string {
	if baseInstruction == "" {
		return "Please provide a clear and accurate response to the given input."
	}

	// Simple variation: add prefixes/suffixes based on temperature
	variations := []string{
		"Think carefully and " + strings.ToLower(baseInstruction),
		baseInstruction + " Be specific and thorough.",
		"Step by step, " + strings.ToLower(baseInstruction),
		baseInstruction + " Provide detailed reasoning.",
		"Carefully consider: " + strings.ToLower(baseInstruction),
	}

	// Temperature-based selection
	if temperature > 1.0 {
		idx := rand.Intn(len(variations))
		return variations[idx]
	}

	return baseInstruction
}

func (c *COPRO) refineInstruction(instruction string, temperature float64) string {
	// Simple refinement strategy - could be made more sophisticated
	refinements := []string{
		"more precisely",
		"step by step",
		"thoroughly and carefully",
		"with detailed reasoning",
		"clearly and accurately",
	}

	if temperature > 0.5 && rand.Float64() < temperature {
		refinement := refinements[rand.Intn(len(refinements))]

		// Add refinement to instruction
		if strings.Contains(instruction, ".") {
			return strings.Replace(instruction, ".", " "+refinement+".", 1)
		} else {
			return instruction + " " + refinement
		}
	}

	return instruction
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
