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

	// LLM-assisted prompt generation components
	PromptGenerator  *LLMPromptGenerator
	CandidateHistory []PromptCandidate // Track previous attempts for learning
}

// PromptCandidate represents a candidate prompt configuration.
type PromptCandidate struct {
	Instruction string
	Prefix      string
	Score       float64
	Generation  int     // Which depth iteration this was generated in
	Diversity   float64 // Semantic diversity score
	Rank        int     // Performance ranking
	AttemptID   string  // Unique identifier for tracking
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

// generateInitialCandidates generates sophisticated initial prompt candidates using LLM assistance.
func (c *COPRO) generateInitialCandidates(ctx context.Context, predictor *modules.Predict, baseInstruction string) []PromptCandidate {
	// Get the signature to understand the task
	signature := predictor.GetSignature()

	// Initialize LLM prompt generator if not already done
	if c.PromptGenerator == nil {
		// Use the predictor's LLM for prompt generation, or default model
		promptLLM := c.PromptModel
		if promptLLM == nil {
			promptLLM = predictor.LLM
		}
		c.PromptGenerator = NewLLMPromptGenerator(promptLLM, signature)
	}

	// Generate sophisticated instructions using LLM assistance
	taskDescription := c.getTaskDescription(signature, baseInstruction)
	instructions, err := c.PromptGenerator.generateBasicInstructions(ctx, taskDescription, c.Breadth, c.InitTemperature)
	if err != nil {
		log.Printf("COPRO: Failed to generate LLM-assisted instructions, falling back to enhanced templates: %v", err)
		// Fallback to enhanced template-based generation
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

// refineCandidates generates sophisticated refined candidates using performance feedback.
func (c *COPRO) refineCandidates(ctx context.Context, predictor *modules.Predict, topCandidates []PromptCandidate, depth int) []PromptCandidate {
	// Add current candidates to history for learning
	c.CandidateHistory = append(c.CandidateHistory, topCandidates...)

	// Try LLM-assisted refinement first
	if c.PromptGenerator != nil && len(c.CandidateHistory) >= 3 {
		// Use LLM to generate refined instructions based on performance history
		refinedInstructions, err := c.PromptGenerator.generateRefinedInstructions(ctx, c.CandidateHistory, c.Breadth, c.InitTemperature*math.Pow(0.8, float64(depth)))
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
		numRefinements := maxInt(1, c.Breadth/len(topCandidates))

		for i := 0; i < numRefinements; i++ {
			temp := c.InitTemperature * math.Pow(0.8, float64(depth))
			refinedInstruction := c.refineInstruction(candidate.Instruction, temp)

			// Ensure we're creating meaningful variations
			if refinedInstruction != candidate.Instruction {
				refinedCandidate := PromptCandidate{
					Instruction: refinedInstruction,
					Prefix:      candidate.Prefix,
					Generation:  depth + 1,
					AttemptID:   fmt.Sprintf("basic_refined_%d_%d", depth, i),
				}
				refined = append(refined, refinedCandidate)
			}
		}
	}

	// Calculate diversity scores
	c.calculatePromptDiversity(refined)
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

	// Evaluate on full training set for proper candidate discrimination (matching Python DSPy)
	maxEval := len(examples) // Use all available examples for accurate evaluation

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

// LLMPromptGenerator handles sophisticated prompt generation using LLM assistance.
type LLMPromptGenerator struct {
	llm                core.LLM
	signature          core.Signature
	candidateCache     map[string]PromptCandidate
	diversityThreshold float64
}

// NewLLMPromptGenerator creates a new LLM-assisted prompt generator.
func NewLLMPromptGenerator(llm core.LLM, signature core.Signature) *LLMPromptGenerator {
	return &LLMPromptGenerator{
		llm:                llm,
		signature:          signature,
		candidateCache:     make(map[string]PromptCandidate),
		diversityThreshold: 0.7,
	}
}

// generateBasicInstructions creates initial high-quality instruction candidates using LLM.
func (lpg *LLMPromptGenerator) generateBasicInstructions(ctx context.Context, taskDescription string, breadth int, temperature float64) ([]string, error) {
	// Create a sophisticated prompt for LLM-assisted instruction generation
	generatorPrompt := fmt.Sprintf(`You are an expert prompt engineer. Your task is to generate %d high-quality, diverse instruction variations for a language model task.

Task Description: %s

Task Fields:
- Input: %s 
- Output: %s

Generate %d distinct, effective instructions that will help a language model perform this task accurately. Each instruction should:
1. Be clear and specific
2. Provide helpful guidance 
3. Use different phrasings and approaches
4. Encourage step-by-step thinking when appropriate
5. Be semantically diverse from the others

Return ONLY the instructions, one per line, without numbering or bullet points.`,
		breadth, taskDescription,
		strings.Join(getFieldNames(lpg.signature.Inputs), ", "),
		strings.Join(getFieldNames(lpg.signature.Outputs), ", "),
		breadth)

	// Use LLM to generate sophisticated instructions
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
		fallbacks := lpg.getFallbackInstructions(taskDescription)
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

	for i := maxInt(0, len(history)-3); i < len(history); i++ {
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
func (lpg *LLMPromptGenerator) getFallbackInstructions(taskDescription string) []string {
	return []string{
		"Analyze the provided information step-by-step and generate a precise, well-reasoned response.",
		"Carefully examine the input data and provide a clear, accurate answer with supporting details.",
		"Think through this problem methodically, considering all relevant factors before responding.",
		"Process the given information thoroughly and deliver a comprehensive, well-structured answer.",
		"Review the provided context carefully and generate a detailed, evidence-based response.",
		"Systematically analyze the input and provide a clear, logical, and well-justified answer.",
		"Consider all aspects of the given information and respond with precision and clarity.",
		"Examine the provided data comprehensively and generate an accurate, well-reasoned output.",
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

// varyInstruction creates sophisticated variations using LLM assistance.
func (c *COPRO) varyInstruction(baseInstruction string, temperature float64) string {
	if baseInstruction == "" {
		return "Please provide a clear and accurate response to the given input."
	}

	// Enhanced variation strategies
	variations := []string{
		"Think step-by-step and " + strings.ToLower(baseInstruction),
		baseInstruction + " Ensure your response is comprehensive and well-reasoned.",
		"Carefully analyze the information, then " + strings.ToLower(baseInstruction),
		baseInstruction + " Provide clear justification for your answer.",
		"Consider all relevant factors and " + strings.ToLower(baseInstruction),
		"Systematically evaluate the input and " + strings.ToLower(baseInstruction),
		baseInstruction + " Support your response with logical reasoning.",
	}

	// Temperature-based selection with enhanced randomness
	if temperature > 0.8 {
		idx := rand.Intn(len(variations))
		return variations[idx]
	}

	return baseInstruction
}

// refineInstruction applies sophisticated refinement strategies.
func (c *COPRO) refineInstruction(instruction string, temperature float64) string {
	// Enhanced refinement strategies with context awareness
	refinements := []string{
		"with methodical analysis",
		"using step-by-step reasoning",
		"with comprehensive evaluation",
		"through careful consideration",
		"with detailed justification",
		"using systematic approach",
		"with thorough examination",
		"through logical analysis",
	}

	// Context-aware refinement placement
	if temperature > 0.4 && rand.Float64() < temperature {
		refinement := refinements[rand.Intn(len(refinements))]

		// Intelligent refinement insertion
		if strings.Contains(instruction, ".") {
			// Insert before final period
			lastDot := strings.LastIndex(instruction, ".")
			return instruction[:lastDot] + " " + refinement + instruction[lastDot:]
		} else if strings.Contains(instruction, ",") {
			// Insert after first comma
			firstComma := strings.Index(instruction, ",")
			return instruction[:firstComma+1] + " " + refinement + "," + instruction[firstComma+1:]
		} else {
			// Append to end
			return instruction + " " + refinement + "."
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

// getFieldNames extracts field names from InputField or OutputField slices.
func getFieldNames(fields interface{}) []string {
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
	taskDesc := c.getTaskDescription(signature, baseInstruction)

	templates := []string{
		fmt.Sprintf("Carefully analyze the provided information and %s with comprehensive reasoning.", strings.ToLower(taskDesc)),
		fmt.Sprintf("Think step-by-step through the given data to %s accurately and precisely.", strings.ToLower(taskDesc)),
		fmt.Sprintf("Systematically evaluate all relevant factors to %s with detailed justification.", strings.ToLower(taskDesc)),
		fmt.Sprintf("Examine the input thoroughly and %s using logical, methodical analysis.", strings.ToLower(taskDesc)),
		fmt.Sprintf("Process the given information comprehensively to %s with clear reasoning.", strings.ToLower(taskDesc)),
	}

	// Ensure we have enough templates
	for len(templates) < c.Breadth {
		templates = append(templates, "Analyze the provided information step-by-step and generate a precise, well-reasoned response.")
	}

	return templates[:c.Breadth]
}
