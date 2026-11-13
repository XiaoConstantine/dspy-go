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
)

// RunMode defines different optimization intensities for MIPRO.
type RunMode string

const (
	LightMode  RunMode = "light"
	MediumMode RunMode = "medium"
	HeavyMode  RunMode = "heavy"
)

// AutoRunSettings defines default configurations for different run modes.
var AutoRunSettings = map[RunMode]struct {
	NumTrials int
	ValSize   int
}{
	LightMode:  {NumTrials: 7, ValSize: 100},
	MediumMode: {NumTrials: 25, ValSize: 300},
	HeavyMode:  {NumTrials: 50, ValSize: 1000},
}

// SearchStrategy defines the interface for optimization search algorithms.
type SearchStrategy interface {
	SuggestParams(ctx context.Context) (map[string]any, error)
	UpdateResults(params map[string]any, score float64) error
	GetBestParams() (map[string]any, float64)
	Initialize(config SearchConfig) error
}

// SearchConfig contains configuration for search strategies.
type SearchConfig struct {
	ParamSpace  map[string][]any
	MaxTrials   int
	Seed        int64
	Constraints map[string]any
}

// TeacherStudentOptimizer handles the teacher-student learning dynamic.
type TeacherStudentOptimizer struct {
	Teacher         core.LLM
	Student         core.LLM
	TeacherSettings map[string]any
	MaxExamples     int

	state struct {
		demonstrations []core.Example
		teacherScores  map[string]float64
	}
}

// Initialize sets up the teacher-student optimization.
func (t *TeacherStudentOptimizer) Initialize(ctx context.Context, program core.Program, dataset core.Dataset) error {
	t.reset()
	return nil
}

func (t *TeacherStudentOptimizer) reset() {
	t.state.demonstrations = make([]core.Example, 0)
	t.state.teacherScores = make(map[string]float64)
}

// GenerateDemonstration creates a high-quality demonstration using the teacher.
func (t *TeacherStudentOptimizer) GenerateDemonstration(ctx context.Context, input core.Example) (core.Example, error) {

	if t.Teacher == nil {
		return core.Example{}, fmt.Errorf("teacher generation failed: no teacher model available, configure a default LLM or use WithModels")
	}
	// A labeled example is its own demonstration. The bare teacher model's
	// answer would be unverified, so it is used only for unlabeled prompts.
	if len(input.Outputs) > 0 {
		return input, nil
	}
	promptValue, ok := input.Inputs["prompt"]
	if !ok || promptValue == nil {
		return core.Example{}, fmt.Errorf("teacher generation failed, missing prompt")
	}
	prompt, ok := promptValue.(string)
	if !ok {
		return core.Example{}, fmt.Errorf("teacher generation failed: prompt must be a string, got %T", promptValue)
	}
	core.RecordLLMCall(ctx, t.Teacher)
	response, err := t.Teacher.Generate(ctx, prompt)
	if err != nil {
		return core.Example{}, fmt.Errorf("teacher generation failed: %w", err)
	}

	return core.Example{
		Inputs:  input.Inputs,
		Outputs: map[string]any{"completion": response.Content},
	}, nil
}

// InstructionGenerator handles the generation of instruction candidates.
type InstructionGenerator struct {
	PromptModel   core.LLM
	MaxCandidates int
	Temperature   float64
}

// GenerateCandidates creates instruction candidates for each predictor.
func (g *InstructionGenerator) GenerateCandidates(
	ctx context.Context,
	program core.Program,
	demos []core.Example,
) (map[int][]string, error) {
	candidates := make(map[int][]string)
	modules := program.GetModules() // Get ordered slice of modules

	for i, module := range modules {
		moduleInstructions := make([]string, g.MaxCandidates)

		// Generate candidates for this module
		for j := 0; j < g.MaxCandidates; j++ {
			instruction, err := g.generateSingleCandidate(ctx, module, demos, j)
			if err != nil {
				return nil, fmt.Errorf("failed to generate candidate %d for module %d: %w", j, i, err)
			}
			// Ensure we never store empty instructions
			if instruction == "" {
				instruction = "Default instruction for module " + fmt.Sprintf("%d", i)
			}
			moduleInstructions[j] = instruction
		}

		candidates[i] = moduleInstructions
	}

	return candidates, nil
}

// instructionTips steer successive proposals apart, so that a model returning
// the same text for the same prompt still yields distinct candidates.
var instructionTips = []string{
	"Keep the instruction short and direct.",
	"State exactly what form the output must take.",
	"Describe how to work through the inputs before answering.",
	"Name the mistakes this task invites and how to avoid them.",
	"Describe the task from a fresh angle.",
}

// generateSingleCandidate asks the prompt model for the index-th alternative
// to the module's instruction, showing it the signature, the current
// instruction, and a few example inputs and outputs.
func (g *InstructionGenerator) generateSingleCandidate(
	ctx context.Context,
	module core.Module,
	demos []core.Example,
	index int,
) (string, error) {
	if g.PromptModel == nil {
		return "", fmt.Errorf("no prompt model available: configure a default LLM or use WithModels")
	}

	signature := module.GetSignature()
	var prompt strings.Builder
	fmt.Fprintf(&prompt, "Generate an instruction for the following signature: %s\n", signature)
	if signature.Instruction != "" {
		fmt.Fprintf(&prompt, "Current instruction: %s\n", signature.Instruction)
	}
	for i, demo := range demos[:min(len(demos), 3)] {
		fmt.Fprintf(&prompt, "Example %d inputs: %v\nExample %d outputs: %v\n", i+1, demo.Inputs, i+1, demo.Outputs)
	}
	fmt.Fprintf(&prompt, "Tip: %s\nWrite a different instruction that would produce better outputs. Reply with the instruction text only.", instructionTips[index%len(instructionTips)])

	core.RecordLLMCall(ctx, g.PromptModel)
	response, err := g.PromptModel.Generate(ctx, prompt.String())
	if err != nil {
		return "", fmt.Errorf("prompt model generation failed: %w", err)
	}

	// Ensure we never return an empty content
	if response == nil || strings.TrimSpace(response.Content) == "" {
		return fmt.Sprintf("Default instruction for %s", signature), nil
	}

	return strings.Trim(strings.TrimSpace(response.Content), "\"`"), nil
}

// MIPROConfig contains all configuration options for the optimizer.
type MIPROConfig struct {
	Mode           RunMode
	NumTrials      int
	ValSize        int
	MiniBatchSize  int
	AdaptiveParams bool
	ScalingFactors struct {
		TrialsPerVariable float64
		BatchSizeScaling  float64
	}
	TeacherSettings map[string]any

	// TPE specific configuration
	TPEGamma        float64
	TPEGenerations  int
	Seed            int64
	NumModules      int // Number of modules to optimize (can be inferred from program)
	MaxLabeledDemos int // Maximum number of labeled demonstrations to use
}

// OptimizationState tracks the progress of optimization.
type OptimizationState struct {
	SuccessfulPatterns []string
	PromptEvolution    []PromptVersion
	TeacherScores      map[string]float64
	CurrentIteration   int
	BestScore          float64
	Convergence        float64
}

// PromptVersion represents a specific version of a prompt template.
type PromptVersion struct {
	Template    string
	Performance float64
	Components  []PromptComponent
}

// PromptComponent represents a specific part of a prompt.
type PromptComponent struct {
	Type    string
	Content string
	Score   float64
}

// MIPROMetrics tracks comprehensive optimization metrics.
type MIPROMetrics struct {
	TeacherPerformance  float64
	StudentPerformance  float64
	PromptEffectiveness map[string]float64
	OptimizationHistory []OptimizationStep
	TokenUsage          *core.TokenInfo
}

// OptimizationStep represents a single step in the optimization process.
type OptimizationStep struct {
	Trial         int
	Performance   float64
	Improvements  []string
	FailurePoints []string
}

type miproMetric = func(example, prediction map[string]any, ctx context.Context) float64

// MIPRO is the main optimizer implementing multi-step interactive prompt optimization.
type MIPRO struct {
	// Core components
	metric               miproMetric
	searchStrategy       SearchStrategy
	searchConfig         SearchConfig
	teacherStudent       *TeacherStudentOptimizer
	instructionGenerator *InstructionGenerator

	// Configuration
	config  MIPROConfig
	state   *OptimizationState
	metrics *MIPROMetrics

	// Models
	promptModel core.LLM
	taskModel   core.LLM

	// Optimization parameters
	maxBootstrappedDemos int
	numCandidates        int

	// rng drives the demonstration split, demonstration sets and minibatches.
	rng *rand.Rand

	// Tracking and logging
	logger *logging.Logger
	mu     sync.RWMutex
}

var (
	_ core.Optimizer        = (*MIPRO)(nil)
	_ core.ExamplesCompiler = (*MIPRO)(nil)
	_ core.ScorerCompiler   = (*MIPRO)(nil)
)

// MIPROOption defines a function type for configuring MIPRO.
type MIPROOption func(*MIPRO)

// WithMode sets the optimization mode.
func WithMode(mode RunMode) MIPROOption {
	return func(m *MIPRO) {
		m.config.Mode = mode
	}
}

// WithNumTrials sets the number of optimization trials.
func WithNumTrials(trials int) MIPROOption {
	return func(m *MIPRO) {
		m.config.NumTrials = trials
	}
}

// WithTeacherSettings configures the teacher model settings.
func WithTeacherSettings(settings map[string]any) MIPROOption {
	return func(m *MIPRO) {
		m.config.TeacherSettings = settings
	}
}

// WithSearchStrategy sets a custom search strategy.
func WithSearchStrategy(strategy SearchStrategy) MIPROOption {
	return func(m *MIPRO) {
		m.searchStrategy = strategy
	}
}

// WithTPEGamma sets the gamma parameter for the TPE optimizer.
func WithTPEGamma(gamma float64) MIPROOption {
	return func(m *MIPRO) {
		// Store in the config for later use when initializing TPE
		m.config.TPEGamma = gamma
	}
}

// WithTPEGenerations sets the number of candidates to generate for each TPE optimization step.
func WithTPEGenerations(generations int) MIPROOption {
	return func(m *MIPRO) {
		m.config.TPEGenerations = generations
	}
}

// WithRandomSeed sets a specific random seed for reproducibility.
func WithRandomSeed(seed int64) MIPROOption {
	return func(m *MIPRO) {
		m.config.Seed = seed
	}
}

func WithMiniBatchSize(size int) MIPROOption {
	return func(m *MIPRO) {
		m.config.MiniBatchSize = size
	}
}

// WithNumModules explicitly sets the number of modules to optimize.
func WithNumModules(numModules int) MIPROOption {
	return func(m *MIPRO) {
		m.config.NumModules = numModules
	}
}

func WithNumCandidates(num int) MIPROOption {
	return func(m *MIPRO) {
		m.numCandidates = num
	}
}

// WithMaxLabeledDemos sets the maximum number of labeled demos to use.
func WithMaxLabeledDemos(maxDemos int) MIPROOption {
	return func(m *MIPRO) {
		m.config.MaxLabeledDemos = maxDemos
	}
}

// WithModels explicitly sets the prompt and task models for MIPRO.
func WithModels(promptModel, taskModel core.LLM) MIPROOption {
	return func(m *MIPRO) {
		m.promptModel = promptModel
		m.taskModel = taskModel
	}
}

// NewMIPRO creates a new MIPRO optimizer instance.
//
// metric is the fallback used when a compile method receives no metric or
// scorer; it may be nil. New code should pass nil and give CompileWithScorer a
// core.Scorer, which takes ctx first and can report errors.
func NewMIPRO(
	metric func(example, prediction map[string]any, ctx context.Context) float64,
	opts ...MIPROOption,
) *MIPRO {
	m := &MIPRO{
		metric: metric,
		config: MIPROConfig{
			Mode: MediumMode,
			ScalingFactors: struct {
				TrialsPerVariable float64
				BatchSizeScaling  float64
			}{
				TrialsPerVariable: 1.5,
				BatchSizeScaling:  0.8,
			},
			TPEGamma:       0.25, // Set default TPE gamma
			TPEGenerations: 20,   // Set default TPE generations
		},
		state: &OptimizationState{
			TeacherScores: make(map[string]float64),
			BestScore:     math.Inf(-1),
		},
		metrics: &MIPROMetrics{
			PromptEffectiveness: make(map[string]float64),
		},
		logger:               logging.GetLogger(),
		numCandidates:        3, // Default to 3 candidates
		maxBootstrappedDemos: 3, // Default to 3 bootstrapped demos
	}

	// Apply options
	for _, opt := range opts {
		opt(m)
	}
	if m.config.NumModules <= 0 {
		m.config.NumModules = 2 // Default to 2 modules (most common case)
	}

	if m.numCandidates <= 0 {
		m.numCandidates = 3
	}
	if m.config.NumTrials <= 0 {
		// Without WithNumTrials the run mode decides how long to search.
		settings, ok := AutoRunSettings[m.config.Mode]
		if !ok {
			settings = AutoRunSettings[MediumMode]
		}
		m.config.NumTrials = settings.NumTrials
	}
	m.rng = rand.New(rand.NewSource(m.config.Seed))
	// Initialize components
	m.initComponents()

	return m
}

// initComponents initializes all required components.
func (m *MIPRO) initComponents() {
	if m.promptModel == nil {
		// Try to get the default LLM. Even when none is available we keep
		// initializing the remaining components so the MIPRO struct is
		// never left half-built; model-dependent calls fail with clear
		// errors instead of nil-pointer panics.
		if defaultLLM := core.GetDefaultLLM(); defaultLLM != nil {
			m.promptModel = defaultLLM
		} else {
			m.logger.Error(context.Background(), "Failed to initialize prompt model")
		}
	}

	if m.taskModel == nil {
		m.taskModel = m.promptModel // Use prompt model as task model if not specified
	}
	m.teacherStudent = &TeacherStudentOptimizer{
		Teacher:         m.promptModel,
		Student:         m.taskModel,
		TeacherSettings: m.config.TeacherSettings,
		MaxExamples:     m.maxBootstrappedDemos,
	}

	m.instructionGenerator = &InstructionGenerator{
		PromptModel:   m.promptModel,
		MaxCandidates: m.numCandidates,
		Temperature:   0.7,
	}
	if m.searchStrategy == nil {
		m.searchStrategy = NewTPEOptimizer(TPEConfig{
			Gamma:            m.config.TPEGamma,
			Seed:             m.config.Seed,
			NumEIGenerations: m.config.TPEGenerations,
			PriorWeight:      1.0,
			BandwidthFactor:  1.0,
		})
	}
	m.searchConfig = SearchConfig{
		ParamSpace: m.paramSpace(m.config.NumModules),
		MaxTrials:  m.config.NumTrials,
		Seed:       m.config.Seed,
	}
}

// demoSetCount is the number of demonstration choices per module. Choice 0
// keeps the module's demonstrations as they are.
const demoSetCount = 5

// paramSpace lists the choices searched for each module. Instruction choice 0
// is the module's current instruction, followed by the generated candidates.
func (m *MIPRO) paramSpace(numModules int) map[string][]any {
	choices := func(n int) []any {
		values := make([]any, n)
		for i := range values {
			values[i] = float64(i)
		}
		return values
	}
	space := make(map[string][]any)
	for i := range max(numModules, 1) {
		space[fmt.Sprintf("module_%d_instruction", i)] = choices(m.numCandidates + 1)
		if m.demosPerSet() > 0 {
			space[fmt.Sprintf("module_%d_demos", i)] = choices(demoSetCount)
		}
	}
	return space
}

// demosPerSet is the number of demonstrations in each demonstration set.
func (m *MIPRO) demosPerSet() int {
	return max(m.maxBootstrappedDemos, m.config.MaxLabeledDemos)
}

func (m *MIPRO) resetCompileState() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = &OptimizationState{
		TeacherScores: make(map[string]float64),
		BestScore:     math.Inf(-1),
	}
	m.metrics = &MIPROMetrics{
		PromptEffectiveness: make(map[string]float64),
	}
	m.rng = rand.New(rand.NewSource(m.config.Seed))
}

// compileSearchConfig sizes the search space to the program being compiled.
func (m *MIPRO) compileSearchConfig(program core.Program) SearchConfig {
	numModules := len(program.Modules)
	if numModules == 0 {
		numModules = m.config.NumModules
	}
	config := SearchConfig{
		ParamSpace:  m.paramSpace(numModules),
		MaxTrials:   m.config.NumTrials,
		Seed:        m.config.Seed,
		Constraints: make(map[string]any, len(m.searchConfig.Constraints)),
	}
	for name, value := range m.searchConfig.Constraints {
		config.Constraints[name] = value
	}
	return config
}

// Compile implements the legacy cursor-based Optimizer API. It consumes the
// dataset once, then optimizes from materialized examples so later phases never
// share or reset the caller's mutable cursor.
func (m *MIPRO) Compile(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
	metric core.Metric,
) (core.Program, error) {
	scorer, err := m.prepareCompile(ctx, program, serializeMetric(metric))
	if err != nil {
		return program, err
	}
	examples, err := core.MaterializeDatasetContext(ctx, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to materialize dataset: %w", err)
	}
	return m.compileExamples(ctx, program, examples, scorer)
}

// CompileExamples optimizes a program from materialized, read-only examples.
func (m *MIPRO) CompileExamples(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	metric core.Metric,
) (core.Program, error) {
	scorer, err := m.prepareCompile(ctx, program, serializeMetric(metric))
	if err != nil {
		return program, err
	}
	return m.compileExamples(ctx, program, examples, scorer)
}

// CompileWithScorer optimizes a program from materialized, read-only examples. It
// calls scorer from up to core.GetConcurrencyLevel goroutines at once. An
// example the scorer cannot score counts as zero in a minibatch and fails a
// full evaluation.
func (m *MIPRO) CompileWithScorer(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	scorer core.Scorer,
) (core.Program, error) {
	scorer, err := m.prepareCompile(ctx, program, scorer)
	if err != nil {
		return program, err
	}
	return m.compileExamples(ctx, program, examples, scorer)
}

// miproMetricScorer adapts a constructor metric, one call at a time.
func miproMetricScorer(metric miproMetric) core.Scorer {
	return serialize(core.ScorerFunc(func(ctx context.Context, example core.Example, prediction map[string]any) (float64, error) {
		return metric(example.Outputs, prediction, ctx), nil
	}))
}

// prepareCompile falls back to the constructor metric when scorer is nil.
func (m *MIPRO) prepareCompile(ctx context.Context, program core.Program, scorer core.Scorer) (core.Scorer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if scorer == nil && m.metric != nil {
		scorer = miproMetricScorer(m.metric)
	}
	if scorer == nil {
		return nil, fmt.Errorf("MIPRO requires a metric function")
	}

	m.resetCompileState()
	if err := m.searchStrategy.Initialize(m.compileSearchConfig(program)); err != nil {
		return nil, fmt.Errorf("failed to initialize search strategy: %w", err)
	}
	m.logger.Info(ctx, "Starting MIPRO optimization with configuration: %+v", m.config)
	m.teacherStudent.reset()
	return scorer, nil
}

func (m *MIPRO) compileExamples(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	scorer core.Scorer,
) (core.Program, error) {
	if err := requireRebindableProgram(program); err != nil {
		return program, err
	}

	// Step 1: Set aside examples for demonstrations, so that no candidate is
	// scored on an example it was shown.
	demoPool, examples := m.splitExamples(examples)
	demos, err := m.generateDemonstrations(ctx, demoPool)
	if err != nil {
		return program, fmt.Errorf("failed to generate demonstrations: %w", err)
	}

	// Step 2: Generate instruction candidates. Choice 0 is each module's
	// current instruction, so the search can keep it.
	instructions, err := m.instructionGenerator.GenerateCandidates(ctx, program, demos)
	if err != nil {
		return program, fmt.Errorf("failed to generate instructions: %w", err)
	}
	for i, module := range program.GetModules() {
		instructions[i] = append([]string{module.GetSignature().Instruction}, instructions[i]...)
	}

	// Step 3: Search instruction and demonstration choices.
	bestProgram, err := m.runOptimizationLoop(ctx, program, examples, demos, instructions, scorer)
	if err != nil {
		return program, fmt.Errorf("optimization failed: %w", err)
	}
	return bestProgram, nil
}

// splitExamples sets aside about a fifth of the examples, and at least one
// demonstration set's worth, as the demonstration pool. The rest are used for
// scoring. Nothing is set aside when demonstrations are disabled or there are
// too few examples to spare.
func (m *MIPRO) splitExamples(examples []core.Example) (demoPool, scored []core.Example) {
	perSet := m.demosPerSet()
	poolSize := min(max(perSet, len(examples)/5), len(examples)/2)
	if perSet == 0 || poolSize == 0 {
		return nil, examples
	}
	inPool := make(map[int]bool, poolSize)
	for _, i := range m.rng.Perm(len(examples))[:poolSize] {
		inPool[i] = true
	}
	for i, example := range examples {
		if inPool[i] {
			demoPool = append(demoPool, example)
		} else {
			scored = append(scored, example)
		}
	}
	return demoPool, scored
}

// demoSets draws the demonstration choices from demos. Set 0 is empty and
// means "keep the module's demonstrations".
func (m *MIPRO) demoSets(demos []core.Example) [][]core.Example {
	sets := make([][]core.Example, demoSetCount)
	perSet := min(m.demosPerSet(), len(demos))
	for i := 1; i < demoSetCount && perSet > 0; i++ {
		for _, j := range m.rng.Perm(len(demos))[:perSet] {
			sets[i] = append(sets[i], demos[j])
		}
	}
	return sets
}

// runOptimizationLoop scores the unchanged program, then NumTrials suggested
// combinations of instruction and demonstrations, and returns the program
// with the highest score over all examples. Ties keep the earlier program, so
// the result never scores below the original on these examples.
func (m *MIPRO) runOptimizationLoop(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	demos []core.Example,
	instructions map[int][]string,
	scorer core.Scorer,
) (core.Program, error) {
	bestProgram := program
	bestScore, err := m.evaluateCandidate(ctx, program, examples, scorer)
	if err != nil {
		return program, fmt.Errorf("failed to evaluate the original program: %w", err)
	}
	m.mu.Lock()
	m.state.BestScore = math.Max(m.state.BestScore, bestScore)
	m.mu.Unlock()
	m.logger.Info(ctx, "Original program score: %f", bestScore)

	demoSets := m.demoSets(demos)
	// A minibatch score is only an estimate, so it guides the search and the
	// best estimates are scored on all examples afterwards.
	useMinibatch := m.config.MiniBatchSize > 0 && m.config.MiniBatchSize < len(examples)
	type trialStats struct {
		params map[string]any
		total  float64
		count  int
	}
	stats := make(map[string]*trialStats)
	var order []string
	fullScores := make(map[string]float64)

	for iteration := 0; iteration < m.config.NumTrials; iteration++ {
		// Get next parameter combination to try
		params, err := m.searchStrategy.SuggestParams(ctx)
		if err != nil {
			return program, fmt.Errorf("failed to get parameters: %w", err)
		}
		m.logger.Info(ctx, "Trial %d/%d with parameters: %v",
			iteration+1, m.config.NumTrials, params)

		key := fmt.Sprint(params)
		candidate := m.createCandidateProgram(program, params, demoSets, instructions)
		score, repeated := fullScores[key]
		switch {
		case useMinibatch:
			batch := make([]core.Example, 0, m.config.MiniBatchSize)
			for _, i := range m.rng.Perm(len(examples))[:m.config.MiniBatchSize] {
				batch = append(batch, examples[i])
			}
			score, err = m.evaluateOnMinibatch(ctx, candidate, batch, len(batch), scorer)
		case !repeated:
			// A combination suggested again keeps its earlier score.
			score, err = m.evaluateCandidate(ctx, candidate, examples, scorer)
			fullScores[key] = score
		}
		if err != nil {
			return program, fmt.Errorf("failed to evaluate candidate: %w", err)
		}
		// Report results back to the TPE optimizer
		if err := m.searchStrategy.UpdateResults(params, score); err != nil {
			return program, fmt.Errorf("failed to update search results: %w", err)
		}
		m.updateOptimizationState(params, score, candidate)

		if stats[key] == nil {
			stats[key] = &trialStats{params: params}
			order = append(order, key)
		}
		stats[key].total += score
		stats[key].count++
		if !useMinibatch && score > bestScore {
			bestScore = score
			bestProgram = candidate
			m.logger.Info(ctx, "New best score: %f", bestScore)
		}
	}

	if useMinibatch {
		const finalists = 3
		mean := func(key string) float64 { return stats[key].total / float64(stats[key].count) }
		sort.SliceStable(order, func(i, j int) bool { return mean(order[i]) > mean(order[j]) })
		for _, key := range order[:min(finalists, len(order))] {
			candidate := m.createCandidateProgram(program, stats[key].params, demoSets, instructions)
			score, err := m.evaluateCandidate(ctx, candidate, examples, scorer)
			if err != nil {
				return program, fmt.Errorf("failed to evaluate candidate: %w", err)
			}
			if score > bestScore {
				bestScore = score
				bestProgram = candidate
				m.logger.Info(ctx, "New best score: %f", bestScore)
			}
		}
	}
	m.logger.Info(ctx, "Best program score: %f", bestScore)
	return bestProgram, nil
}

// Helper functions for the teacher-student dynamic.
func (m *MIPRO) teacherDemonstration(ctx context.Context, example core.Example) (core.Example, error) {
	if m.teacherStudent.Teacher == nil {
		return core.Example{}, fmt.Errorf("teacher demonstration failed: no teacher model available, configure a default LLM or use WithModels")
	}
	prompt, ok := example.Inputs["prompt"].(string)
	if !ok {
		return core.Example{}, fmt.Errorf("teacher demonstration failed: prompt must be a string, got %T", example.Inputs["prompt"])
	}

	// Get a high-quality demonstration from the teacher model
	core.RecordLLMCall(ctx, m.teacherStudent.Teacher)
	teacherResult, err := m.teacherStudent.Teacher.Generate(ctx, prompt)
	if err != nil {
		return core.Example{}, fmt.Errorf("teacher demonstration failed: %w", err)
	}

	// Create a new example with the teacher's output
	return core.Example{
		Inputs:  example.Inputs,
		Outputs: map[string]any{"completion": teacherResult.Content},
	}, nil
}

// Methods for tracking and analyzing performance.
func (m *MIPRO) updateOptimizationState(params map[string]any, score float64, program core.Program) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update state
	m.state.CurrentIteration++
	if score > m.state.BestScore {
		m.state.BestScore = score
	}

	// Track successful patterns
	if score > m.state.BestScore*0.9 {
		m.extractAndStorePatterns(program)
	}
}

func (m *MIPRO) extractAndStorePatterns(program core.Program) {
	// Analyze program structure and extract successful patterns
	for _, module := range program.GetModules() {
		signature := module.GetSignature()
		if signature.Instruction != "" {
			m.state.SuccessfulPatterns = append(m.state.SuccessfulPatterns, signature.Instruction)
		}
	}
}

// Utility functions for tracking metrics and logging.

func (m *MIPRO) updateMetrics(score float64, tokenUsage *core.TokenInfo) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.metrics.OptimizationHistory = append(m.metrics.OptimizationHistory, OptimizationStep{
		Trial:       m.state.CurrentIteration,
		Performance: score,
	})

	if tokenUsage != nil {
		m.metrics.TokenUsage = tokenUsage
	}
}

// generateDemonstrations turns the demonstration pool into demonstrations.
func (m *MIPRO) generateDemonstrations(
	ctx context.Context,
	examples []core.Example,
) ([]core.Example, error) {
	demos := make([]core.Example, 0, len(examples))
	for _, example := range examples {
		demo, err := m.teacherStudent.GenerateDemonstration(ctx, example)
		if err != nil {
			return nil, fmt.Errorf("failed to generate demonstration: %w", err)
		}
		demos = append(demos, demo)
	}

	return demos, nil
}

// createCandidateProgram returns a copy of baseProgram with the instruction
// and demonstration set that params selects for each module. A missing or
// out-of-range choice keeps what the module has.
func (m *MIPRO) createCandidateProgram(
	baseProgram core.Program,
	params map[string]any,
	demoSets [][]core.Example,
	instructions map[int][]string,
) core.Program {
	choice := func(key string) int {
		index, _ := params[key].(float64)
		return int(index)
	}
	newProgram := baseProgram.Clone()
	for i, module := range newProgram.GetModules() {
		candidates := instructions[i]
		if idx := choice(fmt.Sprintf("module_%d_instruction", i)); idx >= 0 && idx < len(candidates) && candidates[idx] != "" {
			module.SetSignature(module.GetSignature().WithInstruction(candidates[idx]))
		}
		if idx := choice(fmt.Sprintf("module_%d_demos", i)); idx > 0 && idx < len(demoSets) && len(demoSets[idx]) > 0 {
			for _, predictor := range demoPredictors(module) {
				if demos := demosForPredictor(predictor.GetSignature(), demoSets[idx]); len(demos) > 0 {
					predictor.SetDemos(demos)
				}
			}
		}
	}
	return newProgram
}

// demosForPredictor builds the demonstrations a predictor with this signature
// can render from labeled program examples: one for each example that carries
// all of its input fields and at least one of its output fields, holding only
// those fields. A predictor in the middle of a program reads its inputs from the
// example's outputs, and any other field would render as "<nil>" in its prompt.
func demosForPredictor(signature core.Signature, examples []core.Example) []core.Example {
	value := func(example core.Example, name string) (any, bool) {
		if v, ok := example.Inputs[name]; ok {
			return v, true
		}
		v, ok := example.Outputs[name]
		return v, ok
	}
	var demos []core.Example
	for _, example := range examples {
		demo := core.Example{Inputs: map[string]any{}, Outputs: map[string]any{}}
		usable := true
		for _, field := range signature.Inputs {
			v, ok := value(example, field.Name)
			usable = usable && ok
			demo.Inputs[field.Name] = v
		}
		for _, field := range signature.Outputs {
			if v, ok := value(example, field.Name); ok {
				demo.Outputs[field.Name] = v
			}
		}
		if usable && len(demo.Outputs) > 0 {
			demos = append(demos, demo)
		}
	}
	return demos
}

// evaluateCandidate evaluates a candidate program.
func (m *MIPRO) evaluateCandidate(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	scorer core.Scorer,
) (float64, error) {
	if len(examples) == 0 {
		return 0, fmt.Errorf("no examples evaluated")
	}

	var totalScore float64
	for _, result := range m.scoreExamples(ctx, program, examples, scorer, true) {
		if result.Err != nil {
			return 0, fmt.Errorf("failed to evaluate example: %w", result.Err)
		}
		totalScore += result.Score
	}
	// Examples skipped after cancellation carry no error of their own.
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	return totalScore / float64(len(examples)), nil
}

func (m *MIPRO) evaluateOnMinibatch(
	ctx context.Context,
	program core.Program,
	examples []core.Example,
	batchSize int,
	scorer core.Scorer,
) (float64, error) {
	if batchSize <= 0 || len(examples) == 0 {
		return 0, fmt.Errorf("no examples in minibatch")
	}
	batchSize = min(batchSize, len(examples))
	batch := examples[:batchSize]

	// Evaluate on the minibatch
	results := m.scoreExamples(ctx, program, batch, scorer, false)
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	var totalScore float64
	for _, result := range results {
		if result.Err != nil {
			m.logger.Warn(ctx, "Failed to evaluate example: %v", result.Err)
			continue
		}
		totalScore += result.Score
	}

	return totalScore / float64(len(batch)), nil
}

// scoreExamples runs program and scorer on each example, up to the package
// concurrency level (core.SetConcurrencyOptions, default 1) at a time.
func (m *MIPRO) scoreExamples(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer, stopOnError bool) []exampleScore {
	return scoreExamples(ctx, core.GetConcurrencyLevel(), stopOnError, examples, func(example core.Example) (float64, error) {
		result, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			return 0, err
		}
		return scorer.Score(ctx, example, result)
	})
}
