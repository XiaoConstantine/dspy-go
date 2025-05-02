package optimizers

import (
	"context"
	"fmt"
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
	SuggestParams(ctx context.Context) (map[string]interface{}, error)
	UpdateResults(params map[string]interface{}, score float64) error
	GetBestParams() (map[string]interface{}, float64)
	Initialize(config SearchConfig) error
}

// SearchConfig contains configuration for search strategies.
type SearchConfig struct {
	ParamSpace  map[string][]interface{}
	MaxTrials   int
	Seed        int64
	Constraints map[string]interface{}
}

// TeacherStudentOptimizer handles the teacher-student learning dynamic.
type TeacherStudentOptimizer struct {
	Teacher         core.LLM
	Student         core.LLM
	TeacherSettings map[string]interface{}
	MaxExamples     int

	state struct {
		demonstrations []core.Example
		teacherScores  map[string]float64
	}
}

// Initialize sets up the teacher-student optimization.
func (t *TeacherStudentOptimizer) Initialize(ctx context.Context, program core.Program, dataset core.Dataset) error {
	t.state.demonstrations = make([]core.Example, 0)
	t.state.teacherScores = make(map[string]float64)
	return nil
}

// GenerateDemonstration creates a high-quality demonstration using the teacher.
func (t *TeacherStudentOptimizer) GenerateDemonstration(ctx context.Context, input core.Example) (core.Example, error) {
	response, err := t.Teacher.Generate(ctx, input.Inputs["prompt"].(string))
	if err != nil {
		return core.Example{}, fmt.Errorf("teacher generation failed: %w", err)
	}

	return core.Example{
		Inputs:  input.Inputs,
		Outputs: map[string]interface{}{"completion": response.Content},
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
			instruction, err := g.generateSingleCandidate(ctx, module, demos)
			if err != nil {
				return nil, fmt.Errorf("failed to generate candidate %d for module %d: %w", j, i, err)
			}
			moduleInstructions[j] = instruction
		}

		candidates[i] = moduleInstructions
	}

	return candidates, nil
}

func (g *InstructionGenerator) generateSingleCandidate(
	ctx context.Context,
	module core.Module,
	demos []core.Example,
) (string, error) {
	prompt := fmt.Sprintf("Generate an instruction for the following signature: %s", module.GetSignature())

	response, err := g.PromptModel.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("prompt model generation failed: %w", err)
	}

	return response.Content, nil
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
	TeacherSettings map[string]interface{}
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

// MIPRO is the main optimizer implementing multi-step interactive prompt optimization.
type MIPRO struct {
	// Core components
	metric               func(example, prediction map[string]interface{}, ctx context.Context) float64
	searchStrategy       SearchStrategy
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

	// Tracking and logging
	logger *logging.Logger
	mu     sync.RWMutex
}

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
func WithTeacherSettings(settings map[string]interface{}) MIPROOption {
	return func(m *MIPRO) {
		m.config.TeacherSettings = settings
	}
}

// NewMIPRO creates a new MIPRO optimizer instance.
func NewMIPRO(
	metric func(example, prediction map[string]interface{}, ctx context.Context) float64,
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
		},
		state: &OptimizationState{
			TeacherScores: make(map[string]float64),
		},
		metrics: &MIPROMetrics{
			PromptEffectiveness: make(map[string]float64),
		},
		logger: logging.GetLogger(),
	}

	// Apply options
	for _, opt := range opts {
		opt(m)
	}

	// Initialize components
	m.initComponents()

	return m
}

// initComponents initializes all required components.
func (m *MIPRO) initComponents() {
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
}

// Compile implements the main optimization loop.
func (m *MIPRO) Compile(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
	metric core.Metric,
) (core.Program, error) {
	m.logger.Info(ctx, "Starting MIPRO optimization with configuration: %+v", m.config)

	// Step 1: Initialize teacher-student learning
	if err := m.teacherStudent.Initialize(ctx, program, dataset); err != nil {
		return program, fmt.Errorf("failed to initialize teacher-student: %w", err)
	}

	// Step 2: Generate initial demonstrations
	demos, err := m.generateDemonstrations(ctx, program, dataset)
	if err != nil {
		return program, fmt.Errorf("failed to generate demonstrations: %w", err)
	}

	// Step 3: Generate instruction candidates
	instructions, err := m.instructionGenerator.GenerateCandidates(ctx, program, demos)
	if err != nil {
		return program, fmt.Errorf("failed to generate instructions: %w", err)
	}

	// Step 4: Run main optimization loop
	bestProgram, err := m.runOptimizationLoop(ctx, program, dataset, demos, instructions)
	if err != nil {
		return program, fmt.Errorf("optimization failed: %w", err)
	}

	// Step 5: Finalize and validate best program
	return m.finalizeProgram(ctx, bestProgram, dataset)
}

func (m *MIPRO) runOptimizationLoop(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
	demos []core.Example,
	instructions map[int][]string,
) (core.Program, error) {
	var bestProgram core.Program
	var bestScore float64

	for iteration := 0; iteration < m.config.NumTrials; iteration++ {
		// Get next parameter combination to try
		params, err := m.searchStrategy.SuggestParams(ctx)
		if err != nil {
			return program, fmt.Errorf("failed to get parameters: %w", err)
		}
		fmt.Println("params:", params) // Add this line

		// Create candidate program with suggested parameters
		candidate := m.createCandidateProgram(program, params, demos, instructions)

		// Evaluate candidate
		score, err := m.evaluateCandidate(ctx, candidate, dataset)
		if err != nil {
			return program, fmt.Errorf("failed to evaluate candidate: %w", err)
		}

		// Update optimization state
		m.updateOptimizationState(params, score, candidate)

		// Check if this is the best program so far
		if score > bestScore {
			bestScore = score
			bestProgram = candidate.Clone()
			m.logger.Info(ctx, "New best score: %f", bestScore)
		}

		// Check for convergence
		if m.hasConverged() {
			m.logger.Info(ctx, "Optimization converged after %d iterations", iteration)
			break
		}
	}

	return bestProgram, nil
}

// Helper functions for the teacher-student dynamic.
func (m *MIPRO) teacherDemonstration(ctx context.Context, example core.Example) (core.Example, error) {
	// Get a high-quality demonstration from the teacher model
	teacherResult, err := m.teacherStudent.Teacher.Generate(ctx, example.Inputs["prompt"].(string))
	if err != nil {
		return core.Example{}, fmt.Errorf("teacher demonstration failed: %w", err)
	}

	// Create a new example with the teacher's output
	return core.Example{
		Inputs:  example.Inputs,
		Outputs: map[string]interface{}{"completion": teacherResult.Content},
	}, nil
}

// Methods for tracking and analyzing performance.
func (m *MIPRO) updateOptimizationState(params map[string]interface{}, score float64, program core.Program) {
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

	// Update convergence measure
	m.updateConvergence(score)
}

func (m *MIPRO) extractAndStorePatterns(program core.Program) {
	// Analyze program structure and extract successful patterns
	for _, predictor := range program.Predictors() {
		signature := predictor.GetSignature()
		if signature.Instruction != "" {
			m.state.SuccessfulPatterns = append(m.state.SuccessfulPatterns, signature.Instruction)
		}
	}
}

func (m *MIPRO) hasConverged() bool {
	// Check various convergence criteria
	return m.state.Convergence > 0.95 ||
		m.state.CurrentIteration >= m.config.NumTrials ||
		(m.state.CurrentIteration > 10 && m.state.BestScore > 0.99)
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

// generateDemonstrations creates initial demonstrations.
func (m *MIPRO) generateDemonstrations(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
) ([]core.Example, error) {
	demos := make([]core.Example, 0, m.maxBootstrappedDemos)

	for i := 0; i < m.maxBootstrappedDemos; i++ {
		example, ok := dataset.Next()
		if !ok {
			break
		}

		demo, err := m.teacherStudent.GenerateDemonstration(ctx, example)
		if err != nil {
			return nil, fmt.Errorf("failed to generate demonstration: %w", err)
		}
		demos = append(demos, demo)
	}

	return demos, nil
}

// createCandidateProgram creates a new program with the given parameters.
func (m *MIPRO) createCandidateProgram(
	baseProgram core.Program,
	params map[string]interface{},
	demos []core.Example,
	instructions map[int][]string,
) core.Program {
	newProgram := baseProgram.Clone()
	modules := newProgram.GetModules() // Get ordered slice of modules

	for i, module := range modules {
		if moduleInstructions, ok := instructions[i]; ok {
			// Use the instructions for this module
			candidateIndex := int(params[fmt.Sprintf("module_%d_instruction", i)].(float64))
			if candidateIndex < len(moduleInstructions) {
				// Update the signature with the new instruction
				signature := module.GetSignature()
				signature = signature.WithInstruction(moduleInstructions[candidateIndex])
				module.SetSignature(signature)
			}
		}
	}

	return newProgram
}

// evaluateCandidate evaluates a candidate program.
func (m *MIPRO) evaluateCandidate(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
) (float64, error) {
	var totalScore float64
	var count int

	dataset.Reset()
	for {
		example, ok := dataset.Next()

		if !ok {
			break
		}

		result, err := program.Execute(ctx, example.Inputs)
		if err != nil {
			return 0, fmt.Errorf("failed to execute program: %w", err)
		}

		score := m.metric(example.Outputs, result, ctx)
		totalScore += score
		count++
	}

	if count == 0 {
		return 0, fmt.Errorf("no examples evaluated")
	}

	return totalScore / float64(count), nil
}

// finalizeProgram performs final validation and cleanup.
func (m *MIPRO) finalizeProgram(
	ctx context.Context,
	program core.Program,
	dataset core.Dataset,
) (core.Program, error) {
	score, err := m.evaluateCandidate(ctx, program, dataset)
	if err != nil {
		return program, fmt.Errorf("final validation failed: %w", err)
	}

	m.logger.Info(ctx, "Final program validation score: %f", score)

	return program, nil
}

// updateOptimizationState updates the optimization state with new results
// func (m *MIPRO) updateOptimizationState(params map[string]interface{}, score float64, program core.Program) {
// 	m.mu.Lock()
// 	defer m.mu.Unlock()
//
// 	m.state.CurrentIteration++
// 	if score > m.state.BestScore {
// 		m.state.BestScore = score
// 	}
//
// 	m.updateConvergence(score)
// 	m.extractAndStorePatterns(program)
// }

// updateConvergence updates the convergence measure.
func (m *MIPRO) updateConvergence(score float64) {
	// Simple convergence calculation based on improvement over best score
	if score > m.state.BestScore {
		m.state.Convergence = 1.0
	} else {
		m.state.Convergence = score / m.state.BestScore
	}
}

// hasConverged checks if optimization has converged
// func (m *MIPRO) hasConverged() bool {
// 	return m.state.Convergence > 0.95 ||
// 		m.state.CurrentIteration >= m.config.NumTrials ||
// 		(m.state.CurrentIteration > 10 && m.state.BestScore > 0.99)
// }
//
// // extractAndStorePatterns extracts successful patterns from the program
// func (m *MIPRO) extractAndStorePatterns(program core.Program) {
// 	for _, module := range program.Modules {
// 		signature := module.GetSignature()
// 		if signature.Instruction != "" {
// 			m.state.SuccessfulPatterns = append(m.state.SuccessfulPatterns, signature.Instruction)
// 		}
// 	}
// }
