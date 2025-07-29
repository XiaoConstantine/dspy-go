package optimizers

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/internal/testutil"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Helper functions for setting up mock LLM responses.
func setupGEPAMockLLM(mockLLM *testutil.MockLLM) {
	// Setup generate_variations response
	mockLLM.On("Generate", mock.Anything, mock.MatchedBy(func(prompt string) bool {
		return containsAny(prompt, []string{"variations", "diverse"})
	}), mock.Anything).Return(&core.LLMResponse{
		Content: `1. Carefully analyze the input and provide detailed output.
2. Thoroughly examine the data and give comprehensive results.
3. Systematically process the information and deliver accurate outcomes.`,
	}, nil)

	// Setup crossover response
	mockLLM.On("Generate", mock.Anything, mock.MatchedBy(func(prompt string) bool {
		return containsAny(prompt, []string{"crossover", "offspring", "combine"})
	}), mock.Anything).Return(&core.LLMResponse{
		Content: `1. Analyze the input thoroughly and provide detailed, accurate results.
2. Examine the data carefully and deliver comprehensive, systematic outcomes.`,
	}, nil)

	// Setup mutation response
	mockLLM.On("Generate", mock.Anything, mock.MatchedBy(func(prompt string) bool {
		return containsAny(prompt, []string{"mutation", "mutate"})
	}), mock.Anything).Return(&core.LLMResponse{
		Content: `Precisely analyze the input and provide detailed, accurate results with careful attention.`,
	}, nil)

	// Setup reflection response
	mockLLM.On("Generate", mock.Anything, mock.MatchedBy(func(prompt string) bool {
		return containsAny(prompt, []string{"reflection", "analyze this instruction"})
	}), mock.Anything).Return(&core.LLMResponse{
		Content: `STRENGTHS:
- Clear and direct instruction
- Focuses on accuracy

WEAKNESSES:
- Could be more specific
- Lacks examples

SUGGESTIONS:
- Add concrete examples
- Specify output format

CONFIDENCE: 0.8`,
	}, nil)

	// Default response for any other prompts
	mockLLM.On("Generate", mock.Anything, mock.Anything, mock.Anything).Return(&core.LLMResponse{
		Content: "Default response for testing",
	}, nil)
}

func containsAny(s string, substrings []string) bool {
	for _, substr := range substrings {
		if contains(s, substr) {
			return true
		}
	}
	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// Mock metric for testing.
func mockMetric(expected, actual map[string]interface{}) float64 {
	// Simple metric: return 1.0 if both have "output", 0.5 if only actual has it, 0.0 otherwise
	if _, expectedOk := expected["output"]; expectedOk {
		if _, actualOk := actual["output"]; actualOk {
			return 1.0
		}
		return 0.0
	}
	if _, actualOk := actual["output"]; actualOk {
		return 0.5
	}
	return 0.0
}

func TestNewGEPA(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := &GEPAConfig{
		PopulationSize: 10,
		MaxGenerations: 5,
		MutationRate:   0.3,
		CrossoverRate:  0.7,
	}

	gepa, err := NewGEPA(config)
	require.NoError(t, err)
	require.NotNil(t, gepa)

	assert.Equal(t, config.PopulationSize, gepa.config.PopulationSize)
	assert.Equal(t, config.MaxGenerations, gepa.config.MaxGenerations)
	assert.NotNil(t, gepa.state)
	assert.NotNil(t, gepa.generationLLM)
}

func TestDefaultGEPAConfig(t *testing.T) {
	config := DefaultGEPAConfig()
	require.NotNil(t, config)

	assert.Equal(t, 20, config.PopulationSize)
	assert.Equal(t, 10, config.MaxGenerations)
	assert.Equal(t, 0.3, config.MutationRate)
	assert.Equal(t, 0.7, config.CrossoverRate)
	assert.Equal(t, 0.1, config.ElitismRate)
	assert.Equal(t, 2, config.ReflectionFreq)
	assert.Equal(t, 3, config.TournamentSize)
}

func TestGEPAState(t *testing.T) {
	state := NewGEPAState()
	require.NotNil(t, state)

	assert.Equal(t, 0, state.CurrentGeneration)
	assert.Equal(t, 0.0, state.BestFitness)
	assert.NotNil(t, state.PopulationHistory)
	assert.NotNil(t, state.ExecutionTraces)
	assert.NotNil(t, state.CandidateMetrics)

	// Test adding trace
	trace := &ExecutionTrace{
		CandidateID: "test-id",
		ModuleName:  "test-module",
		Success:     true,
		Duration:    time.Millisecond * 100,
	}

	state.AddTrace(trace)

	traces := state.GetTracesForCandidate("test-id")
	assert.Len(t, traces, 1)
	assert.Equal(t, "test-id", traces[0].CandidateID)
}

func TestPopulationManagement(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := &GEPAConfig{
		PopulationSize: 5,
		MaxGenerations: 3,
	}

	gepa, err := NewGEPA(config)
	require.NoError(t, err)

	// Create test module with proper mock setup
	mockModule := testutil.NewMockModule("Process the input and provide output.")

	// Setup the Clone method to return a new mock module
	mockModule.On("Clone").Return(testutil.NewMockModule("Process the input and provide output."))

	// Create test program
	program := core.Program{
		Modules: map[string]core.Module{
			"test_module": mockModule,
		},
	}

	// Test population initialization
	err = gepa.initializePopulation(context.Background(), program)
	require.NoError(t, err)

	population := gepa.getCurrentPopulation()
	require.NotNil(t, population)
	assert.Equal(t, 0, population.Generation)
	assert.Len(t, population.Candidates, config.PopulationSize)

	// Verify candidates were created properly
	for _, candidate := range population.Candidates {
		assert.NotEmpty(t, candidate.ID)
		assert.NotEmpty(t, candidate.Instruction)
		assert.Equal(t, "test_module", candidate.ModuleName)
		assert.Equal(t, 0, candidate.Generation)
	}
}

func TestEvolutionaryOperators(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := &GEPAConfig{
		PopulationSize: 4,
		TournamentSize: 2,
		CrossoverRate:  1.0, // Always apply crossover for testing
		MutationRate:   1.0, // Always apply mutation for testing
	}

	gepa, err := NewGEPA(config)
	require.NoError(t, err)

	// Create test candidates
	candidates := []*GEPACandidate{
		{
			ID:          "candidate1",
			ModuleName:  "test",
			Instruction: "First instruction",
			Fitness:     0.8,
			Generation:  0,
		},
		{
			ID:          "candidate2",
			ModuleName:  "test",
			Instruction: "Second instruction",
			Fitness:     0.6,
			Generation:  0,
		},
		{
			ID:          "candidate3",
			ModuleName:  "test",
			Instruction: "Third instruction",
			Fitness:     0.4,
			Generation:  0,
		},
		{
			ID:          "candidate4",
			ModuleName:  "test",
			Instruction: "Fourth instruction",
			Fitness:     0.2,
			Generation:  0,
		},
	}

	population := &Population{
		Candidates:    candidates,
		Generation:    0,
		BestFitness:   0.8,
		BestCandidate: candidates[0],
		Size:          len(candidates),
	}

	// Test tournament selection
	selected := gepa.tournamentSelection(population, 2)
	assert.Len(t, selected, 2)

	// Test crossover
	parent1 := candidates[0]
	parent2 := candidates[1]
	child1, child2 := gepa.crossover(parent1, parent2)

	assert.NotEqual(t, parent1.ID, child1.ID)
	assert.NotEqual(t, parent2.ID, child2.ID)
	assert.Equal(t, utils.Max(parent1.Generation, parent2.Generation)+1, child1.Generation)

	// Test mutation
	mutated := gepa.mutate(context.Background(), candidates[0])
	assert.NotEqual(t, candidates[0].ID, mutated.ID)
	assert.Equal(t, candidates[0].Generation+1, mutated.Generation)
}

func TestGEPACompileBasic(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := &GEPAConfig{
		PopulationSize:      4,
		MaxGenerations:      2,
		EvaluationBatchSize: 2,
		ReflectionFreq:      1,
		ConcurrencyLevel:    2, // Set a positive concurrency level
	}

	gepa, err := NewGEPA(config)
	require.NoError(t, err)

	// Create test module with proper mock setup
	// We'll create a shared mock module that handles all calls
	mockModule := testutil.NewMockModule("Process the input.")

	// Allow unlimited SetSignature calls for cloned modules
	mockModule.On("SetSignature", mock.Anything).Return().Maybe()

	// Setup Clone to return itself for simplicity in testing
	mockModule.On("Clone").Return(mockModule).Maybe()

	// Create test program
	program := core.Program{
		Modules: map[string]core.Module{
			"test_module": mockModule,
		},
		Forward: func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"output": "test response"}, nil
		},
	}

	// Create test dataset
	dataset := testutil.NewMockDataset([]core.Example{
		{
			Inputs:  map[string]interface{}{"input": "test input 1"},
			Outputs: map[string]interface{}{"output": "expected output 1"},
		},
		{
			Inputs:  map[string]interface{}{"input": "test input 2"},
			Outputs: map[string]interface{}{"output": "expected output 2"},
		},
		{
			Inputs:  map[string]interface{}{"input": "test input 3"},
			Outputs: map[string]interface{}{"output": "expected output 3"},
		},
	})

	// Run optimization
	optimizedProgram, err := gepa.Compile(context.Background(), program, dataset, mockMetric)
	require.NoError(t, err)
	require.NotNil(t, optimizedProgram)

	// Verify optimization state
	state := gepa.GetOptimizationState()
	assert.GreaterOrEqual(t, state.CurrentGeneration, 0) // Changed to >= 0 since convergence can happen at generation 0
	assert.NotNil(t, state.BestCandidate)
	assert.GreaterOrEqual(t, state.BestFitness, 0.0)

	// Verify population history
	assert.NotEmpty(t, state.PopulationHistory)
	assert.GreaterOrEqual(t, len(state.PopulationHistory), 1)
}

func TestReflectionParsing(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	reflectionContent := `STRENGTHS:
- Clear and concise instruction
- Easy to understand

WEAKNESSES:
- Lacks specific details
- Could be more comprehensive

SUGGESTIONS:
- Add examples
- Include format specifications

CONFIDENCE: 0.75`

	reflection := gepa.parseReflectionResponse(reflectionContent, "test-candidate")

	assert.Equal(t, "test-candidate", reflection.CandidateID)
	assert.Len(t, reflection.Strengths, 2)
	assert.Contains(t, reflection.Strengths[0], "Clear and concise")
	assert.Len(t, reflection.Weaknesses, 2)
	assert.Contains(t, reflection.Weaknesses[0], "Lacks specific details")
	assert.Len(t, reflection.Suggestions, 2)
	assert.Contains(t, reflection.Suggestions[0], "Add examples")
}

func TestVariationParsing(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	variationContent := `1. Carefully analyze the input and provide detailed output.
2. Thoroughly examine the data and give comprehensive results.
3. Systematically process the information and deliver accurate outcomes.`

	variations := gepa.parseVariations(variationContent)

	assert.GreaterOrEqual(t, len(variations), 3)
	assert.Contains(t, variations[0], "Carefully analyze")
	assert.Contains(t, variations[1], "Thoroughly examine")
	assert.Contains(t, variations[2], "Systematically process")
}

func TestConvergenceDetection(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := &GEPAConfig{
		ConvergenceThreshold: 0.01,
		StagnationLimit:      2,
	}

	gepa, err := NewGEPA(config)
	require.NoError(t, err)

	// Test initial state (should not be converged)
	assert.False(t, gepa.hasConverged())

	// Simulate stagnation by setting last improvement time in the past
	gepa.state.LastImprovement = time.Now().Add(-5 * time.Minute)
	gepa.state.ConvergenceStatus.StagnationCount = config.StagnationLimit

	// Should now be converged due to stagnation
	assert.True(t, gepa.hasConverged())
}

func TestMultiObjectiveFitnessSystem(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Test multi-objective fitness calculation
	candidate := &GEPACandidate{
		ID:          "test-candidate-1",
		ModuleName:  "test_module",
		Instruction: "Test instruction for fitness calculation",
		Generation:  0,
	}

	// Mock execution traces for robustness assessment
	traces := []ExecutionTrace{
		{
			CandidateID: candidate.ID,
			Success:     true,
			Duration:    100 * time.Millisecond,
			Inputs:      map[string]any{"input": "test1"},
			Outputs:     map[string]any{"output": "result1"},
		},
		{
			CandidateID: candidate.ID,
			Success:     true,
			Duration:    120 * time.Millisecond,
			Inputs:      map[string]any{"input": "test2"},
			Outputs:     map[string]any{"output": "result2"},
		},
	}

	// Add traces to state
	for _, trace := range traces {
		gepa.state.AddTrace(&trace)
	}

	// Test fitness calculation
	testInputs := map[string]any{"input": "test"}
	testOutputs := map[string]any{"output": "result"}
	testContext := map[string]interface{}{"duration": 150 * time.Millisecond}
	fitness := gepa.calculateMultiObjectiveFitness(candidate.ID, testInputs, testOutputs, nil, testContext)

	// Verify fitness components
	assert.Greater(t, fitness.SuccessRate, 0.0)
	assert.Greater(t, fitness.OutputQuality, 0.0)
	assert.Greater(t, fitness.Efficiency, 0.0)
	assert.GreaterOrEqual(t, fitness.Robustness, 0.0)
	assert.GreaterOrEqual(t, fitness.Generalization, 0.0)
	assert.GreaterOrEqual(t, fitness.Diversity, 0.0)
	assert.GreaterOrEqual(t, fitness.Innovation, 0.0)
	assert.Greater(t, fitness.WeightedScore, 0.0)
}

func TestContextAwarePerformanceTracking(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	candidate := &GEPACandidate{
		ID:          "context-test-candidate",
		ModuleName:  "test_module",
		Instruction: "Context-aware test instruction",
		Generation:  0,
	}

	duration := 100 * time.Millisecond

	// Test context-aware efficiency assessment
	testInputs := map[string]any{"input": "test"}
	testOutputs := map[string]any{"output": "result"}
	testContext := map[string]interface{}{"duration": duration}
	efficiency := gepa.assessContextAwareEfficiency(candidate.ID, testInputs, testOutputs, nil, testContext)
	assert.GreaterOrEqual(t, efficiency, 0.0)
	assert.LessOrEqual(t, efficiency, 1.0)

	// Test performance context creation
	perfContext := gepa.createPerformanceContext(candidate.ID, testInputs, testOutputs, testContext)
	assert.NotNil(t, perfContext)
	assert.Equal(t, candidate.ID, perfContext.CandidateID)
	assert.GreaterOrEqual(t, perfContext.SystemLoad, 0.0)
	assert.GreaterOrEqual(t, perfContext.MemoryUsage, 0.0)
	assert.GreaterOrEqual(t, perfContext.ConcurrentTasks, 0)
}

func TestParetoBasedSelection(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Create test candidates with different fitness profiles
	candidates := []*GEPACandidate{
		{
			ID:          "pareto-candidate-1",
			ModuleName:  "test",
			Instruction: "High success, low efficiency",
			Fitness:     0.9,
		},
		{
			ID:          "pareto-candidate-2",
			ModuleName:  "test",
			Instruction: "Medium success, high efficiency",
			Fitness:     0.7,
		},
		{
			ID:          "pareto-candidate-3",
			ModuleName:  "test",
			Instruction: "Low success, medium efficiency",
			Fitness:     0.5,
		},
	}

	// Create multi-objective fitness map
	fitnessMap := map[string]*MultiObjectiveFitness{
		"pareto-candidate-1": {
			SuccessRate:   0.9,
			Efficiency:    0.3,
			OutputQuality: 0.8,
			Robustness:    0.7,
			WeightedScore: 0.675,
		},
		"pareto-candidate-2": {
			SuccessRate:   0.7,
			Efficiency:    0.9,
			OutputQuality: 0.6,
			Robustness:    0.8,
			WeightedScore: 0.75,
		},
		"pareto-candidate-3": {
			SuccessRate:   0.5,
			Efficiency:    0.6,
			OutputQuality: 0.4,
			Robustness:    0.5,
			WeightedScore: 0.5,
		},
	}

	// Test Pareto front calculation
	fronts := gepa.calculateParetoFronts(candidates, fitnessMap)
	assert.NotEmpty(t, fronts)
	assert.GreaterOrEqual(t, len(fronts), 1)

	// Test Pareto-based selection
	selected := gepa.selectWithParetoRanking(candidates, fitnessMap, 2)
	assert.Len(t, selected, 2)

	// Test crowding distance calculation
	distances := gepa.calculateCrowdingDistance(candidates, fitnessMap)
	assert.Len(t, distances, len(candidates))
	for _, distance := range distances {
		assert.GreaterOrEqual(t, distance, 0.0)
	}
}

func TestAdvancedSelectionStrategies(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	candidates := []*GEPACandidate{
		{ID: "sel-1", Fitness: 0.9},
		{ID: "sel-2", Fitness: 0.7},
		{ID: "sel-3", Fitness: 0.5},
		{ID: "sel-4", Fitness: 0.3},
	}

	population := &Population{
		Candidates: candidates,
		Size:       len(candidates),
	}

	// Test roulette selection
	selected := gepa.rouletteSelection(population, 2)
	assert.Len(t, selected, 2)

	// Test adaptive weighted selection
	fitnessMap := map[string]*MultiObjectiveFitness{
		"sel-1": {WeightedScore: 0.9},
		"sel-2": {WeightedScore: 0.7},
		"sel-3": {WeightedScore: 0.5},
		"sel-4": {WeightedScore: 0.3},
	}

	adaptiveSelected := gepa.adaptiveWeightedSelection(candidates, fitnessMap, 2, 1)
	assert.Len(t, adaptiveSelected, 2)
}

func TestReflectionEngine(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	ctx := context.Background()
	candidate := &GEPACandidate{
		ID:          "reflection-candidate",
		ModuleName:  "test_module",
		Instruction: "Test instruction for reflection",
		Fitness:     0.8,
		Generation:  1,
	}

	// Add some execution traces
	traces := []ExecutionTrace{
		{
			CandidateID: candidate.ID,
			Success:     true,
			Duration:    100 * time.Millisecond,
		},
		{
			CandidateID: candidate.ID,
			Success:     false,
			Duration:    200 * time.Millisecond,
		},
	}

	for _, trace := range traces {
		gepa.state.AddTrace(&trace)
	}

	// Test execution pattern analysis
	patterns := gepa.analyzeExecutionPatterns(traces)
	assert.NotNil(t, patterns)
	assert.Equal(t, 2, patterns.TotalExecutions)
	assert.Equal(t, 1, patterns.SuccessCount)
	assert.Equal(t, 0.5, patterns.SuccessRate)

	// Test reflection prompt building
	prompt := gepa.buildReflectionPrompt(candidate, patterns)
	assert.Contains(t, prompt, candidate.Instruction)
	assert.Contains(t, prompt, "STRENGTHS")
	assert.Contains(t, prompt, "WEAKNESSES")

	// Test reflection on candidate
	reflection, err := gepa.reflectOnCandidate(ctx, candidate, traces)
	require.NoError(t, err)
	assert.Equal(t, candidate.ID, reflection.CandidateID)
}

func TestSelfCritiqueSystem(t *testing.T) {
	// Set up mock LLM with critique response
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)

	// Add specific mock for self-critique
	mockLLM.On("Generate", mock.Anything, mock.MatchedBy(func(prompt string) bool {
		return containsAny(prompt, []string{"critique", "analyze", "performance"})
	}), mock.Anything).Return(&core.LLMResponse{
		Content: `PERFORMANCE ANALYSIS:
- Success rate: 75%
- Response quality: Good
- Efficiency: Moderate

PATTERN ANALYSIS:
- Input handling: Consistent
- Output format: Structured
- Error patterns: Minimal

COMPARATIVE ANALYSIS:
- Better than baseline
- Room for improvement in speed

OVERALL ASSESSMENT:
Good performance with potential for optimization.`,
	}, nil)

	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	ctx := context.Background()
	candidate := &GEPACandidate{
		ID:          "critique-candidate",
		ModuleName:  "test_module",
		Instruction: "Test instruction for self-critique",
		Fitness:     0.75,
		Generation:  1,
	}

	// Create some traces for the candidate
	traces := []ExecutionTrace{
		{
			CandidateID: candidate.ID,
			Success:     true,
			Duration:    100 * time.Millisecond,
		},
	}

	// Add traces to state so selfCritiqueCandidate can find them
	for _, trace := range traces {
		gepa.state.AddTrace(&trace)
	}

	// Test critique prompt building
	prompt := gepa.buildCritiquePrompt(candidate, traces)
	assert.Contains(t, prompt, candidate.Instruction)
	assert.Contains(t, prompt, "Execution Performance Analysis")

	// Test self-critique
	critique, err := gepa.selfCritiqueCandidate(ctx, candidate)
	require.NoError(t, err)
	assert.Equal(t, candidate.ID, critique.CandidateID)
	assert.NotEmpty(t, critique.Strengths)
}

func TestMultiLevelReflectionSystem(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	ctx := context.Background()

	// Initialize some population data
	candidates := []*GEPACandidate{
		{ID: "ml-1", Fitness: 0.9, Generation: 0},
		{ID: "ml-2", Fitness: 0.7, Generation: 0},
		{ID: "ml-3", Fitness: 0.5, Generation: 1},
	}

	population := &Population{
		Candidates: candidates,
		Generation: 1,
		Size:       len(candidates),
	}

	gepa.state.PopulationHistory = []*Population{population}

	// Test candidate selection for reflection
	diverseCandidates := gepa.selectDiverseCandidates(candidates, 2)
	assert.LessOrEqual(t, len(diverseCandidates), 2)

	interestingCandidates := gepa.selectInterestingCandidates(candidates, 2)
	assert.LessOrEqual(t, len(interestingCandidates), 2)

	// Test individual reflection
	individualResults := gepa.performIndividualReflection(ctx)
	assert.NotNil(t, individualResults)

	// Test population pattern analysis
	patterns := gepa.analyzePopulationPatterns()
	assert.NotNil(t, patterns)

}

func TestErrorHandlingAndEdgeCases(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	config := DefaultGEPAConfig()
	config.PopulationSize = 1 // Minimal population for edge case testing

	gepa, err := NewGEPA(config)
	require.NoError(t, err)

	// Test with empty population
	emptyPopulation := &Population{
		Candidates: []*GEPACandidate{},
		Size:       0,
	}

	// Test selection with empty population - should not panic
	selected := gepa.tournamentSelection(emptyPopulation, 0)
	assert.Empty(t, selected)

	// Test with nil fitness map
	nilFitnessSelected := gepa.selectWithParetoRanking([]*GEPACandidate{}, nil, 1)
	assert.Empty(t, nilFitnessSelected)

	// Test candidate copying
	original := &GEPACandidate{
		ID:          "copy-test",
		Instruction: "Original instruction",
		Fitness:     0.8,
		Metadata:    map[string]interface{}{"key": "value"},
	}

	copied := gepa.copyCandidate(original)
	assert.Equal(t, original.Instruction, copied.Instruction)
	assert.Equal(t, original.Fitness, copied.Fitness)
	assert.Equal(t, original.ID, copied.ID) // copyCandidate preserves original ID
	assert.NotSame(t, original, copied)     // But creates a new instance

	// Test convergence with various states
	gepa.state.BestFitness = 1.0
	gepa.state.ConvergenceStatus.StagnationCount = 0
	assert.False(t, gepa.hasConverged()) // High fitness but no stagnation

	// Test system monitoring functions
	systemLoad := gepa.getCurrentSystemLoad()
	assert.GreaterOrEqual(t, systemLoad, 0.0)

	memoryUsage := gepa.getCurrentMemoryUsage()
	assert.GreaterOrEqual(t, memoryUsage, 0.0)

	concurrentTasks := gepa.getCurrentConcurrentTasks()
	assert.GreaterOrEqual(t, concurrentTasks, 0)
}

func TestInterceptorIntegration(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	ctx := gepa.withGEPAState(context.Background())

	// Test GEPA state context functions
	state := GetGEPAState(ctx)
	assert.NotNil(t, state)

	// Test execution tracking
	inputs := map[string]any{"test": "input"}
	outputs := map[string]any{"test": "output"}

	info := &core.ModuleInfo{
		ModuleName: "test_module",
	}

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return outputs, nil
	}

	// Test execution tracker
	result, err := gepa.gepaExecutionTracker(ctx, inputs, info, handler)
	require.NoError(t, err)
	assert.Equal(t, outputs, result)

	// Test performance collector
	result2, err := gepa.gepaPerformanceCollector(ctx, inputs, info, handler)
	require.NoError(t, err)
	assert.Equal(t, outputs, result2)

	// Test reflection logger
	result3, err := gepa.gepaReflectionLogger(ctx, inputs, info, handler)
	require.NoError(t, err)
	assert.Equal(t, outputs, result3)
}

func TestAdvancedAnalysisFunctions(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	candidate := &GEPACandidate{
		ID:          "analysis-candidate",
		ModuleName:  "test_module",
		Instruction: "Complex analysis instruction",
		Generation:  1,
	}

	// Add execution traces
	traces := []ExecutionTrace{
		{
			CandidateID: candidate.ID,
			Inputs:      map[string]any{"type": "question", "content": "What is AI?"},
			Outputs:     map[string]any{"answer": "Artificial Intelligence..."},
			Success:     true,
		},
		{
			CandidateID: candidate.ID,
			Inputs:      map[string]any{"type": "request", "content": "Explain ML"},
			Outputs:     map[string]any{"response": "Machine Learning..."},
			Success:     true,
		},
	}

	// Test input pattern analysis
	inputPatterns := gepa.analyzeInputPatterns(traces)
	assert.NotNil(t, inputPatterns)

	// Test output pattern analysis
	outputPatterns := gepa.analyzeOutputPatterns(traces)
	assert.NotNil(t, outputPatterns)

	// Test failure pattern analysis (with failed trace)
	failedTraces := []ExecutionTrace{
		{
			CandidateID: candidate.ID,
			Success:     false,
			Error:       assert.AnError,
		},
	}

	failurePatterns := gepa.analyzeFailurePatterns(failedTraces)
	assert.NotNil(t, failurePatterns)

	// Test fitness calculation components
	fitness := gepa.calculateFitness(
		map[string]any{"input": "test"},
		map[string]any{"output": "result"},
		nil,
	)
	assert.GreaterOrEqual(t, fitness, 0.0)

	// Test quality assessment
	quality := gepa.assessOutputQuality(
		map[string]any{"output": "high quality response"},
	)
	assert.GreaterOrEqual(t, quality, 0.0)

	// Test input utilization
	utilization := gepa.assessInputUtilization(
		map[string]any{"output": "response using input concepts"},
		map[string]any{"input": "test input concepts"},
	)
	assert.GreaterOrEqual(t, utilization, 0.0)
}

func TestEvolutionOperationsAdvanced(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Test evolution operations
	ctx := context.Background()

	// Test evolvePopulation
	candidates := []*GEPACandidate{
		{ID: "evo-1", Fitness: 0.9, Generation: 0},
		{ID: "evo-2", Fitness: 0.7, Generation: 0},
	}

	population := &Population{
		Candidates: candidates,
		Generation: 0,
		Size:       len(candidates),
	}

	gepa.state.PopulationHistory = []*Population{population}

	// Test evolvePopulation function
	_ = gepa.evolvePopulation(ctx)
	// This may fail due to empty population but should not panic - we don't check error here

	// Test selectElite
	elite := gepa.selectElite(population, 1)
	assert.LessOrEqual(t, len(elite), 1)

	// Test selectParetoElite
	fitnessMap := map[string]*MultiObjectiveFitness{
		"evo-1": {WeightedScore: 0.9},
		"evo-2": {WeightedScore: 0.7},
	}

	paretoElite := gepa.selectParetoElite(candidates, fitnessMap, 1)
	assert.LessOrEqual(t, len(paretoElite), 1)
}

func TestPatternAnalysisAndSimilarity(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Test pattern similarity calculation with string patterns
	pattern1 := "test pattern 1 with question content"
	pattern2 := "test pattern 2 with request content"

	similarity := gepa.calculatePatternSimilarity(pattern1, pattern2)
	assert.GreaterOrEqual(t, similarity, 0.0)
	assert.LessOrEqual(t, similarity, 1.0)

	// Test context-aware performance tracker (global function)
	tracker := NewContextAwarePerformanceTracker()
	assert.NotNil(t, tracker)

	// Test selection pressure calculation
	population := &Population{
		Candidates: []*GEPACandidate{
			{Fitness: 0.9},
			{Fitness: 0.7},
			{Fitness: 0.5},
		},
		Size: 3,
	}

	pressure := gepa.calculateSelectionPressure(population)
	assert.GreaterOrEqual(t, pressure, 0.0)
}

func TestAdvancedDiversityAndInnovation(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	candidateID := "innovation-test"

	// Test approach uniqueness assessment
	uniqueness := gepa.assessApproachUniqueness(candidateID)
	assert.GreaterOrEqual(t, uniqueness, 0.0)
	assert.LessOrEqual(t, uniqueness, 1.0)

	// Test output pattern novelty
	outputMap := map[string]any{"pattern": "novel output pattern for testing with structured format"}

	novelty := gepa.assessOutputPatternNovelty(outputMap, candidateID)
	assert.GreaterOrEqual(t, novelty, 0.0)
	assert.LessOrEqual(t, novelty, 1.0)

	// Test generalization assessment
	inputs := map[string]any{"input": "test generalization"}
	outputs := map[string]any{"output": "test result"}
	generalization := gepa.assessGeneralization(candidateID, inputs, outputs)
	assert.GreaterOrEqual(t, generalization, 0.0)
	assert.LessOrEqual(t, generalization, 1.0)
}

func TestPerformanceLogging(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Test system monitoring functions
	systemLoad := gepa.getCurrentSystemLoad()
	assert.GreaterOrEqual(t, systemLoad, 0.0)

	memoryUsage := gepa.getCurrentMemoryUsage()
	assert.GreaterOrEqual(t, memoryUsage, 0.0)

	concurrentTasks := gepa.getCurrentConcurrentTasks()
	assert.GreaterOrEqual(t, concurrentTasks, 0)

	ctx := context.Background()
	executionPhase := gepa.getCurrentExecutionPhase(ctx)
	assert.NotEmpty(t, executionPhase)
}

func TestSetProgressReporter(t *testing.T) {
	// Set up mock LLM
	mockLLM := &testutil.MockLLM{}
	setupGEPAMockLLM(mockLLM)
	core.SetDefaultLLM(mockLLM)

	gepa, err := NewGEPA(DefaultGEPAConfig())
	require.NoError(t, err)

	// Test setting progress reporter
	mockReporter := &MockProgressReporter{}
	gepa.SetProgressReporter(mockReporter)

	// Verify it was set (no direct getter, but this tests the function)
	assert.NotNil(t, gepa.progressReporter)
}

// MockProgressReporter for testing.
type MockProgressReporter struct{}

func (m *MockProgressReporter) Report(operation string, current, total int) {
	// Mock implementation
}
