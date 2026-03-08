package optimizers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// BootstrapPopulationFromSeed initializes a GEPA population from a single external seed candidate.
func (g *GEPA) BootstrapPopulationFromSeed(ctx context.Context, seed *GEPACandidate) error {
	if g == nil {
		return fmt.Errorf("optimizers: nil GEPA optimizer")
	}
	if seed == nil {
		return fmt.Errorf("optimizers: nil GEPA seed candidate")
	}
	if strings.TrimSpace(seed.Instruction) == "" {
		return fmt.Errorf("optimizers: GEPA seed candidate must include a non-empty instruction")
	}

	logger := logging.GetLogger()
	logger.Info(ctx, "Bootstrapping GEPA population from seed candidate: module=%s, population_size=%d",
		seed.ModuleName,
		g.config.PopulationSize)

	g.state.mu.Lock()
	if len(g.state.PopulationHistory) > 0 {
		g.state.mu.Unlock()
		return fmt.Errorf("optimizers: GEPA population has already been initialized")
	}
	g.state.mu.Unlock()

	variations, err := g.generateInitialVariations(ctx, seed.Instruction, seedModuleName(seed), g.config.PopulationSize)
	if err != nil {
		logger.Warn(ctx, "Failed to generate seed variations, falling back to seed instruction: %v", err)
		variations = []string{seed.Instruction}
	}

	candidates := make([]*GEPACandidate, 0, g.config.PopulationSize)
	for i, variation := range variations {
		if len(candidates) >= g.config.PopulationSize {
			break
		}

		candidate := g.candidateFromSeed(seed, variation, 0, nil, map[string]interface{}{
			"variation_index":  i,
			"base_instruction": seed.Instruction,
		})
		candidates = append(candidates, candidate)
		g.ensureCandidateMetrics(candidate.ID)
	}

	for len(candidates) < g.config.PopulationSize {
		original := candidates[g.rng.Intn(len(candidates))]
		mutated := g.createMutatedCandidate(original)
		mutated.Generation = 0
		candidates = append(candidates, mutated)
		g.ensureCandidateMetrics(mutated.ID)
	}

	population := &Population{
		Candidates:    candidates,
		Generation:    0,
		BestFitness:   0.0,
		BestCandidate: nil,
		Size:          len(candidates),
	}

	g.state.mu.Lock()
	defer g.state.mu.Unlock()
	if len(g.state.PopulationHistory) > 0 {
		return fmt.Errorf("optimizers: GEPA population has already been initialized")
	}
	g.state.PopulationHistory = []*Population{population}

	return nil
}

// CurrentPopulation returns the latest GEPA population.
func (g *GEPA) CurrentPopulation() *Population {
	return g.getCurrentPopulation()
}

// SetCurrentMultiObjectiveFitnessMap records the current generation's multi-objective fitness map.
func (g *GEPA) SetCurrentMultiObjectiveFitnessMap(fitnessMap map[string]*MultiObjectiveFitness) {
	g.setCurrentMultiObjectiveFitnessMap(fitnessMap)
}

// UpdateBestCandidate records a new best candidate if it improves the optimizer state.
func (g *GEPA) UpdateBestCandidate(candidate *GEPACandidate) {
	g.updateBestCandidate(candidate)
}

// EvolveCurrentPopulation advances GEPA to the next generation using its existing operators.
func (g *GEPA) EvolveCurrentPopulation(ctx context.Context) error {
	return g.evolvePopulation(ctx)
}

// ReflectCurrentPopulation runs GEPA's current reflection logic for the given generation.
func (g *GEPA) ReflectCurrentPopulation(ctx context.Context, generation int) error {
	return g.performReflection(ctx, generation)
}

// HasConverged returns whether GEPA's convergence criteria have been met.
func (g *GEPA) HasConverged() bool {
	return g.hasConverged()
}

func (g *GEPA) candidateFromSeed(seed *GEPACandidate, instruction string, generation int, parentIDs []string, extraMetadata map[string]interface{}) *GEPACandidate {
	return &GEPACandidate{
		ID:          g.generateCandidateID(),
		ModuleName:  seedModuleName(seed),
		Instruction: instruction,
		Generation:  generation,
		ParentIDs:   append([]string(nil), parentIDs...),
		CreatedAt:   time.Now(),
		Metadata:    mergeCandidateMetadata(extraMetadata, seed.Metadata),
	}
}

func (g *GEPA) ensureCandidateMetrics(candidateID string) {
	g.state.mu.Lock()
	defer g.state.mu.Unlock()

	if _, exists := g.state.CandidateMetrics[candidateID]; exists {
		return
	}

	g.state.CandidateMetrics[candidateID] = &CandidateMetrics{
		TotalEvaluations: 0,
		SuccessCount:     0,
		AverageFitness:   0.0,
		BestFitness:      0.0,
		ExecutionTimes:   make([]time.Duration, 0),
		ErrorCounts:      make(map[string]int),
		Metadata:         make(map[string]interface{}),
	}
}

// RecordCandidateFitness updates the stored fitness/metadata for a candidate.
func (s *GEPAState) RecordCandidateFitness(candidate *GEPACandidate, fitness *MultiObjectiveFitness, averageScore float64) {
	if s == nil || candidate == nil || fitness == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	metrics, exists := s.CandidateMetrics[candidate.ID]
	if !exists {
		metrics = &CandidateMetrics{
			ExecutionTimes: make([]time.Duration, 0),
			ErrorCounts:    make(map[string]int),
			Metadata:       make(map[string]interface{}),
		}
		s.CandidateMetrics[candidate.ID] = metrics
	}
	if metrics.Metadata == nil {
		metrics.Metadata = make(map[string]interface{})
	}

	metrics.AverageFitness = candidate.Fitness
	if candidate.Fitness > metrics.BestFitness {
		metrics.BestFitness = candidate.Fitness
	}
	metrics.Metadata["multi_objective_fitness"] = fitness
	metrics.Metadata["average_score"] = averageScore
}

// CloneCandidate returns a deep copy of the candidate so callers can evaluate
// or mutate it without affecting the original lineage record.
func CloneCandidate(candidate *GEPACandidate) *GEPACandidate {
	if candidate == nil {
		return nil
	}

	cloned := &GEPACandidate{
		ID:          candidate.ID,
		ModuleName:  candidate.ModuleName,
		Instruction: candidate.Instruction,
		Generation:  candidate.Generation,
		Fitness:     candidate.Fitness,
		CreatedAt:   candidate.CreatedAt,
	}

	if len(candidate.ParentIDs) > 0 {
		cloned.ParentIDs = append([]string(nil), candidate.ParentIDs...)
	}
	if len(candidate.Demonstrations) > 0 {
		cloned.Demonstrations = append([]core.Example(nil), candidate.Demonstrations...)
	}
	cloned.Metadata = cloneCandidateMetadata(candidate.Metadata)

	return cloned
}

func seedModuleName(seed *GEPACandidate) string {
	if seed != nil && strings.TrimSpace(seed.ModuleName) != "" {
		return seed.ModuleName
	}
	return "agent_artifact"
}

func cloneCandidateMetadata(source map[string]interface{}) map[string]interface{} {
	cloned := make(map[string]interface{}, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}

func mergeCandidateMetadata(extra map[string]interface{}, sources ...map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{})
	// Earlier sources win for conflicting inherited keys; the explicit extra
	// metadata for the new candidate overwrites inherited values last.
	for _, source := range sources {
		for key, value := range source {
			if _, exists := merged[key]; !exists {
				merged[key] = value
			}
		}
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}
