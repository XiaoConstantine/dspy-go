package optimizers

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

const gepaRunStateVersion = 1

type gepaRunSnapshot struct {
	Version int         `json:"version"`
	Config  *GEPAConfig `json:"config,omitempty"`
	State   *GEPAState  `json:"state,omitempty"`
}

func (g *GEPA) prepareRunState() (bool, error) {
	if g == nil || g.config == nil {
		return false, nil
	}

	// Resumed runs assume the caller is providing the same program shape,
	// dataset, and metric semantics as the original run. GEPA does not try to
	// fingerprint those inputs, so changing them under the same RunDir can
	// invalidate the persisted optimization history.
	return g.loadRunState()
}

func (g *GEPA) runStatePath() string {
	if g == nil || g.config == nil {
		return ""
	}

	runDir := filepath.Clean(g.config.RunDir)
	if runDir == "." || runDir == "" {
		return ""
	}

	return filepath.Join(runDir, "gepa_state.json")
}

func (g *GEPA) saveRunState() error {
	if g == nil || g.state == nil {
		return nil
	}

	path := g.runStatePath()
	if path == "" {
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("optimizers: create GEPA run dir: %w", err)
	}

	snapshot := gepaRunSnapshot{
		Version: gepaRunStateVersion,
		Config:  g.config,
		State:   sanitizePersistedGEPAState(g.state),
	}
	rendered, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("optimizers: marshal GEPA run state: %w", err)
	}

	tempPath := path + ".tmp"
	if err := os.WriteFile(tempPath, append(rendered, '\n'), 0o644); err != nil {
		return fmt.Errorf("optimizers: write GEPA run state: %w", err)
	}
	if err := os.Rename(tempPath, path); err != nil {
		return fmt.Errorf("optimizers: finalize GEPA run state: %w", err)
	}

	return nil
}

func (g *GEPA) loadRunState() (bool, error) {
	if g == nil {
		return false, nil
	}

	path := g.runStatePath()
	if path == "" {
		return false, nil
	}

	rendered, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("optimizers: read GEPA run state: %w", err)
	}

	var snapshot gepaRunSnapshot
	if err := json.Unmarshal(rendered, &snapshot); err != nil {
		return false, fmt.Errorf("optimizers: unmarshal GEPA run state: %w", err)
	}
	if snapshot.Version != gepaRunStateVersion {
		if snapshot.Version > gepaRunStateVersion {
			return false, fmt.Errorf(
				"optimizers: GEPA run-state version %d is newer than supported version %d",
				snapshot.Version,
				gepaRunStateVersion,
			)
		}
		return false, fmt.Errorf(
			"optimizers: GEPA run-state version %d is older than supported version %d",
			snapshot.Version,
			gepaRunStateVersion,
		)
	}
	if snapshot.State == nil {
		return false, fmt.Errorf("optimizers: GEPA run state missing optimizer state")
	}

	g.state = snapshot.State
	g.hydrateLoadedState()
	g.state.StopReason = ""
	g.state.StopMetadata = nil

	return true, nil
}

func (g *GEPA) hydrateLoadedState() {
	if g == nil || g.state == nil {
		return
	}

	if g.state.ConvergenceStatus == nil {
		g.state.ConvergenceStatus = &ConvergenceStatus{}
	}
	if g.state.ExecutionTraces == nil {
		g.state.ExecutionTraces = make(map[string][]ExecutionTrace)
	}
	if g.state.CandidateMetrics == nil {
		g.state.CandidateMetrics = make(map[string]*CandidateMetrics)
	}
	if g.state.MultiObjectiveFitnessMap == nil {
		g.state.MultiObjectiveFitnessMap = make(map[string]*MultiObjectiveFitness)
	}
	if g.state.ValidationFrontier == nil {
		g.state.ValidationFrontier = make(map[int]*gepaValidationFrontierEntry)
	}
	if g.state.ValidationCoverage == nil {
		g.state.ValidationCoverage = make(map[string]int)
	}
	if g.state.PerformedMerges == nil {
		g.state.PerformedMerges = make(map[string]bool)
	}
	if g.state.ParetoArchive == nil {
		g.state.ParetoArchive = make([]*GEPACandidate, 0)
	}
	if g.state.ArchiveFitnessMap == nil {
		g.state.ArchiveFitnessMap = make(map[string]*MultiObjectiveFitness)
	}

	// Transient caches are rebuilt on the next evaluation/validation cycle.
	// This intentionally drops cached example-level feedback evidence because it
	// is re-derived from the configured feedback evaluator and execution results.
	g.state.candidateReflections = make(map[string]*ReflectionResult)
	g.state.candidateEvaluations = make(map[string]*gepaCandidateEvaluation)
	g.state.candidateValidationEvals = make(map[string]*gepaCandidateEvaluation)
	g.state.evaluationCaseCache = make(map[string]*gepaCachedEvaluationCase)
}

func sanitizePersistedGEPAState(state *GEPAState) *GEPAState {
	if state == nil {
		return nil
	}

	sanitized := &GEPAState{
		CurrentGeneration:        state.CurrentGeneration,
		BestCandidate:            state.BestCandidate,
		BestFitness:              sanitizePersistedFitness(state.BestFitness),
		BestValidationCandidate:  state.BestValidationCandidate,
		BestValidationFitness:    sanitizePersistedFitness(state.BestValidationFitness),
		PopulationHistory:        state.PopulationHistory,
		ReflectionHistory:        state.ReflectionHistory,
		ConvergenceStatus:        state.ConvergenceStatus,
		StartTime:                state.StartTime,
		LastImprovement:          state.LastImprovement,
		LastValidatedGeneration:  state.LastValidatedGeneration,
		ExecutionTraces:          state.ExecutionTraces,
		CandidateMetrics:         state.CandidateMetrics,
		MultiObjectiveFitnessMap: state.MultiObjectiveFitnessMap,
		ValidationFrontier:       state.ValidationFrontier,
		ValidationCoverage:       state.ValidationCoverage,
		MergeInvocations:         state.MergeInvocations,
		PerformedMerges:          state.PerformedMerges,
		MetricCalls:              state.MetricCalls,
		StopReason:               state.StopReason,
		StopMetadata:             state.StopMetadata,
		ParetoArchive:            state.ParetoArchive,
		ArchiveFitnessMap:        state.ArchiveFitnessMap,
		MaxArchiveSize:           state.MaxArchiveSize,
	}

	return sanitized
}

func sanitizePersistedFitness(value float64) float64 {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return 0
	}
	return value
}
