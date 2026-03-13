package optimizers

import (
	"context"
	"math"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

const (
	gepaStopReasonConverged       = "converged"
	gepaStopReasonMetricBudget    = "metric_budget_exhausted"
	gepaStopReasonScoreThreshold  = "score_threshold_reached"
	gepaStopReasonMaxRuntime      = "max_runtime_exceeded"
	gepaStopReasonCustomStopper   = "custom_stopper"
	gepaStopScoreSourceTraining   = "training"
	gepaStopScoreSourceValidation = "validation"
)

// GEPAStopDecision records why a GEPA run should stop.
type GEPAStopDecision struct {
	Reason   string                 `json:"reason"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// GEPAStopper allows callers to plug additional stopping logic into GEPA.
// Returning nil means the run should continue.
type GEPAStopper func(context.Context, *GEPA) *GEPAStopDecision

// ResetRunControls clears run-scoped stop/budget bookkeeping before a fresh
// Compile call. Merge dedup state is intentionally reset here too, because
// each Compile invocation is treated as an independent GEPA optimization run.
func (s *GEPAState) ResetRunControls(now time.Time) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.StartTime = now
	s.LastImprovement = now
	s.MetricCalls = 0
	s.StopReason = ""
	s.StopMetadata = nil
	s.MergeInvocations = 0
	s.PerformedMerges = make(map[string]bool)
	if s.ConvergenceStatus == nil {
		s.ConvergenceStatus = &ConvergenceStatus{}
	}
	s.ConvergenceStatus.StagnationCount = 0
	s.ConvergenceStatus.IsConverged = false
}

func (s *GEPAState) AddMetricCalls(count int) {
	if s == nil || count <= 0 {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.MetricCalls += count
}

func (s *GEPAState) MetricCallCount() int {
	if s == nil {
		return 0
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.MetricCalls
}

func (s *GEPAState) RecordStopDecision(decision *GEPAStopDecision) {
	if s == nil || decision == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.StopReason = strings.TrimSpace(decision.Reason)
	s.StopMetadata = cloneStringAnyMap(decision.Metadata)
}

func (g *GEPA) recordStopDecision(decision *GEPAStopDecision) {
	if g == nil || g.state == nil || decision == nil {
		return
	}

	g.state.RecordStopDecision(decision)
	logging.GetLogger().Info(context.Background(), "GEPA stopping: reason=%s metadata=%v", decision.Reason, decision.Metadata)
}

func (g *GEPA) evaluateStopDecision(ctx context.Context) *GEPAStopDecision {
	if g == nil || g.state == nil {
		return nil
	}

	if decision := g.metricBudgetStopDecision(); decision != nil {
		return decision
	}
	if decision := g.maxRuntimeStopDecision(); decision != nil {
		return decision
	}
	if decision := g.scoreThresholdStopDecision(); decision != nil {
		return decision
	}
	for _, stopper := range g.config.StopCallbacks {
		if stopper == nil {
			continue
		}
		if decision := stopper(ctx, g); decision != nil {
			if strings.TrimSpace(decision.Reason) == "" {
				decision.Reason = gepaStopReasonCustomStopper
			}
			return decision
		}
	}
	if g.hasConverged() {
		return &GEPAStopDecision{
			Reason: gepaStopReasonConverged,
			Metadata: map[string]interface{}{
				"generation":       g.state.CurrentGeneration,
				"stagnation_count": g.state.ConvergenceStatus.StagnationCount,
			},
		}
	}

	return nil
}

func (g *GEPA) metricBudgetStopDecision() *GEPAStopDecision {
	if g == nil || g.state == nil || g.config.MaxMetricCalls <= 0 {
		return nil
	}

	metricCalls := g.state.MetricCallCount()
	if metricCalls < g.config.MaxMetricCalls {
		return nil
	}

	return &GEPAStopDecision{
		Reason: gepaStopReasonMetricBudget,
		Metadata: map[string]interface{}{
			"metric_calls":     metricCalls,
			"max_metric_calls": g.config.MaxMetricCalls,
		},
	}
}

func (g *GEPA) maxRuntimeStopDecision() *GEPAStopDecision {
	if g == nil || g.state == nil || g.config.MaxRuntime <= 0 {
		return nil
	}

	elapsed := time.Since(g.state.StartTime)
	if elapsed < g.config.MaxRuntime {
		return nil
	}

	return &GEPAStopDecision{
		Reason: gepaStopReasonMaxRuntime,
		Metadata: map[string]interface{}{
			"elapsed":     elapsed.String(),
			"max_runtime": g.config.MaxRuntime.String(),
		},
	}
}

func (g *GEPA) scoreThresholdStopDecision() *GEPAStopDecision {
	// A non-positive threshold disables this stopper by config contract.
	if g == nil || g.state == nil || g.config.ScoreThreshold <= 0 {
		return nil
	}

	bestScore, scoreSource := g.bestScoreForStopping()
	if math.IsInf(bestScore, -1) || bestScore < g.config.ScoreThreshold {
		return nil
	}

	return &GEPAStopDecision{
		Reason: gepaStopReasonScoreThreshold,
		Metadata: map[string]interface{}{
			"best_score":      bestScore,
			"score_threshold": g.config.ScoreThreshold,
			"score_source":    scoreSource,
			"generation":      g.state.CurrentGeneration,
		},
	}
}

func (g *GEPA) bestScoreForStopping() (float64, string) {
	if g == nil || g.state == nil {
		return math.Inf(-1), ""
	}

	g.state.mu.RLock()
	defer g.state.mu.RUnlock()

	switch {
	case g.state.BestValidationCandidate != nil:
		return g.state.BestValidationFitness, gepaStopScoreSourceValidation
	case g.state.BestCandidate != nil:
		return g.state.BestFitness, gepaStopScoreSourceTraining
	default:
		return math.Inf(-1), ""
	}
}
