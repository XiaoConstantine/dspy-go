package optimizers

import "context"

func (g *GEPA) setLatestEvaluationAdapter(adapter *gepaEvaluationAdapter) {
	if g == nil {
		return
	}

	g.evaluationAdapterMu.Lock()
	defer g.evaluationAdapterMu.Unlock()
	g.latestEvaluationAdapter = adapter
}

func (g *GEPA) getLatestEvaluationAdapter() *gepaEvaluationAdapter {
	if g == nil {
		return nil
	}

	g.evaluationAdapterMu.RLock()
	defer g.evaluationAdapterMu.RUnlock()
	return g.latestEvaluationAdapter
}

func (g *GEPA) acceptMutationProposal(ctx context.Context, baseline, proposed *GEPACandidate) *GEPACandidate {
	if baseline == nil || proposed == nil {
		if proposed != nil {
			return proposed
		}
		return baseline
	}
	if baseline == proposed {
		return baseline
	}

	adapter := g.getLatestEvaluationAdapter()
	if adapter == nil {
		return proposed
	}

	baselineEvaluation := g.cachedOrEvaluateCandidate(ctx, baseline, adapter)
	proposedEvaluation := g.evaluateCandidateWithAdapter(ctx, proposed, adapter)
	if proposedEvaluation == nil {
		return baseline
	}

	if baselineEvaluation != nil && totalScoreFromEvaluation(proposedEvaluation) <= totalScoreFromEvaluation(baselineEvaluation) {
		return baseline
	}

	proposed.Fitness = proposedEvaluation.AverageScore
	proposed.Metadata = mergeCandidateMetadata(map[string]interface{}{
		"proposal_accepted":          true,
		"proposal_baseline_total":    totalScoreFromEvaluation(baselineEvaluation),
		"proposal_candidate_total":   totalScoreFromEvaluation(proposedEvaluation),
		"proposal_baseline_average":  averageScoreFromEvaluation(baselineEvaluation),
		"proposal_candidate_average": proposedEvaluation.AverageScore,
	}, proposed.Metadata)

	return proposed
}

func (g *GEPA) cachedOrEvaluateCandidate(ctx context.Context, candidate *GEPACandidate, adapter *gepaEvaluationAdapter) *gepaCandidateEvaluation {
	if candidate == nil {
		return nil
	}

	if g != nil && g.state != nil {
		if evaluation := g.state.GetCandidateEvaluation(candidate.ID); evaluation != nil {
			return evaluation
		}
	}

	return g.evaluateCandidateWithAdapter(ctx, candidate, adapter)
}

func totalScoreFromEvaluation(evaluation *gepaCandidateEvaluation) float64 {
	if evaluation == nil {
		return 0.0
	}

	return evaluation.TotalScore
}

func averageScoreFromEvaluation(evaluation *gepaCandidateEvaluation) float64 {
	if evaluation == nil {
		return 0.0
	}

	return evaluation.AverageScore
}
