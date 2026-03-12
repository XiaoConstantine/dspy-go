package optimizers

import (
	"context"
	"sort"
)

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

func (g *GEPA) acceptCandidateProposal(ctx context.Context, baseline, proposed *GEPACandidate) *GEPACandidate {
	return g.acceptCandidateProposalWithAdapter(ctx, baseline, proposed, g.getLatestEvaluationAdapter(), nil)
}

func (g *GEPA) acceptCandidateProposalWithAdapter(ctx context.Context, baseline, proposed *GEPACandidate, adapter *gepaEvaluationAdapter, extraMetadata map[string]interface{}) *GEPACandidate {
	if baseline == nil || proposed == nil {
		if proposed != nil {
			return proposed
		}
		return baseline
	}
	if baseline == proposed {
		return baseline
	}

	if adapter == nil {
		return proposed
	}

	useLatestAdapterCache := adapter == g.getLatestEvaluationAdapter()

	baselineEvaluation := g.cachedOrEvaluateCandidate(ctx, baseline, adapter, useLatestAdapterCache)
	if useLatestAdapterCache && baselineEvaluation != nil && g != nil && g.state != nil {
		g.state.UpsertCandidateEvaluation(baseline.ID, baselineEvaluation)
	}

	g.ensureCandidateMetrics(proposed.ID)
	proposedEvaluation := g.evaluateCandidateWithAdapter(ctx, proposed, adapter)
	if proposedEvaluation == nil {
		return baseline
	}

	if baselineEvaluation != nil && totalScoreFromEvaluation(proposedEvaluation) <= totalScoreFromEvaluation(baselineEvaluation) {
		return baseline
	}

	proposed.Fitness = proposedEvaluation.AverageScore
	if useLatestAdapterCache && g != nil && g.state != nil {
		g.state.UpsertCandidateEvaluation(proposed.ID, proposedEvaluation)
	}
	metadata := map[string]interface{}{
		"proposal_accepted":          true,
		"proposal_baseline_total":    totalScoreFromEvaluation(baselineEvaluation),
		"proposal_candidate_total":   totalScoreFromEvaluation(proposedEvaluation),
		"proposal_baseline_average":  averageScoreFromEvaluation(baselineEvaluation),
		"proposal_candidate_average": proposedEvaluation.AverageScore,
	}
	for key, value := range extraMetadata {
		metadata[key] = value
	}
	proposed.Metadata = mergeCandidateMetadata(metadata, proposed.Metadata)

	return proposed
}

func (g *GEPA) acceptMutationProposal(ctx context.Context, baseline, proposed *GEPACandidate) *GEPACandidate {
	return g.acceptCandidateProposal(ctx, baseline, proposed)
}

func (g *GEPA) acceptMergeProposal(ctx context.Context, baseline, partner, proposed *GEPACandidate) *GEPACandidate {
	adapter, metadata := g.buildStratifiedMergeAcceptanceAdapter(ctx, baseline, partner)
	accepted := g.acceptCandidateProposalWithAdapter(ctx, baseline, proposed, adapter, metadata)
	if accepted == nil || accepted == baseline {
		return accepted
	}

	latestAdapter := g.getLatestEvaluationAdapter()
	if latestAdapter == nil || latestAdapter == adapter {
		return accepted
	}

	latestEvaluation := g.evaluateCandidateWithAdapter(ctx, accepted, latestAdapter)
	if latestEvaluation == nil {
		return accepted
	}

	accepted.Fitness = latestEvaluation.AverageScore
	if g != nil && g.state != nil {
		g.state.UpsertCandidateEvaluation(accepted.ID, latestEvaluation)
	}
	accepted.Metadata = mergeCandidateMetadata(map[string]interface{}{
		"merge_post_accept_full_batch_total":   totalScoreFromEvaluation(latestEvaluation),
		"merge_post_accept_full_batch_average": latestEvaluation.AverageScore,
	}, accepted.Metadata)

	return accepted
}

func (g *GEPA) buildStratifiedMergeAcceptanceAdapter(ctx context.Context, baseline, partner *GEPACandidate) (*gepaEvaluationAdapter, map[string]interface{}) {
	adapter := g.getLatestEvaluationAdapter()
	if adapter == nil || baseline == nil || partner == nil {
		return adapter, nil
	}

	baselineEvaluation := g.cachedOrEvaluateCandidate(ctx, baseline, adapter, true)
	partnerEvaluation := g.cachedOrEvaluateCandidate(ctx, partner, adapter, true)
	if baselineEvaluation == nil || partnerEvaluation == nil {
		return adapter, nil
	}

	caseIndexes, bucketCounts := stratifiedMergeAcceptanceCaseIndexes(baselineEvaluation, partnerEvaluation)
	if len(caseIndexes) == 0 {
		return adapter, map[string]interface{}{
			"merge_acceptance_mode": "full_batch",
		}
	}

	metadata := map[string]interface{}{
		"merge_acceptance_mode":                 "stratified",
		"merge_acceptance_case_count":           len(caseIndexes),
		"merge_acceptance_source_better_count":  bucketCounts.sourceBetter,
		"merge_acceptance_partner_better_count": bucketCounts.partnerBetter,
		"merge_acceptance_tied_count":           bucketCounts.tied,
	}

	return adapter.subset(caseIndexes), metadata
}

func (g *GEPA) cachedOrEvaluateCandidate(ctx context.Context, candidate *GEPACandidate, adapter *gepaEvaluationAdapter, allowStateCache bool) *gepaCandidateEvaluation {
	if candidate == nil {
		return nil
	}

	if allowStateCache && g != nil && g.state != nil {
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

type mergeAcceptanceBucketCounts struct {
	sourceBetter  int
	partnerBetter int
	tied          int
}

func stratifiedMergeAcceptanceCaseIndexes(sourceEvaluation, partnerEvaluation *gepaCandidateEvaluation) ([]int, mergeAcceptanceBucketCounts) {
	if sourceEvaluation == nil || partnerEvaluation == nil {
		return nil, mergeAcceptanceBucketCounts{}
	}

	caseCount := minInt(len(sourceEvaluation.Cases), len(partnerEvaluation.Cases))
	if caseCount == 0 {
		return nil, mergeAcceptanceBucketCounts{}
	}

	sourceBetter := make([]int, 0, caseCount)
	partnerBetter := make([]int, 0, caseCount)
	tied := make([]int, 0, caseCount)
	for caseIndex := 0; caseIndex < caseCount; caseIndex++ {
		sourceScore := sourceEvaluation.Cases[caseIndex].Score
		partnerScore := partnerEvaluation.Cases[caseIndex].Score
		switch {
		case sourceScore > partnerScore:
			sourceBetter = append(sourceBetter, caseIndex)
		case partnerScore > sourceScore:
			partnerBetter = append(partnerBetter, caseIndex)
		default:
			tied = append(tied, caseIndex)
		}
	}

	buckets := [][]int{sourceBetter, partnerBetter, tied}
	nonEmptyBuckets := 0
	for _, bucket := range buckets {
		if len(bucket) > 0 {
			nonEmptyBuckets++
		}
	}
	if nonEmptyBuckets == 0 {
		return nil, mergeAcceptanceBucketCounts{}
	}

	targetSize := minInt(caseCount, nonEmptyBuckets*2)
	selected := make([]int, 0, targetSize)
	for len(selected) < targetSize {
		progressed := false
		for bucketIndex := range buckets {
			if len(buckets[bucketIndex]) == 0 {
				continue
			}
			selected = append(selected, buckets[bucketIndex][0])
			buckets[bucketIndex] = buckets[bucketIndex][1:]
			progressed = true
			if len(selected) == targetSize {
				break
			}
		}
		if !progressed {
			break
		}
	}
	sort.Ints(selected)

	return selected, mergeAcceptanceBucketCounts{
		sourceBetter:  len(sourceBetter),
		partnerBetter: len(partnerBetter),
		tied:          len(tied),
	}
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}
