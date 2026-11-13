package optimizers

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// Evaluate whole batches or none: a partial validation must never become a
// frontier entry. Cached candidate/example pairs do not consume metric budget.
func (g *GEPA) evaluateWithinBudget(ctx context.Context, candidate *GEPACandidate, adapter *gepaEvaluationAdapter) *gepaCandidateEvaluation {
	if ctx.Err() != nil {
		return nil
	}
	missing := make(map[string]struct{})
	for _, example := range adapter.batch {
		key := evaluationCaseCacheKey(candidate, example)
		if g.state.GetEvaluationCaseCache(key) == nil {
			missing[key] = struct{}{}
		}
	}
	if g.config.MaxMetricCalls > 0 && len(missing) > g.config.MaxMetricCalls-g.state.MetricCallCount() {
		g.recordStopDecision(&GEPAStopDecision{
			Reason: gepaStopReasonMetricBudget,
			Metadata: map[string]any{
				"metric_calls": g.state.MetricCallCount(), "max_metric_calls": g.config.MaxMetricCalls,
				"next_evaluation_calls": len(missing),
			},
		})
		return nil
	}
	return g.evaluateCandidateWithAdapter(ctx, candidate, adapter)
}

func (g *GEPA) validateRetainedCandidate(ctx context.Context, candidate *GEPACandidate, adapter *gepaEvaluationAdapter) bool {
	evaluation := g.state.GetCandidateValidationEvaluation(candidate.ID)
	if !evaluationMatchesBatch(evaluation, candidate, adapter) {
		evaluation = g.evaluateWithinBudget(ctx, g.copyCandidate(candidate), adapter)
		if evaluation == nil || ctx.Err() != nil {
			return false
		}
	}
	candidate.Fitness = evaluation.AverageScore
	candidate.Metadata = mergeCandidateMetadata(map[string]any{"validation_score": evaluation.AverageScore}, candidate.Metadata)
	evaluations := make(map[string]*gepaCandidateEvaluation)
	for _, retained := range g.getCurrentPopulation().Candidates {
		if cached := g.state.GetCandidateValidationEvaluation(retained.ID); cached != nil {
			evaluations[retained.ID] = cached
		}
	}
	evaluations[candidate.ID] = evaluation
	g.state.SetCandidateValidationEvaluations(evaluations)
	frontier, coverage := buildValidationFrontier(evaluations)
	g.state.SetValidationFrontier(frontier, coverage)
	g.setBestValidationCandidate(candidate, evaluation.AverageScore)
	g.updateBestCandidate(candidate)
	g.state.LastValidatedGeneration = g.state.CurrentGeneration
	g.recomputePopulationBest(g.getCurrentPopulation())
	return true
}

func (g *GEPA) evaluationIsPerfect(evaluation *gepaCandidateEvaluation) bool {
	if !g.config.SkipPerfectScore || evaluation == nil || len(evaluation.Cases) == 0 {
		return false
	}
	for _, evalCase := range evaluation.Cases {
		if evalCase.Err != nil || math.IsNaN(evalCase.Score) || evalCase.Score < g.config.PerfectScore {
			return false
		}
	}
	return true
}

func (g *GEPA) validationIsPerfect() bool {
	best := g.state.BestValidationCandidate
	return best != nil && g.evaluationIsPerfect(g.state.GetCandidateValidationEvaluation(best.ID))
}

func (g *GEPA) runCandidateIteration(ctx context.Context, adapter, validationAdapter *gepaEvaluationAdapter) bool {
	current := g.getCurrentPopulation()
	next := &Population{Generation: g.state.CurrentGeneration}
	for _, candidate := range current.Candidates {
		next.Candidates = append(next.Candidates, g.copyCandidate(candidate))
	}
	defer func() {
		g.recomputePopulationBest(next)
		g.state.PopulationHistory = append(g.state.PopulationHistory, next)
	}()

	parent := g.selectCandidateForUpdate(next)
	// Evaluate a copy so rotating training scores never overwrite validation
	// fitness on a retained parent or an earlier population snapshot.
	parentEvaluation := g.evaluateWithinBudget(ctx, g.copyCandidate(parent), adapter)
	if parentEvaluation == nil {
		return false
	}
	g.state.UpsertCandidateEvaluation(parent.ID, parentEvaluation)
	if g.evaluationIsPerfect(parentEvaluation) || g.rng.Float64() >= g.config.MutationRate {
		return true
	}

	selection := g.selectComponentsForUpdate(parent)
	g.applyComponentSelectionState(parent, selection.nextCursor)
	var child *GEPACandidate
	// Retain ancestor merging, but accept it on exactly the same current batch
	// as reflective proposals, without the older stratified acceptance shortcut.
	if g.state.MergeBudgetAvailable(g.config.MaxMergeInvocations) {
		if choice := g.selectAncestorMergePartner(next, parent); choice != nil && g.state.RecordMergeInvocation(choice.mergeKey, g.config.MaxMergeInvocations) {
			child = g.buildAncestorMergedCandidate(parent, choice, g.state.CurrentGeneration)
		}
	}
	if child == nil {
		child = g.proposeFromMinibatch(ctx, parent, selection.modules, parentEvaluation)
	}
	if child == nil {
		return true
	}
	childKey := candidateComponentCacheKey(child)
	for _, retained := range next.Candidates {
		if candidateComponentCacheKey(retained) == childKey {
			// A repeated proposal adds neither a new candidate nor evidence
			// of stagnation, regardless of which parent proposed it.
			return true
		}
	}
	childEvaluation := g.evaluateWithinBudget(ctx, child, adapter)
	if childEvaluation == nil {
		return false
	}
	if !(childEvaluation.TotalScore > parentEvaluation.TotalScore) {
		g.state.ConvergenceStatus.StagnationCount++
		return true
	}
	child.Metadata = mergeCandidateMetadata(map[string]any{
		"proposal_accepted":       true,
		"proposal_baseline_total": parentEvaluation.TotalScore, "proposal_candidate_total": childEvaluation.TotalScore,
	}, child.Metadata)
	previousBest := g.state.BestValidationFitness
	if !g.validateRetainedCandidate(ctx, child, validationAdapter) {
		return false
	}
	g.state.UpsertCandidateEvaluation(child.ID, childEvaluation)
	next.Candidates = append(next.Candidates, child)
	if g.state.BestValidationFitness > previousBest {
		g.state.ConvergenceStatus.StagnationCount = 0
	} else {
		g.state.ConvergenceStatus.StagnationCount++
	}
	return true
}

// A proposal is one reflection-model call over this parent's current minibatch.
// ComponentSelection=all requests a JSON map in the same call, not one call per
// module. Failure, malformed output, and unchanged instructions skip the step.
func (g *GEPA) proposeFromMinibatch(ctx context.Context, parent *GEPACandidate, moduleNames []string, evaluation *gepaCandidateEvaluation) *GEPACandidate {
	if len(moduleNames) == 0 {
		return nil
	}
	focused := g.prepareCandidateForComponentUpdate(parent, moduleNames[0])
	focused.Fitness = evaluation.AverageScore
	prompt := g.buildReflectionMutationPrompt(focused, parent.ID, &ReflectionResult{}, evaluation)
	if len(moduleNames) > 1 {
		instructions := make(map[string]string, len(moduleNames))
		for _, name := range moduleNames {
			instructions[name] = candidateInstructionForModule(parent, name)
		}
		rendered, err := json.Marshal(instructions)
		if err != nil {
			return nil
		}
		prompt += fmt.Sprintf("\n\nRevise these module instructions together: %s\nReturn only a JSON object with these module names as keys and the complete rewritten instructions as string values, instead of a single instruction.", rendered)
	}
	core.RecordModelCall(ctx, g.reflectionLLM)
	response, err := g.reflectionLLM.Generate(ctx, prompt)
	if err != nil || response == nil {
		logging.GetLogger().Warn(ctx, "Skipping GEPA proposal: reflection failed: %v", err)
		return nil
	}
	texts := cloneCandidateComponentTexts(parent)
	if len(moduleNames) == 1 {
		// Preserve multiline instructions; the old variation parser kept only
		// the first nonempty line and silently discarded the rest of a proposal.
		instruction := unwrapInstructionCodeFence(response.Content)
		if label, body, ok := strings.Cut(instruction, ":"); ok {
			switch strings.ToUpper(strings.Trim(strings.TrimSpace(label), "*")) {
			case "REWRITTEN INSTRUCTION", "IMPROVED INSTRUCTION", "INSTRUCTION":
				instruction = strings.TrimSpace(strings.TrimPrefix(body, "**"))
			}
		}
		// Keep legacy single-item proposal normalization (also used for
		// numeric artifacts), but preserve the steps of multiline instructions.
		if !strings.Contains(instruction, "\n") {
			instruction = stripLeadingNumberedMarker(instruction)
		}
		// Remove only matching enclosing quotes, not a quoted final word.
		if len(instruction) >= 2 && (instruction[0] == '"' || instruction[0] == '\'') && instruction[len(instruction)-1] == instruction[0] {
			instruction = strings.TrimSpace(instruction[1 : len(instruction)-1])
		}
		if instruction == "" {
			return nil
		}
		texts[moduleNames[0]] = instruction
	} else {
		content := unwrapInstructionCodeFence(response.Content)
		start, end := strings.Index(content, "{"), strings.LastIndex(content, "}")
		if start < 0 || end <= start {
			return nil
		}
		var proposed map[string]json.RawMessage
		if json.Unmarshal([]byte(content[start:end+1]), &proposed) != nil {
			return nil
		}
		for _, name := range moduleNames {
			var instruction string
			if json.Unmarshal(proposed[name], &instruction) != nil || strings.TrimSpace(instruction) == "" {
				return nil
			}
			texts[name] = strings.TrimSpace(instruction)
		}
	}
	child := g.candidateFromSeed(parent, texts[moduleNames[0]], g.state.CurrentGeneration, []string{parent.ID}, map[string]any{
		"mutation_type": "reflection_guided", "proposal_evidence_used": true,
	})
	child.ModuleName, child.Instruction, child.ComponentTexts = moduleNames[0], texts[moduleNames[0]], texts
	if candidateComponentCacheKey(child) == candidateComponentCacheKey(parent) {
		return nil
	}
	g.state.ReflectionHistory = append(g.state.ReflectionHistory, &ReflectionResult{
		CandidateID: parent.ID, Suggestions: []string{response.Content}, Timestamp: time.Now(),
	})
	g.ensureCandidateMetrics(child.ID)
	return child
}
