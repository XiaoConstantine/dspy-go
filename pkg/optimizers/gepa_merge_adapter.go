package optimizers

import (
	"context"
	"fmt"
	"strings"
	"time"
)

type gepaAncestorMergeChoice struct {
	partner               *GEPACandidate
	ancestor              *GEPACandidate
	adoptedComponents     []string
	conflictingComponents []string
	coverage              int
	mergeKey              string
}

func (g *GEPA) tryAncestorMergeProposal(ctx context.Context, population *Population, source *GEPACandidate, nextGeneration int) *GEPACandidate {
	if g == nil || g.state == nil || !g.state.MergeBudgetAvailable(g.config.MaxMergeInvocations) {
		return nil
	}

	choice := g.selectAncestorMergePartner(population, source)
	if choice == nil {
		return nil
	}
	if !g.state.RecordMergeInvocation(choice.mergeKey, g.config.MaxMergeInvocations) {
		return nil
	}

	merged := g.buildAncestorMergedCandidate(source, choice, nextGeneration)
	if merged == nil {
		return nil
	}

	accepted := g.acceptMergeProposal(ctx, source, choice.partner, merged)
	if accepted == nil || accepted == source {
		return nil
	}
	if accepted.Generation < nextGeneration {
		accepted.Generation = nextGeneration
	}

	return accepted
}

func (g *GEPA) selectAncestorMergePartner(population *Population, source *GEPACandidate) *gepaAncestorMergeChoice {
	if g == nil || g.state == nil || population == nil || source == nil {
		return nil
	}

	_, coverage := g.state.ValidationFrontierSnapshot()
	if len(coverage) == 0 || coverage[source.ID] == 0 {
		return nil
	}

	var best *gepaAncestorMergeChoice
	for _, candidate := range population.Candidates {
		if candidate == nil || candidate.ID == source.ID || coverage[candidate.ID] == 0 {
			continue
		}

		ancestor := g.findClosestCommonAncestor(source, candidate)
		if ancestor == nil || ancestor.ID == source.ID || ancestor.ID == candidate.ID {
			continue
		}

		adopted, conflicts := mergeableAncestorComponents(source, candidate, ancestor)
		if len(adopted) == 0 {
			continue
		}

		choice := &gepaAncestorMergeChoice{
			partner:               candidate,
			ancestor:              ancestor,
			adoptedComponents:     adopted,
			conflictingComponents: conflicts,
			coverage:              coverage[candidate.ID],
			mergeKey:              buildAncestorMergeAttemptKey(source.ID, candidate.ID, ancestor.ID, adopted),
		}
		if g.state.HasRecordedMerge(choice.mergeKey) {
			continue
		}
		if isBetterAncestorMergeChoice(choice, best) {
			best = choice
		}
	}

	return best
}

func isBetterAncestorMergeChoice(candidate, best *gepaAncestorMergeChoice) bool {
	switch {
	case candidate == nil:
		return false
	case best == nil:
		return true
	case len(candidate.adoptedComponents) != len(best.adoptedComponents):
		return len(candidate.adoptedComponents) > len(best.adoptedComponents)
	case candidate.coverage != best.coverage:
		return candidate.coverage > best.coverage
	case len(candidate.conflictingComponents) != len(best.conflictingComponents):
		return len(candidate.conflictingComponents) < len(best.conflictingComponents)
	case candidate.ancestor != nil && best.ancestor != nil && candidate.ancestor.ID != best.ancestor.ID:
		return candidate.ancestor.ID < best.ancestor.ID
	case candidate.partner != nil && best.partner != nil:
		return candidate.partner.ID < best.partner.ID
	default:
		return false
	}
}

func (g *GEPA) buildAncestorMergedCandidate(source *GEPACandidate, choice *gepaAncestorMergeChoice, nextGeneration int) *GEPACandidate {
	if source == nil || choice == nil || choice.partner == nil || choice.ancestor == nil || len(choice.adoptedComponents) == 0 {
		return nil
	}
	if choice.mergeKey == "" {
		choice.mergeKey = buildAncestorMergeAttemptKey(source.ID, choice.partner.ID, choice.ancestor.ID, choice.adoptedComponents)
	}

	componentTexts := cloneCandidateComponentTexts(source)
	if len(componentTexts) == 0 {
		return nil
	}

	for _, moduleName := range choice.adoptedComponents {
		componentTexts[moduleName] = candidateInstructionForModule(choice.partner, moduleName)
	}

	focusedModule := strings.TrimSpace(source.ModuleName)
	if focusedModule == "" || !containsString(choice.adoptedComponents, focusedModule) {
		focusedModule = choice.adoptedComponents[0]
	}

	return &GEPACandidate{
		ID:             g.generateCandidateID(),
		ModuleName:     focusedModule,
		Instruction:    componentTexts[focusedModule],
		ComponentTexts: componentTexts,
		Generation:     nextGeneration,
		Fitness:        source.Fitness,
		ParentIDs:      []string{source.ID, choice.partner.ID},
		CreatedAt:      time.Now(),
		Metadata: mergeCandidateMetadata(map[string]interface{}{
			"proposal_type":                "ancestor_merge",
			"merge_partner_id":             choice.partner.ID,
			"merge_common_ancestor_id":     choice.ancestor.ID,
			"merge_attempt_key":            choice.mergeKey,
			"merge_adopted_components":     append([]string(nil), choice.adoptedComponents...),
			"merge_conflicting_components": append([]string(nil), choice.conflictingComponents...),
		}, source.Metadata, choice.partner.Metadata),
	}
}

func buildAncestorMergeAttemptKey(sourceID, partnerID, ancestorID string, adoptedComponents []string) string {
	components := append([]string(nil), adoptedComponents...)
	if len(components) > 0 {
		components = componentModuleNames(stringSliceSet(components))
	}
	return fmt.Sprintf("%s|%s|%s|%s", sourceID, partnerID, ancestorID, strings.Join(components, ","))
}

func stringSliceSet(values []string) map[string]string {
	if len(values) == 0 {
		return nil
	}

	set := make(map[string]string, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		set[value] = value
	}
	return set
}

func mergeableAncestorComponents(source, partner, ancestor *GEPACandidate) ([]string, []string) {
	moduleNames := candidateUnionComponentNames(source, partner, ancestor)
	if len(moduleNames) == 0 {
		return nil, nil
	}

	adopted := make([]string, 0, len(moduleNames))
	conflicts := make([]string, 0)
	for _, moduleName := range moduleNames {
		ancestorInstruction := candidateInstructionForModule(ancestor, moduleName)
		sourceInstruction := candidateInstructionForModule(source, moduleName)
		partnerInstruction := candidateInstructionForModule(partner, moduleName)

		sourceChanged := sourceInstruction != ancestorInstruction
		partnerChanged := partnerInstruction != ancestorInstruction

		switch {
		case !sourceChanged && partnerChanged:
			adopted = append(adopted, moduleName)
		case sourceChanged && partnerChanged && sourceInstruction != partnerInstruction:
			conflicts = append(conflicts, moduleName)
		}
	}

	return adopted, conflicts
}

func candidateUnionComponentNames(candidates ...*GEPACandidate) []string {
	componentTexts := make(map[string]string)
	for _, candidate := range candidates {
		for moduleName := range cloneCandidateComponentTexts(candidate) {
			componentTexts[moduleName] = ""
		}
	}
	return componentModuleNames(componentTexts)
}

func (g *GEPA) findClosestCommonAncestor(left, right *GEPACandidate) *GEPACandidate {
	if g == nil || left == nil || right == nil {
		return nil
	}

	leftDepths := g.candidateAncestorDepths(left)
	rightDepths := g.candidateAncestorDepths(right)

	bestDepth := -1
	var best *GEPACandidate
	for candidateID, leftDepth := range leftDepths {
		rightDepth, exists := rightDepths[candidateID]
		if !exists {
			continue
		}

		ancestor := g.findCandidateInHistory(candidateID)
		if ancestor == nil {
			continue
		}

		totalDepth := leftDepth + rightDepth
		if best == nil || totalDepth < bestDepth || (totalDepth == bestDepth && ancestor.ID < best.ID) {
			best = ancestor
			bestDepth = totalDepth
		}
	}

	return best
}

func (g *GEPA) candidateAncestorDepths(candidate *GEPACandidate) map[string]int {
	if g == nil || candidate == nil {
		return nil
	}

	type queuedAncestor struct {
		candidate *GEPACandidate
		depth     int
	}

	depths := make(map[string]int)
	queue := []queuedAncestor{{candidate: candidate, depth: 0}}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if current.candidate == nil || strings.TrimSpace(current.candidate.ID) == "" {
			continue
		}
		if existingDepth, exists := depths[current.candidate.ID]; exists && existingDepth <= current.depth {
			continue
		}
		depths[current.candidate.ID] = current.depth

		for _, parentID := range current.candidate.ParentIDs {
			parent := g.findCandidateInHistory(parentID)
			if parent == nil {
				continue
			}
			queue = append(queue, queuedAncestor{candidate: parent, depth: current.depth + 1})
		}
	}

	return depths
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
