package optimizers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

const (
	maxProposalStrengths   = 3
	maxProposalWeaknesses  = 4
	maxProposalSuggestions = 4
)

func cloneReflectionResult(reflection *ReflectionResult) *ReflectionResult {
	if reflection == nil {
		return nil
	}

	return &ReflectionResult{
		CandidateID:     reflection.CandidateID,
		Strengths:       append([]string(nil), reflection.Strengths...),
		Weaknesses:      append([]string(nil), reflection.Weaknesses...),
		Suggestions:     append([]string(nil), reflection.Suggestions...),
		ConfidenceScore: reflection.ConfidenceScore,
		Timestamp:       reflection.Timestamp,
		ReflectionDepth: reflection.ReflectionDepth,
	}
}

func (g *GEPA) reflectionGuidedMutation(ctx context.Context, candidate *GEPACandidate) *GEPACandidate {
	sourceCandidateID, reflection, evaluation := g.resolveProposalGuidance(candidate)
	if reflection == nil || !hasReflectionGuidance(reflection) {
		return nil
	}

	prompt := g.buildReflectionMutationPrompt(candidate, sourceCandidateID, reflection, evaluation)
	core.RecordModelCall(ctx, g.generationLLM)
	response, err := g.generationLLM.Generate(ctx, prompt)
	if err != nil {
		return nil
	}

	proposedInstruction := g.extractInstructionCandidate(response.Content)
	if proposedInstruction == "" || proposedInstruction == candidate.Instruction {
		return nil
	}

	metadata := mergeCandidateMetadata(map[string]interface{}{
		"mutation_type":          "reflection_guided",
		"parent_fitness":         candidate.Fitness,
		"guidance_candidate_id":  sourceCandidateID,
		"reflection_confidence":  reflection.ConfidenceScore,
		"reflection_depth":       reflection.ReflectionDepth,
		"proposal_evidence_used": evaluation != nil,
	}, candidate.Metadata)

	return &GEPACandidate{
		ID:             g.generateCandidateID(),
		ModuleName:     candidate.ModuleName,
		Instruction:    proposedInstruction,
		ComponentTexts: deriveCandidateComponentTexts(candidate, candidate.ModuleName, proposedInstruction),
		Generation:     candidate.Generation + 1,
		ParentIDs:      []string{candidate.ID},
		CreatedAt:      time.Now(),
		Metadata:       metadata,
	}
}

func (g *GEPA) resolveProposalGuidance(candidate *GEPACandidate) (string, *ReflectionResult, *gepaCandidateEvaluation) {
	if candidate == nil || g.state == nil {
		return "", nil, nil
	}

	if reflection := g.state.GetCandidateReflection(candidate.ID); reflection != nil {
		return candidate.ID, reflection, g.state.GetCandidateEvaluation(candidate.ID)
	}

	var bestReflection *ReflectionResult
	bestCandidateID := ""
	for _, parentID := range candidate.ParentIDs {
		if strings.TrimSpace(parentID) == "" {
			continue
		}

		reflection := g.state.GetCandidateReflection(parentID)
		if reflection == nil {
			continue
		}
		if bestReflection == nil || reflection.ConfidenceScore > bestReflection.ConfidenceScore {
			bestReflection = reflection
			bestCandidateID = parentID
		}
	}

	if bestReflection == nil {
		return "", nil, nil
	}

	return bestCandidateID, bestReflection, g.state.GetCandidateEvaluation(bestCandidateID)
}

func (g *GEPA) buildReflectionMutationPrompt(candidate *GEPACandidate, sourceCandidateID string, reflection *ReflectionResult, evaluation *gepaCandidateEvaluation) string {
	strengths := formatPromptList(reflection.Strengths, maxProposalStrengths)
	weaknesses := formatPromptList(reflection.Weaknesses, maxProposalWeaknesses)
	suggestions := formatPromptList(reflection.Suggestions, maxProposalSuggestions)
	caseEvidence := g.formatReflectionCaseEvidence(g.buildReflectionInput(evaluation))

	sourceLabel := sourceCandidateID
	if strings.TrimSpace(sourceLabel) == "" {
		sourceLabel = candidate.ID
	}

	return fmt.Sprintf(`You are improving a GEPA instruction using reflection-guided evidence.

CURRENT INSTRUCTION: "%s"
MODULE: %s
GUIDANCE SOURCE CANDIDATE: %s
CURRENT FITNESS: %.3f

REFLECTION STRENGTHS:
%s

REFLECTION WEAKNESSES:
%s

REFLECTION SUGGESTIONS:
%s

EXAMPLE-LEVEL EVIDENCE:
%s

Rewrite the instruction to preserve the useful strengths while directly addressing the weaknesses and suggestions.

Requirements:
1. Keep the same overall task and module intent.
2. Make the instruction more concrete, actionable, and failure-aware.
3. Do not mention reflection, evidence, scores, or candidates in the rewritten instruction.
4. Return only the rewritten instruction text.

REWRITTEN INSTRUCTION:`,
		candidate.Instruction,
		candidate.ModuleName,
		sourceLabel,
		candidate.Fitness,
		strengths,
		weaknesses,
		suggestions,
		caseEvidence)
}

func hasReflectionGuidance(reflection *ReflectionResult) bool {
	if reflection == nil {
		return false
	}

	return len(reflection.Weaknesses) > 0 || len(reflection.Suggestions) > 0
}

func formatPromptList(items []string, limit int) string {
	if len(items) == 0 {
		return "- none recorded"
	}

	if limit <= 0 || limit > len(items) {
		limit = len(items)
	}

	lines := make([]string, 0, limit)
	for _, item := range items[:limit] {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		lines = append(lines, "- "+item)
	}
	if len(lines) == 0 {
		return "- none recorded"
	}

	return strings.Join(lines, "\n")
}

func (g *GEPA) extractInstructionCandidate(content string) string {
	content = unwrapInstructionCodeFence(content)
	lines := strings.Split(content, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		upper := strings.ToUpper(line)
		switch {
		case strings.HasPrefix(upper, "REWRITTEN INSTRUCTION:"),
			strings.HasPrefix(upper, "IMPROVED INSTRUCTION:"),
			strings.HasPrefix(upper, "MUTATED INSTRUCTION:"),
			strings.HasPrefix(upper, "INSTRUCTION:"):
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				line = strings.TrimSpace(parts[1])
			}
		case strings.HasPrefix(upper, "ORIGINAL:"),
			strings.HasPrefix(upper, "STRENGTHS:"),
			strings.HasPrefix(upper, "WEAKNESSES:"),
			strings.HasPrefix(upper, "SUGGESTIONS:"),
			strings.HasPrefix(upper, "CONFIDENCE:"):
			continue
		}

		line = strings.TrimSpace(strings.TrimPrefix(line, "- "))
		line = stripLeadingNumberedMarker(line)

		line = strings.Trim(line, "`\"'")
		if len(line) > 10 {
			return line
		}
	}

	return ""
}

func unwrapInstructionCodeFence(content string) string {
	content = strings.TrimSpace(content)
	if !strings.HasPrefix(content, "```") {
		return content
	}

	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSpace(content)
	if strings.HasSuffix(content, "```") {
		content = strings.TrimSuffix(content, "```")
		content = strings.TrimSpace(content)
	}

	lines := strings.Split(content, "\n")
	if len(lines) <= 1 {
		return content
	}

	firstLine := strings.TrimSpace(lines[0])
	if isInstructionFenceLanguage(firstLine) {
		return strings.TrimSpace(strings.Join(lines[1:], "\n"))
	}

	return content
}

func isInstructionFenceLanguage(line string) bool {
	if line == "" {
		return true
	}
	for _, r := range line {
		if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '_' && r != '-' && r != '+' {
			return false
		}
	}
	return true
}

func stripLeadingNumberedMarker(line string) string {
	index := 0
	for index < len(line) && line[index] >= '0' && line[index] <= '9' {
		index++
	}
	if index == 0 || index >= len(line) || line[index] != '.' {
		return line
	}

	return strings.TrimSpace(line[index+1:])
}
