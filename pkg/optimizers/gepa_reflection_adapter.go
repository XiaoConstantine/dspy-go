package optimizers

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	maxReflectionWorstCases = 3
	maxReflectionBestCases  = 1
	maxReflectionValueChars = 220
)

type gepaReflectionCaseEvidence struct {
	InputSummary    string
	ExpectedSummary string
	OutputSummary   string
	ErrorSummary    string
	FeedbackSummary string
	FeedbackTarget  string
	Score           float64
}

// gepaReflectionInput adapts example-level candidate evaluation into a compact
// prompt-oriented structure for reflection. This keeps reflection grounded in
// concrete failed/successful cases without changing the outer optimization loop.
type gepaReflectionInput struct {
	TotalCases   int
	SuccessCount int
	FailureCount int
	AverageScore float64
	WorstCases   []gepaReflectionCaseEvidence
	BestCases    []gepaReflectionCaseEvidence
}

func (g *GEPA) buildReflectionInput(evaluation *gepaCandidateEvaluation) *gepaReflectionInput {
	if evaluation == nil || len(evaluation.Cases) == 0 {
		return nil
	}

	result := &gepaReflectionInput{
		TotalCases:   len(evaluation.Cases),
		AverageScore: evaluation.AverageScore,
	}

	type rankedCase struct {
		evidence gepaReflectionCaseEvidence
		errRank  int
	}

	ranked := make([]rankedCase, 0, len(evaluation.Cases))
	for _, evalCase := range evaluation.Cases {
		evidence := gepaReflectionCaseEvidence{
			InputSummary:    summarizeReflectionMap(evalCase.Example.Inputs),
			ExpectedSummary: summarizeReflectionMap(evalCase.Example.Outputs),
			OutputSummary:   summarizeReflectionMap(evalCase.Outputs),
			FeedbackSummary: truncateEvidence(evalCase.Feedback, maxReflectionValueChars),
			FeedbackTarget:  truncateEvidence(evalCase.FeedbackTarget, maxReflectionValueChars),
			Score:           evalCase.Score,
		}
		if evalCase.Err != nil {
			evidence.ErrorSummary = truncateEvidence(evalCase.Err.Error(), maxReflectionValueChars)
			result.FailureCount++
		} else {
			evidence.ErrorSummary = "none"
			result.SuccessCount++
		}

		errRank := 0
		if evalCase.Err != nil {
			errRank = 1
		}
		ranked = append(ranked, rankedCase{
			evidence: evidence,
			errRank:  errRank,
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].errRank != ranked[j].errRank {
			return ranked[i].errRank > ranked[j].errRank
		}
		if ranked[i].evidence.Score != ranked[j].evidence.Score {
			return ranked[i].evidence.Score < ranked[j].evidence.Score
		}
		return ranked[i].evidence.InputSummary < ranked[j].evidence.InputSummary
	})

	for i := 0; i < len(ranked) && i < maxReflectionWorstCases; i++ {
		result.WorstCases = append(result.WorstCases, ranked[i].evidence)
	}

	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].evidence.Score != ranked[j].evidence.Score {
			return ranked[i].evidence.Score > ranked[j].evidence.Score
		}
		return ranked[i].evidence.InputSummary < ranked[j].evidence.InputSummary
	})

	for _, rankedCase := range ranked {
		if rankedCase.errRank != 0 {
			continue
		}
		result.BestCases = append(result.BestCases, rankedCase.evidence)
		if len(result.BestCases) >= maxReflectionBestCases {
			break
		}
	}

	return result
}

func (g *GEPA) formatReflectionCaseEvidence(input *gepaReflectionInput) string {
	if input == nil || input.TotalCases == 0 {
		return "none recorded"
	}

	var builder strings.Builder
	fmt.Fprintf(&builder, "- Cases Evaluated: %d\n", input.TotalCases)
	fmt.Fprintf(&builder, "- Successful Cases: %d\n", input.SuccessCount)
	fmt.Fprintf(&builder, "- Error Cases: %d\n", input.FailureCount)
	fmt.Fprintf(&builder, "- Average Case Score: %.3f\n", input.AverageScore)

	if len(input.WorstCases) > 0 {
		builder.WriteString("\nWorst Cases:\n")
		for i, evidence := range input.WorstCases {
			fmt.Fprintf(&builder,
				"%d. Inputs: %s\n   Expected: %s\n   Actual: %s\n   Score: %.3f\n   Error: %s\n",
				i+1,
				evidence.InputSummary,
				evidence.ExpectedSummary,
				evidence.OutputSummary,
				evidence.Score,
				evidence.ErrorSummary,
			)
			if evidence.FeedbackTarget != "" {
				fmt.Fprintf(&builder, "   Feedback Target: %s\n", evidence.FeedbackTarget)
			}
			if evidence.FeedbackSummary != "" {
				fmt.Fprintf(&builder, "   Feedback: %s\n", evidence.FeedbackSummary)
			}
		}
	}

	if len(input.BestCases) > 0 {
		builder.WriteString("\nRepresentative Successes:\n")
		for i, evidence := range input.BestCases {
			fmt.Fprintf(&builder,
				"%d. Inputs: %s\n   Expected: %s\n   Actual: %s\n   Score: %.3f\n",
				i+1,
				evidence.InputSummary,
				evidence.ExpectedSummary,
				evidence.OutputSummary,
				evidence.Score,
			)
			if evidence.FeedbackTarget != "" {
				fmt.Fprintf(&builder, "   Feedback Target: %s\n", evidence.FeedbackTarget)
			}
			if evidence.FeedbackSummary != "" {
				fmt.Fprintf(&builder, "   Feedback: %s\n", evidence.FeedbackSummary)
			}
		}
	}

	return strings.TrimSpace(builder.String())
}

func summarizeReflectionMap(values map[string]interface{}) string {
	if len(values) == 0 {
		return "{}"
	}

	content, err := json.Marshal(values)
	if err != nil {
		return truncateEvidence(fmt.Sprintf("%v", values), maxReflectionValueChars)
	}

	return truncateEvidence(string(content), maxReflectionValueChars)
}
