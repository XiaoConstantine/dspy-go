package react

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
)

// SelfReflectorACEAdapter bridges the existing SelfReflector with ACE's Adapter interface.
type SelfReflectorACEAdapter struct {
	reflector *SelfReflector
}

// NewSelfReflectorACEAdapter creates an adapter that extracts insights from SelfReflector.
func NewSelfReflectorACEAdapter(reflector *SelfReflector) *SelfReflectorACEAdapter {
	return &SelfReflectorACEAdapter{reflector: reflector}
}

// Extract implements ace.Adapter by converting SelfReflector's reflections to InsightCandidates.
func (a *SelfReflectorACEAdapter) Extract(ctx context.Context) ([]ace.InsightCandidate, error) {
	if a.reflector == nil {
		return nil, nil
	}

	reflections := a.reflector.GetTopReflections(10)
	if len(reflections) == 0 {
		return nil, nil
	}

	candidates := make([]ace.InsightCandidate, 0, len(reflections))
	for _, r := range reflections {
		category := mapReflectionTypeToCategory(r.Type)
		candidates = append(candidates, ace.InsightCandidate{
			Content:    r.Insight,
			Category:   category,
			Confidence: r.Confidence,
			Source:     "self_reflector",
		})
	}

	return candidates, nil
}

func mapReflectionTypeToCategory(rt ReflectionType) string {
	switch rt {
	case ReflectionTypeError:
		return "mistakes"
	case ReflectionTypeStrategy:
		return "strategies"
	case ReflectionTypeLearning:
		return "patterns"
	case ReflectionTypePerformance:
		return "patterns"
	default:
		return "general"
	}
}
