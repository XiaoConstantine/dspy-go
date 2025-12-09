package react

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
)

// DefaultTopReflectionsLimit is the default number of reflections to extract.
const DefaultTopReflectionsLimit = 10

// SelfReflectorACEAdapter bridges the existing SelfReflector with ACE's Adapter interface.
type SelfReflectorACEAdapter struct {
	reflector *SelfReflector
	limit     int
}

// SelfReflectorACEAdapterOption configures a SelfReflectorACEAdapter.
type SelfReflectorACEAdapterOption func(*SelfReflectorACEAdapter)

// WithReflectionsLimit sets the maximum number of reflections to extract.
func WithReflectionsLimit(limit int) SelfReflectorACEAdapterOption {
	return func(a *SelfReflectorACEAdapter) {
		a.limit = limit
	}
}

// NewSelfReflectorACEAdapter creates an adapter that extracts insights from SelfReflector.
func NewSelfReflectorACEAdapter(reflector *SelfReflector, opts ...SelfReflectorACEAdapterOption) *SelfReflectorACEAdapter {
	a := &SelfReflectorACEAdapter{
		reflector: reflector,
		limit:     DefaultTopReflectionsLimit,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Extract implements ace.Adapter by converting SelfReflector's reflections to InsightCandidates.
func (a *SelfReflectorACEAdapter) Extract(ctx context.Context) ([]ace.InsightCandidate, error) {
	if a.reflector == nil {
		return nil, nil
	}

	reflections := a.reflector.GetTopReflections(a.limit)
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
