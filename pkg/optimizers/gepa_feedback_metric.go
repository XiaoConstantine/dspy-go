package optimizers

import (
	"context"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// GEPAFeedback captures optional example-level guidance alongside the scalar
// GEPA score. Feedback is cached by whole-program component texts and example
// content, so evaluators should derive it from those inputs rather than
// candidate identity or generation-local state.
type GEPAFeedback struct {
	Feedback        string                 `json:"feedback,omitempty"`
	TargetComponent string                 `json:"target_component,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// GEPAFeedbackContext provides per-example execution context for optional
// feedback generation without changing the shared core.Metric contract.
type GEPAFeedbackContext struct {
	Candidate *GEPACandidate
	Example   core.Example
	Outputs   map[string]interface{}
	Err       error
}

// GEPAFeedbackEvaluator optionally augments scalar scoring with example-level
// textual guidance that reflection and proposal prompts can consume.
type GEPAFeedbackEvaluator interface {
	EvaluateFeedback(ctx context.Context, expected, actual map[string]interface{}, info *GEPAFeedbackContext) *GEPAFeedback
}

// GEPAFeedbackEvaluatorFunc adapts a function into a GEPAFeedbackEvaluator.
type GEPAFeedbackEvaluatorFunc func(ctx context.Context, expected, actual map[string]interface{}, info *GEPAFeedbackContext) *GEPAFeedback

func (fn GEPAFeedbackEvaluatorFunc) EvaluateFeedback(ctx context.Context, expected, actual map[string]interface{}, info *GEPAFeedbackContext) *GEPAFeedback {
	if fn == nil {
		return nil
	}
	return fn(ctx, expected, actual, info)
}

func cloneGEPAFeedback(feedback *GEPAFeedback) *GEPAFeedback {
	if feedback == nil {
		return nil
	}

	return &GEPAFeedback{
		Feedback:        feedback.Feedback,
		TargetComponent: feedback.TargetComponent,
		Metadata:        cloneStringAnyMap(feedback.Metadata),
	}
}

func normalizeGEPAFeedback(feedback *GEPAFeedback) *GEPAFeedback {
	if feedback == nil {
		return nil
	}

	normalized := cloneGEPAFeedback(feedback)
	normalized.Feedback = strings.TrimSpace(normalized.Feedback)
	normalized.TargetComponent = strings.TrimSpace(normalized.TargetComponent)
	if normalized.Feedback == "" && normalized.TargetComponent == "" && len(normalized.Metadata) == 0 {
		return nil
	}
	return normalized
}
