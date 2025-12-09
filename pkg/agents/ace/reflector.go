package ace

import (
	"context"
	"time"
)

// Reflector analyzes trajectories and produces reflection results.
type Reflector interface {
	Reflect(ctx context.Context, trajectory *Trajectory, learnings []Learning) (*ReflectionResult, error)
}

// UnifiedReflector combines multiple signal sources for comprehensive reflection.
type UnifiedReflector struct {
	adapters []Adapter
	domain   Reflector
}

// Adapter extracts insights from existing systems.
type Adapter interface {
	Extract(ctx context.Context) ([]InsightCandidate, error)
}

// NewUnifiedReflector creates a reflector that combines multiple sources.
func NewUnifiedReflector(adapters []Adapter, domain Reflector) *UnifiedReflector {
	return &UnifiedReflector{
		adapters: adapters,
		domain:   domain,
	}
}

// Reflect analyzes a trajectory and returns combined insights from all sources.
func (ur *UnifiedReflector) Reflect(ctx context.Context, trajectory *Trajectory, learnings []Learning) (*ReflectionResult, error) {
	result := &ReflectionResult{
		TrajectoryID: trajectory.ID,
		ProcessedAt:  time.Now(),
	}

	// Generate feedback for cited learnings based on outcome
	result.LearningUpdates = ur.correlateFeedback(trajectory)

	// Collect insights from adapters (free, no LLM)
	for _, adapter := range ur.adapters {
		insights, err := adapter.Extract(ctx)
		if err != nil {
			continue
		}
		for _, insight := range insights {
			if insight.Category == "mistakes" {
				result.FailurePatterns = append(result.FailurePatterns, insight)
			} else {
				result.SuccessPatterns = append(result.SuccessPatterns, insight)
			}
		}
	}

	// Get domain-specific insights (LLM-based, if provided)
	if ur.domain != nil {
		domainResult, err := ur.domain.Reflect(ctx, trajectory, learnings)
		if err == nil && domainResult != nil {
			result.SuccessPatterns = append(result.SuccessPatterns, domainResult.SuccessPatterns...)
			result.FailurePatterns = append(result.FailurePatterns, domainResult.FailurePatterns...)
			result.LearningUpdates = append(result.LearningUpdates, domainResult.LearningUpdates...)
		}
	}

	return result, nil
}

// correlateFeedback generates learning updates based on trajectory outcome.
func (ur *UnifiedReflector) correlateFeedback(trajectory *Trajectory) []LearningUpdate {
	if len(trajectory.Context.CitedLearnings) == 0 {
		return nil
	}

	var updates []LearningUpdate

	delta := DeltaHelpful
	if trajectory.FinalOutcome == OutcomeFailure {
		delta = DeltaHarmful
	} else if trajectory.FinalOutcome == OutcomePartial && trajectory.Quality < 0.5 {
		delta = DeltaHarmful
	}

	for _, cited := range trajectory.Context.CitedLearnings {
		updates = append(updates, LearningUpdate{
			LearningID: cited,
			Delta:      delta,
			Reason:     string(trajectory.FinalOutcome),
		})
	}

	return updates
}

// SimpleReflector extracts basic insights without LLM calls.
type SimpleReflector struct{}

// NewSimpleReflector creates a basic reflector for testing and simple use cases.
func NewSimpleReflector() *SimpleReflector {
	return &SimpleReflector{}
}

// Reflect analyzes trajectory steps to find patterns.
func (sr *SimpleReflector) Reflect(ctx context.Context, trajectory *Trajectory, learnings []Learning) (*ReflectionResult, error) {
	result := &ReflectionResult{
		TrajectoryID: trajectory.ID,
		ProcessedAt:  time.Now(),
	}

	// Extract patterns from successful trajectories
	if trajectory.FinalOutcome == OutcomeSuccess {
		if len(trajectory.Steps) > 0 && len(trajectory.Steps) <= 3 {
			result.SuccessPatterns = append(result.SuccessPatterns, InsightCandidate{
				Content:    "Completed efficiently with minimal steps",
				Category:   "strategies",
				Confidence: 0.6,
				Source:     "simple_reflector",
			})
		}
	}

	// Extract patterns from failures
	if trajectory.FinalOutcome == OutcomeFailure {
		for _, step := range trajectory.Steps {
			if step.Error != "" {
				result.FailurePatterns = append(result.FailurePatterns, InsightCandidate{
					Content:    "Error in step: " + step.Error,
					Category:   "mistakes",
					Confidence: 0.5,
					Source:     "simple_reflector",
				})
				break // Only capture first error
			}
		}
	}

	return result, nil
}
