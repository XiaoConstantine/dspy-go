package ace

import (
	"context"
)

// PatternSource provides patterns for adaptation to ACE learnings.
type PatternSource interface {
	GetPatterns() map[string]*PatternInfo
}

// PatternInfo describes a pattern from external systems.
type PatternInfo struct {
	Name        string
	Occurrences int
	SuccessRate float64
}

// MetricsSource provides metrics including error patterns.
type MetricsSource interface {
	GetErrorPatterns() map[string]int
}

// SelfReflectorAdapter extracts insights from SelfReflector patterns.
type SelfReflectorAdapter struct {
	patterns       PatternSource
	metrics        MetricsSource
	minOccurrences int
	minSuccessRate float64
}

// NewSelfReflectorAdapter creates an adapter for SelfReflector.
func NewSelfReflectorAdapter(patterns PatternSource, metrics MetricsSource) *SelfReflectorAdapter {
	return &SelfReflectorAdapter{
		patterns:       patterns,
		metrics:        metrics,
		minOccurrences: 3,
		minSuccessRate: 0.7,
	}
}

// Extract converts SelfReflector patterns to InsightCandidates.
func (a *SelfReflectorAdapter) Extract(ctx context.Context) ([]InsightCandidate, error) {
	var insights []InsightCandidate

	// Extract success patterns
	if a.patterns != nil {
		patterns := a.patterns.GetPatterns()
		for name, info := range patterns {
			if info.Occurrences >= a.minOccurrences && info.SuccessRate >= a.minSuccessRate {
				insights = append(insights, InsightCandidate{
					Content:    "Successful tool sequence: " + name,
					Category:   "strategies",
					Confidence: info.SuccessRate,
					Source:     "self_reflector",
				})
			}
		}
	}

	// Extract error patterns
	if a.metrics != nil {
		errorPatterns := a.metrics.GetErrorPatterns()
		for errMsg, count := range errorPatterns {
			if count >= 2 {
				confidence := 0.5 + float64(count)*0.1
				if confidence > 0.9 {
					confidence = 0.9
				}
				insights = append(insights, InsightCandidate{
					Content:    "Recurring error: " + errMsg,
					Category:   "mistakes",
					Confidence: confidence,
					Source:     "self_reflector",
				})
			}
		}
	}

	return insights, nil
}

// ErrorSource provides error and success records for adaptation.
type ErrorSource interface {
	GetErrors() []ErrorInfo
	GetSuccesses() []SuccessInfo
}

// ErrorInfo describes an error record from external systems.
type ErrorInfo struct {
	ErrorType string
	Message   string
	Pattern   string
	Count     int
}

// SuccessInfo describes a success record from external systems.
type SuccessInfo struct {
	SuccessType string
	Description string
	Pattern     string
	Confidence  float64
}

// ErrorRetainerAdapter extracts insights from ErrorRetainer records.
type ErrorRetainerAdapter struct {
	source ErrorSource
}

// NewErrorRetainerAdapter creates an adapter for ErrorRetainer.
func NewErrorRetainerAdapter(source ErrorSource) *ErrorRetainerAdapter {
	return &ErrorRetainerAdapter{source: source}
}

// Extract converts ErrorRetainer records to InsightCandidates.
func (a *ErrorRetainerAdapter) Extract(ctx context.Context) ([]InsightCandidate, error) {
	if a.source == nil {
		return nil, nil
	}

	var insights []InsightCandidate

	// Convert error records to mistake insights
	for _, err := range a.source.GetErrors() {
		content := err.Message
		if err.Pattern != "" {
			content = err.Pattern + ": " + err.Message
		}

		confidence := 0.6
		if err.Count > 1 {
			confidence = 0.7 + float64(err.Count)*0.05
			if confidence > 0.95 {
				confidence = 0.95
			}
		}

		insights = append(insights, InsightCandidate{
			Content:    content,
			Category:   "mistakes",
			Confidence: confidence,
			Source:     "error_retainer",
		})
	}

	// Convert success records to strategy insights
	for _, success := range a.source.GetSuccesses() {
		content := success.Description
		if success.Pattern != "" {
			content = success.Pattern + ": " + success.Description
		}

		insights = append(insights, InsightCandidate{
			Content:    content,
			Category:   "strategies",
			Confidence: success.Confidence,
			Source:     "error_retainer",
		})
	}

	return insights, nil
}

// StaticAdapter provides fixed insights for testing.
type StaticAdapter struct {
	insights []InsightCandidate
}

// NewStaticAdapter creates an adapter with fixed insights.
func NewStaticAdapter(insights []InsightCandidate) *StaticAdapter {
	return &StaticAdapter{insights: insights}
}

// Extract returns the static insights.
func (a *StaticAdapter) Extract(ctx context.Context) ([]InsightCandidate, error) {
	return a.insights, nil
}
