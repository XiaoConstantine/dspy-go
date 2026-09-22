package decide

import (
	"maps"
	"slices"
)

// Decision is evidence for one closed-set output. Its concrete form is one of
// NoulDecision, ScoreDecision, or ChoiceDecision[T].
type Decision interface {
	// Native returns the value placed in Result.Outputs and Process output.
	Native() any
	decision()
}

// NoulDecision is a thresholded Boolean and its provider probability.
// Confidence is distance from Threshold, not a calibrated probability that
// Value is correct.
type NoulDecision struct {
	Value       bool
	Probability float64
	Threshold   float64
	Confidence  float64
}

func (d NoulDecision) Native() any { return d.Value }
func (NoulDecision) decision()     {}

// ScoreDecision is a locally weighted expected value over the provider's raw
// index-keyed distribution. ProviderScore and ProviderConfidence are retained
// for provenance and are not recomputed from Anchors.
type ScoreDecision struct {
	Value              float64
	ProviderScore      float64
	ProviderConfidence float64
	probabilities      map[int]float64
	anchors            []float64
}

func (d ScoreDecision) Native() any { return d.Value }
func (ScoreDecision) decision()     {}

// Probabilities returns a copy of the provider's raw index-keyed distribution.
func (d ScoreDecision) Probabilities() map[int]float64 {
	return maps.Clone(d.probabilities)
}

// Anchors returns a copy of the local numeric anchors used for Value.
func (d ScoreDecision) Anchors() []float64 {
	return slices.Clone(d.anchors)
}

// Clone returns an independent evidence snapshot.
func (d ScoreDecision) Clone() ScoreDecision {
	d.probabilities = maps.Clone(d.probabilities)
	d.anchors = slices.Clone(d.anchors)
	return d
}

// ChoiceDecision retains typed application values separately from provider
// labels. ProviderConfidence describes ProviderValue even when multipliers
// cause Value to differ.
type ChoiceDecision[T any] struct {
	Value              T
	ProviderValue      T
	LocalLabel         string
	ProviderLabel      string
	ProviderConfidence float64
	probabilities      map[string]float64
	multipliers        map[string]float64
}

func (d ChoiceDecision[T]) Native() any { return d.Value }
func (ChoiceDecision[T]) decision()     {}

// Probabilities returns a copy of the provider's raw label distribution.
func (d ChoiceDecision[T]) Probabilities() map[string]float64 {
	return maps.Clone(d.probabilities)
}

// Multipliers returns a copy of the local multipliers used for selection.
func (d ChoiceDecision[T]) Multipliers() map[string]float64 {
	return maps.Clone(d.multipliers)
}

// Clone returns an independent evidence snapshot.
func (d ChoiceDecision[T]) Clone() ChoiceDecision[T] {
	d.probabilities = maps.Clone(d.probabilities)
	d.multipliers = maps.Clone(d.multipliers)
	return d
}
