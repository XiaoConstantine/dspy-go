package decide

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
)

const tunedParameterSchemaVersion = 1

type persistedParameters struct {
	SchemaVersion     int                           `json:"schema_version"`
	Thresholds        map[string]float64            `json:"thresholds"`
	ScoreAnchors      map[string][]float64          `json:"score_anchors"`
	ChoiceMultipliers map[string]map[string]float64 `json:"choice_multipliers"`
}

// SetThreshold changes the local boundary for one Noul output. It does not
// change the provider request.
func (d *Decide) SetThreshold(name string, threshold float64) error {
	return d.setOutputParameter(name, outputNoul, threshold)
}

// SetScoreAnchors changes the local numeric interpretation of one Score
// distribution. Anchors must remain strictly increasing within the declared
// range and do not change the provider request.
func (d *Decide) SetScoreAnchors(name string, anchors []float64) error {
	return d.setOutputParameter(name, outputScore, append([]float64(nil), anchors...))
}

// SetChoiceMultipliers changes local Choice selection. Omitted labels default
// to one. Multipliers do not change provider probabilities or confidence.
func (d *Decide) SetChoiceMultipliers(name string, multipliers map[string]float64) error {
	return d.setOutputParameter(name, outputChoice, maps.Clone(multipliers))
}

func (d *Decide) setOutputParameter(name string, expected outputKind, value any) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, output := range d.outputs {
		if output.outputName() != name {
			continue
		}
		if output.kind() != expected {
			return fmt.Errorf("decide: output %q is %s, not %s", name, output.kind(), expected)
		}
		if err := output.setParameter(value); err != nil {
			return fmt.Errorf("decide: output %q: %w", name, err)
		}
		return nil
	}
	return fmt.Errorf("decide: unknown output %q", name)
}

// GetTunedParameters implements core.ParameterProvider. The returned map is an
// independent, versioned snapshot and never contains the client or API key.
func (d *Decide) GetTunedParameters() map[string]any {
	d.mu.RLock()
	defer d.mu.RUnlock()
	parameters := persistedFromOutputs(d.outputs)
	return map[string]any{
		"schema_version":     parameters.SchemaVersion,
		"thresholds":         parameters.Thresholds,
		"score_anchors":      parameters.ScoreAnchors,
		"choice_multipliers": parameters.ChoiceMultipliers,
	}
}

// SetTunedParameters implements core.ParameterConsumer. Loading is atomic and
// rejects unknown fields and incompatible schema versions.
func (d *Decide) SetTunedParameters(parameters map[string]any) error {
	encoded, err := json.Marshal(parameters)
	if err != nil {
		return fmt.Errorf("decide: encode tuned parameters: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	var persisted persistedParameters
	if err := decoder.Decode(&persisted); err != nil {
		return fmt.Errorf("decide: decode tuned parameters: %w", err)
	}
	if persisted.SchemaVersion != tunedParameterSchemaVersion {
		return fmt.Errorf("decide: unsupported tuned parameter schema version %d", persisted.SchemaVersion)
	}
	if persisted.Thresholds == nil || persisted.ScoreAnchors == nil || persisted.ChoiceMultipliers == nil {
		return fmt.Errorf("decide: tuned parameter sections must all be present")
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	candidate := make([]Output, len(d.outputs))
	for i, output := range d.outputs {
		candidate[i] = output.clone()
	}
	if err := applyPersistedParameters(candidate, persisted); err != nil {
		return fmt.Errorf("decide: %w", err)
	}
	d.outputs = candidate
	return nil
}

func persistedFromOutputs(outputs []Output) persistedParameters {
	result := persistedParameters{
		SchemaVersion:     tunedParameterSchemaVersion,
		Thresholds:        make(map[string]float64),
		ScoreAnchors:      make(map[string][]float64),
		ChoiceMultipliers: make(map[string]map[string]float64),
	}
	for _, output := range outputs {
		switch output.kind() {
		case outputNoul:
			result.Thresholds[output.outputName()] = output.parameter().(float64)
		case outputScore:
			result.ScoreAnchors[output.outputName()] = append([]float64(nil), output.parameter().([]float64)...)
		case outputChoice:
			result.ChoiceMultipliers[output.outputName()] = maps.Clone(output.parameter().(map[string]float64))
		}
	}
	return result
}

func applyPersistedParameters(outputs []Output, parameters persistedParameters) error {
	expectedThresholds := make(map[string]struct{})
	expectedScores := make(map[string]struct{})
	expectedChoices := make(map[string]struct{})
	byName := make(map[string]Output, len(outputs))
	for _, output := range outputs {
		byName[output.outputName()] = output
		switch output.kind() {
		case outputNoul:
			expectedThresholds[output.outputName()] = struct{}{}
		case outputScore:
			expectedScores[output.outputName()] = struct{}{}
		case outputChoice:
			expectedChoices[output.outputName()] = struct{}{}
		}
	}
	if err := validateParameterNames("thresholds", parameters.Thresholds, expectedThresholds); err != nil {
		return err
	}
	if err := validateParameterNames("score_anchors", parameters.ScoreAnchors, expectedScores); err != nil {
		return err
	}
	if err := validateParameterNames("choice_multipliers", parameters.ChoiceMultipliers, expectedChoices); err != nil {
		return err
	}
	for name, value := range parameters.Thresholds {
		if err := byName[name].setParameter(value); err != nil {
			return fmt.Errorf("threshold %q: %w", name, err)
		}
	}
	for name, value := range parameters.ScoreAnchors {
		if err := byName[name].setParameter(value); err != nil {
			return fmt.Errorf("Score anchors %q: %w", name, err)
		}
	}
	for name, value := range parameters.ChoiceMultipliers {
		if err := byName[name].setParameter(value); err != nil {
			return fmt.Errorf("Choice multipliers %q: %w", name, err)
		}
	}
	return nil
}

func validateParameterNames[T any](section string, actual map[string]T, expected map[string]struct{}) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("%s fields do not match the module outputs", section)
	}
	for name := range actual {
		if _, found := expected[name]; !found {
			return fmt.Errorf("%s contains unknown output %q", section, name)
		}
	}
	return nil
}
