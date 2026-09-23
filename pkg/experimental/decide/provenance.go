package decide

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

const resultProvenanceVersion = 1

type scoreLevelProvenance struct {
	Value       float64 `json:"value"`
	Description string  `json:"description"`
}

type choiceOptionProvenance struct {
	Label       string `json:"label"`
	Description string `json:"description"`
	ValueType   string `json:"value_type"`
}

type outputProvenance struct {
	Name          string                   `json:"name"`
	Kind          outputKind               `json:"kind"`
	ScoreLevels   []scoreLevelProvenance   `json:"score_levels,omitempty"`
	ChoiceOptions []choiceOptionProvenance `json:"choice_options,omitempty"`
}

type fieldProvenance struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Prefix      string         `json:"prefix"`
	Type        core.FieldType `json:"type"`
}

type resultProvenancePayload struct {
	Version     int                `json:"version"`
	Instruction string             `json:"instruction"`
	Inputs      []fieldProvenance  `json:"inputs"`
	Outputs     []fieldProvenance  `json:"outputs"`
	Decisions   []outputProvenance `json:"decisions"`
}

func computeResultProvenance(signature core.Signature, outputs []Output) (string, error) {
	payload := resultProvenancePayload{
		Version:     resultProvenanceVersion,
		Instruction: signature.Instruction,
		Inputs:      make([]fieldProvenance, len(signature.Inputs)),
		Outputs:     make([]fieldProvenance, len(signature.Outputs)),
		Decisions:   make([]outputProvenance, len(outputs)),
	}
	for i, field := range signature.Inputs {
		payload.Inputs[i] = fieldProvenance{
			Name: field.Name, Description: field.Description,
			Prefix: field.Prefix, Type: field.Type,
		}
	}
	for i, field := range signature.Outputs {
		payload.Outputs[i] = fieldProvenance{
			Name: field.Name, Description: field.Description,
			Prefix: field.Prefix, Type: field.Type,
		}
	}
	for i, output := range outputs {
		payload.Decisions[i] = output.provenance()
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode result provenance: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}
