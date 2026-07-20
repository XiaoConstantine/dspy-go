//go:build !skip

package datasets

import (
	"encoding/json"
	"fmt"
	"os"
)

type HotPotQAExample struct {
	ID              string  `json:"_id"`
	SupportingFacts [][]any `json:"supporting_facts"`
	Context         [][]any `json:"context"`
	Question        string  `json:"question"`
	Answer          string  `json:"answer"`
	Type            string  `json:"type"`
	Level           string  `json:"level"`
}

func LoadHotpotQA() ([]HotPotQAExample, error) {
	datasetPath, err := EnsureDataset("hotpotqa")
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(datasetPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read HotPotQA dataset: %w", err)
	}
	var examples []HotPotQAExample
	if err := json.Unmarshal(data, &examples); err != nil {
		return nil, fmt.Errorf("failed to parse HotPotQA dataset: %w", err)
	}
	return examples, nil
}
