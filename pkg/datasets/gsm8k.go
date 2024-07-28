package datasets

import (
	"encoding/json"
	"fmt"
	"os"
)

type GSM8KExample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func LoadGSM8K() ([]GSM8KExample, error) {
	datasetPath, err := EnsureDataset("gsm8k")
	if err != nil {
		return nil, err
	}
	file, err := os.Open(datasetPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	defer file.Close()
	var examples []GSM8KExample
	decoder := json.NewDecoder(file)
	for decoder.More() {
		var example GSM8KExample
		if err := decoder.Decode(&example); err != nil {
			return nil, fmt.Errorf("failed to decode example: %w", err)
		}
		examples = append(examples, example)
	}

	return examples, nil
}
