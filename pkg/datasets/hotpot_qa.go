package datasets

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type HotPotQAExample struct {
	ID              string          `json:"_id"`
	SupportingFacts [][]interface{} `json:"supporting_facts"`
	Context         [][]interface{} `json:"context"`
	Question        string          `json:"question"`
	Answer          string          `json:"answer"`
	Type            string          `json:"type"`
	Level           string          `json:"level"`
}

func LoadHotpotQA() ([]HotPotQAExample, error) {
	datasetPath, err := EnsureDataset("hotpotqa")
	if err != nil {
		return nil, err
	}
	file, err := os.Open(datasetPath)
	if err != nil {
		return nil, err
	}
	byteValue, err := io.ReadAll(file)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return nil, err
	}
	var examples []HotPotQAExample
	err = json.Unmarshal(byteValue, &examples)
	if err != nil {
		fmt.Println("Failed to load HotPotQA dataset:", err)
		return nil, err
	}
	return examples, nil
}
