package datasets

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

var (
	gsm8kDatasetURL    = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
	hotPotQADatasetURL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
)

// For testing purposes
func setTestURLs(gsm8k, hotpotqa string) {
	gsm8kDatasetURL = gsm8k
	hotPotQADatasetURL = hotpotqa
}

func EnsureDataset(datasetName string) (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	var suffix string
	switch datasetName {
	case "gsm8k":
		suffix = ".parquet"
	case "hotpotqa":
		suffix = ".json"
	default:
		suffix = ".parquet"
	}
	datasetDir := filepath.Join(homeDir, ".dspy-go", "datasets")
	if err := os.MkdirAll(datasetDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create dataset directory: %w", err)
	}

	datasetPath := filepath.Join(datasetDir, datasetName+suffix)

	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		fmt.Printf("Dataset %s not found locally. Downloading from Hugging Face...\n", datasetName)
		if err := downloadDataset(datasetName, datasetPath); err != nil {
			return "", fmt.Errorf("failed to download dataset: %w", err)
		}
	}

	return datasetPath, nil
}

func downloadDataset(datasetName, datasetPath string) error {
	var url string
	switch datasetName {
	case "gsm8k":
		url = gsm8kDatasetURL
	case "hotpotqa":
		url = hotPotQADatasetURL
	default:
		return fmt.Errorf("unknown dataset: %s", datasetName)
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download dataset: %w", err)
	}
	defer resp.Body.Close()
	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned non-200 status code: %d", resp.StatusCode)
	}
	out, err := os.Create(datasetPath)
	if err != nil {
		return fmt.Errorf("failed to create dataset file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save dataset: %w", err)
	}

	return nil
}
