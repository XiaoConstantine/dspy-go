package datasets

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	GSM8KDatasetURL    = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
	HotPotQADatasetURL = "https://huggingface.co/datasets/hotpot_qa/raw/main/hotpot_dev_fullwiki_v1.json"
)

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
		url = GSM8KDatasetURL
	case "hotpotqa":
		url = HotPotQADatasetURL
	default:
		return fmt.Errorf("unknown dataset: %s", datasetName)
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download dataset: %w", err)
	}
	defer resp.Body.Close()

	out, err := os.Create(datasetPath)
	if err != nil {
		return fmt.Errorf("failed to create dataset file: %w", err)
	}
	defer out.Close()

	// For other datasets, we can copy directly
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to save dataset: %w", err)
	}
	return nil
}
