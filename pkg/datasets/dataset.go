package datasets

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

var (
	gsm8kDatasetURL    = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
	hotPotQADatasetURL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
)

// For testing purposes.
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

	if _, err := os.Stat(datasetPath); err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("failed to stat dataset file: %w", err)
		}
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create download request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download dataset: %w", err)
	}
	defer resp.Body.Close()
	// Check for non-200 status codes
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned non-200 status code: %d", resp.StatusCode)
	}

	// Download to a temporary file and rename into place so an
	// interrupted download never leaves a partial file that later calls
	// would treat as a valid cached dataset.
	tmp, err := os.CreateTemp(filepath.Dir(datasetPath), datasetName+"-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temporary dataset file: %w", err)
	}
	defer func() {
		tmp.Close()
		os.Remove(tmp.Name())
	}()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return fmt.Errorf("failed to save dataset: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("failed to finalize dataset file: %w", err)
	}
	if err := os.Rename(tmp.Name(), datasetPath); err != nil {
		return fmt.Errorf("failed to move dataset into place: %w", err)
	}

	return nil
}

// SimpleDataset implements core.Dataset interface for testing and examples.
type SimpleDataset struct {
	examples []core.Example
	current  int
}

// NewSimpleDataset creates a new SimpleDataset with the given examples.
func NewSimpleDataset(examples []core.Example) *SimpleDataset {
	return &SimpleDataset{
		examples: examples,
		current:  0,
	}
}

// Next returns the next example in the dataset.
func (sd *SimpleDataset) Next() (core.Example, bool) {
	if sd.current >= len(sd.examples) {
		return core.Example{}, false
	}
	example := sd.examples[sd.current]
	sd.current++
	return example, true
}

// Reset resets the dataset iterator to the beginning.
func (sd *SimpleDataset) Reset() {
	sd.current = 0
}
