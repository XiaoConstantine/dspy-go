package datasets

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureDataset(t *testing.T) {
	// Setup
	homeDir, _ := os.UserHomeDir()
	datasetDir := filepath.Join(homeDir, ".dspy-go", "datasets")

	tests := []struct {
		name           string
		datasetName    string
		expectedSuffix string
		setupFunc      func()
		cleanupFunc    func()
	}{
		{
			name:           "GSM8K dataset - not existing",
			datasetName:    "gsm8k",
			expectedSuffix: ".parquet",
			setupFunc: func() {
				os.RemoveAll(datasetDir)
			},
			cleanupFunc: func() {
				os.RemoveAll(datasetDir)
			},
		},
		{
			name:           "HotPotQA dataset - not existing",
			datasetName:    "hotpotqa",
			expectedSuffix: ".json",
			setupFunc: func() {
				os.RemoveAll(datasetDir)
			},
			cleanupFunc: func() {
				os.RemoveAll(datasetDir)
			},
		},
		{
			name:           "Unknown dataset",
			datasetName:    "unknown",
			expectedSuffix: ".parquet",
			setupFunc:      func() {},
			cleanupFunc:    func() {},
		},
		{
			name:           "Existing dataset",
			datasetName:    "existing",
			expectedSuffix: ".parquet",
			setupFunc: func() {
				if err := os.MkdirAll(datasetDir, 0755); err != nil {
					return
				}
				if err := os.WriteFile(filepath.Join(datasetDir, "existing.parquet"), []byte("test"), 0644); err != nil {
					return
				}
			},
			cleanupFunc: func() {
				os.RemoveAll(datasetDir)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupFunc()
			defer tt.cleanupFunc()

			// Mock HTTP server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte("mock dataset content")); err != nil {
					return
				}
			}))
			defer server.Close()
			setTestURLs(server.URL, server.URL)

			path, err := EnsureDataset(tt.datasetName)

			if tt.datasetName == "unknown" {
				if err == nil {
					t.Errorf("Expected error for unknown dataset, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}

				expectedPath := filepath.Join(datasetDir, tt.datasetName+tt.expectedSuffix)
				if path != expectedPath {
					t.Errorf("Expected path %s, got %s", expectedPath, path)
				}

				if _, err := os.Stat(path); os.IsNotExist(err) {
					t.Errorf("Dataset file not created")
				}
			}
		})
	}
}

func TestDownloadDataset(t *testing.T) {
	// Setup
	tempDir, err := os.MkdirTemp("", "dataset-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name        string
		datasetName string
		setupServer func() *httptest.Server
		expectError bool
	}{
		{
			name:        "Successful download - GSM8K",
			datasetName: "gsm8k",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
					if _, err := w.Write([]byte("mock gsm8k content")); err != nil {
						return
					}
				}))
			},
			expectError: false,
		},
		{
			name:        "Successful download - HotPotQA",
			datasetName: "hotpotqa",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
					if _, err := w.Write([]byte("mock hotpotqa content")); err != nil {
						return
					}
				}))
			},
			expectError: false,
		},
		{
			name:        "Unknown dataset",
			datasetName: "unknown",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				}))
			},
			expectError: true,
		},
		{
			name:        "Server error",
			datasetName: "gsm8k",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := tt.setupServer()
			defer server.Close()

			setTestURLs(server.URL, server.URL)

			datasetPath := filepath.Join(tempDir, tt.datasetName+".dataset")
			err := downloadDataset(tt.datasetName, datasetPath)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}

				content, err := os.ReadFile(datasetPath)
				if err != nil {
					t.Errorf("Failed to read downloaded file: %v", err)
				}

				expectedContent := fmt.Sprintf("mock %s content", tt.datasetName)
				if string(content) != expectedContent {
					t.Errorf("Expected content %s, got %s", expectedContent, string(content))
				}
			}
		})
	}
}
