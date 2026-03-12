package tblite

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
)

// ComparisonReport captures baseline vs tuned benchmark results for the same task slice.
type ComparisonReport struct {
	Label                  string                    `json:"label,omitempty"`
	DatasetName            string                    `json:"dataset_name"`
	Split                  string                    `json:"split"`
	Offset                 int                       `json:"offset"`
	Limit                  int                       `json:"limit"`
	StartedAt              time.Time                 `json:"started_at"`
	Duration               time.Duration             `json:"duration"`
	TrainingTaskCount      int                       `json:"training_task_count"`
	ValidationTaskCount    int                       `json:"validation_task_count"`
	TestTaskCount          int                       `json:"test_task_count"`
	BestArtifacts          optimize.AgentArtifacts   `json:"best_artifacts"`
	Baseline               *EvalReport               `json:"baseline,omitempty"`
	Tuned                  *EvalReport               `json:"tuned,omitempty"`
}

// WriteComparisonReport persists a GEPA comparison report as formatted JSON.
func WriteComparisonReport(path string, report *ComparisonReport) error {
	if report == nil {
		return fmt.Errorf("comparison report is required")
	}
	if path == "" {
		return fmt.Errorf("output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	bytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal comparison report: %w", err)
	}
	if err := os.WriteFile(path, bytes, 0o644); err != nil {
		return fmt.Errorf("write comparison report: %w", err)
	}
	return nil
}
