package tblite

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
)

// EvalConfig controls a fixed-slice TBLite batch evaluation.
type EvalConfig struct {
	DatasetName   string
	Split         string
	Offset        int
	Limit         int
	RootDir       string
	KeepArtifacts bool
	Label         string
}

// EvalSummary captures aggregate benchmark metrics for a run.
type EvalSummary struct {
	TotalTasks            int           `json:"total_tasks"`
	PassedTasks           int           `json:"passed_tasks"`
	PassRate              float64       `json:"pass_rate"`
	AverageToolCalls      float64       `json:"average_tool_calls"`
	AverageDuration       time.Duration `json:"average_duration"`
	TotalPromptTokens     int64         `json:"total_prompt_tokens"`
	TotalCompletionTokens int64         `json:"total_completion_tokens"`
	TotalTokens           int64         `json:"total_tokens"`
}

// TaskReport captures the per-task outcome without the materialized task pointer.
type TaskReport struct {
	TaskName    string              `json:"task_name"`
	Category    string              `json:"category,omitempty"`
	Difficulty  string              `json:"difficulty,omitempty"`
	AgentResult *TerminalTaskResult `json:"agent_result,omitempty"`
	TestResult  *TestResult         `json:"test_result,omitempty"`
}

// EvalReport is the stable JSON report emitted by the TBLite batch runner.
type EvalReport struct {
	Label       string        `json:"label,omitempty"`
	DatasetName string        `json:"dataset_name"`
	Split       string        `json:"split"`
	Offset      int           `json:"offset"`
	Limit       int           `json:"limit"`
	StartedAt   time.Time     `json:"started_at"`
	Duration    time.Duration `json:"duration"`
	Summary     EvalSummary   `json:"summary"`
	Tasks       []TaskReport  `json:"tasks"`
}

// LabelOrDefault returns a stable display label for the report.
func (r *EvalReport) LabelOrDefault() string {
	if r == nil || r.Label == "" {
		return "run"
	}
	return r.Label
}

// EvaluateTasks runs a fixed slice of TBLite tasks and returns a JSON-serializable report.
func EvaluateTasks(ctx context.Context, runner *Runner, tasks []datasets.TBLiteTask, cfg EvalConfig) (*EvalReport, error) {
	if runner == nil {
		return nil, fmt.Errorf("tblite runner is required")
	}
	if len(tasks) == 0 {
		return nil, fmt.Errorf("at least one task is required")
	}
	if cfg.RootDir == "" {
		return nil, fmt.Errorf("root dir is required")
	}
	if err := os.MkdirAll(cfg.RootDir, 0o755); err != nil {
		return nil, fmt.Errorf("create root dir: %w", err)
	}

	startedAt := time.Now()
	report := &EvalReport{
		Label:       cfg.Label,
		DatasetName: cfg.DatasetName,
		Split:       cfg.Split,
		Offset:      cfg.Offset,
		Limit:       cfg.Limit,
		StartedAt:   startedAt,
		Tasks:       make([]TaskReport, 0, len(tasks)),
	}

	var totalToolCalls int
	var totalDuration time.Duration

	for _, task := range tasks {
		result, err := runner.EvaluateTask(ctx, task, cfg.RootDir)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
				return nil, fmt.Errorf("evaluate task %q: %w", task.TaskName, err)
			}
			result = &EvaluationResult{
				Task: task.Normalize(),
				AgentResult: &TerminalTaskResult{
					Completed: false,
					Error:     fmt.Sprintf("runner error: %v", err),
				},
			}
		}

		report.Tasks = append(report.Tasks, TaskReport{
			TaskName:    result.Task.TaskName,
			Category:    result.Task.Category,
			Difficulty:  result.Task.Difficulty,
			AgentResult: result.AgentResult,
			TestResult:  result.TestResult,
		})

		report.Summary.TotalTasks++
		if result.TestResult != nil && result.TestResult.Passed {
			report.Summary.PassedTasks++
		}
		if result.AgentResult != nil {
			totalToolCalls += result.AgentResult.ToolCalls
			totalDuration += result.AgentResult.Duration
			report.Summary.TotalPromptTokens += result.AgentResult.TokenUsage.PromptTokens
			report.Summary.TotalCompletionTokens += result.AgentResult.TokenUsage.CompletionTokens
			report.Summary.TotalTokens += result.AgentResult.TokenUsage.TotalTokens
		}

		if !cfg.KeepArtifacts && result.MaterializedTask != nil {
			_ = os.RemoveAll(result.MaterializedTask.RootDir)
		}
	}

	report.Duration = time.Since(startedAt)
	if report.Summary.TotalTasks > 0 {
		report.Summary.PassRate = float64(report.Summary.PassedTasks) / float64(report.Summary.TotalTasks)
		report.Summary.AverageToolCalls = float64(totalToolCalls) / float64(report.Summary.TotalTasks)
		report.Summary.AverageDuration = totalDuration / time.Duration(report.Summary.TotalTasks)
	}

	return report, nil
}

// WriteReport persists a benchmark report as formatted JSON.
func WriteReport(path string, report *EvalReport) error {
	if report == nil {
		return fmt.Errorf("report is required")
	}
	if path == "" {
		return fmt.Errorf("output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	bytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}
	if err := os.WriteFile(path, bytes, 0o644); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}
