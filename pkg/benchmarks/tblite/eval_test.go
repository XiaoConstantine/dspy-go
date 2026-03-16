package tblite

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEvaluateTasks_BuildsSummary(t *testing.T) {
	rootDir := t.TempDir()
	runner := NewRunner(fakeEvalAgent{}, RunnerConfig{MaxTurns: 3})

	tasks := []datasets.TBLiteTask{
		newEvalTask("task-a", "alpha"),
		newEvalTask("task-b", "beta"),
	}

	report, err := EvaluateTasks(context.Background(), runner, tasks, EvalConfig{
		DatasetName: "test/tblite",
		Split:       "train",
		Offset:      0,
		Limit:       2,
		RootDir:     rootDir,
	})
	require.NoError(t, err)

	assert.Equal(t, 2, report.Summary.TotalTasks)
	assert.Equal(t, 2, report.Summary.PassedTasks)
	assert.Equal(t, 1.0, report.Summary.PassRate)
	assert.Equal(t, 1.0, report.Summary.AverageToolCalls)
	assert.Len(t, report.Tasks, 2)
}

func TestWriteReport(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report.json")
	report := &EvalReport{
		DatasetName: "test/tblite",
		Split:       "train",
		Summary:     EvalSummary{TotalTasks: 1},
	}
	require.NoError(t, WriteReport(path, report))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"dataset_name": "test/tblite"`)
}

func TestEvaluateTasks_RecordsTaskErrorsAndContinues(t *testing.T) {
	rootDir := t.TempDir()
	runner := NewRunner(failingEvalAgent{failedTaskID: "task-b"}, RunnerConfig{MaxTurns: 3})

	tasks := []datasets.TBLiteTask{
		newEvalTask("task-a", "alpha"),
		newEvalTask("task-b", "beta"),
	}

	report, err := EvaluateTasks(context.Background(), runner, tasks, EvalConfig{
		DatasetName: "test/tblite",
		Split:       "train",
		Offset:      0,
		Limit:       2,
		RootDir:     rootDir,
	})
	require.NoError(t, err)

	assert.Equal(t, 2, report.Summary.TotalTasks)
	assert.Equal(t, 1, report.Summary.PassedTasks)
	require.Len(t, report.Tasks, 2)
	require.NotNil(t, report.Tasks[1].AgentResult)
	assert.Contains(t, report.Tasks[1].AgentResult.Error, "runner error")
	assert.Nil(t, report.Tasks[1].TestResult)
}

type fakeEvalAgent struct{}

func (fakeEvalAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	target := filepath.Join(req.EnvironmentDir, "result.txt")
	if err := os.WriteFile(target, []byte("ok"), 0o644); err != nil {
		return nil, err
	}
	return &TerminalTaskResult{
		Completed:   true,
		FinalAnswer: "ok",
		ToolCalls:   1,
		TokenUsage: TokenUsage{
			PromptTokens:     5,
			CompletionTokens: 2,
			TotalTokens:      7,
		},
	}, nil
}

type failingEvalAgent struct {
	failedTaskID string
}

func (a failingEvalAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	if req.TaskID == a.failedTaskID {
		return nil, assert.AnError
	}
	return fakeEvalAgent{}.RunTask(ctx, req)
}

func newEvalTask(name, category string) datasets.TBLiteTask {
	return datasets.TBLiteTask{
		TaskName:    name,
		Instruction: "write result.txt",
		Category:    category,
		TestScript:  "#!/bin/sh\nset -eu\n[ \"$(cat result.txt)\" = \"ok\" ]\n",
	}
}
