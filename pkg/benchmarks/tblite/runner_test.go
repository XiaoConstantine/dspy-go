package tblite

import (
	"context"
	"os"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeAgent struct {
	run func(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error)
}

func (f fakeAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	return f.run(ctx, req)
}

func TestRunnerEvaluateTask_RunsAgentAndVerifier(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:    "echo-success",
		Instruction: "Write the expected file",
		TestScript: "#!/bin/sh\nset -eu\n[ \"$(cat \"$DSPY_TBLITE_ENV_DIR/output.txt\")\" = \"done\" ]\n",
	}

	runner := NewRunner(fakeAgent{
		run: func(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
			require.Equal(t, task.TaskName, req.TaskID)
			require.NoError(t, os.WriteFile(req.EnvironmentDir+"/output.txt", []byte("done"), 0o644))
			return &TerminalTaskResult{
				Completed: true,
				ToolCalls: 2,
			}, nil
		},
	}, RunnerConfig{})

	result, err := runner.EvaluateTask(context.Background(), task, t.TempDir())
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.AgentResult)
	require.NotNil(t, result.TestResult)

	assert.True(t, result.AgentResult.Completed)
	assert.True(t, result.TestResult.Passed)
	assert.Equal(t, 0, result.TestResult.ExitCode)
}

func TestRunnerEvaluateTask_RecordsVerifierFailure(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:    "echo-failure",
		Instruction: "Write the wrong value",
		TestScript: "#!/bin/sh\nset -eu\n[ \"$(cat \"$DSPY_TBLITE_ENV_DIR/output.txt\")\" = \"expected\" ]\n",
	}

	runner := NewRunner(fakeAgent{
		run: func(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
			require.NoError(t, os.WriteFile(req.EnvironmentDir+"/output.txt", []byte("actual"), 0o644))
			return &TerminalTaskResult{Completed: true}, nil
		},
	}, RunnerConfig{})

	result, err := runner.EvaluateTask(context.Background(), task, t.TempDir())
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.TestResult)

	assert.False(t, result.TestResult.Passed)
	assert.NotZero(t, result.TestResult.ExitCode)
}

func TestRunnerEvaluateTask_RequiresAgent(t *testing.T) {
	runner := NewRunner(nil, RunnerConfig{})
	_, err := runner.EvaluateTask(context.Background(), datasets.TBLiteTask{TaskName: "missing-agent"}, t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent is required")
}
