package commands

import (
	"bytes"
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/benchmarks/tblite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecodeTerminalTaskRequest(t *testing.T) {
	req, err := decodeTerminalTaskRequest(bytes.NewBufferString(`{
		"task_id":"task-1",
		"task_dir":"/tmp/task",
		"environment_dir":"/tmp/task/environment",
		"instruction":"fix it"
	}`))
	require.NoError(t, err)
	assert.Equal(t, "task-1", req.TaskID)
}

func TestDecodeTerminalTaskRequest_RequiresCoreFields(t *testing.T) {
	_, err := decodeTerminalTaskRequest(bytes.NewBufferString(`{"task_id":"missing-dirs"}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "task_dir is required")
}

func TestRunTerminalTaskCommand_EncodesResult(t *testing.T) {
	var out bytes.Buffer

	err := runTerminalTaskCommand(context.Background(), stubTerminalTaskAgent{
		result: &tblite.TerminalTaskResult{
			Completed:   true,
			FinalAnswer: "done",
		},
	}, tblite.TerminalTaskRequest{TaskID: "task-1"}, &out)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"completed": true,
		"final_answer": "done",
		"duration": 0,
		"token_usage": {
			"prompt_tokens": 0,
			"completion_tokens": 0,
			"total_tokens": 0
		}
	}`, out.String())
}

type stubTerminalTaskAgent struct {
	result *tblite.TerminalTaskResult
	err    error
}

func (s stubTerminalTaskAgent) RunTask(ctx context.Context, req tblite.TerminalTaskRequest) (*tblite.TerminalTaskResult, error) {
	return s.result, s.err
}
