package tblite

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCommandAgentRunTask_UsesJSONStdioContract(t *testing.T) {
	scriptPath := filepath.Join(t.TempDir(), "agent.sh")
	script := `#!/bin/sh
set -eu
cat >/dev/null
printf '{"completed":true,"final_answer":"ok","tool_calls":3}'
`
	require.NoError(t, os.WriteFile(scriptPath, []byte(script), 0o755))

	agent, err := NewCommandAgent(CommandAgentConfig{
		Command: "/bin/sh",
		Args:    []string{scriptPath},
	})
	require.NoError(t, err)

	taskDir := t.TempDir()
	result, err := agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:  "sample-task",
		TaskDir: taskDir,
	})
	require.NoError(t, err)
	require.NotNil(t, result)

	assert.True(t, result.Completed)
	assert.Equal(t, "ok", result.FinalAnswer)
	assert.Equal(t, 3, result.ToolCalls)
}

func TestCommandAgentRunTask_RejectsInvalidJSON(t *testing.T) {
	scriptPath := filepath.Join(t.TempDir(), "agent.sh")
	script := `#!/bin/sh
set -eu
cat >/dev/null
printf 'not-json'
`
	require.NoError(t, os.WriteFile(scriptPath, []byte(script), 0o755))

	agent, err := NewCommandAgent(CommandAgentConfig{
		Command: "/bin/sh",
		Args:    []string{scriptPath},
	})
	require.NoError(t, err)

	_, err = agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:  "sample-task",
		TaskDir: t.TempDir(),
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "parse command agent stdout")
}
