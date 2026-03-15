package tblite

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolCallingAgent_ArtifactsCloneAndTrace(t *testing.T) {
	llm := &toolCallingStubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name": "Finish",
					"arguments": map[string]any{
						"answer": "done",
					},
				},
				"_usage": &core.TokenInfo{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
			},
		},
	}

	agent, err := NewToolCallingAgent(llm, ToolCallingAgentConfig{
		MaxTurns:     4,
		SystemPrompt: "original prompt",
		ToolPolicy:   "read narrow files first and verify before finishing",
	})
	require.NoError(t, err)

	artifacts := agent.GetArtifacts()
	assert.Equal(t, "original prompt", artifacts.Text[optimize.ArtifactSkillPack])
	assert.Equal(t, "read narrow files first and verify before finishing", artifacts.Text[optimize.ArtifactToolPolicy])
	assert.Equal(t, 4, artifacts.Int["max_turns"])

	require.NoError(t, agent.SetArtifacts(optimize.AgentArtifacts{
		Text: map[optimize.ArtifactKey]string{
			optimize.ArtifactSkillPack:  "updated prompt",
			optimize.ArtifactToolPolicy: "prefer short commands and focused verification",
		},
		Int: map[string]int{
			"max_turns": 7,
		},
	}))

	clonedAny, err := agent.Clone()
	require.NoError(t, err)
	cloned := clonedAny.(*ToolCallingAgent)
	assert.Equal(t, "updated prompt", cloned.GetArtifacts().Text[optimize.ArtifactSkillPack])
	assert.Equal(t, "prefer short commands and focused verification", cloned.GetArtifacts().Text[optimize.ArtifactToolPolicy])
	assert.Equal(t, 7, cloned.GetArtifacts().Int["max_turns"])

	taskDir := t.TempDir()
	envDir := filepath.Join(taskDir, "environment")
	require.NoError(t, os.MkdirAll(envDir, 0o755))
	testsDir := filepath.Join(taskDir, "tests")
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	_, err = agent.RunTask(context.Background(), TerminalTaskRequest{
		TaskID:           "trace-task",
		Instruction:      "Finish immediately",
		TaskDir:          taskDir,
		WorkingDirectory: envDir,
		EnvironmentDir:   envDir,
		TestsDir:         testsDir,
		MaxTurns:         1,
	})
	require.NoError(t, err)

	trace := agent.LastExecutionTrace()
	require.NotNil(t, trace)
	assert.Equal(t, "trace-task", trace.Task)
	assert.Equal(t, "finish", trace.TerminationCause)
	require.Len(t, trace.Steps, 1)
	assert.Equal(t, "Finish", trace.Steps[0].Tool)
	assert.Equal(t, int64(5), trace.TokenUsage["total_tokens"])
	require.NotEmpty(t, llm.prompts)
	assert.Contains(t, llm.prompts[0], "TOOL POLICY:\nprefer short commands and focused verification")
}

func TestTerminalTaskRequestFromInput_PreservesAgentTimeout(t *testing.T) {
	req, err := terminalTaskRequestFromInput(map[string]interface{}{
		"task_id":           "timeout-task",
		"instruction":       "do work",
		"task_dir":          "/tmp/task",
		"working_directory": "/tmp/task/environment",
		"environment_dir":   "/tmp/task/environment",
		"tests_dir":         "/tmp/task/tests",
		"agent_timeout":     int64(15 * time.Second),
		"max_turns":         7,
		"test_script_path":  "/tmp/task/test.sh",
		"docker_image":      "example:latest",
	})
	require.NoError(t, err)
	assert.Equal(t, 15*time.Second, req.AgentTimeout)
	assert.Equal(t, 7, req.MaxTurns)
}
