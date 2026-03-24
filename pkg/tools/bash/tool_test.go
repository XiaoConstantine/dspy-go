package bash

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBashToolRunsCommandInWorkspace(t *testing.T) {
	tool, err := NewTool(Config{Root: t.TempDir()})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": "printf 'hello world'",
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Equal(t, "hello world", result.Metadata[core.ToolResultModelTextMeta])
}

func TestBashToolRejectsEscapingWorkingDirectory(t *testing.T) {
	tool, err := NewTool(Config{Root: t.TempDir()})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command":           "pwd",
		"working_directory": "../outside",
	})
	require.NoError(t, err)
	assert.True(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Contains(t, result.Metadata[core.ToolResultModelTextMeta], "escapes workspace root")
}

func TestBashToolReportsFailures(t *testing.T) {
	tool, err := NewTool(Config{Root: t.TempDir()})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": "echo nope && exit 7",
	})
	require.NoError(t, err)
	assert.True(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Contains(t, result.Metadata[core.ToolResultDisplayTextMeta], "command failed")
}

func TestBashToolDoesNotExposeParentSecrets(t *testing.T) {
	t.Setenv("DSPY_GO_SECRET_TEST", "topsecret")

	tool, err := NewTool(Config{Root: t.TempDir()})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": `printf '%s' "${DSPY_GO_SECRET_TEST-unset}"`,
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Equal(t, "unset", result.Metadata[core.ToolResultModelTextMeta])
	assert.NotContains(t, result.Metadata[core.ToolResultDisplayTextMeta], "topsecret")
}

func TestBashToolAllowsExplicitExtraEnvironment(t *testing.T) {
	tool, err := NewTool(Config{
		Root: t.TempDir(),
		ExtraEnv: map[string]string{
			"GH_TOKEN": "gh-secret",
		},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": `printf '%s' "$GH_TOKEN"`,
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Equal(t, "gh-secret", result.Metadata[core.ToolResultModelTextMeta])
}

func TestBashToolAllowsExplicitPassthroughEnvironmentKeys(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "github-secret")

	tool, err := NewTool(Config{
		Root:               t.TempDir(),
		PassthroughEnvKeys: []string{"GITHUB_TOKEN"},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": `printf '%s' "$GITHUB_TOKEN"`,
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Equal(t, "github-secret", result.Metadata[core.ToolResultModelTextMeta])
}

func TestBashToolTimesOut(t *testing.T) {
	tool, err := NewTool(Config{
		Root:    t.TempDir(),
		Timeout: 50 * time.Millisecond,
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), map[string]any{
		"command": "sleep 1",
	})
	require.NoError(t, err)
	assert.True(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Contains(t, result.Metadata[core.ToolResultDisplayTextMeta], "timed out")
}

func TestBashToolCloneAndValidate(t *testing.T) {
	tool, err := NewTool(Config{Root: t.TempDir()})
	require.NoError(t, err)

	require.Error(t, tool.Validate(map[string]any{}))
	cloneable, ok := tool.(core.CloneableTool)
	require.True(t, ok)
	require.NotNil(t, cloneable.CloneTool())
}

func metadataBool(metadata map[string]any, key string) bool {
	value, _ := metadata[key].(bool)
	return value
}
