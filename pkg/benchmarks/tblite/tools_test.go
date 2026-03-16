package tblite

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewTerminalToolset_ReadWriteAndList(t *testing.T) {
	root := t.TempDir()

	toolset, err := NewTerminalToolset(root, ToolsetConfig{})
	require.NoError(t, err)
	require.Len(t, toolset, 4)
	assertToolNames(t, toolset, "list_files", "read_file", "run_command", "write_file")
}

func TestResolveToolPath_RejectsEscapes(t *testing.T) {
	root := t.TempDir()

	_, err := resolveToolPath(root, "../outside.txt")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "escapes benchmark workspace")
}

func TestListEntriesRecursive(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(root, "nested"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "nested", "file.txt"), []byte("hello"), 0o644))

	entries, err := listEntries(root, true)
	require.NoError(t, err)
	assert.Contains(t, entries, "nested")
	assert.Contains(t, entries, "nested/file.txt")
}

func TestTruncateString(t *testing.T) {
	assert.Equal(t, "abcdef", agentutil.TruncateString("abcdef", 6))
	assert.Equal(t, "ab...", agentutil.TruncateString("abcdef", 5))
	assert.Equal(t, "ab", agentutil.TruncateString("abcdef", 2))
}

func TestNewTerminalToolset_SandboxAndCommand(t *testing.T) {
	root := t.TempDir()

	toolset, err := NewTerminalToolset(root, ToolsetConfig{})
	require.NoError(t, err)

	byName := map[string]core.Tool{}
	for _, tool := range toolset {
		byName[tool.Name()] = tool
	}

	ctx := context.Background()
	writeResult, err := byName["write_file"].Execute(ctx, map[string]any{
		"path":    "notes/todo.txt",
		"content": "ship the benchmark",
	})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(writeResult), "wrote")

	readResult, err := byName["read_file"].Execute(ctx, map[string]any{
		"path": "notes/todo.txt",
	})
	require.NoError(t, err)
	assert.Equal(t, "ship the benchmark", agentutil.StringifyToolResult(readResult))

	listResult, err := byName["list_files"].Execute(ctx, map[string]any{
		"path":      ".",
		"recursive": true,
	})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(listResult), "notes/todo.txt")

	commandResult, err := byName["run_command"].Execute(ctx, map[string]any{
		"command": "cat notes/todo.txt",
	})
	require.NoError(t, err)
	assert.Equal(t, "ship the benchmark", strings.TrimSpace(agentutil.StringifyToolResult(commandResult)))

	sandboxResult, err := byName["read_file"].Execute(ctx, map[string]any{
		"path": "../escape.txt",
	})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(sandboxResult), "escapes benchmark workspace")
}

func TestNewTerminalToolset_ResolvesTaskAndContainerAliases(t *testing.T) {
	taskRoot := t.TempDir()
	envRoot := filepath.Join(taskRoot, "environment")
	testsDir := filepath.Join(taskRoot, "tests")
	require.NoError(t, os.MkdirAll(filepath.Join(envRoot, "api"), 0o755))
	require.NoError(t, os.MkdirAll(testsDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(envRoot, "api", "app.py"), []byte("print('ok')"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(taskRoot, "test.sh"), []byte("#!/bin/sh\necho test\n"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(taskRoot, "instruction.txt"), []byte("do the task"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(testsDir, "check.txt"), []byte("check"), 0o644))

	toolset, err := NewTerminalToolset(envRoot, ToolsetConfig{
		TaskRoot:         taskRoot,
		TestsDir:         testsDir,
		ContainerEnvRoot: "/app",
	})
	require.NoError(t, err)

	byName := map[string]core.Tool{}
	for _, tool := range toolset {
		byName[tool.Name()] = tool
	}

	ctx := context.Background()

	readApp, err := byName["read_file"].Execute(ctx, map[string]any{"path": "/app/api/app.py"})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readApp), "print('ok')")

	readAppRelative, err := byName["read_file"].Execute(ctx, map[string]any{"path": "app/api/app.py"})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readAppRelative), "print('ok')")

	readTests, err := byName["read_file"].Execute(ctx, map[string]any{"path": "tests/check.txt"})
	require.NoError(t, err)
	assert.Equal(t, "check", agentutil.StringifyToolResult(readTests))

	readTaskScript, err := byName["read_file"].Execute(ctx, map[string]any{"path": "test.sh"})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readTaskScript), "echo test")

	readInstruction, err := byName["read_file"].Execute(ctx, map[string]any{"path": "instruction.txt"})
	require.NoError(t, err)
	assert.Equal(t, "do the task", agentutil.StringifyToolResult(readInstruction))

	readEnvironmentAlias, err := byName["read_file"].Execute(ctx, map[string]any{"path": "environment/api/app.py"})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readEnvironmentAlias), "print('ok')")

	readHostPath, err := byName["read_file"].Execute(ctx, map[string]any{"path": filepath.Join(taskRoot, "test.sh")})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readHostPath), "echo test")
}

func TestNewTerminalToolset_SanitizesToolErrors(t *testing.T) {
	taskRoot := t.TempDir()
	envRoot := filepath.Join(taskRoot, "environment")
	testsDir := filepath.Join(taskRoot, "tests")
	require.NoError(t, os.MkdirAll(envRoot, 0o755))
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	toolset, err := NewTerminalToolset(envRoot, ToolsetConfig{
		TaskRoot:         taskRoot,
		TestsDir:         testsDir,
		ContainerEnvRoot: "/app",
	})
	require.NoError(t, err)

	byName := map[string]core.Tool{}
	for _, tool := range toolset {
		byName[tool.Name()] = tool
	}

	result, err := byName["read_file"].Execute(context.Background(), map[string]any{
		"path": "/app/missing.txt",
	})
	require.NoError(t, err)
	output := agentutil.StringifyToolResult(result)
	assert.Contains(t, output, "/app/missing.txt")
	assert.NotContains(t, output, filepath.ToSlash(taskRoot))
	assert.NotContains(t, output, filepath.ToSlash(envRoot))
}

func TestNewTerminalToolset_RunCommandUsesAliasWorkingDirectory(t *testing.T) {
	taskRoot := t.TempDir()
	envRoot := filepath.Join(taskRoot, "environment")
	testsDir := filepath.Join(taskRoot, "tests")
	require.NoError(t, os.MkdirAll(envRoot, 0o755))
	require.NoError(t, os.MkdirAll(testsDir, 0o755))

	runner := &recordingCommandRunner{}
	toolset, err := NewTerminalToolset(envRoot, ToolsetConfig{
		TaskRoot:      taskRoot,
		TestsDir:      testsDir,
		CommandRunner: runner,
	})
	require.NoError(t, err)

	byName := map[string]core.Tool{}
	for _, tool := range toolset {
		byName[tool.Name()] = tool
	}

	result, err := byName["run_command"].Execute(context.Background(), map[string]any{
		"command":           "pwd",
		"working_directory": "tests",
	})
	require.NoError(t, err)
	assert.Equal(t, filepath.Clean(testsDir), runner.workingDir)
	assert.Equal(t, "ok", strings.TrimSpace(agentutil.StringifyToolResult(result)))
}

func TestNewTerminalToolset_RejectsSymlinkEscapes(t *testing.T) {
	root := t.TempDir()
	outsideDir := t.TempDir()
	outsideFile := filepath.Join(outsideDir, "outside.txt")
	require.NoError(t, os.WriteFile(outsideFile, []byte("outside"), 0o644))
	require.NoError(t, os.Symlink(outsideFile, filepath.Join(root, "link.txt")))

	toolset, err := NewTerminalToolset(root, ToolsetConfig{})
	require.NoError(t, err)

	byName := map[string]core.Tool{}
	for _, tool := range toolset {
		byName[tool.Name()] = tool
	}

	readResult, err := byName["read_file"].Execute(context.Background(), map[string]any{
		"path": "link.txt",
	})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(readResult), "escapes benchmark workspace")

	writeResult, err := byName["write_file"].Execute(context.Background(), map[string]any{
		"path":    "link.txt",
		"content": "mutated",
	})
	require.NoError(t, err)
	assert.Contains(t, agentutil.StringifyToolResult(writeResult), "escapes benchmark workspace")

	content, err := os.ReadFile(outsideFile)
	require.NoError(t, err)
	assert.Equal(t, "outside", string(content))
}

func TestHostCommandRunner_ExtraEnvOverridesHostEnv(t *testing.T) {
	t.Setenv("DSPY_TBLITE_ENV_ORDER", "host")

	output, err := (hostCommandRunner{}).Run(context.Background(), t.TempDir(), `printf %s "$DSPY_TBLITE_ENV_ORDER"`, []string{
		"DSPY_TBLITE_ENV_ORDER=task",
	})
	require.NoError(t, err)
	assert.Equal(t, "task", string(output))
}

func assertToolNames(t *testing.T, toolset []core.Tool, names ...string) {
	t.Helper()

	actual := make([]string, 0, len(toolset))
	for _, tool := range toolset {
		actual = append(actual, tool.Name())
	}
	assert.ElementsMatch(t, names, actual)
}

type recordingCommandRunner struct {
	workingDir string
	command    string
}

func (r *recordingCommandRunner) Run(ctx context.Context, workingDir string, command string, extraEnv []string) ([]byte, error) {
	r.workingDir = workingDir
	r.command = command
	return []byte("ok"), nil
}
