package tblite

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
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
	assert.Equal(t, "abcdef", truncateString("abcdef", 6))
	assert.Equal(t, "ab...", truncateString("abcdef", 5))
	assert.Equal(t, "ab", truncateString("abcdef", 2))
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
	assert.Contains(t, stringifyToolResult(writeResult), "wrote")

	readResult, err := byName["read_file"].Execute(ctx, map[string]any{
		"path": "notes/todo.txt",
	})
	require.NoError(t, err)
	assert.Equal(t, "ship the benchmark", stringifyToolResult(readResult))

	listResult, err := byName["list_files"].Execute(ctx, map[string]any{
		"path":      ".",
		"recursive": true,
	})
	require.NoError(t, err)
	assert.Contains(t, stringifyToolResult(listResult), "notes/todo.txt")

	commandResult, err := byName["run_command"].Execute(ctx, map[string]any{
		"command": "cat notes/todo.txt",
	})
	require.NoError(t, err)
	assert.Equal(t, "ship the benchmark", strings.TrimSpace(stringifyToolResult(commandResult)))

	sandboxResult, err := byName["read_file"].Execute(ctx, map[string]any{
		"path": "../escape.txt",
	})
	require.NoError(t, err)
	assert.Contains(t, stringifyToolResult(sandboxResult), "escapes benchmark workspace")
}

func assertToolNames(t *testing.T, toolset []core.Tool, names ...string) {
	t.Helper()

	actual := make([]string, 0, len(toolset))
	for _, tool := range toolset {
		actual = append(actual, tool.Name())
	}
	assert.ElementsMatch(t, names, actual)
}
