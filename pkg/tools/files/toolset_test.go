package files

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewToolsetRequiresRoot(t *testing.T) {
	toolset, err := NewToolset(Config{})
	require.Error(t, err)
	assert.Nil(t, toolset)
}

func TestFilesToolsRoundTrip(t *testing.T) {
	root := t.TempDir()
	toolset, err := NewToolset(Config{
		Root:               root,
		ModelOutputLimit:   64,
		DisplayOutputLimit: 256,
	})
	require.NoError(t, err)

	toolMap := byName(toolset.Tools())
	ctx := context.Background()

	writeResult, err := toolMap["write"].Execute(ctx, map[string]any{
		"path":    "notes/plan.md",
		"content": "# Plan\nship it\n",
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(writeResult.Metadata, core.ToolResultIsErrorMeta))
	assert.Equal(t, "notes/plan.md", detailsMap(writeResult)["path"])

	readResult, err := toolMap["read"].Execute(ctx, map[string]any{
		"path": "notes/plan.md",
	})
	require.NoError(t, err)
	assert.Equal(t, "# Plan\nship it", readResult.Metadata[core.ToolResultModelTextMeta])

	editResult, err := toolMap["edit"].Execute(ctx, map[string]any{
		"path":     "notes/plan.md",
		"old_text": "ship it",
		"new_text": "ship it safely",
	})
	require.NoError(t, err)
	assert.False(t, metadataBool(editResult.Metadata, core.ToolResultIsErrorMeta))

	readUpdated, err := toolMap["read"].Execute(ctx, map[string]any{
		"path": "notes/plan.md",
	})
	require.NoError(t, err)
	assert.Contains(t, readUpdated.Metadata[core.ToolResultDisplayTextMeta], "ship it safely")

	listResult, err := toolMap["ls"].Execute(ctx, map[string]any{
		"path":      ".",
		"recursive": true,
	})
	require.NoError(t, err)
	modelText, _ := listResult.Metadata[core.ToolResultModelTextMeta].(string)
	assert.Contains(t, modelText, "notes/")
	assert.Contains(t, modelText, "notes/plan.md")
}

func TestFilesToolsRejectEscapes(t *testing.T) {
	root := t.TempDir()
	toolset, err := NewToolset(Config{Root: root})
	require.NoError(t, err)

	toolMap := byName(toolset.Tools())
	result, err := toolMap["read"].Execute(context.Background(), map[string]any{
		"path": "../outside.txt",
	})
	require.NoError(t, err)
	assert.True(t, metadataBool(result.Metadata, core.ToolResultIsErrorMeta))
	assert.Contains(t, result.Metadata[core.ToolResultModelTextMeta], "escapes workspace root")
}

func TestFilesToolsCloneAndValidate(t *testing.T) {
	toolset, err := NewToolset(Config{Root: t.TempDir()})
	require.NoError(t, err)

	writeTool := byName(toolset.Tools())["write"]
	require.Error(t, writeTool.Validate(map[string]any{"path": "only.txt"}))

	cloneable, ok := writeTool.(core.CloneableTool)
	require.True(t, ok)
	cloned := cloneable.CloneTool()
	require.NotNil(t, cloned)
	assert.Equal(t, writeTool.Name(), cloned.Name())
}

func TestFilesToolsetRootIsAbsolute(t *testing.T) {
	toolset, err := NewToolset(Config{Root: "."})
	require.NoError(t, err)
	assert.True(t, filepath.IsAbs(toolset.Root()))
}

func TestFilesToolsAllowSymlinkedRoot(t *testing.T) {
	base := t.TempDir()
	realRoot := filepath.Join(base, "real")
	linkRoot := filepath.Join(base, "link")
	require.NoError(t, os.MkdirAll(realRoot, 0o755))
	require.NoError(t, os.Symlink(realRoot, linkRoot))

	toolset, err := NewToolset(Config{Root: linkRoot})
	require.NoError(t, err)

	toolMap := byName(toolset.Tools())
	_, err = toolMap["write"].Execute(context.Background(), map[string]any{
		"path":    "nested/file.txt",
		"content": "hello",
	})
	require.NoError(t, err)

	readResult, err := toolMap["read"].Execute(context.Background(), map[string]any{
		"path": "nested/file.txt",
	})
	require.NoError(t, err)
	assert.Equal(t, "hello", readResult.Metadata[core.ToolResultModelTextMeta])
	assert.FileExists(t, filepath.Join(realRoot, "nested", "file.txt"))
}

func byName(tools []core.Tool) map[string]core.Tool {
	result := make(map[string]core.Tool, len(tools))
	for _, tool := range tools {
		result[tool.Name()] = tool
	}
	return result
}

func metadataBool(metadata map[string]any, key string) bool {
	value, _ := metadata[key].(bool)
	return value
}

func detailsMap(result core.ToolResult) map[string]any {
	details, _ := result.Annotations[core.ToolResultDetailsAnnotation].(map[string]any)
	return details
}
