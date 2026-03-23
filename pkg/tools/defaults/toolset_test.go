package defaults

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultToolsetProvidesMinimalPack(t *testing.T) {
	toolset, err := NewToolset(Config{Root: t.TempDir()})
	require.NoError(t, err)

	names := make([]string, 0, len(toolset.Tools()))
	for _, tool := range toolset.Tools() {
		names = append(names, tool.Name())
	}
	assert.Equal(t, []string{"ls", "read", "write", "edit", "bash"}, names)
}

func TestDefaultToolsetReturnsCopies(t *testing.T) {
	toolset, err := NewToolset(Config{Root: t.TempDir()})
	require.NoError(t, err)

	first := toolset.Tools()
	second := toolset.Tools()
	require.Len(t, first, 5)
	require.Len(t, second, 5)
	first[0] = nil
	assert.NotNil(t, second[0])
}

func TestDefaultToolsetToolsAreCloneable(t *testing.T) {
	toolset, err := NewToolset(Config{Root: t.TempDir()})
	require.NoError(t, err)

	for _, tool := range toolset.Tools() {
		cloneable, ok := tool.(core.CloneableTool)
		require.True(t, ok, tool.Name())
		assert.NotNil(t, cloneable.CloneTool(), tool.Name())
	}
}
