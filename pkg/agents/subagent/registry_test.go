package subagent

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistry_RegisterAndTool(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	err := registry.Register("", ToolConfig{
		Name:        "research",
		Description: "Research worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{output: map[string]any{"final_answer": "ok", "completed": true}}, nil
		},
		SessionPolicy: SessionPolicyDerived,
	})
	require.NoError(t, err)

	tool, err := registry.Tool("research")
	require.NoError(t, err)
	info, ok := InfoFromTool(tool)
	require.True(t, ok)
	assert.Equal(t, "research", info.Name)
	assert.Equal(t, "derived", info.SessionPolicy)
}

func TestRegistry_RejectsDuplicateName(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	cfg := ToolConfig{
		Name:        "verify",
		Description: "Verifier worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{output: map[string]any{"final_answer": "ok"}}, nil
		},
	}
	require.NoError(t, registry.Register("", cfg))

	err := registry.Register("", cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already registered")
}

func TestRegistry_ToolsReturnsClonedToolsInOrder(t *testing.T) {
	t.Parallel()

	registry := NewRegistry()
	require.NoError(t, registry.Register("alpha", ToolConfig{
		Description: "Alpha worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{output: map[string]any{"final_answer": "a"}}, nil
		},
	}))
	require.NoError(t, registry.Register("beta", ToolConfig{
		Description: "Beta worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{output: map[string]any{"final_answer": "b"}}, nil
		},
	}))

	tools := registry.Tools()
	require.Len(t, tools, 2)
	assert.Equal(t, []string{"alpha", "beta"}, []string{tools[0].Name(), tools[1].Name()})

	first, err := registry.Tool("alpha")
	require.NoError(t, err)
	assert.NotSame(t, first, tools[0])

	_, cloneable := tools[0].(core.CloneableTool)
	assert.True(t, cloneable)
}
