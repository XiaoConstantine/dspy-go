package skills

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryStore_SaveLoadAndBest(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	require.NoError(t, store.Save(ctx, Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Use the repository runbook.",
		Version: 1,
		Metadata: map[string]string{
			"source": "seed",
		},
	}))
	require.NoError(t, store.Save(ctx, Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Use the updated repository runbook.",
		Version: 2,
	}))
	require.NoError(t, store.Save(ctx, Skill{
		Name:    "other-skill",
		Domain:  "repo:other",
		Content: "Other domain skill.",
		Version: 1,
	}))

	loaded, err := store.Load(ctx, "repo:test")
	require.NoError(t, err)
	require.Len(t, loaded, 2)
	assert.Equal(t, 2, loaded[0].Version)
	assert.Equal(t, 1, loaded[1].Version)

	best, err := store.Best(ctx, "repo:test")
	require.NoError(t, err)
	require.NotNil(t, best)
	assert.Equal(t, 2, best.Version)
	assert.Equal(t, "Use the updated repository runbook.", best.Content)
}

func TestFileStore_PersistsAcrossHandles(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "skills.json")

	store := NewFileStore(path)
	require.NoError(t, store.Save(ctx, Skill{
		Name:      "repo-skill",
		Domain:    "repo:test",
		Content:   "Use repo-aware search first.",
		Version:   3,
		CreatedAt: time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC),
		Metadata: map[string]string{
			"model": "gpt-5-mini",
		},
	}))

	reloaded := NewFileStore(path)
	loaded, err := reloaded.Load(ctx, "repo:test")
	require.NoError(t, err)
	require.Len(t, loaded, 1)
	assert.Equal(t, 3, loaded[0].Version)
	assert.Equal(t, "Use repo-aware search first.", loaded[0].Content)
	assert.Equal(t, "gpt-5-mini", loaded[0].Metadata["model"])

	best, err := reloaded.Best(ctx, "repo:test")
	require.NoError(t, err)
	require.NotNil(t, best)
	assert.Equal(t, 3, best.Version)
}

func TestFileStore_Save_ReplacesMatchingSkillVersion(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "skills.json")
	store := NewFileStore(path)

	require.NoError(t, store.Save(ctx, Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "v1 content",
		Version: 1,
	}))
	first, err := store.Best(ctx, "repo:test")
	require.NoError(t, err)
	require.NotNil(t, first)

	require.NoError(t, store.Save(ctx, Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "replacement content",
		Version: 1,
	}))

	loaded, err := store.Load(ctx, "repo:test")
	require.NoError(t, err)
	require.Len(t, loaded, 1)
	assert.Equal(t, "replacement content", loaded[0].Content)
	assert.Equal(t, first.CreatedAt, loaded[0].CreatedAt)
}

func TestInjector_InjectBest_AppliesSkillPackToAgent(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	require.NoError(t, store.Save(ctx, Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Prefer deterministic repo-aware debugging steps.",
		Version: 2,
	}))

	injector := NewInjector(store)
	agent := &stubOptimizableAgent{artifacts: optimize.AgentArtifacts{Text: map[optimize.ArtifactKey]string{}}}
	require.NoError(t, agent.SetArtifacts(optimize.AgentArtifacts{
		Text: map[optimize.ArtifactKey]string{
			optimize.ArtifactToolPolicy: "Prefer low-cost tools first.",
		},
	}))

	applied, err := injector.InjectBest(ctx, agent, "repo:test")
	require.NoError(t, err)
	require.NotNil(t, applied)
	assert.Equal(t, 2, applied.Version)

	artifacts := agent.GetArtifacts()
	assert.Equal(t, "Prefer deterministic repo-aware debugging steps.", artifacts.Text[optimize.ArtifactSkillPack])
	assert.Equal(t, "Prefer low-cost tools first.", artifacts.Text[optimize.ArtifactToolPolicy])
}

func TestInjector_InjectBest_NoSkillIsNoOp(t *testing.T) {
	injector := NewInjector(NewMemoryStore())
	agent := &stubOptimizableAgent{
		artifacts: optimize.AgentArtifacts{
			Text: map[optimize.ArtifactKey]string{
				optimize.ArtifactToolPolicy: "keep existing",
			},
		},
	}

	applied, err := injector.InjectBest(context.Background(), agent, "repo:missing")
	require.NoError(t, err)
	assert.Nil(t, applied)
	assert.Equal(t, "keep existing", agent.artifacts.Text[optimize.ArtifactToolPolicy])
}

type stubOptimizableAgent struct {
	artifacts optimize.AgentArtifacts
}

func (s *stubOptimizableAgent) Execute(context.Context, map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

func (s *stubOptimizableAgent) GetCapabilities() []core.Tool { return nil }

func (s *stubOptimizableAgent) GetMemory() agents.Memory { return nil }

func (s *stubOptimizableAgent) GetArtifacts() optimize.AgentArtifacts {
	return s.artifacts.Clone()
}

func (s *stubOptimizableAgent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	s.artifacts = artifacts.Clone()
	return nil
}

func (s *stubOptimizableAgent) Clone() (optimize.OptimizableAgent, error) {
	return &stubOptimizableAgent{artifacts: s.artifacts.Clone()}, nil
}
