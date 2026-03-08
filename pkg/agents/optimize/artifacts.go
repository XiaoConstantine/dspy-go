package optimize

import "github.com/XiaoConstantine/dspy-go/pkg/agents"

// ArtifactKey identifies a mutable artifact on an optimizable agent.
type ArtifactKey string

const (
	ArtifactSkillPack        ArtifactKey = "skill_pack"
	ArtifactPlannerPrompt    ArtifactKey = "planner_prompt"
	ArtifactToolPolicy       ArtifactKey = "tool_policy"
	ArtifactMemoryTemplate   ArtifactKey = "memory_template"
	ArtifactReflectionPrompt ArtifactKey = "reflection_prompt"
	ArtifactContextPolicy    ArtifactKey = "context_policy"
)

// AgentArtifacts groups mutable agent configuration surfaced to optimizers.
type AgentArtifacts struct {
	Text map[ArtifactKey]string
	Int  map[string]int
	Bool map[string]bool
}

// Clone returns a deep copy of the artifact maps so parent and child candidates
// cannot accidentally share mutable state.
func (a AgentArtifacts) Clone() AgentArtifacts {
	cloned := AgentArtifacts{
		Text: make(map[ArtifactKey]string, len(a.Text)),
		Int:  make(map[string]int, len(a.Int)),
		Bool: make(map[string]bool, len(a.Bool)),
	}

	for key, value := range a.Text {
		cloned.Text[key] = value
	}
	for key, value := range a.Int {
		cloned.Int[key] = value
	}
	for key, value := range a.Bool {
		cloned.Bool[key] = value
	}

	return cloned
}

// OptimizableAgent is a parallel interface that exposes mutable artifacts
// without widening the base agents.Agent contract.
type OptimizableAgent interface {
	agents.Agent

	GetArtifacts() AgentArtifacts
	SetArtifacts(AgentArtifacts) error
	Clone() (OptimizableAgent, error)
}
