package optimize

import "github.com/XiaoConstantine/dspy-go/pkg/agents"

// ArtifactKey identifies a mutable artifact on an optimizable agent.
type ArtifactKey string

const (
	ArtifactSkillPack          ArtifactKey = "skill_pack"
	ArtifactPlannerPrompt      ArtifactKey = "planner_prompt"
	ArtifactToolPolicy         ArtifactKey = "tool_policy"
	ArtifactMemoryTemplate     ArtifactKey = "memory_template"
	ArtifactReflectionPrompt   ArtifactKey = "reflection_prompt"
	ArtifactContextPolicy      ArtifactKey = "context_policy"
	ArtifactRLMOuterPrompt     ArtifactKey = "rlm_outer_prompt"
	ArtifactRLMIterationPrompt ArtifactKey = "rlm_iteration_prompt"
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

func mergeArtifacts(base, overlay AgentArtifacts) AgentArtifacts {
	merged := base.Clone()
	if merged.Text == nil {
		merged.Text = make(map[ArtifactKey]string)
	}
	if merged.Int == nil {
		merged.Int = make(map[string]int)
	}
	if merged.Bool == nil {
		merged.Bool = make(map[string]bool)
	}

	for key, value := range overlay.Text {
		merged.Text[key] = value
	}
	for key, value := range overlay.Int {
		merged.Int[key] = value
	}
	for key, value := range overlay.Bool {
		merged.Bool[key] = value
	}

	return merged
}

// OptimizableAgent is a parallel interface that exposes mutable artifacts
// without widening the base agents.Agent contract.
//
// Implementations that also expose LastExecutionTrace() *agents.ExecutionTrace
// allow evaluators to attach richer step-level side information without
// forcing that method into the base interface.
type OptimizableAgent interface {
	agents.Agent

	GetArtifacts() AgentArtifacts
	SetArtifacts(AgentArtifacts) error
	Clone() (OptimizableAgent, error)
}
