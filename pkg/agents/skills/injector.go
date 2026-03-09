package skills

import (
	"context"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
)

// Injector loads persisted skills and applies them to an optimizable agent.
type Injector struct {
	Store Store
}

// NewInjector creates an injector backed by the provided store.
func NewInjector(store Store) *Injector {
	return &Injector{Store: store}
}

// InjectBest loads the best skill for a domain and writes it into the agent's skill pack artifact.
// When no skill exists for the domain, the agent is left unchanged and nil is returned.
func (i *Injector) InjectBest(ctx context.Context, agent optimize.OptimizableAgent, domain string) (*Skill, error) {
	if i == nil || i.Store == nil {
		return nil, nil
	}

	normalizedDomain := strings.TrimSpace(domain)
	if normalizedDomain == "" {
		return nil, nil
	}

	skill, err := i.Store.Best(ctx, normalizedDomain)
	if err != nil || skill == nil {
		return skill, err
	}

	if err := Apply(agent, *skill); err != nil {
		return nil, err
	}

	applied := skill.Clone()
	return &applied, nil
}

// Apply writes a concrete skill into the agent's skill pack artifact while preserving the rest of the artifact set.
func Apply(agent optimize.OptimizableAgent, skill Skill) error {
	artifacts := agent.GetArtifacts()
	if artifacts.Text == nil {
		artifacts.Text = make(map[optimize.ArtifactKey]string)
	}
	artifacts.Text[optimize.ArtifactSkillPack] = skill.Content
	return agent.SetArtifacts(artifacts)
}
