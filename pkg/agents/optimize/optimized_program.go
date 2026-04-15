package optimize

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// OptimizationTargetKind identifies the value family represented by a target.
type OptimizationTargetKind string

const (
	OptimizationTargetText OptimizationTargetKind = "text"
	OptimizationTargetInt  OptimizationTargetKind = "int"
	OptimizationTargetBool OptimizationTargetKind = "bool"
)

const (
	optimizedAgentProgramSchemaV1  = "dspy-go.optimized-agent-program"
	optimizedAgentProgramVersionV1 = 1
)

// OptimizationTargetDescriptor maps a stable user-facing target ID onto one
// mutable artifact slot.
type OptimizationTargetDescriptor struct {
	ID          string                 `json:"id"`
	Kind        OptimizationTargetKind `json:"kind"`
	Description string                 `json:"description,omitempty"`
	ArtifactKey ArtifactKey            `json:"artifact_key,omitempty"`
	IntKey      string                 `json:"int_key,omitempty"`
	BoolKey     string                 `json:"bool_key,omitempty"`
}

// OptimizedAgentProgram is the shared persisted envelope for optimized agent
// artifacts across native, ReAct, and RLM-backed agents.
type OptimizedAgentProgram struct {
	Schema      string                 `json:"schema"`
	Version     int                    `json:"version"`
	AgentType   string                 `json:"agent_type,omitempty"`
	TargetOrder []string               `json:"target_order,omitempty"`
	Text        map[string]string      `json:"text,omitempty"`
	Int         map[string]int         `json:"int,omitempty"`
	Bool        map[string]bool        `json:"bool,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type optimizationTargetLister interface {
	ListOptimizationTargets() []OptimizationTargetDescriptor
}

type optimizationAgentTyper interface {
	OptimizationAgentType() string
}

type optimizationArtifactUpdater interface {
	UpdateArtifacts(func(AgentArtifacts) (AgentArtifacts, error)) error
}

func normalizeOptimizationTargetDescriptors(descriptors []OptimizationTargetDescriptor) []OptimizationTargetDescriptor {
	normalized := make([]OptimizationTargetDescriptor, 0, len(descriptors))
	seen := make(map[string]struct{}, len(descriptors))
	for _, descriptor := range descriptors {
		descriptor.ID = strings.TrimSpace(descriptor.ID)
		descriptor.IntKey = strings.TrimSpace(descriptor.IntKey)
		descriptor.BoolKey = strings.TrimSpace(descriptor.BoolKey)
		descriptor.Description = strings.TrimSpace(descriptor.Description)
		if descriptor.ID == "" {
			continue
		}
		if _, exists := seen[descriptor.ID]; exists {
			continue
		}
		switch descriptor.Kind {
		case OptimizationTargetText:
			if descriptor.ArtifactKey == "" {
				continue
			}
		case OptimizationTargetInt:
			if descriptor.IntKey == "" {
				continue
			}
		case OptimizationTargetBool:
			if descriptor.BoolKey == "" {
				continue
			}
		default:
			continue
		}
		seen[descriptor.ID] = struct{}{}
		normalized = append(normalized, descriptor)
	}
	return normalized
}

func defaultOptimizationTargetsForArtifacts(artifacts AgentArtifacts) []OptimizationTargetDescriptor {
	descriptors := make([]OptimizationTargetDescriptor, 0, len(artifacts.Text)+len(artifacts.Int)+len(artifacts.Bool))
	for key := range artifacts.Text {
		descriptors = append(descriptors, OptimizationTargetDescriptor{
			ID:          fmt.Sprintf("artifact.text.%s", key),
			Kind:        OptimizationTargetText,
			Description: fmt.Sprintf("Text artifact %q", key),
			ArtifactKey: key,
		})
	}
	for key := range artifacts.Int {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		descriptors = append(descriptors, OptimizationTargetDescriptor{
			ID:          "artifact.int." + key,
			Kind:        OptimizationTargetInt,
			Description: fmt.Sprintf("Integer artifact %q", key),
			IntKey:      key,
		})
	}
	for key := range artifacts.Bool {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		descriptors = append(descriptors, OptimizationTargetDescriptor{
			ID:          "artifact.bool." + key,
			Kind:        OptimizationTargetBool,
			Description: fmt.Sprintf("Boolean artifact %q", key),
			BoolKey:     key,
		})
	}
	sort.Slice(descriptors, func(i, j int) bool {
		return descriptors[i].ID < descriptors[j].ID
	})
	return normalizeOptimizationTargetDescriptors(descriptors)
}

func optimizationTargetsForAgent(agent OptimizableAgent, artifacts AgentArtifacts) ([]OptimizationTargetDescriptor, string) {
	var descriptors []OptimizationTargetDescriptor
	if lister, ok := agent.(optimizationTargetLister); ok {
		descriptors = normalizeOptimizationTargetDescriptors(lister.ListOptimizationTargets())
	}
	if len(descriptors) == 0 {
		descriptors = defaultOptimizationTargetsForArtifacts(artifacts)
	}

	agentType := ""
	if typer, ok := agent.(optimizationAgentTyper); ok {
		agentType = strings.TrimSpace(typer.OptimizationAgentType())
	}

	return descriptors, agentType
}

// Validate verifies that the program envelope uses a supported schema/version.
func (p *OptimizedAgentProgram) Validate() error {
	if p == nil {
		return fmt.Errorf("optimize: nil optimized agent program")
	}
	if strings.TrimSpace(p.Schema) != optimizedAgentProgramSchemaV1 {
		return fmt.Errorf("optimize: unsupported optimized agent program schema %q", p.Schema)
	}
	if p.Version != optimizedAgentProgramVersionV1 {
		return fmt.Errorf("optimize: unsupported optimized agent program version %d", p.Version)
	}
	return nil
}

// ToArtifacts converts the persisted target values back into an AgentArtifacts overlay.
func (p *OptimizedAgentProgram) ToArtifacts(descriptors []OptimizationTargetDescriptor) (AgentArtifacts, error) {
	if err := p.Validate(); err != nil {
		return AgentArtifacts{}, err
	}

	normalized := normalizeOptimizationTargetDescriptors(descriptors)
	descriptorByID := make(map[string]OptimizationTargetDescriptor, len(normalized))
	for _, descriptor := range normalized {
		descriptorByID[descriptor.ID] = descriptor
	}

	artifacts := AgentArtifacts{
		Text: make(map[ArtifactKey]string),
		Int:  make(map[string]int),
		Bool: make(map[string]bool),
	}

	for id, value := range p.Text {
		descriptor, exists := descriptorByID[id]
		if !exists || descriptor.Kind != OptimizationTargetText {
			continue
		}
		artifacts.Text[descriptor.ArtifactKey] = value
	}

	for id, value := range p.Int {
		descriptor, exists := descriptorByID[id]
		if !exists || descriptor.Kind != OptimizationTargetInt {
			continue
		}
		artifacts.Int[descriptor.IntKey] = value
	}

	for id, value := range p.Bool {
		descriptor, exists := descriptorByID[id]
		if !exists || descriptor.Kind != OptimizationTargetBool {
			continue
		}
		artifacts.Bool[descriptor.BoolKey] = value
	}

	return artifacts, nil
}

// ExportOptimizedAgentProgram exports the current mutable artifact state behind
// an agent into the shared persisted envelope.
func ExportOptimizedAgentProgram(agent OptimizableAgent) (*OptimizedAgentProgram, error) {
	if agent == nil {
		return nil, fmt.Errorf("optimize: nil agent")
	}
	return ExportOptimizedAgentProgramFromArtifacts(agent, agent.GetArtifacts())
}

// ExportOptimizedAgentProgramFromArtifacts exports a provided artifact set using
// the target mapping for the given agent.
func ExportOptimizedAgentProgramFromArtifacts(agent OptimizableAgent, artifacts AgentArtifacts) (*OptimizedAgentProgram, error) {
	if agent == nil {
		return nil, fmt.Errorf("optimize: nil agent")
	}

	descriptors, agentType := optimizationTargetsForAgent(agent, artifacts)
	program := &OptimizedAgentProgram{
		Schema:      optimizedAgentProgramSchemaV1,
		Version:     optimizedAgentProgramVersionV1,
		AgentType:   agentType,
		TargetOrder: make([]string, 0, len(descriptors)),
		Text:        make(map[string]string),
		Int:         make(map[string]int),
		Bool:        make(map[string]bool),
	}

	for _, descriptor := range descriptors {
		program.TargetOrder = append(program.TargetOrder, descriptor.ID)
		switch descriptor.Kind {
		case OptimizationTargetText:
			if value, ok := artifacts.Text[descriptor.ArtifactKey]; ok {
				program.Text[descriptor.ID] = value
			}
		case OptimizationTargetInt:
			if value, ok := artifacts.Int[descriptor.IntKey]; ok {
				program.Int[descriptor.ID] = value
			}
		case OptimizationTargetBool:
			if value, ok := artifacts.Bool[descriptor.BoolKey]; ok {
				program.Bool[descriptor.ID] = value
			}
		}
	}

	return program, nil
}

// ApplyOptimizedAgentProgram validates and applies a persisted optimized-agent
// program onto the provided agent by converting it back into artifact values.
func ApplyOptimizedAgentProgram(agent OptimizableAgent, program *OptimizedAgentProgram) error {
	if agent == nil {
		return fmt.Errorf("optimize: nil agent")
	}
	if program == nil {
		return fmt.Errorf("optimize: nil optimized agent program")
	}

	if updater, ok := agent.(optimizationArtifactUpdater); ok {
		return updater.UpdateArtifacts(func(current AgentArtifacts) (AgentArtifacts, error) {
			descriptors, agentType := optimizationTargetsForAgent(agent, current)
			if program.AgentType != "" && agentType != "" && program.AgentType != agentType {
				return AgentArtifacts{}, fmt.Errorf("optimize: optimized agent program type %q does not match agent type %q", program.AgentType, agentType)
			}

			overlay, err := program.ToArtifacts(descriptors)
			if err != nil {
				return AgentArtifacts{}, err
			}
			return mergeArtifacts(current, overlay), nil
		})
	}

	current := agent.GetArtifacts()
	descriptors, agentType := optimizationTargetsForAgent(agent, current)
	if program.AgentType != "" && agentType != "" && program.AgentType != agentType {
		return fmt.Errorf("optimize: optimized agent program type %q does not match agent type %q", program.AgentType, agentType)
	}

	overlay, err := program.ToArtifacts(descriptors)
	if err != nil {
		return err
	}

	return agent.SetArtifacts(mergeArtifacts(current, overlay))
}

// WriteOptimizedAgentProgram serializes the program envelope to disk.
func WriteOptimizedAgentProgram(path string, program *OptimizedAgentProgram) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("optimize: output path is required")
	}
	if err := program.Validate(); err != nil {
		return err
	}

	data, err := json.MarshalIndent(program, "", "  ")
	if err != nil {
		return fmt.Errorf("optimize: marshal optimized agent program: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("optimize: write optimized agent program: %w", err)
	}
	return nil
}

// ReadOptimizedAgentProgram deserializes a persisted optimized-agent program.
func ReadOptimizedAgentProgram(path string) (*OptimizedAgentProgram, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("optimize: input path is required")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("optimize: read optimized agent program: %w", err)
	}

	var program OptimizedAgentProgram
	if err := json.Unmarshal(data, &program); err != nil {
		return nil, fmt.Errorf("optimize: unmarshal optimized agent program: %w", err)
	}
	if err := program.Validate(); err != nil {
		return nil, err
	}
	return &program, nil
}
