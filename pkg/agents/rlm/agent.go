package rlm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

const (
	ArtifactMaxIterations            = "rlm_max_iterations"
	ArtifactMaxTokens                = "rlm_max_tokens"
	ArtifactUseIterationDemos        = "rlm_use_iteration_demos"
	ArtifactCompactIterationPrompt   = "rlm_compact_iteration_instructions"
	ArtifactAdaptiveIterationEnabled = "rlm_adaptive_iteration_enabled"
	defaultAgentIDPrefix             = "rlm-agent"
	agentTypeRLM                     = "rlm"
)

// Agent wraps an RLM module with the agents.Agent and optimize.OptimizableAgent contracts.
type Agent struct {
	id        string
	module    *modrlm.RLM
	memory    agents.Memory
	artifacts optimize.AgentArtifacts
	lastTrace *agents.ExecutionTrace
	mu        sync.RWMutex
}

var _ optimize.OptimizableAgent = (*Agent)(nil)

// NewAgent creates an optimizable agent around an RLM module.
func NewAgent(id string, module *modrlm.RLM) *Agent {
	if strings.TrimSpace(id) == "" {
		id = fmt.Sprintf("%s-%d", defaultAgentIDPrefix, time.Now().UnixNano())
	}

	agent := &Agent{
		id:     id,
		module: module,
	}
	if module != nil {
		agent.artifacts = artifactsFromConfig(module.Config())
	}
	return agent
}

// Execute runs the wrapped RLM module and records the most recent execution trace.
func (a *Agent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	if a == nil {
		err := fmt.Errorf("rlm agent is not initialized")
		return nil, err
	}
	if a.module == nil {
		err := fmt.Errorf("rlm agent is not initialized")
		a.storeTrace(buildMinimalFailureTrace("", input, err))
		return nil, err
	}

	contextPayload, ok := input["context"]
	if !ok {
		err := modrlm.ErrMissingContext
		a.storeTrace(buildMinimalFailureTrace(a.id, input, err))
		return nil, err
	}
	query, ok := input["query"].(string)
	if !ok {
		err := modrlm.ErrMissingQuery
		a.storeTrace(buildMinimalFailureTrace(a.id, input, err))
		return nil, err
	}

	result, trace, err := a.module.CompleteWithTrace(ctx, contextPayload, query)

	output := map[string]interface{}{}
	if result != nil {
		output["answer"] = result.Response
	}

	a.storeTrace(a.buildExecutionTrace(input, output, err, trace))
	if err != nil {
		return nil, err
	}
	return output, nil
}

// GetCapabilities returns nil because the RLM-backed agent does not expose tool capabilities through the agent interface.
func (a *Agent) GetCapabilities() []core.Tool {
	return nil
}

// GetMemory returns the configured memory store, if any.
func (a *Agent) GetMemory() agents.Memory {
	if a == nil {
		return nil
	}
	return a.memory
}

// GetArtifacts returns a defensive copy of the current artifact set.
func (a *Agent) GetArtifacts() optimize.AgentArtifacts {
	if a == nil {
		return optimize.AgentArtifacts{}
	}

	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.artifacts.Clone()
}

// SetArtifacts applies a full artifact set to the wrapped RLM module.
func (a *Agent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	if a == nil || a.module == nil {
		return fmt.Errorf("rlm agent is not initialized")
	}

	cfg := a.module.Config()
	applyArtifactsToConfig(&cfg, artifacts)
	a.module.SetConfig(cfg)

	a.mu.Lock()
	a.artifacts = artifactsFromConfig(cfg)
	a.mu.Unlock()

	return nil
}

// Clone returns a clone-safe copy of the agent and the wrapped RLM module.
func (a *Agent) Clone() (optimize.OptimizableAgent, error) {
	if a == nil || a.module == nil {
		return nil, fmt.Errorf("rlm agent is not initialized")
	}

	clonedModule, ok := a.module.Clone().(*modrlm.RLM)
	if !ok {
		return nil, fmt.Errorf("rlm agent clone produced unexpected module type")
	}

	cloned := &Agent{
		id:     a.id,
		module: clonedModule,
		// Memory is intentionally shared because agents.Memory has no clone contract today.
		// This mirrors existing agent behavior and keeps the wrapper compatible with stateful stores.
		memory:    a.memory,
		artifacts: a.GetArtifacts(),
	}
	if trace := a.LastExecutionTrace(); trace != nil {
		cloned.lastTrace = trace
	}
	return cloned, nil
}

// LastExecutionTrace returns the most recent execution trace, if any.
func (a *Agent) LastExecutionTrace() *agents.ExecutionTrace {
	if a == nil {
		return nil
	}

	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.lastTrace.Clone()
}

func (a *Agent) storeTrace(trace *agents.ExecutionTrace) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.lastTrace = trace
}

func (a *Agent) buildExecutionTrace(input map[string]interface{}, output map[string]interface{}, err error, trace *modrlm.RLMTrace) *agents.ExecutionTrace {
	if trace == nil {
		return buildMinimalFailureTrace(a.id, input, err)
	}

	steps := make([]agents.TraceStep, 0, len(trace.Steps))
	toolUsageCount := make(map[string]int)
	status := agents.TraceStatusSuccess

	for _, step := range trace.Steps {
		toolName := ""
		if step.Action != "" && step.Action != "final" {
			toolName = step.Action
			toolUsageCount[toolName]++
		}
		if !step.Success && status == agents.TraceStatusSuccess {
			status = agents.TraceStatusPartial
		}

		steps = append(steps, agents.TraceStep{
			Index:       step.Index,
			Thought:     step.Thought,
			ActionRaw:   step.Action,
			Tool:        toolName,
			Observation: step.Observation,
			Duration:    step.Duration,
			Success:     step.Success,
			Error:       step.Error,
		})
	}

	if err != nil {
		status = agents.TraceStatusFailure
	}

	tokenUsage := map[string]int64{
		"prompt_tokens":     int64(trace.Usage.PromptTokens),
		"completion_tokens": int64(trace.Usage.CompletionTokens),
		"total_tokens":      int64(trace.Usage.TotalTokens),
	}

	contextMetadata := map[string]interface{}{
		"iterations":                     trace.Iterations,
		"termination_cause":              trace.TerminationCause,
		"adaptive_iteration_enabled":     a.artifacts.Bool[ArtifactAdaptiveIterationEnabled],
		"compact_iteration_instructions": a.artifacts.Bool[ArtifactCompactIterationPrompt],
		"use_iteration_demos":            a.artifacts.Bool[ArtifactUseIterationDemos],
		"max_iterations":                 a.artifacts.Int[ArtifactMaxIterations],
		"max_tokens":                     a.artifacts.Int[ArtifactMaxTokens],
	}

	executionTrace := &agents.ExecutionTrace{
		AgentID:          a.id,
		AgentType:        agentTypeRLM,
		Task:             inputString(input, "query"),
		Input:            core.ShallowCopyMap(input),
		Output:           core.ShallowCopyMap(output),
		Steps:            steps,
		Status:           status,
		StartedAt:        trace.StartedAt,
		CompletedAt:      trace.CompletedAt,
		ProcessingTime:   trace.ProcessingTime,
		TokenUsage:       tokenUsage,
		ToolUsageCount:   toolUsageCount,
		ContextMetadata:  contextMetadata,
		TerminationCause: trace.TerminationCause,
	}
	if err != nil {
		executionTrace.Error = err.Error()
	} else if trace.Error != "" {
		executionTrace.Error = trace.Error
	}

	return executionTrace
}

func buildMinimalFailureTrace(agentID string, input map[string]interface{}, err error) *agents.ExecutionTrace {
	trace := &agents.ExecutionTrace{
		AgentID:          agentID,
		AgentType:        agentTypeRLM,
		Task:             inputString(input, "query"),
		Input:            core.ShallowCopyMap(input),
		Output:           map[string]interface{}{},
		Status:           agents.TraceStatusFailure,
		StartedAt:        time.Now(),
		CompletedAt:      time.Now(),
		ProcessingTime:   0,
		ToolUsageCount:   map[string]int{},
		ContextMetadata:  map[string]interface{}{},
		TerminationCause: "error",
	}
	if err != nil {
		trace.Error = err.Error()
	}
	return trace
}

func inputString(input map[string]interface{}, key string) string {
	if input == nil {
		return ""
	}
	raw, ok := input[key]
	if !ok {
		return ""
	}
	value, ok := raw.(string)
	if !ok {
		return ""
	}
	return value
}

func artifactsFromConfig(cfg modrlm.Config) optimize.AgentArtifacts {
	artifacts := optimize.AgentArtifacts{
		Text: map[optimize.ArtifactKey]string{
			optimize.ArtifactRLMOuterPrompt:     cfg.OuterInstruction,
			optimize.ArtifactRLMIterationPrompt: cfg.IterationInstruction,
		},
		Int: map[string]int{
			ArtifactMaxIterations: cfg.MaxIterations,
			ArtifactMaxTokens:     cfg.MaxTokens,
		},
		Bool: map[string]bool{
			ArtifactUseIterationDemos:        cfg.UseIterationDemos,
			ArtifactCompactIterationPrompt:   cfg.CompactIterationInstructions,
			ArtifactAdaptiveIterationEnabled: cfg.AdaptiveIteration != nil && cfg.AdaptiveIteration.Enabled,
		},
	}

	if artifacts.Text[optimize.ArtifactRLMOuterPrompt] == "" {
		artifacts.Text[optimize.ArtifactRLMOuterPrompt] = modrlm.DefaultOuterInstruction()
	}
	if artifacts.Text[optimize.ArtifactRLMIterationPrompt] == "" {
		artifacts.Text[optimize.ArtifactRLMIterationPrompt] = modrlm.DefaultIterationInstruction(cfg.CompactIterationInstructions)
	}

	return artifacts
}

func applyArtifactsToConfig(cfg *modrlm.Config, artifacts optimize.AgentArtifacts) {
	if cfg == nil {
		return
	}

	if value, ok := artifacts.Text[optimize.ArtifactRLMOuterPrompt]; ok && strings.TrimSpace(value) != "" {
		cfg.OuterInstruction = value
	}
	if value, ok := artifacts.Text[optimize.ArtifactRLMIterationPrompt]; ok && strings.TrimSpace(value) != "" {
		cfg.IterationInstruction = value
	}

	if value, ok := artifacts.Int[ArtifactMaxIterations]; ok && value > 0 {
		cfg.MaxIterations = value
	}
	if value, ok := artifacts.Int[ArtifactMaxTokens]; ok && value >= 0 {
		cfg.MaxTokens = value
	}

	if value, ok := artifacts.Bool[ArtifactUseIterationDemos]; ok {
		cfg.UseIterationDemos = value
	}
	if value, ok := artifacts.Bool[ArtifactCompactIterationPrompt]; ok {
		cfg.CompactIterationInstructions = value
	}
	if value, ok := artifacts.Bool[ArtifactAdaptiveIterationEnabled]; ok {
		if value {
			if cfg.AdaptiveIteration == nil {
				defaultAdaptive := modrlm.DefaultAdaptiveIterationConfig()
				cfg.AdaptiveIteration = &defaultAdaptive
			} else {
				cfg.AdaptiveIteration.Enabled = true
			}
		} else {
			cfg.AdaptiveIteration = nil
		}
	}
}
