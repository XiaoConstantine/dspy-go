package tblite

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
)

var _ optimize.OptimizableAgent = (*NativeAgent)(nil)

// Execute adapts the benchmark task contract to the generic agents.Agent API.
func (a *NativeAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	req, err := terminalTaskRequestFromInput(input)
	if err != nil {
		return nil, err
	}

	result, err := a.RunTask(ctx, req)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return map[string]interface{}{"completed": false, "error": "native agent returned nil result"}, nil
	}

	return map[string]interface{}{
		"completed":    result.Completed,
		"final_answer": result.FinalAnswer,
		"error":        result.Error,
		"tool_calls":   result.ToolCalls,
		"trace_path":   result.TracePath,
	}, nil
}

func (a *NativeAgent) GetCapabilities() []core.Tool {
	return nil
}

func (a *NativeAgent) GetMemory() agents.Memory {
	if a == nil {
		return nil
	}
	return a.memory
}

func (a *NativeAgent) GetArtifacts() optimize.AgentArtifacts {
	artifacts := optimize.AgentArtifacts{
		Text: make(map[optimize.ArtifactKey]string),
		Int:  make(map[string]int),
		Bool: make(map[string]bool),
	}
	if a == nil {
		return artifacts
	}
	artifacts.Text[optimize.ArtifactSkillPack] = a.config.SystemPrompt
	artifacts.Text[optimize.ArtifactToolPolicy] = a.config.ToolPolicy
	artifacts.Int["max_turns"] = a.config.MaxTurns
	return artifacts
}

func (a *NativeAgent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	if a == nil {
		return fmt.Errorf("tblite native agent is nil")
	}
	if prompt, ok := artifacts.Text[optimize.ArtifactSkillPack]; ok && prompt != "" {
		a.config.SystemPrompt = prompt
	}
	if policy, ok := artifacts.Text[optimize.ArtifactToolPolicy]; ok && policy != "" {
		a.config.ToolPolicy = policy
	}
	if maxTurns, ok := artifacts.Int["max_turns"]; ok && maxTurns > 0 {
		a.config.MaxTurns = maxTurns
	}
	return nil
}

func (a *NativeAgent) Clone() (optimize.OptimizableAgent, error) {
	if a == nil {
		return nil, fmt.Errorf("tblite native agent is nil")
	}
	return &NativeAgent{
		llm:    a.llm,
		config: a.config,
		memory: agents.NewInMemoryStore(),
	}, nil
}

func (a *NativeAgent) LastExecutionTrace() *agents.ExecutionTrace {
	if a == nil {
		return nil
	}
	a.traceMu.RLock()
	defer a.traceMu.RUnlock()
	return a.lastTrace.Clone()
}

func terminalTaskRequestFromInput(input map[string]interface{}) (TerminalTaskRequest, error) {
	if input == nil {
		return TerminalTaskRequest{}, fmt.Errorf("native agent input is required")
	}

	req := TerminalTaskRequest{
		TaskID:           agentutil.StringValue(input["task_id"]),
		Instruction:      agentutil.StringValue(input["instruction"]),
		TaskDir:          agentutil.StringValue(input["task_dir"]),
		WorkingDirectory: agentutil.StringValue(input["working_directory"]),
		EnvironmentDir:   agentutil.StringValue(input["environment_dir"]),
		TestsDir:         agentutil.StringValue(input["tests_dir"]),
		TestScriptPath:   agentutil.StringValue(input["test_script_path"]),
		DockerImage:      agentutil.StringValue(input["docker_image"]),
		MaxTurns:         agentutil.IntValue(input["max_turns"]),
		AgentTimeout:     agentutil.DurationValue(input["agent_timeout"]),
	}
	if req.TaskID == "" {
		return TerminalTaskRequest{}, fmt.Errorf("task_id is required")
	}
	if req.TaskDir == "" || req.EnvironmentDir == "" {
		return TerminalTaskRequest{}, fmt.Errorf("task_dir and environment_dir are required")
	}
	if rawEnv, ok := input["container_env"].([]string); ok {
		req.ContainerEnv = append([]string(nil), rawEnv...)
	} else if rawEnvAny, ok := input["container_env"].([]interface{}); ok {
		req.ContainerEnv = make([]string, 0, len(rawEnvAny))
		for _, item := range rawEnvAny {
			req.ContainerEnv = append(req.ContainerEnv, agentutil.StringValue(item))
		}
	}
	req.ContainerID = agentutil.StringValue(input["container_id"])
	return req, nil
}
