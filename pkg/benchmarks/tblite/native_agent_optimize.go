package tblite

import (
	"context"
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
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
		TaskID:           stringValue(input["task_id"]),
		Instruction:      stringValue(input["instruction"]),
		TaskDir:          stringValue(input["task_dir"]),
		WorkingDirectory: stringValue(input["working_directory"]),
		EnvironmentDir:   stringValue(input["environment_dir"]),
		TestsDir:         stringValue(input["tests_dir"]),
		TestScriptPath:   stringValue(input["test_script_path"]),
		DockerImage:      stringValue(input["docker_image"]),
		MaxTurns:         intValue(input["max_turns"]),
		AgentTimeout:     durationValue(input["agent_timeout"]),
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
			req.ContainerEnv = append(req.ContainerEnv, stringValue(item))
		}
	}
	req.ContainerID = stringValue(input["container_id"])
	return req, nil
}

func stringValue(value interface{}) string {
	if str, ok := value.(string); ok {
		return str
	}
	return ""
}

func intValue(value interface{}) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func durationValue(value interface{}) time.Duration {
	switch typed := value.(type) {
	case time.Duration:
		return typed
	case int:
		return time.Duration(typed)
	case int32:
		return time.Duration(typed)
	case int64:
		return time.Duration(typed)
	case float64:
		return time.Duration(typed)
	default:
		return 0
	}
}
