package tblite

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	sharednative "github.com/XiaoConstantine/dspy-go/pkg/agents/native"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
)

const (
	traceFileName = ".dspy_tblite_trace.json"
	debugFileName = ".dspy_tblite_debug.json"
)

// NativeAgentConfig controls the TBLite adapter over the shared native agent.
type NativeAgentConfig struct {
	MaxTurns       int
	MaxTokens      int
	Temperature    float64
	Toolset        ToolsetConfig
	SystemPrompt   string
	ToolPolicy     string
	FinishToolText string
}

// NativeAgent adapts the shared dspy-go native tool-calling harness to TBLite tasks.
type NativeAgent struct {
	llm       core.LLM
	config    NativeAgentConfig
	memory    agents.Memory
	traceMu   sync.RWMutex
	lastTrace *agents.ExecutionTrace
}

// ToolCallTrace captures a benchmark task run for later inspection.
type ToolCallTrace struct {
	TaskID      string              `json:"task_id"`
	Model       string              `json:"model"`
	Provider    string              `json:"provider"`
	Instruction string              `json:"instruction"`
	StartedAt   time.Time           `json:"started_at"`
	Duration    time.Duration       `json:"duration"`
	Completed   bool                `json:"completed"`
	FinalAnswer string              `json:"final_answer,omitempty"`
	Error       string              `json:"error,omitempty"`
	Steps       []ToolCallTraceStep `json:"steps"`
}

// ToolCallTraceStep captures one LLM/tool step in the benchmark loop.
type ToolCallTraceStep struct {
	Index         int               `json:"index"`
	AssistantText string            `json:"assistant_text,omitempty"`
	ToolName      string            `json:"tool_name,omitempty"`
	Arguments     map[string]any    `json:"arguments,omitempty"`
	Observation   string            `json:"observation,omitempty"`
	IsError       bool              `json:"is_error,omitempty"`
	Usage         TokenUsage        `json:"usage,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

type taskDebugSnapshot struct {
	TaskID         string `json:"task_id"`
	Model          string `json:"model"`
	Provider       string `json:"provider"`
	Error          string `json:"error"`
	Instruction    string `json:"instruction"`
	TaskPrompt     string `json:"task_prompt"`
	SystemPrompt   string `json:"system_prompt"`
	ToolPolicy     string `json:"tool_policy"`
	FinishToolText string `json:"finish_tool_text"`
	MaxTurns       int    `json:"max_turns"`
	TracePath      string `json:"trace_path,omitempty"`
}

// NewNativeAgent creates a benchmark agent that uses the shared native harness.
func NewNativeAgent(llm core.LLM, cfg NativeAgentConfig) (*NativeAgent, error) {
	if llm == nil {
		return nil, fmt.Errorf("llm is required")
	}
	if !supportsFunctionCalling(llm) {
		return nil, fmt.Errorf("llm %s does not support tool calling", llm.ModelID())
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 40
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 2048
	}
	if cfg.FinishToolText == "" {
		cfg.FinishToolText = "Call this tool when the task is complete and summarize what changed in the answer."
	}
	if strings.TrimSpace(cfg.SystemPrompt) == "" {
		cfg.SystemPrompt = defaultToolCallingSystemPrompt
	}
	if strings.TrimSpace(cfg.ToolPolicy) == "" {
		cfg.ToolPolicy = defaultToolCallingToolPolicy
	}
	return &NativeAgent{
		llm:    llm,
		config: cfg,
		memory: agents.NewInMemoryStore(),
	}, nil
}

// RunTask executes one materialized terminal benchmark task.
func (a *NativeAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	runCtx := ctx
	var cancel context.CancelFunc
	if req.AgentTimeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, req.AgentTimeout)
		defer cancel()
	}

	toolCfg := a.config.Toolset
	toolCfg.TaskRoot = req.TaskDir
	toolCfg.TestsDir = req.TestsDir
	toolCfg.ContainerEnvRoot = containerEnvValue(req.ContainerEnv, EnvTaskEnvDir, "")
	if req.ContainerID != "" && toolCfg.CommandRunner == nil {
		toolCfg.CommandRunner = dockerCommandRunner{
			runtime: &dockerTaskRuntime{
				containerID:      req.ContainerID,
				taskRoot:         req.TaskDir,
				environmentRoot:  req.EnvironmentDir,
				containerEnvRoot: containerEnvValue(req.ContainerEnv, EnvTaskEnvDir, containerEnvDir),
			},
			extraEnv:  req.ContainerEnv,
			shellPath: toolCfg.ShellPath,
		}
	}

	toolset, err := NewTerminalToolset(req.EnvironmentDir, toolCfg)
	if err != nil {
		return nil, err
	}

	sharedAgent, err := sharednative.NewAgent(a.llm, sharednative.Config{
		MaxTurns:       a.config.MaxTurns,
		MaxTokens:      a.config.MaxTokens,
		Temperature:    a.config.Temperature,
		SystemPrompt:   a.config.SystemPrompt,
		ToolPolicy:     a.config.ToolPolicy,
		FinishToolText: a.config.FinishToolText,
	})
	if err != nil {
		return nil, err
	}
	for _, tool := range toolset {
		if err := sharedAgent.RegisterTool(tool); err != nil {
			return nil, fmt.Errorf("register tool %q: %w", tool.Name(), err)
		}
	}

	taskPrompt := a.buildTaskPrompt(req)
	resultMap, err := sharedAgent.Execute(runCtx, map[string]interface{}{
		"task":      taskPrompt,
		"task_id":   req.TaskID,
		"max_turns": a.maxTurns(req),
	})
	a.traceMu.Lock()
	a.lastTrace = sharedAgent.LastExecutionTrace()
	a.traceMu.Unlock()
	if err != nil {
		nativeTrace := sharedAgent.LastNativeTrace()
		result := &TerminalTaskResult{
			Completed: false,
			Error:     err.Error(),
		}
		traceFile, _ := a.populateTraceResult(result, nativeTrace, req)
		_ = writeDebugSnapshot(req.TaskDir, taskDebugSnapshot{
			TaskID:         req.TaskID,
			Model:          a.llm.ModelID(),
			Provider:       a.llm.ProviderName(),
			Error:          err.Error(),
			Instruction:    req.Instruction,
			TaskPrompt:     taskPrompt,
			SystemPrompt:   a.config.SystemPrompt,
			ToolPolicy:     a.config.ToolPolicy,
			FinishToolText: a.config.FinishToolText,
			MaxTurns:       a.maxTurns(req),
			TracePath:      traceFile,
		})
		return result, nil
	}

	nativeTrace := sharedAgent.LastNativeTrace()
	if nativeTrace == nil {
		return nil, fmt.Errorf("shared native agent did not record a trace")
	}
	result := &TerminalTaskResult{
		Completed:   boolValue(resultMap["completed"]),
		FinalAnswer: agentutil.StringValue(resultMap["final_answer"]),
		Error:       agentutil.StringValue(resultMap["error"]),
	}
	if _, err := a.populateTraceResult(result, nativeTrace, req); err != nil {
		return nil, err
	}
	return result, nil
}

type dockerCommandRunner struct {
	runtime   *dockerTaskRuntime
	extraEnv  []string
	shellPath string
}

func (r dockerCommandRunner) Run(ctx context.Context, workingDir string, command string, extraEnv []string) ([]byte, error) {
	env := append([]string{}, r.extraEnv...)
	env = append(env, extraEnv...)
	return r.runtime.Exec(ctx, workingDir, r.shellPath, command, env)
}

// populateTraceResult fills trace-derived fields (Duration, ToolCalls,
// TokenUsage, TracePath, Metadata) on an existing result. It returns the
// trace file path for callers that need it independently.
func (a *NativeAgent) populateTraceResult(result *TerminalTaskResult, nativeTrace *sharednative.Trace, req TerminalTaskRequest) (string, error) {
	if result.Metadata == nil {
		result.Metadata = map[string]any{}
	}
	result.Metadata["provider"] = a.llm.ProviderName()
	result.Metadata["model"] = a.llm.ModelID()
	if nativeTrace == nil {
		return "", nil
	}
	trace := nativeTraceToToolCallTrace(nativeTrace, req.Instruction)
	traceFile, writeErr := writeTrace(req.TaskDir, trace)
	result.Duration = nativeTrace.Duration
	result.ToolCalls = countExecutedTools(trace.Steps)
	result.TokenUsage = tokenUsageFromShared(nativeTrace.TokenUsage)
	result.TracePath = traceFile
	result.Metadata["turns"] = len(trace.Steps)
	return traceFile, writeErr
}

func (a *NativeAgent) buildTaskPrompt(req TerminalTaskRequest) string {
	var b strings.Builder
	b.WriteString("TASK INSTRUCTION:\n")
	b.WriteString(req.Instruction)
	b.WriteString("\n\n")
	b.WriteString("WORKSPACE:\n")
	b.WriteString("- task id: " + req.TaskID + "\n")
	b.WriteString("- working directory: " + promptWorkingDirectory(req) + "\n")
	b.WriteString("- tests directory: " + promptTestsDirectory(req) + "\n")
	b.WriteString("- test script: " + promptTestScript(req) + "\n")
	b.WriteString(fmt.Sprintf("- max turns: %d\n\n", a.maxTurns(req)))
	b.WriteString("Use the available tools to inspect, edit, run commands, verify progress, and call Finish when done.\n")
	b.WriteString("If a health check, evaluation script, or verifier-style command succeeds and the required outputs exist, call Finish immediately instead of spending turns on extra inspection.\n")
	return b.String()
}

func promptWorkingDirectory(req TerminalTaskRequest) string {
	if root := containerEnvValue(req.ContainerEnv, EnvTaskEnvDir, ""); strings.TrimSpace(root) != "" {
		return root
	}
	if strings.TrimSpace(req.TaskDir) != "" {
		return containerEnvDir
	}
	return req.WorkingDirectory
}

func promptTestsDirectory(req TerminalTaskRequest) string {
	if root := containerEnvValue(req.ContainerEnv, EnvTaskTestsDir, ""); strings.TrimSpace(root) != "" {
		return root
	}
	if strings.TrimSpace(req.TaskDir) != "" {
		return containerTestsDir
	}
	return req.TestsDir
}

func promptTestScript(req TerminalTaskRequest) string {
	if strings.TrimSpace(req.TaskDir) != "" {
		return containerTaskRoot + "/test.sh"
	}
	return req.TestScriptPath
}

func (a *NativeAgent) maxTurns(req TerminalTaskRequest) int {
	if req.MaxTurns > 0 {
		return req.MaxTurns
	}
	return a.config.MaxTurns
}

func writeTrace(taskDir string, trace ToolCallTrace) (string, error) {
	tracePath := filepath.Join(taskDir, traceFileName)
	traceBytes, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal trace: %w", err)
	}
	if err := os.WriteFile(tracePath, traceBytes, 0o644); err != nil {
		return "", fmt.Errorf("write trace file: %w", err)
	}
	return tracePath, nil
}

func writeDebugSnapshot(taskDir string, snapshot taskDebugSnapshot) string {
	debugPath := filepath.Join(taskDir, debugFileName)
	debugBytes, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return ""
	}
	if err := os.WriteFile(debugPath, debugBytes, 0o644); err != nil {
		return ""
	}
	return debugPath
}

func nativeTraceToToolCallTrace(trace *sharednative.Trace, instruction string) ToolCallTrace {
	if trace == nil {
		return ToolCallTrace{Instruction: instruction}
	}

	steps := make([]ToolCallTraceStep, 0, len(trace.Steps))
	for _, step := range trace.Steps {
		steps = append(steps, ToolCallTraceStep{
			Index:         step.Index,
			AssistantText: step.AssistantText,
			ToolName:      step.ToolName,
			Arguments:     maps.Clone(step.Arguments),
			Observation:   step.Observation,
			IsError:       step.IsError,
			Usage:         tokenUsageFromShared(step.Usage),
			Metadata:      maps.Clone(step.Metadata),
		})
	}

	return ToolCallTrace{
		TaskID:      trace.TaskID,
		Model:       trace.Model,
		Provider:    trace.Provider,
		Instruction: instruction,
		StartedAt:   trace.StartedAt,
		Duration:    trace.Duration,
		Completed:   trace.Completed,
		FinalAnswer: trace.FinalAnswer,
		Error:       trace.Error,
		Steps:       steps,
	}
}

func tokenUsageFromShared(usage sharednative.TokenUsage) TokenUsage {
	return TokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	}
}

func countExecutedTools(steps []ToolCallTraceStep) int {
	count := 0
	for _, step := range steps {
		if step.ToolName != "" && !strings.EqualFold(step.ToolName, "finish") {
			count++
		}
	}
	return count
}

func supportsFunctionCalling(llm core.LLM) bool {
	for _, capability := range llm.Capabilities() {
		if capability == core.CapabilityToolCalling {
			return true
		}
	}
	return false
}

func containerEnvValue(env []string, key string, fallback string) string {
	prefix := key + "="
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) {
			value := strings.TrimSpace(strings.TrimPrefix(entry, prefix))
			if value != "" {
				return value
			}
		}
	}
	return fallback
}

func boolValue(value interface{}) bool {
	if typed, ok := value.(bool); ok {
		return typed
	}
	return false
}

const defaultToolCallingSystemPrompt = `You are a terminal task benchmark agent.

Work only inside the provided benchmark workspace.
Use the available tools to inspect files, edit files, and run commands.
Prefer concrete progress over narration.`

const defaultToolCallingToolPolicy = `Use narrow, evidence-seeking tool calls.
Start by locating the relevant files before making edits.
Prefer short targeted reads and focused shell commands over broad dumps.
After meaningful edits, run the smallest relevant verification before continuing.
Do not loop indefinitely: if you have enough evidence and the task is complete, call Finish.
When a health check, evaluation script, or verifier-style command succeeds and required outputs are present, stop and call Finish immediately.`
