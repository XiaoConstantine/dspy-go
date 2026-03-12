package tblite

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	toolspkg "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

const traceFileName = ".dspy_tblite_trace.json"

// ToolCallingAgentConfig controls the native tool-calling benchmark agent.
type ToolCallingAgentConfig struct {
	MaxTurns       int
	MaxTokens      int
	Temperature    float64
	Toolset        ToolsetConfig
	SystemPrompt   string
	ToolPolicy     string
	FinishToolText string
}

// ToolCallingAgent executes terminal benchmark tasks with native provider tool calling.
type ToolCallingAgent struct {
	llm       core.LLM
	config    ToolCallingAgentConfig
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

// NewToolCallingAgent creates a benchmark agent that uses provider-native tool calling.
func NewToolCallingAgent(llm core.LLM, cfg ToolCallingAgentConfig) (*ToolCallingAgent, error) {
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
	if cfg.Temperature == 0 {
		cfg.Temperature = 0
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
	return &ToolCallingAgent{
		llm:    llm,
		config: cfg,
		memory: agents.NewInMemoryStore(),
	}, nil
}

// RunTask executes one materialized terminal benchmark task.
func (a *ToolCallingAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	runCtx := ctx
	var cancel context.CancelFunc
	if req.AgentTimeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, req.AgentTimeout)
		defer cancel()
	}

	toolCfg := a.config.Toolset
	if req.ContainerID != "" && toolCfg.CommandRunner == nil {
		toolCfg.CommandRunner = dockerCommandRunner{
			runtime: &dockerTaskRuntime{
				containerID: req.ContainerID,
				taskRoot:    req.TaskDir,
			},
			extraEnv:  req.ContainerEnv,
			shellPath: toolCfg.ShellPath,
		}
	}

	toolRegistry := toolspkg.NewInMemoryToolRegistry()
	toolset, err := NewTerminalToolset(req.EnvironmentDir, toolCfg)
	if err != nil {
		return nil, err
	}
	for _, tool := range toolset {
		if err := toolRegistry.Register(tool); err != nil {
			return nil, fmt.Errorf("register tool %q: %w", tool.Name(), err)
		}
	}

	functions, err := toolspkg.BuildFunctionSchemas(toolRegistry)
	if err != nil {
		return nil, fmt.Errorf("build tool schemas: %w", err)
	}
	functions = append(functions, toolspkg.BuildFinishFunctionSchema(a.config.FinishToolText))

	startedAt := time.Now()
	trace := ToolCallTrace{
		TaskID:      req.TaskID,
		Model:       a.llm.ModelID(),
		Provider:    a.llm.ProviderName(),
		Instruction: req.Instruction,
		StartedAt:   startedAt,
		Steps:       make([]ToolCallTraceStep, 0, a.maxTurns(req)),
	}
	totalUsage := TokenUsage{}
	transcript := make([]toolTurn, 0, a.maxTurns(req))

	for turn := 0; turn < a.maxTurns(req); turn++ {
		prompt := a.buildPrompt(req, transcript)
		options := []core.GenerateOption{core.WithMaxTokens(a.config.MaxTokens)}
		if a.config.Temperature >= 0 {
			options = append(options, core.WithTemperature(a.config.Temperature))
		}
		result, err := a.llm.GenerateWithFunctions(runCtx, prompt, functions, options...)
		if err != nil {
			trace.Duration = time.Since(startedAt)
			trace.Error = err.Error()
			a.storeExecutionTrace(req, trace, totalUsage)
			traceFile, writeErr := writeTrace(req.TaskDir, trace)
			if writeErr != nil {
				return nil, fmt.Errorf("write failed trace: %w", writeErr)
			}
			return &TerminalTaskResult{
				Completed:  false,
				Error:      err.Error(),
				Duration:   trace.Duration,
				TokenUsage: totalUsage,
				TracePath:  traceFile,
				Metadata: map[string]any{
					"turns": len(trace.Steps),
				},
			}, nil
		}

		step := ToolCallTraceStep{Index: turn + 1}
		addUsage(&totalUsage, result["_usage"])
		if usage, ok := result["_usage"].(*core.TokenInfo); ok {
			step.Usage = tokenUsageFromCore(usage)
		}

		if content, ok := result["content"].(string); ok && strings.TrimSpace(content) != "" {
			step.AssistantText = strings.TrimSpace(content)
		}

		call, hasCall := result["function_call"].(map[string]any)
		if !hasCall {
			trace.Steps = append(trace.Steps, step)
			transcript = append(transcript, toolTurn{
				AssistantText: step.AssistantText,
				Observation:   "Model returned text without a tool call. It must use a tool or Finish.",
				IsError:       true,
			})
			continue
		}

		toolName, _ := call["name"].(string)
		arguments, _ := call["arguments"].(map[string]any)
		if arguments == nil {
			arguments = map[string]any{}
		}
		step.ToolName = toolName
		step.Arguments = core.ShallowCopyMap(arguments)

		if strings.EqualFold(toolName, "finish") {
			finalAnswer := extractFinishAnswer(arguments, step.AssistantText)
			trace.Completed = true
			trace.FinalAnswer = finalAnswer
			trace.Steps = append(trace.Steps, step)
			trace.Duration = time.Since(startedAt)
			a.storeExecutionTrace(req, trace, totalUsage)
			traceFile, err := writeTrace(req.TaskDir, trace)
			if err != nil {
				return nil, err
			}
			return &TerminalTaskResult{
				Completed:   true,
				FinalAnswer: finalAnswer,
				Duration:    trace.Duration,
				ToolCalls:   countExecutedTools(trace.Steps),
				TokenUsage:  totalUsage,
				TracePath:   traceFile,
				Metadata: map[string]any{
					"provider": a.llm.ProviderName(),
					"model":    a.llm.ModelID(),
					"turns":    len(trace.Steps),
				},
			}, nil
		}

		tool, err := toolRegistry.Get(toolName)
		if err != nil {
			step.IsError = true
			step.Observation = fmt.Sprintf("unknown tool %q: %v", toolName, err)
			trace.Steps = append(trace.Steps, step)
			transcript = append(transcript, toolTurn{
				ToolName:      toolName,
				Arguments:     step.Arguments,
				Observation:   step.Observation,
				IsError:       true,
				AssistantText: step.AssistantText,
			})
			continue
		}

		if err := tool.Validate(arguments); err != nil {
			step.IsError = true
			step.Observation = fmt.Sprintf("invalid tool arguments: %v", err)
			trace.Steps = append(trace.Steps, step)
			transcript = append(transcript, toolTurn{
				ToolName:      toolName,
				Arguments:     step.Arguments,
				Observation:   step.Observation,
				IsError:       true,
				AssistantText: step.AssistantText,
			})
			continue
		}

		toolResult, err := tool.Execute(runCtx, arguments)
		if err != nil {
			step.IsError = true
			step.Observation = fmt.Sprintf("tool execution failed: %v", err)
		} else {
			step.Observation = stringifyToolResult(toolResult)
			if isError, _ := toolResult.Metadata["isError"].(bool); isError {
				step.IsError = true
			}
		}

		trace.Steps = append(trace.Steps, step)
		transcript = append(transcript, toolTurn{
			ToolName:      toolName,
			Arguments:     step.Arguments,
			Observation:   step.Observation,
			IsError:       step.IsError,
			AssistantText: step.AssistantText,
		})
	}

	trace.Duration = time.Since(startedAt)
	trace.Error = fmt.Sprintf("max turns reached without Finish after %d turns", a.maxTurns(req))
	a.storeExecutionTrace(req, trace, totalUsage)
	traceFile, err := writeTrace(req.TaskDir, trace)
	if err != nil {
		return nil, err
	}
	return &TerminalTaskResult{
		Completed:  false,
		Error:      trace.Error,
		Duration:   trace.Duration,
		ToolCalls:  countExecutedTools(trace.Steps),
		TokenUsage: totalUsage,
		TracePath:  traceFile,
		Metadata: map[string]any{
			"provider": a.llm.ProviderName(),
			"model":    a.llm.ModelID(),
			"turns":    len(trace.Steps),
		},
	}, nil
}

const defaultToolCallingSystemPrompt = `You are a terminal task benchmark agent.

Work only inside the provided benchmark workspace.
Use the available tools to inspect files, edit files, and run commands.
Prefer concrete progress over narration.`

const defaultToolCallingToolPolicy = `Use narrow, evidence-seeking tool calls.
Start by locating the relevant files before making edits.
Prefer short targeted reads and focused shell commands over broad dumps.
After meaningful edits, run the smallest relevant verification before continuing.
Do not loop indefinitely: if you have enough evidence and the task is complete, call Finish.`

type toolTurn struct {
	AssistantText string
	ToolName      string
	Arguments     map[string]any
	Observation   string
	IsError       bool
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

func (a *ToolCallingAgent) buildPrompt(req TerminalTaskRequest, turns []toolTurn) string {
	var b strings.Builder
	b.WriteString(a.config.SystemPrompt)
	b.WriteString("\n\n")
	b.WriteString("TASK INSTRUCTION:\n")
	b.WriteString(req.Instruction)
	b.WriteString("\n\n")
	if strings.TrimSpace(a.config.ToolPolicy) != "" {
		b.WriteString("TOOL POLICY:\n")
		b.WriteString(a.config.ToolPolicy)
		b.WriteString("\n\n")
	}
	b.WriteString("WORKSPACE:\n")
	b.WriteString("- task id: " + req.TaskID + "\n")
	b.WriteString("- working directory: " + req.WorkingDirectory + "\n")
	b.WriteString("- tests directory: " + req.TestsDir + "\n")
	b.WriteString("- test script: " + req.TestScriptPath + "\n\n")
	b.WriteString("You must use tools to work in the workspace and call Finish when done.\n")

	if len(turns) == 0 {
		b.WriteString("\nNo prior steps yet.\n")
		return b.String()
	}

	b.WriteString("\nPRIOR STEPS:\n")
	for i, turn := range turns {
		fmt.Fprintf(&b, "Step %d:\n", i+1)
		if turn.AssistantText != "" {
			fmt.Fprintf(&b, "Assistant: %s\n", turn.AssistantText)
		}
		if turn.ToolName != "" {
			fmt.Fprintf(&b, "Tool: %s\n", turn.ToolName)
			if len(turn.Arguments) > 0 {
				argBytes, _ := json.Marshal(turn.Arguments)
				fmt.Fprintf(&b, "Arguments: %s\n", string(argBytes))
			}
		}
		if turn.Observation != "" {
			if turn.IsError {
				fmt.Fprintf(&b, "Observation (error): %s\n", turn.Observation)
			} else {
				fmt.Fprintf(&b, "Observation: %s\n", turn.Observation)
			}
		}
		b.WriteString("\n")
	}

	return b.String()
}

func (a *ToolCallingAgent) maxTurns(req TerminalTaskRequest) int {
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

func extractFinishAnswer(arguments map[string]any, fallback string) string {
	if answer, ok := arguments["answer"].(string); ok && strings.TrimSpace(answer) != "" {
		return strings.TrimSpace(answer)
	}
	return strings.TrimSpace(fallback)
}

func stringifyToolResult(result core.ToolResult) string {
	text := strings.TrimSpace(fmt.Sprint(result.Data))
	if text == "" {
		text = "(no output)"
	}
	if isError, _ := result.Metadata["isError"].(bool); isError {
		return "tool reported error: " + text
	}
	return text
}

func supportsFunctionCalling(llm core.LLM) bool {
	for _, capability := range llm.Capabilities() {
		if capability == core.CapabilityToolCalling {
			return true
		}
	}
	return false
}

func addUsage(total *TokenUsage, raw any) {
	usage, ok := raw.(*core.TokenInfo)
	if !ok || usage == nil {
		return
	}
	total.PromptTokens += int64(usage.PromptTokens)
	total.CompletionTokens += int64(usage.CompletionTokens)
	total.TotalTokens += int64(usage.TotalTokens)
}

func tokenUsageFromCore(usage *core.TokenInfo) TokenUsage {
	if usage == nil {
		return TokenUsage{}
	}
	return TokenUsage{
		PromptTokens:     int64(usage.PromptTokens),
		CompletionTokens: int64(usage.CompletionTokens),
		TotalTokens:      int64(usage.TotalTokens),
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

func (a *ToolCallingAgent) storeExecutionTrace(req TerminalTaskRequest, trace ToolCallTrace, usage TokenUsage) {
	if a == nil {
		return
	}

	toolUsage := make(map[string]int)
	steps := make([]agents.TraceStep, 0, len(trace.Steps))
	for _, step := range trace.Steps {
		if strings.TrimSpace(step.ToolName) != "" && !strings.EqualFold(step.ToolName, "finish") {
			toolUsage[step.ToolName]++
		}
		steps = append(steps, agents.TraceStep{
			Index:       step.Index,
			Thought:     step.AssistantText,
			Tool:        step.ToolName,
			Arguments:   core.ShallowCopyMap(step.Arguments),
			Observation: step.Observation,
			Success:     !step.IsError,
			Error:       boolError(step.IsError, step.Observation),
		})
	}

	status := agents.TraceStatusFailure
	if trace.Completed {
		status = agents.TraceStatusSuccess
	}
	if !trace.Completed && len(steps) > 0 {
		status = agents.TraceStatusPartial
	}

	execTrace := &agents.ExecutionTrace{
		AgentID:        fmt.Sprintf("tblite-%s-%s", a.llm.ProviderName(), a.llm.ModelID()),
		AgentType:      "tblite-tool-calling",
		Task:           req.TaskID,
		Input:          map[string]interface{}{"instruction": req.Instruction, "working_directory": req.WorkingDirectory, "tests_dir": req.TestsDir},
		Output:         map[string]interface{}{"completed": trace.Completed, "final_answer": trace.FinalAnswer},
		Steps:          steps,
		Status:         status,
		Error:          trace.Error,
		StartedAt:      trace.StartedAt,
		CompletedAt:    trace.StartedAt.Add(trace.Duration),
		ProcessingTime: trace.Duration,
		TokenUsage: map[string]int64{
			"prompt_tokens":     usage.PromptTokens,
			"completion_tokens": usage.CompletionTokens,
			"total_tokens":      usage.TotalTokens,
		},
		ToolUsageCount:   toolUsage,
		ContextMetadata:  map[string]interface{}{"turns": len(trace.Steps)},
		TerminationCause: traceTerminationCause(trace),
	}

	a.traceMu.Lock()
	a.lastTrace = execTrace
	a.traceMu.Unlock()
}

func traceTerminationCause(trace ToolCallTrace) string {
	if trace.Completed {
		return "finish"
	}
	if strings.Contains(strings.ToLower(trace.Error), "max turns") {
		return "max_turns"
	}
	if strings.TrimSpace(trace.Error) != "" {
		return "error"
	}
	return "unknown"
}

func boolError(isError bool, observation string) string {
	if !isError {
		return ""
	}
	return observation
}
