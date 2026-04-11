package native

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/skills"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/subagent"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	toolspkg "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

const defaultMaxConsecutiveNoCallResponses = 3

// Config controls the shared native tool-calling agent behavior.
type Config struct {
	MaxTurns                      int
	MaxTokens                     int
	Temperature                   float64
	SystemPrompt                  string
	ToolPolicy                    string
	FinishToolText                string
	Memory                        agents.Memory
	SessionID                     string
	SessionBranchID               string
	SessionRecallLimit            int
	SessionRecallMaxChars         int
	SessionEventStore             sessionevent.SessionEventStore
	SkillDomain                   string
	SkillStore                    skills.Store
	MaxConsecutiveNoCallResponses int
	ToolInterceptors              []core.ToolInterceptor
	OnEvent                       func(agents.AgentEvent)
}

// TokenUsage captures aggregate token usage for one tool-calling run.
type TokenUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// Trace captures a native tool-calling run with step-level detail.
type Trace struct {
	TaskID      string        `json:"task_id,omitempty"`
	Model       string        `json:"model"`
	Provider    string        `json:"provider"`
	Task        string        `json:"task"`
	StartedAt   time.Time     `json:"started_at"`
	Duration    time.Duration `json:"duration"`
	Completed   bool          `json:"completed"`
	FinalAnswer string        `json:"final_answer,omitempty"`
	Error       string        `json:"error,omitempty"`
	TokenUsage  TokenUsage    `json:"token_usage"`
	Steps       []TraceStep   `json:"steps"`
}

// Clone returns a deep copy of the trace.
func (t *Trace) Clone() *Trace {
	if t == nil {
		return nil
	}

	cloned := &Trace{
		TaskID:      t.TaskID,
		Model:       t.Model,
		Provider:    t.Provider,
		Task:        t.Task,
		StartedAt:   t.StartedAt,
		Duration:    t.Duration,
		Completed:   t.Completed,
		FinalAnswer: t.FinalAnswer,
		Error:       t.Error,
		TokenUsage:  t.TokenUsage,
	}
	if len(t.Steps) > 0 {
		cloned.Steps = make([]TraceStep, len(t.Steps))
		for i, step := range t.Steps {
			cloned.Steps[i] = step.Clone()
		}
	}
	return cloned
}

// TraceStep captures one model/tool turn.
type TraceStep struct {
	Index              int               `json:"index"`
	AssistantText      string            `json:"assistant_text,omitempty"`
	ToolName           string            `json:"tool_name,omitempty"`
	Arguments          map[string]any    `json:"arguments,omitempty"`
	Observation        string            `json:"observation,omitempty"`
	ObservationDisplay string            `json:"observation_display,omitempty"`
	ObservationDetails map[string]any    `json:"observation_details,omitempty"`
	IsError            bool              `json:"is_error,omitempty"`
	Synthetic          bool              `json:"synthetic,omitempty"`
	Redacted           bool              `json:"redacted,omitempty"`
	Truncated          bool              `json:"truncated,omitempty"`
	Usage              TokenUsage        `json:"usage,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
}

// Clone returns a deep copy of the step.
func (s TraceStep) Clone() TraceStep {
	return TraceStep{
		Index:              s.Index,
		AssistantText:      s.AssistantText,
		ToolName:           s.ToolName,
		Arguments:          maps.Clone(s.Arguments),
		Observation:        s.Observation,
		ObservationDisplay: s.ObservationDisplay,
		ObservationDetails: maps.Clone(s.ObservationDetails),
		IsError:            s.IsError,
		Synthetic:          s.Synthetic,
		Redacted:           s.Redacted,
		Truncated:          s.Truncated,
		Usage:              s.Usage,
		Metadata:           maps.Clone(s.Metadata),
	}
}

type toolTurn struct {
	LLMTurnIndex     int
	AssistantText    string
	AssistantContent []core.ContentBlock
	ToolCallID       string
	ToolName         string
	Arguments        map[string]any
	ToolCallMetadata map[string]any
	Observation      string
	IsError          bool
}

// Agent executes tasks using provider-native tool calling with a shared dspy-go harness.
type Agent struct {
	llm          core.LLM
	config       Config
	toolRegistry *toolspkg.InMemoryToolRegistry
	memory       agents.Memory
	sessions     *agents.SessionStore
	sessionEvent sessionevent.SessionEventStore

	loadedSkill          *skills.Skill
	skillLoadErr         error
	persistedSkillPrompt string

	traceMu         sync.RWMutex
	lastTrace       *agents.ExecutionTrace
	lastNativeTrace *Trace
}

var _ optimize.OptimizableAgent = (*Agent)(nil)

// NewAgent creates a shared native tool-calling agent.
func NewAgent(llm core.LLM, cfg Config) (*Agent, error) {
	if llm == nil {
		return nil, fmt.Errorf("llm is required")
	}
	if !supportsToolCalling(llm) {
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
		cfg.SystemPrompt = defaultSystemPrompt
	}
	if strings.TrimSpace(cfg.ToolPolicy) == "" {
		cfg.ToolPolicy = defaultToolPolicy
	}
	if cfg.MaxConsecutiveNoCallResponses <= 0 {
		cfg.MaxConsecutiveNoCallResponses = defaultMaxConsecutiveNoCallResponses
	}
	if cfg.SessionRecallLimit <= 0 {
		cfg.SessionRecallLimit = defaultSessionRecallLimit
	}
	if cfg.SessionRecallMaxChars <= 0 {
		cfg.SessionRecallMaxChars = defaultSessionRecallMaxChars
	}
	if cfg.Memory == nil {
		cfg.Memory = agents.NewInMemoryStore()
	}
	cfg.ToolInterceptors = append([]core.ToolInterceptor(nil), cfg.ToolInterceptors...)

	agent := &Agent{
		llm:          llm,
		config:       cfg,
		toolRegistry: toolspkg.NewInMemoryToolRegistry(),
		memory:       cfg.Memory,
		sessions:     agents.NewSessionStore(cfg.Memory),
		sessionEvent: cfg.SessionEventStore,
	}

	if cfg.SkillStore != nil && strings.TrimSpace(cfg.SkillDomain) != "" {
		skill, err := cfg.SkillStore.Best(context.Background(), cfg.SkillDomain)
		agent.skillLoadErr = err
		if err != nil {
			logger := logging.GetLogger()
			logger.Warn(context.Background(), "Failed to load skill for native agent domain %s: %v", cfg.SkillDomain, err)
		} else if skill != nil {
			cloned := skill.Clone()
			agent.loadedSkill = &cloned
			agent.persistedSkillPrompt = strings.TrimSpace(cloned.Content)
		}
	}

	return agent, nil
}

// RegisterTool adds a tool to the shared registry.
func (a *Agent) RegisterTool(tool core.Tool) error {
	if a == nil {
		return fmt.Errorf("tool-calling agent is nil")
	}
	return a.toolRegistry.Register(tool)
}

// Execute runs one native tool-calling task.
func (a *Agent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	if a == nil {
		return nil, fmt.Errorf("tool-calling agent is nil")
	}

	task := agentutil.StringValue(input["task"])
	if strings.TrimSpace(task) == "" {
		return nil, fmt.Errorf("task is required")
	}
	taskID := agentutil.StringValue(input["task_id"])
	maxTurns := a.maxTurns(input)
	sessionID := a.sessionID(input)
	sessionContext, sessionErr := a.loadSessionContext(ctx, input)
	if sessionID != "" {
		sessionEventData := map[string]any{
			"task_id":        taskID,
			"session_id":     sessionID,
			"source":         sessionContext.Source,
			"record_count":   sessionContext.RecordCount,
			"entry_count":    sessionContext.EntryCount,
			"summary_count":  sessionContext.SummaryCount,
			"recall_chars":   len(sessionContext.Recall),
			"branch_id":      sessionContext.BranchID,
			"head_entry_id":  sessionContext.HeadEntryID,
			"forked_from_id": sessionContext.ForkedFromEntryID,
		}
		if sessionErr != nil {
			sessionEventData["error"] = sessionErr.Error()
		}
		a.emitEvent(agents.EventSessionLoaded, sessionEventData)
	}
	sessionRecall := sessionContext.Recall
	if sessionErr != nil {
		sessionRecall = ""
	}

	functions, err := toolspkg.BuildFunctionSchemas(a.toolRegistry)
	if err != nil {
		return nil, fmt.Errorf("build tool schemas: %w", err)
	}
	functions = append(functions, toolspkg.BuildFinishFunctionSchema(a.config.FinishToolText))

	startedAt := time.Now()
	trace := &Trace{
		TaskID:    taskID,
		Model:     a.llm.ModelID(),
		Provider:  a.llm.ProviderName(),
		Task:      task,
		StartedAt: startedAt,
		Steps:     make([]TraceStep, 0, maxTurns),
	}

	totalUsage := TokenUsage{}
	transcript := make([]toolTurn, 0, maxTurns)
	noCallStreak := 0

	a.emitEvent(agents.EventRunStarted, map[string]any{
		"task_id":               taskID,
		"task":                  task,
		"max_turns":             maxTurns,
		"model":                 a.llm.ModelID(),
		"provider":              a.llm.ProviderName(),
		"session_id":            sessionID,
		"session_source":        sessionContext.Source,
		"session_runs":          sessionContext.RecordCount,
		"session_entries":       sessionContext.EntryCount,
		"session_summaries":     sessionContext.SummaryCount,
		"session_branch_id":     sessionContext.BranchID,
		"session_head_entry_id": sessionContext.HeadEntryID,
		"session_recall_chars":  len(sessionRecall),
	})

	for turn := 0; turn < maxTurns; turn++ {
		options := []core.GenerateOption{core.WithMaxTokens(a.config.MaxTokens)}
		options = append(options, core.WithTemperature(a.config.Temperature))

		a.emitEvent(agents.EventLLMTurnStarted, map[string]any{
			"task_id":   taskID,
			"turn":      turn + 1,
			"max_turns": maxTurns,
		})

		result, err := a.generateToolCallResponse(ctx, task, sessionRecall, transcript, turn+1, maxTurns, functions, options...)
		if err != nil {
			trace.Duration = time.Since(startedAt)
			trace.Error = err.Error()
			trace.TokenUsage = totalUsage
			a.storeTraces(ctx, input, trace)
			a.emitEvent(agents.EventLLMTurnFinished, map[string]any{
				"task_id":     taskID,
				"turn":        turn + 1,
				"tool_calls":  0,
				"usage_total": int64(0),
				"error":       err.Error(),
			})
			a.emitEvent(agents.EventRunFailed, map[string]any{
				"task_id": taskID,
				"error":   err.Error(),
			})
			a.emitEvent(agents.EventRunFinished, map[string]any{
				"task_id":    taskID,
				"completed":  false,
				"turns":      len(trace.Steps),
				"tool_calls": countExecutedTools(trace.Steps),
				"error":      trace.Error,
			})
			return nil, err
		}

		step := TraceStep{Index: turn + 1}
		addUsage(&totalUsage, result["_usage"])
		if usage, ok := result["_usage"].(*core.TokenInfo); ok {
			step.Usage = tokenUsageFromCore(usage)
		}

		if content, ok := result["content"].(string); ok && strings.TrimSpace(content) != "" {
			step.AssistantText = strings.TrimSpace(content)
		}
		assistantContent, _ := result["content_blocks"].([]core.ContentBlock)
		if metadata := toolResponseDiagnosticMetadata(result); len(metadata) > 0 {
			step.Metadata = metadata
		}

		calls, err := extractFunctionCalls(result)
		if err != nil {
			a.emitEvent(agents.EventLLMTurnFinished, map[string]any{
				"task_id":        taskID,
				"turn":           turn + 1,
				"assistant_text": step.AssistantText,
				"tool_calls":     0,
				"usage_total":    step.Usage.TotalTokens,
				"parse_error":    err.Error(),
			})
			step.Observation = err.Error()
			step.ObservationDisplay = err.Error()
			step.IsError = true
			trace.Steps = append(trace.Steps, step)
			trace.Duration = time.Since(startedAt)
			trace.Error = err.Error()
			trace.TokenUsage = totalUsage
			a.storeTraces(ctx, input, trace)
			a.emitEvent(agents.EventRunFinished, map[string]any{
				"task_id":    taskID,
				"completed":  false,
				"turns":      len(trace.Steps),
				"tool_calls": countExecutedTools(trace.Steps),
				"error":      trace.Error,
			})
			return map[string]interface{}{
				"completed":   false,
				"error":       trace.Error,
				"tool_calls":  countExecutedTools(trace.Steps),
				"turns":       len(trace.Steps),
				"token_usage": tokenUsageToMap(totalUsage),
			}, nil
		}
		a.emitEvent(agents.EventLLMTurnFinished, map[string]any{
			"task_id":        taskID,
			"turn":           turn + 1,
			"assistant_text": step.AssistantText,
			"tool_calls":     len(calls),
			"usage_total":    step.Usage.TotalTokens,
		})
		if len(calls) == 0 {
			noCallStreak++
			observation := "Model returned text without a tool call. It must use a tool or Finish."
			if reason := toolResponseDiagnosticReason(step.Metadata); reason != "" {
				observation += " Provider diagnostic: " + reason
			}
			if finishReason := toolResponseDiagnosticFinishReason(step.Metadata); finishReason != "" {
				observation += " Finish reason: " + finishReason
			}
			step.Observation = observation
			step.ObservationDisplay = observation
			step.IsError = true
			trace.Steps = append(trace.Steps, step)
			transcript = append(transcript, toolTurn{
				LLMTurnIndex:     turn + 1,
				AssistantText:    step.AssistantText,
				AssistantContent: cloneContentBlocks(assistantContent),
				Observation:      observation,
				IsError:          true,
			})
			if noCallStreak >= a.config.MaxConsecutiveNoCallResponses {
				trace.Duration = time.Since(startedAt)
				trace.Error = fmt.Sprintf("repeated model responses without tool calls after %d turns", noCallStreak)
				trace.TokenUsage = totalUsage
				a.storeTraces(ctx, input, trace)
				a.emitEvent(agents.EventRunFinished, map[string]any{
					"task_id":    taskID,
					"completed":  false,
					"turns":      len(trace.Steps),
					"tool_calls": countExecutedTools(trace.Steps),
					"error":      trace.Error,
				})
				return map[string]interface{}{
					"completed":   false,
					"error":       trace.Error,
					"tool_calls":  countExecutedTools(trace.Steps),
					"turns":       len(trace.Steps),
					"token_usage": tokenUsageToMap(totalUsage),
				}, nil
			}
			continue
		}
		noCallStreak = 0

		for callIndex, call := range calls {
			callStep := step
			if callIndex > 0 {
				callStep.Index = len(trace.Steps) + 1
				callStep.AssistantText = ""
			}

			toolName := call.Name
			arguments := maps.Clone(call.Arguments)
			if arguments == nil {
				arguments = map[string]any{}
			}
			callStep.ToolName = toolName
			// Trace the originally proposed call arguments. Interceptors may rewrite
			// the executed arguments, but proposed values remain useful for debugging
			// model behavior and policy decisions.
			callStep.Arguments = maps.Clone(arguments)
			proposedEvent := map[string]any{
				"task_id":   taskID,
				"turn":      turn + 1,
				"tool_name": toolName,
				"arguments": maps.Clone(arguments),
			}

			if strings.EqualFold(toolName, "finish") {
				a.emitEvent(agents.EventToolCallProposed, enrichSubagentEventData(nil, proposedEvent, nil))
				finalAnswer := extractFinishAnswer(arguments, callStep.AssistantText)
				trace.Completed = true
				trace.FinalAnswer = finalAnswer
				trace.Steps = append(trace.Steps, callStep)
				trace.Duration = time.Since(startedAt)
				trace.TokenUsage = totalUsage
				a.storeTraces(ctx, input, trace)
				a.emitEvent(agents.EventRunFinished, map[string]any{
					"task_id":      taskID,
					"completed":    true,
					"turns":        len(trace.Steps),
					"tool_calls":   countExecutedTools(trace.Steps),
					"final_answer": finalAnswer,
				})
				return map[string]interface{}{
					"completed":    true,
					"final_answer": finalAnswer,
					"tool_calls":   countExecutedTools(trace.Steps),
					"turns":        len(trace.Steps),
					"token_usage":  tokenUsageToMap(totalUsage),
				}, nil
			}

			tool, err := a.toolRegistry.Get(toolName)
			if err != nil {
				a.emitEvent(agents.EventToolCallProposed, enrichSubagentEventData(nil, proposedEvent, nil))
				callStep.IsError = true
				callStep.Observation = fmt.Sprintf("unknown tool %q: %v", toolName, err)
				callStep.ObservationDisplay = callStep.Observation
				trace.Steps = append(trace.Steps, callStep)
				transcript = append(transcript, toolTurn{
					LLMTurnIndex:     turn + 1,
					AssistantText:    callStep.AssistantText,
					AssistantContent: cloneContentBlocks(assistantContent),
					ToolCallID:       call.ID,
					ToolName:         toolName,
					Arguments:        callStep.Arguments,
					ToolCallMetadata: maps.Clone(call.Metadata),
					Observation:      callStep.Observation,
					IsError:          true,
				})
				continue
			}

			if err := tool.Validate(arguments); err != nil {
				a.emitEvent(agents.EventToolCallProposed, enrichSubagentEventData(tool, proposedEvent, nil))
				callStep.IsError = true
				callStep.Observation = fmt.Sprintf("invalid tool arguments: %v", err)
				callStep.ObservationDisplay = callStep.Observation
				trace.Steps = append(trace.Steps, callStep)
				transcript = append(transcript, toolTurn{
					LLMTurnIndex:     turn + 1,
					AssistantText:    callStep.AssistantText,
					AssistantContent: cloneContentBlocks(assistantContent),
					ToolCallID:       call.ID,
					ToolName:         toolName,
					Arguments:        callStep.Arguments,
					ToolCallMetadata: maps.Clone(call.Metadata),
					Observation:      callStep.Observation,
					IsError:          true,
				})
				continue
			}

			a.emitEvent(agents.EventToolCallProposed, enrichSubagentEventData(tool, proposedEvent, nil))
			a.emitEvent(agents.EventToolCallStarted, enrichSubagentEventData(tool, map[string]any{
				"task_id":   taskID,
				"turn":      turn + 1,
				"tool_name": toolName,
				"arguments": maps.Clone(arguments),
			}, nil))

			observation, err := a.executeTool(ctx, tool, arguments)
			if errors.Is(err, core.ErrToolBlocked) {
				reason := blockedReason(err)
				observation = agents.BlockedToolObservation(toolName, reason)
				callStep.IsError = true
				a.emitEvent(agents.EventToolCallBlocked, enrichSubagentEventData(tool, map[string]any{
					"task_id":   taskID,
					"turn":      turn + 1,
					"tool_name": toolName,
					"arguments": maps.Clone(arguments),
					"reason":    reason,
				}, observation.Details))
			} else if err != nil {
				callStep.IsError = true
				observation = agents.ToolObservation{
					ModelText:   fmt.Sprintf("tool execution failed: %v", err),
					DisplayText: fmt.Sprintf("tool execution failed: %v", err),
					IsError:     true,
				}
			}

			callStep.Observation = observation.ModelText
			callStep.ObservationDisplay = observation.DisplayText
			callStep.ObservationDetails = observation.Details
			callStep.IsError = callStep.IsError || observation.IsError
			callStep.Synthetic = callStep.Synthetic || observation.Synthetic
			callStep.Redacted = observation.Redacted
			callStep.Truncated = observation.Truncated

			trace.Steps = append(trace.Steps, callStep)
			transcript = append(transcript, toolTurn{
				LLMTurnIndex:     turn + 1,
				AssistantText:    callStep.AssistantText,
				AssistantContent: cloneContentBlocks(assistantContent),
				ToolCallID:       call.ID,
				ToolName:         toolName,
				Arguments:        callStep.Arguments,
				ToolCallMetadata: maps.Clone(call.Metadata),
				Observation:      callStep.Observation,
				IsError:          callStep.IsError,
			})
			a.emitEvent(agents.EventToolCallFinished, enrichSubagentEventData(tool, map[string]any{
				"task_id":   taskID,
				"turn":      turn + 1,
				"tool_name": toolName,
				"is_error":  callStep.IsError,
				"synthetic": callStep.Synthetic,
				"redacted":  callStep.Redacted,
				"truncated": callStep.Truncated,
			}, observation.Details))
		}
	}

	trace.Duration = time.Since(startedAt)
	trace.Error = fmt.Sprintf("max turns reached without Finish after %d turns", maxTurns)
	trace.TokenUsage = totalUsage
	a.storeTraces(ctx, input, trace)
	a.emitEvent(agents.EventRunFinished, map[string]any{
		"task_id":    taskID,
		"completed":  false,
		"turns":      len(trace.Steps),
		"tool_calls": countExecutedTools(trace.Steps),
		"error":      trace.Error,
	})
	return map[string]interface{}{
		"completed":   false,
		"error":       trace.Error,
		"tool_calls":  countExecutedTools(trace.Steps),
		"turns":       len(trace.Steps),
		"token_usage": tokenUsageToMap(totalUsage),
	}, nil
}

// GetCapabilities returns the currently registered tools.
func (a *Agent) GetCapabilities() []core.Tool {
	if a == nil {
		return nil
	}
	return a.toolRegistry.List()
}

// GetMemory returns the attached memory store.
func (a *Agent) GetMemory() agents.Memory {
	if a == nil {
		return nil
	}
	return a.memory
}

// GetArtifacts returns prompt/config backed artifacts.
func (a *Agent) GetArtifacts() optimize.AgentArtifacts {
	artifacts := optimize.AgentArtifacts{
		Text: make(map[optimize.ArtifactKey]string),
		Int:  make(map[string]int),
		Bool: make(map[string]bool),
	}
	if a == nil {
		return artifacts
	}
	artifacts.Text[optimize.ArtifactSkillPack] = composeSystemPrompt(a.config.SystemPrompt, a.persistedSkillPrompt)
	artifacts.Text[optimize.ArtifactToolPolicy] = a.config.ToolPolicy
	artifacts.Int["max_turns"] = a.config.MaxTurns
	return artifacts
}

// GetLoadedSkill returns the constructor-loaded persisted skill, if one was applied.
func (a *Agent) GetLoadedSkill() *skills.Skill {
	if a == nil || a.loadedSkill == nil {
		return nil
	}

	cloned := a.loadedSkill.Clone()
	return &cloned
}

// GetSkillLoadError returns the last constructor-time persisted skill load error.
func (a *Agent) GetSkillLoadError() error {
	if a == nil {
		return nil
	}
	return a.skillLoadErr
}

// SetArtifacts updates the prompt/config backed artifacts.
func (a *Agent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	if a == nil {
		return fmt.Errorf("tool-calling agent is nil")
	}
	if prompt, ok := artifacts.Text[optimize.ArtifactSkillPack]; ok && strings.TrimSpace(prompt) != "" {
		a.config.SystemPrompt = prompt
		a.persistedSkillPrompt = ""
		a.loadedSkill = nil
		a.skillLoadErr = nil
		// Explicit prompt overrides are authoritative. Clearing constructor-time
		// skill wiring avoids reloading persisted skills on later Clone() calls.
		a.config.SkillStore = nil
		a.config.SkillDomain = ""
	}
	if policy, ok := artifacts.Text[optimize.ArtifactToolPolicy]; ok && policy != "" {
		a.config.ToolPolicy = policy
	}
	if maxTurns, ok := artifacts.Int["max_turns"]; ok && maxTurns > 0 {
		a.config.MaxTurns = maxTurns
	}
	return nil
}

// Clone creates a fresh agent with the same LLM, config, and registered tools.
func (a *Agent) Clone() (optimize.OptimizableAgent, error) {
	if a == nil {
		return nil, fmt.Errorf("tool-calling agent is nil")
	}

	cfg := a.config
	cfg.Memory = nil
	cfg.SessionID = ""
	cfg.SessionBranchID = ""
	cfg.SessionEventStore = nil
	cloned, err := NewAgent(a.llm, cfg)
	if err != nil {
		return nil, err
	}
	for _, tool := range a.toolRegistry.List() {
		toolClone, err := cloneRegisteredTool(tool)
		if err != nil {
			return nil, err
		}
		if err := cloned.RegisterTool(toolClone); err != nil {
			return nil, fmt.Errorf("register cloned tool %q: %w", tool.Name(), err)
		}
	}
	return cloned, nil
}

// LastExecutionTrace returns the most recent execution trace.
func (a *Agent) LastExecutionTrace() *agents.ExecutionTrace {
	if a == nil {
		return nil
	}
	a.traceMu.RLock()
	defer a.traceMu.RUnlock()
	return a.lastTrace.Clone()
}

// LastNativeTrace returns the richer native-tool trace from the most recent run.
func (a *Agent) LastNativeTrace() *Trace {
	if a == nil {
		return nil
	}
	a.traceMu.RLock()
	defer a.traceMu.RUnlock()
	return a.lastNativeTrace.Clone()
}

func (a *Agent) maxTurns(input map[string]interface{}) int {
	if override := agentutil.IntValue(input["max_turns"]); override > 0 {
		return override
	}
	return a.config.MaxTurns
}

func (a *Agent) emitEvent(eventType string, data map[string]any) {
	if a == nil {
		return
	}
	agents.EmitEvent(a.config.OnEvent, eventType, data)
}

func enrichSubagentEventData(tool core.Tool, data map[string]any, details map[string]any) map[string]any {
	info, ok := subagent.InfoFromTool(tool)
	if !ok {
		return data
	}
	enriched := maps.Clone(data)
	enriched["subagent"] = true
	enriched["subagent_name"] = info.Name
	enriched["session_policy"] = info.SessionPolicy
	if details != nil {
		if completed, ok := details["completed"].(bool); ok {
			enriched["child_completed"] = completed
		}
	}
	return enriched
}

func (a *Agent) executeTool(ctx context.Context, tool core.Tool, arguments map[string]any) (agents.ToolObservation, error) {
	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return tool.Execute(ctx, args)
	}
	info := core.NewToolInfoFromTool(tool)
	result, err := core.ChainToolInterceptors(a.config.ToolInterceptors...)(ctx, arguments, info, handler)
	if err != nil {
		return agents.ToolObservation{}, err
	}
	return agents.NormalizeToolResult(result), nil
}

func (a *Agent) generateToolCallResponse(ctx context.Context, task string, sessionRecall string, turns []toolTurn, currentTurn int, maxTurns int, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	core.RecordLLMCall(ctx, a.llm)
	if supportsNativeToolCalling(a.llm) {
		chatLLM, ok := a.llm.(core.ToolCallingChatLLM)
		if !ok {
			return nil, fmt.Errorf("llm %s advertised native tool calling without chat tool support", a.llm.ModelID())
		}
		return chatLLM.GenerateWithTools(ctx, a.buildToolMessages(task, sessionRecall, turns, currentTurn, maxTurns), functions, options...)
	}
	return a.llm.GenerateWithFunctions(ctx, a.buildPrompt(task, sessionRecall, turns, currentTurn, maxTurns), functions, options...)
}

func (a *Agent) buildToolMessages(task string, sessionRecall string, turns []toolTurn, currentTurn int, maxTurns int) []core.ChatMessage {
	messages := []core.ChatMessage{
		{
			Role:    "user",
			Content: []core.ContentBlock{core.NewTextBlock(a.buildPrompt(task, sessionRecall, nil, currentTurn, maxTurns))},
		},
	}
	for i := 0; i < len(turns); i++ {
		turn := turns[i]
		if strings.TrimSpace(turn.ToolName) != "" {
			group := []toolTurn{turn}
			for j := i + 1; j < len(turns); j++ {
				next := turns[j]
				if strings.TrimSpace(next.ToolName) == "" || next.LLMTurnIndex != turn.LLMTurnIndex {
					break
				}
				group = append(group, next)
				i = j
			}

			assistant := core.ChatMessage{Role: "assistant"}
			for _, grouped := range group {
				if len(assistant.Content) == 0 {
					if len(grouped.AssistantContent) > 0 {
						assistant.Content = cloneContentBlocks(grouped.AssistantContent)
					} else if strings.TrimSpace(grouped.AssistantText) != "" {
						assistant.Content = []core.ContentBlock{core.NewTextBlock(grouped.AssistantText)}
					}
				}
				assistant.ToolCalls = append(assistant.ToolCalls, core.ToolCall{
					ID:        grouped.ToolCallID,
					Name:      grouped.ToolName,
					Arguments: maps.Clone(grouped.Arguments),
					Metadata:  maps.Clone(grouped.ToolCallMetadata),
				})
			}
			if len(assistant.Content) > 0 || len(assistant.ToolCalls) > 0 {
				messages = append(messages, assistant)
			}
			for _, grouped := range group {
				messages = append(messages, core.ChatMessage{
					Role: "tool",
					ToolResult: &core.ChatToolResult{
						ToolCallID: grouped.ToolCallID,
						Name:       grouped.ToolName,
						Content:    []core.ContentBlock{core.NewTextBlock(grouped.Observation)},
						IsError:    grouped.IsError,
					},
				})
			}
			continue
		}

		assistant := core.ChatMessage{Role: "assistant"}
		if len(turn.AssistantContent) > 0 {
			assistant.Content = cloneContentBlocks(turn.AssistantContent)
		} else if strings.TrimSpace(turn.AssistantText) != "" {
			assistant.Content = []core.ContentBlock{core.NewTextBlock(turn.AssistantText)}
		}
		if len(assistant.Content) > 0 {
			messages = append(messages, assistant)
		}
		if strings.TrimSpace(turn.Observation) != "" {
			messages = append(messages, core.ChatMessage{
				Role:    "user",
				Content: []core.ContentBlock{core.NewTextBlock("System observation: " + turn.Observation)},
			})
		}
	}
	if reminder := turnBudgetReminder(currentTurn, maxTurns); reminder != "" {
		messages = append(messages, core.ChatMessage{
			Role:    "user",
			Content: []core.ContentBlock{core.NewTextBlock(reminder)},
		})
	}
	return messages
}

func (a *Agent) buildPrompt(task string, sessionRecall string, turns []toolTurn, currentTurn int, maxTurns int) string {
	var b strings.Builder
	b.WriteString(composeSystemPrompt(a.config.SystemPrompt, a.persistedSkillPrompt))
	b.WriteString("\n\n")
	if strings.TrimSpace(a.config.ToolPolicy) != "" {
		b.WriteString("TOOL POLICY:\n")
		b.WriteString(a.config.ToolPolicy)
		b.WriteString("\n\n")
	}
	b.WriteString("TASK:\n")
	b.WriteString(task)
	b.WriteString("\n\n")
	if strings.TrimSpace(sessionRecall) != "" {
		b.WriteString("SESSION RECALL:\n")
		b.WriteString(sessionRecall)
		b.WriteString("\n\n")
	}
	if currentTurn > 0 && maxTurns > 0 {
		fmt.Fprintf(&b, "TURN BUDGET: turn %d of %d (%d remaining)\n", currentTurn, maxTurns, maxTurns-currentTurn+1)
		b.WriteString("If the task appears complete or your latest verification/check succeeded, call Finish immediately instead of doing extra inspection.\n\n")
	}
	b.WriteString("You must use tools to work on the task and call Finish when done.\n")

	if len(turns) == 0 {
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

func composeSystemPrompt(base, persistedSkill string) string {
	base = strings.TrimSpace(base)
	persistedSkill = strings.TrimSpace(persistedSkill)
	if persistedSkill == "" {
		return base
	}
	if base == "" {
		return "SKILL PACK:\n" + persistedSkill
	}
	return base + "\n\nSKILL PACK:\n" + persistedSkill
}

func turnBudgetReminder(currentTurn int, maxTurns int) string {
	if currentTurn <= 0 || maxTurns <= 0 {
		return ""
	}
	remaining := maxTurns - currentTurn + 1
	switch {
	case remaining <= 1:
		return fmt.Sprintf("Final turn. If the task is complete or your latest verification succeeded, call Finish now.")
	case remaining <= 3:
		return fmt.Sprintf("%d turns remaining. If the task is complete or your latest verification succeeded, stop inspecting and call Finish.", remaining)
	default:
		return ""
	}
}

func (a *Agent) storeTraces(ctx context.Context, input map[string]interface{}, trace *Trace) {
	execTrace := nativeTraceToExecutionTrace(a, input, trace)
	a.traceMu.Lock()
	a.lastNativeTrace = trace.Clone()
	a.lastTrace = execTrace
	a.traceMu.Unlock()
	a.persistSessionRecord(ctx, input, trace)
}

func nativeTraceToExecutionTrace(a *Agent, input map[string]interface{}, trace *Trace) *agents.ExecutionTrace {
	if trace == nil {
		return nil
	}

	steps := make([]agents.TraceStep, 0, len(trace.Steps))
	toolUsage := make(map[string]int)
	for _, step := range trace.Steps {
		if step.ToolName != "" && !strings.EqualFold(step.ToolName, "finish") {
			toolUsage[step.ToolName]++
		}
		steps = append(steps, agents.TraceStep{
			Index:              step.Index,
			Thought:            step.AssistantText,
			Tool:               step.ToolName,
			Arguments:          maps.Clone(step.Arguments),
			Observation:        step.Observation,
			ObservationDisplay: step.ObservationDisplay,
			ObservationDetails: maps.Clone(step.ObservationDetails),
			Success:            !step.IsError,
			Error:              boolError(step.IsError, step.Observation),
			Synthetic:          step.Synthetic,
			Redacted:           step.Redacted,
			Truncated:          step.Truncated,
		})
	}

	status := agents.TraceStatusFailure
	switch {
	case trace.Completed:
		status = agents.TraceStatusSuccess
	case len(steps) > 0:
		status = agents.TraceStatusPartial
	}

	contextMetadata := map[string]interface{}{
		"turns": len(trace.Steps),
	}
	if sessionID := a.sessionID(input); sessionID != "" {
		contextMetadata["session_id"] = sessionID
	}

	return &agents.ExecutionTrace{
		AgentID:        fmt.Sprintf("native-%s-%s", a.llm.ProviderName(), a.llm.ModelID()),
		AgentType:      "native",
		Task:           firstNonEmpty(trace.TaskID, trace.Task),
		Input:          maps.Clone(input),
		Output:         map[string]interface{}{"completed": trace.Completed, "final_answer": trace.FinalAnswer},
		Steps:          steps,
		Status:         status,
		Error:          trace.Error,
		StartedAt:      trace.StartedAt,
		CompletedAt:    trace.StartedAt.Add(trace.Duration),
		ProcessingTime: trace.Duration,
		TokenUsage: map[string]int64{
			"prompt_tokens":     trace.TokenUsage.PromptTokens,
			"completion_tokens": trace.TokenUsage.CompletionTokens,
			"total_tokens":      trace.TokenUsage.TotalTokens,
		},
		ToolUsageCount:   toolUsage,
		ContextMetadata:  contextMetadata,
		TerminationCause: traceTerminationCause(trace),
	}
}

func traceTerminationCause(trace *Trace) string {
	if trace == nil {
		return "unknown"
	}
	if trace.Completed {
		return "finish"
	}
	switch {
	case strings.Contains(strings.ToLower(trace.Error), "max turns"):
		return "max_turns"
	case strings.Contains(strings.ToLower(trace.Error), "without tool calls"):
		return "no_tool_calls"
	case strings.TrimSpace(trace.Error) != "":
		return "error"
	default:
		return "unknown"
	}
}

func extractFunctionCalls(result map[string]any) ([]core.ToolCall, error) {
	if result == nil {
		return nil, nil
	}
	if rawCalls, ok := result["tool_calls"].([]core.ToolCall); ok && len(rawCalls) > 0 {
		calls := make([]core.ToolCall, 0, len(rawCalls))
		for _, rawCall := range rawCalls {
			calls = append(calls, core.ToolCall{
				ID:        rawCall.ID,
				Name:      rawCall.Name,
				Arguments: maps.Clone(rawCall.Arguments),
				Metadata:  maps.Clone(rawCall.Metadata),
			})
		}
		return calls, nil
	}
	if call, ok := result["function_call"].(map[string]any); ok {
		arguments, _ := call["arguments"].(map[string]any)
		metadata, _ := call["metadata"].(map[string]any)
		return []core.ToolCall{{
			ID:        agentutil.StringValue(call["id"]),
			Name:      agentutil.StringValue(call["name"]),
			Arguments: maps.Clone(arguments),
			Metadata:  maps.Clone(metadata),
		}}, nil
	}
	return nil, nil
}

func extractFinishAnswer(arguments map[string]any, fallback string) string {
	if answer, ok := arguments["answer"].(string); ok && strings.TrimSpace(answer) != "" {
		return strings.TrimSpace(answer)
	}
	return strings.TrimSpace(fallback)
}

func supportsToolCalling(llm core.CapabilityProvider) bool {
	for _, capability := range llm.Capabilities() {
		if capability == core.CapabilityToolCalling {
			return true
		}
	}
	return false
}

func supportsNativeToolCalling(llm core.LLM) bool {
	if llm == nil {
		return false
	}
	if _, ok := unwrapLLM(llm).(core.ToolCallingChatLLM); ok {
		return true
	}
	return false
}

func unwrapLLM(llm core.LLM) core.LLM {
	type unwrapCapable interface {
		Unwrap() core.LLM
	}

	current := llm
	for current != nil {
		unwrapped, ok := current.(unwrapCapable)
		if !ok {
			return current
		}
		next := unwrapped.Unwrap()
		if next == nil || next == current {
			return current
		}
		current = next
	}
	return llm
}

func cloneRegisteredTool(tool core.Tool) (core.Tool, error) {
	if tool == nil {
		return nil, fmt.Errorf("cannot clone nil tool")
	}
	cloneable, ok := tool.(core.CloneableTool)
	if !ok {
		return nil, fmt.Errorf("tool %q does not implement core.CloneableTool", tool.Name())
	}
	cloned := cloneable.CloneTool()
	if cloned == nil {
		return nil, fmt.Errorf("tool %q returned nil clone", tool.Name())
	}
	return cloned, nil
}

func cloneContentBlocks(blocks []core.ContentBlock) []core.ContentBlock {
	if len(blocks) == 0 {
		return nil
	}
	cloned := make([]core.ContentBlock, len(blocks))
	for i, block := range blocks {
		cloned[i] = block
		if block.Metadata != nil {
			cloned[i].Metadata = maps.Clone(block.Metadata)
		}
		if len(block.Data) > 0 {
			cloned[i].Data = append([]byte(nil), block.Data...)
		}
	}
	return cloned
}

func blockedReason(err error) string {
	var blocked *core.ToolBlockedError
	if errors.As(err, &blocked) && blocked != nil && strings.TrimSpace(blocked.Reason) != "" {
		return blocked.Reason
	}
	return "blocked by tool policy"
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

func tokenUsageToMap(usage TokenUsage) map[string]any {
	return map[string]any{
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
	}
}

func countExecutedTools(steps []TraceStep) int {
	count := 0
	for _, step := range steps {
		if step.ToolName != "" && !strings.EqualFold(step.ToolName, "finish") {
			count++
		}
	}
	return count
}

func toolResponseDiagnosticMetadata(result map[string]any) map[string]string {
	raw, ok := result["provider_diagnostic"].(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}

	metadata := make(map[string]string)
	for _, key := range []string{"provider", "provider_mode", "reason", "finish_reason"} {
		if value, ok := raw[key]; ok {
			if text := strings.TrimSpace(fmt.Sprint(value)); text != "" && text != "<nil>" {
				metadata[key] = text
			}
		}
	}
	if value, ok := raw["candidate_count"]; ok {
		metadata["candidate_count"] = strings.TrimSpace(fmt.Sprint(value))
	}
	if value, ok := raw["part_count"]; ok {
		metadata["part_count"] = strings.TrimSpace(fmt.Sprint(value))
	}
	return metadata
}

func toolResponseDiagnosticReason(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}
	return metadata["reason"]
}

func toolResponseDiagnosticFinishReason(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}
	return metadata["finish_reason"]
}

func boolError(isError bool, observation string) string {
	if !isError {
		return ""
	}
	return observation
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

const defaultSystemPrompt = `You are a tool-calling agent.

Work only inside the provided task scope.
Use the available tools to inspect files, edit files, and run commands.
Prefer concrete progress over narration.`

const defaultToolPolicy = `Use narrow, evidence-seeking tool calls.
Start by locating the relevant files before making edits.
Prefer short targeted reads and focused shell commands over broad dumps.
After meaningful edits, run the smallest relevant verification before continuing.
Do not loop indefinitely: if you have enough evidence and the task is complete, call Finish.`
