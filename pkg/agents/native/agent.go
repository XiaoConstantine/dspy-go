package native

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/skills"
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
	EventSink                     agents.EventSink
	SessionEventSink              SessionEventSink
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

	artifactMu sync.RWMutex
	traceMu    sync.RWMutex
	lastTrace  *agents.ExecutionTrace
}

var (
	_ optimize.OptimizableAgent   = (*Agent)(nil)
	_ agents.ScopedExecutionAgent = (*Agent)(nil)
)

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

// ExecuteWithTrace runs one task and returns its operation-scoped trace.
func (a *Agent) ExecuteWithTrace(ctx context.Context, input map[string]any) (agents.AgentExecutionResult, error) {
	capturedCtx, trace := agents.WithExecutionTraceCapture(ctx)
	output, err := a.Execute(capturedCtx, input)
	return agents.AgentExecutionResult{Output: output, Trace: trace()}, err
}

// Execute runs one native tool-calling task.
func (a *Agent) Execute(ctx context.Context, input map[string]any) (map[string]any, error) {
	if a == nil {
		return nil, fmt.Errorf("tool-calling agent is nil")
	}

	task := agentutil.StringValue(input["task"])
	if strings.TrimSpace(task) == "" {
		return nil, fmt.Errorf("task is required")
	}
	taskID := agentutil.StringValue(input["task_id"])

	a.artifactMu.RLock()
	runConfig := a.config
	persistedSkillPrompt := a.persistedSkillPrompt
	a.artifactMu.RUnlock()
	runConfig.ToolInterceptors = append([]core.ToolInterceptor(nil), runConfig.ToolInterceptors...)
	maxTurns := runConfig.MaxTurns
	if override := agentutil.IntValue(input["max_turns"]); override > 0 {
		maxTurns = override
	}

	sessionID := a.sessionID(input)
	sessionContext, sessionErr := a.loadSessionContext(ctx, input)
	if sessionID != "" {
		a.emitSessionEvent(ctx, SessionLoadedEvent{
			TaskID:            taskID,
			SessionID:         sessionID,
			Source:            sessionContext.Source,
			RecordCount:       sessionContext.RecordCount,
			EntryCount:        sessionContext.EntryCount,
			SummaryCount:      sessionContext.SummaryCount,
			RecallChars:       len(sessionContext.Recall),
			BranchID:          sessionContext.BranchID,
			HeadEntryID:       sessionContext.HeadEntryID,
			ForkedFromEntryID: sessionContext.ForkedFromEntryID,
			Err:               sessionErr,
		})
	}
	sessionRecall := sessionContext.Recall
	if sessionErr != nil {
		sessionRecall = ""
	}

	initialPrompt := buildNativeInitialPrompt(runConfig, persistedSkillPrompt, task, sessionRecall)
	initialMessages := []agents.Message{agents.NewTextMessage(agents.RoleUser, initialPrompt)}
	baseModel, err := agents.NewLLMAdapter(a.llm, agents.WithPromptRenderer(renderNativeModelPrompt))
	if err != nil {
		return nil, err
	}
	model := &nativeRunModel{Model: baseModel, maxTurns: maxTurns}
	var typedEvents []agents.ExecutionEvent
	eventSink := agents.EventSinkFunc(func(eventCtx context.Context, event agents.ExecutionEvent) {
		stored := agents.CloneExecutionEvent(event)
		typedEvents = append(typedEvents, stored)
		if runConfig.EventSink != nil {
			runConfig.EventSink.EmitEvent(eventCtx, agents.CloneExecutionEvent(event))
		}
	})
	executor := func(executeCtx context.Context, tool core.Tool, arguments map[string]any) (core.ToolResult, error) {
		handler := func(handlerCtx context.Context, args map[string]any) (core.ToolResult, error) {
			return tool.Execute(handlerCtx, args)
		}
		return core.ChainToolInterceptors(runConfig.ToolInterceptors...)(
			executeCtx, arguments, core.NewToolInfoFromTool(tool), handler,
		)
	}

	loopResult, runErr := agents.RunLoop(ctx, model, a.toolRegistry.List(), initialMessages, agents.LoopConfig{
		RunID: taskID, Task: task, MaxTurns: maxTurns,
		Completion: agents.FinishCompletion(runConfig.FinishToolText),
		Options: []core.GenerateOption{
			core.WithMaxTokens(runConfig.MaxTokens),
			core.WithTemperature(runConfig.Temperature),
		},
		Events:                    eventSink,
		Execute:                   executor,
		MaxConsecutiveNoToolCalls: runConfig.MaxConsecutiveNoCallResponses,
	})

	if runErr != nil && len(typedEvents) == 0 {
		return nil, runErr
	}
	execTrace, err := agents.ExecutionTraceFromEvents(typedEvents, agents.ExecutionTraceConfig{
		RunID:     taskID,
		AgentID:   fmt.Sprintf("native-%s-%s", a.llm.ProviderName(), a.llm.ModelID()),
		AgentType: "native",
		Input:     maps.Clone(input),
		Task:      task,
		ContextMetadata: func() map[string]any {
			metadata := map[string]any{}
			if sessionID := a.sessionID(input); sessionID != "" {
				metadata["session_id"] = sessionID
			}
			return metadata
		}(),
	})
	if err != nil {
		return nil, err
	}
	a.storeTrace(ctx, input, execTrace, typedEvents)
	if runErr != nil {
		return nil, runErr
	}

	output := map[string]any{
		"completed":   loopResult.StopReason == agents.StopReasonFinish,
		"tool_calls":  executionTraceToolCalls(execTrace),
		"turns":       len(execTrace.Steps),
		"token_usage": loopTokenUsageMap(loopResult.Usage),
	}
	if loopResult.FinalAnswer != "" {
		output["final_answer"] = loopResult.FinalAnswer
	}
	if loopResult.StopReason != agents.StopReasonFinish {
		errorText := execTrace.Error
		if loopResult.StopReason == agents.StopReasonMaxTurns {
			errorText = fmt.Sprintf("max turns reached without Finish after %d turns", loopResult.Turns)
		}
		output["error"] = errorText
	}
	return output, nil
}

type nativeRunModel struct {
	agents.Model
	turn     int
	maxTurns int
}

func (m *nativeRunModel) Complete(ctx context.Context, request agents.ModelRequest) (agents.ModelResponse, error) {
	m.turn++
	request.Messages = agents.CloneMessages(request.Messages)
	feedback := fmt.Sprintf("TURN BUDGET: turn %d of %d (%d remaining)\nIf the task appears complete or your latest verification/check succeeded, call Finish immediately instead of doing extra inspection.", m.turn, m.maxTurns, m.maxTurns-m.turn+1)
	if reminder := turnBudgetReminder(m.turn, m.maxTurns); reminder != "" {
		feedback += "\n" + reminder
	}
	if diagnostic := latestNoToolDiagnostic(request.Messages); diagnostic != "" {
		feedback += "\n" + diagnostic
	}
	request.Messages = append(request.Messages, agents.NewTextMessage(agents.RoleUser, feedback))
	return m.Model.Complete(ctx, request)
}

func latestNoToolDiagnostic(messages []agents.Message) string {
	for index := len(messages) - 1; index >= 0; index-- {
		message := messages[index]
		if message.Role != agents.RoleAssistant {
			continue
		}
		if len(message.ToolCalls) != 0 {
			return ""
		}
		diagnostics, _ := message.Metadata[agents.ModelDiagnosticsMetadataKey].(map[string]any)
		provider, _ := diagnostics["provider_diagnostic"].(map[string]any)
		parts := make([]string, 0, 2)
		if reason := strings.TrimSpace(fmt.Sprint(provider["reason"])); reason != "" && reason != "<nil>" {
			parts = append(parts, "Provider diagnostic: "+reason)
		}
		if reason := strings.TrimSpace(fmt.Sprint(provider["finish_reason"])); reason != "" && reason != "<nil>" {
			parts = append(parts, "Finish reason: "+reason)
		}
		return strings.Join(parts, " ")
	}
	return ""
}

func renderNativeModelPrompt(request agents.ModelRequest) (string, error) {
	var builder strings.Builder
	for index, message := range request.Messages {
		if index == 0 && message.Role == agents.RoleUser {
			builder.WriteString(message.TextContent())
			continue
		}
		switch message.Role {
		case agents.RoleAssistant:
			if text := strings.TrimSpace(message.TextContent()); text != "" {
				fmt.Fprintf(&builder, "\nAssistant: %s\n", text)
			}
			for _, call := range message.ToolCalls {
				fmt.Fprintf(&builder, "Tool: %s\nArguments: %v\n", call.Name, call.Arguments)
			}
		case agents.RoleTool:
			if message.ToolResult != nil {
				label := "Observation"
				if message.ToolResult.IsError {
					label = "Observation (error)"
				}
				fmt.Fprintf(&builder, "%s: %s\n", label, contentBlocksText(message.ToolResult.Content))
			}
		case agents.RoleUser:
			fmt.Fprintf(&builder, "\n%s\n", message.TextContent())
		}
	}
	return builder.String(), nil
}

func buildNativeInitialPrompt(config Config, persistedSkillPrompt, task, sessionRecall string) string {
	var builder strings.Builder
	builder.WriteString(composeSystemPrompt(config.SystemPrompt, persistedSkillPrompt))
	builder.WriteString("\n\n")
	if strings.TrimSpace(config.ToolPolicy) != "" {
		builder.WriteString("TOOL POLICY:\n")
		builder.WriteString(config.ToolPolicy)
		builder.WriteString("\n\n")
	}
	builder.WriteString("TASK:\n")
	builder.WriteString(task)
	builder.WriteString("\n\n")
	if strings.TrimSpace(sessionRecall) != "" {
		builder.WriteString("SESSION RECALL:\n")
		builder.WriteString(sessionRecall)
		builder.WriteString("\n\n")
	}
	builder.WriteString("You must use tools to work on the task and call Finish when done.\n")
	return builder.String()
}

func contentBlocksText(blocks []core.ContentBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		if block.Type == core.FieldTypeText && strings.TrimSpace(block.Text) != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func loopTokenUsageMap(usage *core.TokenInfo) map[string]int64 {
	if usage == nil {
		return map[string]int64{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	}
	return map[string]int64{
		"prompt_tokens": int64(usage.PromptTokens), "completion_tokens": int64(usage.CompletionTokens),
		"total_tokens": int64(usage.TotalTokens),
	}
}

// AgentContract describes native agent input/output fields for composition.
func (a *Agent) AgentContract() agents.AgentContract {
	return agents.AgentContract{
		Inputs:       []agents.AgentField{{Name: "task", Description: "Task for the native coding agent", Required: true}},
		Outputs:      []agents.AgentField{{Name: "final_answer", Description: "Final task result"}, {Name: "completed", Description: "Whether the task completed", Required: true}},
		PrimaryInput: "task",
	}
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

	a.artifactMu.RLock()
	defer a.artifactMu.RUnlock()
	return a.artifactsLocked()
}

// OptimizationAgentType returns the stable persisted optimization envelope type.
func (a *Agent) OptimizationAgentType() string {
	return "native"
}

// ListOptimizationTargets returns the stable optimization targets supported by the native agent.
func (a *Agent) ListOptimizationTargets() []optimize.OptimizationTargetDescriptor {
	return []optimize.OptimizationTargetDescriptor{
		{
			ID:          "root.system",
			Kind:        optimize.OptimizationTargetText,
			Description: "Primary system prompt and persisted skill guidance.",
			ArtifactKey: optimize.ArtifactSkillPack,
		},
		{
			ID:          "root.tool_policy",
			Kind:        optimize.OptimizationTargetText,
			Description: "Tool-use policy and guardrails.",
			ArtifactKey: optimize.ArtifactToolPolicy,
		},
		{
			ID:          "root.max_turns",
			Kind:        optimize.OptimizationTargetInt,
			Description: "Maximum tool-calling turns for one task.",
			IntKey:      "max_turns",
		},
	}
}

// ExportOptimizedProgram exports the native agent's current artifacts into the shared persisted envelope.
func (a *Agent) ExportOptimizedProgram() (*optimize.OptimizedAgentProgram, error) {
	return optimize.ExportOptimizedAgentProgram(a)
}

// ApplyOptimizedProgram applies a shared persisted optimization envelope onto the native agent.
func (a *Agent) ApplyOptimizedProgram(program *optimize.OptimizedAgentProgram) error {
	return optimize.ApplyOptimizedAgentProgram(a, program)
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
	a.artifactMu.Lock()
	defer a.artifactMu.Unlock()
	return a.setArtifactsLocked(artifacts)
}

// UpdateArtifacts atomically reads, transforms, and reapplies the native agent artifacts.
func (a *Agent) UpdateArtifacts(update func(optimize.AgentArtifacts) (optimize.AgentArtifacts, error)) error {
	if a == nil {
		return fmt.Errorf("tool-calling agent is nil")
	}
	if update == nil {
		return fmt.Errorf("artifact update function is nil")
	}

	a.artifactMu.Lock()
	defer a.artifactMu.Unlock()

	next, err := update(a.artifactsLocked())
	if err != nil {
		return err
	}
	return a.setArtifactsLocked(next)
}

// Clone creates a fresh agent with the same LLM, config, and registered tools.
func (a *Agent) Clone() (optimize.OptimizableAgent, error) {
	if a == nil {
		return nil, fmt.Errorf("tool-calling agent is nil")
	}

	a.artifactMu.RLock()
	cfg := a.config
	a.artifactMu.RUnlock()
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

func (a *Agent) artifactsLocked() optimize.AgentArtifacts {
	artifacts := optimize.AgentArtifacts{
		Text: make(map[optimize.ArtifactKey]string),
		Int:  make(map[string]int),
		Bool: make(map[string]bool),
	}
	artifacts.Text[optimize.ArtifactSkillPack] = composeSystemPrompt(a.config.SystemPrompt, a.persistedSkillPrompt)
	artifacts.Text[optimize.ArtifactToolPolicy] = a.config.ToolPolicy
	artifacts.Int["max_turns"] = a.config.MaxTurns
	return artifacts
}

func (a *Agent) setArtifactsLocked(artifacts optimize.AgentArtifacts) error {
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

// LastExecutionTrace returns the most recent execution trace.
func (a *Agent) LastExecutionTrace() *agents.ExecutionTrace {
	if a == nil {
		return nil
	}
	a.traceMu.RLock()
	defer a.traceMu.RUnlock()
	return a.lastTrace.Clone()
}

func (a *Agent) emitSessionEvent(ctx context.Context, payload SessionEventPayload) {
	if a == nil || a.config.SessionEventSink == nil || payload == nil {
		return
	}
	a.config.SessionEventSink.EmitSessionEvent(ctx, SessionEvent{
		Timestamp: time.Now().UTC(),
		Payload:   payload,
	})
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

func (a *Agent) storeTrace(ctx context.Context, input map[string]any, trace *agents.ExecutionTrace, events []agents.ExecutionEvent) {
	a.traceMu.Lock()
	a.lastTrace = trace.Clone()
	a.traceMu.Unlock()
	agents.PublishExecutionTrace(ctx, trace)
	a.persistSessionRecord(ctx, input, trace, events)
}

func supportsToolCalling(llm core.CapabilityProvider) bool {
	for _, capability := range llm.Capabilities() {
		if capability == core.CapabilityToolCalling {
			return true
		}
	}
	return false
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
