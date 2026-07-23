package agents

import (
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ExecutionTraceConfig describes wrapper-owned trace fields that are not
// present in provider-neutral run events.
type ExecutionTraceConfig struct {
	RunID           string
	AgentID         string
	AgentType       string
	Input           map[string]any
	Task            string
	ContextMetadata map[string]any
}

type runEventState struct {
	runID        string
	started      *RunStartedEvent
	startedAt    time.Time
	terminal     *RunFinishedEvent
	finishedAt   time.Time
	messages     []runMessageRecord
	turnUsage    map[int]*core.TokenInfo
	toolOutcomes map[int]map[int]ToolCallFinishedEvent
}

type runMessageRecord struct {
	turn      int
	message   Message
	createdAt time.Time
}

// ExecutionTraceFromEvents builds a reusable execution trace from canonical run
// events instead of wrapper-private trace state.
func ExecutionTraceFromEvents(events []ExecutionEvent, config ExecutionTraceConfig) (*ExecutionTrace, error) {
	state, err := collectRunEventState(events, config.RunID)
	if err != nil {
		return nil, err
	}
	if state.started == nil && state.terminal == nil && len(state.messages) == 0 {
		return nil, nil
	}

	task := strings.TrimSpace(config.Task)
	if task == "" && state.started != nil {
		task = strings.TrimSpace(state.started.Task)
	}
	status := TraceStatusFailure
	terminalStatus := RunStatusFailed
	stopReason := StopReasonError
	finalAnswer := ""
	errText := ""
	if state.terminal != nil {
		status = traceStatusFromRunStatus(state.terminal.Status)
		terminalStatus = state.terminal.Status
		stopReason = state.terminal.StopReason
		finalAnswer = strings.TrimSpace(state.terminal.FinalAnswer)
		if state.terminal.Err != nil {
			errText = state.terminal.Err.Error()
		} else if strings.TrimSpace(state.terminal.Diagnostic) != "" {
			errText = strings.TrimSpace(state.terminal.Diagnostic)
		}
	}

	steps, toolUsage, tokenUsage, err := executionTraceStepsFromState(state)
	if err != nil {
		return nil, err
	}
	contextMetadata := cloneAnyMap(config.ContextMetadata)
	if contextMetadata == nil {
		contextMetadata = map[string]any{}
	}
	contextMetadata["turns"] = len(steps)

	output := map[string]any{"completed": terminalStatus == RunStatusCompleted}
	if finalAnswer != "" {
		output["final_answer"] = finalAnswer
	}

	startedAt := state.startedAt.UTC()
	completedAt := state.finishedAt.UTC()
	if completedAt.IsZero() {
		completedAt = startedAt
	}
	if startedAt.IsZero() && !completedAt.IsZero() {
		startedAt = completedAt
	}

	return &ExecutionTrace{
		AgentID:        config.AgentID,
		AgentType:      config.AgentType,
		Task:           task,
		Input:          cloneAnyMap(config.Input),
		Output:         output,
		Steps:          steps,
		Status:         status,
		Error:          errText,
		StartedAt:      startedAt,
		CompletedAt:    completedAt,
		ProcessingTime: completedAt.Sub(startedAt),
		TokenUsage:     tokenUsage,
		ToolUsageCount: toolUsage,
		ContextMetadata: func() map[string]any {
			if len(contextMetadata) == 0 {
				return nil
			}
			return contextMetadata
		}(),
		TerminationCause: string(stopReason),
	}, nil
}

func collectRunEventState(events []ExecutionEvent, selectedRunID string) (runEventState, error) {
	selected, runID, err := selectExecutionLifecycle(events, selectedRunID)
	if err != nil {
		return runEventState{}, err
	}
	state := runEventState{runID: runID, turnUsage: map[int]*core.TokenInfo{}, toolOutcomes: map[int]map[int]ToolCallFinishedEvent{}}
	for _, event := range selected {
		eventRunID, ok := executionEventRunID(event.Payload)
		if !ok {
			continue
		}
		if strings.TrimSpace(eventRunID) != runID {
			return runEventState{}, fmt.Errorf("run lifecycle %q contains event for run %q", runID, strings.TrimSpace(eventRunID))
		}
		switch payload := event.Payload.(type) {
		case RunStartedEvent:
			copied := payload
			state.started = &copied
			state.startedAt = event.Timestamp.UTC()
		case RunFinishedEvent:
			copied := payload
			state.terminal = &copied
			state.finishedAt = event.Timestamp.UTC()
		case TurnFinishedEvent:
			if payload.Usage != nil {
				state.turnUsage[payload.Turn] = &core.TokenInfo{
					PromptTokens:     payload.Usage.PromptTokens,
					CompletionTokens: payload.Usage.CompletionTokens,
					TotalTokens:      payload.Usage.TotalTokens,
				}
			}
		case MessageAddedEvent:
			state.messages = append(state.messages, runMessageRecord{turn: payload.Turn, message: payload.Message.Clone(), createdAt: event.Timestamp.UTC()})
		case ToolCallFinishedEvent:
			if state.toolOutcomes[payload.Turn] == nil {
				state.toolOutcomes[payload.Turn] = map[int]ToolCallFinishedEvent{}
			}
			copied := cloneEventPayload(payload).(ToolCallFinishedEvent)
			state.toolOutcomes[payload.Turn][payload.ToolIndex] = copied
		}
	}
	return state, nil
}

func selectExecutionLifecycle(events []ExecutionEvent, selectedRunID string) ([]ExecutionEvent, string, error) {
	selectedRunID = strings.TrimSpace(selectedRunID)
	starts := make([]int, 0, 2)
	for index, event := range events {
		started, ok := event.Payload.(RunStartedEvent)
		if !ok {
			continue
		}
		if selectedRunID == "" || strings.TrimSpace(started.RunID) == selectedRunID {
			starts = append(starts, index)
		}
	}
	if len(starts) > 1 {
		return nil, "", fmt.Errorf("event buffer contains %d matching run lifecycles; select one lifecycle before projection", len(starts))
	}
	if len(starts) == 0 {
		if selectedRunID != "" {
			return nil, selectedRunID, nil
		}
		return events, "", nil
	}
	start := starts[0]
	runID := strings.TrimSpace(events[start].Payload.(RunStartedEvent).RunID)
	end := len(events)
	for index := start + 1; index < len(events); index++ {
		if _, ok := events[index].Payload.(RunStartedEvent); ok {
			end = index
			break
		}
		if finished, ok := events[index].Payload.(RunFinishedEvent); ok && strings.TrimSpace(finished.RunID) == runID {
			end = index + 1
			break
		}
	}
	return events[start:end], runID, nil
}

func executionEventRunID(payload EventPayload) (string, bool) {
	switch event := payload.(type) {
	case RunStartedEvent:
		return event.RunID, true
	case RunFinishedEvent:
		return event.RunID, true
	case TurnStartedEvent:
		return event.RunID, true
	case TurnFinishedEvent:
		return event.RunID, true
	case MessageAddedEvent:
		return event.RunID, true
	case ToolCallProposedEvent:
		return event.RunID, true
	case ToolExecutionStartedEvent:
		return event.RunID, true
	case ToolCallFinishedEvent:
		return event.RunID, true
	default:
		return "", false
	}
}

func executionTraceStepsFromState(state runEventState) ([]TraceStep, map[string]int, map[string]int64, error) {
	steps := make([]TraceStep, 0, len(state.messages))
	toolUsage := make(map[string]int)
	tokens := aggregateTurnUsage(state.turnUsage)
	for _, record := range state.messages {
		message := record.message
		if message.Role != RoleAssistant {
			continue
		}
		assistantText := message.TextContent()
		if len(message.ToolCalls) == 0 {
			if state.terminal != nil && state.terminal.Status == RunStatusCompleted && state.terminal.StopReason == StopReasonText && record.turn == state.terminal.Turns {
				steps = append(steps, TraceStep{
					Index:   len(steps) + 1,
					Thought: assistantText,
					Success: true,
				})
				continue
			}
			observation := "Model returned text without a tool call. It must use a tool or Finish."
			if diagnostic := providerDiagnosticText(message.Metadata); diagnostic != "" {
				observation += " " + diagnostic
			}
			steps = append(steps, TraceStep{
				Index:              len(steps) + 1,
				Thought:            assistantText,
				Observation:        observation,
				ObservationDisplay: observation,
				Success:            false,
				Error:              observation,
			})
			continue
		}
		for index, call := range message.ToolCalls {
			step := TraceStep{
				Index:     len(steps) + 1,
				Tool:      call.Name,
				Arguments: cloneAnyMap(call.Arguments),
				Success:   true,
			}
			if index == 0 {
				step.Thought = assistantText
			}
			result, err := strictToolOutcomeResult(state, record.turn, index, call)
			if err != nil {
				return nil, nil, nil, err
			}
			if state.toolOutcomes[record.turn][index].Outcome != ToolCallOutcomeFinish {
				toolUsage[call.Name]++
			}
			if result != nil {
				step.Observation = contentBlocksText(result.ToolResult.Content)
				step.ObservationDisplay = contentBlocksText(result.ToolResult.DisplayContent)
				step.ObservationDetails = cloneAnyMap(result.ToolResult.Details)
				step.Success = !result.ToolResult.IsError
				if result.ToolResult.IsError {
					step.Error = step.Observation
				}
				step.Synthetic = result.ToolResult.Synthetic
				step.Redacted = result.ToolResult.Redacted
				step.Truncated = result.ToolResult.Truncated
			}
			steps = append(steps, step)
		}
	}
	return steps, toolUsage, tokens, nil
}

func strictToolOutcomeResult(state runEventState, turn, toolIndex int, call core.ToolCall) (*Message, error) {
	outcome, ok := state.toolOutcomes[turn][toolIndex]
	if !ok {
		return nil, fmt.Errorf("tool call %q at turn %d index %d has no finished lifecycle", call.Name, turn, toolIndex)
	}
	if outcome.Call.Name != call.Name || outcome.Call.ID != call.ID {
		return nil, fmt.Errorf("tool call %q at turn %d index %d does not match finished lifecycle", call.Name, turn, toolIndex)
	}
	if outcome.Outcome == ToolCallOutcomeFinish {
		if outcome.Result != nil {
			return nil, fmt.Errorf("Finish call at turn %d index %d unexpectedly has a result", turn, toolIndex)
		}
		return nil, nil
	}
	if outcome.Result == nil {
		return nil, fmt.Errorf("tool call %q at turn %d index %d finished without a result", call.Name, turn, toolIndex)
	}
	result := outcome.Result.Clone()
	if result.ToolResult == nil || result.ToolResult.Name != call.Name || result.ToolResult.ToolCallID != call.ID {
		return nil, fmt.Errorf("tool result for %q at turn %d index %d does not match its call", call.Name, turn, toolIndex)
	}
	return &result, nil
}

func aggregateTurnUsage(turnUsage map[int]*core.TokenInfo) map[string]int64 {
	tokens := map[string]int64{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	for _, usage := range turnUsage {
		if usage == nil {
			continue
		}
		tokens["prompt_tokens"] += int64(usage.PromptTokens)
		tokens["completion_tokens"] += int64(usage.CompletionTokens)
		tokens["total_tokens"] += int64(usage.TotalTokens)
	}
	return tokens
}

func traceStatusFromRunStatus(status RunStatus) TraceStatus {
	switch status {
	case RunStatusCompleted:
		return TraceStatusSuccess
	case RunStatusStopped:
		return TraceStatusPartial
	default:
		return TraceStatusFailure
	}
}

func providerDiagnosticText(metadata map[string]any) string {
	diagnostics, _ := metadata[ModelDiagnosticsMetadataKey].(map[string]any)
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
