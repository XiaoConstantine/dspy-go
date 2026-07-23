package sessionevent

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// EventProjectionConfig describes wrapper-owned session entry fields that are
// not present in provider-neutral run events.
type EventProjectionConfig struct {
	RunID    string
	TaskID   string
	Source   string
	UserText string
}

type projectedRunState struct {
	runID        string
	started      *agents.RunStartedEvent
	startedAt    time.Time
	terminal     *agents.RunFinishedEvent
	finishedAt   time.Time
	messages     []projectedMessage
	turnUsage    map[int]*core.TokenInfo
	toolOutcomes map[int]map[int]agents.ToolCallFinishedEvent
}

type projectedMessage struct {
	turn      int
	message   agents.Message
	createdAt time.Time
}

// EntriesFromEvents persists canonical run messages and terminal events
// directly to sessionevent entries.
func EntriesFromEvents(events []agents.ExecutionEvent, sessionID, branchID string, config EventProjectionConfig) ([]SessionEntry, error) {
	if strings.TrimSpace(sessionID) == "" || strings.TrimSpace(branchID) == "" {
		return nil, nil
	}
	state, err := collectProjectedRunState(events, config.RunID)
	if err != nil {
		return nil, err
	}
	if state.started == nil && state.terminal == nil && len(state.messages) == 0 {
		return nil, nil
	}

	provider := ""
	model := ""
	taskText := strings.TrimSpace(config.UserText)
	if state.started != nil {
		provider = strings.TrimSpace(state.started.Provider)
		model = strings.TrimSpace(state.started.Model)
		if taskText == "" {
			taskText = strings.TrimSpace(state.started.Task)
		}
	}
	source := strings.TrimSpace(config.Source)
	if source == "" {
		source = "agents"
	}

	entries := make([]SessionEntry, 0, len(state.messages)+2)
	appendEntry := func(entry SessionEntry) {
		entry.SessionID = sessionID
		entry.BranchID = branchID
		entries = append(entries, entry)
	}

	if taskText != "" {
		appendEntry(SessionEntry{
			Kind:       EntryKindUserMessage,
			Role:       string(agents.RoleUser),
			CreatedAt:  fallbackEntryTime(state.startedAt, entries),
			SearchText: taskText,
			Payload: map[string]any{
				"text":    taskText,
				"task_id": strings.TrimSpace(config.TaskID),
			},
			Metadata: map[string]any{
				"source":   source,
				"task_id":  strings.TrimSpace(config.TaskID),
				"provider": provider,
				"model":    model,
			},
		})
	}

	lastAssistantText := ""
	consumedOutcomes := map[int]map[int]bool{}
	for _, record := range state.messages {
		message := record.message
		switch message.Role {
		case agents.RoleAssistant:
			text := strings.TrimSpace(message.TextContent())
			usage := state.turnUsage[record.turn]
			if text != "" {
				lastAssistantText = text
				appendEntry(SessionEntry{
					Kind:             EntryKindAssistantMessage,
					Role:             string(agents.RoleAssistant),
					CreatedAt:        fallbackEntryTime(record.createdAt, entries),
					SearchText:       text,
					PromptTokens:     tokenValue(usage, func(info *core.TokenInfo) int { return info.PromptTokens }),
					CompletionTokens: tokenValue(usage, func(info *core.TokenInfo) int { return info.CompletionTokens }),
					TotalTokens:      tokenValue(usage, func(info *core.TokenInfo) int { return info.TotalTokens }),
					Payload: map[string]any{
						"text": text,
						"turn": record.turn,
					},
					Metadata: map[string]any{
						"source": source,
						"turn":   record.turn,
					},
				})
			}
			for index, call := range message.ToolCalls {
				outcome, ok := state.toolOutcomes[record.turn][index]
				if !ok {
					return nil, fmt.Errorf("tool call %q in turn %d index %d has no finished lifecycle", call.Name, record.turn, index)
				}
				if outcome.Call.Name != call.Name || outcome.Call.ID != call.ID {
					return nil, fmt.Errorf("tool call %q in turn %d index %d does not match finished lifecycle", call.Name, record.turn, index)
				}
				if outcome.Outcome == agents.ToolCallOutcomeFinish {
					continue
				}
				appendEntry(SessionEntry{
					Kind:       EntryKindToolCall,
					Role:       string(agents.RoleAssistant),
					ToolName:   call.Name,
					CreatedAt:  fallbackEntryTime(record.createdAt.Add(time.Duration(index+1)*time.Nanosecond), entries),
					SearchText: strings.TrimSpace(call.Name),
					Payload: map[string]any{
						"arguments":    cloneAnyMap(call.Arguments),
						"tool_call_id": strings.TrimSpace(call.ID),
						"tool_index":   index,
						"turn":         record.turn,
					},
					Metadata: map[string]any{
						"source":       source,
						"turn":         record.turn,
						"tool_call_id": strings.TrimSpace(call.ID),
						"tool_index":   index,
					},
				})
			}
		case agents.RoleTool:
			if message.ToolResult == nil {
				continue
			}
			toolIndex, ok := consumeToolOutcomeIndex(state, record.turn, message, consumedOutcomes)
			if !ok {
				return nil, fmt.Errorf("tool result for %q in turn %d has no matching finished lifecycle", message.ToolResult.Name, record.turn)
			}
			appendEntry(SessionEntry{
				Kind:       EntryKindToolResult,
				ToolName:   message.ToolResult.Name,
				CreatedAt:  fallbackEntryTime(record.createdAt, entries),
				IsError:    message.ToolResult.IsError,
				Synthetic:  message.ToolResult.Synthetic,
				Redacted:   message.ToolResult.Redacted,
				Truncated:  message.ToolResult.Truncated,
				SearchText: firstNonEmptyText(strings.TrimSpace(contentBlocksText(message.ToolResult.DisplayContent)), strings.TrimSpace(contentBlocksText(message.ToolResult.Content))),
				Payload: map[string]any{
					"tool_call_id":          strings.TrimSpace(message.ToolResult.ToolCallID),
					"observation":           strings.TrimSpace(contentBlocksText(message.ToolResult.Content)),
					"observation_display":   strings.TrimSpace(contentBlocksText(message.ToolResult.DisplayContent)),
					"details":               cloneAnyMap(message.ToolResult.Details),
					"provider_visible_text": strings.TrimSpace(contentBlocksText(message.ToolResult.Content)),
					"tool_index":            toolIndex,
					"turn":                  record.turn,
				},
				Metadata: map[string]any{
					"source":       source,
					"tool_call_id": strings.TrimSpace(message.ToolResult.ToolCallID),
					"tool_index":   toolIndex,
					"turn":         record.turn,
				},
			})
		}
	}

	finalAnswer := ""
	status := agents.RunStatusFailed
	stopReason := agents.StopReasonError
	diagnostic := ""
	errText := ""
	finishedAt := time.Time{}
	if state.terminal != nil {
		finalAnswer = strings.TrimSpace(state.terminal.FinalAnswer)
		status = state.terminal.Status
		stopReason = state.terminal.StopReason
		diagnostic = strings.TrimSpace(state.terminal.Diagnostic)
		finishedAt = state.finishedAt
		if state.terminal.Err != nil {
			errText = state.terminal.Err.Error()
		}
	}
	if finalAnswer != "" && finalAnswer != lastAssistantText {
		appendEntry(SessionEntry{
			Kind:       EntryKindAssistantMessage,
			Role:       string(agents.RoleAssistant),
			CreatedAt:  fallbackEntryTime(finishedAt.Add(-time.Nanosecond), entries),
			SearchText: finalAnswer,
			Payload: map[string]any{
				"text":  finalAnswer,
				"final": true,
			},
			Metadata: map[string]any{
				"source": source,
			},
		})
	}

	tokens := aggregateTurnUsage(state.turnUsage)
	completed := status == agents.RunStatusCompleted
	appendEntry(SessionEntry{
		Kind:             EntryKindSystemEvent,
		CreatedAt:        fallbackEntryTime(finishedAt, entries),
		SearchText:       firstNonEmptyText(finalAnswer, errText, diagnostic, string(stopReason), taskText),
		PromptTokens:     tokens["prompt_tokens"],
		CompletionTokens: tokens["completion_tokens"],
		TotalTokens:      tokens["total_tokens"],
		Payload: map[string]any{
			"event":             "run_finished",
			"task_id":           strings.TrimSpace(config.TaskID),
			"completed":         completed,
			"final_answer":      finalAnswer,
			"error":             errText,
			"diagnostic":        diagnostic,
			"provider":          provider,
			"model":             model,
			"status":            string(status),
			"stop_reason":       string(stopReason),
			"started_at":        state.startedAt.UTC().Format(time.RFC3339Nano),
			"duration_ms":       fallbackEntryTime(finishedAt, entries).Sub(fallbackEntryTime(state.startedAt, nil)).Milliseconds(),
			"prompt_tokens":     tokens["prompt_tokens"],
			"completion_tokens": tokens["completion_tokens"],
			"total_tokens":      tokens["total_tokens"],
		},
		Metadata: map[string]any{
			"source": source,
		},
	})
	return entries, nil
}

func collectProjectedRunState(events []agents.ExecutionEvent, selectedRunID string) (projectedRunState, error) {
	selected, runID, err := selectProjectedLifecycle(events, selectedRunID)
	if err != nil {
		return projectedRunState{}, err
	}
	state := projectedRunState{runID: runID, turnUsage: map[int]*core.TokenInfo{}, toolOutcomes: map[int]map[int]agents.ToolCallFinishedEvent{}}
	for _, event := range selected {
		eventRunID, ok := projectionRunID(event.Payload)
		if !ok {
			continue
		}
		if strings.TrimSpace(eventRunID) != runID {
			return projectedRunState{}, fmt.Errorf("run lifecycle %q contains event for run %q", runID, strings.TrimSpace(eventRunID))
		}
		switch payload := event.Payload.(type) {
		case agents.RunStartedEvent:
			copied := payload
			state.started = &copied
			state.startedAt = event.Timestamp.UTC()
		case agents.RunFinishedEvent:
			copied := payload
			state.terminal = &copied
			state.finishedAt = event.Timestamp.UTC()
		case agents.TurnFinishedEvent:
			if payload.Usage != nil {
				state.turnUsage[payload.Turn] = &core.TokenInfo{PromptTokens: payload.Usage.PromptTokens, CompletionTokens: payload.Usage.CompletionTokens, TotalTokens: payload.Usage.TotalTokens}
			}
		case agents.MessageAddedEvent:
			state.messages = append(state.messages, projectedMessage{turn: payload.Turn, message: payload.Message.Clone(), createdAt: event.Timestamp.UTC()})
		case agents.ToolCallFinishedEvent:
			if state.toolOutcomes[payload.Turn] == nil {
				state.toolOutcomes[payload.Turn] = map[int]agents.ToolCallFinishedEvent{}
			}
			state.toolOutcomes[payload.Turn][payload.ToolIndex] = payload
		}
	}
	return state, nil
}

func selectProjectedLifecycle(events []agents.ExecutionEvent, selectedRunID string) ([]agents.ExecutionEvent, string, error) {
	selectedRunID = strings.TrimSpace(selectedRunID)
	starts := make([]int, 0, 2)
	for index, event := range events {
		started, ok := event.Payload.(agents.RunStartedEvent)
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
	runID := strings.TrimSpace(events[start].Payload.(agents.RunStartedEvent).RunID)
	end := len(events)
	for index := start + 1; index < len(events); index++ {
		if _, ok := events[index].Payload.(agents.RunStartedEvent); ok {
			end = index
			break
		}
		if finished, ok := events[index].Payload.(agents.RunFinishedEvent); ok && strings.TrimSpace(finished.RunID) == runID {
			end = index + 1
			break
		}
	}
	return events[start:end], runID, nil
}

func projectionRunID(payload agents.EventPayload) (string, bool) {
	switch event := payload.(type) {
	case agents.RunStartedEvent:
		return event.RunID, true
	case agents.RunFinishedEvent:
		return event.RunID, true
	case agents.TurnStartedEvent:
		return event.RunID, true
	case agents.TurnFinishedEvent:
		return event.RunID, true
	case agents.MessageAddedEvent:
		return event.RunID, true
	case agents.ToolCallProposedEvent:
		return event.RunID, true
	case agents.ToolExecutionStartedEvent:
		return event.RunID, true
	case agents.ToolCallFinishedEvent:
		return event.RunID, true
	default:
		return "", false
	}
}

func consumeToolOutcomeIndex(state projectedRunState, turn int, message agents.Message, consumed map[int]map[int]bool) (int, bool) {
	if message.ToolResult == nil {
		return 0, false
	}
	indexes := make([]int, 0, len(state.toolOutcomes[turn]))
	for index := range state.toolOutcomes[turn] {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		if consumed[turn] != nil && consumed[turn][index] {
			continue
		}
		outcome := state.toolOutcomes[turn][index]
		if outcome.Result == nil || outcome.Result.ToolResult == nil {
			continue
		}
		result := outcome.Result.ToolResult
		if outcome.Call.Name != message.ToolResult.Name || result.Name != message.ToolResult.Name {
			continue
		}
		if outcome.Call.ID != message.ToolResult.ToolCallID || result.ToolCallID != message.ToolResult.ToolCallID {
			continue
		}
		if consumed[turn] == nil {
			consumed[turn] = map[int]bool{}
		}
		consumed[turn][index] = true
		return index, true
	}
	return 0, false
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

func fallbackEntryTime(ts time.Time, entries []SessionEntry) time.Time {
	if !ts.IsZero() {
		return ts.UTC()
	}
	if len(entries) == 0 {
		return time.Now().UTC()
	}
	return entries[len(entries)-1].CreatedAt.Add(time.Nanosecond).UTC()
}

func tokenValue(info *core.TokenInfo, get func(*core.TokenInfo) int) int64 {
	if info == nil {
		return 0
	}
	return int64(get(info))
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

func firstNonEmptyText(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func cloneAnyMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	cloned := make(map[string]any, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}
