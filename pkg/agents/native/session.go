package native

import (
	"context"
	goerrors "errors"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/agentutil"
)

const (
	defaultSessionRecallLimit    = 3
	defaultSessionRecallMaxChars = 1200
	maxSessionHighlights         = 3
	maxSessionHighlightChars     = 160
	maxSessionTaskChars          = 160
	maxSessionAnswerChars        = 220
)

func (a *Agent) sessionID(input map[string]any) string {
	if id := strings.TrimSpace(agentutil.StringValue(input["session_id"])); id != "" {
		return id
	}
	return strings.TrimSpace(a.config.SessionID)
}

func (a *Agent) sessionStore() *agents.SessionStore {
	if a == nil {
		return nil
	}
	return a.sessions
}

func (a *Agent) sessionEventStore() sessionevent.SessionEventStore {
	if a == nil {
		return nil
	}
	return a.sessionEvent
}

func (a *Agent) loadSessionContext(input map[string]any) ([]agents.SessionRecord, string, error) {
	sessionID := a.sessionID(input)
	if sessionID == "" {
		return nil, "", nil
	}

	records, err := a.sessionStore().Recent(sessionID, a.config.SessionRecallLimit)
	if err != nil {
		return nil, "", err
	}
	return records, buildSessionRecall(records, a.config.SessionRecallMaxChars), nil
}

func (a *Agent) persistSessionRecord(ctx context.Context, input map[string]any, trace *Trace) {
	sessionID := a.sessionID(input)
	if sessionID == "" || trace == nil {
		return
	}

	eventStore := a.sessionEventStore()
	record := sessionRecordFromTrace(a, input, trace, sessionID)
	snapshotErr := a.sessionStore().Append(record)

	var (
		eventEntries []sessionevent.SessionEntry
		eventBranch  string
		eventErr     error
	)
	if eventStore != nil {
		eventEntries, eventBranch, eventErr = a.persistSessionEventTrace(ctx, eventStore, trace, sessionID)
	}

	eventData := map[string]any{
		"session_id": sessionID,
		"record_id":  record.ID,
		"task_id":    trace.TaskID,
		"success":    snapshotErr == nil,
		"completed":  trace.Completed,
	}
	if snapshotErr != nil {
		eventData["error"] = snapshotErr.Error()
	}
	if eventStore != nil {
		eventData["event_store_enabled"] = true
		eventData["event_store_success"] = eventErr == nil
		eventData["event_entry_count"] = len(eventEntries)
		if eventBranch != "" {
			eventData["event_branch_id"] = eventBranch
		}
		if len(eventEntries) > 0 {
			eventData["event_head_entry_id"] = eventEntries[len(eventEntries)-1].ID
		}
		if eventErr != nil {
			eventData["event_store_error"] = eventErr.Error()
		}
	}
	a.emitEvent(agents.EventSessionPersisted, eventData)
}

func (a *Agent) persistSessionEventTrace(ctx context.Context, store sessionevent.SessionEventStore, trace *Trace, sessionID string) ([]sessionevent.SessionEntry, string, error) {
	if store == nil || trace == nil || strings.TrimSpace(sessionID) == "" {
		return nil, "", nil
	}

	branchID, err := ensureSessionEventBranch(ctx, store, sessionID, trace)
	if err != nil {
		return nil, "", err
	}

	entries := sessionEventEntriesFromTrace(trace, sessionID, branchID)
	if len(entries) == 0 {
		return nil, branchID, nil
	}

	inserted, err := store.AppendEntries(ctx, entries)
	if err != nil {
		return nil, branchID, err
	}
	return inserted, branchID, nil
}

func ensureSessionEventBranch(ctx context.Context, store sessionevent.SessionEventStore, sessionID string, trace *Trace) (string, error) {
	session, err := store.GetSession(ctx, sessionID)
	if err == nil {
		if branchID := strings.TrimSpace(session.ActiveBranchID); branchID != "" {
			return branchID, nil
		}
		branches, listErr := store.ListBranches(ctx, sessionID)
		if listErr != nil {
			return "", listErr
		}
		if len(branches) > 0 {
			branchID := branches[0].ID
			if setErr := store.SetActiveBranch(ctx, sessionID, branchID); setErr != nil {
				return "", setErr
			}
			return branchID, nil
		}
		return "", fmt.Errorf("session event store session %q has no active branch", sessionID)
	}
	if !isResourceNotFound(err) {
		return "", err
	}

	title := ""
	if trace != nil {
		title = strings.TrimSpace(trace.Task)
	}
	_, branch, createErr := store.CreateSession(ctx, sessionevent.CreateSessionParams{
		ID:    sessionID,
		Title: title,
	})
	if createErr == nil {
		return branch.ID, nil
	}
	session, getErr := store.GetSession(ctx, sessionID)
	if getErr == nil && strings.TrimSpace(session.ActiveBranchID) != "" {
		return session.ActiveBranchID, nil
	}
	if getErr != nil && !isResourceNotFound(getErr) {
		return "", fmt.Errorf("create session event branch recovery failed: %w", goerrors.Join(createErr, getErr))
	}
	return "", createErr
}

func sessionEventEntriesFromTrace(trace *Trace, sessionID, branchID string) []sessionevent.SessionEntry {
	if trace == nil || strings.TrimSpace(sessionID) == "" || strings.TrimSpace(branchID) == "" {
		return nil
	}

	entries := make([]sessionevent.SessionEntry, 0, len(trace.Steps)*3+3)
	baseTime := time.Now().UTC()
	offset := 0
	nextTime := func() time.Time {
		// The trace only captures run-level timing, not per-entry timestamps. Use a
		// monotonic nanosecond offset to preserve deterministic entry ordering.
		t := baseTime.Add(time.Duration(offset) * time.Nanosecond)
		offset++
		return t
	}

	appendEntry := func(entry sessionevent.SessionEntry) {
		entry.SessionID = sessionID
		entry.BranchID = branchID
		if entry.CreatedAt.IsZero() {
			entry.CreatedAt = nextTime()
		}
		entries = append(entries, entry)
	}

	appendEntry(sessionevent.SessionEntry{
		Kind:       sessionevent.EntryKindUserMessage,
		Role:       "user",
		CreatedAt:  nextTime(),
		SearchText: strings.TrimSpace(trace.Task),
		Payload: map[string]any{
			"text":    strings.TrimSpace(trace.Task),
			"task_id": strings.TrimSpace(trace.TaskID),
		},
		Metadata: map[string]any{
			"source":   "native",
			"task_id":  strings.TrimSpace(trace.TaskID),
			"provider": strings.TrimSpace(trace.Provider),
			"model":    strings.TrimSpace(trace.Model),
		},
	})

	lastAssistantText := ""
	for _, step := range trace.Steps {
		if text := strings.TrimSpace(step.AssistantText); text != "" {
			lastAssistantText = text
			appendEntry(sessionevent.SessionEntry{
				Kind:             sessionevent.EntryKindAssistantMessage,
				Role:             "assistant",
				CreatedAt:        nextTime(),
				SearchText:       text,
				PromptTokens:     step.Usage.PromptTokens,
				CompletionTokens: step.Usage.CompletionTokens,
				TotalTokens:      step.Usage.TotalTokens,
				Payload: map[string]any{
					"text":       text,
					"step_index": step.Index,
				},
				Metadata: map[string]any{
					"source":     "native",
					"step_index": step.Index,
				},
			})
		}

		if strings.TrimSpace(step.ToolName) == "" || strings.EqualFold(step.ToolName, "finish") {
			continue
		}

		appendEntry(sessionevent.SessionEntry{
			Kind:       sessionevent.EntryKindToolCall,
			Role:       "assistant",
			ToolName:   step.ToolName,
			CreatedAt:  nextTime(),
			SearchText: strings.TrimSpace(step.ToolName),
			Payload: map[string]any{
				"arguments":  core.ShallowCopyMap(step.Arguments),
				"step_index": step.Index,
			},
			Metadata: map[string]any{
				"source":     "native",
				"step_index": step.Index,
			},
		})

		appendEntry(sessionevent.SessionEntry{
			Kind:       sessionevent.EntryKindToolResult,
			ToolName:   step.ToolName,
			CreatedAt:  nextTime(),
			IsError:    step.IsError,
			Synthetic:  step.Synthetic,
			Redacted:   step.Redacted,
			Truncated:  step.Truncated,
			SearchText: firstNonEmpty(strings.TrimSpace(step.ObservationDisplay), strings.TrimSpace(step.Observation)),
			Payload: map[string]any{
				"observation":         strings.TrimSpace(step.Observation),
				"observation_display": strings.TrimSpace(step.ObservationDisplay),
				"details":             core.ShallowCopyMap(step.ObservationDetails),
				"step_index":          step.Index,
			},
			Metadata: map[string]any{
				"source":     "native",
				"step_index": step.Index,
			},
		})
	}

	if finalAnswer := strings.TrimSpace(trace.FinalAnswer); finalAnswer != "" && finalAnswer != lastAssistantText {
		appendEntry(sessionevent.SessionEntry{
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			CreatedAt:  nextTime(),
			SearchText: finalAnswer,
			Payload: map[string]any{
				"text":  finalAnswer,
				"final": true,
			},
			Metadata: map[string]any{
				"source": "native",
			},
		})
	}

	appendEntry(sessionevent.SessionEntry{
		Kind:             sessionevent.EntryKindSystemEvent,
		CreatedAt:        nextTime(),
		SearchText:       firstNonEmpty(strings.TrimSpace(trace.FinalAnswer), strings.TrimSpace(trace.Error), strings.TrimSpace(trace.Task)),
		PromptTokens:     trace.TokenUsage.PromptTokens,
		CompletionTokens: trace.TokenUsage.CompletionTokens,
		TotalTokens:      trace.TokenUsage.TotalTokens,
		Payload: map[string]any{
			"event":             "run_finished",
			"task_id":           strings.TrimSpace(trace.TaskID),
			"completed":         trace.Completed,
			"final_answer":      strings.TrimSpace(trace.FinalAnswer),
			"error":             strings.TrimSpace(trace.Error),
			"provider":          strings.TrimSpace(trace.Provider),
			"model":             strings.TrimSpace(trace.Model),
			"started_at":        trace.StartedAt.UTC().Format(time.RFC3339Nano),
			"duration_ms":       trace.Duration.Milliseconds(),
			"prompt_tokens":     trace.TokenUsage.PromptTokens,
			"completion_tokens": trace.TokenUsage.CompletionTokens,
			"total_tokens":      trace.TokenUsage.TotalTokens,
		},
		Metadata: map[string]any{
			"source": "native",
		},
	})

	return entries
}

func isResourceNotFound(err error) bool {
	var dspyErr *dspyerrors.Error
	return goerrors.As(err, &dspyErr) && dspyErr.Code() == dspyerrors.ResourceNotFound
}

func buildSessionRecall(records []agents.SessionRecord, maxChars int) string {
	if len(records) == 0 {
		return ""
	}
	if maxChars <= 0 {
		maxChars = defaultSessionRecallMaxChars
	}

	const footer = "Use this prior session context when it is relevant. Avoid repeating already completed work unless the new task requires it."

	var builder strings.Builder
	header := "Recent session runs:\n"
	builder.WriteString(header)
	currentLen := len([]rune(header))
	footerLen := len([]rune(footer))

	for i, record := range records {
		recordText := renderSessionRecallRecord(i+1, record)
		recordLen := len([]rune(recordText))
		if currentLen+recordLen+footerLen > maxChars {
			break
		}
		builder.WriteString(recordText)
		currentLen += recordLen
	}

	if currentLen+footerLen <= maxChars {
		builder.WriteString(footer)
		return builder.String()
	}
	return agentutil.TruncateString(builder.String(), maxChars)
}

func sessionRecordFromTrace(a *Agent, input map[string]any, trace *Trace, sessionID string) agents.SessionRecord {
	if trace == nil {
		return agents.SessionRecord{}
	}

	completedAt := trace.StartedAt.Add(trace.Duration)
	return agents.SessionRecord{
		ID:          fmt.Sprintf("%s-%d", firstNonEmpty(trace.TaskID, "run"), trace.StartedAt.UTC().UnixNano()),
		SessionID:   sessionID,
		AgentID:     fmt.Sprintf("native-%s-%s", a.llm.ProviderName(), a.llm.ModelID()),
		AgentType:   "native",
		TaskID:      trace.TaskID,
		Task:        trace.Task,
		StartedAt:   trace.StartedAt.UTC(),
		CompletedAt: completedAt.UTC(),
		Completed:   trace.Completed,
		FinalAnswer: strings.TrimSpace(trace.FinalAnswer),
		Error:       strings.TrimSpace(trace.Error),
		Highlights:  traceHighlights(trace.Steps),
		Metadata: map[string]any{
			"provider":          trace.Provider,
			"model":             trace.Model,
			"turns":             len(trace.Steps),
			"tool_calls":        countExecutedTools(trace.Steps),
			"termination_cause": traceTerminationCause(trace),
			"task_id":           firstNonEmpty(trace.TaskID, agentutil.StringValue(input["task_id"])),
			"prompt_tokens":     trace.TokenUsage.PromptTokens,
			"completion_tokens": trace.TokenUsage.CompletionTokens,
			"total_tokens":      trace.TokenUsage.TotalTokens,
		},
	}
}

func traceHighlights(steps []TraceStep) []string {
	highlights := make([]string, 0, maxSessionHighlights)
	for _, step := range steps {
		if len(highlights) >= maxSessionHighlights {
			break
		}
		if strings.EqualFold(step.ToolName, "finish") {
			continue
		}
		text := strings.TrimSpace(step.Observation)
		if text == "" {
			continue
		}

		prefix := strings.TrimSpace(step.ToolName)
		switch {
		case prefix != "" && step.IsError:
			text = fmt.Sprintf("%s error: %s", prefix, text)
		case prefix != "":
			text = fmt.Sprintf("%s: %s", prefix, text)
		case step.IsError:
			text = "error: " + text
		}
		text = agentutil.TruncateString(text, maxSessionHighlightChars)
		if text == "" || containsHighlight(highlights, text) {
			continue
		}
		highlights = append(highlights, text)
	}
	return highlights
}

func containsHighlight(highlights []string, want string) bool {
	for _, highlight := range highlights {
		if highlight == want {
			return true
		}
	}
	return false
}

func renderSessionRecallRecord(index int, record agents.SessionRecord) string {
	var builder strings.Builder
	fmt.Fprintf(&builder, "%d. Task: %s\n", index, agentutil.TruncateString(strings.TrimSpace(record.Task), maxSessionTaskChars))
	if !record.StartedAt.IsZero() {
		fmt.Fprintf(&builder, "   Started: %s\n", record.StartedAt.UTC().Format(time.RFC3339))
	}
	if record.Completed {
		builder.WriteString("   Outcome: completed\n")
	} else {
		builder.WriteString("   Outcome: incomplete\n")
	}
	if answer := strings.TrimSpace(record.FinalAnswer); answer != "" {
		fmt.Fprintf(&builder, "   Final answer: %s\n", agentutil.TruncateString(answer, maxSessionAnswerChars))
	}
	if errText := strings.TrimSpace(record.Error); errText != "" {
		fmt.Fprintf(&builder, "   Error: %s\n", agentutil.TruncateString(errText, maxSessionAnswerChars))
	}
	for _, highlight := range record.Highlights {
		fmt.Fprintf(&builder, "   - %s\n", agentutil.TruncateString(strings.TrimSpace(highlight), maxSessionHighlightChars))
	}
	return builder.String()
}
