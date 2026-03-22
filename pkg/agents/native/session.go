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
	defaultSessionRecallLimit      = 3
	defaultSessionRecallMaxChars   = 1200
	defaultSessionEventEntryFactor = 4
	maxSessionHighlights           = 3
	maxSessionHighlightChars       = 160
	maxSessionTaskChars            = 160
	maxSessionAnswerChars          = 220
)

type sessionContext struct {
	Recall            string
	Source            string
	RecordCount       int
	EntryCount        int
	SummaryCount      int
	BranchID          string
	HeadEntryID       string
	ForkedFromEntryID string
}

type sessionEventBranchState struct {
	BranchID          string
	HeadEntryID       string
	ForkedFromEntryID string
}

func (a *Agent) sessionID(input map[string]any) string {
	if id := strings.TrimSpace(agentutil.StringValue(input["session_id"])); id != "" {
		return id
	}
	return strings.TrimSpace(a.config.SessionID)
}

func (a *Agent) sessionBranchID(input map[string]any) string {
	if id := strings.TrimSpace(agentutil.StringValue(input["session_branch_id"])); id != "" {
		return id
	}
	return strings.TrimSpace(a.config.SessionBranchID)
}

func (a *Agent) sessionBranchName(input map[string]any) string {
	return strings.TrimSpace(agentutil.StringValue(input["session_branch_name"]))
}

func (a *Agent) sessionForkFromEntryID(input map[string]any) string {
	return strings.TrimSpace(agentutil.StringValue(input["session_fork_from_entry_id"]))
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

func (a *Agent) loadSessionContext(ctx context.Context, input map[string]any) (sessionContext, error) {
	sessionID := a.sessionID(input)
	if sessionID == "" {
		return sessionContext{}, nil
	}

	if eventStore := a.sessionEventStore(); eventStore != nil {
		loadedContext, err := a.loadSessionEventContext(ctx, input, eventStore, sessionID)
		if err == nil {
			return loadedContext, nil
		}

		explicitBranchRequest := a.sessionBranchID(input) != "" || a.sessionForkFromEntryID(input) != ""
		if explicitBranchRequest || !isResourceNotFound(err) {
			return sessionContext{}, err
		}
	}

	records, err := a.sessionStore().Recent(sessionID, a.config.SessionRecallLimit)
	if err != nil {
		return sessionContext{}, err
	}
	return sessionContext{
		Recall:      buildSessionRecall(records, a.config.SessionRecallMaxChars),
		Source:      "snapshot",
		RecordCount: len(records),
	}, nil
}

func (a *Agent) loadSessionEventContext(ctx context.Context, input map[string]any, store sessionevent.SessionEventStore, sessionID string) (sessionContext, error) {
	branchState, err := a.resolveSessionEventBranch(ctx, input, store, sessionID)
	if err != nil {
		return sessionContext{}, err
	}

	result := sessionContext{
		Source:            "event_store",
		BranchID:          branchState.BranchID,
		HeadEntryID:       branchState.HeadEntryID,
		ForkedFromEntryID: branchState.ForkedFromEntryID,
	}
	if strings.TrimSpace(branchState.HeadEntryID) == "" {
		return result, nil
	}

	lineageOpts := sessionevent.LoadOptions{
		MaxEntries: maxSessionEventEntries(a.config.SessionRecallLimit),
	}
	entries, err := store.LoadLineage(ctx, sessionID, branchState.HeadEntryID, lineageOpts)
	if err != nil {
		return sessionContext{}, err
	}

	summaries, err := store.LoadSummaries(ctx, sessionID, branchState.BranchID, maxSessionEventSummaries(a.config.SessionRecallLimit))
	if err != nil {
		return sessionContext{}, err
	}

	result.EntryCount = len(entries)
	result.SummaryCount = len(summaries)
	result.Recall = buildSessionEventRecall(branchState.BranchID, summaries, entries, a.config.SessionRecallMaxChars)
	return result, nil
}

func (a *Agent) resolveSessionEventBranch(ctx context.Context, input map[string]any, store sessionevent.SessionEventStore, sessionID string) (sessionEventBranchState, error) {
	if forkFromID := a.sessionForkFromEntryID(input); forkFromID != "" {
		forked, err := store.ForkBranch(ctx, sessionID, forkFromID, a.sessionBranchName(input), nil)
		if err != nil {
			return sessionEventBranchState{}, err
		}
		if err := store.SetActiveBranch(ctx, sessionID, forked.ID); err != nil {
			return sessionEventBranchState{}, err
		}
		return sessionEventBranchState{
			BranchID:          forked.ID,
			HeadEntryID:       strings.TrimSpace(forked.HeadEntryID),
			ForkedFromEntryID: forkFromID,
		}, nil
	}

	if branchID := a.sessionBranchID(input); branchID != "" {
		head, err := switchSessionEventBranch(ctx, store, sessionID, branchID)
		if err != nil {
			return sessionEventBranchState{}, err
		}
		return sessionEventBranchState{
			BranchID:    branchID,
			HeadEntryID: sessionHeadEntryID(head),
		}, nil
	}

	session, err := store.GetSession(ctx, sessionID)
	if err != nil {
		return sessionEventBranchState{}, err
	}

	branchID := strings.TrimSpace(session.ActiveBranchID)
	if branchID == "" {
		branches, err := store.ListBranches(ctx, sessionID)
		if err != nil {
			return sessionEventBranchState{}, err
		}
		if len(branches) == 0 {
			return sessionEventBranchState{}, fmt.Errorf("session event store session %q has no branches", sessionID)
		}
		branchID = branches[0].ID
		if err := store.SetActiveBranch(ctx, sessionID, branchID); err != nil {
			return sessionEventBranchState{}, err
		}
	}

	head, err := store.GetBranchHead(ctx, sessionID, branchID)
	if err != nil {
		return sessionEventBranchState{}, err
	}
	return sessionEventBranchState{
		BranchID:    branchID,
		HeadEntryID: sessionHeadEntryID(head),
	}, nil
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

func sessionHeadEntryID(head *sessionevent.SessionEntry) string {
	if head == nil {
		return ""
	}
	return strings.TrimSpace(head.ID)
}

func maxSessionEventEntries(limit int) int {
	if limit <= 0 {
		limit = defaultSessionRecallLimit
	}
	return limit * defaultSessionEventEntryFactor
}

func maxSessionEventSummaries(limit int) int {
	if limit <= 0 {
		return defaultSessionRecallLimit
	}
	return limit
}

func buildSessionEventRecall(branchID string, summaries []sessionevent.SessionSummary, entries []sessionevent.SessionEntry, maxChars int) string {
	if len(summaries) == 0 && len(entries) == 0 {
		return ""
	}
	if maxChars <= 0 {
		maxChars = defaultSessionRecallMaxChars
	}

	const footer = "Use this prior session context when it is relevant. Avoid repeating already completed work unless the new task requires it."

	var builder strings.Builder
	header := "Recent session branch context:\n"
	builder.WriteString(header)
	currentLen := len([]rune(header))
	footerLen := len([]rune(footer))

	if branchID = strings.TrimSpace(branchID); branchID != "" {
		branchLine := fmt.Sprintf("Active branch: %s\n", branchID)
		if currentLen+len([]rune(branchLine))+footerLen <= maxChars {
			builder.WriteString(branchLine)
			currentLen += len([]rune(branchLine))
		}
	}

	if len(summaries) > 0 {
		// Until sessionevent supports summary-aware lineage elision, summaries and raw
		// entries may overlap in recall. Keep both for now so resume stays lossless.
		summaryHeader := "Branch summaries:\n"
		if currentLen+len([]rune(summaryHeader))+footerLen <= maxChars {
			builder.WriteString(summaryHeader)
			currentLen += len([]rune(summaryHeader))
			for i := len(summaries) - 1; i >= 0; i-- {
				summaryText := renderSessionEventSummary(summaries[i])
				summaryLen := len([]rune(summaryText))
				if currentLen+summaryLen+footerLen > maxChars {
					break
				}
				builder.WriteString(summaryText)
				currentLen += summaryLen
			}
		}
	}

	if len(entries) > 0 {
		lineageHeader := "Branch lineage:\n"
		if currentLen+len([]rune(lineageHeader))+footerLen <= maxChars {
			builder.WriteString(lineageHeader)
			currentLen += len([]rune(lineageHeader))
			for i, entry := range entries {
				entryText := renderSessionEventRecallEntry(i+1, entry)
				entryLen := len([]rune(entryText))
				if currentLen+entryLen+footerLen > maxChars {
					break
				}
				builder.WriteString(entryText)
				currentLen += entryLen
			}
		}
	}

	if currentLen+footerLen <= maxChars {
		builder.WriteString(footer)
		return builder.String()
	}
	return agentutil.TruncateString(builder.String(), maxChars)
}

func renderSessionEventSummary(summary sessionevent.SessionSummary) string {
	return fmt.Sprintf("- %s\n", agentutil.TruncateString(strings.TrimSpace(summary.SummaryText), maxSessionAnswerChars))
}

func renderSessionEventRecallEntry(index int, entry sessionevent.SessionEntry) string {
	switch entry.Kind {
	case sessionevent.EntryKindUserMessage:
		return fmt.Sprintf("%d. User: %s\n", index, agentutil.TruncateString(strings.TrimSpace(agentutil.StringValue(entry.Payload["text"])), maxSessionTaskChars))
	case sessionevent.EntryKindAssistantMessage:
		return fmt.Sprintf("%d. Assistant: %s\n", index, agentutil.TruncateString(strings.TrimSpace(agentutil.StringValue(entry.Payload["text"])), maxSessionAnswerChars))
	case sessionevent.EntryKindToolCall:
		return fmt.Sprintf("%d. Tool call %s\n", index, agentutil.TruncateString(strings.TrimSpace(entry.ToolName), maxSessionTaskChars))
	case sessionevent.EntryKindToolResult:
		prefix := fmt.Sprintf("%d. Tool result %s", index, agentutil.TruncateString(strings.TrimSpace(entry.ToolName), maxSessionTaskChars))
		body := agentutil.StringValue(entry.Payload["observation_display"])
		if strings.TrimSpace(body) == "" {
			body = agentutil.StringValue(entry.Payload["observation"])
		}
		if entry.IsError {
			return fmt.Sprintf("%s (error): %s\n", prefix, agentutil.TruncateString(strings.TrimSpace(body), maxSessionAnswerChars))
		}
		return fmt.Sprintf("%s: %s\n", prefix, agentutil.TruncateString(strings.TrimSpace(body), maxSessionAnswerChars))
	case sessionevent.EntryKindSystemEvent:
		if event := strings.TrimSpace(agentutil.StringValue(entry.Payload["event"])); event == "run_finished" {
			if finalAnswer := strings.TrimSpace(agentutil.StringValue(entry.Payload["final_answer"])); finalAnswer != "" {
				return fmt.Sprintf("%d. Final result: %s\n", index, agentutil.TruncateString(finalAnswer, maxSessionAnswerChars))
			}
			if errText := strings.TrimSpace(agentutil.StringValue(entry.Payload["error"])); errText != "" {
				return fmt.Sprintf("%d. Run error: %s\n", index, agentutil.TruncateString(errText, maxSessionAnswerChars))
			}
		}
		return fmt.Sprintf("%d. System event: %s\n", index, agentutil.TruncateString(strings.TrimSpace(agentutil.StringValue(entry.Payload["event"])), maxSessionAnswerChars))
	default:
		return fmt.Sprintf("%d. %s\n", index, agentutil.TruncateString(strings.TrimSpace(entry.SearchText), maxSessionAnswerChars))
	}
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
