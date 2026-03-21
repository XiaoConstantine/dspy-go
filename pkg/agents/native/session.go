package native

import (
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
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

func (a *Agent) persistSessionRecord(input map[string]any, trace *Trace) {
	sessionID := a.sessionID(input)
	if sessionID == "" || trace == nil {
		return
	}

	record := sessionRecordFromTrace(a, input, trace, sessionID)
	if err := a.sessionStore().Append(record); err != nil {
		a.emitEvent(agents.EventSessionPersisted, map[string]any{
			"session_id": sessionID,
			"record_id":  record.ID,
			"task_id":    trace.TaskID,
			"success":    false,
			"error":      err.Error(),
		})
		return
	}

	a.emitEvent(agents.EventSessionPersisted, map[string]any{
		"session_id": sessionID,
		"record_id":  record.ID,
		"task_id":    trace.TaskID,
		"success":    true,
		"completed":  trace.Completed,
	})
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
