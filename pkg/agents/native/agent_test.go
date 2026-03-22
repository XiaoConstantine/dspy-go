package native

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewAgent_RequiresToolCalling(t *testing.T) {
	_, err := NewAgent(&stubLLM{}, Config{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not support tool calling")
}

func TestAgent_Execute_CompletesWithToolAndFinish(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Write a note and finish.",
		"task_id": "task-1",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Equal(t, "done", result["final_answer"])

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.True(t, trace.Completed)
	assert.Equal(t, "done", trace.FinalAnswer)
	assert.Len(t, trace.Steps, 2)
	assert.Equal(t, "write_note", trace.Steps[0].ToolName)
	assert.Equal(t, int64(9), trace.TokenUsage.PromptTokens)
	assert.Equal(t, int64(5), trace.TokenUsage.CompletionTokens)
}

func TestAgent_Execute_UsesModelTextInTranscriptAndDisplayTextInTrace(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "done"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
			return core.ToolResult{
				Data: "raw output",
				Metadata: map[string]any{
					core.ToolResultModelTextMeta:   "summary for model",
					core.ToolResultDisplayTextMeta: "full output for operator",
				},
				Annotations: map[string]any{
					core.ToolResultDetailsAnnotation: map[string]any{"content": "done"},
				},
			}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task": "Write a note and finish.",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], "summary for model")
	assert.NotContains(t, llm.prompts[1], "full output for operator")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.Equal(t, "summary for model", trace.Steps[0].Observation)
	assert.Equal(t, "full output for operator", trace.Steps[0].ObservationDisplay)
	assert.Equal(t, map[string]any{"content": "done"}, trace.Steps[0].ObservationDetails)

	execTrace := agent.LastExecutionTrace()
	require.NotNil(t, execTrace)
	require.Len(t, execTrace.Steps, 2)
	assert.Equal(t, "summary for model", execTrace.Steps[0].Observation)
	assert.Equal(t, "full output for operator", execTrace.Steps[0].ObservationDisplay)
	assert.Equal(t, map[string]any{"content": "done"}, execTrace.Steps[0].ObservationDetails)
}

func TestAgent_Execute_FailsFastAfterRepeatedNoCallResponses(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			noCallResult(),
			noCallResult(),
			noCallResult(),
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 20})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Use a tool before finishing.",
		"task_id": "task-no-call",
	})
	require.NoError(t, err)
	require.False(t, result["completed"].(bool))
	assert.Contains(t, result["error"], "repeated model responses without tool calls")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.Len(t, trace.Steps, 3)
	assert.Equal(t, "empty_content_and_function_call", trace.Steps[0].Metadata["reason"])
}

func TestAgent_Execute_ProcessesMultipleToolCallsInOneResponse(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"tool_calls": []core.ToolCall{
					{
						Name:      "write_note",
						Arguments: map[string]any{"content": "first"},
					},
					{
						Name:      "write_note",
						Arguments: map[string]any{"content": "second"},
					},
				},
				"_usage": &core.TokenInfo{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)

	observed := make([]string, 0, 2)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
			observed = append(observed, params["content"].(string))
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Write two notes and finish.",
		"task_id": "task-multi-call",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Equal(t, []string{"first", "second"}, observed)

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 3)
	assert.Equal(t, "write_note", trace.Steps[0].ToolName)
	assert.Equal(t, "write_note", trace.Steps[1].ToolName)
	assert.Equal(t, "Finish", trace.Steps[2].ToolName)
}

func TestAgent_Execute_UsesNativeToolCallingWhenAvailable(t *testing.T) {
	llm := &nativeStubLLM{
		stubLLM: stubLLM{
			capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
			results: []map[string]any{
				{
					"function_call": map[string]any{
						"name":      "Finish",
						"arguments": map[string]any{"answer": "done"},
					},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{"task": "Finish immediately."})
	require.NoError(t, err)
	assert.Len(t, llm.messages, 1)
	assert.Empty(t, llm.prompts)
}

func TestAgent_Execute_EmitsEventsAndHandlesBlockedTools(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "blocked"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 4,
		ToolInterceptors: []core.ToolInterceptor{
			func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
				return core.ToolResult{}, &core.ToolBlockedError{Reason: "approval denied"}
			},
		},
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Write a note and finish.",
		"task_id": "task-blocked",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], `Tool "write_note" blocked: approval denied`)

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.True(t, trace.Steps[0].Synthetic)
	assert.True(t, trace.Steps[0].IsError)
	assert.Contains(t, trace.Steps[0].Observation, "approval denied")

	eventTypes := make([]string, 0, len(events))
	for _, event := range events {
		eventTypes = append(eventTypes, event.Type)
	}
	assert.Contains(t, eventTypes, agents.EventRunStarted)
	assert.Contains(t, eventTypes, agents.EventLLMTurnStarted)
	assert.Contains(t, eventTypes, agents.EventLLMTurnFinished)
	assert.Contains(t, eventTypes, agents.EventToolCallProposed)
	assert.Contains(t, eventTypes, agents.EventToolCallStarted)
	assert.Contains(t, eventTypes, agents.EventToolCallBlocked)
	assert.Contains(t, eventTypes, agents.EventToolCallFinished)
	assert.Contains(t, eventTypes, agents.EventRunFinished)
}

func TestAgent_Execute_HandlesGenericInterceptorError(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "write_note",
					"arguments": map[string]any{"content": "value"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{
		MaxTurns: 4,
		ToolInterceptors: []core.ToolInterceptor{
			func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
				return core.ToolResult{}, fmt.Errorf("approval backend unavailable")
			},
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task": "Write a note and finish.",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], "tool execution failed: approval backend unavailable")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 2)
	assert.False(t, trace.Steps[0].Synthetic)
	assert.True(t, trace.Steps[0].IsError)
	assert.Contains(t, trace.Steps[0].Observation, "approval backend unavailable")
	assert.Contains(t, trace.Steps[0].ObservationDisplay, "approval backend unavailable")

	execTrace := agent.LastExecutionTrace()
	require.NotNil(t, execTrace)
	require.Len(t, execTrace.Steps, 2)
	assert.Contains(t, execTrace.Steps[0].Observation, "approval backend unavailable")
	assert.Contains(t, execTrace.Steps[0].ObservationDisplay, "approval backend unavailable")
}

func TestAgent_Execute_EmitsLLMTurnFinishedOnGenerateError(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task":    "this will fail",
		"task_id": "task-generate-error",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no more stubbed results")

	eventTypes := make([]string, 0, len(events))
	var llmFinished *agents.AgentEvent
	for i := range events {
		eventTypes = append(eventTypes, events[i].Type)
		if events[i].Type == agents.EventLLMTurnFinished {
			llmFinished = &events[i]
		}
	}

	assert.Contains(t, eventTypes, agents.EventLLMTurnStarted)
	assert.Contains(t, eventTypes, agents.EventLLMTurnFinished)
	assert.Contains(t, eventTypes, agents.EventRunFailed)
	assert.Contains(t, eventTypes, agents.EventRunFinished)
	require.NotNil(t, llmFinished)
	assert.Equal(t, 0, llmFinished.Data["tool_calls"])
	assert.Equal(t, int64(0), llmFinished.Data["usage_total"])
	assert.Contains(t, llmFinished.Data["error"], "no more stubbed results")
}

func TestAgent_Execute_PersistsAndReloadsSessionRecall(t *testing.T) {
	memory := agents.NewInMemoryStore()

	firstLLM := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "first answer"},
				},
			},
		},
	}
	firstAgent, err := NewAgent(firstLLM, Config{
		MaxTurns:  1,
		Memory:    memory,
		SessionID: "session-1",
	})
	require.NoError(t, err)

	_, err = firstAgent.Execute(context.Background(), map[string]interface{}{
		"task":    "First task",
		"task_id": "task-1",
	})
	require.NoError(t, err)

	secondLLM := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "second answer"},
				},
			},
		},
	}
	secondAgent, err := NewAgent(secondLLM, Config{
		MaxTurns:  1,
		Memory:    memory,
		SessionID: "session-1",
	})
	require.NoError(t, err)

	_, err = secondAgent.Execute(context.Background(), map[string]interface{}{
		"task":    "Second task",
		"task_id": "task-2",
	})
	require.NoError(t, err)

	require.Len(t, secondLLM.prompts, 1)
	assert.Contains(t, secondLLM.prompts[0], "SESSION RECALL:")
	assert.Contains(t, secondLLM.prompts[0], "First task")
	assert.Contains(t, secondLLM.prompts[0], "first answer")

	records, err := agents.NewSessionStore(memory).Recent("session-1", 10)
	require.NoError(t, err)
	require.Len(t, records, 2)
	assert.Equal(t, "First task", records[0].Task)
	assert.Equal(t, "Second task", records[1].Task)
}

func TestAgent_Execute_EmitsSessionEvents(t *testing.T) {
	memory := agents.NewInMemoryStore()
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:  1,
		Memory:    memory,
		SessionID: "session-events",
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Eventful task",
		"task_id": "event-task",
	})
	require.NoError(t, err)

	var sessionLoaded *agents.AgentEvent
	var sessionPersisted *agents.AgentEvent
	for i := range events {
		switch events[i].Type {
		case agents.EventSessionLoaded:
			sessionLoaded = &events[i]
		case agents.EventSessionPersisted:
			sessionPersisted = &events[i]
		}
	}

	require.NotNil(t, sessionLoaded)
	assert.Equal(t, "session-events", sessionLoaded.Data["session_id"])
	assert.Equal(t, 0, sessionLoaded.Data["record_count"])
	assert.Equal(t, 0, sessionLoaded.Data["recall_chars"])

	require.NotNil(t, sessionPersisted)
	assert.Equal(t, "session-events", sessionPersisted.Data["session_id"])
	assert.Equal(t, true, sessionPersisted.Data["success"])
	assert.Equal(t, true, sessionPersisted.Data["completed"])
}

func TestAgent_Execute_DualWritesSessionEventStore(t *testing.T) {
	memory := agents.NewInMemoryStore()
	eventStore, err := sessionevent.NewSQLiteStore(filepath.Join(t.TempDir(), "session-events.db"))
	require.NoError(t, err)
	defer eventStore.Close()

	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
				"_usage": &core.TokenInfo{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		Memory:            memory,
		SessionID:         "session-dual-write",
		SessionEventStore: eventStore,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Persist to both stores",
		"task_id": "dual-write-task",
	})
	require.NoError(t, err)
	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.Equal(t, int64(11), trace.TokenUsage.PromptTokens)
	assert.Equal(t, int64(7), trace.TokenUsage.CompletionTokens)
	assert.Equal(t, int64(18), trace.TokenUsage.TotalTokens)

	records, err := agents.NewSessionStore(memory).Recent("session-dual-write", 10)
	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "Persist to both stores", records[0].Task)

	session, err := eventStore.GetSession(context.Background(), "session-dual-write")
	require.NoError(t, err)
	require.NotEmpty(t, session.ActiveBranchID)

	head, err := eventStore.GetBranchHead(context.Background(), session.ID, session.ActiveBranchID)
	require.NoError(t, err)
	require.NotNil(t, head)

	derivedEntries := sessionEventEntriesFromTrace(trace, session.ID, session.ActiveBranchID)
	require.Len(t, derivedEntries, 3)
	assert.Equal(t, int64(18), derivedEntries[2].TotalTokens)

	lineage, err := eventStore.LoadLineage(context.Background(), session.ID, head.ID, sessionevent.LoadOptions{})
	require.NoError(t, err)
	require.Len(t, lineage, 3)
	assert.Equal(t, sessionevent.EntryKindUserMessage, lineage[0].Kind)
	assert.Equal(t, sessionevent.EntryKindAssistantMessage, lineage[1].Kind)
	assert.Equal(t, sessionevent.EntryKindSystemEvent, lineage[2].Kind)
	assert.Equal(t, "Persist to both stores", lineage[0].Payload["text"])
	assert.Equal(t, "done", lineage[1].Payload["text"])
	assert.Equal(t, "run_finished", lineage[2].Payload["event"])
	assert.Equal(t, int64(11), lineage[2].PromptTokens)
	assert.Equal(t, int64(7), lineage[2].CompletionTokens)
	assert.Equal(t, int64(18), lineage[2].TotalTokens)

	var persisted *agents.AgentEvent
	for i := range events {
		if events[i].Type == agents.EventSessionPersisted {
			persisted = &events[i]
		}
	}
	require.NotNil(t, persisted)
	assert.Equal(t, true, persisted.Data["success"])
	assert.Equal(t, true, persisted.Data["event_store_success"])
	assert.Equal(t, 3, persisted.Data["event_entry_count"])
	assert.Equal(t, session.ActiveBranchID, persisted.Data["event_branch_id"])
}

func TestAgent_Execute_ReportsSessionEventStoreFailureWithoutBreakingSnapshotPersistence(t *testing.T) {
	memory := agents.NewInMemoryStore()
	eventStore, err := sessionevent.NewSQLiteStore(filepath.Join(t.TempDir(), "session-events.db"))
	require.NoError(t, err)
	require.NoError(t, eventStore.Close())

	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	var events []agents.AgentEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		Memory:            memory,
		SessionID:         "session-event-failure",
		SessionEventStore: eventStore,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Event store should fail only for dual write",
		"task_id": "dual-write-failure",
	})
	require.NoError(t, err)

	records, err := agents.NewSessionStore(memory).Recent("session-event-failure", 10)
	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "Event store should fail only for dual write", records[0].Task)

	var persisted *agents.AgentEvent
	for i := range events {
		if events[i].Type == agents.EventSessionPersisted {
			persisted = &events[i]
		}
	}
	require.NotNil(t, persisted)
	assert.Equal(t, true, persisted.Data["success"])
	assert.Equal(t, true, persisted.Data["event_store_enabled"])
	assert.Equal(t, false, persisted.Data["event_store_success"])
	assert.Contains(t, persisted.Data["event_store_error"], "closed")
}

func TestSessionEventEntriesFromTrace_DeduplicatesFinalAnswerAssistantEntry(t *testing.T) {
	trace := &Trace{
		TaskID:      "task-final-answer",
		Task:        "Summarize and finish",
		Provider:    "stub",
		Model:       "stub-model",
		StartedAt:   time.Date(2026, time.March, 21, 12, 0, 0, 0, time.UTC),
		Completed:   true,
		FinalAnswer: "done",
		TokenUsage:  TokenUsage{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13},
		Steps: []TraceStep{
			{
				Index:         1,
				AssistantText: "done",
				ToolName:      "Finish",
			},
		},
	}

	entries := sessionEventEntriesFromTrace(trace, "session-1", "branch-1")
	require.Len(t, entries, 3)
	assert.Equal(t, sessionevent.EntryKindUserMessage, entries[0].Kind)
	assert.Equal(t, sessionevent.EntryKindAssistantMessage, entries[1].Kind)
	assert.Equal(t, "done", entries[1].Payload["text"])
	assert.Equal(t, sessionevent.EntryKindSystemEvent, entries[2].Kind)
	assert.Equal(t, "done", entries[2].Payload["final_answer"])
}

func TestEnsureSessionEventBranch_JoinsCreateAndRecoveryErrors(t *testing.T) {
	createErr := errors.New("create failed")
	recoveryErr := errors.New("recovery lookup failed")
	store := &stubSessionEventStore{
		getSessionErrs: []error{
			dspyerrors.New(dspyerrors.ResourceNotFound, "missing"),
			recoveryErr,
		},
		createSessionErr: createErr,
	}

	_, err := ensureSessionEventBranch(context.Background(), store, "session-1", &Trace{Task: "task"})
	require.Error(t, err)
	assert.ErrorIs(t, err, createErr)
	assert.ErrorIs(t, err, recoveryErr)
	assert.Contains(t, err.Error(), "create session event branch recovery failed")
}

func TestAgent_Execute_PersistsFailedRunsToSessionStore(t *testing.T) {
	memory := agents.NewInMemoryStore()
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			noCallResult(),
			noCallResult(),
			noCallResult(),
		},
	}

	agent, err := NewAgent(llm, Config{
		MaxTurns:  20,
		Memory:    memory,
		SessionID: "session-failure",
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Fail without tool calls",
		"task_id": "task-failure",
	})
	require.NoError(t, err)
	require.False(t, result["completed"].(bool))

	records, err := agents.NewSessionStore(memory).Recent("session-failure", 10)
	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.False(t, records[0].Completed)
	assert.Equal(t, "Fail without tool calls", records[0].Task)
	assert.Contains(t, records[0].Error, "repeated model responses without tool calls")
}

func TestBuildSessionRecall_StopsBeforePartialRecord(t *testing.T) {
	records := []agents.SessionRecord{
		{
			Task:        "First task",
			StartedAt:   time.Date(2026, time.March, 21, 10, 0, 0, 0, time.UTC),
			Completed:   true,
			FinalAnswer: "first answer",
		},
		{
			Task:        "Second task that should not be partially included",
			StartedAt:   time.Date(2026, time.March, 21, 10, 5, 0, 0, time.UTC),
			Completed:   true,
			FinalAnswer: "second answer",
		},
	}

	fullFirst := renderSessionRecallRecord(1, records[0])
	maxChars := len([]rune("Recent session runs:\n")) + len([]rune(fullFirst)) + len([]rune("Use this prior session context when it is relevant. Avoid repeating already completed work unless the new task requires it."))
	recall := buildSessionRecall(records, maxChars)

	assert.Contains(t, recall, "First task")
	assert.NotContains(t, recall, "Second task")
	assert.NotContains(t, recall, "...")
}

func TestAgent_Execute_FallsBackToFunctionsForWrappedFunctionOnlyLLM(t *testing.T) {
	base := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "done"},
				},
			},
		},
	}

	agent, err := NewAgent(core.NewModelContextDecorator(base), Config{MaxTurns: 1})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Len(t, base.prompts, 1)
	assert.Contains(t, base.prompts[0], "TURN BUDGET: turn 1 of 1")
	assert.Contains(t, base.prompts[0], "call Finish immediately")
	assert.NotContains(t, base.prompts[0], "SESSION RECALL:")
}

func TestAgent_Clone_CopiesCloneableTools(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)
	tool := &simpleTool{name: "write_note"}
	require.NoError(t, agent.RegisterTool(tool))

	clonedAny, err := agent.Clone()
	require.NoError(t, err)

	cloned, ok := clonedAny.(*Agent)
	require.True(t, ok)

	originalTools := agent.toolRegistry.List()
	clonedTools := cloned.toolRegistry.List()
	require.Len(t, originalTools, 1)
	require.Len(t, clonedTools, 1)
	assert.NotSame(t, originalTools[0], clonedTools[0])
	assert.NotSame(t, agent.memory, cloned.memory)
	assert.Empty(t, cloned.config.SessionID)
	assert.Nil(t, cloned.sessionEvent)
}

func TestAgent_Clone_FailsForNonCloneableTools(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(nonCloneableTool{name: "plain"}))

	_, err = agent.Clone()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not implement core.CloneableTool")
}

func TestTurnBudgetReminder(t *testing.T) {
	assert.Equal(t, "", turnBudgetReminder(1, 5))
	assert.Contains(t, turnBudgetReminder(3, 5), "3 turns remaining")
	assert.Contains(t, turnBudgetReminder(5, 5), "Final turn")
}

func noCallResult() map[string]any {
	return map[string]any{
		"content": "No content or function call received from model",
		"provider_diagnostic": map[string]any{
			"provider":      "google",
			"provider_mode": "tools",
			"reason":        "empty_content_and_function_call",
			"finish_reason": "STOP",
		},
	}
}

type simpleTool struct {
	name string
	run  func(context.Context, map[string]interface{}) (core.ToolResult, error)
}

func (t simpleTool) Name() string        { return t.name }
func (t simpleTool) Description() string { return t.name }
func (t simpleTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t simpleTool) CanHandle(context.Context, string) bool { return false }
func (t simpleTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	if t.run == nil {
		return core.ToolResult{}, nil
	}
	return t.run(ctx, params)
}
func (t simpleTool) Validate(map[string]interface{}) error { return nil }
func (t simpleTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{}}
}
func (t *simpleTool) CloneTool() core.Tool {
	if t == nil {
		return nil
	}
	cloned := *t
	return &cloned
}

type nonCloneableTool struct {
	name string
}

func (t nonCloneableTool) Name() string        { return t.name }
func (t nonCloneableTool) Description() string { return t.name }
func (t nonCloneableTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t nonCloneableTool) CanHandle(context.Context, string) bool { return false }
func (t nonCloneableTool) Execute(context.Context, map[string]interface{}) (core.ToolResult, error) {
	return core.ToolResult{}, nil
}
func (t nonCloneableTool) Validate(map[string]interface{}) error { return nil }
func (t nonCloneableTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{}}
}

type stubLLM struct {
	results      []map[string]any
	index        int
	capabilities []core.Capability
	prompts      []string
}

func (m *stubLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected Generate call")
}

func (m *stubLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("unexpected GenerateWithJSON call")
}

func (m *stubLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	m.prompts = append(m.prompts, prompt)
	if m.index >= len(m.results) {
		return nil, fmt.Errorf("no more stubbed results")
	}
	result := m.results[m.index]
	m.index++
	return result, nil
}

func (m *stubLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbedding call")
}

func (m *stubLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("unexpected CreateEmbeddings call")
}

func (m *stubLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerate call")
}

func (m *stubLLM) GenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("unexpected GenerateWithContent call")
}

func (m *stubLLM) StreamGenerateWithContent(context.Context, []core.ContentBlock, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("unexpected StreamGenerateWithContent call")
}

func (m *stubLLM) ProviderName() string            { return "stub" }
func (m *stubLLM) ModelID() string                 { return "stub-model" }
func (m *stubLLM) Capabilities() []core.Capability { return m.capabilities }

type nativeStubLLM struct {
	stubLLM
	messages [][]core.ChatMessage
}

func (m *nativeStubLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	m.messages = append(m.messages, append([]core.ChatMessage(nil), messages...))
	if m.index >= len(m.results) {
		return nil, fmt.Errorf("no more stubbed results")
	}
	result := m.results[m.index]
	m.index++
	return result, nil
}

type stubSessionEventStore struct {
	getSessionErrs   []error
	getSessionIndex  int
	createSessionErr error
}

func (s *stubSessionEventStore) CreateSession(context.Context, sessionevent.CreateSessionParams) (*sessionevent.Session, *sessionevent.SessionBranch, error) {
	if s.createSessionErr != nil {
		return nil, nil, s.createSessionErr
	}
	return &sessionevent.Session{ID: "session-1", ActiveBranchID: "branch-1"}, &sessionevent.SessionBranch{ID: "branch-1", SessionID: "session-1"}, nil
}

func (s *stubSessionEventStore) AppendEntries(context.Context, []sessionevent.SessionEntry) ([]sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected AppendEntries call")
}

func (s *stubSessionEventStore) AppendSummary(context.Context, sessionevent.SessionSummary) error {
	return fmt.Errorf("unexpected AppendSummary call")
}

func (s *stubSessionEventStore) SetActiveBranch(context.Context, string, string) error {
	return fmt.Errorf("unexpected SetActiveBranch call")
}

func (s *stubSessionEventStore) ForkBranch(context.Context, string, string, string, map[string]any) (*sessionevent.SessionBranch, error) {
	return nil, fmt.Errorf("unexpected ForkBranch call")
}

func (s *stubSessionEventStore) GetSession(context.Context, string) (*sessionevent.Session, error) {
	if s.getSessionIndex < len(s.getSessionErrs) {
		err := s.getSessionErrs[s.getSessionIndex]
		s.getSessionIndex++
		if err != nil {
			return nil, err
		}
	}
	return &sessionevent.Session{ID: "session-1", ActiveBranchID: "branch-1"}, nil
}

func (s *stubSessionEventStore) ListBranches(context.Context, string) ([]sessionevent.SessionBranch, error) {
	return nil, fmt.Errorf("unexpected ListBranches call")
}

func (s *stubSessionEventStore) GetEntry(context.Context, string, string) (*sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected GetEntry call")
}

func (s *stubSessionEventStore) GetBranchHead(context.Context, string, string) (*sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected GetBranchHead call")
}

func (s *stubSessionEventStore) LoadLineage(context.Context, string, string, sessionevent.LoadOptions) ([]sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected LoadLineage call")
}

func (s *stubSessionEventStore) LoadSummaries(context.Context, string, string, int) ([]sessionevent.SessionSummary, error) {
	return nil, fmt.Errorf("unexpected LoadSummaries call")
}
