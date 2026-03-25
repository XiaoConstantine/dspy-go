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
	"github.com/XiaoConstantine/dspy-go/pkg/agents/subagent"
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

func TestAgent_Execute_EmitsProposedEventForFinish(t *testing.T) {
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
		MaxTurns: 1,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Finish immediately.",
		"task_id": "task-finish-proposed",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	proposed := findEvent(events, agents.EventToolCallProposed, "Finish")
	require.NotNil(t, proposed)
	_, hasSubagent := proposed.Data["subagent"]
	assert.False(t, hasSubagent)
}

func TestAgent_Execute_EnrichesSubagentToolEvents(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "researcher",
					"arguments": map[string]any{"query": "inspect auth"},
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
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	tool, err := subagent.AsTool(subagent.ToolConfig{
		Name:          "researcher",
		Description:   "Delegated research worker.",
		SessionPolicy: subagent.SessionPolicyDerived,
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &subagentChildAgent{
				output: map[string]any{"final_answer": "delegated result", "completed": true},
				trace: &agents.ExecutionTrace{
					AgentID:   "child-1",
					AgentType: "native",
					Status:    agents.TraceStatusSuccess,
				},
			}, nil
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(tool))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":       "Investigate auth and finish.",
		"task_id":    "task-subagent",
		"session_id": "parent-session",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	proposed := findEvent(events, agents.EventToolCallProposed, "researcher")
	started := findEvent(events, agents.EventToolCallStarted, "researcher")
	finished := findEvent(events, agents.EventToolCallFinished, "researcher")
	require.NotNil(t, proposed)
	require.NotNil(t, started)
	require.NotNil(t, finished)

	for _, event := range []*agents.AgentEvent{proposed, started, finished} {
		assert.Equal(t, true, event.Data["subagent"])
		assert.Equal(t, "researcher", event.Data["subagent_name"])
		assert.Equal(t, "derived", event.Data["session_policy"])
	}
	assert.Equal(t, true, finished.Data["child_completed"])
}

func TestAgent_Execute_EnrichesBlockedSubagentToolEvents(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "researcher",
					"arguments": map[string]any{"query": "blocked"},
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

	tool, err := subagent.AsTool(subagent.ToolConfig{
		Name:          "researcher",
		Description:   "Delegated research worker.",
		SessionPolicy: subagent.SessionPolicyEphemeral,
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &subagentChildAgent{
				output: map[string]any{"final_answer": "should not run", "completed": true},
			}, nil
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(tool))

	result, err := agent.Execute(context.Background(), map[string]interface{}{
		"task":    "Attempt blocked research and finish.",
		"task_id": "task-subagent-blocked",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	blocked := findEvent(events, agents.EventToolCallBlocked, "researcher")
	require.NotNil(t, blocked)
	assert.Equal(t, true, blocked.Data["subagent"])
	assert.Equal(t, "researcher", blocked.Data["subagent_name"])
	assert.Equal(t, "ephemeral", blocked.Data["session_policy"])
	assert.NotContains(t, blocked.Data, "child_completed")
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

func TestAgent_Execute_LoadsSessionRecallFromSessionEventStore(t *testing.T) {
	memory := agents.NewInMemoryStore()
	eventStore := newNativeTestSessionEventStore(t)
	ctx := context.Background()

	session, branch, err := eventStore.CreateSession(ctx, sessionevent.CreateSessionParams{
		ID:    "session-event-read",
		Title: "Prior session",
	})
	require.NoError(t, err)

	inserted, err := eventStore.AppendEntries(ctx, []sessionevent.SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       sessionevent.EntryKindUserMessage,
			Role:       "user",
			SearchText: "Prior event-store task",
			Payload:    map[string]any{"text": "Prior event-store task"},
		},
		{
			SessionID:  session.ID,
			BranchID:   branch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "Prior event-store answer",
			Payload:    map[string]any{"text": "Prior event-store answer"},
		},
	})
	require.NoError(t, err)
	require.Len(t, inserted, 2)

	require.NoError(t, eventStore.AppendSummary(ctx, sessionevent.SessionSummary{
		SessionID:    session.ID,
		BranchID:     branch.ID,
		StartEntryID: inserted[0].ID,
		EndEntryID:   inserted[1].ID,
		SummaryText:  "Condensed summary from event store",
	}))

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
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]interface{}{
		"task":    "Continue prior work",
		"task_id": "resume-task",
	})
	require.NoError(t, err)

	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "SESSION RECALL:")
	assert.Contains(t, llm.prompts[0], "Condensed summary from event store")
	assert.Contains(t, llm.prompts[0], "Prior event-store task")
	assert.Contains(t, llm.prompts[0], "Prior event-store answer")

	var loaded *agents.AgentEvent
	for i := range events {
		if events[i].Type == agents.EventSessionLoaded {
			loaded = &events[i]
			break
		}
	}
	require.NotNil(t, loaded)
	assert.Equal(t, "event_store", loaded.Data["source"])
	assert.Equal(t, 0, loaded.Data["record_count"])
	assert.Equal(t, 2, loaded.Data["entry_count"])
	assert.Equal(t, 1, loaded.Data["summary_count"])
	assert.Equal(t, branch.ID, loaded.Data["branch_id"])
	assert.Equal(t, inserted[1].ID, loaded.Data["head_entry_id"])
}

func TestAgent_Execute_UsesRequestedSessionBranchForRecallAndPersistence(t *testing.T) {
	eventStore := newNativeTestSessionEventStore(t)
	ctx := context.Background()

	session, mainBranch, err := eventStore.CreateSession(ctx, sessionevent.CreateSessionParams{
		ID:    "session-branch-switch",
		Title: "Branch switch",
	})
	require.NoError(t, err)

	mainEntries, err := eventStore.AppendEntries(ctx, []sessionevent.SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindUserMessage,
			Role:       "user",
			SearchText: "shared root",
			Payload:    map[string]any{"text": "shared root"},
		},
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "shared assistant",
			Payload:    map[string]any{"text": "shared assistant"},
		},
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "main branch only",
			Payload:    map[string]any{"text": "main branch only"},
		},
	})
	require.NoError(t, err)
	require.Len(t, mainEntries, 3)

	altBranch, err := eventStore.ForkBranch(ctx, session.ID, mainEntries[1].ID, "alt-path", nil)
	require.NoError(t, err)

	_, err = eventStore.AppendEntries(ctx, []sessionevent.SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   altBranch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "alternate branch only",
			Payload:    map[string]any{"text": "alternate branch only"},
		},
	})
	require.NoError(t, err)

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
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]interface{}{
		"task":              "Continue alternate branch",
		"session_branch_id": altBranch.ID,
	})
	require.NoError(t, err)

	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "Active branch: "+altBranch.ID)
	assert.Contains(t, llm.prompts[0], "alternate branch only")
	assert.NotContains(t, llm.prompts[0], "main branch only")

	updatedSession, err := eventStore.GetSession(ctx, session.ID)
	require.NoError(t, err)
	assert.Equal(t, altBranch.ID, updatedSession.ActiveBranchID)

	forkedHead, err := eventStore.GetBranchHead(ctx, session.ID, altBranch.ID)
	require.NoError(t, err)
	require.NotNil(t, forkedHead)
	lineage, err := eventStore.LoadLineage(ctx, session.ID, forkedHead.ID, sessionevent.LoadOptions{})
	require.NoError(t, err)
	assert.NotEmpty(t, lineage)
	assert.NotContains(t, sessionEventPayloadTexts(lineage), "main branch only")

	var loaded *agents.AgentEvent
	for i := range events {
		if events[i].Type == agents.EventSessionLoaded {
			loaded = &events[i]
			break
		}
	}
	require.NotNil(t, loaded)
	assert.Equal(t, "event_store", loaded.Data["source"])
	assert.Equal(t, altBranch.ID, loaded.Data["branch_id"])
}

func TestAgent_Execute_ForksSessionBranchFromRequestedEntry(t *testing.T) {
	eventStore := newNativeTestSessionEventStore(t)
	ctx := context.Background()

	session, mainBranch, err := eventStore.CreateSession(ctx, sessionevent.CreateSessionParams{
		ID:    "session-fork-request",
		Title: "Fork request",
	})
	require.NoError(t, err)

	mainEntries, err := eventStore.AppendEntries(ctx, []sessionevent.SessionEntry{
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindUserMessage,
			Role:       "user",
			SearchText: "shared root",
			Payload:    map[string]any{"text": "shared root"},
		},
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "fork point",
			Payload:    map[string]any{"text": "fork point"},
		},
		{
			SessionID:  session.ID,
			BranchID:   mainBranch.ID,
			Kind:       sessionevent.EntryKindAssistantMessage,
			Role:       "assistant",
			SearchText: "main branch only",
			Payload:    map[string]any{"text": "main branch only"},
		},
	})
	require.NoError(t, err)
	require.Len(t, mainEntries, 3)

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
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		OnEvent: func(event agents.AgentEvent) {
			events = append(events, event)
		},
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]interface{}{
		"task":                       "Investigate forked branch",
		"session_fork_from_entry_id": mainEntries[1].ID,
		"session_branch_name":        "explore-fork",
	})
	require.NoError(t, err)

	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "shared root")
	assert.Contains(t, llm.prompts[0], "fork point")
	assert.NotContains(t, llm.prompts[0], "main branch only")

	updatedSession, err := eventStore.GetSession(ctx, session.ID)
	require.NoError(t, err)
	assert.NotEqual(t, mainBranch.ID, updatedSession.ActiveBranchID)

	branches, err := eventStore.ListBranches(ctx, session.ID)
	require.NoError(t, err)
	require.Len(t, branches, 2)

	var forked *sessionevent.SessionBranch
	for i := range branches {
		if branches[i].ID == updatedSession.ActiveBranchID {
			forked = &branches[i]
			break
		}
	}
	require.NotNil(t, forked)
	assert.Equal(t, "explore-fork", forked.Name)
	assert.Equal(t, mainEntries[1].ID, forked.OriginEntryID)

	forkedHead, err := eventStore.GetBranchHead(ctx, session.ID, forked.ID)
	require.NoError(t, err)
	require.NotNil(t, forkedHead)
	lineage, err := eventStore.LoadLineage(ctx, session.ID, forkedHead.ID, sessionevent.LoadOptions{})
	require.NoError(t, err)
	assert.NotEmpty(t, lineage)
	assert.NotContains(t, sessionEventPayloadTexts(lineage), "main branch only")
	assert.Contains(t, sessionEventPayloadTexts(lineage), "shared root")

	var loaded *agents.AgentEvent
	for i := range events {
		if events[i].Type == agents.EventSessionLoaded {
			loaded = &events[i]
			break
		}
	}
	require.NotNil(t, loaded)
	assert.Equal(t, "event_store", loaded.Data["source"])
	assert.Equal(t, mainEntries[1].ID, loaded.Data["forked_from_id"])
	assert.Equal(t, forked.ID, loaded.Data["branch_id"])
	assert.Equal(t, mainEntries[1].ID, loaded.Data["head_entry_id"])
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

func TestResolveSessionEventBranch_DoesNotSwitchActiveBranchBeforeHeadLoads(t *testing.T) {
	headErr := errors.New("head unavailable")
	store := &stubSessionEventStore{
		getBranchHeadErr: headErr,
	}
	agent := &Agent{
		config: Config{
			SessionBranchID: "branch-2",
		},
	}

	_, err := agent.resolveSessionEventBranch(context.Background(), map[string]any{}, store, "session-1")
	require.Error(t, err)
	assert.ErrorIs(t, err, headErr)
	assert.Equal(t, 0, store.setActiveBranchCalls)
	assert.Equal(t, "", store.setActiveBranchID)
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

func newNativeTestSessionEventStore(t *testing.T) *sessionevent.SQLiteStore {
	t.Helper()

	store, err := sessionevent.NewSQLiteStore(filepath.Join(t.TempDir(), "session-events.db"))
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, store.Close())
	})
	return store
}

func sessionEventPayloadTexts(entries []sessionevent.SessionEntry) []string {
	texts := make([]string, 0, len(entries))
	for _, entry := range entries {
		for _, key := range []string{"text", "final_answer", "event", "observation_display", "observation"} {
			text := fmt.Sprint(entry.Payload[key])
			if text == "" || text == "<nil>" {
				continue
			}
			texts = append(texts, text)
		}
		if entry.SearchText != "" {
			texts = append(texts, entry.SearchText)
		}
	}
	return texts
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

type subagentChildAgent struct {
	output map[string]any
	err    error
	trace  *agents.ExecutionTrace
}

func (s *subagentChildAgent) Execute(context.Context, map[string]any) (map[string]any, error) {
	return core.ShallowCopyMap(s.output), s.err
}

func (s *subagentChildAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *subagentChildAgent) GetMemory() agents.Memory {
	return agents.NewInMemoryStore()
}

func (s *subagentChildAgent) LastExecutionTrace() *agents.ExecutionTrace {
	if s == nil || s.trace == nil {
		return nil
	}
	return s.trace.Clone()
}

func findEvent(events []agents.AgentEvent, eventType, toolName string) *agents.AgentEvent {
	for i := range events {
		if events[i].Type != eventType {
			continue
		}
		if events[i].Data["tool_name"] != toolName {
			continue
		}
		return &events[i]
	}
	return nil
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
	getSessionErrs       []error
	getSessionIndex      int
	createSessionErr     error
	getBranchHead        *sessionevent.SessionEntry
	getBranchHeadErr     error
	setActiveBranchErr   error
	setActiveBranchID    string
	setActiveBranchCalls int
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

func (s *stubSessionEventStore) SetActiveBranch(_ context.Context, _ string, branchID string) error {
	s.setActiveBranchCalls++
	s.setActiveBranchID = branchID
	if s.setActiveBranchErr != nil {
		return s.setActiveBranchErr
	}
	return nil
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

func (s *stubSessionEventStore) ListSessions(context.Context) ([]sessionevent.Session, error) {
	return nil, fmt.Errorf("unexpected ListSessions call")
}

func (s *stubSessionEventStore) GetEntry(context.Context, string, string) (*sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected GetEntry call")
}

func (s *stubSessionEventStore) GetBranchHead(context.Context, string, string) (*sessionevent.SessionEntry, error) {
	if s.getBranchHeadErr != nil {
		return nil, s.getBranchHeadErr
	}
	if s.getBranchHead != nil {
		cloned := *s.getBranchHead
		return &cloned, nil
	}
	return nil, nil
}

func (s *stubSessionEventStore) LoadLineage(context.Context, string, string, sessionevent.LoadOptions) ([]sessionevent.SessionEntry, error) {
	return nil, fmt.Errorf("unexpected LoadLineage call")
}

func (s *stubSessionEventStore) LoadSummaries(context.Context, string, string, int) ([]sessionevent.SessionSummary, error) {
	return nil, fmt.Errorf("unexpected LoadSummaries call")
}
