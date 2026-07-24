package native

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/sessionevent"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/skills"
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

func TestNewAgent_LoadsPersistedSkillPrompt(t *testing.T) {
	store := skills.NewMemoryStore()
	require.NoError(t, store.Save(context.Background(), skills.Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Prefer repository-specific debugging heuristics.",
		Version: 2,
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

	agent, err := NewAgent(llm, Config{
		MaxTurns:     1,
		SystemPrompt: "You are a careful code reviewer.",
		SkillStore:   store,
		SkillDomain:  "repo:test",
	})
	require.NoError(t, err)

	loadedSkill := agent.GetLoadedSkill()
	require.NotNil(t, loadedSkill)
	assert.Equal(t, "repo:test", loadedSkill.Domain)
	assert.Equal(t, 2, loadedSkill.Version)
	assert.NoError(t, agent.GetSkillLoadError())

	artifacts := agent.GetArtifacts()
	assert.Contains(t, artifacts.Text[optimize.ArtifactSkillPack], "You are a careful code reviewer.")
	assert.Contains(t, artifacts.Text[optimize.ArtifactSkillPack], "Prefer repository-specific debugging heuristics.")

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "You are a careful code reviewer.")
	assert.Contains(t, llm.prompts[0], "SKILL PACK:")
	assert.Contains(t, llm.prompts[0], "Prefer repository-specific debugging heuristics.")
}

func TestNewAgent_PersistedSkillNoOpInputs(t *testing.T) {
	tests := []struct {
		name   string
		config Config
	}{
		{
			name: "nil store",
			config: Config{
				MaxTurns:     1,
				SystemPrompt: "You are a careful code reviewer.",
				SkillDomain:  "repo:test",
			},
		},
		{
			name: "empty domain",
			config: Config{
				MaxTurns:     1,
				SystemPrompt: "You are a careful code reviewer.",
				SkillStore:   skills.NewMemoryStore(),
			},
		},
		{
			name: "missing persisted skill",
			config: Config{
				MaxTurns:     1,
				SystemPrompt: "You are a careful code reviewer.",
				SkillStore:   skills.NewMemoryStore(),
				SkillDomain:  "repo:missing",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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

			agent, err := NewAgent(llm, tt.config)
			require.NoError(t, err)
			assert.Nil(t, agent.GetLoadedSkill())
			assert.NoError(t, agent.GetSkillLoadError())

			result, err := agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
			require.NoError(t, err)
			require.True(t, result["completed"].(bool))
			require.Len(t, llm.prompts, 1)
			assert.NotContains(t, llm.prompts[0], "SKILL PACK:")
		})
	}
}

func TestNewAgent_PersistedSkillLoadErrorIsNonFatal(t *testing.T) {
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

	agent, err := NewAgent(llm, Config{
		MaxTurns:     1,
		SystemPrompt: "You are a careful code reviewer.",
		SkillStore:   nativeErrorSkillStore{err: errors.New("load failed")},
		SkillDomain:  "repo:test",
	})
	require.NoError(t, err)
	assert.Nil(t, agent.GetLoadedSkill())
	require.EqualError(t, agent.GetSkillLoadError(), "load failed")

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "You are a careful code reviewer.")
	assert.NotContains(t, llm.prompts[0], "SKILL PACK:")
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
		run: func(ctx context.Context, params map[string]any) (core.ToolResult, error) {
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]any{
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
		run: func(ctx context.Context, params map[string]any) (core.ToolResult, error) {
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

	result, err := agent.Execute(context.Background(), map[string]any{
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

func TestAgent_Execute_PreflightFailureDoesNotReplaceLastTrace(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{{"function_call": map[string]any{
			"name": "Finish", "arguments": map[string]any{"answer": "first"},
		}}},
	}
	agent, err := NewAgent(llm, Config{MaxTurns: 1})
	require.NoError(t, err)
	_, err = agent.Execute(context.Background(), map[string]any{"task": "first task", "task_id": "first"})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "Finish"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "second task", "task_id": "second"})
	assert.Nil(t, result)
	require.ErrorContains(t, err, "conflicts with executable tool")
	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.Equal(t, "first", trace.TaskID)
	assert.Equal(t, agents.RunStatusCompleted, trace.Status)
	assert.Equal(t, agents.StopReasonFinish, trace.StopReason)
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

	result, err := agent.Execute(context.Background(), map[string]any{
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
	require.Len(t, llm.prompts, 3)
	assert.Contains(t, llm.prompts[1], "Provider diagnostic: empty_content_and_function_call")
	assert.Contains(t, trace.Steps[0].Observation, "Provider diagnostic: empty_content_and_function_call")
}

func TestAgent_Execute_NoToolDiagnosticCountsOnlyConsecutiveEmptyTurns(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{"function_call": map[string]any{"name": "write_note", "arguments": map[string]any{"content": "done"}}},
			noCallResult(), noCallResult(), noCallResult(),
		},
	}
	agent, err := NewAgent(llm, Config{MaxTurns: 20})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]any{"task": "work then stall"})
	require.NoError(t, err)
	assert.Contains(t, result["error"], "after 3 turns")
	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	assert.Equal(t, agents.StopReasonNoToolCalls, trace.StopReason)
	assert.Contains(t, trace.Error, "after 3 turns")
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
		run: func(ctx context.Context, params map[string]any) (core.ToolResult, error) {
			observed = append(observed, params["content"].(string))
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]any{
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

	_, err = agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	assert.Len(t, llm.messages, 1)
	assert.Empty(t, llm.prompts)
}

func TestAgent_Execute_PreservesToolCallMetadataAcrossNativeRounds(t *testing.T) {
	llm := &nativeStubLLM{
		stubLLM: stubLLM{
			capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
			results: []map[string]any{
				{
					"content_blocks": []core.ContentBlock{
						{
							Type: core.FieldTypeText,
							Text: "Need to inspect package layout",
							Metadata: map[string]any{
								"gemini_thought":           true,
								"gemini_thought_signature": "sig-thought",
							},
						},
					},
					"tool_calls": []core.ToolCall{
						{
							ID:        "call-1",
							Name:      "write_note",
							Arguments: map[string]any{"content": "done"},
							Metadata: map[string]any{
								"gemini_thought":           true,
								"gemini_thought_signature": "sig-call",
							},
						},
					},
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
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]any) (core.ToolResult, error) {
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]any{
		"task": "Write a note and finish.",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.messages, 2)
	require.GreaterOrEqual(t, len(llm.messages[1]), 3)
	require.Len(t, llm.messages[1][1].Content, 1)
	require.Len(t, llm.messages[1][1].ToolCalls, 1)

	assert.Equal(t, "Need to inspect package layout", llm.messages[1][1].Content[0].Text)
	assert.Equal(t, "sig-thought", llm.messages[1][1].Content[0].Metadata["gemini_thought_signature"])
	assert.Equal(t, true, llm.messages[1][1].Content[0].Metadata["gemini_thought"])

	toolCall := llm.messages[1][1].ToolCalls[0]
	assert.Equal(t, "call-1", toolCall.ID)
	assert.Equal(t, "write_note", toolCall.Name)
	assert.Equal(t, "sig-call", toolCall.Metadata["gemini_thought_signature"])
	assert.Equal(t, true, toolCall.Metadata["gemini_thought"])

	require.NotNil(t, llm.messages[1][2].ToolResult)
	assert.Equal(t, "call-1", llm.messages[1][2].ToolResult.ToolCallID)
	assert.Equal(t, "write_note", llm.messages[1][2].ToolResult.Name)
}

func TestAgent_Execute_GroupsMultipleToolCallsFromOneNativeTurn(t *testing.T) {
	llm := &nativeStubLLM{
		stubLLM: stubLLM{
			capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
			results: []map[string]any{
				{
					"tool_calls": []core.ToolCall{
						{
							ID:        "call-1",
							Name:      "write_note",
							Arguments: map[string]any{"content": "first"},
							Metadata: map[string]any{
								"gemini_thought_signature": "sig-call",
							},
						},
						{
							ID:        "call-2",
							Name:      "write_note",
							Arguments: map[string]any{"content": "second"},
						},
					},
				},
				{
					"function_call": map[string]any{
						"name":      "Finish",
						"arguments": map[string]any{"answer": "done"},
					},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{MaxTurns: 4})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{
		name: "write_note",
		run: func(ctx context.Context, params map[string]any) (core.ToolResult, error) {
			return core.ToolResult{Data: params["content"]}, nil
		},
	}))

	result, err := agent.Execute(context.Background(), map[string]any{
		"task": "Write two notes and finish.",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Equal(t, 3, result["turns"])
	assert.Equal(t, 2, result["tool_calls"])
	require.Len(t, llm.messages, 2)
	require.GreaterOrEqual(t, len(llm.messages[1]), 4)
	require.Len(t, llm.messages[1][1].ToolCalls, 2)

	assert.Equal(t, "call-1", llm.messages[1][1].ToolCalls[0].ID)
	assert.Equal(t, "sig-call", llm.messages[1][1].ToolCalls[0].Metadata["gemini_thought_signature"])
	assert.Equal(t, "call-2", llm.messages[1][1].ToolCalls[1].ID)

	require.NotNil(t, llm.messages[1][2].ToolResult)
	assert.Equal(t, "call-1", llm.messages[1][2].ToolResult.ToolCallID)
	require.NotNil(t, llm.messages[1][3].ToolResult)
	assert.Equal(t, "call-2", llm.messages[1][3].ToolResult.ToolCallID)
	lastMessage := llm.messages[1][len(llm.messages[1])-1]
	assert.Contains(t, lastMessage.Content[0].Text, "TURN BUDGET: turn 2 of 4")
	assert.Contains(t, lastMessage.Content[0].Text, "3 turns remaining")
	var secondRequestText strings.Builder
	for _, message := range llm.messages[1] {
		for _, block := range message.Content {
			secondRequestText.WriteString(block.Text)
		}
	}
	assert.NotContains(t, secondRequestText.String(), "TURN BUDGET: turn 1 of 4")

	trace := agent.LastNativeTrace()
	require.NotNil(t, trace)
	require.Len(t, trace.Steps, 3)
	assert.Equal(t, "first", trace.Steps[0].Observation)
	assert.Equal(t, "second", trace.Steps[1].Observation)
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 4,
		ToolInterceptors: []core.ToolInterceptor{
			func(ctx context.Context, args map[string]any, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
				return core.ToolResult{}, &core.ToolBlockedError{Reason: "approval denied"}
			},
		},
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]any{
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

	require.NoError(t, agents.ValidateEventLifecycle(events))
	require.NotNil(t, findToolProposedEvent(events, "write_note"))
	require.NotNil(t, findToolStartedEvent(events, "write_note"))
	blocked := findToolFinishedEvent(events, "write_note")
	require.NotNil(t, blocked)
	assert.Equal(t, agents.ToolCallOutcomeBlocked, blocked.Outcome)
	var blockedErr *core.ToolBlockedError
	require.ErrorAs(t, blocked.Err, &blockedErr)
	assert.Equal(t, "approval denied", blockedErr.Reason)
	require.NotNil(t, findRunFinishedEvent(events))
}

func TestAgent_Execute_EmitsTypedExecutionEvents(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{{
			"function_call": map[string]any{
				"name":      "Finish",
				"arguments": map[string]any{"answer": "done"},
			},
		}},
	}

	var typed []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			typed = append(typed, event)
		}),
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]any{
		"task":    "Finish immediately.",
		"task_id": "task-typed-events",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.NotEmpty(t, typed)
	require.NoError(t, agents.ValidateEventLifecycle(typed))

	typeNames := make([]string, 0, len(typed))
	for _, event := range typed {
		typeNames = append(typeNames, fmt.Sprintf("%T", event.Payload))
	}
	assert.Equal(t, "agents.RunStartedEvent", typeNames[0])
	assert.Contains(t, typeNames, "agents.ToolCallProposedEvent")
	assert.Contains(t, typeNames, "agents.ToolCallFinishedEvent")
	assert.Equal(t, "agents.RunFinishedEvent", typeNames[len(typeNames)-1])
}

func TestAgent_Execute_ExternalEventConsumerCannotMutateInternalTrace(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "lookup",
					"arguments": map[string]any{"query": "original"},
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
		MaxTurns: 2,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			switch payload := event.Payload.(type) {
			case agents.ToolCallProposedEvent:
				payload.Call.Arguments["query"] = "mutated"
			case agents.ToolCallFinishedEvent:
				if payload.Result != nil && payload.Result.ToolResult != nil && len(payload.Result.ToolResult.Content) > 0 {
					payload.Result.ToolResult.Content[0].Text = "mutated"
				}
			}
		}),
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "lookup", run: func(context.Context, map[string]any) (core.ToolResult, error) {
		return core.ToolResult{Data: "found"}, nil
	}}))

	result, err := agent.Execute(context.Background(), map[string]any{
		"task":    "Look up once and finish.",
		"task_id": "task-mutation-isolation",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	execTrace := agent.LastExecutionTrace()
	require.NotNil(t, execTrace)
	require.Len(t, execTrace.Steps, 2)
	assert.Equal(t, "original", execTrace.Steps[0].Arguments["query"])
	assert.Equal(t, "found", execTrace.Steps[0].Observation)
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)

	result, err := agent.Execute(context.Background(), map[string]any{
		"task":    "Finish immediately.",
		"task_id": "task-finish-proposed",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	require.NoError(t, agents.ValidateEventLifecycle(events))
	require.NotNil(t, findToolProposedEvent(events, "Finish"))
	assert.Nil(t, findToolStartedEvent(events, "Finish"))
	finished := findToolFinishedEvent(events, "Finish")
	require.NotNil(t, finished)
	assert.Equal(t, agents.ToolCallOutcomeFinish, finished.Outcome)
}

func TestAgent_Execute_PreservesSubagentResultDetailsInTypedEvents(t *testing.T) {
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 4,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
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

	result, err := agent.Execute(context.Background(), map[string]any{
		"task":       "Investigate auth and finish.",
		"task_id":    "task-subagent",
		"session_id": "parent-session",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	require.NoError(t, agents.ValidateEventLifecycle(events))
	require.NotNil(t, findToolProposedEvent(events, "researcher"))
	require.NotNil(t, findToolStartedEvent(events, "researcher"))
	finished := findToolFinishedEvent(events, "researcher")
	require.NotNil(t, finished)
	require.NotNil(t, finished.Result)
	require.NotNil(t, finished.Result.ToolResult)
	details := finished.Result.ToolResult.Details
	assert.Equal(t, true, details["subagent"])
	assert.Equal(t, "researcher", details["subagent_name"])
	assert.Equal(t, "derived", details["session_policy"])
	assert.Equal(t, true, details["completed"])
}

func TestAgent_Execute_ReportsBlockedSubagentAsTypedToolOutcome(t *testing.T) {
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

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 4,
		ToolInterceptors: []core.ToolInterceptor{
			func(ctx context.Context, args map[string]any, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
				return core.ToolResult{}, &core.ToolBlockedError{Reason: "approval denied"}
			},
		},
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
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

	result, err := agent.Execute(context.Background(), map[string]any{
		"task":    "Attempt blocked research and finish.",
		"task_id": "task-subagent-blocked",
	})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))

	require.NoError(t, agents.ValidateEventLifecycle(events))
	blocked := findToolFinishedEvent(events, "researcher")
	require.NotNil(t, blocked)
	assert.Equal(t, agents.ToolCallOutcomeBlocked, blocked.Outcome)
	var blockedErr *core.ToolBlockedError
	require.ErrorAs(t, blocked.Err, &blockedErr)
	assert.Equal(t, "approval denied", blockedErr.Reason)
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
			func(ctx context.Context, args map[string]any, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
				return core.ToolResult{}, fmt.Errorf("approval backend unavailable")
			},
		},
	})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note"}))

	result, err := agent.Execute(context.Background(), map[string]any{
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

func TestAgent_Execute_EmitsTypedTerminalLifecycleOnGenerateError(t *testing.T) {
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}

	var events []agents.ExecutionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns: 1,
		EventSink: agents.EventSinkFunc(func(_ context.Context, event agents.ExecutionEvent) {
			events = append(events, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]any{
		"task":    "this will fail",
		"task_id": "task-generate-error",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no more stubbed results")

	require.NoError(t, agents.ValidateEventLifecycle(events))
	turnFinished := findTurnFinishedEvent(events, 1)
	require.NotNil(t, turnFinished)
	assert.Equal(t, agents.OperationStatusFailed, turnFinished.Status)
	assert.Equal(t, 0, turnFinished.ToolCallCount)
	require.Error(t, turnFinished.Err)
	assert.Contains(t, turnFinished.Err.Error(), "no more stubbed results")
	runFinished := findRunFinishedEvent(events)
	require.NotNil(t, runFinished)
	assert.Equal(t, agents.RunStatusFailed, runFinished.Status)
	require.Error(t, runFinished.Err)
	assert.Contains(t, runFinished.Err.Error(), "no more stubbed results")
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

	_, err = firstAgent.Execute(context.Background(), map[string]any{
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

	_, err = secondAgent.Execute(context.Background(), map[string]any{
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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:  1,
		Memory:    memory,
		SessionID: "session-events",
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]any{
		"task":    "Eventful task",
		"task_id": "event-task",
	})
	require.NoError(t, err)

	require.Len(t, typedSession, 2)
	loaded, ok := typedSession[0].Payload.(SessionLoadedEvent)
	require.True(t, ok)
	assert.Equal(t, "session-events", loaded.SessionID)
	assert.Equal(t, 0, loaded.RecordCount)
	assert.Equal(t, 0, loaded.RecallChars)

	persisted, ok := typedSession[1].Payload.(SessionPersistedEvent)
	require.True(t, ok)
	assert.Equal(t, "session-events", persisted.SessionID)
	assert.True(t, persisted.Success)
	assert.True(t, persisted.Completed)
	assert.False(t, persisted.EventStoreEnabled)
	assert.False(t, persisted.EventStoreSuccess)
}

func TestSessionEventSinkFunc_TypedNilIsNoOp(t *testing.T) {
	var sink SessionEventSinkFunc
	sink.EmitSessionEvent(context.Background(), SessionEvent{Payload: SessionLoadedEvent{SessionID: "session-1"}})
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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		Memory:            memory,
		SessionID:         "session-dual-write",
		SessionEventStore: eventStore,
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]any{
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

	require.Len(t, typedSession, 2)
	typedPersisted, ok := typedSession[1].Payload.(SessionPersistedEvent)
	require.True(t, ok)
	assert.Equal(t, "session-dual-write", typedPersisted.SessionID)
	assert.True(t, typedPersisted.Success)
	assert.True(t, typedPersisted.EventStoreEnabled)
	assert.True(t, typedPersisted.EventStoreSuccess)
	assert.Equal(t, 3, typedPersisted.EventEntryCount)
	assert.Equal(t, session.ActiveBranchID, typedPersisted.EventBranchID)
	require.NoError(t, typedPersisted.Err)
	require.NoError(t, typedPersisted.EventStoreErr)
}

func TestAgent_Execute_PersistsToolCallOrderingAndCorrelationToSessionEventStore(t *testing.T) {
	memory := agents.NewInMemoryStore()
	eventStore := newNativeTestSessionEventStore(t)
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{"function_call": map[string]any{"id": "call-1", "name": "write_note", "arguments": map[string]any{"content": "done"}}},
			{"function_call": map[string]any{"id": "finish-1", "name": "Finish", "arguments": map[string]any{"answer": "all set"}}},
		},
	}
	agent, err := NewAgent(llm, Config{MaxTurns: 4, Memory: memory, SessionID: "session-tool-order", SessionEventStore: eventStore})
	require.NoError(t, err)
	require.NoError(t, agent.RegisterTool(simpleTool{name: "write_note", run: func(context.Context, map[string]any) (core.ToolResult, error) {
		return core.ToolResult{Data: "tool ok", Metadata: map[string]any{core.ToolResultDisplayTextMeta: "tool display"}}, nil
	}}))

	_, err = agent.Execute(context.Background(), map[string]any{"task": "Do one tool call then finish", "task_id": "tool-order-task"})
	require.NoError(t, err)

	session, err := eventStore.GetSession(context.Background(), "session-tool-order")
	require.NoError(t, err)
	head, err := eventStore.GetBranchHead(context.Background(), session.ID, session.ActiveBranchID)
	require.NoError(t, err)
	lineage, err := eventStore.LoadLineage(context.Background(), session.ID, head.ID, sessionevent.LoadOptions{})
	require.NoError(t, err)
	require.Len(t, lineage, 5)
	assert.Equal(t, []sessionevent.EntryKind{
		sessionevent.EntryKindUserMessage,
		sessionevent.EntryKindToolCall,
		sessionevent.EntryKindToolResult,
		sessionevent.EntryKindAssistantMessage,
		sessionevent.EntryKindSystemEvent,
	}, []sessionevent.EntryKind{lineage[0].Kind, lineage[1].Kind, lineage[2].Kind, lineage[3].Kind, lineage[4].Kind})
	assert.Equal(t, "call-1", lineage[1].Payload["tool_call_id"])
	assert.Equal(t, "call-1", lineage[2].Payload["tool_call_id"])
	assert.Equal(t, "tool display", lineage[2].Payload["observation_display"])
	assert.Equal(t, "all set", lineage[3].Payload["text"])
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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		Memory:            memory,
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]any{
		"task":    "Continue prior work",
		"task_id": "resume-task",
	})
	require.NoError(t, err)

	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "SESSION RECALL:")
	assert.Contains(t, llm.prompts[0], "Condensed summary from event store")
	assert.Contains(t, llm.prompts[0], "Prior event-store task")
	assert.Contains(t, llm.prompts[0], "Prior event-store answer")

	require.Len(t, typedSession, 2)
	loaded, ok := typedSession[0].Payload.(SessionLoadedEvent)
	require.True(t, ok)
	assert.Equal(t, "event_store", loaded.Source)
	assert.Equal(t, 0, loaded.RecordCount)
	assert.Equal(t, 2, loaded.EntryCount)
	assert.Equal(t, 1, loaded.SummaryCount)
	assert.Equal(t, branch.ID, loaded.BranchID)
	assert.Equal(t, inserted[1].ID, loaded.HeadEntryID)
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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]any{
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

	require.Len(t, typedSession, 2)
	loaded, ok := typedSession[0].Payload.(SessionLoadedEvent)
	require.True(t, ok)
	assert.Equal(t, "event_store", loaded.Source)
	assert.Equal(t, altBranch.ID, loaded.BranchID)
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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		SessionID:         session.ID,
		SessionEventStore: eventStore,
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(ctx, map[string]any{
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

	require.Len(t, typedSession, 2)
	loaded, ok := typedSession[0].Payload.(SessionLoadedEvent)
	require.True(t, ok)
	assert.Equal(t, "event_store", loaded.Source)
	assert.Equal(t, mainEntries[1].ID, loaded.ForkedFromEntryID)
	assert.Equal(t, forked.ID, loaded.BranchID)
	assert.Equal(t, mainEntries[1].ID, loaded.HeadEntryID)
}

func TestAgent_Execute_ReportsSessionEventStoreFailureWithoutBreakingSnapshotPersistence(t *testing.T) {
	memory := agents.NewInMemoryStore()
	eventStoreErr := errors.New("sentinel append failure")
	eventStore := &stubSessionEventStore{appendEntriesErr: eventStoreErr}

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

	var typedSession []SessionEvent
	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		Memory:            memory,
		SessionID:         "session-event-failure",
		SessionEventStore: eventStore,
		SessionEventSink: SessionEventSinkFunc(func(_ context.Context, event SessionEvent) {
			typedSession = append(typedSession, event)
		}),
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]any{
		"task":    "Event store should fail only for dual write",
		"task_id": "dual-write-failure",
	})
	require.NoError(t, err)

	records, err := agents.NewSessionStore(memory).Recent("session-event-failure", 10)
	require.NoError(t, err)
	require.Len(t, records, 1)
	assert.Equal(t, "Event store should fail only for dual write", records[0].Task)

	require.Len(t, typedSession, 2)
	typedPersisted, ok := typedSession[1].Payload.(SessionPersistedEvent)
	require.True(t, ok)
	assert.Equal(t, "session-event-failure", typedPersisted.SessionID)
	assert.True(t, typedPersisted.Success)
	assert.True(t, typedPersisted.EventStoreEnabled)
	assert.False(t, typedPersisted.EventStoreSuccess)
	assert.Equal(t, "branch-1", typedPersisted.EventBranchID)
	require.NoError(t, typedPersisted.Err)
	require.ErrorIs(t, typedPersisted.EventStoreErr, eventStoreErr)
	assert.Equal(t, eventStoreErr.Error(), typedPersisted.EventStoreErr.Error())
}

func TestSessionEntriesFromEvents_DeduplicatesFinalAnswerAssistantEntry(t *testing.T) {
	base := time.Date(2026, time.March, 21, 12, 0, 0, 0, time.UTC)
	entries, err := sessionevent.EntriesFromEvents([]agents.ExecutionEvent{
		{Timestamp: base, Payload: agents.RunStartedEvent{RunID: "run-1", Task: "Summarize and finish", Model: "stub-model", Provider: "stub"}},
		{Timestamp: base.Add(time.Second), Payload: agents.MessageAddedEvent{RunID: "run-1", Turn: 1, Message: agents.Message{Role: agents.RoleAssistant, ToolCalls: []core.ToolCall{{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}}}}},
		{Timestamp: base.Add(1500 * time.Millisecond), Payload: agents.ToolCallFinishedEvent{RunID: "run-1", Turn: 1, ToolIndex: 0, Call: core.ToolCall{ID: "finish-1", Name: "Finish", Arguments: map[string]any{"answer": "done"}}, Outcome: agents.ToolCallOutcomeFinish, Status: agents.OperationStatusCompleted}},
		{Timestamp: base.Add(2 * time.Second), Payload: agents.TurnFinishedEvent{RunID: "run-1", Turn: 1, Status: agents.OperationStatusCompleted, Usage: &core.TokenInfo{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13}}},
		{Timestamp: base.Add(3 * time.Second), Payload: agents.RunFinishedEvent{RunID: "run-1", Status: agents.RunStatusCompleted, StopReason: agents.StopReasonFinish, FinalAnswer: "done"}},
	}, "session-1", "branch-1", sessionevent.EventProjectionConfig{RunID: "run-1", TaskID: "task-final-answer", Source: "native"})
	require.NoError(t, err)
	require.Len(t, entries, 3)
	assert.Equal(t, sessionevent.EntryKindUserMessage, entries[0].Kind)
	assert.Equal(t, sessionevent.EntryKindAssistantMessage, entries[1].Kind)
	assert.Equal(t, "done", entries[1].Payload["text"])
	assert.Equal(t, sessionevent.EntryKindSystemEvent, entries[2].Kind)
	assert.Equal(t, "done", entries[2].Payload["final_answer"])
}

func TestBuildSessionEventRecall_IncludesStoppedRunDiagnostic(t *testing.T) {
	recall := buildSessionEventRecall("branch-1", nil, []sessionevent.SessionEntry{{
		Kind:       sessionevent.EntryKindSystemEvent,
		SearchText: "repeated model responses without tool calls after 3 turns",
		Payload: map[string]any{
			"event":       "run_finished",
			"stop_reason": "no_tool_calls",
			"diagnostic":  "repeated model responses without tool calls after 3 turns",
		},
	}}, 400)
	assert.Contains(t, recall, "Run stopped (no_tool_calls): repeated model responses without tool calls after 3 turns")
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

	result, err := agent.Execute(context.Background(), map[string]any{
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

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	assert.Len(t, base.prompts, 1)
	assert.Contains(t, base.prompts[0], "TURN BUDGET: turn 1 of 1")
	assert.Contains(t, base.prompts[0], "Final turn")
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

func TestAgent_Clone_DoesNotDuplicatePersistedSkillPrompt(t *testing.T) {
	store := skills.NewMemoryStore()
	require.NoError(t, store.Save(context.Background(), skills.Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Prefer repository-specific debugging heuristics.",
		Version: 2,
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

	agent, err := NewAgent(llm, Config{
		MaxTurns:     1,
		SystemPrompt: "You are a careful code reviewer.",
		SkillStore:   store,
		SkillDomain:  "repo:test",
	})
	require.NoError(t, err)

	clonedAny, err := agent.Clone()
	require.NoError(t, err)
	cloned, ok := clonedAny.(*Agent)
	require.True(t, ok)

	result, err := cloned.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 1)
	assert.Equal(t, 1, strings.Count(llm.prompts[0], "SKILL PACK:"))
	assert.Equal(t, 1, strings.Count(llm.prompts[0], "Prefer repository-specific debugging heuristics."))
	assert.Equal(t, 1, strings.Count(llm.prompts[0], "You are a careful code reviewer."))
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

func TestAgent_SetArtifacts_ClearsPersistedSkillOverlay(t *testing.T) {
	store := skills.NewMemoryStore()
	require.NoError(t, store.Save(context.Background(), skills.Skill{
		Name:    "repo-skill",
		Domain:  "repo:test",
		Content: "Prefer repository-specific debugging heuristics.",
		Version: 2,
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

	agent, err := NewAgent(llm, Config{
		MaxTurns:     1,
		SystemPrompt: "You are a careful code reviewer.",
		SkillStore:   store,
		SkillDomain:  "repo:test",
	})
	require.NoError(t, err)

	require.NoError(t, agent.SetArtifacts(optimize.AgentArtifacts{
		Text: map[optimize.ArtifactKey]string{
			optimize.ArtifactSkillPack: "Manual override prompt.",
		},
	}))

	assert.Nil(t, agent.GetLoadedSkill())
	assert.NoError(t, agent.GetSkillLoadError())
	assert.Equal(t, "Manual override prompt.", agent.GetArtifacts().Text[optimize.ArtifactSkillPack])

	result, err := agent.Execute(context.Background(), map[string]any{"task": "Finish immediately."})
	require.NoError(t, err)
	require.True(t, result["completed"].(bool))
	require.Len(t, llm.prompts, 1)
	assert.Contains(t, llm.prompts[0], "Manual override prompt.")
	assert.NotContains(t, llm.prompts[0], "SKILL PACK:")
	assert.NotContains(t, llm.prompts[0], "Prefer repository-specific debugging heuristics.")
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

type nativeErrorSkillStore struct {
	err error
}

func (s nativeErrorSkillStore) Save(context.Context, skills.Skill) error { return s.err }

func (s nativeErrorSkillStore) Load(context.Context, string) ([]skills.Skill, error) {
	return nil, s.err
}

func (s nativeErrorSkillStore) Best(context.Context, string) (*skills.Skill, error) {
	return nil, s.err
}

type simpleTool struct {
	name string
	run  func(context.Context, map[string]any) (core.ToolResult, error)
}

type subagentChildAgent struct {
	output map[string]any
	err    error
	trace  *agents.ExecutionTrace
}

func (s *subagentChildAgent) Execute(context.Context, map[string]any) (map[string]any, error) {
	return maps.Clone(s.output), s.err
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

func findToolProposedEvent(events []agents.ExecutionEvent, toolName string) *agents.ToolCallProposedEvent {
	for _, event := range events {
		payload, ok := event.Payload.(agents.ToolCallProposedEvent)
		if ok && payload.Call.Name == toolName {
			return &payload
		}
	}
	return nil
}

func findToolStartedEvent(events []agents.ExecutionEvent, toolName string) *agents.ToolExecutionStartedEvent {
	for _, event := range events {
		payload, ok := event.Payload.(agents.ToolExecutionStartedEvent)
		if ok && payload.Call.Name == toolName {
			return &payload
		}
	}
	return nil
}

func findToolFinishedEvent(events []agents.ExecutionEvent, toolName string) *agents.ToolCallFinishedEvent {
	for _, event := range events {
		payload, ok := event.Payload.(agents.ToolCallFinishedEvent)
		if ok && payload.Call.Name == toolName {
			return &payload
		}
	}
	return nil
}

func findTurnFinishedEvent(events []agents.ExecutionEvent, turn int) *agents.TurnFinishedEvent {
	for _, event := range events {
		payload, ok := event.Payload.(agents.TurnFinishedEvent)
		if ok && payload.Turn == turn {
			return &payload
		}
	}
	return nil
}

func findRunFinishedEvent(events []agents.ExecutionEvent) *agents.RunFinishedEvent {
	for _, event := range events {
		if payload, ok := event.Payload.(agents.RunFinishedEvent); ok {
			return &payload
		}
	}
	return nil
}

func (t simpleTool) Name() string        { return t.name }
func (t simpleTool) Description() string { return t.name }
func (t simpleTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.name, Description: t.name}
}
func (t simpleTool) CanHandle(context.Context, string) bool { return false }
func (t simpleTool) Execute(ctx context.Context, params map[string]any) (core.ToolResult, error) {
	if t.run == nil {
		return core.ToolResult{}, nil
	}
	return t.run(ctx, params)
}
func (t simpleTool) Validate(map[string]any) error { return nil }
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
func (t nonCloneableTool) Execute(context.Context, map[string]any) (core.ToolResult, error) {
	return core.ToolResult{}, nil
}
func (t nonCloneableTool) Validate(map[string]any) error { return nil }
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

func (m *stubLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, fmt.Errorf("unexpected GenerateWithJSON call")
}

func (m *stubLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
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
	appendEntriesErr     error
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

func (s *stubSessionEventStore) AppendEntries(_ context.Context, entries []sessionevent.SessionEntry) ([]sessionevent.SessionEntry, error) {
	if s.appendEntriesErr != nil {
		return nil, s.appendEntriesErr
	}
	inserted := make([]sessionevent.SessionEntry, len(entries))
	copy(inserted, entries)
	for i := range inserted {
		if strings.TrimSpace(inserted[i].ID) == "" {
			inserted[i].ID = fmt.Sprintf("entry-%d", i+1)
		}
		if inserted[i].CreatedAt.IsZero() {
			inserted[i].CreatedAt = time.Date(2026, time.July, 23, 0, 0, i+1, 0, time.UTC)
		}
	}
	return inserted, nil
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
