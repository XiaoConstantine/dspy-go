package native

import (
	"context"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgent_SessionControls_RequireEventStore(t *testing.T) {
	agent, err := NewAgent(&stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
	}, Config{SessionID: "session-1"})
	require.NoError(t, err)

	_, err = agent.GetSessionState(context.Background(), "")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrSessionEventStoreUnavailable)

	err = agent.SwitchSessionBranch(context.Background(), "", "branch-1")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrSessionEventStoreUnavailable)

	_, err = agent.ForkActiveSession(context.Background(), "", "alt", true)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrSessionEventStoreUnavailable)
}

func TestAgent_GetSessionState_UsesConfiguredSessionID(t *testing.T) {
	eventStore := newNativeTestSessionEventStore(t)
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
		MaxTurns:          1,
		SessionID:         "session-state",
		SessionEventStore: eventStore,
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task": "Create state",
	})
	require.NoError(t, err)

	state, err := agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, state)
	require.NotNil(t, state.Session)
	require.NotNil(t, state.ActiveBranch)
	require.NotNil(t, state.HeadEntry)
	assert.Equal(t, "session-state", state.Session.ID)
	assert.Equal(t, state.Session.ActiveBranchID, state.ActiveBranch.ID)
	assert.Len(t, state.Branches, 1)
}

func TestAgent_ForkActiveSession_CanLeaveActiveBranchUnchanged(t *testing.T) {
	eventStore := newNativeTestSessionEventStore(t)
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
		MaxTurns:          1,
		SessionID:         "session-fork-control",
		SessionEventStore: eventStore,
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task": "Seed active branch",
	})
	require.NoError(t, err)

	before, err := agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, before.ActiveBranch)

	forked, err := agent.ForkActiveSession(context.Background(), "", "inactive-fork", false)
	require.NoError(t, err)
	assert.Equal(t, "inactive-fork", forked.Name)

	after, err := agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, after.ActiveBranch)
	assert.Equal(t, before.ActiveBranch.ID, after.ActiveBranch.ID)
	assert.NotEqual(t, forked.ID, after.ActiveBranch.ID)
}

func TestAgent_SessionControls_SupportLifecycleWithoutRawBranchKeys(t *testing.T) {
	eventStore := newNativeTestSessionEventStore(t)
	llm := &stubLLM{
		capabilities: []core.Capability{core.CapabilityCompletion, core.CapabilityToolCalling},
		results: []map[string]any{
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "main done"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "branch done"},
				},
			},
			{
				"function_call": map[string]any{
					"name":      "Finish",
					"arguments": map[string]any{"answer": "back on main"},
				},
			},
		},
	}

	agent, err := NewAgent(llm, Config{
		MaxTurns:          1,
		SessionID:         "session-lifecycle",
		SessionEventStore: eventStore,
	})
	require.NoError(t, err)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task": "main first",
	})
	require.NoError(t, err)
	require.Len(t, llm.prompts, 1)

	state, err := agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, state.ActiveBranch)
	originalBranchID := state.ActiveBranch.ID

	forked, err := agent.ForkActiveSession(context.Background(), "", "alt-branch", true)
	require.NoError(t, err)
	assert.Equal(t, "alt-branch", forked.Name)

	state, err = agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, state.ActiveBranch)
	assert.Equal(t, forked.ID, state.ActiveBranch.ID)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task": "branch second",
	})
	require.NoError(t, err)
	require.Len(t, llm.prompts, 2)
	assert.Contains(t, llm.prompts[1], "SESSION RECALL:")
	assert.Contains(t, llm.prompts[1], "main first")

	err = agent.SwitchSessionBranch(context.Background(), "", originalBranchID)
	require.NoError(t, err)

	state, err = agent.GetSessionState(context.Background(), "")
	require.NoError(t, err)
	require.NotNil(t, state.ActiveBranch)
	assert.Equal(t, originalBranchID, state.ActiveBranch.ID)

	_, err = agent.Execute(context.Background(), map[string]interface{}{
		"task": "main third",
	})
	require.NoError(t, err)
	require.Len(t, llm.prompts, 3)
	assert.Contains(t, llm.prompts[2], "main first")
	assert.NotContains(t, llm.prompts[2], "branch second")
}
