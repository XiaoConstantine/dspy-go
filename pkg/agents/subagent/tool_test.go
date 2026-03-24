package subagent

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAsTool_UsesBuildAgentPerInvocation(t *testing.T) {
	t.Parallel()

	buildCalls := 0
	tool, err := AsTool(ToolConfig{
		Name:        "research",
		Description: "Run a specialized research worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			buildCalls++
			return &stubAgent{
				output: map[string]any{
					"final_answer": "done",
					"completed":    true,
				},
				trace: &agents.ExecutionTrace{AgentType: "stub"},
			}, nil
		},
	})
	require.NoError(t, err)

	_, err = tool.Execute(context.Background(), map[string]any{"task": "first"})
	require.NoError(t, err)
	_, err = tool.Execute(context.Background(), map[string]any{"task": "second"})
	require.NoError(t, err)

	assert.Equal(t, 2, buildCalls)
}

func TestAsTool_ClonesStoredOptimizableAgentPerInvocation(t *testing.T) {
	t.Parallel()

	cloneCount := 0
	base := &cloneableStubAgent{
		cloneCount: &cloneCount,
		output: map[string]any{
			"final_answer": "ok",
			"completed":    true,
		},
	}

	tool, err := AsTool(ToolConfig{
		Name:        "verify",
		Description: "Run a verifier worker.",
		Agent:       base,
	})
	require.NoError(t, err)

	_, err = tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	_, err = tool.Execute(context.Background(), nil)
	require.NoError(t, err)

	assert.Equal(t, 2, cloneCount)
}

func TestAsTool_RejectsNonCloneableStoredAgent(t *testing.T) {
	t.Parallel()

	_, err := AsTool(ToolConfig{
		Name:        "unsafe",
		Description: "Unsafe stored agent.",
		Agent:       &stubAgent{},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "optimize.OptimizableAgent")
}

func TestTool_ProjectsBestEffortParentContext(t *testing.T) {
	t.Parallel()

	var gotParent ParentContext
	var gotChildInput map[string]any

	tool, err := AsTool(ToolConfig{
		Name:        "worker",
		Description: "Worker tool.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				output: map[string]any{
					"final_answer": "done",
					"completed":    true,
				},
			}, nil
		},
		StaticParentContext: ParentContext{
			ParentAgentType: "native",
		},
		BuildInput: func(args map[string]any, parent ParentContext) (map[string]any, error) {
			gotParent = parent
			gotChildInput = map[string]any{
				"task":       args["task"],
				"session_id": parent.SessionID,
				"task_id":    parent.TaskID,
			}
			return gotChildInput, nil
		},
	})
	require.NoError(t, err)

	ctx := WithParentContext(context.Background(), ParentContext{
		TaskID:          "task-123",
		ParentAgentID:   "parent-1",
		SessionID:       "session-1",
		Input:           map[string]any{"question": "parent input"},
		ParentAgentType: "override-type",
	})
	_, err = tool.Execute(ctx, map[string]any{"task": "inspect this"})
	require.NoError(t, err)

	assert.Equal(t, "task-123", gotParent.TaskID)
	assert.Equal(t, "parent-1", gotParent.ParentAgentID)
	assert.Equal(t, "override-type", gotParent.ParentAgentType)
	assert.Equal(t, "session-1", gotParent.SessionID)
	assert.Equal(t, map[string]any{"question": "parent input"}, gotParent.Input)
	assert.Equal(t, "inspect this", gotChildInput["task"])
}

func TestTool_DefaultResultUsesFinalAnswerAndTraceDetails(t *testing.T) {
	t.Parallel()

	tool, err := AsTool(ToolConfig{
		Name:        "research",
		Description: "Research worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				output: map[string]any{
					"final_answer": "Found two relevant files.",
					"completed":    true,
				},
				trace: &agents.ExecutionTrace{
					AgentID:   "child-1",
					AgentType: "native",
					Status:    agents.TraceStatusSuccess,
				},
			}, nil
		},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "Found two relevant files.", core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta))

	details, _ := result.Annotations[core.ToolResultDetailsAnnotation].(map[string]any)
	require.NotNil(t, details)
	assert.Equal(t, true, details["subagent"])
	assert.Equal(t, "research", details["subagent_name"])
	trace, ok := details["trace"].(SubagentTraceRef)
	require.True(t, ok)
	assert.Equal(t, "research", trace.Name)
	require.NotNil(t, trace.Trace)
	assert.Equal(t, "child-1", trace.Trace.AgentID)
}

func TestTool_TranslatesChildErrorIntoErrorToolResult(t *testing.T) {
	t.Parallel()

	tool, err := AsTool(ToolConfig{
		Name:        "fragile",
		Description: "Fragile worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				err: errors.New("boom"),
				trace: &agents.ExecutionTrace{
					AgentID: "child-err",
				},
			}, nil
		},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, true, result.Metadata[core.ToolResultIsErrorMeta])

	details, _ := result.Annotations[core.ToolResultDetailsAnnotation].(map[string]any)
	require.NotNil(t, details)
	trace, ok := details["trace"].(SubagentTraceRef)
	require.True(t, ok)
	require.NotNil(t, trace.Trace)
	assert.Equal(t, "child-err", trace.Trace.AgentID)
}

func TestTool_TimeoutReturnsErrorToolResult(t *testing.T) {
	t.Parallel()

	tool, err := AsTool(ToolConfig{
		Name:        "slow",
		Description: "Slow worker.",
		Timeout:     10 * time.Millisecond,
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &blockingAgent{}, nil
		},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, true, result.Metadata[core.ToolResultIsErrorMeta])
	assert.Contains(t, core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta), "failed")
}

func TestTool_SessionPolicyDerivedAddsChildSessionID(t *testing.T) {
	t.Parallel()

	var seen map[string]any
	tool, err := AsTool(ToolConfig{
		Name:          "deep research",
		Description:   "Derived-session worker.",
		SessionPolicy: SessionPolicyDerived,
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				output: map[string]any{"final_answer": "ok", "completed": true},
			}, nil
		},
		BuildInput: func(args map[string]any, parent ParentContext) (map[string]any, error) {
			seen = map[string]any{"task": args["task"]}
			return seen, nil
		},
	})
	require.NoError(t, err)

	ctx := WithParentContext(context.Background(), ParentContext{SessionID: "parent-session"})
	_, err = tool.Execute(ctx, map[string]any{"task": "find evidence"})
	require.NoError(t, err)

	assert.Equal(t, "parent-session/deep-research", seen["session_id"])
}

func TestTool_SessionPolicyEphemeralStripsSessionKeys(t *testing.T) {
	t.Parallel()

	var seen map[string]any
	tool, err := AsTool(ToolConfig{
		Name:          "isolated",
		Description:   "Ephemeral worker.",
		SessionPolicy: SessionPolicyEphemeral,
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				output: map[string]any{"final_answer": "ok", "completed": true},
			}, nil
		},
		BuildInput: func(args map[string]any, parent ParentContext) (map[string]any, error) {
			seen = map[string]any{
				"task":                       args["task"],
				"session_id":                 parent.SessionID,
				"session_branch_id":          "branch-from-builder",
				"session_fork_from_entry_id": "entry-from-builder",
			}
			return seen, nil
		},
	})
	require.NoError(t, err)

	ctx := WithParentContext(context.Background(), ParentContext{
		SessionID: "parent-session",
		BranchID:  "parent-branch",
	})
	_, err = tool.Execute(ctx, map[string]any{
		"task":                       "inspect",
		"session_id":                 "explicit-session",
		"session_branch_id":          "explicit-branch",
		"session_fork_from_entry_id": "explicit-entry",
	})
	require.NoError(t, err)

	_, hasSession := seen["session_id"]
	_, hasBranch := seen["session_branch_id"]
	_, hasFork := seen["session_fork_from_entry_id"]
	assert.False(t, hasSession)
	assert.False(t, hasBranch)
	assert.False(t, hasFork)
}

func TestTool_DefaultResultFallsBackToJSONSummary(t *testing.T) {
	t.Parallel()

	tool, err := AsTool(ToolConfig{
		Name:        "structured",
		Description: "Structured worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{
				output: map[string]any{
					"issues": []map[string]any{
						{"path": "auth.go", "message": "missing nil check"},
					},
					"completed": true,
				},
			}, nil
		},
	})
	require.NoError(t, err)

	result, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)

	modelText := core.ToolResultMetadataString(result.Metadata, core.ToolResultModelTextMeta)
	assert.Contains(t, modelText, "\"issues\"")
	assert.Contains(t, modelText, "\"auth.go\"")
}

func TestTool_ImplementsCloneableTool(t *testing.T) {
	t.Parallel()

	tool, err := AsTool(ToolConfig{
		Name:        "cloneable",
		Description: "Cloneable worker.",
		BuildAgent: func(context.Context, map[string]any) (agents.Agent, error) {
			return &stubAgent{output: map[string]any{"final_answer": "ok"}}, nil
		},
	})
	require.NoError(t, err)

	cloneable, ok := tool.(core.CloneableTool)
	require.True(t, ok)
	cloned := cloneable.CloneTool()
	require.NotNil(t, cloned)
	assert.NotSame(t, tool, cloned)
}

type stubAgent struct {
	output map[string]any
	err    error
	trace  *agents.ExecutionTrace
}

func (s *stubAgent) Execute(context.Context, map[string]any) (map[string]any, error) {
	return core.ShallowCopyMap(s.output), s.err
}

func (s *stubAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *stubAgent) GetMemory() agents.Memory {
	return agents.NewInMemoryStore()
}

func (s *stubAgent) LastExecutionTrace() *agents.ExecutionTrace {
	return s.trace.Clone()
}

type cloneableStubAgent struct {
	cloneCount *int
	output     map[string]any
}

func (s *cloneableStubAgent) Execute(context.Context, map[string]any) (map[string]any, error) {
	return core.ShallowCopyMap(s.output), nil
}

func (s *cloneableStubAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *cloneableStubAgent) GetMemory() agents.Memory {
	return agents.NewInMemoryStore()
}

func (s *cloneableStubAgent) GetArtifacts() optimize.AgentArtifacts {
	return optimize.AgentArtifacts{}
}

func (s *cloneableStubAgent) SetArtifacts(optimize.AgentArtifacts) error {
	return nil
}

func (s *cloneableStubAgent) Clone() (optimize.OptimizableAgent, error) {
	*s.cloneCount = *s.cloneCount + 1
	return &cloneableStubAgent{
		cloneCount: s.cloneCount,
		output:     core.ShallowCopyMap(s.output),
	}, nil
}

type blockingAgent struct{}

func (b *blockingAgent) Execute(ctx context.Context, _ map[string]any) (map[string]any, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (b *blockingAgent) GetCapabilities() []core.Tool {
	return nil
}

func (b *blockingAgent) GetMemory() agents.Memory {
	return agents.NewInMemoryStore()
}
