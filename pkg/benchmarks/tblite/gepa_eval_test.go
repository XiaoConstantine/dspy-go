package tblite

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeOptimizableTaskAgent struct {
	artifacts  optimize.AgentArtifacts
	trace      *agents.ExecutionTrace
	shouldPass bool
}

func (a *fakeOptimizableTaskAgent) RunTask(ctx context.Context, req TerminalTaskRequest) (*TerminalTaskResult, error) {
	content := "bad"
	if a.shouldPass {
		content = "ok"
	}
	if err := os.WriteFile(filepath.Join(req.EnvironmentDir, "result.txt"), []byte(content), 0o644); err != nil {
		return nil, err
	}
	a.trace = &agents.ExecutionTrace{
		Task:             req.TaskID,
		Status:           agents.TraceStatusSuccess,
		TerminationCause: "finish",
		TokenUsage:       map[string]int64{"total_tokens": 7},
	}
	return &TerminalTaskResult{
		Completed:  true,
		ToolCalls:  1,
		TokenUsage: TokenUsage{TotalTokens: 7},
	}, nil
}

func (a *fakeOptimizableTaskAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}
func (a *fakeOptimizableTaskAgent) GetCapabilities() []core.Tool { return nil }
func (a *fakeOptimizableTaskAgent) GetMemory() agents.Memory     { return agents.NewInMemoryStore() }
func (a *fakeOptimizableTaskAgent) GetArtifacts() optimize.AgentArtifacts {
	return a.artifacts.Clone()
}
func (a *fakeOptimizableTaskAgent) SetArtifacts(artifacts optimize.AgentArtifacts) error {
	a.artifacts = artifacts.Clone()
	return nil
}
func (a *fakeOptimizableTaskAgent) Clone() (optimize.OptimizableAgent, error) {
	cloned := *a
	cloned.artifacts = a.artifacts.Clone()
	cloned.trace = a.trace.Clone()
	return &cloned, nil
}
func (a *fakeOptimizableTaskAgent) LastExecutionTrace() *agents.ExecutionTrace {
	return a.trace.Clone()
}

func TestGEPAEvaluator_EvaluatesRealTask(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:    "tblite-pass",
		Instruction: "write result.txt",
		TestScript:  "#!/bin/sh\nset -eu\n[ \"$(cat \"$DSPY_TBLITE_ENV_DIR/result.txt\")\" = \"ok\" ]\n",
	}

	evaluator := NewGEPAEvaluator(GEPAEvaluatorConfig{
		RootDir:           t.TempDir(),
		MaxTurns:          3,
		UseTaskContainers: false,
	})

	result, err := evaluator.Evaluate(context.Background(), &fakeOptimizableTaskAgent{
		shouldPass: true,
		artifacts: optimize.AgentArtifacts{
			Text: map[optimize.ArtifactKey]string{
				optimize.ArtifactSkillPack: "use tools carefully",
			},
		},
	}, ExampleFromTask(task))
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, 1.0, result.Score)
	require.NotNil(t, result.SideInfo)
	assert.Equal(t, 1.0, result.SideInfo.Scores["tblite_pass"])
	assert.Equal(t, 1, result.SideInfo.Diagnostics["tool_calls"])
	require.NotNil(t, result.SideInfo.Trace)
	assert.Equal(t, "tblite-pass", result.SideInfo.Trace.Task)
}

func TestGEPAEvaluator_UsesPartialVerifierCredit(t *testing.T) {
	task := datasets.TBLiteTask{
		TaskName:    "tblite-partial",
		Instruction: "emit partial verifier output",
		TestScript:  "#!/bin/sh\nset -eu\necho '=========================== short test summary info ============================'\necho 'FAILED tests/test_outputs.py::test_one'\necho '==================== 2 failed, 1 passed in 0.37s ===================='\nexit 1\n",
	}

	evaluator := NewGEPAEvaluator(GEPAEvaluatorConfig{
		RootDir:           t.TempDir(),
		MaxTurns:          3,
		UseTaskContainers: false,
	})

	result, err := evaluator.Evaluate(context.Background(), &fakeOptimizableTaskAgent{
		shouldPass: false,
		artifacts: optimize.AgentArtifacts{
			Text: map[optimize.ArtifactKey]string{
				optimize.ArtifactToolPolicy: "use targeted inspection",
			},
		},
	}, ExampleFromTask(task))
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.InDelta(t, 1.0/3.0, result.Score, 0.0001)
	require.NotNil(t, result.SideInfo)
	assert.InDelta(t, 1.0/3.0, result.SideInfo.Scores["tblite_pass"], 0.0001)
	assert.Equal(t, 1.0/3.0, result.SideInfo.Diagnostics["verifier_pass_fraction"])
}

func TestVerifierPassFraction(t *testing.T) {
	score := verifierPassFraction(
		"==================== 2 failed, 1 passed in 0.37s ====================",
		"==================== 1 skipped, 1 error in 0.10s ====================",
	)

	assert.InDelta(t, 1.0/5.0, score, 0.0001)
}
