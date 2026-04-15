package optimize

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type targetAwareMockAgent struct {
	*mockOptimizableAgent
}

type updateTrackingAgent struct {
	artifacts    AgentArtifacts
	updateCalled bool
	getCalled    bool
	setCalled    bool
}

func newTargetAwareMockAgent() *targetAwareMockAgent {
	agent := &targetAwareMockAgent{mockOptimizableAgent: newMockOptimizableAgent()}
	agent.artifacts = AgentArtifacts{
		Text: map[ArtifactKey]string{
			ArtifactSkillPack:  "Use the repository debugging guide.",
			ArtifactToolPolicy: "Prefer deterministic tools first.",
		},
		Int: map[string]int{
			"max_turns": 8,
		},
		Bool: map[string]bool{},
	}
	return agent
}

func (m *targetAwareMockAgent) Clone() (OptimizableAgent, error) {
	cloned, err := m.mockOptimizableAgent.Clone()
	if err != nil {
		return nil, err
	}
	return &targetAwareMockAgent{mockOptimizableAgent: cloned.(*mockOptimizableAgent)}, nil
}

func (m *targetAwareMockAgent) ListOptimizationTargets() []OptimizationTargetDescriptor {
	return []OptimizationTargetDescriptor{
		{
			ID:          "root.system",
			Kind:        OptimizationTargetText,
			Description: "System guidance",
			ArtifactKey: ArtifactSkillPack,
		},
		{
			ID:          "root.tool_policy",
			Kind:        OptimizationTargetText,
			Description: "Tool policy",
			ArtifactKey: ArtifactToolPolicy,
		},
		{
			ID:          "root.max_turns",
			Kind:        OptimizationTargetInt,
			Description: "Max turns",
			IntKey:      "max_turns",
		},
	}
}

func (m *targetAwareMockAgent) OptimizationAgentType() string {
	return "mock"
}

func (a *updateTrackingAgent) Execute(context.Context, map[string]interface{}) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

func (a *updateTrackingAgent) GetCapabilities() []core.Tool { return nil }

func (a *updateTrackingAgent) GetMemory() agents.Memory { return nil }

func (a *updateTrackingAgent) GetArtifacts() AgentArtifacts {
	a.getCalled = true
	return a.artifacts.Clone()
}

func (a *updateTrackingAgent) SetArtifacts(AgentArtifacts) error {
	a.setCalled = true
	return nil
}

func (a *updateTrackingAgent) UpdateArtifacts(update func(AgentArtifacts) (AgentArtifacts, error)) error {
	a.updateCalled = true
	next, err := update(a.artifacts.Clone())
	if err != nil {
		return err
	}
	a.artifacts = next.Clone()
	return nil
}

func (a *updateTrackingAgent) Clone() (OptimizableAgent, error) {
	cloned := *a
	cloned.artifacts = a.artifacts.Clone()
	return &cloned, nil
}

func (a *updateTrackingAgent) ListOptimizationTargets() []OptimizationTargetDescriptor {
	return newTargetAwareMockAgent().ListOptimizationTargets()
}

func (a *updateTrackingAgent) OptimizationAgentType() string {
	return "mock"
}

type artifactScoringEvaluator struct{}

func (artifactScoringEvaluator) Evaluate(_ context.Context, agent OptimizableAgent, ex AgentExample) (*EvalResult, error) {
	artifacts := agent.GetArtifacts()
	score := 0.0

	if strings.Contains(strings.ToLower(artifacts.Text[ArtifactSkillPack]), "carefully") {
		score += 0.4
	}
	if strings.Contains(strings.ToLower(artifacts.Text[ArtifactToolPolicy]), "evidence-seeking") {
		score += 0.3
	}
	if artifacts.Int["max_turns"] == 12 {
		score += 0.3
	}

	trace := &agents.ExecutionTrace{
		AgentID:        "mock",
		AgentType:      "mock",
		Task:           ex.ID,
		Status:         agents.TraceStatusSuccess,
		StartedAt:      time.Now().Add(-10 * time.Millisecond),
		CompletedAt:    time.Now(),
		ProcessingTime: 10 * time.Millisecond,
		TokenUsage: map[string]int64{
			"total": 25,
		},
		ToolUsageCount: map[string]int{
			"lookup": 1,
		},
		ContextMetadata: map[string]interface{}{
			"artifact_skill_pack": artifacts.Text[ArtifactSkillPack],
		},
		TerminationCause: "score",
	}

	return &EvalResult{
		Score: score,
		SideInfo: &SideInfo{
			Trace: trace,
			Scores: map[string]float64{
				"artifact_score": score,
			},
			Diagnostics: map[string]interface{}{
				"skill_pack":  artifacts.Text[ArtifactSkillPack],
				"tool_policy": artifacts.Text[ArtifactToolPolicy],
				"max_turns":   artifacts.Int["max_turns"],
			},
		},
	}, nil
}

func TestOptimizedAgentProgram_RoundTripWithNamedTargets(t *testing.T) {
	agent := newTargetAwareMockAgent()

	program, err := ExportOptimizedAgentProgram(agent)
	require.NoError(t, err)
	require.NotNil(t, program)

	assert.Equal(t, optimizedAgentProgramSchemaV1, program.Schema)
	assert.Equal(t, optimizedAgentProgramVersionV1, program.Version)
	assert.Equal(t, "mock", program.AgentType)
	assert.Equal(t, "Use the repository debugging guide.", program.Text["root.system"])
	assert.Equal(t, 8, program.Int["root.max_turns"])

	tmpPath := filepath.Join(t.TempDir(), "optimized-agent-program.json")
	require.NoError(t, WriteOptimizedAgentProgram(tmpPath, program))

	loaded, err := ReadOptimizedAgentProgram(tmpPath)
	require.NoError(t, err)
	require.NotNil(t, loaded)

	agent.artifacts.Text[ArtifactSkillPack] = "stale prompt"
	agent.artifacts.Int["max_turns"] = 3
	require.NoError(t, ApplyOptimizedAgentProgram(agent, loaded))

	assert.Equal(t, "Use the repository debugging guide.", agent.artifacts.Text[ArtifactSkillPack])
	assert.Equal(t, "Prefer deterministic tools first.", agent.artifacts.Text[ArtifactToolPolicy])
	assert.Equal(t, 8, agent.artifacts.Int["max_turns"])
}

func TestApplyOptimizedAgentProgram_RejectsMismatchedAgentType(t *testing.T) {
	agent := newTargetAwareMockAgent()
	program, err := ExportOptimizedAgentProgram(agent)
	require.NoError(t, err)

	program.AgentType = "different-agent"
	err = ApplyOptimizedAgentProgram(agent, program)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not match")
}

func TestApplyOptimizedAgentProgram_RejectsUnsupportedSchemaAndVersion(t *testing.T) {
	agent := newTargetAwareMockAgent()
	program, err := ExportOptimizedAgentProgram(agent)
	require.NoError(t, err)

	program.Schema = "unsupported.schema"
	err = ApplyOptimizedAgentProgram(agent, program)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported optimized agent program schema")

	program.Schema = optimizedAgentProgramSchemaV1
	program.Version = optimizedAgentProgramVersionV1 + 1
	err = ApplyOptimizedAgentProgram(agent, program)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported optimized agent program version")
}

func TestNormalizeOptimizationTargetDescriptors_PreservesDeclaredOrder(t *testing.T) {
	descriptors := normalizeOptimizationTargetDescriptors([]OptimizationTargetDescriptor{
		{ID: "root.tool_policy", Kind: OptimizationTargetText, ArtifactKey: ArtifactToolPolicy},
		{ID: "root.system", Kind: OptimizationTargetText, ArtifactKey: ArtifactSkillPack},
		{ID: "root.max_turns", Kind: OptimizationTargetInt, IntKey: "max_turns"},
	})

	require.Len(t, descriptors, 3)
	assert.Equal(t, "root.tool_policy", descriptors[0].ID)
	assert.Equal(t, "root.system", descriptors[1].ID)
	assert.Equal(t, "root.max_turns", descriptors[2].ID)
}

func TestExportOptimizedAgentProgram_SkipsMissingArtifactKeys(t *testing.T) {
	agent := newTargetAwareMockAgent()
	delete(agent.artifacts.Text, ArtifactToolPolicy)
	delete(agent.artifacts.Int, "max_turns")

	program, err := ExportOptimizedAgentProgram(agent)
	require.NoError(t, err)

	assert.Equal(t, "Use the repository debugging guide.", program.Text["root.system"])
	_, hasToolPolicy := program.Text["root.tool_policy"]
	_, hasMaxTurns := program.Int["root.max_turns"]
	assert.False(t, hasToolPolicy)
	assert.False(t, hasMaxTurns)

	restoreTarget := newTargetAwareMockAgent()
	require.NoError(t, ApplyOptimizedAgentProgram(restoreTarget, program))
	assert.Equal(t, "Prefer deterministic tools first.", restoreTarget.artifacts.Text[ArtifactToolPolicy])
	assert.Equal(t, 8, restoreTarget.artifacts.Int["max_turns"])
}

func TestApplyOptimizedAgentProgram_IgnoresUnknownTargetIDs(t *testing.T) {
	agent := newTargetAwareMockAgent()
	program, err := ExportOptimizedAgentProgram(agent)
	require.NoError(t, err)

	program.Text["root.system"] = "Use the patched system prompt."
	program.Text["root.dropped"] = "obsolete setting"
	program.Int["root.unknown_int"] = 99
	program.Bool["root.unknown_bool"] = true

	err = ApplyOptimizedAgentProgram(agent, program)
	require.NoError(t, err)
	assert.Equal(t, "Use the patched system prompt.", agent.artifacts.Text[ArtifactSkillPack])
	assert.Equal(t, "Prefer deterministic tools first.", agent.artifacts.Text[ArtifactToolPolicy])
}

func TestApplyOptimizedAgentProgram_UsesAtomicArtifactUpdater(t *testing.T) {
	agent := &updateTrackingAgent{
		artifacts: AgentArtifacts{
			Text: map[ArtifactKey]string{
				ArtifactSkillPack:  "old system",
				ArtifactToolPolicy: "old tool policy",
			},
			Int: map[string]int{
				"max_turns": 4,
			},
			Bool: map[string]bool{},
		},
	}

	program := &OptimizedAgentProgram{
		Schema:    optimizedAgentProgramSchemaV1,
		Version:   optimizedAgentProgramVersionV1,
		AgentType: "mock",
		Text: map[string]string{
			"root.system": "new system",
		},
		Int: map[string]int{
			"root.max_turns": 9,
		},
	}

	require.NoError(t, ApplyOptimizedAgentProgram(agent, program))
	assert.True(t, agent.updateCalled)
	assert.False(t, agent.getCalled)
	assert.False(t, agent.setCalled)
	assert.Equal(t, "new system", agent.artifacts.Text[ArtifactSkillPack])
	assert.Equal(t, "old tool policy", agent.artifacts.Text[ArtifactToolPolicy])
	assert.Equal(t, 9, agent.artifacts.Int["max_turns"])
}

func TestRunGEPAWorkflow_BaselineOptimizeReplay(t *testing.T) {
	setupAgentMultiArtifactGEPAMockLLM(t)

	agent := newTargetAwareMockAgent()
	artifactPath := filepath.Join(t.TempDir(), "optimized-program.json")
	workflow, err := RunGEPAWorkflow(context.Background(), agent, GEPAWorkflowRequest{
		Evaluator: artifactScoringEvaluator{},
		TrainingExamples: []AgentExample{
			{ID: "train-1"},
		},
		ValidationExamples: []AgentExample{
			{ID: "validation-1"},
		},
		ReplayExamples: []AgentExample{
			{ID: "replay-1"},
		},
		PassThreshold: 0.9,
		Config: GEPAAdapterConfig{
			PopulationSize:  3,
			MaxGenerations:  1,
			ReflectionFreq:  1,
			EvalConcurrency: 1,
			PrimaryArtifact: ArtifactSkillPack,
			ArtifactKeys:    []ArtifactKey{ArtifactSkillPack, ArtifactToolPolicy},
			IntMutationPlans: map[string]IntMutationConfig{
				"max_turns": {Min: 8, Max: 12, Step: 4},
			},
			ValidationSplit: 0,
		},
		ApplyBest:    true,
		ArtifactPath: artifactPath,
	})
	require.NoError(t, err)
	require.NotNil(t, workflow)
	require.NotNil(t, workflow.BaselineRun)
	require.NotNil(t, workflow.Optimization)
	require.NotNil(t, workflow.OptimizedProgram)
	require.NotNil(t, workflow.ReplayRun)

	assert.Equal(t, "mock", workflow.OptimizedProgram.AgentType)
	assert.GreaterOrEqual(t, workflow.ReplayRun.AverageScore, workflow.BaselineRun.AverageScore)

	loadedProgram, err := ReadOptimizedAgentProgram(artifactPath)
	require.NoError(t, err)
	require.NotNil(t, loadedProgram)
	assert.Equal(t, workflow.OptimizedProgram.Schema, loadedProgram.Schema)
	assert.Equal(t, workflow.OptimizedProgram.Version, loadedProgram.Version)
	assert.Equal(t, workflow.OptimizedProgram.AgentType, loadedProgram.AgentType)
	assert.Equal(t, workflow.OptimizedProgram.TargetOrder, loadedProgram.TargetOrder)
	assert.Equal(t, workflow.OptimizedProgram.Text, loadedProgram.Text)
	assert.Equal(t, workflow.OptimizedProgram.Int, loadedProgram.Int)

	loadedArtifacts, err := loadedProgram.ToArtifacts(agent.ListOptimizationTargets())
	require.NoError(t, err)
	assert.Equal(t, workflow.Optimization.BestArtifacts.Text, loadedArtifacts.Text)
	assert.Equal(t, workflow.Optimization.BestArtifacts.Int, loadedArtifacts.Int)

	appliedArtifacts := agent.GetArtifacts()
	assert.Equal(t, workflow.Optimization.BestArtifacts.Text[ArtifactSkillPack], appliedArtifacts.Text[ArtifactSkillPack])
	assert.Equal(t, workflow.Optimization.BestArtifacts.Text[ArtifactToolPolicy], appliedArtifacts.Text[ArtifactToolPolicy])
	assert.Equal(t, workflow.Optimization.BestArtifacts.Int["max_turns"], appliedArtifacts.Int["max_turns"])

	replayAgent := newTargetAwareMockAgent()
	require.NoError(t, ApplyOptimizedAgentProgram(replayAgent, loadedProgram))

	harness := &Harness{
		Evaluator:     artifactScoringEvaluator{},
		PassThreshold: 0.9,
	}
	manualReplay, err := harness.Run(context.Background(), replayAgent, []AgentExample{{ID: "replay-1"}})
	require.NoError(t, err)
	assert.Equal(t, workflow.ReplayRun.AverageScore, manualReplay.AverageScore)
	assert.Equal(t, workflow.ReplayRun.PassedExamples, manualReplay.PassedExamples)
}
