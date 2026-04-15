package optimize

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// GEPAWorkflowRequest describes one baseline -> optimize -> save/restore ->
// replay optimization flow for an optimizable agent.
type GEPAWorkflowRequest struct {
	Evaluator          AgentEvaluator
	TrainingExamples   []AgentExample
	ValidationExamples []AgentExample
	BaselineExamples   []AgentExample
	ReplayExamples     []AgentExample
	PassThreshold      float64
	Config             GEPAAdapterConfig
	ProgressReporter   core.ProgressReporter
	ApplyBest          bool
	ArtifactPath       string
}

// GEPAWorkflowResult captures the user-facing outputs of a full optimization workflow.
type GEPAWorkflowResult struct {
	BaselineRun      *HarnessRunResult
	Optimization     *GEPAOptimizeResult
	OptimizedProgram *OptimizedAgentProgram
	ReplayRun        *HarnessRunResult
}

// RunGEPAWorkflow executes a full agent-optimization workflow using the
// existing GEPAAgentOptimizer control plane.
func RunGEPAWorkflow(ctx context.Context, baseAgent OptimizableAgent, req GEPAWorkflowRequest) (*GEPAWorkflowResult, error) {
	if baseAgent == nil {
		return nil, fmt.Errorf("optimize: nil base agent")
	}
	if req.Evaluator == nil {
		return nil, fmt.Errorf("optimize: workflow evaluator is required")
	}
	if len(req.TrainingExamples) == 0 {
		return nil, fmt.Errorf("optimize: workflow requires at least one training example")
	}

	harness := &Harness{
		Evaluator:     req.Evaluator,
		PassThreshold: req.PassThreshold,
	}

	baselineExamples := append([]AgentExample(nil), req.BaselineExamples...)
	if len(baselineExamples) == 0 {
		baselineExamples = append(baselineExamples, req.TrainingExamples...)
		baselineExamples = append(baselineExamples, req.ValidationExamples...)
	}

	var baselineRun *HarnessRunResult
	if len(baselineExamples) > 0 {
		baselineAgent, err := baseAgent.Clone()
		if err != nil {
			return nil, fmt.Errorf("optimize: clone baseline agent: %w", err)
		}
		baselineRun, err = harness.Run(ctx, baselineAgent, baselineExamples)
		if err != nil {
			return nil, fmt.Errorf("optimize: baseline harness run: %w", err)
		}
	}

	optimizer := NewGEPAAgentOptimizer(baseAgent, req.Evaluator, req.Config)
	optimization, err := optimizer.Optimize(ctx, GEPAOptimizeRequest{
		SeedArtifacts:      baseAgent.GetArtifacts(),
		TrainingExamples:   append([]AgentExample(nil), req.TrainingExamples...),
		ValidationExamples: append([]AgentExample(nil), req.ValidationExamples...),
		ProgressReporter:   req.ProgressReporter,
	})
	if err != nil {
		return nil, err
	}

	program, err := ExportOptimizedAgentProgramFromArtifacts(baseAgent, optimization.BestArtifacts)
	if err != nil {
		return nil, fmt.Errorf("optimize: export optimized agent program: %w", err)
	}
	if req.ArtifactPath != "" {
		if err := WriteOptimizedAgentProgram(req.ArtifactPath, program); err != nil {
			return nil, err
		}
	}

	if req.ApplyBest {
		if err := ApplyOptimizedAgentProgram(baseAgent, program); err != nil {
			return nil, fmt.Errorf("optimize: apply optimized program to base agent: %w", err)
		}
	}

	replayExamples := append([]AgentExample(nil), req.ReplayExamples...)
	if len(replayExamples) == 0 {
		replayExamples = baselineExamples
	}

	var replayRun *HarnessRunResult
	if len(replayExamples) > 0 {
		replayAgent, err := baseAgent.Clone()
		if err != nil {
			return nil, fmt.Errorf("optimize: clone replay agent: %w", err)
		}
		if err := ApplyOptimizedAgentProgram(replayAgent, program); err != nil {
			return nil, fmt.Errorf("optimize: apply optimized program to replay agent: %w", err)
		}
		replayRun, err = harness.Run(ctx, replayAgent, replayExamples)
		if err != nil {
			return nil, fmt.Errorf("optimize: replay harness run: %w", err)
		}
	}

	return &GEPAWorkflowResult{
		BaselineRun:      baselineRun,
		Optimization:     optimization,
		OptimizedProgram: program,
		ReplayRun:        replayRun,
	}, nil
}
