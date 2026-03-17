package optimize

import (
	"context"
	"fmt"
	"math"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

// GEPAOptimizeRequest configures one end-to-end GEPA optimization run for an agent artifact set.
//
// SeedArtifacts and examples are treated as trusted harness inputs. They may be
// embedded into model prompts during optimization, so callers should source them
// from trusted corpora or explicitly sanitize them before invoking Optimize.
type GEPAOptimizeRequest struct {
	SeedArtifacts      AgentArtifacts
	TrainingExamples   []AgentExample
	ValidationExamples []AgentExample
	ProgressReporter   core.ProgressReporter
}

// GEPAOptimizeResult captures the best candidate and resulting artifacts from a GEPA run.
type GEPAOptimizeResult struct {
	BestCandidate            *optimizers.GEPACandidate
	BestArtifacts            AgentArtifacts
	BestValidationEvaluation *GEPACandidateEvaluation
	TrainingExampleCount     int
	ValidationExampleCount   int
	OptimizationState        *optimizers.GEPAState
}

// Optimize runs GEPA against agent artifacts using the mainline whole-program engine.
func (o *GEPAAgentOptimizer) Optimize(ctx context.Context, req GEPAOptimizeRequest) (*GEPAOptimizeResult, error) {
	if o == nil {
		return nil, fmt.Errorf("optimize: nil GEPA agent optimizer")
	}
	if o.evaluator == nil {
		return nil, fmt.Errorf("optimize: nil agent evaluator")
	}

	trainExamples, validationExamples, err := o.partitionExamples(req)
	if err != nil {
		return nil, err
	}

	engineConfig := o.buildEngineConfig(len(trainExamples))
	if len(validationExamples) > 0 {
		engineConfig.ValidationExamples = core.DatasetToSlice(o.buildOptimizationDataset(validationExamples))
	}

	engine, err := optimizers.NewGEPA(engineConfig)
	if err != nil {
		return nil, fmt.Errorf("optimize: create GEPA engine: %w", err)
	}
	if req.ProgressReporter != nil {
		engine.SetProgressReporter(req.ProgressReporter)
	}

	allExamples := append(append([]AgentExample(nil), trainExamples...), validationExamples...)

	program, err := o.buildOptimizationProgram(req.SeedArtifacts, allExamples)
	if err != nil {
		return nil, err
	}

	dataset := o.buildOptimizationDataset(trainExamples)
	optimizedProgram, err := engine.Compile(ctx, program, dataset, o.buildOptimizationMetric())
	if err != nil {
		return nil, err
	}

	state := engine.GetOptimizationState()
	bestCandidate, _, _ := state.BestCandidateForApplication()
	var bestValidation *GEPACandidateEvaluation
	if len(validationExamples) > 0 {
		bestCandidate, bestValidation, err = o.evaluateBestCandidateOnValidation(ctx, state, req.SeedArtifacts, validationExamples)
		if err != nil {
			return nil, err
		}
	}

	bestArtifacts, err := o.resolveBestArtifacts(bestCandidate, optimizedProgram, req.SeedArtifacts)
	if err != nil {
		return nil, err
	}

	return &GEPAOptimizeResult{
		BestCandidate:            bestCandidate,
		BestArtifacts:            bestArtifacts,
		BestValidationEvaluation: bestValidation,
		TrainingExampleCount:     len(trainExamples),
		ValidationExampleCount:   len(validationExamples),
		OptimizationState:        state,
	}, nil
}

func (o *GEPAAgentOptimizer) buildEngineConfig(trainingCount int) *optimizers.GEPAConfig {
	engineConfig := optimizers.DefaultGEPAConfig()
	engineConfig.PopulationSize = o.config.PopulationSize
	engineConfig.MaxGenerations = o.config.MaxGenerations
	engineConfig.ReflectionFreq = o.config.ReflectionFreq
	engineConfig.ConcurrencyLevel = o.config.EvalConcurrency
	engineConfig.StagnationLimit = o.config.StagnationLimit
	if trainingCount > 0 {
		engineConfig.EvaluationBatchSize = trainingCount
		if o.config.SearchBatchSize > 0 && o.config.SearchBatchSize < engineConfig.EvaluationBatchSize {
			engineConfig.EvaluationBatchSize = o.config.SearchBatchSize
		}
	}
	return engineConfig
}

func (o *GEPAAgentOptimizer) partitionExamples(req GEPAOptimizeRequest) ([]AgentExample, []AgentExample, error) {
	train := append([]AgentExample(nil), req.TrainingExamples...)
	validation := append([]AgentExample(nil), req.ValidationExamples...)

	if len(train) == 0 {
		return nil, nil, fmt.Errorf("optimize: GEPA optimization requires at least one training example")
	}

	if len(validation) == 0 && o.config.ValidationSplit > 0 && len(train) > 1 {
		validationCount := int(math.Ceil(float64(len(train)) * o.config.ValidationSplit))
		if validationCount >= len(train) {
			validationCount = len(train) - 1
		}
		if validationCount > 0 {
			split := len(train) - validationCount
			validation = append(validation, train[split:]...)
			train = append([]AgentExample(nil), train[:split]...)
		}
	}

	if len(train) == 0 {
		return nil, nil, fmt.Errorf("optimize: validation split consumed all training examples")
	}

	return train, validation, nil
}

func (o *GEPAAgentOptimizer) evaluateBestCandidateOnValidation(ctx context.Context, state *optimizers.GEPAState, seed AgentArtifacts, validationExamples []AgentExample) (*optimizers.GEPACandidate, *GEPACandidateEvaluation, error) {
	bestCandidate, _, _ := state.BestCandidateForApplication()
	if bestCandidate == nil {
		return nil, nil, fmt.Errorf("optimize: no GEPA best candidate available for validation evaluation")
	}

	if cachedEvaluation, err := o.cachedValidationEvaluation(state, bestCandidate, seed, validationExamples); err != nil {
		return nil, nil, fmt.Errorf("optimize: decode cached GEPA validation evaluation: %w", err)
	} else if cachedEvaluation != nil {
		return bestCandidate, cachedEvaluation, nil
	}

	evaluation, err := o.evaluateCandidateWithBase(ctx, optimizers.CloneCandidate(bestCandidate), seed, validationExamples)
	if err != nil {
		return nil, nil, fmt.Errorf("optimize: evaluate GEPA best candidate on validation examples: %w", err)
	}

	return bestCandidate, evaluation, nil
}

func (o *GEPAAgentOptimizer) cachedValidationEvaluation(state *optimizers.GEPAState, candidate *optimizers.GEPACandidate, seed AgentArtifacts, validationExamples []AgentExample) (*GEPACandidateEvaluation, error) {
	if o == nil || state == nil || candidate == nil {
		return nil, nil
	}

	cachedEvaluation := state.GetCandidateValidationEvaluation(candidate.ID)
	if cachedEvaluation == nil {
		return nil, nil
	}

	artifacts, err := o.candidateArtifactsWithBase(candidate, seed)
	if err != nil {
		return nil, err
	}

	run := &HarnessRunResult{
		Results: make([]HarnessExampleResult, 0, len(cachedEvaluation.Cases)),
	}
	traces := make([]optimizers.ExecutionTrace, 0, len(cachedEvaluation.Cases))
	for idx, evalCase := range cachedEvaluation.Cases {
		example := validationAgentExample(validationExamples, idx, evalCase.Example)
		evalResult := cachedValidationEvalResult(evalCase.Score, evalCase.Err, evalCase.Feedback, evalCase.FeedbackTarget, evalCase.FeedbackMetadata)

		run.Results = append(run.Results, HarnessExampleResult{
			ExampleID: example.ID,
			Result:    evalResult,
		})
		run.CompletedExamples++
		run.AverageScore += evalResult.Score
		if evalResult.Score >= o.config.PassThreshold {
			run.PassedExamples++
		} else {
			run.FailedExamples++
		}
		if evalCase.Err != nil {
			run.EvaluationErrors++
		}
		traces = append(traces, buildGEPATrace(candidate, example, evalResult, o.config.PassThreshold))
	}
	if run.CompletedExamples > 0 {
		run.AverageScore /= float64(run.CompletedExamples)
	}

	fitness := buildMultiObjectiveFitness(run)
	evaluatedCandidate := optimizers.CloneCandidate(candidate)
	evaluatedCandidate.Fitness = fitness.WeightedScore
	return &GEPACandidateEvaluation{
		Candidate:    evaluatedCandidate,
		Artifacts:    artifacts,
		Run:          run,
		Fitness:      fitness,
		Traces:       traces,
		AverageScore: cachedEvaluation.AverageScore,
	}, nil
}

func (o *GEPAAgentOptimizer) resolveBestArtifacts(bestCandidate *optimizers.GEPACandidate, optimizedProgram core.Program, seed AgentArtifacts) (AgentArtifacts, error) {
	if bestCandidate != nil {
		return o.candidateArtifactsWithBase(bestCandidate, seed)
	}

	if len(optimizedProgram.Modules) == 0 {
		return AgentArtifacts{}, fmt.Errorf("optimize: no optimized GEPA artifacts available")
	}

	artifacts := seed.Clone()
	if artifacts.Text == nil {
		artifacts.Text = make(map[ArtifactKey]string)
	}
	specs, err := o.buildArtifactProgramSpecs(seed)
	if err != nil {
		return AgentArtifacts{}, err
	}
	for _, spec := range specs {
		module, exists := optimizedProgram.Modules[spec.moduleName]
		if !exists {
			continue
		}
		instruction := module.GetSignature().Instruction
		switch spec.kind {
		case artifactProgramKindText:
			artifacts.Text[spec.textKey] = instruction
		case artifactProgramKindInt:
			artifacts.Int[spec.intKey] = parseIntArtifactInstruction(spec.intKey, instruction, artifacts.Int[spec.intKey], spec.intPlan)
		}
	}
	return artifacts, nil
}

func validationAgentExample(validationExamples []AgentExample, idx int, example core.Example) AgentExample {
	if idx >= 0 && idx < len(validationExamples) {
		return cloneAgentExample(validationExamples[idx])
	}

	return AgentExample{
		ID:      fmt.Sprintf("validation-case-%d", idx),
		Inputs:  core.ShallowCopyMap(example.Inputs),
		Outputs: core.ShallowCopyMap(example.Outputs),
	}
}

func cachedValidationEvalResult(score float64, err error, feedback, feedbackTarget string, feedbackMetadata map[string]interface{}) *EvalResult {
	diagnostics := make(map[string]interface{})
	if err != nil {
		diagnostics["evaluation_error"] = err.Error()
	}
	if feedback != "" {
		diagnostics["gepa_feedback"] = feedback
	}
	if feedbackTarget != "" {
		diagnostics["gepa_feedback_target"] = feedbackTarget
	}
	for key, value := range feedbackMetadata {
		diagnostics[key] = value
	}

	return &EvalResult{
		Score: score,
		SideInfo: &SideInfo{
			Diagnostics: diagnostics,
			Scores: map[string]float64{
				"cached_validation_score": score,
			},
		},
	}
}
