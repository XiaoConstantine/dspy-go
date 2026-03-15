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

	engine, err := optimizers.NewGEPA(o.buildEngineConfig(len(trainExamples)))
	if err != nil {
		return nil, fmt.Errorf("optimize: create GEPA engine: %w", err)
	}
	if req.ProgressReporter != nil {
		engine.SetProgressReporter(req.ProgressReporter)
	}

	program, err := o.buildOptimizationProgram(req.SeedArtifacts, trainExamples)
	if err != nil {
		return nil, err
	}

	dataset := o.buildOptimizationDataset(trainExamples)
	optimizedProgram, err := engine.Compile(ctx, program, dataset, o.buildOptimizationMetric())
	if err != nil {
		return nil, err
	}

	state := engine.GetOptimizationState()
	bestCandidate := state.BestCandidate
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
	bestCandidate := bestCandidateFromState(state)
	if bestCandidate == nil {
		return nil, nil, fmt.Errorf("optimize: no GEPA best candidate available for validation evaluation")
	}

	evaluation, err := o.evaluateCandidateWithBase(ctx, optimizers.CloneCandidate(bestCandidate), seed, validationExamples)
	if err != nil {
		return nil, nil, fmt.Errorf("optimize: evaluate GEPA best candidate on validation examples: %w", err)
	}

	return bestCandidate, evaluation, nil
}

func bestCandidateFromState(state *optimizers.GEPAState) *optimizers.GEPACandidate {
	if state == nil {
		return nil
	}

	if state.BestCandidate != nil {
		return optimizers.CloneCandidate(state.BestCandidate)
	}

	var bestArchiveCandidate *optimizers.GEPACandidate
	bestArchiveFitness := math.Inf(-1)
	for _, candidate := range state.ParetoArchive {
		if candidate == nil {
			continue
		}

		fitness := candidate.Fitness
		if archiveFitness, exists := state.ArchiveFitnessMap[candidate.ID]; exists && archiveFitness != nil {
			fitness = archiveFitness.WeightedScore
		}

		if bestArchiveCandidate == nil || fitness > bestArchiveFitness || (fitness == bestArchiveFitness && candidate.ID < bestArchiveCandidate.ID) {
			bestArchiveCandidate = candidate
			bestArchiveFitness = fitness
		}
	}
	if bestArchiveCandidate != nil {
		return optimizers.CloneCandidate(bestArchiveCandidate)
	}

	return nil
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

func failedCandidateEvaluation(candidate *optimizers.GEPACandidate, examples []AgentExample, err error) *GEPACandidateEvaluation {
	run := &HarnessRunResult{
		Results:           make([]HarnessExampleResult, 0, len(examples)),
		AverageScore:      0,
		PassedExamples:    0,
		FailedExamples:    len(examples),
		CompletedExamples: len(examples),
		EvaluationErrors:  len(examples),
	}

	failureResult := evaluationFailureResult(err)
	traces := make([]optimizers.ExecutionTrace, 0, len(examples))
	for _, example := range examples {
		run.Results = append(run.Results, HarnessExampleResult{
			ExampleID: example.ID,
			Result:    failureResult,
		})
		traces = append(traces, buildGEPATrace(candidate, example, failureResult, 1.0))
	}

	zeroFitness := &optimizers.MultiObjectiveFitness{}
	if candidate != nil {
		candidate.Fitness = 0
	}

	return &GEPACandidateEvaluation{
		Candidate:    candidate,
		Run:          run,
		Fitness:      zeroFitness,
		Traces:       traces,
		AverageScore: 0,
	}
}
