package optimize

import (
	"context"
	"fmt"
	"math"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/sourcegraph/conc/pool"
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

// Optimize runs the shared GEPA evolution loop against agent artifacts using the adapter bridge.
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

	engine, err := optimizers.NewGEPA(o.buildEngineConfig())
	if err != nil {
		return nil, fmt.Errorf("optimize: create GEPA engine: %w", err)
	}

	seedCandidate, err := o.SeedCandidate(req.SeedArtifacts)
	if err != nil {
		return nil, err
	}
	if err := engine.BootstrapPopulationFromSeed(ctx, seedCandidate); err != nil {
		return nil, fmt.Errorf("optimize: bootstrap GEPA population: %w", err)
	}

	if err := optimizers.RunEvolutionLoop(ctx, optimizers.EvolutionLoopConfig{
		MaxGenerations: o.config.MaxGenerations,
		ReflectionFreq: o.config.ReflectionFreq,
		PhaseName:      "GEPA Agent Optimization",
	}, optimizers.EvolutionLoopHooks{
		BeforeGeneration: func(ctx context.Context, generation int) error {
			engine.GetOptimizationState().CurrentGeneration = generation
			return nil
		},
		EvaluateGeneration: func(ctx context.Context, generation int) error {
			return o.evaluateCurrentPopulation(ctx, engine, trainExamples)
		},
		AfterEvaluation: func(ctx context.Context, generation int) error {
			population := engine.CurrentPopulation()
			if population != nil {
				engine.GetOptimizationState().UpdateParetoArchive(population.Candidates, engine.GetOptimizationState().MultiObjectiveFitnessMap)
			}
			return nil
		},
		Reflect: func(ctx context.Context, generation int) error {
			return engine.ReflectCurrentPopulation(ctx, generation)
		},
		HasConverged: func(ctx context.Context, generation int) bool {
			return engine.HasConverged()
		},
		Evolve: func(ctx context.Context, generation int) error {
			return engine.EvolveCurrentPopulation(ctx)
		},
		ReportProgress: func(phase string, current, total int) {
			if req.ProgressReporter != nil {
				req.ProgressReporter.Report(phase, current, total)
			}
		},
	}); err != nil {
		return nil, err
	}

	bestCandidate := engine.GetOptimizationState().BestCandidate
	var bestValidation *GEPACandidateEvaluation
	if len(validationExamples) > 0 {
		bestCandidate, bestValidation, err = o.selectBestCandidate(ctx, engine.GetOptimizationState(), validationExamples)
		if err != nil {
			return nil, err
		}
	}

	if bestCandidate == nil {
		return nil, fmt.Errorf("optimize: no best GEPA candidate found")
	}

	bestArtifacts, err := o.CandidateArtifacts(bestCandidate)
	if err != nil {
		return nil, err
	}

	return &GEPAOptimizeResult{
		BestCandidate:            bestCandidate,
		BestArtifacts:            bestArtifacts,
		BestValidationEvaluation: bestValidation,
		TrainingExampleCount:     len(trainExamples),
		ValidationExampleCount:   len(validationExamples),
		OptimizationState:        engine.GetOptimizationState(),
	}, nil
}

func (o *GEPAAgentOptimizer) buildEngineConfig() *optimizers.GEPAConfig {
	engineConfig := optimizers.DefaultGEPAConfig()
	engineConfig.PopulationSize = o.config.PopulationSize
	engineConfig.MaxGenerations = o.config.MaxGenerations
	engineConfig.ReflectionFreq = o.config.ReflectionFreq
	engineConfig.ConcurrencyLevel = o.config.EvalConcurrency
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

func (o *GEPAAgentOptimizer) evaluateCurrentPopulation(ctx context.Context, engine *optimizers.GEPA, examples []AgentExample) error {
	population := engine.CurrentPopulation()
	if population == nil {
		return fmt.Errorf("optimize: no current GEPA population")
	}

	evaluations, err := o.evaluatePopulationCandidates(ctx, population.Candidates, examples)
	if err != nil {
		return err
	}

	state := engine.GetOptimizationState()
	fitnessMap := make(map[string]*optimizers.MultiObjectiveFitness, len(evaluations))

	var generationBest *optimizers.GEPACandidate
	bestFitness := -1.0

	for _, evaluation := range evaluations {
		if evaluation == nil || evaluation.Candidate == nil || evaluation.Fitness == nil {
			continue
		}

		fitnessMap[evaluation.Candidate.ID] = evaluation.Fitness
		state.RecordCandidateFitness(evaluation.Candidate, evaluation.Fitness, evaluation.AverageScore)
		for _, trace := range evaluation.Traces {
			trace := trace
			state.AddTrace(&trace)
		}

		if evaluation.Candidate.Fitness > bestFitness {
			bestFitness = evaluation.Candidate.Fitness
			generationBest = evaluation.Candidate
		}
	}

	population.BestCandidate = generationBest
	if generationBest != nil {
		population.BestFitness = generationBest.Fitness
		engine.UpdateBestCandidate(generationBest)
	}

	engine.SetCurrentMultiObjectiveFitnessMap(fitnessMap)
	return nil
}

func (o *GEPAAgentOptimizer) evaluatePopulationCandidates(ctx context.Context, candidates []*optimizers.GEPACandidate, examples []AgentExample) ([]*GEPACandidateEvaluation, error) {
	evaluations := make([]*GEPACandidateEvaluation, len(candidates))
	evaluationPool := pool.New().WithContext(ctx).WithMaxGoroutines(o.config.EvalConcurrency)

	for idx, candidate := range candidates {
		idx := idx
		candidate := candidate
		evaluationPool.Go(func(ctx context.Context) error {
			if err := ctx.Err(); err != nil {
				return err
			}

			evaluation, err := o.EvaluateCandidate(ctx, candidate, examples)
			if err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				evaluations[idx] = failedCandidateEvaluation(candidate, examples, err)
				return nil
			}
			evaluations[idx] = evaluation
			return nil
		})
	}

	if err := evaluationPool.Wait(); err != nil {
		return nil, err
	}

	return evaluations, nil
}

func (o *GEPAAgentOptimizer) selectBestCandidate(ctx context.Context, state *optimizers.GEPAState, validationExamples []AgentExample) (*optimizers.GEPACandidate, *GEPACandidateEvaluation, error) {
	candidates := state.ParetoArchive
	if len(candidates) == 0 && state.BestCandidate != nil {
		candidates = []*optimizers.GEPACandidate{state.BestCandidate}
	}
	if len(candidates) == 0 {
		return nil, nil, fmt.Errorf("optimize: no GEPA candidates available for final selection")
	}

	candidateSet := dedupeCandidates(candidates)
	validationCandidates := make([]*optimizers.GEPACandidate, 0, len(candidateSet))
	originalByID := make(map[string]*optimizers.GEPACandidate, len(candidateSet))
	for _, candidate := range candidateSet {
		validationCandidates = append(validationCandidates, optimizers.CloneCandidate(candidate))
		originalByID[candidate.ID] = candidate
	}

	evaluations, err := o.evaluatePopulationCandidates(ctx, validationCandidates, validationExamples)
	if err != nil {
		return nil, nil, fmt.Errorf("optimize: evaluate GEPA validation candidates: %w", err)
	}

	var bestCandidate *optimizers.GEPACandidate
	var bestEvaluation *GEPACandidateEvaluation
	for _, evaluation := range evaluations {
		if evaluation == nil || evaluation.Candidate == nil || evaluation.Fitness == nil {
			continue
		}
		if bestEvaluation == nil || evaluation.Fitness.WeightedScore > bestEvaluation.Fitness.WeightedScore {
			bestCandidate = originalByID[evaluation.Candidate.ID]
			bestEvaluation = evaluation
		}
	}

	if bestCandidate == nil || bestEvaluation == nil {
		return nil, nil, fmt.Errorf("optimize: no successful GEPA validation evaluation found")
	}

	return bestCandidate, bestEvaluation, nil
}
func dedupeCandidates(candidates []*optimizers.GEPACandidate) []*optimizers.GEPACandidate {
	seen := make(map[string]struct{}, len(candidates))
	deduped := make([]*optimizers.GEPACandidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}
		if _, exists := seen[candidate.ID]; exists {
			continue
		}
		seen[candidate.ID] = struct{}{}
		deduped = append(deduped, candidate)
	}
	return deduped
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
	candidate.Fitness = 0

	return &GEPACandidateEvaluation{
		Candidate:    candidate,
		Run:          run,
		Fitness:      zeroFitness,
		Traces:       traces,
		AverageScore: 0,
	}
}
