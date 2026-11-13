package optimizers

import (
	"context"
	"fmt"
	"maps"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/sourcegraph/conc/pool"
)

type BootstrapFewShot struct {
	Metric          func(example map[string]any, prediction map[string]any, ctx context.Context) bool
	MaxBootstrapped int
	// MetricThreshold is the score a prediction needs in CompileWithScorer to
	// become a demonstration. Zero means unset and keeps every prediction that
	// scores above zero, so a threshold of exactly zero cannot be expressed; a
	// negative threshold also keeps predictions that score zero.
	MetricThreshold float64
}

var _ core.ScorerCompiler = (*BootstrapFewShot)(nil)

func NewBootstrapFewShot(metric func(example map[string]any, prediction map[string]any, ctx context.Context) bool, maxBootstrapped int) *BootstrapFewShot {
	return &BootstrapFewShot{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
	}
}

// Compile implements the core.Optimizer interface.
func (b *BootstrapFewShot) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	// Convert core.Dataset to trainset format
	var trainset []map[string]any
	dataset.Reset()
	for {
		example, hasNext := dataset.Next()
		if !hasNext {
			break
		}
		// Combine inputs and outputs into a single map for compatibility
		trainExample := make(map[string]any)
		for k, v := range example.Inputs {
			trainExample[k] = v
		}
		for k, v := range example.Outputs {
			trainExample[k] = v
		}
		trainset = append(trainset, trainExample)
	}

	// Use program as both student and teacher (self-improvement)
	return b.compileInternal(ctx, program, program, trainset, b.metricKeeps(trainset))
}

// CompileWithScorer bootstraps demonstrations from materialized, read-only
// examples, using program as its own teacher. A teacher prediction becomes a
// demonstration when scorer scores it at MetricThreshold or above (above zero
// when the threshold is unset); one the scorer cannot score is not kept. An
// example the teacher fails on is skipped, and compiling fails only when
// every teacher call fails. The program must have exactly one predictor, whose
// existing demonstrations are replaced. It
// calls scorer from up to core.GetConcurrencyLevel goroutines at once and
// ignores the constructor metric.
func (b *BootstrapFewShot) CompileWithScorer(ctx context.Context, program core.Program, examples []core.Example, scorer core.Scorer) (core.Program, error) {
	if scorer == nil {
		return program, errors.New(errors.InvalidInput, "BootstrapFewShot requires a scorer")
	}
	trainset := make([]map[string]any, len(examples))
	for i, example := range examples {
		trainset[i] = make(map[string]any, len(example.Inputs)+len(example.Outputs))
		maps.Copy(trainset[i], example.Inputs)
		maps.Copy(trainset[i], example.Outputs)
	}
	keep := func(ctx context.Context, index int, prediction map[string]any) bool {
		score, err := scorer.Score(ctx, examples[index], prediction)
		if err != nil {
			logging.GetLogger().Warn(ctx, "BootstrapFewShot: scorer failed: %v", err)
			return false
		}
		if b.MetricThreshold == 0 {
			return score > 0
		}
		return score >= b.MetricThreshold
	}
	return b.compileInternal(ctx, program, program, trainset, keep)
}

// CompileLegacy provides backward compatibility for the old interface.
func (b *BootstrapFewShot) CompileLegacy(ctx context.Context, student, teacher core.Program, trainset []map[string]any) (core.Program, error) {
	return b.compileInternal(ctx, student, teacher, trainset, b.metricKeeps(trainset))
}

// metricKeeps decides with the constructor metric.
func (b *BootstrapFewShot) metricKeeps(trainset []map[string]any) func(context.Context, int, map[string]any) bool {
	return func(ctx context.Context, index int, prediction map[string]any) bool {
		return b.Metric(trainset[index], prediction, ctx)
	}
}

// compileInternal contains the original implementation logic. keep reports
// whether the teacher's prediction for trainset[index] becomes a demonstration.
func (b *BootstrapFewShot) compileInternal(ctx context.Context, student, teacher core.Program, trainset []map[string]any, keep func(ctx context.Context, index int, prediction map[string]any) bool) (core.Program, error) {
	for _, program := range []core.Program{student, teacher} {
		if err := requireRebindableProgram(program); err != nil {
			return student, err
		}
	}
	compiledStudent := student.Clone()
	// Start from a student without demonstrations, so compiling an already
	// compiled program bootstraps again instead of counting its old demos. A
	// program-level demonstration fits only one predictor's fields, so more
	// than one predictor is rejected rather than given another's demos.
	predictorCount := 0
	for _, module := range compiledStudent.Modules {
		for _, predictor := range demoPredictors(module) {
			predictor.SetDemos(nil)
			predictorCount++
		}
	}
	if predictorCount > 1 {
		return student, errors.New(errors.InvalidInput, fmt.Sprintf("BootstrapFewShot supports programs with one predictor, got %d: a demonstration of the whole program does not match each predictor's fields", predictorCount))
	}
	// Modules keep their own LLMs unless a teacher LLM is configured.
	teacherLLM := core.GetTeacherLLM()
	if ctx == nil {
		ctx = context.Background()
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}

	ctx = core.WithExecutionState(ctx)
	ctx, span := core.StartSpan(ctx, "Compilation")

	defer core.EndSpan(ctx)

	// Walk the trainset in order, one batch of concurrent teacher calls at a
	// time, until enough predictions are kept. Stopping at the first
	// MaxBootstrapped examples would return too few demonstrations whenever
	// some of them fail the metric.
	var results []core.Example
	var teacherErr error
	teacherSucceeded := false
	batchSize := max(core.GetConcurrencyLevel(), 1)
	for start := 0; start < len(trainset) && len(results) < b.MaxBootstrapped; start += batchSize {
		if b.enoughBootstrappedDemos(compiledStudent) {
			logging.GetLogger().Info(ctx, "Enough bootstrapped demos, breaking loop")
			break
		}
		batch := trainset[start:min(start+batchSize, len(trainset))]
		kept := make([]*core.Example, len(batch))
		errs := make([]error, len(batch))
		p := pool.New().WithMaxGoroutines(batchSize)
		for i, ex := range batch {
			p.Go(func() {
				exampleCtx, exampleSpan := core.StartSpan(ctx, "Example")
				defer core.EndSpan(exampleCtx)

				exampleSpan.WithAnnotation("Example", ex)
				prediction, err := b.predictWithTeacher(exampleCtx, teacher, teacherLLM, ex)
				if err != nil {
					exampleSpan.WithError(err)
					errs[i] = err
					return
				}
				exampleSpan.WithAnnotation("prediction", prediction)

				if keep(exampleCtx, start+i, prediction) {
					kept[i] = &core.Example{Inputs: ex, Outputs: prediction}
				}
			})
		}
		p.Wait()

		// A cancelled run has scored only part of the trainset; report it instead
		// of returning the demonstrations that happened to finish.
		if err := ctx.Err(); err != nil {
			span.WithError(err)
			return compiledStudent, fmt.Errorf("error during compilation: %w", err)
		}
		// A failed teacher call yields no demonstration; the walk continues.
		for i, err := range errs {
			if err == nil {
				teacherSucceeded = true
				continue
			}
			logging.GetLogger().Warn(ctx, "BootstrapFewShot: teacher failed on example %d: %v", start+i, err)
			if teacherErr == nil {
				teacherErr = err
			}
		}
		for _, example := range kept {
			if example != nil && len(results) < b.MaxBootstrapped {
				results = append(results, *example)
			}
		}
	}
	// A teacher that never works is a broken setup, not a hard example.
	if teacherErr != nil && !teacherSucceeded {
		span.WithError(teacherErr)
		return compiledStudent, fmt.Errorf("error during compilation: every teacher call failed: %w", teacherErr)
	}

	for _, result := range results {
		if err := b.addDemonstrations(compiledStudent, result, ctx); err != nil {
			span.WithError(err)
			return compiledStudent, fmt.Errorf("error adding demonstrations: %w", err)
		}
		if b.enoughBootstrappedDemos(compiledStudent) {
			break
		}
	}

	span.WithAnnotation("compiledStudent", compiledStudent)
	return compiledStudent, nil
}

func (b *BootstrapFewShot) predictWithTeacher(ctx context.Context, teacher core.Program, teacherLLM core.LLM, example map[string]any) (map[string]any, error) {
	// Clone the teacher program; a configured teacher LLM replaces its modules' LLMs.
	teacherClone := teacher.Clone()
	if teacherLLM != nil {
		for _, module := range teacherClone.Modules {
			if predictor, ok := module.(interface{ SetLLM(core.LLM) }); ok {
				predictor.SetLLM(teacherLLM)
			}
		}
	}
	if ctx == nil {
		ctx = context.Background()
	}

	ctx = core.WithExecutionState(ctx)
	ctx, span := core.StartSpan(ctx, "TeacherPrediction")
	defer core.EndSpan(ctx)

	span.WithAnnotation("Example", example)
	outputs, err := teacherClone.Execute(ctx, example)
	if err != nil {
		span.WithError(err)
		return nil, err
	}

	span.WithAnnotation("outputs", outputs)
	return outputs, nil

}

// demoPredictors returns the Predict modules that hold module's demonstrations,
// looking inside composites such as ChainOfThought.
func demoPredictors(module core.Module) []*modules.Predict {
	if predictor, ok := module.(*modules.Predict); ok {
		return []*modules.Predict{predictor}
	}
	composite, ok := module.(interface{ GetSubModules() []core.Module })
	if !ok {
		return nil
	}
	var predictors []*modules.Predict
	for _, sub := range composite.GetSubModules() {
		predictors = append(predictors, demoPredictors(sub)...)
	}
	return predictors
}

func (b *BootstrapFewShot) enoughBootstrappedDemos(program core.Program) bool {
	for _, module := range program.Modules {
		for _, predictor := range demoPredictors(module) {
			if len(predictor.GetDemos()) < b.MaxBootstrapped {
				return false
			}
		}
	}
	return true
}

func (b *BootstrapFewShot) addDemonstrations(program core.Program, demo core.Example, ctx context.Context) error {
	if ctx == nil {
		return errors.New(errors.InvalidInput, "cannot add demonstrations: context is nil")
	}

	ctx, span := core.StartSpan(ctx, "AddDemonstrations")
	defer core.EndSpan(ctx)
	span.WithAnnotation("demo_inputs", demo.Inputs)
	span.WithAnnotation("demo_outputs", demo.Outputs)
	for moduleName, module := range program.Modules {
		for _, predictor := range demoPredictors(module) {
			currentDemos := predictor.GetDemos()
			if len(currentDemos) >= b.MaxBootstrapped {
				span.WithAnnotation("skipped_module", moduleName)
				span.WithAnnotation("reason", "max_demos_reached")
				return errors.WithFields(
					errors.New(errors.ResourceExhausted, fmt.Sprintf("max demonstrations reached for module %s", moduleName)),
					errors.Fields{
						"module":        moduleName,
						"max_demos":     b.MaxBootstrapped,
						"current_demos": len(currentDemos),
					})
			}
			newDemos := append(currentDemos, demo)
			predictor.SetDemos(newDemos)
			span.WithAnnotation("added_to_module", moduleName)
			span.WithAnnotation("total_demos", len(newDemos))
			return nil
		}
	}
	return errors.New(errors.ResourceNotFound, "no suitable module found for adding demonstrations")
}
