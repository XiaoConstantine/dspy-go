package optimizers

import (
	"context"
	"fmt"
	"maps"
	"sync"
	"sync/atomic"

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
	// become a demonstration. Zero keeps every prediction that scores above zero.
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
// when the threshold is unset); one the scorer cannot score is not kept. It
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
	compiledStudent := student.Clone()
	teacherLLM := core.GetTeacherLLM()
	if teacherLLM == nil {
		teacherLLM = core.GetDefaultLLM()
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}

	ctx = core.WithExecutionState(ctx)
	ctx, span := core.StartSpan(ctx, "Compilation")

	defer core.EndSpan(ctx)

	var (
		resultsMu sync.Mutex
		results   []core.Example
		processed atomic.Int32
		errCh     = make(chan error, 1)
	)
	examplesNeeded := b.MaxBootstrapped
	if examplesNeeded > len(trainset) {
		examplesNeeded = len(trainset)
	}

	p := pool.New().WithMaxGoroutines(core.GetConcurrencyLevel())

	for i := 0; i < examplesNeeded; i++ {
		if b.enoughBootstrappedDemos(compiledStudent) {
			logger := logging.GetLogger()
			logger.Info(ctx, "Enough bootstrapped demos, breaking loop")
			break
		}

		ex := trainset[i]
		p.Go(func() {
			exampleCtx, exampleSpan := core.StartSpan(ctx, "Example")
			defer core.EndSpan(exampleCtx)

			exampleSpan.WithAnnotation("Example", ex)
			prediction, err := b.predictWithTeacher(exampleCtx, teacher, teacherLLM, ex)
			if err != nil {
				exampleSpan.WithError(err)
				select {
				case errCh <- err:
				default:
				}
				return
			}
			exampleSpan.WithAnnotation("prediction", prediction)

			if keep(exampleCtx, i, prediction) {
				resultsMu.Lock()
				results = append(results, core.Example{
					Inputs:  ex,
					Outputs: prediction,
				})
				resultsMu.Unlock()
			}

			processed.Add(1)
		})
	}

	p.Wait()

	select {
	case err := <-errCh:
		span.WithError(err)
		return compiledStudent, fmt.Errorf("error during compilation: %w", err)
	default:
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
	// Clone the teacher program and set its LLM to the teacher LLM
	teacherClone := teacher.Clone()
	for _, module := range teacherClone.Modules {
		if predictor, ok := module.(interface{ SetLLM(core.LLM) }); ok {
			predictor.SetLLM(teacherLLM)
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

func (b *BootstrapFewShot) enoughBootstrappedDemos(program core.Program) bool {
	for _, module := range program.Modules {
		if predictor, ok := module.(interface{ GetDemos() []core.Example }); ok {
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
		predictor, ok := module.(*modules.Predict)
		if !ok {
			continue
		}

		currentDemos := predictor.GetDemos()

		if len(currentDemos) < b.MaxBootstrapped {

			newDemos := append(currentDemos, demo)
			predictor.SetDemos(newDemos)
			span.WithAnnotation("added_to_module", moduleName)
			span.WithAnnotation("total_demos", len(newDemos))
			return nil
		} else {
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
	}
	return errors.New(errors.ResourceNotFound, "no suitable module found for adding demonstrations")
}
