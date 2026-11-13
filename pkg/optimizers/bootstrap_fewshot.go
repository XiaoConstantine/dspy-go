package optimizers

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sync"

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
// every teacher call fails. Each predictor of the program, including one
// nested in a module such as ChainOfThought, gets demonstrations of its own
// inputs and outputs in the kept runs; they replace its existing ones. It
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

// bootstrapRun is one teacher run the metric kept: the example, the program's
// prediction, and the last call of each teacher predictor by name.
type bootstrapRun struct {
	example    map[string]any
	prediction map[string]any
	calls      map[string]core.Example
}

// compileInternal bootstraps demonstrations for student from teacher runs.
// keep reports whether the teacher's run on trainset[index] is kept. Each
// student predictor gets one demonstration per kept run, built from what the
// teacher predictor of the same name received and returned in that run.
func (b *BootstrapFewShot) compileInternal(ctx context.Context, student, teacher core.Program, trainset []map[string]any, keep func(ctx context.Context, index int, prediction map[string]any) bool) (core.Program, error) {
	for _, program := range []core.Program{student, teacher} {
		if err := requireRebindableProgram(program); err != nil {
			return student, err
		}
	}
	compiledStudent := student.Clone()
	// Start from a student without demonstrations, so compiling an already
	// compiled program bootstraps again instead of counting its old demos.
	names, predictors := namedPredictors(compiledStudent)
	for _, predictor := range predictors {
		predictor.SetDemos(nil)
	}
	// A student predictor learns from the teacher predictor of the same name.
	// Two single-predictor programs pair up whatever their module names.
	teacherNames, _ := namedPredictors(teacher)
	teacherName := func(name string) string { return name }
	if len(names) == 1 && len(teacherNames) == 1 {
		teacherName = func(string) string { return teacherNames[0] }
	} else if len(names) > 0 && !slices.Equal(names, teacherNames) {
		return student, errors.New(errors.InvalidInput, fmt.Sprintf("BootstrapFewShot: student predictors %v and teacher predictors %v must have the same names", names, teacherNames))
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
	// time, until enough runs are kept. Stopping at the first
	// MaxBootstrapped examples would return too few demonstrations whenever
	// some of them fail the metric.
	var runs []bootstrapRun
	var teacherErr error
	teacherSucceeded := false
	batchSize := max(core.GetConcurrencyLevel(), 1)
	for start := 0; start < len(trainset) && len(runs) < b.MaxBootstrapped; start += batchSize {
		batch := trainset[start:min(start+batchSize, len(trainset))]
		kept := make([]*bootstrapRun, len(batch))
		errs := make([]error, len(batch))
		p := pool.New().WithMaxGoroutines(batchSize)
		for i, ex := range batch {
			p.Go(func() {
				exampleCtx, exampleSpan := core.StartSpan(ctx, "Example")
				defer core.EndSpan(exampleCtx)

				exampleSpan.WithAnnotation("Example", ex)
				prediction, calls, err := b.predictWithTeacher(exampleCtx, teacher, teacherLLM, ex)
				if err != nil {
					exampleSpan.WithError(err)
					errs[i] = err
					return
				}
				exampleSpan.WithAnnotation("prediction", prediction)

				if keep(exampleCtx, start+i, prediction) {
					kept[i] = &bootstrapRun{example: ex, prediction: prediction, calls: calls}
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
		for _, run := range kept {
			if run != nil && len(runs) < b.MaxBootstrapped {
				runs = append(runs, *run)
			}
		}
	}
	// A teacher that never works is a broken setup, not a hard example.
	if teacherErr != nil && !teacherSucceeded {
		span.WithError(teacherErr)
		return compiledStudent, fmt.Errorf("error during compilation: every teacher call failed: %w", teacherErr)
	}
	if len(runs) > 0 && len(names) == 0 {
		err := errors.New(errors.ResourceNotFound, "no suitable module found for adding demonstrations")
		span.WithError(err)
		return compiledStudent, fmt.Errorf("error adding demonstrations: %w", err)
	}

	for _, run := range runs {
		demoCtx, demoSpan := core.StartSpan(ctx, "AddDemonstrations")
		demoSpan.WithAnnotation("demo_inputs", run.example)
		demoSpan.WithAnnotation("demo_outputs", run.prediction)
		for _, name := range names {
			predictor := predictors[name]
			call, ok := run.calls[teacherName(name)]
			if !ok && len(names) == 1 {
				// The only predictor could not be traced, as when a module
				// answers without calling it; the program's run stands in.
				call, ok = core.Example{Inputs: run.example, Outputs: run.prediction}, true
			}
			if !ok {
				// The run did not call this predictor.
				continue
			}
			// A demonstration holds only the predictor's own fields.
			signature := predictor.GetSignature()
			demo := core.Example{Inputs: map[string]any{}, Outputs: map[string]any{}}
			for _, field := range signature.Inputs {
				if value, ok := call.Inputs[field.Name]; ok {
					demo.Inputs[field.Name] = value
				}
			}
			for _, field := range signature.Outputs {
				if value, ok := call.Outputs[field.Name]; ok {
					demo.Outputs[field.Name] = value
				}
			}
			predictor.SetDemos(append(predictor.GetDemos(), demo))
			demoSpan.WithAnnotation("added_to_module", name)
		}
		core.EndSpan(demoCtx)
	}

	span.WithAnnotation("compiledStudent", compiledStudent)
	return compiledStudent, nil
}

// predictWithTeacher runs a clone of teacher on example. It returns the
// program's prediction and the last successful call of each predictor by name.
func (b *BootstrapFewShot) predictWithTeacher(ctx context.Context, teacher core.Program, teacherLLM core.LLM, example map[string]any) (map[string]any, map[string]core.Example, error) {
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

	_, predictors := namedPredictors(teacherClone)
	nameOf := make(map[*modules.Predict]string, len(predictors))
	for name, predictor := range predictors {
		nameOf[predictor] = name
	}
	var mu sync.Mutex
	calls := make(map[string]core.Example)
	ctx = modules.WithPredictionObserver(ctx, func(predictor *modules.Predict, inputs, outputs map[string]any) {
		name, ok := nameOf[predictor]
		if !ok {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		calls[name] = core.Example{Inputs: maps.Clone(inputs), Outputs: maps.Clone(outputs)}
	})

	span.WithAnnotation("Example", example)
	outputs, err := teacherClone.Execute(ctx, example)
	if err != nil {
		span.WithError(err)
		return nil, nil, err
	}

	span.WithAnnotation("outputs", outputs)
	mu.Lock()
	defer mu.Unlock()
	return outputs, maps.Clone(calls), nil
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

// namedPredictors returns program's predictors by name, and the names sorted.
// A module's first predictor is named after the module and any further ones
// "name[i]".
func namedPredictors(program core.Program) ([]string, map[string]*modules.Predict) {
	predictors := make(map[string]*modules.Predict)
	for moduleName, module := range program.Modules {
		for i, predictor := range demoPredictors(module) {
			name := moduleName
			if i > 0 {
				name = fmt.Sprintf("%s[%d]", moduleName, i)
			}
			predictors[name] = predictor
		}
	}
	return slices.Sorted(maps.Keys(predictors)), predictors
}
