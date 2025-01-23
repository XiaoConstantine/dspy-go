package optimizers

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/sourcegraph/conc/pool"
)

type BootstrapFewShot struct {
	Metric          func(example map[string]interface{}, prediction map[string]interface{}, ctx context.Context) bool
	MaxBootstrapped int
}

func NewBootstrapFewShot(metric func(example map[string]interface{}, prediction map[string]interface{}, ctx context.Context) bool, maxBootstrapped int) *BootstrapFewShot {
	return &BootstrapFewShot{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
	}
}

func (b *BootstrapFewShot) Compile(ctx context.Context, student, teacher core.Program, trainset []map[string]interface{}) (core.Program, error) {
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
		results   []struct {
			demo core.Example
			ctx  context.Context
		}
		processed int32
		errCh     = make(chan error, 1)
	)
	examplesNeeded := b.MaxBootstrapped
	if examplesNeeded > len(trainset) {
		examplesNeeded = len(trainset)
	}

	p := pool.New().WithMaxGoroutines(core.GlobalConfig.ConcurrencyLevel)

	for i := 0; i < examplesNeeded; i++ {
		if b.enoughBootstrappedDemos(compiledStudent) {
			log.Println("Enough bootstrapped demos, breaking loop")
			break
		}

		ex := trainset[i]
		p.Go(func() {
			exampleCtx, exampleSpan := core.StartSpan(ctx, "Example")
			defer core.EndSpan(exampleCtx)

			exampleSpan.WithAnnotation("Example", ex)
			prediction, err := b.predictWithTeacher(ctx, teacher, teacherLLM, ex)
			if err != nil {
				exampleSpan.WithError(err)
				select {
				case errCh <- err:
				default:
				}
				return
			}
			exampleSpan.WithAnnotation("prediction", prediction)

			if b.Metric(ex, prediction, exampleCtx) {
				resultsMu.Lock()
				results = append(results, struct {
					demo core.Example
					ctx  context.Context
				}{
					demo: core.Example{
						Inputs:  ex,
						Outputs: prediction,
					},
					ctx: exampleCtx,
				})
				resultsMu.Unlock()
			}

			atomic.AddInt32(&processed, 1)
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
		if err := b.addDemonstrations(compiledStudent, result.demo, result.ctx); err != nil {
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

func (b *BootstrapFewShot) predictWithTeacher(ctx context.Context, teacher core.Program, teacherLLM core.LLM, example map[string]interface{}) (map[string]interface{}, error) {
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
		return fmt.Errorf("cannot add demonstrations: context is nil")

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
			return fmt.Errorf("max demonstrations reached for module %s", moduleName)
		}
	}
	return fmt.Errorf("no suitable module found for adding demonstrations")
}
