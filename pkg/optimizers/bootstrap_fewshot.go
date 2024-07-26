package optimizers

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type BootstrapFewShot struct {
	Metric          func(example map[string]interface{}, prediction map[string]interface{}, traces *[]core.Trace) bool
	MaxBootstrapped int
}

func NewBootstrapFewShot(metric func(example map[string]interface{}, prediction map[string]interface{}, traces *[]core.Trace) bool, maxBootstrapped int) *BootstrapFewShot {
	return &BootstrapFewShot{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
	}
}

func (b *BootstrapFewShot) Compile(student, teacher core.Program, trainset []map[string]interface{}) (core.Program, error) {
	compiledStudent := student.Clone()

	for _, example := range trainset {
		if b.enoughBootstrappedDemos(compiledStudent) {

			break
		}

		traces := &[]core.Trace{}
		ctx := context.WithValue(context.Background(), "traces", traces)

		prediction, err := teacher.Execute(ctx, example)
		if err != nil {
			return compiledStudent, fmt.Errorf("error executing teacher: %w", err)
		}

		if b.Metric(example, prediction, traces) {
			if err := b.addDemonstrations(compiledStudent, traces); err != nil {
				return compiledStudent, fmt.Errorf("error adding demonstrations: %w", err)
			}
		}
	}

	return compiledStudent, nil
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

func (b *BootstrapFewShot) addDemonstrations(program core.Program, traces *[]core.Trace) error {
	for _, trace := range *traces {
		for moduleName, module := range program.Modules {
			if predictor, ok := module.(interface {
				GetDemos() []core.Example
				SetDemos([]core.Example)
			}); ok {
				demos := predictor.GetDemos()

				if len(demos) < b.MaxBootstrapped {
					demo := core.Example{
						Inputs:  trace.Inputs,
						Outputs: trace.Outputs,
					}
					demos = append(demos, demo)
					predictor.SetDemos(demos)
					return nil

				} else {
					return fmt.Errorf("max demonstrations reached for module %s", moduleName)
				}
			}
		}
	}
	return nil
}
