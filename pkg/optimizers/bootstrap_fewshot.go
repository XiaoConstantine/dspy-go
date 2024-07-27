package optimizers

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
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
	// Use the teacher LLM if available, otherwise use the default LLM
	teacherLLM := config.GetTeacherLLM()
	if teacherLLM == nil {
		teacherLLM = config.GetDefaultLLM()
	}
	for _, example := range trainset {
		if b.enoughBootstrappedDemos(compiledStudent) {

			break
		}

		traces := &[]core.Trace{}
		ctx := context.WithValue(context.Background(), utils.TracesContextKey, traces)
		// Use the teacher LLM for prediction
		prediction, err := b.predictWithTeacher(ctx, teacher, teacherLLM, example)
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

func (b *BootstrapFewShot) predictWithTeacher(ctx context.Context, teacher core.Program, teacherLLM core.LLM, example map[string]interface{}) (map[string]interface{}, error) {
	// Clone the teacher program and set its LLM to the teacher LLM
	teacherClone := teacher.Clone()
	for _, module := range teacherClone.Modules {
		if predictor, ok := module.(interface{ SetLLM(core.LLM) }); ok {
			predictor.SetLLM(teacherLLM)
		}
	}

	return teacherClone.Execute(ctx, example)
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
