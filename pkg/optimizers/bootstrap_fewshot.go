package optimizers

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type BootstrapFewShot struct {
	Metric          func(example map[string]interface{}, prediction map[string]interface{}, trace *core.Trace) bool
	MaxBootstrapped int
}

func NewBootstrapFewShot(metric func(example map[string]interface{}, prediction map[string]interface{}, trace *core.Trace) bool, maxBootstrapped int) *BootstrapFewShot {
	return &BootstrapFewShot{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
	}
}

func (b *BootstrapFewShot) Compile(ctx context.Context, student, teacher core.Program, trainset []map[string]interface{}) (core.Program, error) {
	compiledStudent := student.Clone()
	// Use the teacher LLM if available, otherwise use the default LLM
	teacherLLM := config.GetTeacherLLM()
	if teacherLLM == nil {
		teacherLLM = config.GetDefaultLLM()
	}
	ctx = core.WithTraceManager(ctx)
	tm := core.GetTraceManager(ctx)
	compilationTrace := tm.StartTrace("Compilation", "Compilation")
	defer tm.EndTrace()
	for _, example := range trainset {

		if b.enoughBootstrappedDemos(compiledStudent) {
			log.Println("Enough bootstrapped demos, breaking loop")

			break
		}
		exampleTrace := tm.StartTrace("Example", "Example")
		exampleTrace.SetInputs(example) // Set the inputs on the example trace

		// Use the teacher LLM for prediction
		prediction, err := b.predictWithTeacher(ctx, teacher, teacherLLM, example)
		if err != nil {
			exampleTrace.SetError(err)
			tm.EndTrace()
			compilationTrace.SetError(err)
			return compiledStudent, fmt.Errorf("error executing teacher: %w", err)
		}
		exampleTrace.SetOutputs(prediction) // Set the outputs on the example trace

		if b.Metric(example, prediction, exampleTrace) {
			if err := b.addDemonstrations(compiledStudent, exampleTrace); err != nil {
				exampleTrace.SetError(err)
				tm.EndTrace()
				compilationTrace.SetError(err)

				return compiledStudent, fmt.Errorf("error adding demonstrations: %w", err)
			}
		}
		tm.EndTrace() // End the example trace
	}
	compilationTrace.SetOutputs(map[string]interface{}{
		"compiledStudent": compiledStudent,
	})
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
	tm := core.GetTraceManager(ctx)
	trace := tm.StartTrace("TeacherPrediction", "Prediction")
	defer tm.EndTrace()
	trace.SetInputs(example) // Set the inputs on the trace

	outputs, err := teacherClone.Execute(ctx, example)
	if err != nil {
		trace.SetError(err)
		return nil, err
	}

	trace.SetOutputs(outputs)
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

func (b *BootstrapFewShot) addDemonstrations(program core.Program, trace *core.Trace) error {
	if trace == nil {
		return fmt.Errorf("trace is nil")
	}

	for moduleName, module := range program.Modules {
		predictor, ok := module.(*modules.Predict)
		if !ok {
			continue
		}

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
	return fmt.Errorf("no suitable module found for adding demonstrations")
}
