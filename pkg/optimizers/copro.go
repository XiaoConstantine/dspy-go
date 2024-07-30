package optimizers

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type Copro struct {
	Metric          func(example, prediction map[string]interface{}, trace *core.Trace) bool
	MaxBootstrapped int
	SubOptimizer    core.Optimizer
}

func NewCopro(metric func(example, prediction map[string]interface{}, trace *core.Trace) bool, maxBootstrapped int, subOptimizer core.Optimizer) *Copro {
	return &Copro{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
		SubOptimizer:    subOptimizer,
	}
}
func (c *Copro) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	compiledProgram := program.Clone()
	ctx = core.WithTraceManager(ctx)
	tm := core.GetTraceManager(ctx)
	compilationTrace := tm.StartTrace("CoproCompilation", "Compilation")
	defer tm.EndTrace()

	wrappedMetric := func(expected, actual map[string]interface{}) float64 {
		// Get the current trace from the context
		tm := core.GetTraceManager(ctx)
		currentTrace := tm.CurrentTrace

		if c.Metric(expected, actual, currentTrace) {
			return 1.0
		}
		return 0.0
	}
	for moduleName, module := range compiledProgram.Modules {
		moduleTrace := tm.StartTrace(fmt.Sprintf("Module_%s", moduleName), "ModuleCompilation")

		compiledModule, err := c.compileModule(ctx, module, dataset, wrappedMetric)
		if err != nil {
			moduleTrace.SetError(err)
			tm.EndTrace()
			compilationTrace.SetError(err)
			return compiledProgram, fmt.Errorf("error compiling module %s: %w", moduleName, err)
		}

		compiledProgram.Modules[moduleName] = compiledModule
		moduleTrace.SetOutputs(map[string]interface{}{"compiledModule": compiledModule})
		tm.EndTrace()
	}

	compilationTrace.SetOutputs(map[string]interface{}{"compiledProgram": compiledProgram})
	return compiledProgram, nil
}

func (c *Copro) compileModule(ctx context.Context, module core.Module, dataset core.Dataset, metric core.Metric) (core.Module, error) {
	switch m := module.(type) {
	case *modules.Predict:
		// Create a temporary Program with just this Predict module
		tempProgram := core.NewProgram(map[string]core.Module{"predict": m}, nil)
		// Compile using the SubOptimizer
		compiledProgram, err := c.SubOptimizer.Compile(ctx, tempProgram, dataset, metric)
		if err != nil {
			return nil, err
		}

		// Extract the optimized Predict module from the compiled Program
		optimizedPredict, ok := compiledProgram.Modules["predict"]
		if !ok {
			return nil, fmt.Errorf("compiled program does not contain 'predict' module")
		}

		return optimizedPredict, nil

	case core.Composable:
		subModules := m.GetSubModules()
		for i, subModule := range subModules {
			compiledSubModule, err := c.compileModule(ctx, subModule, dataset, metric)
			if err != nil {
				return nil, err
			}
			subModules[i] = compiledSubModule
		}
		m.SetSubModules(subModules)
		return m, nil

	default:
		// For non-Predict, non-Composable modules, return as-is

		return m, nil
	}
}
