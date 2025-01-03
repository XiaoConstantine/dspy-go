package optimizers

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type Copro struct {
	Metric          func(example, prediction map[string]interface{}, ctx context.Context) bool
	MaxBootstrapped int
	SubOptimizer    core.Optimizer
}

func NewCopro(metric func(example, prediction map[string]interface{}, ctx context.Context) bool, maxBootstrapped int, subOptimizer core.Optimizer) *Copro {
	return &Copro{
		Metric:          metric,
		MaxBootstrapped: maxBootstrapped,
		SubOptimizer:    subOptimizer,
	}
}
func (c *Copro) Compile(ctx context.Context, program core.Program, dataset core.Dataset, metric core.Metric) (core.Program, error) {
	compiledProgram := program.Clone()
	// Ensure execution state exists
	if core.GetExecutionState(ctx) == nil {
		ctx = core.WithExecutionState(ctx)
	}

	ctx, compilationSpan := core.StartSpan(ctx, "CoproCompilation")
	defer core.EndSpan(ctx)

	wrappedMetric := func(expected, actual map[string]interface{}) float64 {
		metricCtx, metricSpan := core.StartSpan(ctx, "MetricEvaluation")
		defer core.EndSpan(metricCtx)

		metricSpan.WithAnnotation("expected", expected)
		metricSpan.WithAnnotation("actual", actual)

		// Use the context-based metric
		if c.Metric(expected, actual, metricCtx) {
			metricSpan.WithAnnotation("result", 1.0)
			return 1.0
		}

		metricSpan.WithAnnotation("result", 0.0)

		return 0.0
	}
	for moduleName, module := range compiledProgram.Modules {
		moduleCtx, moduleSpan := core.StartSpan(ctx, fmt.Sprintf("Module_%s", moduleName))

		compiledModule, err := c.compileModule(ctx, module, dataset, wrappedMetric)
		if err != nil {
			moduleSpan.WithError(err)
			core.EndSpan(moduleCtx)
			compilationSpan.WithError(err)

			return compiledProgram, fmt.Errorf("error compiling module %s: %w", moduleName, err)
		}

		compiledProgram.Modules[moduleName] = compiledModule
		moduleSpan.WithAnnotation("compiledModule", compiledModule)
		core.EndSpan(moduleCtx)

	}

	compilationSpan.WithAnnotation("compiledProgram", compiledProgram)

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
