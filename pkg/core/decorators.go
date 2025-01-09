package core

import "context"

// BaseDecorator provides common functionality for all LLM decorators.
type BaseDecorator struct {
	LLM
}

// ModelContextDecorator adds model context tracking.
type ModelContextDecorator struct {
	BaseDecorator
}

func NewModelContextDecorator(base LLM) *ModelContextDecorator {
	return &ModelContextDecorator{
		BaseDecorator: BaseDecorator{LLM: base},
	}
}

func (d *BaseDecorator) Unwrap() LLM {
	return d.LLM
}

func (d *ModelContextDecorator) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	if state := GetExecutionState(ctx); state != nil {
		state.WithModelID(d.ModelID())
	}
	return d.LLM.Generate(ctx, prompt, options...)
}

// Helper function to compose multiple decorators.
func Chain(base LLM, decorators ...func(LLM) LLM) LLM {
	result := base
	for _, d := range decorators {
		result = d(result)
	}
	return result
}
