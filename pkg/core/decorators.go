package core

import (
	"context"
	"fmt"
)

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

func (d *BaseDecorator) GenerateWithTools(ctx context.Context, messages []ChatMessage, tools []map[string]any, opts ...GenerateOption) (map[string]any, error) {
	chatLLM, ok := d.LLM.(ToolCallingChatLLM)
	if !ok {
		return nil, fmt.Errorf("wrapped llm %s does not support native tool calling", d.ModelID())
	}
	return chatLLM.GenerateWithTools(ctx, messages, tools, opts...)
}

func (d *ModelContextDecorator) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	if state := GetExecutionState(ctx); state != nil {
		state.WithModelID(d.ModelID())
	}
	return d.LLM.Generate(ctx, prompt, options...)
}

func (d *ModelContextDecorator) GenerateWithTools(ctx context.Context, messages []ChatMessage, tools []map[string]any, opts ...GenerateOption) (map[string]any, error) {
	if state := GetExecutionState(ctx); state != nil {
		state.WithModelID(d.ModelID())
	}
	return d.BaseDecorator.GenerateWithTools(ctx, messages, tools, opts...)
}

// Helper function to compose multiple decorators.
func Chain(base LLM, decorators ...func(LLM) LLM) LLM {
	result := base
	for _, d := range decorators {
		result = d(result)
	}
	return result
}
