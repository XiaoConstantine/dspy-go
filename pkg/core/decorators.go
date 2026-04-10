package core

import (
	"context"
	"fmt"
)

// Deprecated: prefer explicit RecordLLMCall calls at LLM boundaries instead of
// wrapping LLM implementations with decorators.
// BaseDecorator provides common functionality for compatibility with older code.
type BaseDecorator struct {
	LLM
}

// Deprecated: prefer explicit RecordLLMCall calls at LLM boundaries instead of
// wrapping LLM implementations with decorators.
// ModelContextDecorator adds model context tracking.
type ModelContextDecorator struct {
	BaseDecorator
}

// Deprecated: prefer explicit RecordLLMCall calls at LLM boundaries instead of
// wrapping LLM implementations with decorators.
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
	RecordLLMCall(ctx, d.LLM)
	return d.LLM.Generate(ctx, prompt, options...)
}

func (d *ModelContextDecorator) GenerateWithTools(ctx context.Context, messages []ChatMessage, tools []map[string]any, opts ...GenerateOption) (map[string]any, error) {
	RecordLLMCall(ctx, d.LLM)
	return d.BaseDecorator.GenerateWithTools(ctx, messages, tools, opts...)
}

// Deprecated: prefer explicit RecordLLMCall calls at LLM boundaries instead of
// composing LLM decorators.
// Helper function to compose multiple decorators.
func Chain(base LLM, decorators ...func(LLM) LLM) LLM {
	result := base
	for _, d := range decorators {
		result = d(result)
	}
	return result
}
