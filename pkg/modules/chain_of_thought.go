package modules

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type ChainOfThought struct {
	Predict *Predict
}

var (
	_ core.Module             = (*ChainOfThought)(nil)
	_ core.InterceptableModule = (*ChainOfThought)(nil)
	_ core.Composable         = (*ChainOfThought)(nil)
)

func NewChainOfThought(signature core.Signature) *ChainOfThought {
	modifiedSignature := appendRationaleField(signature)
	return &ChainOfThought{
		Predict: NewPredict(modifiedSignature),
	}
}

// WithName sets a semantic name for this ChainOfThought instance.
func (c *ChainOfThought) WithName(name string) *ChainOfThought {
	c.Predict.DisplayName = name
	return c
}

// GetDisplayName returns the display name for this ChainOfThought module.
func (c *ChainOfThought) GetDisplayName() string {
	// If the inner Predict module has a custom display name, use it.
	if c.Predict.DisplayName != "" {
		return c.Predict.DisplayName
	}
	return "ChainOfThought"
}

// GetModuleType returns "ChainOfThought".
func (c *ChainOfThought) GetModuleType() string {
	return "ChainOfThought"
}

// GetSignature returns the signature from the internal Predict module.
func (c *ChainOfThought) GetSignature() core.Signature {
	return c.Predict.GetSignature()
}

// SetSignature sets the signature on the internal Predict module with rationale field.
func (c *ChainOfThought) SetSignature(signature core.Signature) {
	modifiedSignature := appendRationaleField(signature)
	c.Predict.SetSignature(modifiedSignature)
}

// SetLLM sets the LLM on the internal Predict module.
func (c *ChainOfThought) SetLLM(llm core.LLM) {
	c.Predict.SetLLM(llm)
}

// Clone creates a deep copy of the ChainOfThought module.
func (c *ChainOfThought) Clone() core.Module {
	return &ChainOfThought{
		Predict: c.Predict.Clone().(*Predict),
	}
}

// Compose creates a new module that chains this module with the next module.
func (c *ChainOfThought) Compose(next core.Module) core.Module {
	return core.NewModuleChain(c, next)
}

// GetSubModules returns the sub-modules of this ChainOfThought.
func (c *ChainOfThought) GetSubModules() []core.Module {
	return []core.Module{c.Predict}
}

// SetSubModules sets the sub-modules (expects exactly one Predict module).
func (c *ChainOfThought) SetSubModules(modules []core.Module) {
	if len(modules) == 1 {
		if predict, ok := modules[0].(*Predict); ok {
			c.Predict = predict
		}
	}
}

// WithDefaultOptions sets default options by configuring the underlying Predict module.
func (c *ChainOfThought) WithDefaultOptions(opts ...core.Option) *ChainOfThought {
	// Simply delegate to the Predict module's WithDefaultOptions
	c.Predict.WithDefaultOptions(opts...)
	return c
}

func (c *ChainOfThought) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Use ChainOfThought's own display name
	displayName := c.GetDisplayName()

	metadata := map[string]interface{}{
		"module_type":   c.GetModuleType(),
		"module_config": c.GetSignature().String(),
	}
	ctx, span := core.StartSpanWithContext(ctx, "ChainOfThought", displayName, metadata)
	defer core.EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)
	outputs, err := c.Predict.Process(ctx, inputs, opts...)
	if err != nil {
		span.WithError(err)
		return nil, err
	}
	span.WithAnnotation("outputs", outputs)

	return outputs, nil
}

// ProcessWithInterceptors executes the module's logic with interceptor support.
func (c *ChainOfThought) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	// Use provided interceptors, or fall back to Predict's interceptors if none provided
	if interceptors == nil {
		interceptors = c.Predict.GetInterceptors()
	}

	// Create correct ModuleInfo for ChainOfThought (not Predict)
	info := core.NewModuleInfo(c.GetDisplayName(), c.GetModuleType(), c.GetSignature())
	
	// Create handler that calls our own Process method
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return c.Process(ctx, inputs, opts...)
	}
	
	// Chain interceptors and execute
	chainedInterceptor := core.ChainModuleInterceptors(interceptors...)
	return chainedInterceptor(ctx, inputs, info, handler, opts...)
}

// SetInterceptors sets the default interceptors for this module instance.
func (c *ChainOfThought) SetInterceptors(interceptors []core.ModuleInterceptor) {
	c.Predict.SetInterceptors(interceptors)
}

// GetInterceptors returns the current interceptors for this module.
func (c *ChainOfThought) GetInterceptors() []core.ModuleInterceptor {
	return c.Predict.GetInterceptors()
}

// ClearInterceptors removes all interceptors from this module.
func (c *ChainOfThought) ClearInterceptors() {
	c.Predict.ClearInterceptors()
}

func appendRationaleField(signature core.Signature) core.Signature {
	newSignature := signature
	rationaleField := core.OutputField{
		Field: core.NewField("rationale",
			core.WithDescription("Step-by-step reasoning process"),
		),
	}
	newSignature.Outputs = append([]core.OutputField{rationaleField}, newSignature.Outputs...)

	return newSignature
}
