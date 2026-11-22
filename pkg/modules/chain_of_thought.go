package modules

import (
	"context"
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

type ChainOfThought struct {
	Predict *Predict

	// structuredOutput is set by WithStructuredOutput. Process then runs the
	// module's interceptors itself, because Predict no longer does.
	structuredOutput bool
}

var (
	_ core.Module              = (*ChainOfThought)(nil)
	_ core.InterceptableModule = (*ChainOfThought)(nil)
	_ core.Composable          = (*ChainOfThought)(nil)
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
	if c.Predict.GetAdapter() != nil {
		modifiedSignature = renameOutput(modifiedSignature, rationaleField, reasoningField, true)
	}
	c.Predict.SetSignature(modifiedSignature)
}

// SetLLM sets the LLM on the internal Predict module.
func (c *ChainOfThought) SetLLM(llm core.LLM) {
	c.Predict.SetLLM(llm)
}

// Clone creates a deep copy of the ChainOfThought module.
func (c *ChainOfThought) Clone() core.Module {
	return &ChainOfThought{
		Predict:          c.Predict.Clone().(*Predict),
		structuredOutput: c.structuredOutput,
	}
}

// Compose creates a new module that chains this module with the next module.
func (c *ChainOfThought) Compose(next core.Module) core.Module {
	return core.NewModuleChain(c, next)
}

// GetDemos returns the demonstrations of the underlying Predict module.
func (c *ChainOfThought) GetDemos() []core.Example {
	return c.Predict.GetDemos()
}

// SetDemos sets the demonstrations of the underlying Predict module.
func (c *ChainOfThought) SetDemos(demos []core.Example) {
	c.Predict.SetDemos(demos)
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

// WithStructuredOutput enables native JSON structured output for ChainOfThought.
// This uses the LLM's GenerateWithJSON capability to produce structured responses
// that include both reasoning and output fields, eliminating parsing errors.
//
// The output will include a "reasoning" field containing the step-by-step thought process.
//
// Usage:
//
//	cot := modules.NewChainOfThought(signature).WithStructuredOutput()
func (c *ChainOfThought) WithStructuredOutput() *ChainOfThought {
	return c.WithStructuredOutputConfig(interceptors.DefaultChainOfThoughtStructuredConfig())
}

// WithStructuredOutputConfig enables structured output with custom CoT configuration.
func (c *ChainOfThought) WithStructuredOutputConfig(config interceptors.ChainOfThoughtStructuredConfig) *ChainOfThought {
	interceptor := interceptors.ChainOfThoughtStructuredInterceptor(config)
	// The interceptor replaces Predict's XML interceptor, so a model without
	// JSON support must fall back to text parsing, not to unparsed XML mode.
	c.Predict.enableXMLMode = false
	c.Predict.xmlConfig = nil
	c.structuredOutput = true
	c.SetInterceptors([]core.ModuleInterceptor{interceptor})
	return c
}

// WithAdapter makes the module use a structured-output adapter, as
// Predict.WithAdapter does. With an adapter the module asks for its reasoning
// in a field named "reasoning", which a model with native reasoning fills from
// its reasoning summary; without one the field is named "rationale", as before.
// For one release the outputs of a module with an adapter also carry the
// reasoning as "rationale". A nil adapter restores text parsing and the
// outputs it has always returned.
func (c *ChainOfThought) WithAdapter(adapter core.Adapter) *ChainOfThought {
	c.Predict.WithAdapter(adapter)
	c.structuredOutput = false
	signature := c.Predict.GetSignature()
	if adapter != nil {
		signature = renameOutput(signature, rationaleField, reasoningField, true)
	} else {
		signature = renameOutput(signature, reasoningField, rationaleField, false)
	}
	c.Predict.SetSignature(signature)
	return c
}

func (c *ChainOfThought) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	if c.structuredOutput {
		return c.ProcessWithInterceptors(ctx, inputs, nil, opts...)
	}
	outputs, err := c.processCore(ctx, inputs, opts...)
	return c.withReasoningAlias(outputs), err
}

// processCore runs the prediction without the module's interceptors.
func (c *ChainOfThought) processCore(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Use ChainOfThought's own display name
	displayName := c.GetDisplayName()

	metadata := map[string]any{
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
	// The alias is added after the prediction is recorded, so recorded
	// demonstrations keep only the field the signature declares.
	outputs, err := c.processWithInterceptors(ctx, inputs, interceptors, opts...)
	return c.withReasoningAlias(outputs), err
}

func (c *ChainOfThought) processWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (outputs map[string]any, err error) {
	// Structured-output interceptors can return without calling Predict.Process.
	if record := predictionRecorder(ctx, c.Predict, inputs); record != nil {
		ctx = withoutPredictionTrace(ctx, c.Predict)
		defer func() { record(outputs, err) }()
	}
	// Use provided interceptors, or fall back to Predict's interceptors if none provided
	if interceptors == nil {
		interceptors = c.Predict.GetInterceptors()
	}

	// Create correct ModuleInfo for ChainOfThought (not Predict)
	info := core.NewModuleInfo(c.GetDisplayName(), c.GetModuleType(), c.GetSignature()).
		WithLLM(c.Predict.LLM).
		WithDefaultOptions(c.Predict.defaultOptions).
		WithDemos(c.Predict.Demos)

	// Chain interceptors and execute
	chainedInterceptor := core.ChainModuleInterceptors(interceptors...)
	return chainedInterceptor(ctx, inputs, info, c.processCore, opts...)
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

// NewTypedChainOfThought creates a new type-safe ChainOfThought module from a typed signature.
// Typed modules use text-based parsing by default since they typically rely on prefixes.
func NewTypedChainOfThought[TInput, TOutput any]() *ChainOfThought {
	typedSig := core.NewTypedSignatureCached[TInput, TOutput]()
	legacySig := typedSig.ToLegacySignature()

	cot := NewChainOfThought(legacySig)
	// Use clearer variable names for type display
	var i TInput
	var o TOutput
	cot.Predict.DisplayName = fmt.Sprintf("TypedChainOfThought[%T,%T]", i, o)

	return cot
}

// The reasoning output is named "rationale" without an adapter and "reasoning"
// with one. For one release, outputs produced with an adapter also carry the
// reasoning as "rationale", so code that reads the old name keeps working after
// it opts in. Without an adapter the outputs are unchanged.
const (
	rationaleField = "rationale"
	reasoningField = "reasoning"
)

// withReasoningAlias copies an adapter's "reasoning" output to "rationale". It
// never overwrites a field the outputs already have, and it leaves the name
// alone when the signature declares its own output by it, so a caller's field
// that the model omitted stays absent.
func (c *ChainOfThought) withReasoningAlias(outputs map[string]any) map[string]any {
	if outputs == nil || c.Predict.GetAdapter() == nil {
		return outputs
	}
	value, has := outputs[reasoningField]
	if _, taken := outputs[rationaleField]; !has || taken {
		return outputs
	}
	for _, field := range c.GetSignature().Outputs {
		if field.Name == rationaleField {
			return outputs
		}
	}
	outputs[rationaleField] = value
	return outputs
}

// renameOutput renames the output field from to to and sets whether a model's
// native reasoning may fill it. A signature that already has an output named
// to, or none named from, is returned unchanged.
func renameOutput(signature core.Signature, from, to string, nativeReasoning bool) core.Signature {
	index := -1
	for i, output := range signature.Outputs {
		switch output.Name {
		case to:
			return signature
		case from:
			index = i
		}
	}
	if index < 0 {
		return signature
	}
	renamed := signature
	renamed.Outputs = append([]core.OutputField(nil), signature.Outputs...)
	field := &renamed.Outputs[index]
	field.Name = to
	field.NativeReasoning = nativeReasoning
	if field.Prefix == from+":" {
		field.Prefix = to + ":"
	}
	return renamed
}

func appendRationaleField(signature core.Signature) core.Signature {
	for _, output := range signature.Outputs {
		if output.Name == "rationale" {
			return signature
		}
	}
	newSignature := signature
	rationaleField := core.OutputField{
		Field: core.NewField("rationale",
			core.WithDescription("Step-by-step reasoning process"),
		),
	}
	newSignature.Outputs = append([]core.OutputField{rationaleField}, newSignature.Outputs...)

	return newSignature
}
