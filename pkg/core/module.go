package core

import (
	"context"
	"errors"

	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// Module represents a basic unit of computation in DSPy.
type Module interface {
	// Process executes the module's logic
	Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error)

	// GetSignature returns the module's input and output signature
	GetSignature() Signature

	SetSignature(signature Signature)

	// SetLLM sets the language model for the module
	SetLLM(llm LLM)

	// Clone creates a deep copy of the module
	Clone() Module

	// GetDisplayName returns a human-readable name for this module instance
	GetDisplayName() string

	// GetModuleType returns the category/type of this module
	GetModuleType() string
}

// TypeSafeModule extends Module with compile-time type safety.
type TypeSafeModule interface {
	Module

	// ProcessTyped executes the module's logic with type-safe inputs/outputs
	ProcessTyped(ctx context.Context, inputs any, opts ...Option) (any, error)
}

// InterceptableModule extends Module with interceptor support.
// This interface provides backward-compatible enhancement for modules that support interceptors.
type InterceptableModule interface {
	Module

	// ProcessWithInterceptors executes the module's logic with interceptor support
	ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error)

	// SetInterceptors sets the default interceptors for this module instance
	SetInterceptors(interceptors []ModuleInterceptor)

	// GetInterceptors returns the current interceptors for this module
	GetInterceptors() []ModuleInterceptor

	// ClearInterceptors removes all interceptors from this module
	ClearInterceptors()
}

type Option func(*ModuleOptions)

type StreamHandler func(chunk StreamChunk) error

// WithStreamHandler returns an option to enable streaming.
func WithStreamHandler(handler StreamHandler) Option {
	return func(o *ModuleOptions) {
		o.StreamHandler = handler
	}
}

// ModuleOptions holds configuration that can be passed to modules.
type ModuleOptions struct {
	// LLM generation options
	GenerateOptions []GenerateOption

	StreamHandler StreamHandler
}

// WithGenerateOptions adds LLM generation options.
func WithGenerateOptions(opts ...GenerateOption) Option {
	return func(o *ModuleOptions) {
		o.GenerateOptions = append(o.GenerateOptions, opts...)
	}
}

// Clone creates a copy of ModuleOptions.
func (o *ModuleOptions) Clone() *ModuleOptions {
	if o == nil {
		return nil
	}
	return &ModuleOptions{
		GenerateOptions: append([]GenerateOption{}, o.GenerateOptions...),
		StreamHandler:   o.StreamHandler, // Copy the StreamHandler reference
	}
}

// MergeWith merges this options with other options, with other taking precedence.
func (o *ModuleOptions) MergeWith(other *ModuleOptions) *ModuleOptions {
	if other == nil {
		return o.Clone()
	}
	merged := o.Clone()
	if merged == nil {
		merged = &ModuleOptions{}
	}
	merged.GenerateOptions = append(merged.GenerateOptions, other.GenerateOptions...)
	// If other has a StreamHandler, it takes precedence
	if other.StreamHandler != nil {
		merged.StreamHandler = other.StreamHandler
	}
	return merged
}

func WithOptions(opts ...Option) Option {
	return func(o *ModuleOptions) {
		for _, opt := range opts {
			opt(o)
		}
	}
}

// BaseModule provides a basic implementation of the Module interface.
type BaseModule struct {
	Signature    Signature
	LLM          LLM
	DisplayName  string
	ModuleType   string
	interceptors []ModuleInterceptor // Default interceptors for this module instance
}

// GetSignature returns the module's signature.
func (bm *BaseModule) GetSignature() Signature {
	return bm.Signature
}

// SetLLM sets the language model for the module.
func (bm *BaseModule) SetLLM(llm LLM) {
	bm.LLM = llm
}

func (bm *BaseModule) SetSignature(signature Signature) {
	bm.Signature = signature
}

// GetDisplayName returns the human-readable name for this module instance.
func (bm *BaseModule) GetDisplayName() string {
	if bm.DisplayName != "" {
		return bm.DisplayName
	}
	return bm.GetModuleType()
}

// GetModuleType returns the category/type of this module.
func (bm *BaseModule) GetModuleType() string {
	if bm.ModuleType != "" {
		return bm.ModuleType
	}
	return "BaseModule"
}

// Process is a placeholder implementation and should be overridden by specific modules.
func (bm *BaseModule) Process(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
	return nil, errors.New("Process method not implemented")
}

// Clone creates a deep copy of the BaseModule.
func (bm *BaseModule) Clone() Module {
	// Deep copy interceptors
	interceptors := make([]ModuleInterceptor, len(bm.interceptors))
	copy(interceptors, bm.interceptors)

	return &BaseModule{
		Signature:    bm.Signature,
		LLM:          bm.LLM, // Note: This is a shallow copy of the LLM
		DisplayName:  bm.DisplayName,
		ModuleType:   bm.ModuleType,
		interceptors: interceptors,
	}
}

// ProcessWithInterceptors executes the module's logic with interceptor support.
// If no interceptors are provided, it falls back to using the module's default interceptors.
// Note: This method should be called on the concrete module type, not BaseModule directly.
func (bm *BaseModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, opts ...Option) (map[string]any, error) {
	// This is a fallback implementation that uses the BaseModule's Process method.
	// Concrete types should override this method to call their own Process implementation.
	return bm.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, bm.Process, opts...)
}

// ProcessWithInterceptorsImpl is a helper method that implements the interceptor logic.
// It accepts a process function as a parameter to allow concrete types to pass their own Process method.
func (bm *BaseModule) ProcessWithInterceptorsImpl(ctx context.Context, inputs map[string]any, interceptors []ModuleInterceptor, processFunc func(context.Context, map[string]any, ...Option) (map[string]any, error), opts ...Option) (map[string]any, error) {
	// Use provided interceptors, or fall back to module's default interceptors
	if interceptors == nil {
		interceptors = bm.interceptors
	}

	// Create module info for interceptors
	info := NewModuleInfo(bm.GetDisplayName(), bm.GetModuleType(), bm.GetSignature())

	// Create the base handler that calls the provided process function
	handler := func(ctx context.Context, inputs map[string]any, opts ...Option) (map[string]any, error) {
		return processFunc(ctx, inputs, opts...)
	}

	// Chain the interceptors
	chainedInterceptor := ChainModuleInterceptors(interceptors...)

	// Execute with interceptors
	return chainedInterceptor(ctx, inputs, info, handler, opts...)
}

// SetInterceptors sets the default interceptors for this module instance.
func (bm *BaseModule) SetInterceptors(interceptors []ModuleInterceptor) {
	bm.interceptors = make([]ModuleInterceptor, len(interceptors))
	copy(bm.interceptors, interceptors)
}

// GetInterceptors returns the current interceptors for this module.
func (bm *BaseModule) GetInterceptors() []ModuleInterceptor {
	result := make([]ModuleInterceptor, len(bm.interceptors))
	copy(result, bm.interceptors)
	return result
}

// ClearInterceptors removes all interceptors from this module.
func (bm *BaseModule) ClearInterceptors() {
	bm.interceptors = nil
}

// NewModule creates a new base module with the given signature.
func NewModule(signature Signature) *BaseModule {
	return &BaseModule{
		Signature:    signature,
		interceptors: make([]ModuleInterceptor, 0),
	}
}

// ValidateInputs checks if the provided inputs match the module's input signature.
func (bm *BaseModule) ValidateInputs(inputs map[string]any) error {
	for _, field := range bm.Signature.Inputs {
		if _, ok := inputs[field.Name]; !ok {
			return errors.New("missing required input: " + field.Name)
		}
	}
	return nil
}

// FormatOutputs ensures that the output map contains all fields specified in the output signature.
func (bm *BaseModule) FormatOutputs(outputs map[string]any) map[string]any {
	formattedOutputs := make(map[string]any)
	for _, field := range bm.Signature.Outputs {
		if value, ok := outputs[field.Name]; ok {
			formattedOutputs[field.Name] = value
		} else {
			formattedOutputs[field.Name] = nil
		}
	}
	return formattedOutputs
}

// Composable is an interface for modules that can be composed with other modules.
type Composable interface {
	Module
	Compose(next Module) Module
	GetSubModules() []Module
	SetSubModules([]Module)
}

// ModuleChain represents a chain of modules.
type ModuleChain struct {
	BaseModule
	Modules []Module
}

// NewModuleChain creates a new module chain.
func NewModuleChain(modules ...Module) *ModuleChain {
	// Compute the combined signature
	var inputs []InputField
	var outputs []OutputField
	for i, m := range modules {
		sig := m.GetSignature()
		if i == 0 {
			inputs = sig.Inputs
		}
		if i == len(modules)-1 {
			outputs = sig.Outputs
		}
	}

	return &ModuleChain{
		BaseModule: BaseModule{
			Signature: Signature{Inputs: inputs, Outputs: outputs},
		},
		Modules: modules,
	}
}

// ProcessTyped provides type-safe module processing using generics.
func ProcessTyped[TInput, TOutput any](ctx context.Context, module Module, inputs TInput, opts ...Option) (TOutput, error) {
	var zero TOutput

	// Convert typed inputs to legacy format
	legacyInputs, err := utils.ConvertTypedInputsToLegacy(inputs)
	if err != nil {
		return zero, err
	}

	// Call the legacy Process method
	legacyOutputs, err := module.Process(ctx, legacyInputs, opts...)
	if err != nil {
		return zero, err
	}

	// Convert legacy outputs to typed format
	typedOutputs, err := utils.ConvertLegacyOutputsToTyped[TOutput](legacyOutputs)
	if err != nil {
		return zero, err
	}

	return typedOutputs, nil
}

// ProcessTypedWithValidation provides type-safe processing with signature validation.
func ProcessTypedWithValidation[TInput, TOutput any](ctx context.Context, module Module, inputs TInput, opts ...Option) (TOutput, error) {
	var zero TOutput

	// Create typed signature for validation
	typedSig := NewTypedSignature[TInput, TOutput]()

	// Validate inputs
	if err := typedSig.ValidateInput(inputs); err != nil {
		return zero, err
	}

	// Process with type conversion
	result, err := ProcessTyped[TInput, TOutput](ctx, module, inputs, opts...)
	if err != nil {
		return zero, err
	}

	// Validate outputs
	if err := typedSig.ValidateOutput(result); err != nil {
		return zero, err
	}

	return result, nil
}
