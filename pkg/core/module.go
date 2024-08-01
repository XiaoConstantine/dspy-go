package core

import (
	"context"
	"errors"
)

// Module represents a basic unit of computation in DSPy.
type Module interface {
	// Process executes the module's logic
	Process(ctx context.Context, inputs map[string]any) (map[string]any, error)

	// GetSignature returns the module's input and output signature
	GetSignature() Signature

	// SetLLM sets the language model for the module
	SetLLM(llm LLM)

	// Clone creates a deep copy of the module
	Clone() Module
}

// BaseModule provides a basic implementation of the Module interface.
type BaseModule struct {
	Signature Signature
	LLM       LLM
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

// Process is a placeholder implementation and should be overridden by specific modules.
func (bm *BaseModule) Process(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	return nil, errors.New("Process method not implemented")
}

// Clone creates a deep copy of the BaseModule.
func (bm *BaseModule) Clone() Module {
	return &BaseModule{
		Signature: bm.Signature,
		LLM:       bm.LLM, // Note: This is a shallow copy of the LLM
	}
}

// NewModule creates a new base module with the given signature.
func NewModule(signature Signature) *BaseModule {
	return &BaseModule{
		Signature: signature,
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
