package core

import (
	"context"
	"errors"
	"reflect"
)

// Program represents a complete DSPy pipeline or workflow.
type Program struct {
	Modules map[string]Module
	Forward func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)
}

// NewProgram creates a new Program with the given modules and forward function.
func NewProgram(modules map[string]Module, forward func(context.Context, map[string]interface{}) (map[string]interface{}, error)) Program {
	return Program{
		Modules: modules,
		Forward: forward,
	}
}

// Execute runs the program with the given inputs.
func (p Program) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	if p.Forward == nil {
		return nil, errors.New("forward function is not defined")
	}

	ctx = WithTraceManager(ctx)
	tm := GetTraceManager(ctx)
	trace := tm.StartTrace("Program", "Program")
	defer tm.EndTrace()

	trace.SetInputs(inputs)
	outputs, err := p.Forward(ctx, inputs)
	if err != nil {
		trace.SetError(err)
		return nil, err
	}
	trace.SetOutputs(outputs)

	return outputs, nil
}

// GetSignature returns the overall signature of the program
// This would need to be defined based on the Forward function's expected inputs and outputs.
func (p Program) GetSignature() Signature {
	var inputs []InputField
	var outputs []OutputField

	// Since modules are in a map, we can't rely on order.
	// We'll use the first module we find for inputs and outputs.
	for _, module := range p.Modules {
		sig := module.GetSignature()
		inputs = sig.Inputs
		outputs = sig.Outputs
		break
	}
	return NewSignature(inputs, outputs)
}

// Clone creates a deep copy of the Program.
func (p Program) Clone() Program {
	modulesCopy := make(map[string]Module)
	for name, module := range p.Modules {
		modulesCopy[name] = module.Clone()
	}

	return Program{
		Modules: modulesCopy,
		Forward: p.Forward, // Note: We're copying the pointer to the forward function
	}
}

// Equal checks if two Programs are equivalent.
func (p Program) Equal(other Program) bool {
	if p.Forward == nil && other.Forward != nil || p.Forward != nil && other.Forward == nil {
		return false
	}
	if len(p.Modules) != len(other.Modules) {
		return false
	}
	for name, module := range p.Modules {
		otherModule, exists := other.Modules[name]
		if !exists {
			return false
		}
		if !reflect.DeepEqual(module.GetSignature(), otherModule.GetSignature()) {
			return false
		}
	}
	return true
}

// AddModule adds a new module to the Program.
func (p *Program) AddModule(name string, module Module) {
	p.Modules[name] = module
}

// SetForward sets the forward function for the Program.
func (p *Program) SetForward(forward func(context.Context, map[string]interface{}) (map[string]interface{}, error)) {
	p.Forward = forward
}
