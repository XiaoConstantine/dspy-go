package core

import (
	"context"
	"errors"
	"reflect"
	"sort"
)

// ForwardFactory rebuilds a program forward function against a specific module map.
// Programs created with a factory can safely rebind their forward logic during Clone().
type ForwardFactory func(modules map[string]Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error)

// Program represents a complete DSPy pipeline or workflow.
type Program struct {
	Modules map[string]Module
	Forward func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error)

	forwardFactory ForwardFactory
}

// NewProgram creates a new Program with the given modules and forward function.
func NewProgram(modules map[string]Module, forward func(context.Context, map[string]interface{}) (map[string]interface{}, error)) Program {
	return Program{
		Modules: modules,
		Forward: forward,
	}
}

// NewProgramWithForwardFactory creates a program whose forward function can be
// safely rebound against cloned or replaced modules.
func NewProgramWithForwardFactory(modules map[string]Module, factory ForwardFactory) Program {
	program := Program{
		Modules:        modules,
		forwardFactory: factory,
	}
	if factory != nil {
		program.Forward = factory(modules)
	}
	return program
}

// Execute runs the program with the given inputs.
func (p Program) Execute(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	if p.Forward == nil {
		return nil, errors.New("forward function is not defined")
	}
	// Ensure we have execution state
	if GetExecutionState(ctx) == nil {
		ctx = WithExecutionState(ctx)
	}

	ctx, span := StartSpan(ctx, "Program")
	defer EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)
	outputs, err := p.Forward(ctx, inputs)
	if err != nil {
		span.WithError(err)
		return nil, err
	}

	span.WithAnnotation("outputs", outputs)
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

	return p.RebindModules(modulesCopy)
}

// RebindModules returns a new program with the provided module map.
// When a forward factory is available, the returned program rebuilds its forward
// function against the supplied modules; otherwise it preserves the original
// forward function pointer.
func (p Program) RebindModules(modules map[string]Module) Program {
	rebound := Program{
		Modules:        modules,
		Forward:        p.Forward,
		forwardFactory: p.forwardFactory,
	}
	if p.forwardFactory != nil {
		rebound.Forward = p.forwardFactory(modules)
	}
	return rebound
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
	p.forwardFactory = nil
}

// SetForwardFactory sets a clone-safe forward factory for the Program.
func (p *Program) SetForwardFactory(factory ForwardFactory) {
	p.forwardFactory = factory
	if factory != nil {
		p.Forward = factory(p.Modules)
	} else {
		p.Forward = nil
	}
}

// GetModules returns all modules in deterministic alphabetical order by name.
// This ensures consistent behavior across operations that iterate over modules.
func (p *Program) GetModules() []Module {
	moduleNames := make([]string, 0, len(p.Modules))
	for name := range p.Modules {
		moduleNames = append(moduleNames, name)
	}
	sort.Strings(moduleNames)

	modules := make([]Module, len(moduleNames))
	for i, name := range moduleNames {
		modules[i] = p.Modules[name]
	}
	return modules
}
