package decide

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

// SystemOneClient is the narrow provider capability required by Decide.
type SystemOneClient interface {
	SystemOne(context.Context, typesafe.SystemOneRequest) (*typesafe.SystemOneResponse, error)
}

// Result contains native outputs and evidence derived from the same provider
// response. Decisions are snapshots; mutating their exported maps does not
// alter module parameters or later results.
type Result struct {
	Outputs   map[string]any
	Decisions map[string]Decision
	Model     string
	Usage     typesafe.Usage
	RequestID string
}

// Decide is a closed-set core.Module backed by a System One client. It is a
// sibling of modules.Predict: it has no prompts, demonstrations, or text LLM.
type Decide struct {
	mu           sync.RWMutex
	client       SystemOneClient
	signature    core.Signature
	signatureErr error
	outputs      []Output
	displayName  string
}

var _ core.Module = (*Decide)(nil)
var _ core.ParameterProvider = (*Decide)(nil)
var _ core.ParameterConsumer = (*Decide)(nil)

// New constructs a Decide module. Every signature output must have exactly one
// matching declaration.
func New(client SystemOneClient, signature core.Signature, outputs ...Output) (*Decide, error) {
	if client == nil || isNilClient(client) {
		return nil, fmt.Errorf("decide: System One client must not be nil")
	}
	clonedOutputs := make([]Output, len(outputs))
	for i, output := range outputs {
		if output == nil {
			return nil, fmt.Errorf("decide: output declaration %d is nil", i)
		}
		clonedOutputs[i] = output.clone()
		if err := clonedOutputs[i].validate(); err != nil {
			return nil, fmt.Errorf("decide: %w", err)
		}
	}
	if err := validateSignature(signature, clonedOutputs); err != nil {
		return nil, fmt.Errorf("decide: %w", err)
	}
	return &Decide{
		client:      client,
		signature:   cloneSignature(signature),
		outputs:     clonedOutputs,
		displayName: "Decide",
	}, nil
}

// WithName sets a human-readable module name.
func (d *Decide) WithName(name string) *Decide {
	d.mu.Lock()
	defer d.mu.Unlock()
	if strings.TrimSpace(name) == "" {
		d.displayName = "Decide"
	} else {
		d.displayName = name
	}
	return d
}

// Process returns native values for ordinary program composition. Use
// ProcessDecision when provider evidence is needed.
func (d *Decide) Process(ctx context.Context, inputs map[string]any, _ ...core.Option) (map[string]any, error) {
	result, err := d.ProcessDecision(ctx, inputs)
	if err != nil {
		return nil, err
	}
	return maps.Clone(result.Outputs), nil
}

// ProcessDecision returns native outputs plus typed evidence from one request.
func (d *Decide) ProcessDecision(ctx context.Context, inputs map[string]any) (*Result, error) {
	if ctx == nil {
		return nil, fmt.Errorf("decide: context must not be nil")
	}
	signature, declared, client, signatureErr := d.snapshot()
	if signatureErr != nil {
		return nil, signatureErr
	}
	if err := validateInputs(signature, inputs); err != nil {
		return nil, err
	}

	state := make(map[string]any, len(signature.Inputs))
	for _, input := range signature.Inputs {
		state[input.Name] = inputs[input.Name]
	}
	questions := make(map[string]typesafe.Question, len(declared))
	outputFields := make(map[string]core.OutputField, len(signature.Outputs))
	for _, field := range signature.Outputs {
		outputFields[field.Name] = field
	}
	for _, output := range declared {
		questions[output.outputName()] = output.question(questionInstructions(signature, outputFields[output.outputName()]))
	}

	response, err := client.SystemOne(ctx, typesafe.SystemOneRequest{State: state, Questions: questions})
	if err != nil {
		return nil, fmt.Errorf("decide: System One request failed: %w", err)
	}
	if response == nil {
		return nil, fmt.Errorf("decide: System One client returned a nil response")
	}
	if len(response.Answers) != len(declared) {
		return nil, fmt.Errorf("decide: response answer names do not match declared outputs")
	}

	result := &Result{
		Outputs:   make(map[string]any, len(declared)),
		Decisions: make(map[string]Decision, len(declared)),
		Model:     response.Model,
		Usage:     response.Usage,
		RequestID: response.RequestID,
	}
	for _, output := range declared {
		name := output.outputName()
		answer, found := response.Answers[name]
		if !found || answer == nil {
			return nil, fmt.Errorf("decide: response is missing answer %q", name)
		}
		decision, native, err := output.decode(answer)
		if err != nil {
			return nil, fmt.Errorf("decide: %w", err)
		}
		result.Decisions[name] = decision
		result.Outputs[name] = native
	}
	return result, nil
}

// GetSignature returns an independent copy of the current signature.
func (d *Decide) GetSignature() core.Signature {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return cloneSignature(d.signature)
}

// SetSignature applies instruction, input, and field-description changes when
// output names remain compatible. Because core.Module cannot return an error
// here, an incompatible signature is recorded and the next Process call fails
// before contacting the provider. Setting a compatible signature clears it.
func (d *Decide) SetSignature(signature core.Signature) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.signature = cloneSignature(signature)
	if err := validateSignature(signature, d.outputs); err != nil {
		d.signatureErr = fmt.Errorf("decide: incompatible signature: %w", err)
	} else {
		d.signatureErr = nil
	}
}

// SetLLM is intentionally a no-op. Decide is backed by its explicitly supplied
// System One client, and a program-wide generative LLM must not replace it.
func (*Decide) SetLLM(core.LLM) {}

// Clone creates an independent copy of signatures and local parameters. The
// concurrency-safe System One client is intentionally shared.
func (d *Decide) Clone() core.Module {
	d.mu.RLock()
	defer d.mu.RUnlock()
	outputs := make([]Output, len(d.outputs))
	for i, output := range d.outputs {
		outputs[i] = output.clone()
	}
	return &Decide{
		client:       d.client,
		signature:    cloneSignature(d.signature),
		signatureErr: d.signatureErr,
		outputs:      outputs,
		displayName:  d.displayName,
	}
}

// GetDisplayName returns the configured name or "Decide".
func (d *Decide) GetDisplayName() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.displayName == "" {
		return "Decide"
	}
	return d.displayName
}

// GetModuleType identifies this module without treating it as Predict.
func (*Decide) GetModuleType() string { return "Decide" }

func (d *Decide) snapshot() (core.Signature, []Output, SystemOneClient, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	outputs := make([]Output, len(d.outputs))
	for i, output := range d.outputs {
		outputs[i] = output.clone()
	}
	return cloneSignature(d.signature), outputs, d.client, d.signatureErr
}

func validateSignature(signature core.Signature, outputs []Output) error {
	if len(signature.Outputs) == 0 {
		return fmt.Errorf("signature requires at least one output")
	}
	if len(signature.Outputs) != len(outputs) {
		return fmt.Errorf("signature has %d outputs but %d decisions were declared", len(signature.Outputs), len(outputs))
	}
	inputNames := make(map[string]struct{}, len(signature.Inputs))
	for _, input := range signature.Inputs {
		if strings.TrimSpace(input.Name) == "" {
			return fmt.Errorf("signature input names must not be empty")
		}
		if _, duplicate := inputNames[input.Name]; duplicate {
			return fmt.Errorf("signature contains duplicate input %q", input.Name)
		}
		inputNames[input.Name] = struct{}{}
	}
	declared := make(map[string]struct{}, len(outputs))
	for _, output := range outputs {
		name := output.outputName()
		if _, duplicate := declared[name]; duplicate {
			return fmt.Errorf("decision output %q was declared more than once", name)
		}
		declared[name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(signature.Outputs))
	for _, field := range signature.Outputs {
		if strings.TrimSpace(field.Name) == "" {
			return fmt.Errorf("signature output names must not be empty")
		}
		if _, duplicate := seen[field.Name]; duplicate {
			return fmt.Errorf("signature contains duplicate output %q", field.Name)
		}
		seen[field.Name] = struct{}{}
		if _, found := declared[field.Name]; !found {
			return fmt.Errorf("signature output %q has no decision declaration", field.Name)
		}
	}
	return nil
}

func validateInputs(signature core.Signature, inputs map[string]any) error {
	for _, input := range signature.Inputs {
		if _, found := inputs[input.Name]; !found {
			return fmt.Errorf("decide: missing required input %q", input.Name)
		}
	}
	return nil
}

func questionInstructions(signature core.Signature, output core.OutputField) any {
	question := output.Description
	if question == "" {
		question = fmt.Sprintf("Decide `%s`.", output.Name)
	}
	instructions := map[string]any{"question": question}
	if signature.Instruction != "" {
		instructions["task"] = signature.Instruction
	}
	if len(signature.Inputs) > 0 {
		inputs := make([]map[string]string, len(signature.Inputs))
		for i, input := range signature.Inputs {
			inputs[i] = map[string]string{"name": input.Name}
			if input.Description != "" {
				inputs[i]["description"] = input.Description
			}
		}
		instructions["inputs"] = inputs
	}
	return instructions
}

func cloneSignature(signature core.Signature) core.Signature {
	signature.Inputs = slices.Clone(signature.Inputs)
	signature.Outputs = slices.Clone(signature.Outputs)
	return signature
}

func isNilClient(client SystemOneClient) bool {
	value := reflect.ValueOf(client)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
