package agents

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// AgentField describes one named agent input or output without prescribing a
// transport representation.
type AgentField struct {
	Name        string
	Description string
	Required    bool
}

// AgentContract describes the named inputs and outputs accepted by an agent.
// PrimaryInput identifies the field used for one unlabelled text prompt.
type AgentContract struct {
	Inputs       []AgentField
	Outputs      []AgentField
	PrimaryInput string
}

// AgentContractProvider is the optional composition contract implemented by
// agents that accept named inputs other than legacy question.
type AgentContractProvider interface {
	AgentContract() AgentContract
}

// AgentInputAdapter optionally normalizes transport-produced named input before
// Agent.Execute is called.
type AgentInputAdapter interface {
	AdaptAgentInput(map[string]any) (map[string]any, error)
}

// AgentExecutionResult keeps output and trace correlated to one execution.
type AgentExecutionResult struct {
	Output map[string]any
	Trace  *ExecutionTrace
}

// ScopedExecutionAgent optionally returns an execution-scoped trace instead of
// requiring consumers to race against a mutable LastExecutionTrace accessor.
type ScopedExecutionAgent interface {
	ExecuteWithTrace(context.Context, map[string]any) (AgentExecutionResult, error)
}

type executionTraceCaptureKey struct{}

type executionTraceCapture struct {
	mu    sync.Mutex
	trace *ExecutionTrace
}

// WithExecutionTraceCapture creates an operation-local trace capture used by
// agent wrappers implementing ScopedExecutionAgent.
func WithExecutionTraceCapture(ctx context.Context) (context.Context, func() *ExecutionTrace) {
	capture := &executionTraceCapture{}
	captured := context.WithValue(ctx, executionTraceCaptureKey{}, capture)
	return captured, func() *ExecutionTrace {
		capture.mu.Lock()
		defer capture.mu.Unlock()
		return capture.trace.Clone()
	}
}

// PublishExecutionTrace records a trace in an operation-local capture, if one
// is present in ctx.
func PublishExecutionTrace(ctx context.Context, trace *ExecutionTrace) {
	capture, _ := ctx.Value(executionTraceCaptureKey{}).(*executionTraceCapture)
	if capture == nil {
		return
	}
	capture.mu.Lock()
	capture.trace = trace.Clone()
	capture.mu.Unlock()
}

// ContractFromSignature converts a DSPy signature into an agent composition
// contract without inventing question or task field names.
func ContractFromSignature(signature core.Signature) AgentContract {
	contract := AgentContract{
		Inputs:  make([]AgentField, 0, len(signature.Inputs)),
		Outputs: make([]AgentField, 0, len(signature.Outputs)),
	}
	for _, input := range signature.Inputs {
		contract.Inputs = append(contract.Inputs, AgentField{
			Name: input.Name, Description: input.Description, Required: true,
		})
	}
	for _, output := range signature.Outputs {
		contract.Outputs = append(contract.Outputs, AgentField{
			Name: output.Name, Description: output.Description, Required: true,
		})
	}
	if len(contract.Inputs) > 0 {
		contract.PrimaryInput = contract.Inputs[0].Name
	}
	return contract
}

// Validate checks that field names and the primary input are coherent.
func (c AgentContract) Validate() error {
	seen := make(map[string]struct{}, len(c.Inputs))
	for _, field := range c.Inputs {
		name := strings.TrimSpace(field.Name)
		if name == "" {
			return fmt.Errorf("agent input field name is required")
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("duplicate agent input field %q", name)
		}
		seen[name] = struct{}{}
	}
	if primary := strings.TrimSpace(c.PrimaryInput); primary != "" {
		if _, exists := seen[primary]; !exists {
			return fmt.Errorf("primary input %q is not an input field", primary)
		}
	}
	return nil
}
