package agents

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// AgentModule adapts a reusable Harness to DSPy's core.Module composition
// contract. Named signature inputs are mapped to Prompt.Fields.
type AgentModule struct {
	mu           sync.Mutex
	harness      *Harness
	signature    core.Signature
	displayName  string
	signatureErr error
	modelErr     error
}

var _ core.Module = (*AgentModule)(nil)

// NewAgentModule creates a module backed by an existing Harness.
func NewAgentModule(harness *Harness, signature core.Signature) (*AgentModule, error) {
	if harness == nil {
		return nil, fmt.Errorf("harness is required")
	}
	if err := validateAgentModuleSignature(signature); err != nil {
		return nil, err
	}
	source, err := instructionSourceFromProvider(signature)
	if err != nil {
		return nil, err
	}
	if err := harness.replaceInstructions(source); err != nil {
		return nil, err
	}
	return &AgentModule{harness: harness, signature: cloneSignature(signature), displayName: "AgentModule"}, nil
}

// Process invokes the harness with named signature fields.
func (m *AgentModule) Process(ctx context.Context, inputs map[string]any, options ...core.Option) (map[string]any, error) {
	if m == nil {
		return nil, fmt.Errorf("agent module is nil")
	}
	if len(options) > 0 {
		return nil, fmt.Errorf("agent module does not support per-call core.Option values")
	}
	m.mu.Lock()
	configErr := errors.Join(m.signatureErr, m.modelErr)
	signature := cloneSignature(m.signature)
	harness := m.harness.fresh()
	m.mu.Unlock()
	if configErr != nil {
		return nil, configErr
	}
	result, err := harness.Prompt(ctx, Prompt{Fields: cloneAnyMap(inputs)})
	if err != nil {
		return nil, err
	}
	output := make(map[string]any, len(signature.Outputs))
	answerAssigned := false
	for _, field := range signature.Outputs {
		switch strings.ToLower(strings.TrimSpace(field.Name)) {
		case "completed":
			output[field.Name] = result.StopReason == StopReasonFinish || result.StopReason == StopReasonText
		case "status":
			if result.StopReason == StopReasonFinish || result.StopReason == StopReasonText {
				output[field.Name] = string(RunStatusCompleted)
			} else {
				output[field.Name] = string(RunStatusStopped)
			}
		case "stop_reason":
			output[field.Name] = string(result.StopReason)
		default:
			if !answerAssigned {
				output[field.Name] = result.FinalAnswer
				answerAssigned = true
			}
		}
	}
	return output, nil
}

// GetSignature returns an ownership-safe signature copy.
func (m *AgentModule) GetSignature() core.Signature {
	if m == nil {
		return core.Signature{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return cloneSignature(m.signature)
}

// SetSignature replaces pre-run signature instructions and field mapping.
func (m *AgentModule) SetSignature(signature core.Signature) {
	if m == nil {
		return
	}
	err := validateAgentModuleSignature(signature)
	var source InstructionSource
	if err == nil {
		source, err = instructionSourceFromProvider(signature)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if err == nil {
		err = m.harness.replaceInstructions(source)
	}
	if err != nil {
		m.signatureErr = fmt.Errorf("set agent module signature: %w", err)
		return
	}
	m.signature = cloneSignature(signature)
	m.signatureErr = nil
}

// SetLLM replaces the Harness model for subsequent runs.
func (m *AgentModule) SetLLM(llm core.LLM) {
	if m == nil {
		return
	}
	model, err := NewLLMAdapter(llm)
	m.mu.Lock()
	defer m.mu.Unlock()
	if err == nil {
		err = m.harness.replaceModel(model)
	}
	if err != nil {
		m.modelErr = fmt.Errorf("set agent module LLM: %w", err)
	} else {
		m.modelErr = nil
	}
}

// Clone creates an independently stateful module snapshot.
func (m *AgentModule) Clone() core.Module {
	if m == nil {
		return (*AgentModule)(nil)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return &AgentModule{
		harness: m.harness.fresh(), signature: cloneSignature(m.signature),
		displayName: m.displayName, signatureErr: m.signatureErr, modelErr: m.modelErr,
	}
}

func (m *AgentModule) GetDisplayName() string {
	if m == nil {
		return "AgentModule"
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.displayName
}

func (*AgentModule) GetModuleType() string { return "agent_harness" }

func validateAgentModuleSignature(signature core.Signature) error {
	if len(signature.Inputs) == 0 {
		return fmt.Errorf("agent module signature requires at least one input")
	}
	if len(signature.Outputs) == 0 {
		return fmt.Errorf("agent module signature requires at least one output")
	}
	answerFields := 0
	for _, field := range signature.Outputs {
		switch strings.ToLower(strings.TrimSpace(field.Name)) {
		case "completed", "status", "stop_reason":
		default:
			answerFields++
		}
	}
	if answerFields != 1 {
		return fmt.Errorf("agent module signature requires exactly one answer output, got %d", answerFields)
	}
	return nil
}
