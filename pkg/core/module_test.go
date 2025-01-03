package core

import (
	"context"
	"reflect"
	"testing"
)

// MockLLM is a mock implementation of the LLM interface for testing.
type MockLLM struct{}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	return &LLMResponse{Content: "mock response"}, nil
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"response": "mock response"}, nil
}

// TestBaseModule tests the BaseModule struct and its methods.
func TestBaseModule(t *testing.T) {
	sig := NewSignature(
		[]InputField{{Field: Field{Name: "input"}}},
		[]OutputField{{Field: Field{Name: "output"}}},
	)
	bm := NewModule(sig)

	if !reflect.DeepEqual(bm.GetSignature(), sig) {
		t.Error("GetSignature did not return the correct signature")
	}

	mockLLM := &MockLLM{}
	bm.SetLLM(mockLLM)
	if bm.LLM != mockLLM {
		t.Error("SetLLM did not set the LLM correctly")
	}

	_, err := bm.Process(context.Background(), map[string]any{"input": "test"})
	if err == nil || err.Error() != "Process method not implemented" {
		t.Error("Expected 'Process method not implemented' error")
	}

	clone := bm.Clone()
	if !reflect.DeepEqual(clone.GetSignature(), bm.GetSignature()) {
		t.Error("Cloned module does not have the same signature")
	}
}

// TestModuleChain tests the ModuleChain struct and its methods.
func TestModuleChain(t *testing.T) {
	module1 := NewModule(NewSignature(
		[]InputField{{Field: Field{Name: "input1"}}},
		[]OutputField{{Field: Field{Name: "output1"}}},
	))
	module2 := NewModule(NewSignature(
		[]InputField{{Field: Field{Name: "input2"}}},
		[]OutputField{{Field: Field{Name: "output2"}}},
	))

	chain := NewModuleChain(module1, module2)

	if len(chain.Modules) != 2 {
		t.Errorf("Expected 2 modules in chain, got %d", len(chain.Modules))
	}

	sig := chain.GetSignature()
	if len(sig.Inputs) != 1 || sig.Inputs[0].Name != "input1" {
		t.Error("Chain signature inputs are incorrect")
	}
	if len(sig.Outputs) != 1 || sig.Outputs[0].Name != "output2" {
		t.Error("Chain signature outputs are incorrect")
	}
}
