package core

import "testing"

func TestSignature(t *testing.T) {
	inputs := []InputField{
		{Field: Field{Name: "input1", Description: "First input"}},
		{Field: Field{Name: "input2", Description: "Second input"}},
	}
	outputs := []OutputField{
		{Field: Field{Name: "output1", Description: "First output"}},
	}

	sig := NewSignature(inputs, outputs)

	if len(sig.Inputs) != 2 {
		t.Errorf("Expected 2 inputs, got %d", len(sig.Inputs))
	}
	if len(sig.Outputs) != 1 {
		t.Errorf("Expected 1 output, got %d", len(sig.Outputs))
	}

	sigWithInstruction := sig.WithInstruction("Test instruction")
	if sigWithInstruction.Instruction != "Test instruction" {
		t.Errorf("Expected instruction 'Test instruction', got '%s'", sigWithInstruction.Instruction)
	}
}
