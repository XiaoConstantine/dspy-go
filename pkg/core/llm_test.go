package core

import (
	"context"
	"testing"
)

// TestLLMRegistry tests the LLMRegistry.
//
//	func TestLLMRegistry(t *testing.T) {
//		registry := NewLLMRegistry()
//
//		// Test registering an LLM
//		registry.Register("test", func() (LLM, error) {
//			return &MockLLM{}, nil
//		})
//
//		// Test creating a registered LLM
//		llm, err := registry.Create("test")
//		if err != nil {
//			t.Errorf("Unexpected error creating LLM: %v", err)
//		}
//		if _, ok := llm.(*MockLLM); !ok {
//			t.Error("Created LLM is not of expected type")
//		}
//
//		// Test creating an unregistered LLM
//		_, err = registry.Create("nonexistent")
//		if err == nil {
//			t.Error("Expected error when creating unregistered LLM, got nil")
//		}
//	}
//
// // TestGenerateOptions tests the GenerateOptions and related functions.
func TestGenerateOptions(t *testing.T) {
	opts := &GenerateOptions{}

	WithMaxTokens(100)(opts)
	if opts.MaxTokens != 100 {
		t.Errorf("Expected MaxTokens 100, got %d", opts.MaxTokens)
	}

	WithTemperature(0.7)(opts)
	if opts.Temperature != 0.7 {
		t.Errorf("Expected Temperature 0.7, got %f", opts.Temperature)
	}

	WithTopP(0.9)(opts)
	if opts.TopP != 0.9 {
		t.Errorf("Expected TopP 0.9, got %f", opts.TopP)
	}

	WithPresencePenalty(1.0)(opts)
	if opts.PresencePenalty != 1.0 {
		t.Errorf("Expected PresencePenalty 1.0, got %f", opts.PresencePenalty)
	}

	WithFrequencyPenalty(1.5)(opts)
	if opts.FrequencyPenalty != 1.5 {
		t.Errorf("Expected FrequencyPenalty 1.5, got %f", opts.FrequencyPenalty)
	}

	WithStopSequences("stop1", "stop2")(opts)
	if len(opts.Stop) != 2 || opts.Stop[0] != "stop1" || opts.Stop[1] != "stop2" {
		t.Errorf("Expected Stop sequences [stop1 stop2], got %v", opts.Stop)
	}
}

// TestMockLLM tests the MockLLM implementation.
func TestMockLLM(t *testing.T) {
	llm := &MockLLM{}

	response, err := llm.Generate(context.Background(), "test prompt")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if response.Content != "mock response" {
		t.Errorf("Expected 'mock response', got '%s'", response.Content)
	}

	jsonResponse, err := llm.GenerateWithJSON(context.Background(), "test prompt")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if jsonResponse["response"] != "mock response" {
		t.Errorf("Expected {'response': 'mock response'}, got %v", jsonResponse)
	}
}
