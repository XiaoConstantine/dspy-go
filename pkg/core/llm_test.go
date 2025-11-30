package core

import (
	"context"
	"testing"
	"time"
)

// TestGenerateOptions tests the GenerateOptions and related functions.
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

// TestTransportConfig_ToTransport tests the TransportConfig.ToTransport method.
func TestTransportConfig_ToTransport(t *testing.T) {
	config := TransportConfig{
		MaxIdleConns:        50,
		MaxIdleConnsPerHost: 25,
		MaxConnsPerHost:     50,
		IdleConnTimeout:     60 * time.Second,
		TLSHandshakeTimeout: 5 * time.Second,
	}

	transport := config.ToTransport()

	if transport.MaxIdleConns != 50 {
		t.Errorf("Expected MaxIdleConns 50, got %d", transport.MaxIdleConns)
	}
	if transport.MaxIdleConnsPerHost != 25 {
		t.Errorf("Expected MaxIdleConnsPerHost 25, got %d", transport.MaxIdleConnsPerHost)
	}
	if transport.MaxConnsPerHost != 50 {
		t.Errorf("Expected MaxConnsPerHost 50, got %d", transport.MaxConnsPerHost)
	}
	if transport.IdleConnTimeout != 60*time.Second {
		t.Errorf("Expected IdleConnTimeout 60s, got %v", transport.IdleConnTimeout)
	}
	if transport.TLSHandshakeTimeout != 5*time.Second {
		t.Errorf("Expected TLSHandshakeTimeout 5s, got %v", transport.TLSHandshakeTimeout)
	}
}

// TestDefaultTransportConfig tests the DefaultTransportConfig function.
func TestDefaultTransportConfig(t *testing.T) {
	config := DefaultTransportConfig()

	if config.MaxIdleConns != 100 {
		t.Errorf("Expected MaxIdleConns 100, got %d", config.MaxIdleConns)
	}
	if config.MaxIdleConnsPerHost != 100 {
		t.Errorf("Expected MaxIdleConnsPerHost 100, got %d", config.MaxIdleConnsPerHost)
	}
	if config.MaxConnsPerHost != 100 {
		t.Errorf("Expected MaxConnsPerHost 100, got %d", config.MaxConnsPerHost)
	}
	if config.IdleConnTimeout != 90*time.Second {
		t.Errorf("Expected IdleConnTimeout 90s, got %v", config.IdleConnTimeout)
	}
	if config.TLSHandshakeTimeout != 10*time.Second {
		t.Errorf("Expected TLSHandshakeTimeout 10s, got %v", config.TLSHandshakeTimeout)
	}
}

// TestBaseLLM_IsLocal tests the IsLocal method.
func TestBaseLLM_IsLocal(t *testing.T) {
	tests := []struct {
		provider string
		expected bool
	}{
		{"gemini", false},
		{"google", false},
		{"openai", false},
		{"anthropic", false},
		{"ollama", true},
		{"llamacpp", true},
		{"litellm", false},
		{"unknown", false},
	}

	for _, tt := range tests {
		t.Run(tt.provider, func(t *testing.T) {
			llm := NewBaseLLM(tt.provider, "test-model", nil, nil)
			if llm.IsLocal() != tt.expected {
				t.Errorf("IsLocal() for provider %q: expected %v, got %v",
					tt.provider, tt.expected, llm.IsLocal())
			}
		})
	}
}

// TestBaseLLM_WorkloadType tests the WorkloadType method.
func TestBaseLLM_WorkloadType(t *testing.T) {
	// Remote API should be I/O-bound
	remoteLLM := NewBaseLLM("gemini", "test-model", nil, nil)
	if remoteLLM.WorkloadType() != WorkloadIOBound {
		t.Errorf("Expected WorkloadIOBound for gemini, got %v", remoteLLM.WorkloadType())
	}

	// Local model should be CPU-bound
	localLLM := NewBaseLLM("ollama", "test-model", nil, nil)
	if localLLM.WorkloadType() != WorkloadCPUBound {
		t.Errorf("Expected WorkloadCPUBound for ollama, got %v", localLLM.WorkloadType())
	}
}

// TestWithTransportConfig tests the WithTransportConfig option.
func TestWithTransportConfig(t *testing.T) {
	customConfig := TransportConfig{
		MaxConnsPerHost:     200,
		MaxIdleConnsPerHost: 150,
		MaxIdleConns:        200,
		IdleConnTimeout:     120 * time.Second,
		TLSHandshakeTimeout: 15 * time.Second,
	}

	llm := NewBaseLLM("gemini", "test-model", nil, nil, WithTransportConfig(customConfig))

	// Verify the LLM was created (transport is internal, but we can check the client exists)
	if llm.GetHTTPClient() == nil {
		t.Error("Expected HTTP client to be set")
	}
}
