package core

import (
	"context"
	"testing"
	"time"
)

// RegistryMockLLM is a simple mock implementation for registry testing.
type RegistryMockLLM struct {
	*BaseLLM
	providerName string
	modelID      string
}

func NewRegistryMockLLM(providerName, modelID string) *RegistryMockLLM {
	capabilities := []Capability{CapabilityCompletion, CapabilityChat}
	return &RegistryMockLLM{
		BaseLLM:      NewBaseLLM(providerName, ModelID(modelID), capabilities, nil),
		providerName: providerName,
		modelID:      modelID,
	}
}

func (m *RegistryMockLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	return &LLMResponse{Content: "mock response for " + prompt}, nil
}

func (m *RegistryMockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"mock": "json response"}, nil
}

func (m *RegistryMockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...GenerateOption) (map[string]interface{}, error) {
	return map[string]interface{}{"function": "called"}, nil
}

func (m *RegistryMockLLM) CreateEmbedding(ctx context.Context, input string, options ...EmbeddingOption) (*EmbeddingResult, error) {
	return &EmbeddingResult{Vector: []float32{0.1, 0.2, 0.3}}, nil
}

func (m *RegistryMockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...EmbeddingOption) (*BatchEmbeddingResult, error) {
	results := make([]EmbeddingResult, len(inputs))
	for i := range inputs {
		results[i] = EmbeddingResult{Vector: []float32{0.1, 0.2, 0.3}}
	}
	return &BatchEmbeddingResult{Embeddings: results}, nil
}

func (m *RegistryMockLLM) StreamGenerate(ctx context.Context, prompt string, options ...GenerateOption) (*StreamResponse, error) {
	chunkChan := make(chan StreamChunk, 1)
	chunkChan <- StreamChunk{Content: "mock stream", Done: true}
	close(chunkChan)
	return &StreamResponse{ChunkChannel: chunkChan, Cancel: func() {}}, nil
}

func (m *RegistryMockLLM) GenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*LLMResponse, error) {
	// Convert content blocks to a simple text representation for mock response
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}
	return &LLMResponse{Content: "mock response for " + textContent}, nil
}

func (m *RegistryMockLLM) StreamGenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*StreamResponse, error) {
	// Create a simple mock stream response
	chunkChan := make(chan StreamChunk, 1)
	var textContent string
	for _, block := range content {
		textContent += block.String() + " "
	}

	chunkChan <- StreamChunk{
		Content: "mock stream response for " + textContent,
		Done:    true,
	}
	close(chunkChan)

	return &StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       func() {},
	}, nil
}

// MockProviderFactory creates RegistryMockLLM instances.
func MockProviderFactory(ctx context.Context, config ProviderConfig, modelID ModelID) (LLM, error) {
	return NewRegistryMockLLM(config.Name, string(modelID)), nil
}

func TestLLMRegistry_RegisterProvider(t *testing.T) {
	registry := NewLLMRegistry()

	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	providers := registry.ListProviders()
	if len(providers) != 1 || providers[0] != "mock" {
		t.Errorf("Expected 1 provider 'mock', got %v", providers)
	}
}

func TestLLMRegistry_CreateLLM(t *testing.T) {
	registry := NewLLMRegistry()
	ctx := context.Background()

	// Register mock provider
	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Load configuration
	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		},
	}

	err = registry.LoadFromConfig(ctx, config)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Create LLM
	llm, err := registry.CreateLLM(ctx, "test-api-key", ModelID("test-model"))
	if err != nil {
		t.Fatalf("Failed to create LLM: %v", err)
	}

	if llm.ProviderName() != "mock" {
		t.Errorf("Expected provider 'mock', got '%s'", llm.ProviderName())
	}

	if llm.ModelID() != "test-model" {
		t.Errorf("Expected model 'test-model', got '%s'", llm.ModelID())
	}
}

func TestLLMRegistry_IsModelSupported(t *testing.T) {
	registry := NewLLMRegistry()
	ctx := context.Background()

	// Register mock provider
	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Load configuration
	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID: "test-model",
				},
			},
		},
	}

	err = registry.LoadFromConfig(ctx, config)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Test supported model
	if !registry.IsModelSupported(ModelID("test-model")) {
		t.Error("Expected test-model to be supported")
	}

	// Test unsupported model
	if registry.IsModelSupported(ModelID("unsupported-model")) {
		t.Error("Expected unsupported-model to not be supported")
	}
}

func TestLLMRegistry_RefreshProvider(t *testing.T) {
	registry := NewLLMRegistry()
	ctx := context.Background()

	// Register mock provider
	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Initial configuration
	config := ProviderConfig{
		Name: "mock",
		Models: map[string]ModelConfig{
			"model1": {ID: "model1"},
		},
	}

	err = registry.LoadFromConfig(ctx, map[string]ProviderConfig{"mock": config})
	if err != nil {
		t.Fatalf("Failed to load initial config: %v", err)
	}

	// Check initial model is supported
	if !registry.IsModelSupported(ModelID("model1")) {
		t.Error("Expected model1 to be supported initially")
	}

	// Refresh with new configuration
	newConfig := ProviderConfig{
		Name: "mock",
		Models: map[string]ModelConfig{
			"model2": {ID: "model2"},
		},
	}

	err = registry.RefreshProvider(ctx, "mock", newConfig)
	if err != nil {
		t.Fatalf("Failed to refresh provider: %v", err)
	}

	// Check old model is no longer supported
	if registry.IsModelSupported(ModelID("model1")) {
		t.Error("Expected model1 to no longer be supported after refresh")
	}

	// Check new model is supported
	if !registry.IsModelSupported(ModelID("model2")) {
		t.Error("Expected model2 to be supported after refresh")
	}
}

func TestLLMRegistry_BackwardCompatibility(t *testing.T) {
	registry := NewLLMRegistry()

	// Test backward compatibility with Anthropic models
	// The registry should be able to infer the provider
	if !registry.IsModelSupported(ModelAnthropicSonnet) {
		t.Error("Expected Anthropic Sonnet to be supported via backward compatibility")
	}

	providerName, found := registry.GetModelProvider(ModelAnthropicSonnet)
	if !found || providerName != "anthropic" {
		t.Errorf("Expected to find 'anthropic' provider for Sonnet, got '%s', found: %v", providerName, found)
	}

	// Test with Gemini models
	if !registry.IsModelSupported(ModelGoogleGeminiFlash) {
		t.Error("Expected Gemini Flash to be supported via backward compatibility")
	}

	providerName, found = registry.GetModelProvider(ModelGoogleGeminiFlash)
	if !found || providerName != "google" {
		t.Errorf("Expected to find 'google' provider for Gemini Flash, got '%s', found: %v", providerName, found)
	}
}

func TestLLMRegistry_ThreadSafety(t *testing.T) {
	registry := NewLLMRegistry()
	ctx := context.Background()

	// Register provider
	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {ID: "test-model"},
			},
		},
	}

	err = registry.LoadFromConfig(ctx, config)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Test concurrent access
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- true }()

			// Test creating LLMs concurrently
			_, err := registry.CreateLLM(ctx, "test-key", ModelID("test-model"))
			if err != nil {
				t.Errorf("Failed to create LLM concurrently: %v", err)
			}

			// Test checking model support concurrently
			_ = registry.IsModelSupported(ModelID("test-model"))

			// Test listing providers concurrently
			_ = registry.ListProviders()
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Timed out waiting for concurrent operations")
		}
	}
}

func TestRegistryConfig_Integration(t *testing.T) {
	ctx := context.Background()

	// Test full registry configuration
	registryConfig := RegistryConfig{
		Providers: map[string]ProviderConfig{
			"mock": {
				Name: "mock",
				Models: map[string]ModelConfig{
					"test-model": {
						ID:           "test-model",
						Name:         "Test Model",
						Capabilities: []string{"completion", "chat"},
					},
				},
			},
		},
		DefaultProvider: "mock",
	}

	// Register the mock provider factory first (this would normally be done by the provider package)
	registry := GetRegistry()
	err := registry.RegisterProvider("mock", MockProviderFactory)
	if err != nil {
		t.Fatalf("Failed to register mock provider: %v", err)
	}

	// Initialize registry with configuration
	err = InitializeRegistry(ctx, registryConfig)
	if err != nil {
		t.Fatalf("Failed to initialize registry: %v", err)
	}

	// Test that the configuration was loaded
	if !registry.IsModelSupported(ModelID("test-model")) {
		t.Error("Expected test-model to be supported after registry initialization")
	}

	// Test creating LLM through global config functions
	err = ConfigureDefaultLLMFromRegistry(ctx, "test-key", ModelID("test-model"))
	if err != nil {
		t.Fatalf("Failed to configure default LLM from registry: %v", err)
	}

	if GlobalConfig.DefaultLLM == nil {
		t.Error("Expected default LLM to be configured")
	}

	if GlobalConfig.DefaultLLM.ModelID() != "test-model" {
		t.Errorf("Expected default LLM model 'test-model', got '%s'", GlobalConfig.DefaultLLM.ModelID())
	}
}
