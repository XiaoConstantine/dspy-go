package llms

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type DefaultLLMFactory struct{}

var (
	defaultFactory     *DefaultLLMFactory
	defaultFactoryOnce sync.Once
	registryInitOnce   sync.Once
)

// resetFactoryForTesting resets the factory for testing purposes
// This should only be called from tests.
func resetFactoryForTesting() {
	defaultFactory = nil
	defaultFactoryOnce = sync.Once{}
	registryInitOnce = sync.Once{}
	core.DefaultFactory = nil
	// Reset the global registry as well
	core.GlobalRegistry = core.NewLLMRegistry()
}

func ensureFactory() {
	defaultFactoryOnce.Do(func() {
		defaultFactory = &DefaultLLMFactory{}
		core.DefaultFactory = defaultFactory
		// Initialize registry on first factory creation
		ensureRegistryInitialized()
	})
}

// ensureRegistryInitialized initializes the global registry with default providers.
func ensureRegistryInitialized() {
	registryInitOnce.Do(func() {
		// Initialize the global registry if not already done
		if core.GlobalRegistry == nil {
			core.GlobalRegistry = core.NewLLMRegistry()
		}

		// Register default providers
		registry := core.GetRegistry()

		// Register Anthropic provider
		if err := registry.RegisterProvider("anthropic", AnthropicProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register anthropic provider: %v", err))
		}

		// Register Google/Gemini provider
		if err := registry.RegisterProvider("google", GeminiProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register google provider: %v", err))
		}

		// Register Ollama provider
		if err := registry.RegisterProvider("ollama", OllamaProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register ollama provider: %v", err))
		}

		// Register LlamaCpp provider if it exists
		if err := registry.RegisterProvider("llamacpp", LlamacppProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register llamacpp provider: %v", err))
		}

		// Load default model configurations
		loadDefaultModelConfigurations(registry)
	})
}

// loadDefaultModelConfigurations loads the default model configurations into the registry.
func loadDefaultModelConfigurations(registry core.LLMRegistry) {
	defaultConfigs := map[string]core.ProviderConfig{
		"anthropic": {
			Name: "anthropic",
			Models: map[string]core.ModelConfig{
				string(core.ModelAnthropicHaiku): {
					ID:           string(core.ModelAnthropicHaiku),
					Name:         "Claude 3 Haiku",
					Capabilities: []string{"completion", "chat", "json"},
				},
				string(core.ModelAnthropicSonnet): {
					ID:           string(core.ModelAnthropicSonnet),
					Name:         "Claude 3.5 Sonnet",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelAnthropicOpus): {
					ID:           string(core.ModelAnthropicOpus),
					Name:         "Claude 3 Opus",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
			},
		},
		"google": {
			Name: "google",
			Models: map[string]core.ModelConfig{
				string(core.ModelGoogleGeminiFlash): {
					ID:           string(core.ModelGoogleGeminiFlash),
					Name:         "Gemini 2.0 Flash",
					Capabilities: []string{"completion", "chat", "json", "embedding", "streaming", "tool-calling"},
				},
				string(core.ModelGoogleGeminiPro): {
					ID:           string(core.ModelGoogleGeminiPro),
					Name:         "Gemini 2.5 Pro",
					Capabilities: []string{"completion", "chat", "json", "embedding", "streaming", "tool-calling"},
				},
				string(core.ModelGoogleGeminiFlashThinking): {
					ID:           string(core.ModelGoogleGeminiFlashThinking),
					Name:         "Gemini 2.0 Flash Thinking",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelGoogleGeminiFlashLite): {
					ID:           string(core.ModelGoogleGeminiFlashLite),
					Name:         "Gemini 2.0 Flash Lite",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := registry.LoadFromConfig(ctx, defaultConfigs); err != nil {
		panic(fmt.Sprintf("failed to load default model configurations: %v", err))
	}
}

// NewLLM creates a new LLM instance based on the provided model ID.
// This function now uses the registry system for dynamic model creation.
func NewLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	ensureFactory()

	// Try to use the registry first
	registry := core.GetRegistry()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	llm, err := registry.CreateLLM(ctx, apiKey, modelID)
	if err != nil {
		// If registry fails, fall back to the old hardcoded approach for backward compatibility
		llm, err = createLLMFallback(apiKey, modelID)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM with registry and fallback: %w", err)
		}
	}

	// Apply caching if enabled (this happens before other decorators)
	llm = cache.WrapWithCache(llm, nil) // nil means use environment/default config

	// Apply other decorators
	return core.Chain(llm,
		func(l core.LLM) core.LLM { return core.NewModelContextDecorator(l) },
	), nil
}

// createLLMFallback provides backward compatibility with the old factory approach.
func createLLMFallback(apiKey string, modelID core.ModelID) (core.LLM, error) {
	var llm core.LLM
	var err error

	switch {
	case modelID == core.ModelAnthropicHaiku || modelID == core.ModelAnthropicSonnet || modelID == core.ModelAnthropicOpus:
		llm, err = NewAnthropicLLM(apiKey, anthropic.ModelID(modelID))
	case modelID == core.ModelGoogleGeminiFlash || modelID == core.ModelGoogleGeminiPro || modelID == core.ModelGoogleGeminiFlashThinking || modelID == core.ModelGoogleGeminiFlashLite:
		llm, err = NewGeminiLLM(apiKey, modelID)
	case strings.HasPrefix(string(modelID), "ollama:"):
		parts := strings.SplitN(string(modelID), ":", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("invalid Ollama model ID format. Use 'ollama:<model_name>'")
		}
		llm, err = NewOllamaLLM("http://localhost:11434", parts[1])
	case strings.HasPrefix(string(modelID), "llamacpp:"):
		return NewLlamacppLLM("http://localhost:8080")
	default:
		return nil, fmt.Errorf("unsupported model ID: %s", modelID)
	}

	return llm, err
}

// Implement the LLMFactory interface.
func (f *DefaultLLMFactory) CreateLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	return NewLLM(apiKey, modelID)
}

func init() {
	ensureFactory()
}

func EnsureFactory() {
	ensureFactory()
}

// Provider factories are defined in their respective LLM files
// (anthrophic.go, gemini.go, ollama.go, llamacpp.go)
