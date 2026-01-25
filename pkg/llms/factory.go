package llms

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

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

		// Register OpenAI and compatible providers
		if err := registry.RegisterProvider("openai", OpenAIProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register openai provider: %v", err))
		}
		if err := registry.RegisterProvider("litellm", LiteLLMProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register litellm provider: %v", err))
		}
		if err := registry.RegisterProvider("localai", LocalAIProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register localai provider: %v", err))
		}
		if err := registry.RegisterProvider("fastchat", FastChatProviderFactory); err != nil {
			panic(fmt.Sprintf("failed to register fastchat provider: %v", err))
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
				string(core.ModelGoogleGeminiFlashLite): {
					ID:           string(core.ModelGoogleGeminiFlashLite),
					Name:         "Gemini 2.0 Flash Lite",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
			},
		},
		"openai": {
			Name: "openai",
			Models: map[string]core.ModelConfig{
				string(core.ModelOpenAIGPT4): {
					ID:           string(core.ModelOpenAIGPT4),
					Name:         "GPT-4",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelOpenAIGPT4Turbo): {
					ID:           string(core.ModelOpenAIGPT4Turbo),
					Name:         "GPT-4 Turbo",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelOpenAIGPT35Turbo): {
					ID:           string(core.ModelOpenAIGPT35Turbo),
					Name:         "GPT-3.5 Turbo",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelOpenAIGPT4o): {
					ID:           string(core.ModelOpenAIGPT4o),
					Name:         "GPT-4o",
					Capabilities: []string{"completion", "chat", "json", "streaming", "embedding"},
				},
				string(core.ModelOpenAIGPT4oMini): {
					ID:           string(core.ModelOpenAIGPT4oMini),
					Name:         "GPT-4o Mini",
					Capabilities: []string{"completion", "chat", "json", "streaming", "embedding"},
				},
			},
		},
		"litellm": {
			Name: "litellm",
			Models: map[string]core.ModelConfig{
				string(core.ModelLiteLLMGPT4): {
					ID:           string(core.ModelLiteLLMGPT4),
					Name:         "GPT-4 via LiteLLM",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelLiteLLMClaude3): {
					ID:           string(core.ModelLiteLLMClaude3),
					Name:         "Claude 3 Sonnet via LiteLLM",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
				string(core.ModelLiteLLMLlama2): {
					ID:           string(core.ModelLiteLLMLlama2),
					Name:         "Llama 2 70B via LiteLLM",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
				string(core.ModelLiteLLMGemini): {
					ID:           string(core.ModelLiteLLMGemini),
					Name:         "Gemini Pro via LiteLLM",
					Capabilities: []string{"completion", "chat", "json", "streaming"},
				},
			},
		},
		"localai": {
			Name: "localai",
			Models: map[string]core.ModelConfig{
				string(core.ModelLocalAILlama2): {
					ID:           string(core.ModelLocalAILlama2),
					Name:         "Llama 2 7B Chat",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
				string(core.ModelLocalAICodeLlama): {
					ID:           string(core.ModelLocalAICodeLlama),
					Name:         "CodeLlama 13B Instruct",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
				string(core.ModelLocalAIAlpaca): {
					ID:           string(core.ModelLocalAIAlpaca),
					Name:         "Alpaca 7B",
					Capabilities: []string{"completion", "chat"},
				},
				string(core.ModelLocalAIVicuna): {
					ID:           string(core.ModelLocalAIVicuna),
					Name:         "Vicuna 7B",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
			},
		},
		"fastchat": {
			Name: "fastchat",
			Models: map[string]core.ModelConfig{
				string(core.ModelFastChatVicuna): {
					ID:           string(core.ModelFastChatVicuna),
					Name:         "Vicuna 7B v1.5",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
				string(core.ModelFastChatAlpaca): {
					ID:           string(core.ModelFastChatAlpaca),
					Name:         "Alpaca 13B",
					Capabilities: []string{"completion", "chat"},
				},
				string(core.ModelFastChatCodeLlama): {
					ID:           string(core.ModelFastChatCodeLlama),
					Name:         "CodeLlama 7B Instruct",
					Capabilities: []string{"completion", "chat", "streaming"},
				},
				string(core.ModelFastChatLlama2): {
					ID:           string(core.ModelFastChatLlama2),
					Name:         "Llama 2 7B Chat",
					Capabilities: []string{"completion", "chat", "streaming"},
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

	modelStr := string(modelID)
	switch {
	case modelID == core.ModelAnthropicHaiku || modelID == core.ModelAnthropicSonnet || modelID == core.ModelAnthropicOpus ||
		modelID == core.ModelAnthropicClaude45Opus || modelID == core.ModelAnthropicClaude4Opus ||
		modelID == core.ModelAnthropicClaude4Sonnet || modelID == core.ModelAnthropicClaude45Sonnet ||
		strings.HasPrefix(modelStr, "claude-") || strings.HasPrefix(modelStr, "opus-") || strings.HasPrefix(modelStr, "sonnet-"):
		llm, err = NewAnthropicLLM(apiKey, normalizeModelName(modelStr))
	case modelID == core.ModelGoogleGeminiFlash || modelID == core.ModelGoogleGeminiPro || modelID == core.ModelGoogleGeminiFlashLite:
		llm, err = NewGeminiLLM(apiKey, modelID)
	case modelID == core.ModelOpenAIGPT4 || modelID == core.ModelOpenAIGPT4Turbo || modelID == core.ModelOpenAIGPT35Turbo ||
		modelID == core.ModelOpenAIGPT4o || modelID == core.ModelOpenAIGPT4oMini ||
		modelID == core.ModelOpenAIGPT41 || modelID == core.ModelOpenAIGPT41Mini || modelID == core.ModelOpenAIGPT41Nano ||
		modelID == core.ModelOpenAIO1 || modelID == core.ModelOpenAIO1Pro || modelID == core.ModelOpenAIO1Mini ||
		modelID == core.ModelOpenAIO3 || modelID == core.ModelOpenAIO3Mini ||
		modelID == core.ModelOpenAIGPT5 || modelID == core.ModelOpenAIGPT5Mini || modelID == core.ModelOpenAIGPT5Nano ||
		modelID == core.ModelOpenAIGPT52 || modelID == core.ModelOpenAIGPT52Instant || modelID == core.ModelOpenAIGPT52Thinking ||
		modelID == core.ModelOpenAIGPT52ThinkHigh || modelID == core.ModelOpenAIGPT52Pro || modelID == core.ModelOpenAIGPT52Codex ||
		strings.HasPrefix(modelStr, "gpt-") || strings.HasPrefix(modelStr, "o1") || strings.HasPrefix(modelStr, "o3"):
		llm, err = NewOpenAI(modelID, apiKey)
	case strings.HasPrefix(string(modelID), "ollama:"):
		parts := strings.SplitN(string(modelID), ":", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("invalid Ollama model ID format. Use 'ollama:<model_name>'")
		}
		llm, err = NewOllamaLLM(core.ModelID(parts[1]))
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

// isValidModelInList checks if a model ID is in the provided list of valid models.
// This provides O(n) lookup which is acceptable for small lists of valid models.
func isValidModelInList(modelID core.ModelID, validModels []core.ModelID) bool {
	for _, validModel := range validModels {
		if modelID == validModel {
			return true
		}
	}
	return false
}
