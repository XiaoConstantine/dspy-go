package llms

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/registry"
	llm "github.com/XiaoConstantine/llm-go"
	llmmodels "github.com/XiaoConstantine/llm-go/models"
)

// DefaultLLMFactory creates the compatibility adapter backed by llm-go.
type DefaultLLMFactory struct{}

var (
	defaultFactory     *DefaultLLMFactory
	defaultFactoryOnce sync.Once
	registryInitOnce   sync.Once
	factoryInitErr     error
	registryInitErr    error
)

var defaultProviderFactories = []struct {
	name    string
	factory core.ProviderFactory
}{
	{"anthropic", AnthropicProviderFactory},
	{"google", GeminiProviderFactory},
	{"openai", OpenAIProviderFactory},
	{"openai-codex", OpenAICodexProviderFactory},
	{"ollama", OllamaProviderFactory},
	{"llamacpp", LlamacppProviderFactory},
	{"litellm", LiteLLMProviderFactory},
	{"localai", LocalAIProviderFactory},
	{"fastchat", FastChatProviderFactory},
}

func init() {
	core.SetRegistryConstructor(func() core.LLMRegistry {
		return registry.NewLLMRegistry()
	})
	_ = ensureFactory()
}

func ensureFactory() error {
	defaultFactoryOnce.Do(func() {
		defaultFactory = &DefaultLLMFactory{}
		core.SetDefaultFactory(defaultFactory)
		factoryInitErr = ensureRegistryInitialized()
	})
	return factoryInitErr
}

func ensureRegistryInitialized() error {
	registryInitOnce.Do(func() {
		llmRegistry := core.GetRegistry()
		for _, provider := range defaultProviderFactories {
			if err := llmRegistry.RegisterProvider(provider.name, provider.factory); err != nil {
				registryInitErr = fmt.Errorf("register %s provider: %w", provider.name, err)
				return
			}
		}

		configs := defaultProviderConfigurations()
		// Load providers individually and in reverse registration order so
		// duplicate compatible model IDs resolve deterministically to native
		// catalog providers such as OpenAI.
		for index := len(defaultProviderFactories) - 1; index >= 0; index-- {
			name := defaultProviderFactories[index].name
			if err := llmRegistry.LoadFromConfig(context.Background(), map[string]core.ProviderConfig{
				name: configs[name],
			}); err != nil {
				registryInitErr = fmt.Errorf("load %s provider models: %w", name, err)
				return
			}
		}
	})
	return registryInitErr
}

func defaultProviderConfigurations() map[string]core.ProviderConfig {
	configs := make(map[string]core.ProviderConfig, len(defaultProviderFactories))
	for _, provider := range defaultProviderFactories {
		configs[provider.name] = core.ProviderConfig{
			Name:   provider.name,
			Models: make(map[string]core.ModelConfig),
		}
	}

	for _, model := range llmmodels.BuiltinCatalog().Models("") {
		config, registered := configs[model.Provider]
		if !registered {
			continue
		}
		config.Models[model.ID] = registryModelConfig(model.Provider, model.ID, model.Name, model.Capabilities)
		configs[model.Provider] = config
	}

	compatibleCapabilities := []llm.Capability{
		llm.CapabilityGeneration,
		llm.CapabilityStreaming,
		llm.CapabilityTools,
		llm.CapabilityJSON,
		llm.CapabilityVision,
	}
	for _, provider := range []string{"ollama", "llamacpp", "litellm", "localai", "fastchat"} {
		config := configs[provider]
		for _, modelID := range core.ProviderModels[provider] {
			id := string(modelID)
			if _, exists := config.Models[id]; exists {
				continue
			}
			config.Models[id] = registryModelConfig(provider, id, id, compatibleCapabilities)
		}
		configs[provider] = config
	}

	return configs
}

func registryModelConfig(provider, id, name string, capabilities []llm.Capability) core.ModelConfig {
	if name == "" {
		name = id
	}
	converted := coreCapabilities(capabilities)
	switch provider {
	case "google", "openai", "ollama", "llamacpp", "litellm", "localai", "fastchat":
		converted = appendCapability(converted, core.CapabilityEmbedding)
	}
	capabilityNames := make([]string, len(converted))
	for index, capability := range converted {
		capabilityNames[index] = string(capability)
	}
	return core.ModelConfig{ID: id, Name: name, Capabilities: capabilityNames}
}

// NewLLM creates a cached dspy-go LLM backed by an llm-go generator.
func NewLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	if err := ensureFactory(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	model, err := core.GetRegistry().CreateLLM(ctx, apiKey, modelID)
	if err != nil {
		return nil, fmt.Errorf("create LLM: %w", err)
	}
	return cache.WrapWithCache(model, nil), nil
}

// CreateLLM implements core.LLMFactory.
func (*DefaultLLMFactory) CreateLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	return NewLLM(apiKey, modelID)
}

// EnsureFactory initializes dspy-go's default llm-go-backed provider registry.
func EnsureFactory() {
	_ = ensureFactory()
}

// resetFactoryForTesting resets package-global factory state.
func resetFactoryForTesting() {
	defaultFactory = nil
	defaultFactoryOnce = sync.Once{}
	registryInitOnce = sync.Once{}
	factoryInitErr = nil
	registryInitErr = nil
	core.SetDefaultFactory(nil)
	core.SetRegistry(core.NewLLMRegistry())
}
