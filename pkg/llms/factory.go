package llms

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/registry"
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
	})
	return registryInitErr
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
