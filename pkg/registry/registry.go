package registry

import (
	"context"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// DefaultLLMRegistry is the in-memory default implementation of core.LLMRegistry.
type DefaultLLMRegistry struct {
	mu              sync.RWMutex
	providers       map[string]core.ProviderFactory
	providerConfigs map[string]core.ProviderConfig
	modelToProvider map[core.ModelID]string
	defaultProvider string
}

// NewLLMRegistry creates a new in-memory registry implementation.
func NewLLMRegistry() *DefaultLLMRegistry {
	return &DefaultLLMRegistry{
		providers:       make(map[string]core.ProviderFactory),
		providerConfigs: make(map[string]core.ProviderConfig),
		modelToProvider: make(map[core.ModelID]string),
	}
}

// RegisterProvider registers a new provider factory.
func (r *DefaultLLMRegistry) RegisterProvider(name string, factory core.ProviderFactory) error {
	if name == "" {
		return errors.New(errors.InvalidInput, "provider name cannot be empty")
	}
	if factory == nil {
		return errors.New(errors.InvalidInput, "provider factory cannot be nil")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.providers[name] = factory
	return nil
}

// UnregisterProvider removes a provider.
func (r *DefaultLLMRegistry) UnregisterProvider(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.providers, name)
	delete(r.providerConfigs, name)
	for modelID, providerName := range r.modelToProvider {
		if providerName == name {
			delete(r.modelToProvider, modelID)
		}
	}
	return nil
}

// CreateLLM creates an LLM instance using the registry with backward compatibility.
func (r *DefaultLLMRegistry) CreateLLM(ctx context.Context, apiKey string, modelID core.ModelID) (core.LLM, error) {
	r.mu.RLock()
	providerName, exists := r.modelToProvider[modelID]
	if !exists {
		providerName = core.InferProviderFromModelID(modelID)
		if providerName == "" {
			r.mu.RUnlock()
			return nil, errors.WithFields(
				errors.New(errors.ModelNotSupported, "model not supported"),
				errors.Fields{"model_id": modelID},
			)
		}
	}

	factory, exists := r.providers[providerName]
	if !exists {
		r.mu.RUnlock()
		return nil, errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": providerName},
		)
	}

	config := core.ProviderConfig{
		Name:   providerName,
		APIKey: apiKey,
	}
	if storedConfig, exists := r.providerConfigs[providerName]; exists {
		config = storedConfig.Clone()
		if apiKey != "" {
			config.APIKey = apiKey
		}
	}
	r.mu.RUnlock()

	return factory(ctx, config, modelID)
}

// CreateLLMWithConfig creates an LLM instance with specific configuration.
func (r *DefaultLLMRegistry) CreateLLMWithConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	r.mu.RLock()
	factory, exists := r.providers[config.Name]
	if !exists {
		r.mu.RUnlock()
		return nil, errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": config.Name},
		)
	}
	r.mu.RUnlock()

	return factory(ctx, config.Clone(), modelID)
}

// LoadFromConfig loads provider configurations from a config structure.
func (r *DefaultLLMRegistry) LoadFromConfig(ctx context.Context, configs map[string]core.ProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, config := range configs {
		if config.Name == "" {
			config.Name = name
		}

		storedConfig := config.Clone()
		r.providerConfigs[name] = storedConfig

		for modelIDStr := range storedConfig.Models {
			modelID := core.ModelID(modelIDStr)
			r.modelToProvider[modelID] = name
		}
	}
	return nil
}

// ListProviders returns all registered provider names.
func (r *DefaultLLMRegistry) ListProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	providers := make([]string, 0, len(r.providers))
	for name := range r.providers {
		providers = append(providers, name)
	}
	return providers
}

// GetProviderConfig returns the configuration for a provider.
func (r *DefaultLLMRegistry) GetProviderConfig(name string) (core.ProviderConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	config, exists := r.providerConfigs[name]
	if !exists {
		return core.ProviderConfig{}, false
	}
	return config.Clone(), true
}

// IsModelSupported checks if a model is supported by any provider.
func (r *DefaultLLMRegistry) IsModelSupported(modelID core.ModelID) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if _, exists := r.modelToProvider[modelID]; exists {
		return true
	}
	return core.InferProviderFromModelID(modelID) != ""
}

// GetModelProvider returns the provider name for a model.
func (r *DefaultLLMRegistry) GetModelProvider(modelID core.ModelID) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if provider, exists := r.modelToProvider[modelID]; exists {
		return provider, true
	}

	provider := core.InferProviderFromModelID(modelID)
	return provider, provider != ""
}

// SetDefaultProvider sets the default provider for models.
func (r *DefaultLLMRegistry) SetDefaultProvider(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[name]; !exists {
		return errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": name},
		)
	}

	r.defaultProvider = name
	return nil
}

// RefreshProvider reloads a provider configuration.
func (r *DefaultLLMRegistry) RefreshProvider(ctx context.Context, name string, config core.ProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[name]; !exists {
		return errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": name},
		)
	}

	storedConfig := config.Clone()
	r.providerConfigs[name] = storedConfig

	for modelID, providerName := range r.modelToProvider {
		if providerName == name {
			delete(r.modelToProvider, modelID)
		}
	}

	for modelIDStr := range storedConfig.Models {
		modelID := core.ModelID(modelIDStr)
		r.modelToProvider[modelID] = name
	}

	return nil
}

var _ core.LLMRegistry = (*DefaultLLMRegistry)(nil)
