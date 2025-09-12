package core

import (
	"context"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// ProviderConfig represents configuration for a specific provider.
type ProviderConfig struct {
	// Provider name (e.g., "anthropic", "google", "ollama")
	Name string `json:"name" yaml:"name"`

	// API configuration
	APIKey  string `json:"api_key,omitempty" yaml:"api_key,omitempty"`
	BaseURL string `json:"base_url,omitempty" yaml:"base_url,omitempty"`

	// Model-specific configuration
	Models map[string]ModelConfig `json:"models,omitempty" yaml:"models,omitempty"`

	// Provider-specific parameters
	Params map[string]interface{} `json:"params,omitempty" yaml:"params,omitempty"`

	// Endpoint configuration
	Endpoint *EndpointConfig `json:"endpoint,omitempty" yaml:"endpoint,omitempty"`
}

// ModelConfig represents configuration for a specific model.
type ModelConfig struct {
	// Model identifier
	ID string `json:"id" yaml:"id"`

	// Display name
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// Model capabilities
	Capabilities []string `json:"capabilities,omitempty" yaml:"capabilities,omitempty"`

	// Model-specific parameters
	Params map[string]interface{} `json:"params,omitempty" yaml:"params,omitempty"`

	// Default generation options
	DefaultOptions *GenerateOptions `json:"default_options,omitempty" yaml:"default_options,omitempty"`
}

// ProviderFactory creates LLM instances for a specific provider.
type ProviderFactory func(ctx context.Context, config ProviderConfig, modelID ModelID) (LLM, error)

// LLMRegistry manages dynamic registration and creation of LLM providers.
type LLMRegistry interface {
	// RegisterProvider registers a new provider factory
	RegisterProvider(name string, factory ProviderFactory) error

	// UnregisterProvider removes a provider
	UnregisterProvider(name string) error

	// CreateLLM creates an LLM instance using the registry
	CreateLLM(ctx context.Context, apiKey string, modelID ModelID) (LLM, error)

	// CreateLLMWithConfig creates an LLM instance with specific configuration
	CreateLLMWithConfig(ctx context.Context, config ProviderConfig, modelID ModelID) (LLM, error)

	// LoadFromConfig loads provider configurations from a config structure
	LoadFromConfig(ctx context.Context, configs map[string]ProviderConfig) error

	// ListProviders returns all registered provider names
	ListProviders() []string

	// GetProviderConfig returns the configuration for a provider
	GetProviderConfig(name string) (ProviderConfig, bool)

	// IsModelSupported checks if a model is supported by any provider
	IsModelSupported(modelID ModelID) bool

	// GetModelProvider returns the provider name for a model
	GetModelProvider(modelID ModelID) (string, bool)

	// SetDefaultProvider sets the default provider for models
	SetDefaultProvider(name string) error

	// RefreshProvider reloads a provider configuration
	RefreshProvider(ctx context.Context, name string, config ProviderConfig) error
}

// DefaultLLMRegistry is the default implementation of LLMRegistry.
type DefaultLLMRegistry struct {
	mu              sync.RWMutex
	providers       map[string]ProviderFactory
	providerConfigs map[string]ProviderConfig
	modelToProvider map[ModelID]string
	defaultProvider string
}

// NewLLMRegistry creates a new instance of DefaultLLMRegistry.
func NewLLMRegistry() *DefaultLLMRegistry {
	return &DefaultLLMRegistry{
		providers:       make(map[string]ProviderFactory),
		providerConfigs: make(map[string]ProviderConfig),
		modelToProvider: make(map[ModelID]string),
	}
}

// RegisterProvider registers a new provider factory.
func (r *DefaultLLMRegistry) RegisterProvider(name string, factory ProviderFactory) error {
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

	// Remove model mappings for this provider
	for modelID, providerName := range r.modelToProvider {
		if providerName == name {
			delete(r.modelToProvider, modelID)
		}
	}

	return nil
}

// CreateLLM creates an LLM instance using the registry with backward compatibility.
func (r *DefaultLLMRegistry) CreateLLM(ctx context.Context, apiKey string, modelID ModelID) (LLM, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Find the provider for this model
	providerName, exists := r.modelToProvider[modelID]
	if !exists {
		// Try to infer provider from model ID for backward compatibility
		providerName = r.inferProviderFromModelID(modelID)
		if providerName == "" {
			return nil, errors.WithFields(
				errors.New(errors.ModelNotSupported, "model not supported"),
				errors.Fields{"model_id": modelID})
		}
	}

	// Get the provider factory
	factory, exists := r.providers[providerName]
	if !exists {
		return nil, errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": providerName})
	}

	// Create a basic config for backward compatibility
	config := ProviderConfig{
		Name:   providerName,
		APIKey: apiKey,
	}

	// If we have a stored config, use it
	if storedConfig, exists := r.providerConfigs[providerName]; exists {
		config = storedConfig
		// Override API key if provided
		if apiKey != "" {
			config.APIKey = apiKey
		}
	}

	return factory(ctx, config, modelID)
}

// CreateLLMWithConfig creates an LLM instance with specific configuration.
func (r *DefaultLLMRegistry) CreateLLMWithConfig(ctx context.Context, config ProviderConfig, modelID ModelID) (LLM, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	factory, exists := r.providers[config.Name]
	if !exists {
		return nil, errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": config.Name})
	}

	return factory(ctx, config, modelID)
}

// LoadFromConfig loads provider configurations from a config structure.
func (r *DefaultLLMRegistry) LoadFromConfig(ctx context.Context, configs map[string]ProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, config := range configs {
		// Validate config
		if config.Name == "" {
			config.Name = name
		}

		// Store provider configuration
		r.providerConfigs[name] = config

		// Map models to providers
		for modelIDStr := range config.Models {
			modelID := ModelID(modelIDStr)
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
func (r *DefaultLLMRegistry) GetProviderConfig(name string) (ProviderConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	config, exists := r.providerConfigs[name]
	return config, exists
}

// IsModelSupported checks if a model is supported by any provider.
func (r *DefaultLLMRegistry) IsModelSupported(modelID ModelID) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, exists := r.modelToProvider[modelID]
	if exists {
		return true
	}

	// Check if we can infer the provider for backward compatibility
	return r.inferProviderFromModelID(modelID) != ""
}

// GetModelProvider returns the provider name for a model.
func (r *DefaultLLMRegistry) GetModelProvider(modelID ModelID) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	provider, exists := r.modelToProvider[modelID]
	if exists {
		return provider, true
	}

	// Try to infer for backward compatibility
	provider = r.inferProviderFromModelID(modelID)
	return provider, provider != ""
}

// SetDefaultProvider sets the default provider for models.
func (r *DefaultLLMRegistry) SetDefaultProvider(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[name]; !exists {
		return errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": name})
	}

	r.defaultProvider = name
	return nil
}

// RefreshProvider reloads a provider configuration.
func (r *DefaultLLMRegistry) RefreshProvider(ctx context.Context, name string, config ProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if provider is registered
	if _, exists := r.providers[name]; !exists {
		return errors.WithFields(
			errors.New(errors.ProviderNotFound, "provider not found"),
			errors.Fields{"provider": name})
	}

	// Update configuration
	r.providerConfigs[name] = config

	// Update model mappings
	for modelID, providerName := range r.modelToProvider {
		if providerName == name {
			delete(r.modelToProvider, modelID)
		}
	}

	// Add new model mappings
	for modelIDStr := range config.Models {
		modelID := ModelID(modelIDStr)
		r.modelToProvider[modelID] = name
	}

	return nil
}

// inferProviderFromModelID infers the provider name from model ID for backward compatibility.
func (r *DefaultLLMRegistry) inferProviderFromModelID(modelID ModelID) string {
	modelStr := string(modelID)

	// Anthropic models
	if modelID == ModelAnthropicHaiku || modelID == ModelAnthropicSonnet || modelID == ModelAnthropicOpus {
		return "anthropic"
	}

	// Google models
	if modelID == ModelGoogleGeminiFlash || modelID == ModelGoogleGeminiPro ||
		modelID == ModelGoogleGeminiFlashLite {
		return "google"
	}

	// Ollama models
	if len(modelStr) > 7 && modelStr[:7] == "ollama:" {
		return "ollama"
	}

	// LlamaCpp models
	if len(modelStr) > 9 && modelStr[:9] == "llamacpp:" {
		return "llamacpp"
	}

	return ""
}

// GlobalRegistry is the global registry instance.
var (
	GlobalRegistry LLMRegistry
	registryOnce   sync.Once
)

// RegistryConfig represents the configuration for the registry system.
type RegistryConfig struct {
	// Providers configuration
	Providers map[string]ProviderConfig `json:"providers,omitempty" yaml:"providers,omitempty"`

	// Default provider
	DefaultProvider string `json:"default_provider,omitempty" yaml:"default_provider,omitempty"`

	// Cache settings
	CacheEnabled bool          `json:"cache_enabled,omitempty" yaml:"cache_enabled,omitempty"`
	CacheTTL     time.Duration `json:"cache_ttl,omitempty" yaml:"cache_ttl,omitempty"`
}

// InitializeRegistry initializes the global registry with the given configuration.
func InitializeRegistry(ctx context.Context, config RegistryConfig) error {
	// Use existing registry or create new one
	var registry LLMRegistry
	if GlobalRegistry != nil {
		registry = GlobalRegistry
	} else {
		registry = NewLLMRegistry()
	}

	// Load provider configurations
	if len(config.Providers) > 0 {
		if err := registry.LoadFromConfig(ctx, config.Providers); err != nil {
			return errors.Wrap(err, errors.ConfigurationError, "failed to load provider configurations")
		}
	}

	// Set default provider
	if config.DefaultProvider != "" {
		if err := registry.SetDefaultProvider(config.DefaultProvider); err != nil {
			return errors.Wrap(err, errors.ConfigurationError, "failed to set default provider")
		}
	}

	GlobalRegistry = registry
	return nil
}

// GetRegistry returns the global registry instance.
func GetRegistry() LLMRegistry {
	registryOnce.Do(func() {
		if GlobalRegistry == nil {
			// Initialize with empty config for backward compatibility
			GlobalRegistry = NewLLMRegistry()
		}
	})
	return GlobalRegistry
}

// RegisterProviderFactory is a convenience function to register a provider factory.
func RegisterProviderFactory(name string, factory ProviderFactory) error {
	return GetRegistry().RegisterProvider(name, factory)
}

// CreateLLMFromRegistry creates an LLM instance using the global registry.
func CreateLLMFromRegistry(ctx context.Context, apiKey string, modelID ModelID) (LLM, error) {
	return GetRegistry().CreateLLM(ctx, apiKey, modelID)
}

// Backward compatibility functions.
func init() {
	// Initialize the global registry
	GlobalRegistry = NewLLMRegistry()
}
