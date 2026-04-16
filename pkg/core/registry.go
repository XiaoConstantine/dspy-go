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
	Params map[string]any `json:"params,omitempty" yaml:"params,omitempty"`

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
	Params map[string]any `json:"params,omitempty" yaml:"params,omitempty"`

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

// GlobalRegistry is the global registry instance.
var (
	globalRegistryMu sync.RWMutex

	// GlobalRegistry is the package-level registry for backward compatibility.
	// Prefer GetRegistry and SetRegistry over direct mutation.
	GlobalRegistry LLMRegistry
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
	registry := GetRegistry()

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

	SetRegistry(registry)
	return nil
}

// SetRegistry replaces the package-level registry.
func SetRegistry(registry LLMRegistry) {
	globalRegistryMu.Lock()
	defer globalRegistryMu.Unlock()
	GlobalRegistry = registry
}

// GetRegistry returns the global registry instance.
func GetRegistry() LLMRegistry {
	globalRegistryMu.RLock()
	registry := GlobalRegistry
	globalRegistryMu.RUnlock()
	if registry != nil {
		return registry
	}

	globalRegistryMu.Lock()
	defer globalRegistryMu.Unlock()
	if GlobalRegistry == nil {
		// Initialize with empty config for backward compatibility
		GlobalRegistry = NewLLMRegistry()
	}
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
