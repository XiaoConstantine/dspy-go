package core

import (
	"context"
	"sync"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// RegistryConstructor creates a concrete LLMRegistry backend for the
// compatibility-facing core registry entry points.
type RegistryConstructor func() LLMRegistry

var (
	registryConstructorMu sync.RWMutex
	registryConstructor   RegistryConstructor
)

// SetRegistryConstructor installs or replaces the constructor used by
// NewLLMRegistry and the GetRegistry lazy-init path.
func SetRegistryConstructor(constructor RegistryConstructor) {
	registryConstructorMu.Lock()
	defer registryConstructorMu.Unlock()
	registryConstructor = constructor
}

func getRegistryConstructor() RegistryConstructor {
	registryConstructorMu.RLock()
	defer registryConstructorMu.RUnlock()
	return registryConstructor
}

// NewLLMRegistry creates a lazily constructed registry wrapper backed by the
// currently installed registry constructor.
func NewLLMRegistry() LLMRegistry {
	return &lazyRegistry{}
}

type lazyRegistry struct {
	mu       sync.RWMutex
	delegate LLMRegistry
}

func (r *lazyRegistry) resolve() (LLMRegistry, error) {
	r.mu.RLock()
	if r.delegate != nil {
		delegate := r.delegate
		r.mu.RUnlock()
		return delegate, nil
	}
	r.mu.RUnlock()

	constructor := getRegistryConstructor()
	if constructor == nil {
		return nil, errors.New(
			errors.ConfigurationError,
			"llm registry backend is not installed; import pkg/llms or set a registry constructor explicitly",
		)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.delegate == nil {
		r.delegate = constructor()
	}
	return r.delegate, nil
}

func (r *lazyRegistry) RegisterProvider(name string, factory ProviderFactory) error {
	delegate, err := r.resolve()
	if err != nil {
		return err
	}
	return delegate.RegisterProvider(name, factory)
}

func (r *lazyRegistry) UnregisterProvider(name string) error {
	delegate, err := r.resolve()
	if err != nil {
		return err
	}
	return delegate.UnregisterProvider(name)
}

func (r *lazyRegistry) CreateLLM(ctx context.Context, apiKey string, modelID ModelID) (LLM, error) {
	delegate, err := r.resolve()
	if err != nil {
		return nil, err
	}
	return delegate.CreateLLM(ctx, apiKey, modelID)
}

func (r *lazyRegistry) CreateLLMWithConfig(ctx context.Context, config ProviderConfig, modelID ModelID) (LLM, error) {
	delegate, err := r.resolve()
	if err != nil {
		return nil, err
	}
	return delegate.CreateLLMWithConfig(ctx, config, modelID)
}

func (r *lazyRegistry) LoadFromConfig(ctx context.Context, configs map[string]ProviderConfig) error {
	delegate, err := r.resolve()
	if err != nil {
		return err
	}
	return delegate.LoadFromConfig(ctx, configs)
}

func (r *lazyRegistry) ListProviders() []string {
	delegate, err := r.resolve()
	if err != nil {
		return nil
	}
	return delegate.ListProviders()
}

func (r *lazyRegistry) GetProviderConfig(name string) (ProviderConfig, bool) {
	delegate, err := r.resolve()
	if err != nil {
		return ProviderConfig{}, false
	}
	return delegate.GetProviderConfig(name)
}

func (r *lazyRegistry) IsModelSupported(modelID ModelID) bool {
	delegate, err := r.resolve()
	if err != nil {
		return false
	}
	return delegate.IsModelSupported(modelID)
}

func (r *lazyRegistry) GetModelProvider(modelID ModelID) (string, bool) {
	delegate, err := r.resolve()
	if err != nil {
		return "", false
	}
	return delegate.GetModelProvider(modelID)
}

func (r *lazyRegistry) SetDefaultProvider(name string) error {
	delegate, err := r.resolve()
	if err != nil {
		return err
	}
	return delegate.SetDefaultProvider(name)
}

func (r *lazyRegistry) RefreshProvider(ctx context.Context, name string, config ProviderConfig) error {
	delegate, err := r.resolve()
	if err != nil {
		return err
	}
	return delegate.RefreshProvider(ctx, name, config)
}

var _ LLMRegistry = (*lazyRegistry)(nil)
