package core

import (
	"context"
	"sync"
	"time"
)

// LLMFactory defines a simple interface for creating LLM instances.
// This maintains compatibility with existing code while allowing for configuration.
type LLMFactory interface {
	// CreateLLM creates a new LLM instance. It uses the global configuration
	// from core.GlobalConfig for client settings.
	CreateLLM(apiKey string, modelID ModelID) (LLM, error)
}

// RegistryBasedFactory implements LLMFactory using the registry system.
type RegistryBasedFactory struct {
	registry LLMRegistry
}

// NewRegistryBasedFactory creates a new factory that uses the registry system.
func NewRegistryBasedFactory(registry LLMRegistry) *RegistryBasedFactory {
	return &RegistryBasedFactory{
		registry: registry,
	}
}

// CreateLLM creates an LLM instance using the registry.
func (f *RegistryBasedFactory) CreateLLM(apiKey string, modelID ModelID) (LLM, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return f.registry.CreateLLM(ctx, apiKey, modelID)
}

var defaultFactoryMu sync.RWMutex

// DefaultFactory is the global factory instance used by the configuration system.
// Prefer GetDefaultFactory and SetDefaultFactory over direct mutation.
var DefaultFactory LLMFactory

// SetDefaultFactory replaces the package default LLM factory.
func SetDefaultFactory(factory LLMFactory) {
	defaultFactoryMu.Lock()
	defer defaultFactoryMu.Unlock()
	DefaultFactory = factory
}

// InitializeDefaultFactory initializes the default factory with the registry system.
func InitializeDefaultFactory() {
	defaultFactoryMu.Lock()
	defer defaultFactoryMu.Unlock()
	if DefaultFactory == nil {
		DefaultFactory = NewRegistryBasedFactory(GetRegistry())
	}
}

// GetDefaultFactory returns the default factory, initializing it if necessary.
func GetDefaultFactory() LLMFactory {
	defaultFactoryMu.RLock()
	factory := DefaultFactory
	defaultFactoryMu.RUnlock()
	if factory != nil {
		return factory
	}

	InitializeDefaultFactory()

	defaultFactoryMu.RLock()
	defer defaultFactoryMu.RUnlock()
	return DefaultFactory
}
