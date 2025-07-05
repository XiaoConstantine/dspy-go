package core

import (
	"context"
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

// DefaultFactory is the global factory instance used by the configuration system.
var DefaultFactory LLMFactory

// InitializeDefaultFactory initializes the default factory with the registry system.
func InitializeDefaultFactory() {
	if DefaultFactory == nil {
		DefaultFactory = NewRegistryBasedFactory(GetRegistry())
	}
}

// GetDefaultFactory returns the default factory, initializing it if necessary.
func GetDefaultFactory() LLMFactory {
	if DefaultFactory == nil {
		InitializeDefaultFactory()
	}
	return DefaultFactory
}
