package core

import (
	"context"
	"fmt"
	"time"
)

type Config struct {
	DefaultLLM       LLM
	TeacherLLM       LLM
	ConcurrencyLevel int

	// Registry configuration
	Registry *RegistryConfig `json:"registry,omitempty" yaml:"registry,omitempty"`
}

var GlobalConfig = &Config{
	// default concurrency 1
	ConcurrencyLevel: 1,
}

// ConfigureDefaultLLM sets up the default LLM to be used across the package.
func ConfigureDefaultLLM(apiKey string, modelID ModelID) error {
	llmInstance, err := DefaultFactory.CreateLLM(apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure default LLM: %w", err)
	}
	GlobalConfig.DefaultLLM = llmInstance
	return nil
}

// ConfigureTeacherLLM sets up the teacher LLM.
func ConfigureTeacherLLM(apiKey string, modelID ModelID) error {
	llmInstance, err := DefaultFactory.CreateLLM(apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure teacher LLM: %w", err)
	}
	GlobalConfig.TeacherLLM = llmInstance
	return nil
}

// GetDefaultLLM returns the default LLM.
func GetDefaultLLM() LLM {
	return GlobalConfig.DefaultLLM
}

// GetTeacherLLM returns the teacher LLM.
func GetTeacherLLM() LLM {
	return GlobalConfig.TeacherLLM
}

func SetConcurrencyOptions(level int) {
	if level > 0 {
		GlobalConfig.ConcurrencyLevel = level
	} else {
		GlobalConfig.ConcurrencyLevel = 1 // Reset to default value for invalid inputs
	}
}

// ConfigureFromRegistryConfig initializes the global configuration with registry settings.
func ConfigureFromRegistryConfig(ctx context.Context, config RegistryConfig) error {
	// Initialize the registry with the provided configuration
	if err := InitializeRegistry(ctx, config); err != nil {
		return fmt.Errorf("failed to initialize registry: %w", err)
	}

	// Update global config to include registry configuration
	GlobalConfig.Registry = &config

	return nil
}

// ConfigureDefaultLLMFromRegistry sets up the default LLM using the registry.
func ConfigureDefaultLLMFromRegistry(ctx context.Context, apiKey string, modelID ModelID) error {
	registry := GetRegistry()
	llmInstance, err := registry.CreateLLM(ctx, apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure default LLM from registry: %w", err)
	}
	GlobalConfig.DefaultLLM = llmInstance
	return nil
}

// ConfigureTeacherLLMFromRegistry sets up the teacher LLM using the registry.
func ConfigureTeacherLLMFromRegistry(ctx context.Context, apiKey string, modelID ModelID) error {
	registry := GetRegistry()
	llmInstance, err := registry.CreateLLM(ctx, apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure teacher LLM from registry: %w", err)
	}
	GlobalConfig.TeacherLLM = llmInstance
	return nil
}

// LoadLLMFromConfig creates an LLM instance from a complete provider configuration.
func LoadLLMFromConfig(ctx context.Context, providerConfig ProviderConfig, modelID ModelID) (LLM, error) {
	registry := GetRegistry()
	return registry.CreateLLMWithConfig(ctx, providerConfig, modelID)
}

// RefreshRegistryProvider reloads a provider configuration.
func RefreshRegistryProvider(ctx context.Context, providerName string, config ProviderConfig) error {
	registry := GetRegistry()
	return registry.RefreshProvider(ctx, providerName, config)
}

// IsModelSupportedInRegistry checks if a model is supported by the registry.
func IsModelSupportedInRegistry(modelID ModelID) bool {
	registry := GetRegistry()
	return registry.IsModelSupported(modelID)
}

// GetSupportedModels returns all models supported by all providers in the registry.
func GetSupportedModels() map[string][]string {
	registry := GetRegistry()
	providers := registry.ListProviders()
	result := make(map[string][]string)

	for _, providerName := range providers {
		config, exists := registry.GetProviderConfig(providerName)
		if !exists {
			continue
		}

		models := make([]string, 0, len(config.Models))
		for modelID := range config.Models {
			models = append(models, modelID)
		}
		result[providerName] = models
	}

	return result
}

// CreateLLMWithTimeout creates an LLM with a specified timeout.
func CreateLLMWithTimeout(apiKey string, modelID ModelID, timeout time.Duration) (LLM, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	registry := GetRegistry()
	return registry.CreateLLM(ctx, apiKey, modelID)
}
