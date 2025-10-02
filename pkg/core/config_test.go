package core

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestConfigureDefaultLLM(t *testing.T) {
	// Save original state
	originalDefaultLLM := GlobalConfig.DefaultLLM
	originalDefaultFactory := DefaultFactory
	defer func() {
		GlobalConfig.DefaultLLM = originalDefaultLLM
		DefaultFactory = originalDefaultFactory
	}()

	// Set up mock factory
	mockFactory := &MockLLMFactory{}
	DefaultFactory = mockFactory

	t.Run("ConfigureDefaultLLM_Success", func(t *testing.T) {
		err := ConfigureDefaultLLM("test-key", "test-model")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if GlobalConfig.DefaultLLM == nil {
			t.Error("Expected DefaultLLM to be set")
		}
	})

	t.Run("ConfigureDefaultLLM_Error", func(t *testing.T) {
		mockFactory.ShouldError = true
		err := ConfigureDefaultLLM("test-key", "test-model")
		if err == nil {
			t.Error("Expected error when factory fails")
		}
		if err.Error() != "failed to configure default LLM: mock factory error" {
			t.Errorf("Expected specific error message, got %v", err)
		}
	})
}

func TestConfigureTeacherLLM(t *testing.T) {
	// Save original state
	originalTeacherLLM := GlobalConfig.TeacherLLM
	originalDefaultFactory := DefaultFactory
	defer func() {
		GlobalConfig.TeacherLLM = originalTeacherLLM
		DefaultFactory = originalDefaultFactory
	}()

	// Set up mock factory
	mockFactory := &MockLLMFactory{}
	DefaultFactory = mockFactory

	t.Run("ConfigureTeacherLLM_Success", func(t *testing.T) {
		err := ConfigureTeacherLLM("test-key", "test-model")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if GlobalConfig.TeacherLLM == nil {
			t.Error("Expected TeacherLLM to be set")
		}
	})

	t.Run("ConfigureTeacherLLM_Error", func(t *testing.T) {
		mockFactory.ShouldError = true
		err := ConfigureTeacherLLM("test-key", "test-model")
		if err == nil {
			t.Error("Expected error when factory fails")
		}
		if err.Error() != "failed to configure teacher LLM: mock factory error" {
			t.Errorf("Expected specific error message, got %v", err)
		}
	})
}

func TestGetDefaultLLM(t *testing.T) {
	// Save original state
	originalDefaultLLM := GlobalConfig.DefaultLLM
	defer func() {
		GlobalConfig.DefaultLLM = originalDefaultLLM
	}()

	mockLLM := &MockLLM{}
	GlobalConfig.DefaultLLM = mockLLM

	result := GetDefaultLLM()
	if result != mockLLM {
		t.Error("Expected to get the configured default LLM")
	}
}

func TestGetTeacherLLM(t *testing.T) {
	// Save original state
	originalTeacherLLM := GlobalConfig.TeacherLLM
	defer func() {
		GlobalConfig.TeacherLLM = originalTeacherLLM
	}()

	mockLLM := &MockLLM{}
	GlobalConfig.TeacherLLM = mockLLM

	result := GetTeacherLLM()
	if result != mockLLM {
		t.Error("Expected to get the configured teacher LLM")
	}
}

func TestSetConcurrencyOptions(t *testing.T) {
	// Save original state
	originalConcurrency := GlobalConfig.ConcurrencyLevel
	defer func() {
		GlobalConfig.ConcurrencyLevel = originalConcurrency
	}()

	t.Run("SetConcurrencyOptions_ValidLevel", func(t *testing.T) {
		SetConcurrencyOptions(5)
		if GlobalConfig.ConcurrencyLevel != 5 {
			t.Errorf("Expected ConcurrencyLevel 5, got %d", GlobalConfig.ConcurrencyLevel)
		}
	})

	t.Run("SetConcurrencyOptions_InvalidLevel", func(t *testing.T) {
		SetConcurrencyOptions(0)
		if GlobalConfig.ConcurrencyLevel != 1 {
			t.Errorf("Expected ConcurrencyLevel to reset to 1, got %d", GlobalConfig.ConcurrencyLevel)
		}

		SetConcurrencyOptions(-1)
		if GlobalConfig.ConcurrencyLevel != 1 {
			t.Errorf("Expected ConcurrencyLevel to reset to 1, got %d", GlobalConfig.ConcurrencyLevel)
		}
	})
}

func TestConfigureFromRegistryConfig(t *testing.T) {
	// Save original state
	originalRegistry := GlobalConfig.Registry
	defer func() {
		GlobalConfig.Registry = originalRegistry
	}()

	ctx := context.Background()

	t.Run("ConfigureFromRegistryConfig_Success", func(t *testing.T) {
		config := RegistryConfig{
			Providers: map[string]ProviderConfig{
				"mock": {
					Name: "mock",
					Models: map[string]ModelConfig{
						"test-model": {
							ID:   "test-model",
							Name: "Test Model",
						},
					},
				},
			},
		}

		err := ConfigureFromRegistryConfig(ctx, config)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if GlobalConfig.Registry == nil {
			t.Error("Expected Registry to be set")
		}
	})

	t.Run("ConfigureFromRegistryConfig_Error", func(t *testing.T) {
		// Test with invalid default provider that doesn't exist
		config := RegistryConfig{
			Providers: map[string]ProviderConfig{
				"mock": {
					Name: "mock",
					Models: map[string]ModelConfig{
						"test-model": {
							ID:   "test-model",
							Name: "Test Model",
						},
					},
				},
			},
			DefaultProvider: "nonexistent", // This should fail
		}

		err := ConfigureFromRegistryConfig(ctx, config)
		if err == nil {
			t.Error("Expected error with invalid default provider")
		}
	})
}

func TestConfigureDefaultLLMFromRegistry(t *testing.T) {
	// Save original state
	originalDefaultLLM := GlobalConfig.DefaultLLM
	defer func() {
		GlobalConfig.DefaultLLM = originalDefaultLLM
	}()

	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		},
	}
	if err := registry.LoadFromConfig(context.Background(), config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	ctx := context.Background()

	t.Run("ConfigureDefaultLLMFromRegistry_Success", func(t *testing.T) {
		err := ConfigureDefaultLLMFromRegistry(ctx, "test-key", "test-model")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if GlobalConfig.DefaultLLM == nil {
			t.Error("Expected DefaultLLM to be set")
		}
	})

	t.Run("ConfigureDefaultLLMFromRegistry_Error", func(t *testing.T) {
		err := ConfigureDefaultLLMFromRegistry(ctx, "test-key", "nonexistent-model")
		if err == nil {
			t.Error("Expected error with nonexistent model")
		}
	})
}

func TestConfigureTeacherLLMFromRegistry(t *testing.T) {
	// Save original state
	originalTeacherLLM := GlobalConfig.TeacherLLM
	defer func() {
		GlobalConfig.TeacherLLM = originalTeacherLLM
	}()

	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		},
	}
	if err := registry.LoadFromConfig(context.Background(), config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	ctx := context.Background()

	t.Run("ConfigureTeacherLLMFromRegistry_Success", func(t *testing.T) {
		err := ConfigureTeacherLLMFromRegistry(ctx, "test-key", "test-model")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if GlobalConfig.TeacherLLM == nil {
			t.Error("Expected TeacherLLM to be set")
		}
	})

	t.Run("ConfigureTeacherLLMFromRegistry_Error", func(t *testing.T) {
		err := ConfigureTeacherLLMFromRegistry(ctx, "test-key", "nonexistent-model")
		if err == nil {
			t.Error("Expected error with nonexistent model")
		}
	})
}

func TestLoadLLMFromConfig(t *testing.T) {
	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	ctx := context.Background()

	t.Run("LoadLLMFromConfig_Success", func(t *testing.T) {
		config := ProviderConfig{
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		}

		llm, err := LoadLLMFromConfig(ctx, config, "test-model")
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if llm == nil {
			t.Error("Expected LLM to be returned")
		}
	})
}

func TestRefreshRegistryProvider(t *testing.T) {
	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	ctx := context.Background()

	t.Run("RefreshRegistryProvider_Success", func(t *testing.T) {
		config := ProviderConfig{
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		}

		err := RefreshRegistryProvider(ctx, "mock", config)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	})
}

func TestIsModelSupportedInRegistry(t *testing.T) {
	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		},
	}
	if err := registry.LoadFromConfig(context.Background(), config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	t.Run("IsModelSupportedInRegistry_Supported", func(t *testing.T) {
		result := IsModelSupportedInRegistry("test-model")
		if !result {
			t.Error("Expected test-model to be supported")
		}
	})

	t.Run("IsModelSupportedInRegistry_NotSupported", func(t *testing.T) {
		result := IsModelSupportedInRegistry("nonexistent-model")
		if result {
			t.Error("Expected nonexistent-model to not be supported")
		}
	})
}

func TestGetSupportedModels(t *testing.T) {
	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}
	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model-1": {
					ID:   "test-model-1",
					Name: "Test Model 1",
				},
				"test-model-2": {
					ID:   "test-model-2",
					Name: "Test Model 2",
				},
			},
		},
	}
	if err := registry.LoadFromConfig(context.Background(), config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	t.Run("GetSupportedModels_Success", func(t *testing.T) {
		result := GetSupportedModels()
		if len(result) != 1 {
			t.Errorf("Expected 1 provider, got %d", len(result))
		}
		if len(result["mock"]) != 2 {
			t.Errorf("Expected 2 models for mock provider, got %d", len(result["mock"]))
		}
	})
}

func TestCreateLLMWithTimeout(t *testing.T) {
	// Set up registry with mock provider
	registry := NewLLMRegistry()
	if err := registry.RegisterProvider("mock", MockProviderFactory); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	config := map[string]ProviderConfig{
		"mock": {
			Name: "mock",
			Models: map[string]ModelConfig{
				"test-model": {
					ID:   "test-model",
					Name: "Test Model",
				},
			},
		},
	}
	if err := registry.LoadFromConfig(context.Background(), config); err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// Override the global registry temporarily
	originalGlobalRegistry := GlobalRegistry
	defer func() {
		// Restore original registry getter
		GlobalRegistry = originalGlobalRegistry
	}()

	GlobalRegistry = registry

	t.Run("CreateLLMWithTimeout_Success", func(t *testing.T) {
		llm, err := CreateLLMWithTimeout("test-key", "test-model", 5*time.Second)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if llm == nil {
			t.Error("Expected LLM to be returned")
		}
	})

	t.Run("CreateLLMWithTimeout_Error", func(t *testing.T) {
		_, err := CreateLLMWithTimeout("test-key", "nonexistent-model", 5*time.Second)
		if err == nil {
			t.Error("Expected error with nonexistent model")
		}
	})
}

// TestSetDefaultLLM verifies that SetDefaultLLM correctly sets GlobalConfig.DefaultLLM
// This test ensures the fix for issue #145 works correctly.
func TestSetDefaultLLM(t *testing.T) {
	// Save original state
	originalDefaultLLM := GlobalConfig.DefaultLLM
	defer func() {
		GlobalConfig.DefaultLLM = originalDefaultLLM
	}()

	t.Run("SetDefaultLLM_SetsGlobalConfig", func(t *testing.T) {
		// Create a mock LLM
		mockLLM := &MockLLM{}

		// Call SetDefaultLLM
		SetDefaultLLM(mockLLM)

		// Verify it was set in GlobalConfig.DefaultLLM
		if GlobalConfig.DefaultLLM != mockLLM {
			t.Error("SetDefaultLLM should set GlobalConfig.DefaultLLM")
		}

		// Verify GetDefaultLLM returns the same instance
		if GetDefaultLLM() != mockLLM {
			t.Error("GetDefaultLLM should return the LLM set by SetDefaultLLM")
		}
	})

	t.Run("SetDefaultLLM_NilValue", func(t *testing.T) {
		// Should be able to set nil (clearing the default)
		SetDefaultLLM(nil)

		if GlobalConfig.DefaultLLM != nil {
			t.Error("SetDefaultLLM should be able to set nil")
		}

		if GetDefaultLLM() != nil {
			t.Error("GetDefaultLLM should return nil when default is nil")
		}
	})
}

// MockLLMFactory is a mock factory for testing.
type MockLLMFactory struct {
	ShouldError bool
}

func (f *MockLLMFactory) CreateLLM(apiKey string, modelID ModelID) (LLM, error) {
	if f.ShouldError {
		return nil, fmt.Errorf("mock factory error")
	}
	return &MockLLM{}, nil
}
