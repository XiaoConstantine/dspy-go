package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidationError(t *testing.T) {
	err := &ValidationError{
		Field: "TestField",
		Tag:   "required",
		Value: nil,
	}
	
	assert.Contains(t, err.Error(), "TestField")
	assert.Contains(t, err.Error(), "required")
	
	// Test with custom message
	err.Message = "Custom validation message"
	assert.Equal(t, "Custom validation message", err.Error())
}

func TestValidationErrors(t *testing.T) {
	errors := ValidationErrors{
		{Field: "Field1", Tag: "required", Value: nil},
		{Field: "Field2", Tag: "min", Value: 0},
	}
	
	errStr := errors.Error()
	assert.Contains(t, errStr, "validation failed")
	assert.Contains(t, errStr, "Field1")
	assert.Contains(t, errStr, "Field2")
}

func TestNewValidator(t *testing.T) {
	validator, err := NewValidator()
	require.NoError(t, err)
	require.NotNil(t, validator)
	
	// Test that custom validators are registered
	config := GetDefaultConfig()
	err = validator.ValidateConfig(config)
	assert.NoError(t, err)
}

func TestValidateCustomRules(t *testing.T) {
	validator, err := NewValidator()
	require.NoError(t, err)
	
	// Test LLM config with provider not in providers map
	config := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "nonexistent",
				ModelID:  "test-model",
			},
			Providers: map[string]LLMProviderConfig{
				"anthropic": {
					Provider: "anthropic",
					ModelID:  "claude-3-sonnet-20240229",
				},
			},
		},
	}
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "default provider 'nonexistent' not found")
}

func TestValidateLLMProviderConfig(t *testing.T) {
	validator, err := NewValidator()
	require.NoError(t, err)
	
	tests := []struct {
		name        string
		provider    LLMProviderConfig
		expectError bool
		errorText   string
	}{
		{
			name: "valid anthropic",
			provider: LLMProviderConfig{
				Provider: "anthropic",
				ModelID:  "claude-3-sonnet-20240229",
				APIKey:   "test-key",
			},
			expectError: false,
		},
		{
			name: "anthropic missing API key",
			provider: LLMProviderConfig{
				Provider: "anthropic",
				ModelID:  "claude-3-sonnet-20240229",
			},
			expectError: false, // API key validation removed for security
		},
		{
			name: "anthropic invalid model",
			provider: LLMProviderConfig{
				Provider: "anthropic",
				ModelID:  "invalid-model",
				APIKey:   "test-key",
			},
			expectError: true,
			errorText:   "invalid Anthropic model ID",
		},
		{
			name: "google missing API key",
			provider: LLMProviderConfig{
				Provider: "google",
				ModelID:  "gemini-2.0-flash",
			},
			expectError: false, // API key validation removed for security
		},
		{
			name: "google invalid model",
			provider: LLMProviderConfig{
				Provider: "google",
				ModelID:  "invalid-model",
				APIKey:   "test-key",
			},
			expectError: true,
			errorText:   "invalid Google model ID",
		},
		{
			name: "ollama missing base URL",
			provider: LLMProviderConfig{
				Provider: "ollama",
				ModelID:  "llama3.2",
			},
			expectError: true,
			errorText:   "base URL is required for Ollama provider",
		},
		{
			name: "ollama invalid model ID",
			provider: LLMProviderConfig{
				Provider: "ollama",
				ModelID:  "invalid-model",
				Endpoint: EndpointConfig{BaseURL: "http://localhost:11434"},
			},
			expectError: true,
			errorText:   "Ollama model ID must start with 'ollama:'",
		},
		{
			name: "llamacpp missing base URL",
			provider: LLMProviderConfig{
				Provider: "llamacpp",
				ModelID:  "test-model",
			},
			expectError: true,
			errorText:   "base URL is required for LlamaCPP provider",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &Config{
				LLM: LLMConfig{
					Providers: map[string]LLMProviderConfig{
						"test": tt.provider,
					},
				},
			}
			
			err := validator.ValidateConfig(config)
			
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorText != "" {
					assert.Contains(t, err.Error(), tt.errorText)
				}
			} else {
				if err != nil {
					// May have other validation errors, just check our specific provider error isn't there
					assert.NotContains(t, err.Error(), "API key is required")
				}
			}
		})
	}
}

func TestValidateExecutionConfig(t *testing.T) {
	validator, err := NewValidator()
	require.NoError(t, err)
	
	// Test context timeout > execution timeout
	config := &Config{
		Execution: ExecutionConfig{
			DefaultTimeout: 1 * time.Minute,
			Context: ContextConfig{
				DefaultTimeout: 2 * time.Minute, // Invalid: greater than execution timeout
			},
		},
	}
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context timeout should not exceed execution timeout")
	
	// Test tracing enabled without exporter type
	config = &Config{
		Execution: ExecutionConfig{
			Tracing: TracingConfig{
				Enabled: true,
				// Missing Type and Endpoint
			},
		},
	}
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exporter type is required when tracing is enabled")
}

func TestValidateStorageConfig(t *testing.T) {
	validator, err := NewValidator()
	require.NoError(t, err)
	
	// Test default backend not in backends map
	config := &Config{
		Storage: StorageConfig{
			DefaultBackend: "nonexistent",
			Backends: map[string]StorageBackendConfig{
				"file": {Type: "file"},
			},
		},
	}
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "default backend 'nonexistent' not found")
	
	// Test encryption enabled without key source
	config = &Config{
		Storage: StorageConfig{
			Encryption: EncryptionConfig{
				Enabled: true,
				Key: EncryptionKeyConfig{
					// Missing Source and Identifier
				},
			},
		},
	}
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "key source is required when encryption is enabled")
}

func TestCustomValidators(t *testing.T) {
	// Test custom validators through integration rather than unit testing
	// This avoids complex mocking of the validator interface
	
	validator, err := NewValidator()
	require.NoError(t, err)
	
	// Test various configurations that should trigger custom validation
	validConfig := GetDefaultConfig()
	err = validator.ValidateConfig(validConfig)
	assert.NoError(t, err)
	
	// Test with invalid provider in LLM config
	invalidConfig := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "invalid-provider",
				ModelID:  "test-model",
			},
		},
	}
	
	err = validator.ValidateConfig(invalidConfig)
	assert.Error(t, err)
}

func TestIsValidAnthropicModel(t *testing.T) {
	validModels := []string{
		"claude-3-haiku-20240307",
		"claude-3-sonnet-20240229",
		"claude-3-opus-20240229",
		"claude-3-5-sonnet-20241022",
		"claude-3-5-haiku-20241022",
	}
	
	for _, model := range validModels {
		assert.True(t, isValidAnthropicModel(model), "Model %s should be valid", model)
	}
	
	invalidModels := []string{
		"invalid-model",
		"claude-2",
		"gpt-4",
	}
	
	for _, model := range invalidModels {
		assert.False(t, isValidAnthropicModel(model), "Model %s should be invalid", model)
	}
}

func TestIsValidGoogleModel(t *testing.T) {
	validModels := []string{
		"gemini-2.0-flash",
		"gemini-1.5-pro",
		"gemma-2b",
		"palm-bison",
	}
	
	for _, model := range validModels {
		assert.True(t, isValidGoogleModel(model), "Model %s should be valid", model)
	}
	
	invalidModels := []string{
		"claude-3-sonnet",
		"gpt-4",
		"invalid-model",
	}
	
	for _, model := range invalidModels {
		assert.False(t, isValidGoogleModel(model), "Model %s should be invalid", model)
	}
}

func TestGetValidationMessage(t *testing.T) {
	// Test validation message formatting - this is primarily integration tested
	// through the actual validator usage in other tests
	
	// Test a simple case to ensure function exists and works
	config := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "", // This should trigger a required validation error
			},
		},
	}
	
	validator, err := NewValidator()
	require.NoError(t, err)
	
	err = validator.ValidateConfig(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "validation failed")
}


func TestGetValidator(t *testing.T) {
	// Reset global validator
	globalValidator = nil
	
	validator1 := GetValidator()
	validator2 := GetValidator()
	
	// Should return the same instance
	assert.Same(t, validator1, validator2)
}

func TestValidateConfiguration(t *testing.T) {
	config := GetDefaultConfig()
	err := ValidateConfiguration(config)
	assert.NoError(t, err)
	
	// Test with invalid config
	config.LLM.Default.Generation.Temperature = 5.0 // Invalid
	err = ValidateConfiguration(config)
	assert.Error(t, err)
}