package config

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	config := GetDefaultConfig()

	// Test that we get a valid configuration
	require.NotNil(t, config)

	// Test LLM configuration
	assert.Equal(t, "anthropic", config.LLM.Default.Provider)
	assert.Equal(t, "claude-3-5-sonnet-20250929", config.LLM.Default.ModelID)
	assert.Equal(t, 8192, config.LLM.Default.Generation.MaxTokens)
	assert.Equal(t, 0.5, config.LLM.Default.Generation.Temperature)

	// Test that providers map is populated
	assert.Contains(t, config.LLM.Providers, "anthropic")
	assert.Contains(t, config.LLM.Providers, "google")
	assert.Contains(t, config.LLM.Providers, "ollama")
	assert.Contains(t, config.LLM.Providers, "llamacpp")

	// Test logging configuration
	assert.Equal(t, "INFO", config.Logging.Level)
	assert.Len(t, config.Logging.Outputs, 1)
	assert.Equal(t, "console", config.Logging.Outputs[0].Type)

	// Test execution configuration
	assert.Equal(t, 5*time.Minute, config.Execution.DefaultTimeout)
	assert.Equal(t, 10, config.Execution.MaxConcurrency)

	// Test that configuration validates
	err := config.Validate()
	assert.NoError(t, err)
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      *Config
		expectError bool
		errorField  string
	}{
		{
			name:        "valid default config",
			config:      GetDefaultConfig(),
			expectError: false,
		},
		{
			name: "missing LLM provider",
			config: &Config{
				LLM: LLMConfig{
					Default: LLMProviderConfig{
						Provider: "", // Invalid: empty provider
						ModelID:  "test-model",
					},
				},
			},
			expectError: true,
			errorField:  "Provider",
		},
		{
			name: "invalid temperature",
			config: &Config{
				LLM: LLMConfig{
					Default: LLMProviderConfig{
						Provider: "anthropic",
						ModelID:  "claude-3-sonnet-20240229",
						Generation: GenerationConfig{
							Temperature: 3.0, // Invalid: > 2.0
						},
					},
				},
			},
			expectError: true,
			errorField:  "Temperature",
		},
		{
			name: "invalid max tokens",
			config: &Config{
				LLM: LLMConfig{
					Default: LLMProviderConfig{
						Provider: "anthropic",
						ModelID:  "claude-3-sonnet-20240229",
						Generation: GenerationConfig{
							MaxTokens: 0, // Invalid: must be > 0
						},
					},
				},
			},
			expectError: true,
			errorField:  "MaxTokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorField != "" {
					assert.Contains(t, err.Error(), tt.errorField)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfigManager(t *testing.T) {
	// Create a temporary config file
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test_config.yaml")

	// Create manager with custom path and only file source to avoid environment interference
	manager, err := NewManager(
		WithConfigPath(configPath),
		WithSources(NewFileSource()),
	)
	require.NoError(t, err)

	// Test loading default configuration
	err = manager.Load()
	require.NoError(t, err)

	config := manager.Get()
	require.NotNil(t, config)

	// Test that we can get specific configuration sections
	llmConfig := manager.GetLLMConfig()
	require.NotNil(t, llmConfig)
	assert.Equal(t, "anthropic", llmConfig.Default.Provider)

	loggingConfig := manager.GetLoggingConfig()
	require.NotNil(t, loggingConfig)
	assert.Equal(t, "INFO", loggingConfig.Level)

	// Test saving configuration
	err = manager.Save()
	require.NoError(t, err)

	// Verify file was created
	assert.FileExists(t, configPath)

	// Test loading from saved file
	manager2, err := NewManager(WithConfigPath(configPath))
	require.NoError(t, err)

	err = manager2.Load()
	require.NoError(t, err)

	config2 := manager2.Get()
	assert.Equal(t, config.LLM.Default.Provider, config2.LLM.Default.Provider)
}

func TestConfigDiscovery(t *testing.T) {
	tempDir := t.TempDir()

	// Create a config file in temp directory
	configPath := filepath.Join(tempDir, "dspy.yaml")
	defaultConfig := GetDefaultConfig()

	manager := &Manager{config: defaultConfig}
	err := manager.SaveToFile(configPath)
	require.NoError(t, err)

	// Test discovery
	discovery := NewDiscoveryWithPaths([]string{tempDir})
	discoveredFiles, err := discovery.Discover()
	require.NoError(t, err)

	assert.Len(t, discoveredFiles, 1)
	assert.Contains(t, discoveredFiles[0], "dspy.yaml")
}

func TestEnvironmentSource(t *testing.T) {
	// Set environment variables
	os.Setenv("DSPY_LLM_DEFAULT_PROVIDER", "google")
	os.Setenv("DSPY_LLM_DEFAULT_MODEL_ID", "gemini-2.0-flash")
	os.Setenv("DSPY_LLM_GLOBAL_CONCURRENCY_LEVEL", "5")
	os.Setenv("DSPY_LOGGING_LEVEL", "DEBUG")

	defer func() {
		os.Unsetenv("DSPY_LLM_DEFAULT_PROVIDER")
		os.Unsetenv("DSPY_LLM_DEFAULT_MODEL_ID")
		os.Unsetenv("DSPY_LLM_GLOBAL_CONCURRENCY_LEVEL")
		os.Unsetenv("DSPY_LOGGING_LEVEL")
	}()

	config := GetDefaultConfig()
	envSource := NewEnvironmentSource()

	err := envSource.Load(config, nil)
	require.NoError(t, err)

	// Check that environment variables were applied
	assert.Equal(t, "google", config.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", config.LLM.Default.ModelID)
	assert.Equal(t, 5, config.LLM.GlobalSettings.ConcurrencyLevel)
	assert.Equal(t, "DEBUG", config.Logging.Level)
}

func TestFileSource(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test.yaml")

	// Create a test config file
	configYAML := `
llm:
  default:
    provider: "google"
    model_id: "gemini-2.0-flash"
    generation:
      max_tokens: 4096
      temperature: 0.7
  global_settings:
    concurrency_level: 3

logging:
  level: "WARN"
  sample_rate: 2

execution:
  default_timeout: "10m"
  max_concurrency: 20
`

	err := os.WriteFile(configPath, []byte(configYAML), 0644)
	require.NoError(t, err)

	config := GetDefaultConfig()
	fileSource := NewFileSource()

	err = fileSource.Load(config, []string{configPath})
	require.NoError(t, err)

	// Check that file configuration was applied
	assert.Equal(t, "google", config.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", config.LLM.Default.ModelID)
	assert.Equal(t, 4096, config.LLM.Default.Generation.MaxTokens)
	assert.Equal(t, 0.7, config.LLM.Default.Generation.Temperature)
	assert.Equal(t, 3, config.LLM.GlobalSettings.ConcurrencyLevel)
	assert.Equal(t, "WARN", config.Logging.Level)
	assert.Equal(t, uint32(2), config.Logging.SampleRate)
	assert.Equal(t, 10*time.Minute, config.Execution.DefaultTimeout)
	assert.Equal(t, 20, config.Execution.MaxConcurrency)
}

func TestMultiSource(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test.yaml")

	// Create a test config file
	configYAML := `
llm:
  default:
    provider: "google"
    model_id: "gemini-2.0-flash"
    generation:
      temperature: 0.7

logging:
  level: "WARN"
`

	err := os.WriteFile(configPath, []byte(configYAML), 0644)
	require.NoError(t, err)

	// Set environment variables (higher priority)
	os.Setenv("DSPY_LLM_DEFAULT_PROVIDER", "anthropic")
	os.Setenv("DSPY_LOGGING_LEVEL", "ERROR")

	defer func() {
		os.Unsetenv("DSPY_LLM_DEFAULT_PROVIDER")
		os.Unsetenv("DSPY_LOGGING_LEVEL")
	}()

	config := GetDefaultConfig()

	// Create multi-source with file and environment
	multiSource := NewMultiSource(
		NewFileSource(),
		NewEnvironmentSource(),
	)

	err = multiSource.Load(config, []string{configPath})
	require.NoError(t, err)

	// Environment should override file
	assert.Equal(t, "anthropic", config.LLM.Default.Provider)       // From env
	assert.Equal(t, "gemini-2.0-flash", config.LLM.Default.ModelID) // From file
	assert.Equal(t, 0.7, config.LLM.Default.Generation.Temperature) // From file
	assert.Equal(t, "ERROR", config.Logging.Level)                  // From env
}

func TestGlobalManager(t *testing.T) {
	// Reset global manager
	globalManager = nil
	globalManagerOnce = sync.Once{}

	// Create a clean manager for testing
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test_config.yaml")

	manager, err := NewManager(
		WithConfigPath(configPath),
		WithSources(NewFileSource()), // Only file source to avoid external interference
	)
	require.NoError(t, err)

	SetGlobalManager(manager)

	manager2 := GetGlobalManager()
	assert.Same(t, manager, manager2)

	// Test loading global config
	err = LoadGlobalConfig()
	require.NoError(t, err)

	config := GetGlobalConfig()
	require.NotNil(t, config)
	assert.Equal(t, "anthropic", config.LLM.Default.Provider)
}

func TestConfigUpdate(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test_config.yaml")

	manager, err := NewManager(
		WithConfigPath(configPath),
		WithSources(NewFileSource()),
	)
	require.NoError(t, err)

	err = manager.Load()
	require.NoError(t, err)

	// Test updating configuration
	err = manager.Update(func(config *Config) error {
		config.LLM.Default.Provider = "google"
		config.LLM.Default.ModelID = "gemini-2.0-flash"
		config.LLM.GlobalSettings.ConcurrencyLevel = 5
		return nil
	})
	require.NoError(t, err)

	config := manager.Get()
	assert.Equal(t, "google", config.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", config.LLM.Default.ModelID)
	assert.Equal(t, 5, config.LLM.GlobalSettings.ConcurrencyLevel)
}

func TestConfigWatcher(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "test_config.yaml")

	var watchedConfig *Config
	watcher := func(config *Config) error {
		watchedConfig = config
		return nil
	}

	manager, err := NewManager(
		WithConfigPath(configPath),
		WithSources(NewFileSource()),
		WithWatcher(watcher),
	)
	require.NoError(t, err)

	err = manager.Load()
	require.NoError(t, err)

	// Update configuration to trigger watcher
	err = manager.Update(func(config *Config) error {
		config.LLM.Default.Provider = "google"
		return nil
	})
	require.NoError(t, err)

	// Verify watcher was called
	require.NotNil(t, watchedConfig)
	assert.Equal(t, "google", watchedConfig.LLM.Default.Provider)
}

func TestConfigValidationErrors(t *testing.T) {
	config := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "invalid_provider",
				ModelID:  "test-model",
				Generation: GenerationConfig{
					MaxTokens:   -1,  // Invalid
					Temperature: 5.0, // Invalid
					TopP:        2.0, // Invalid
				},
			},
		},
	}

	err := config.Validate()
	require.Error(t, err)

	// Check that it's a ValidationErrors type
	if validationErrs, ok := err.(ValidationErrors); ok {
		assert.Greater(t, len(validationErrs), 0)

		// Check that specific fields are mentioned
		errStr := err.Error()
		assert.Contains(t, errStr, "Provider")
		assert.Contains(t, errStr, "MaxTokens")
		assert.Contains(t, errStr, "Temperature")
	}
}

func TestProviderSpecificValidation(t *testing.T) {
	config := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "anthropic",
				ModelID:  "invalid-anthropic-model",
			},
			Providers: map[string]LLMProviderConfig{
				"anthropic": {
					Provider: "anthropic",
					ModelID:  "invalid-anthropic-model",
				},
			},
		},
	}

	// This should trigger custom validation
	validator, err := NewValidator()
	require.NoError(t, err)

	err = validator.ValidateConfig(config)
	// Note: The current validation doesn't implement provider-specific model validation
	// This test demonstrates where such validation would be added
	if err != nil {
		t.Logf("Validation error (expected for invalid model): %v", err)
	}
}

// Test helper to verify the configuration system is working end-to-end.
func TestConfigurationSystemIntegration(t *testing.T) {
	tempDir := t.TempDir()

	// Create a test scenario with file and environment configuration
	configPath := filepath.Join(tempDir, "dspy.yaml")
	configYAML := `
llm:
  default:
    provider: "google"
    model_id: "gemini-2.0-flash"
    api_key: "test-key-from-file"
    endpoint:
      timeout: "45s"
      retry:
        max_retries: 3
        initial_backoff: "1s"
        max_backoff: "30s"
        backoff_multiplier: 2.0
    generation:
      max_tokens: 4096
      temperature: 0.7
      top_p: 1.0
      presence_penalty: 0.0
      frequency_penalty: 0.0
      top_k: 50
    embedding:
      batch_size: 32
  teacher:
    provider: "google"
    model_id: "gemini-2.0-flash"
    api_key: "test-teacher-key"
    endpoint:
      timeout: "60s"
      retry:
        max_retries: 3
        initial_backoff: "1s"
        max_backoff: "30s"
        backoff_multiplier: 2.0
    generation:
      max_tokens: 8192
      temperature: 0.5
      top_p: 1.0
      presence_penalty: 0.0
      frequency_penalty: 0.0
      top_k: 50
    embedding:
      batch_size: 32
  global_settings:
    concurrency_level: 2
    log_requests: true

logging:
  level: "DEBUG"
  outputs:
    - type: "console"
      format: "json"
      colors: false

execution:
  default_timeout: "15m"
  max_concurrency: 8
  context:
    default_timeout: "5m"
    buffer_size: 1000
  tracing:
    enabled: true
    sampling_rate: 0.1
    exporter:
      type: "jaeger"
      endpoint: "http://localhost:14268"

modules:
  chain_of_thought:
    max_steps: 10
  multi_chain_comparison:
    num_chains: 3
    comparison_strategy: "majority_vote"
  react:
    max_cycles: 10
    action_timeout: "30s"
  refine:
    max_iterations: 5
    refinement_strategy: "iterative_improvement"
  predict:
    default_settings:
      top_k: 50
    caching:
      enabled: true
      ttl: "1h"
      max_size: 1000
      type: "memory"

agents:
  default:
    max_history: 100
    timeout: "5m"
    tool_use:
      max_tools: 10
      timeout: "1m"
  memory:
    type: "buffered"
    capacity: 1000
    persistence:
      enabled: true
      sync_interval: "5m"
  workflows:
    default_timeout: "10m"
    max_parallel: 5
    persistence:
      enabled: true
      backend: "file"

tools:
  registry:
    max_tools: 100
  mcp:
    default_timeout: "30s"
    connection_pool:
      max_connections: 10
      connection_timeout: "10s"
      idle_timeout: "5m"
  functions:
    max_execution_time: "30s"
    sandbox:
      type: "docker"
      resource_limits:
        memory_mb: 512
        cpu_cores: 1
        execution_timeout: "30s"

optimizers:
  bootstrap_few_shot:
    max_examples: 10
    bootstrap_iterations: 3
  mipro:
    population_size: 20
    num_generations: 10
  copro:
    max_iterations: 20
  simba:
    num_candidates: 10
    selection_strategy: "tournament"
  tpe:
    num_trials: 100
    num_startup_trials: 10

storage:
  default_backend: "file"
  compression:
    enabled: true
    algorithm: "gzip"
    level: 6
  encryption:
    enabled: true
    algorithm: "aes256"
    key_derivation: "pbkdf2"
    key:
      source: "env"
      identifier: "TEST_ENCRYPTION_KEY"
      rotation:
        enabled: false
        interval: "24h"
`

	err := os.WriteFile(configPath, []byte(configYAML), 0644)
	require.NoError(t, err)

	// Set environment variable overrides
	os.Setenv("DSPY_LLM_DEFAULT_API_KEY", "env-override-key")
	os.Setenv("DSPY_LLM_GLOBAL_CONCURRENCY_LEVEL", "4")
	os.Setenv("DSPY_EXECUTION_MAX_CONCURRENCY", "12")

	defer func() {
		os.Unsetenv("DSPY_LLM_DEFAULT_API_KEY")
		os.Unsetenv("DSPY_LLM_GLOBAL_CONCURRENCY_LEVEL")
		os.Unsetenv("DSPY_EXECUTION_MAX_CONCURRENCY")
	}()

	// Create manager with file and environment sources
	manager, err := NewManager(
		WithConfigPath(configPath),
		WithSources(
			NewFileSource(),
			NewEnvironmentSource(),
		),
	)
	require.NoError(t, err)

	// Load configuration
	err = manager.Load()
	require.NoError(t, err)

	config := manager.Get()
	require.NotNil(t, config)

	// Verify file configuration was loaded
	assert.Equal(t, "google", config.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", config.LLM.Default.ModelID)
	assert.Equal(t, 4096, config.LLM.Default.Generation.MaxTokens)
	assert.Equal(t, 0.7, config.LLM.Default.Generation.Temperature)
	assert.Equal(t, 45*time.Second, config.LLM.Default.Endpoint.Timeout)
	assert.Equal(t, true, config.LLM.GlobalSettings.LogRequests)
	assert.Equal(t, "DEBUG", config.Logging.Level)
	assert.Equal(t, "json", config.Logging.Outputs[0].Format)
	assert.Equal(t, false, config.Logging.Outputs[0].Colors)
	assert.Equal(t, 15*time.Minute, config.Execution.DefaultTimeout)
	assert.Equal(t, true, config.Execution.Tracing.Enabled)
	assert.Equal(t, 0.1, config.Execution.Tracing.SamplingRate)

	// Verify environment overrides took effect
	assert.Equal(t, "env-override-key", config.LLM.Default.APIKey) // From env
	assert.Equal(t, 4, config.LLM.GlobalSettings.ConcurrencyLevel) // From env
	assert.Equal(t, 12, config.Execution.MaxConcurrency)           // From env

	// Verify configuration validates
	err = config.Validate()
	assert.NoError(t, err)

	// Test that we can save and reload
	newConfigPath := filepath.Join(tempDir, "saved_config.yaml")
	err = manager.SaveToFile(newConfigPath)
	require.NoError(t, err)

	// Load with new manager
	manager2, err := NewManager(WithConfigPath(newConfigPath))
	require.NoError(t, err)

	err = manager2.Load()
	require.NoError(t, err)

	config2 := manager2.Get()

	// Verify the saved configuration is equivalent
	assert.Equal(t, config.LLM.Default.Provider, config2.LLM.Default.Provider)
	assert.Equal(t, config.LLM.Default.ModelID, config2.LLM.Default.ModelID)
	assert.Equal(t, config.LLM.Default.APIKey, config2.LLM.Default.APIKey)
	assert.Equal(t, config.LLM.GlobalSettings.ConcurrencyLevel, config2.LLM.GlobalSettings.ConcurrencyLevel)
	assert.Equal(t, config.Execution.MaxConcurrency, config2.Execution.MaxConcurrency)
}
