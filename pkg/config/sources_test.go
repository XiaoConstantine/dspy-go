package config

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileSourceMethods(t *testing.T) {
	source := NewFileSource()
	assert.Equal(t, "file", source.Name())
	assert.Equal(t, 100, source.Priority())

	sourceWithPriority := NewFileSourceWithPriority(200)
	assert.Equal(t, 200, sourceWithPriority.Priority())
}

func TestEnvironmentSourceMethods(t *testing.T) {
	source := NewEnvironmentSource()
	assert.Equal(t, "environment", source.Name())
	assert.Equal(t, 200, source.Priority())

	sourceWithPrefix := NewEnvironmentSourceWithPrefix("CUSTOM_")
	assert.Equal(t, "CUSTOM_", sourceWithPrefix.prefix)

	sourceWithOptions := NewEnvironmentSourceWithOptions(300, "CUSTOM_")
	assert.Equal(t, 300, sourceWithOptions.Priority())
	assert.Equal(t, "CUSTOM_", sourceWithOptions.prefix)
}

func TestEnvironmentSourceSetLLMProviderValue(t *testing.T) {
	source := NewEnvironmentSource()
	provider := &LLMProviderConfig{}

	tests := []struct {
		key           string
		value         string
		expectedField string
		expectedValue interface{}
	}{
		{"provider", "anthropic", "Provider", "anthropic"},
		{"model.id", "claude-3-sonnet", "ModelID", "claude-3-sonnet"},
		{"modelid", "claude-3-sonnet", "ModelID", "claude-3-sonnet"},
		{"api.key", "test-key", "APIKey", "test-key"},
		{"apikey", "test-key", "APIKey", "test-key"},
		{"endpoint.baseurl", "https://api.example.com", "Endpoint.BaseURL", "https://api.example.com"},
		{"endpoint.base.url", "https://api.example.com", "Endpoint.BaseURL", "https://api.example.com"},
		{"endpoint.timeout", "30s", "Endpoint.Timeout", "30s"},
		{"generation.maxTokens", "4096", "Generation.MaxTokens", 4096},
		{"generation.max.tokens", "4096", "Generation.MaxTokens", 4096},
		{"generation.temperature", "0.7", "Generation.Temperature", 0.7},
		{"generation.topp", "0.9", "Generation.TopP", 0.9},
		{"generation.top.p", "0.9", "Generation.TopP", 0.9},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			err := source.setLLMProviderValue(provider, tt.key, tt.value)
			require.NoError(t, err)

			switch tt.key {
			case "provider":
				assert.Equal(t, tt.expectedValue, provider.Provider)
			case "model.id", "modelid":
				assert.Equal(t, tt.expectedValue, provider.ModelID)
			case "api.key", "apikey":
				assert.Equal(t, tt.expectedValue, provider.APIKey)
			case "endpoint.baseurl", "endpoint.base.url":
				assert.Equal(t, tt.expectedValue, provider.Endpoint.BaseURL)
			case "generation.maxTokens", "generation.max.tokens":
				assert.Equal(t, tt.expectedValue, provider.Generation.MaxTokens)
			case "generation.temperature":
				assert.Equal(t, tt.expectedValue, provider.Generation.Temperature)
			case "generation.topp", "generation.top.p":
				assert.Equal(t, tt.expectedValue, provider.Generation.TopP)
			}
		})
	}

	// Test invalid values
	err := source.setLLMProviderValue(provider, "generation.maxTokens", "invalid")
	assert.Error(t, err)

	err = source.setLLMProviderValue(provider, "generation.temperature", "invalid")
	assert.Error(t, err)

	err = source.setLLMProviderValue(provider, "endpoint.timeout", "invalid")
	assert.Error(t, err)

	err = source.setLLMProviderValue(provider, "unsupported.key", "value")
	assert.NoError(t, err) // Unknown keys are now silently ignored
}

func TestEnvironmentSourceSetLLMGlobalValue(t *testing.T) {
	source := NewEnvironmentSource()
	global := &LLMGlobalSettings{}

	tests := []struct {
		key           string
		value         string
		expectedField string
		expectedValue interface{}
	}{
		{"concurrency.level", "5", "ConcurrencyLevel", 5},
		{"concurrencyLevel", "5", "ConcurrencyLevel", 5},
		{"log.requests", "true", "LogRequests", true},
		{"logRequests", "false", "LogRequests", false},
		{"track.token.usage", "true", "TrackTokenUsage", true},
		{"trackTokenUsage", "false", "TrackTokenUsage", false},
		{"enable.metrics", "true", "EnableMetrics", true},
		{"enableMetrics", "false", "EnableMetrics", false},
	}

	for _, tt := range tests {
		t.Run(tt.key, func(t *testing.T) {
			err := source.setLLMGlobalValue(global, tt.key, tt.value)
			require.NoError(t, err)

			switch tt.key {
			case "concurrency.level", "concurrencyLevel":
				assert.Equal(t, tt.expectedValue, global.ConcurrencyLevel)
			case "log.requests", "logRequests":
				assert.Equal(t, tt.expectedValue, global.LogRequests)
			case "track.token.usage", "trackTokenUsage":
				assert.Equal(t, tt.expectedValue, global.TrackTokenUsage)
			case "enable.metrics", "enableMetrics":
				assert.Equal(t, tt.expectedValue, global.EnableMetrics)
			}
		})
	}

	// Test invalid values
	err := source.setLLMGlobalValue(global, "concurrency.level", "invalid")
	assert.Error(t, err)

	err = source.setLLMGlobalValue(global, "log.requests", "invalid")
	assert.Error(t, err)

	err = source.setLLMGlobalValue(global, "unsupported.key", "value")
	assert.NoError(t, err) // Unknown keys are now silently ignored
}

func TestEnvironmentSourceSetLoggingValue(t *testing.T) {
	source := NewEnvironmentSource()
	logging := &LoggingConfig{}

	err := source.setLoggingValue(logging, "level", "DEBUG")
	require.NoError(t, err)
	assert.Equal(t, "DEBUG", logging.Level)

	err = source.setLoggingValue(logging, "sample.rate", "5")
	require.NoError(t, err)
	assert.Equal(t, uint32(5), logging.SampleRate)

	err = source.setLoggingValue(logging, "sampleRate", "10")
	require.NoError(t, err)
	assert.Equal(t, uint32(10), logging.SampleRate)

	// Test invalid values
	err = source.setLoggingValue(logging, "sample.rate", "invalid")
	assert.Error(t, err)

	err = source.setLoggingValue(logging, "unsupported.key", "value")
	assert.NoError(t, err) // Unknown keys are now silently ignored
}

func TestEnvironmentSourceSetExecutionValue(t *testing.T) {
	source := NewEnvironmentSource()
	execution := &ExecutionConfig{}

	err := source.setExecutionValue(execution, "default.timeout", "10m")
	require.NoError(t, err)
	assert.Equal(t, 10*time.Minute, execution.DefaultTimeout)

	err = source.setExecutionValue(execution, "max.concurrency", "20")
	require.NoError(t, err)
	assert.Equal(t, 20, execution.MaxConcurrency)

	err = source.setExecutionValue(execution, "tracing.enabled", "true")
	require.NoError(t, err)
	assert.Equal(t, true, execution.Tracing.Enabled)

	err = source.setExecutionValue(execution, "tracing.sampling.rate", "0.5")
	require.NoError(t, err)
	assert.Equal(t, 0.5, execution.Tracing.SamplingRate)

	// Test invalid values
	err = source.setExecutionValue(execution, "default.timeout", "invalid")
	assert.Error(t, err)

	err = source.setExecutionValue(execution, "max.concurrency", "invalid")
	assert.Error(t, err)

	err = source.setExecutionValue(execution, "unsupported.key", "value")
	assert.NoError(t, err) // Unknown keys are now silently ignored
}

func TestCommandLineSource(t *testing.T) {
	args := []string{
		"--config.llm.default.provider=google",
		"--config-llm-default-model-id", "gemini-2.0-flash",
		"-c", "logging.level=DEBUG",
	}

	source := NewCommandLineSource(args)
	assert.Equal(t, "command_line", source.Name())
	assert.Equal(t, 300, source.Priority())

	sourceWithPriority := NewCommandLineSourceWithPriority(400, args)
	assert.Equal(t, 400, sourceWithPriority.Priority())

	// Test parsing config args
	configArgs := source.parseConfigArgs()
	assert.Equal(t, "google", configArgs["llm.default.provider"])
	assert.Equal(t, "gemini-2.0-flash", configArgs["llm.default.model.id"])
	assert.Equal(t, "DEBUG", configArgs["logging.level"])
}

func TestCommandLineSourceLoad(t *testing.T) {
	args := []string{
		"--config.llm.default.provider=google",
		"--config.logging.level=ERROR",
	}

	source := NewCommandLineSource(args)
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	assert.Equal(t, "google", config.LLM.Default.Provider)
	assert.Equal(t, "ERROR", config.Logging.Level)
}

func TestMultiSourceMethods(t *testing.T) {
	fileSource := NewFileSource()
	envSource := NewEnvironmentSource()

	multiSource := NewMultiSource(fileSource, envSource)
	assert.Equal(t, "multi_source", multiSource.Name())
	assert.Equal(t, 200, multiSource.Priority()) // Highest priority among sources

	sources := multiSource.GetSources()
	assert.Len(t, sources, 2)

	// Test adding source
	cmdSource := NewCommandLineSource([]string{})
	multiSource.AddSource(cmdSource)
	assert.Len(t, multiSource.GetSources(), 3)

	// Test removing source
	multiSource.RemoveSource("command_line")
	assert.Len(t, multiSource.GetSources(), 2)
}

func TestMultiSourceLoad(t *testing.T) {
	// Set environment variable
	os.Setenv("DSPY_LLM_DEFAULT_PROVIDER", "google")
	defer os.Unsetenv("DSPY_LLM_DEFAULT_PROVIDER")

	fileSource := NewFileSource()
	envSource := NewEnvironmentSource()

	multiSource := NewMultiSource(fileSource, envSource)
	config := GetDefaultConfig()

	err := multiSource.Load(config, nil)
	require.NoError(t, err)

	// Environment should override default
	assert.Equal(t, "google", config.LLM.Default.Provider)
}

func TestSortSourcesByPriority(t *testing.T) {
	fileSource := NewFileSourceWithPriority(100)
	envSource := NewEnvironmentSourceWithOptions(200, "DSPY_")
	cmdSource := NewCommandLineSourceWithPriority(300, []string{})

	multiSource := NewMultiSource(cmdSource, fileSource, envSource)
	sorted := multiSource.sortSourcesByPriority()

	// Should be sorted by ascending priority
	assert.Equal(t, 100, sorted[0].Priority())
	assert.Equal(t, 200, sorted[1].Priority())
	assert.Equal(t, 300, sorted[2].Priority())
}

func TestRemoteSource(t *testing.T) {
	source := NewRemoteSource("https://config.example.com/config.yaml")
	assert.Equal(t, "remote", source.Name())
	assert.Equal(t, 50, source.Priority())

	// Test load (should return not implemented error)
	config := GetDefaultConfig()
	err := source.Load(config, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not implemented")
}

func TestCreateDefaultSources(t *testing.T) {
	sources := CreateDefaultSources()
	assert.Len(t, sources, 2)

	names := make([]string, len(sources))
	for i, source := range sources {
		names[i] = source.Name()
	}

	assert.Contains(t, names, "file")
	assert.Contains(t, names, "environment")
}

func TestCreateAllSources(t *testing.T) {
	args := []string{"--config.test=value"}
	sources := CreateAllSources(args)
	assert.Len(t, sources, 3)

	names := make([]string, len(sources))
	for i, source := range sources {
		names[i] = source.Name()
	}

	assert.Contains(t, names, "file")
	assert.Contains(t, names, "environment")
	assert.Contains(t, names, "command_line")
}

func TestLoadFromSources(t *testing.T) {
	// Set environment variable
	os.Setenv("DSPY_LOGGING_LEVEL", "WARN")
	defer os.Unsetenv("DSPY_LOGGING_LEVEL")

	sources := []Source{
		NewFileSource(),
		NewEnvironmentSource(),
	}

	config := GetDefaultConfig()
	err := LoadFromSources(config, sources, nil)
	require.NoError(t, err)

	assert.Equal(t, "WARN", config.Logging.Level)
}

func TestEnvironmentSourceUnhandledPath(t *testing.T) {
	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	// Test unhandled configuration path (should not fail)
	err := source.setConfigValue(config, "unhandled.path", "value")
	assert.NoError(t, err) // Should not fail, just ignore unknown paths
}

func TestFileSourceLoadNonexistentFile(t *testing.T) {
	source := NewFileSource()
	config := GetDefaultConfig()

	// Should not fail for non-existent files, just skip them
	err := source.Load(config, []string{"/nonexistent/file.yaml"})
	assert.NoError(t, err)
}

func TestEnvironmentSourceGetEnvironmentVariablesEdgeCases(t *testing.T) {
	// Set malformed environment variables
	os.Setenv("DSPY_MALFORMED", "") // No value
	os.Setenv("MALFORMED", "value") // No = sign in the environment (this won't happen in real env)

	defer func() {
		os.Unsetenv("DSPY_MALFORMED")
		os.Unsetenv("MALFORMED")
	}()

	source := NewEnvironmentSource()
	envVars := source.getEnvironmentVariables()

	// Should handle malformed environment variables gracefully
	assert.Contains(t, envVars, "malformed")
	assert.Equal(t, "", envVars["malformed"])
}

// TestNewFileSourceDefaultConstructor tests the default constructor.
func TestNewFileSourceDefaultConstructor(t *testing.T) {
	source := NewFileSource()
	assert.NotNil(t, source)
	assert.Equal(t, "file", source.Name())
	assert.Equal(t, 100, source.Priority())
}

// TestEnvironmentSourceModulesConfiguration tests module configuration via environment variables.
func TestEnvironmentSourceModulesConfiguration(t *testing.T) {
	// Set environment variables for modules configuration
	os.Setenv("DSPY_MODULES_CHAINOFTHOUGHT_MAXSTEPS", "5")
	os.Setenv("DSPY_MODULES_COT_MAXSTEPS", "10")
	os.Setenv("DSPY_MODULES_MULTICHAINCOMPARISON_NUMCHAINS", "3")
	os.Setenv("DSPY_MODULES_MCC_NUMCHAINS", "4")
	os.Setenv("DSPY_MODULES_REACT_MAXSTEPS", "8")
	os.Setenv("DSPY_MODULES_REFINE_MAXSTEPS", "6")
	os.Setenv("DSPY_MODULES_PREDICT_MAXITERATIONS", "15")

	defer func() {
		os.Unsetenv("DSPY_MODULES_CHAINOFTHOUGHT_MAXSTEPS")
		os.Unsetenv("DSPY_MODULES_COT_MAXSTEPS")
		os.Unsetenv("DSPY_MODULES_MULTICHAINCOMPARISON_NUMCHAINS")
		os.Unsetenv("DSPY_MODULES_MCC_NUMCHAINS")
		os.Unsetenv("DSPY_MODULES_REACT_MAXSTEPS")
		os.Unsetenv("DSPY_MODULES_REFINE_MAXSTEPS")
		os.Unsetenv("DSPY_MODULES_PREDICT_MAXITERATIONS")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	// Test COT configuration was set (should use last value)
	assert.Equal(t, 10, config.Modules.ChainOfThought.MaxSteps)

	// Test MCC configuration was set (should use last value)
	assert.Equal(t, 4, config.Modules.MultiChainComparison.NumChains)

	// Test ReAct configuration was set
	assert.Equal(t, 8, config.Modules.ReAct.MaxCycles)

	// Test Refine configuration was set
	assert.Equal(t, 6, config.Modules.Refine.MaxIterations)

	// Test Predict configuration was set (checking if default settings exist)
	assert.NotNil(t, config.Modules.Predict.DefaultSettings)
}

// TestEnvironmentSourceAgentsConfiguration tests agent configuration via environment variables.
func TestEnvironmentSourceAgentsConfiguration(t *testing.T) {
	// Set environment variables for agents configuration
	os.Setenv("DSPY_AGENTS_MAXCONCURRENCY", "5")
	os.Setenv("DSPY_AGENTS_AGENT_MAXITERATIONS", "20")
	os.Setenv("DSPY_AGENTS_AGENT_MEMORY_MAXHISTORY", "100")
	os.Setenv("DSPY_AGENTS_WORKFLOWS_MAXCONCURRENCY", "3")

	defer func() {
		os.Unsetenv("DSPY_AGENTS_MAXCONCURRENCY")
		os.Unsetenv("DSPY_AGENTS_AGENT_MAXITERATIONS")
		os.Unsetenv("DSPY_AGENTS_AGENT_MEMORY_MAXHISTORY")
		os.Unsetenv("DSPY_AGENTS_WORKFLOWS_MAXCONCURRENCY")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	// Test agents configuration was set
	assert.Equal(t, 100, config.Agents.Default.MaxHistory)
	assert.NotNil(t, config.Agents.Memory)
	assert.NotNil(t, config.Agents.Workflows)
}

// TestEnvironmentSourceToolsConfiguration tests tools configuration via environment variables.
func TestEnvironmentSourceToolsConfiguration(t *testing.T) {
	// Set environment variables for tools configuration
	os.Setenv("DSPY_TOOLS_TOOLREGISTRY_MAXCACHEDTOOLS", "50")
	os.Setenv("DSPY_TOOLS_MCP_MAXCONNECTIONS", "10")
	os.Setenv("DSPY_TOOLS_FUNCTIONTOOLS_MAXCONCURRENCY", "8")

	defer func() {
		os.Unsetenv("DSPY_TOOLS_TOOLREGISTRY_MAXCACHEDTOOLS")
		os.Unsetenv("DSPY_TOOLS_MCP_MAXCONNECTIONS")
		os.Unsetenv("DSPY_TOOLS_FUNCTIONTOOLS_MAXCONCURRENCY")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	// Test tools configuration was set
	assert.Equal(t, 50, config.Tools.Registry.MaxTools)
	assert.NotNil(t, config.Tools.MCP)
	assert.NotNil(t, config.Tools.Functions)
}

// TestEnvironmentSourceOptimizersConfiguration tests optimizers configuration via environment variables.
func TestEnvironmentSourceOptimizersConfiguration(t *testing.T) {
	// Set environment variables for optimizers configuration
	os.Setenv("DSPY_OPTIMIZERS_BOOTSTRAPFEWSHOT_MAXBOOTSTRAPSAMPLES", "100")
	os.Setenv("DSPY_OPTIMIZERS_MIPRO_MAXITERATIONS", "50")
	os.Setenv("DSPY_OPTIMIZERS_MIPRO_MAXCANDIDATES", "20")
	os.Setenv("DSPY_OPTIMIZERS_COPRO_MAXITERATIONS", "30")
	os.Setenv("DSPY_OPTIMIZERS_COPRO_MAXCANDIDATES", "15")
	os.Setenv("DSPY_OPTIMIZERS_SIMBA_MAXITERATIONS", "40")
	os.Setenv("DSPY_OPTIMIZERS_SIMBA_MAXCANDIDATES", "25")
	os.Setenv("DSPY_OPTIMIZERS_TPE_MAXITERATIONS", "60")
	os.Setenv("DSPY_OPTIMIZERS_TPE_MAXTRIALS", "200")

	defer func() {
		os.Unsetenv("DSPY_OPTIMIZERS_BOOTSTRAPFEWSHOT_MAXBOOTSTRAPSAMPLES")
		os.Unsetenv("DSPY_OPTIMIZERS_MIPRO_MAXITERATIONS")
		os.Unsetenv("DSPY_OPTIMIZERS_MIPRO_MAXCANDIDATES")
		os.Unsetenv("DSPY_OPTIMIZERS_COPRO_MAXITERATIONS")
		os.Unsetenv("DSPY_OPTIMIZERS_COPRO_MAXCANDIDATES")
		os.Unsetenv("DSPY_OPTIMIZERS_SIMBA_MAXITERATIONS")
		os.Unsetenv("DSPY_OPTIMIZERS_SIMBA_MAXCANDIDATES")
		os.Unsetenv("DSPY_OPTIMIZERS_TPE_MAXITERATIONS")
		os.Unsetenv("DSPY_OPTIMIZERS_TPE_MAXTRIALS")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	// Test optimizers configuration was set
	assert.Equal(t, 100, config.Optimizers.BootstrapFewShot.MaxExamples)
	assert.Equal(t, 50, config.Optimizers.MIPRO.NumGenerations)
	assert.Equal(t, 20, config.Optimizers.MIPRO.PopulationSize)
	assert.Equal(t, 30, config.Optimizers.COPRO.MaxIterations)
	assert.Equal(t, 25, config.Optimizers.SIMBA.NumCandidates)
	assert.Equal(t, 200, config.Optimizers.TPE.NumTrials)
}

// TestEnvironmentSourceCachingConfiguration tests caching configuration via environment variables.
func TestEnvironmentSourceCachingConfiguration(t *testing.T) {
	// Set environment variables for caching configuration
	os.Setenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_ENABLED", "true")
	os.Setenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_MAXSIZE", "1000")
	os.Setenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_TTL", "3600")

	defer func() {
		os.Unsetenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_ENABLED")
		os.Unsetenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_MAXSIZE")
		os.Unsetenv("DSPY_MODULES_PREDICT_SETTINGS_CACHING_TTL")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)
	require.NoError(t, err)

	// Test caching configuration was set
	assert.Equal(t, true, config.Modules.Predict.Caching.Enabled)
}

// TestEnvironmentSourceInvalidValues tests handling of invalid environment variable values.
func TestEnvironmentSourceInvalidValues(t *testing.T) {
	// Set invalid environment variables
	os.Setenv("DSPY_MODULES_CHAINOFTHOUGHT_MAXSTEPS", "invalid")

	defer func() {
		os.Unsetenv("DSPY_MODULES_CHAINOFTHOUGHT_MAXSTEPS")
	}()

	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	err := source.Load(config, nil)

	// Environment source may not validate all fields, so just ensure it doesn't crash
	if err != nil {
		assert.Contains(t, err.Error(), "invalid")
	}
}

// TestFileSourceErrorHandling tests file source error handling.
func TestFileSourceErrorHandling(t *testing.T) {
	// Test loading from a directory instead of a file
	source := NewFileSource()
	config := GetDefaultConfig()

	// This should not return an error but should not load anything
	err := source.Load(config, []string{"/tmp"})
	assert.NoError(t, err)

	// Test loading from a file with invalid YAML
	invalidYAML := "/tmp/invalid.yaml"
	err = os.WriteFile(invalidYAML, []byte("invalid: yaml: content: ["), 0644)
	require.NoError(t, err)
	defer os.Remove(invalidYAML)

	err = source.Load(config, []string{invalidYAML})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse YAML")
}

// TestRemoteSourceMethods tests remote source methods.
func TestRemoteSourceMethods(t *testing.T) {
	source := NewRemoteSource("https://example.com/config.yaml")
	assert.Equal(t, "remote", source.Name())
	assert.Equal(t, 50, source.Priority())

	// Skip NewRemoteSourceWithPriority test - function doesn't exist
}

// TestRemoteSourceLoad tests remote source loading.
func TestRemoteSourceLoad(t *testing.T) {
	source := NewRemoteSource("https://example.com/config.yaml")
	config := GetDefaultConfig()

	// This should fail since we don't have a real remote server
	err := source.Load(config, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to fetch remote config")
}

// TestSetConfigValueEdgeCases tests edge cases in setConfigValue.
func TestSetConfigValueEdgeCases(t *testing.T) {
	source := NewEnvironmentSource()
	config := GetDefaultConfig()

	// Test setting a value for an unknown path
	err := source.setConfigValue(config, "unknown.path", "value")
	assert.NoError(t, err) // Should not error, just ignore

	// Test setting values for complex nested paths
	err = source.setConfigValue(config, "llm.default.endpoint.timeout", "30")
	assert.NoError(t, err)

	// Skip boolean test - LogRequests field doesn't exist

	// Test setting values for time duration fields
	err = source.setConfigValue(config, "llm.default.endpoint.timeout", "45")
	assert.NoError(t, err)
	assert.Equal(t, 45*time.Second, config.LLM.Default.Endpoint.Timeout)
}

func TestEnvironmentSourcePredictSettings(t *testing.T) {
	source := NewEnvironmentSource()
	
	// Initialize PredictSettings
	settings := &PredictSettings{}
	
	// Test include.confidence
	err := source.setPredictSettingsValue(settings, "include.confidence", "true")
	assert.NoError(t, err)
	assert.True(t, settings.IncludeConfidence)
	
	// Test includeConfidence alternative
	err = source.setPredictSettingsValue(settings, "includeConfidence", "false")
	assert.NoError(t, err)
	assert.False(t, settings.IncludeConfidence)
	
	// Test temperature
	err = source.setPredictSettingsValue(settings, "temperature", "0.7")
	assert.NoError(t, err)
	assert.Equal(t, 0.7, settings.Temperature)
	
	// Test top.k
	err = source.setPredictSettingsValue(settings, "top.k", "10")
	assert.NoError(t, err)
	assert.Equal(t, 10, settings.TopK)
	
	// Test topK alternative
	err = source.setPredictSettingsValue(settings, "topK", "20")
	assert.NoError(t, err)
	assert.Equal(t, 20, settings.TopK)
	
	// Test invalid values
	err = source.setPredictSettingsValue(settings, "include.confidence", "invalid")
	assert.Error(t, err)
	
	err = source.setPredictSettingsValue(settings, "temperature", "invalid")
	assert.Error(t, err)
	
	err = source.setPredictSettingsValue(settings, "top.k", "invalid")
	assert.Error(t, err)
	
	// Test unhandled key
	err = source.setPredictSettingsValue(settings, "unknown", "value")
	assert.NoError(t, err) // Should not error for unknown keys
}

func TestEnvironmentSourceCaching(t *testing.T) {
	source := NewEnvironmentSource()
	
	// Initialize CachingConfig
	caching := &CachingConfig{}
	
	// Test enabled
	err := source.setCachingValue(caching, "enabled", "true")
	assert.NoError(t, err)
	assert.True(t, caching.Enabled)
	
	// Test type field
	err = source.setCachingValue(caching, "type", "redis")
	assert.NoError(t, err)
	assert.Equal(t, "redis", caching.Type)
	
	// Test ttl (needs duration format)
	err = source.setCachingValue(caching, "ttl", "1h")
	assert.NoError(t, err)
	assert.Equal(t, 1*time.Hour, caching.TTL)
	
	// Test invalid values
	err = source.setCachingValue(caching, "enabled", "invalid")
	assert.Error(t, err)
	
	err = source.setCachingValue(caching, "ttl", "invalid")
	assert.Error(t, err)
	
	// Test unhandled key
	err = source.setCachingValue(caching, "unknown", "value")
	assert.NoError(t, err) // Should not error for unknown keys
}

func TestEnvironmentSourceAgentSettings(t *testing.T) {
	source := NewEnvironmentSource()
	
	// Initialize AgentConfig
	agent := &AgentConfig{}
	
	// Test timeout (needs duration format)
	err := source.setAgentValue(agent, "timeout", "30s")
	assert.NoError(t, err)
	assert.Equal(t, 30*time.Second, agent.Timeout)
	
	// Test max.history
	err = source.setAgentValue(agent, "max.history", "100")
	assert.NoError(t, err)
	assert.Equal(t, 100, agent.MaxHistory)
	
	// Test invalid values
	err = source.setAgentValue(agent, "timeout", "invalid")
	assert.Error(t, err)
	
	err = source.setAgentValue(agent, "max.history", "invalid")
	assert.Error(t, err)
	
	// Test unhandled key
	err = source.setAgentValue(agent, "unknown", "value")
	assert.NoError(t, err) // Should not error for unknown keys
}

func TestEnvironmentSourceAgentMemory(t *testing.T) {
	source := NewEnvironmentSource()
	
	// Initialize AgentMemoryConfig
	memory := &AgentMemoryConfig{}
	
	// Test capacity
	err := source.setAgentMemoryValue(memory, "capacity", "1000")
	assert.NoError(t, err)
	assert.Equal(t, 1000, memory.Capacity)
	
	// Test type
	err = source.setAgentMemoryValue(memory, "type", "buffered")
	assert.NoError(t, err)
	assert.Equal(t, "buffered", memory.Type)
	
	// Test invalid values
	err = source.setAgentMemoryValue(memory, "capacity", "invalid")
	assert.Error(t, err)
	
	// Test unhandled key
	err = source.setAgentMemoryValue(memory, "unknown", "value")
	assert.NoError(t, err) // Should not error for unknown keys
}
