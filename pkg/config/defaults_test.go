package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetDefaultLLMProviderConfig(t *testing.T) {
	// Test getting anthropic config
	anthropicConfig := GetDefaultLLMProviderConfig("anthropic")
	require.NotNil(t, anthropicConfig)
	assert.Equal(t, "anthropic", anthropicConfig.Provider)
	assert.Equal(t, "claude-3-sonnet-20240229", anthropicConfig.ModelID)
	assert.Equal(t, "https://api.anthropic.com", anthropicConfig.Endpoint.BaseURL)
	
	// Test getting google config
	googleConfig := GetDefaultLLMProviderConfig("google")
	require.NotNil(t, googleConfig)
	assert.Equal(t, "google", googleConfig.Provider)
	assert.Equal(t, "gemini-2.0-flash", googleConfig.ModelID)
	assert.Equal(t, "https://generativelanguage.googleapis.com", googleConfig.Endpoint.BaseURL)
	
	// Test getting ollama config
	ollamaConfig := GetDefaultLLMProviderConfig("ollama")
	require.NotNil(t, ollamaConfig)
	assert.Equal(t, "ollama", ollamaConfig.Provider)
	assert.Equal(t, "ollama:llama3.2", ollamaConfig.ModelID)
	assert.Equal(t, "http://localhost:11434", ollamaConfig.Endpoint.BaseURL)
	
	// Test getting llamacpp config
	llamacppConfig := GetDefaultLLMProviderConfig("llamacpp")
	require.NotNil(t, llamacppConfig)
	assert.Equal(t, "llamacpp", llamacppConfig.Provider)
	assert.Equal(t, "llamacpp:default", llamacppConfig.ModelID)
	assert.Equal(t, "http://localhost:8080", llamacppConfig.Endpoint.BaseURL)
	
	// Test getting non-existent provider (should fallback to anthropic)
	fallbackConfig := GetDefaultLLMProviderConfig("nonexistent")
	require.NotNil(t, fallbackConfig)
	assert.Equal(t, "anthropic", fallbackConfig.Provider)
}

func TestGetDefaultGenerationConfig(t *testing.T) {
	config := GetDefaultGenerationConfig()
	require.NotNil(t, config)
	
	assert.Equal(t, 8192, config.MaxTokens)
	assert.Equal(t, 0.5, config.Temperature)
	assert.Equal(t, 0.9, config.TopP)
	assert.Equal(t, 0.0, config.PresencePenalty)
	assert.Equal(t, 0.0, config.FrequencyPenalty)
	assert.Empty(t, config.StopSequences)
}

func TestGetDefaultEndpointConfig(t *testing.T) {
	config := GetDefaultEndpointConfig()
	require.NotNil(t, config)
	
	assert.Equal(t, "", config.BaseURL)
	assert.Equal(t, "", config.Path)
	assert.Equal(t, "application/json", config.Headers["Content-Type"])
	assert.Equal(t, 30*time.Second, config.Timeout)
	
	assert.Equal(t, 3, config.Retry.MaxRetries)
	assert.Equal(t, 1*time.Second, config.Retry.InitialBackoff)
	assert.Equal(t, 30*time.Second, config.Retry.MaxBackoff)
	assert.Equal(t, 2.0, config.Retry.BackoffMultiplier)
}

func TestGetDefaultRetryConfig(t *testing.T) {
	config := GetDefaultRetryConfig()
	require.NotNil(t, config)
	
	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialBackoff)
	assert.Equal(t, 30*time.Second, config.MaxBackoff)
	assert.Equal(t, 2.0, config.BackoffMultiplier)
}

func TestGetDefaultEmbeddingConfig(t *testing.T) {
	config := GetDefaultEmbeddingConfig()
	require.NotNil(t, config)
	
	assert.Equal(t, "", config.Model)
	assert.Equal(t, 32, config.BatchSize)
	assert.NotNil(t, config.Params)
	assert.Empty(t, config.Params)
}

func TestMergeWithDefaults(t *testing.T) {
	// Test with nil config
	result := MergeWithDefaults(nil)
	require.NotNil(t, result)
	assert.Equal(t, "anthropic", result.LLM.Default.Provider)
	
	// Test with partial config
	partial := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "google",
				ModelID:  "gemini-2.0-flash",
			},
		},
	}
	
	result = MergeWithDefaults(partial)
	require.NotNil(t, result)
	
	// Should keep the provided values
	assert.Equal(t, "google", result.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", result.LLM.Default.ModelID)
	
	// Should fill in defaults for missing fields
	assert.Equal(t, "INFO", result.Logging.Level)
	assert.Equal(t, 5*time.Minute, result.Execution.DefaultTimeout)
}

func TestMergeWithDefaultsEmptyFields(t *testing.T) {
	// Test with config that has empty/zero values
	partial := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				// Empty provider should trigger default merge
			},
			Teacher: LLMProviderConfig{
				// Empty teacher should trigger default merge
			},
			Providers: nil, // Empty providers should trigger default merge
			GlobalSettings: LLMGlobalSettings{
				ConcurrencyLevel: 0, // Zero value should trigger default merge
			},
		},
		Logging: LoggingConfig{
			Level: "", // Empty level should trigger default merge
		},
		Execution: ExecutionConfig{
			DefaultTimeout: 0, // Zero timeout should trigger default merge
		},
	}
	
	result := MergeWithDefaults(partial)
	require.NotNil(t, result)
	
	// Should use defaults for empty/zero fields
	assert.Equal(t, "anthropic", result.LLM.Default.Provider)
	assert.Empty(t, result.LLM.Teacher.Provider) // Teacher remains empty by default
	assert.NotEmpty(t, result.LLM.Providers)
	assert.Equal(t, 1, result.LLM.GlobalSettings.ConcurrencyLevel)
	assert.Equal(t, "INFO", result.Logging.Level)
	assert.Equal(t, 5*time.Minute, result.Execution.DefaultTimeout)
}

func TestMergeWithDefaultsPreservesNonEmptyValues(t *testing.T) {
	// Test that non-empty values are preserved
	partial := &Config{
		LLM: LLMConfig{
			Default: LLMProviderConfig{
				Provider: "google",
				ModelID:  "gemini-2.0-flash",
			},
			Providers: map[string]LLMProviderConfig{
				"custom": {
					Provider: "custom",
					ModelID:  "custom-model",
				},
			},
			GlobalSettings: LLMGlobalSettings{
				ConcurrencyLevel: 5,
			},
		},
		Logging: LoggingConfig{
			Level: "DEBUG",
		},
		Execution: ExecutionConfig{
			DefaultTimeout: 10 * time.Minute,
		},
	}
	
	result := MergeWithDefaults(partial)
	require.NotNil(t, result)
	
	// Should preserve non-empty values
	assert.Equal(t, "google", result.LLM.Default.Provider)
	assert.Equal(t, "gemini-2.0-flash", result.LLM.Default.ModelID)
	assert.Len(t, result.LLM.Providers, 1)
	assert.Equal(t, "custom", result.LLM.Providers["custom"].Provider)
	assert.Equal(t, 5, result.LLM.GlobalSettings.ConcurrencyLevel)
	assert.Equal(t, "DEBUG", result.Logging.Level)
	assert.Equal(t, 10*time.Minute, result.Execution.DefaultTimeout)
}

func TestValidateDefaults(t *testing.T) {
	err := ValidateDefaults()
	assert.NoError(t, err, "Default configuration should be valid")
}

func TestDefaultConfigProviders(t *testing.T) {
	config := GetDefaultConfig()
	
	// Test that all expected providers are present
	expectedProviders := []string{"anthropic", "google", "ollama", "llamacpp"}
	for _, provider := range expectedProviders {
		providerConfig, exists := config.LLM.Providers[provider]
		assert.True(t, exists, "Provider %s should exist", provider)
		assert.Equal(t, provider, providerConfig.Provider)
		assert.NotEmpty(t, providerConfig.ModelID)
	}
}

func TestDefaultConfigLogging(t *testing.T) {
	config := GetDefaultConfig()
	
	assert.Equal(t, "INFO", config.Logging.Level)
	assert.Equal(t, uint32(1), config.Logging.SampleRate)
	assert.Len(t, config.Logging.Outputs, 1)
	
	output := config.Logging.Outputs[0]
	assert.Equal(t, "console", output.Type)
	assert.Equal(t, "text", output.Format)
	assert.True(t, output.Colors)
	
	assert.NotEmpty(t, config.Logging.DefaultFields)
	assert.Equal(t, "dspy-go", config.Logging.DefaultFields["service"])
}

func TestDefaultConfigExecution(t *testing.T) {
	config := GetDefaultConfig()
	
	assert.Equal(t, 5*time.Minute, config.Execution.DefaultTimeout)
	assert.Equal(t, 10, config.Execution.MaxConcurrency)
	
	assert.Equal(t, 2*time.Minute, config.Execution.Context.DefaultTimeout)
	assert.Equal(t, 1000, config.Execution.Context.BufferSize)
	
	assert.False(t, config.Execution.Tracing.Enabled)
	assert.Equal(t, 0.1, config.Execution.Tracing.SamplingRate)
	assert.Equal(t, "jaeger", config.Execution.Tracing.Exporter.Type)
}

func TestDefaultConfigModules(t *testing.T) {
	config := GetDefaultConfig()
	
	// Chain of Thought
	assert.Equal(t, 10, config.Modules.ChainOfThought.MaxSteps)
	assert.True(t, config.Modules.ChainOfThought.IncludeReasoning)
	assert.Equal(t, "\n---\n", config.Modules.ChainOfThought.StepDelimiter)
	
	// Multi-Chain Comparison
	assert.Equal(t, 3, config.Modules.MultiChainComparison.NumChains)
	assert.Equal(t, "majority_vote", config.Modules.MultiChainComparison.ComparisonStrategy)
	assert.True(t, config.Modules.MultiChainComparison.ParallelExecution)
	
	// ReAct
	assert.Equal(t, 5, config.Modules.ReAct.MaxCycles)
	assert.Equal(t, 30*time.Second, config.Modules.ReAct.ActionTimeout)
	assert.True(t, config.Modules.ReAct.IncludeIntermediateSteps)
	
	// Refine
	assert.Equal(t, 3, config.Modules.Refine.MaxIterations)
	assert.Equal(t, 0.95, config.Modules.Refine.ConvergenceThreshold)
	assert.Equal(t, "iterative_improvement", config.Modules.Refine.RefinementStrategy)
	
	// Predict
	assert.True(t, config.Modules.Predict.DefaultSettings.IncludeConfidence)
	assert.Equal(t, 0.5, config.Modules.Predict.DefaultSettings.Temperature)
	assert.Equal(t, 50, config.Modules.Predict.DefaultSettings.TopK)
	
	assert.True(t, config.Modules.Predict.Caching.Enabled)
	assert.Equal(t, 1*time.Hour, config.Modules.Predict.Caching.TTL)
	assert.Equal(t, 1000, config.Modules.Predict.Caching.MaxSize)
	assert.Equal(t, "memory", config.Modules.Predict.Caching.Type)
}

func TestDefaultConfigAgents(t *testing.T) {
	config := GetDefaultConfig()
	
	// Default agent
	assert.Equal(t, 100, config.Agents.Default.MaxHistory)
	assert.Equal(t, 5*time.Minute, config.Agents.Default.Timeout)
	assert.Equal(t, 10, config.Agents.Default.ToolUse.MaxTools)
	assert.Equal(t, 30*time.Second, config.Agents.Default.ToolUse.Timeout)
	assert.False(t, config.Agents.Default.ToolUse.ParallelExecution)
	
	// Memory
	assert.Equal(t, "buffered", config.Agents.Memory.Type)
	assert.Equal(t, 1000, config.Agents.Memory.Capacity)
	assert.False(t, config.Agents.Memory.Persistence.Enabled)
	assert.Equal(t, "./data/agent_memory", config.Agents.Memory.Persistence.Path)
	assert.Equal(t, 5*time.Minute, config.Agents.Memory.Persistence.SyncInterval)
	
	// Workflows
	assert.Equal(t, 10*time.Minute, config.Agents.Workflows.DefaultTimeout)
	assert.Equal(t, 5, config.Agents.Workflows.MaxParallel)
	assert.False(t, config.Agents.Workflows.Persistence.Enabled)
	assert.Equal(t, "file", config.Agents.Workflows.Persistence.Backend)
}

func TestDefaultConfigTools(t *testing.T) {
	config := GetDefaultConfig()
	
	// Registry
	assert.Equal(t, 100, config.Tools.Registry.MaxTools)
	assert.Contains(t, config.Tools.Registry.DiscoveryPaths, "./tools")
	assert.Contains(t, config.Tools.Registry.DiscoveryPaths, "./plugins")
	assert.True(t, config.Tools.Registry.AutoDiscovery)
	
	// MCP
	assert.Empty(t, config.Tools.MCP.Servers)
	assert.Equal(t, 30*time.Second, config.Tools.MCP.DefaultTimeout)
	assert.Equal(t, 10, config.Tools.MCP.ConnectionPool.MaxConnections)
	assert.Equal(t, 10*time.Second, config.Tools.MCP.ConnectionPool.ConnectionTimeout)
	assert.Equal(t, 5*time.Minute, config.Tools.MCP.ConnectionPool.IdleTimeout)
	
	// Functions
	assert.Equal(t, 30*time.Second, config.Tools.Functions.MaxExecutionTime)
}

func TestDefaultConfigOptimizers(t *testing.T) {
	config := GetDefaultConfig()
	
	// Bootstrap Few-Shot
	assert.Equal(t, 50, config.Optimizers.BootstrapFewShot.MaxExamples)
	assert.Equal(t, "claude-3-opus-20240229", config.Optimizers.BootstrapFewShot.TeacherModel)
	assert.Equal(t, "claude-3-sonnet-20240229", config.Optimizers.BootstrapFewShot.StudentModel)
	assert.Equal(t, 3, config.Optimizers.BootstrapFewShot.BootstrapIterations)
	
	// MIPRO
	assert.Equal(t, 20, config.Optimizers.MIPRO.PopulationSize)
	assert.Equal(t, 10, config.Optimizers.MIPRO.NumGenerations)
	assert.Equal(t, 0.1, config.Optimizers.MIPRO.MutationRate)
	assert.Equal(t, 0.8, config.Optimizers.MIPRO.CrossoverRate)
	
	// COPRO
	assert.Equal(t, 10, config.Optimizers.COPRO.MaxIterations)
	assert.Equal(t, 0.01, config.Optimizers.COPRO.ConvergenceThreshold)
	assert.Equal(t, 0.1, config.Optimizers.COPRO.LearningRate)
	
	// SIMBA
	assert.Equal(t, 10, config.Optimizers.SIMBA.NumCandidates)
	assert.Equal(t, "tournament", config.Optimizers.SIMBA.SelectionStrategy)
	assert.Equal(t, "accuracy", config.Optimizers.SIMBA.EvaluationMetric)
	
	// TPE
	assert.Equal(t, 100, config.Optimizers.TPE.NumTrials)
	assert.Equal(t, 10, config.Optimizers.TPE.NumStartupTrials)
	assert.Equal(t, 0.15, config.Optimizers.TPE.Percentile)
	assert.Equal(t, int64(42), config.Optimizers.TPE.RandomSeed)
}

