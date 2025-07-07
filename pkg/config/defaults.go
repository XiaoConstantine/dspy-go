package config

import (
	"time"
)

// GetDefaultConfig returns the default configuration for dspy-go.
func GetDefaultConfig() *Config {
	return &Config{
		LLM:        getDefaultLLMConfig(),
		Logging:    getDefaultLoggingConfig(),
		Execution:  getDefaultExecutionConfig(),
		Modules:    getDefaultModulesConfig(),
		Agents:     getDefaultAgentsConfig(),
		Tools:      getDefaultToolsConfig(),
		Optimizers: getDefaultOptimizersConfig(),
	}
}

// getDefaultLLMConfig returns default LLM configuration.
func getDefaultLLMConfig() LLMConfig {
	defaultEndpoint := EndpointConfig{
		BaseURL: "https://api.anthropic.com",
		Path:    "/v1/messages",
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		Timeout: 30 * time.Second,
		Retry: RetryConfig{
			MaxRetries:        3,
			InitialBackoff:    1 * time.Second,
			MaxBackoff:        30 * time.Second,
			BackoffMultiplier: 2.0,
		},
	}
	
	defaultGeneration := GenerationConfig{
		MaxTokens:        8192,
		Temperature:      0.5,
		TopP:             0.9,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		StopSequences:    []string{},
	}
	
	return LLMConfig{
		Default: LLMProviderConfig{
			Provider: "anthropic",
			ModelID:  "claude-3-sonnet-20240229",
			APIKey:   "", // Should be provided via environment or config file
			Endpoint: defaultEndpoint,
			Generation: defaultGeneration,
			Embedding: EmbeddingConfig{
				Model:     "text-embedding-ada-002",
				BatchSize: 32,
				Params:    map[string]interface{}{},
			},
			Capabilities: []string{
				"completion",
				"chat",
				"json",
				"streaming",
			},
		},
		Teacher: LLMProviderConfig{}, // Empty - optional
		Providers: getDefaultProviders(), // Populate with default providers
		GlobalSettings: LLMGlobalSettings{
			ConcurrencyLevel: 1,
			LogRequests:      false,
			TrackTokenUsage:  true,
			EnableMetrics:    true,
		},
	}
}

// getDefaultProviders returns default provider configurations.
func getDefaultProviders() map[string]LLMProviderConfig {
	return map[string]LLMProviderConfig{
		"anthropic": {
			Provider: "anthropic",
			ModelID:  "claude-3-sonnet-20240229",
			APIKey:   "",
			Endpoint: EndpointConfig{
				BaseURL: "https://api.anthropic.com",
				Path:    "/v1/messages",
				Headers: map[string]string{
					"Content-Type":      "application/json",
					"anthropic-version": "2023-06-01",
				},
				Timeout: 30 * time.Second,
				Retry: RetryConfig{
					MaxRetries:        3,
					InitialBackoff:    1 * time.Second,
					MaxBackoff:        30 * time.Second,
					BackoffMultiplier: 2.0,
				},
			},
			Generation: GenerationConfig{
				MaxTokens:        8192,
				Temperature:      0.5,
				TopP:             0.9,
				PresencePenalty:  0.0,
				FrequencyPenalty: 0.0,
				StopSequences:    []string{},
			},
			Embedding: EmbeddingConfig{
				Model:     "text-embedding-ada-002",
				BatchSize: 32,
				Params:    map[string]interface{}{},
			},
			Capabilities: []string{
				"completion",
				"chat",
				"json",
				"streaming",
			},
		},
		"google": {
			Provider: "google",
			ModelID:  "gemini-2.0-flash",
			APIKey:   "",
			Endpoint: EndpointConfig{
				BaseURL: "https://generativelanguage.googleapis.com",
				Path:    "/v1beta/models",
				Headers: map[string]string{
					"Content-Type": "application/json",
				},
				Timeout: 30 * time.Second,
				Retry: RetryConfig{
					MaxRetries:        3,
					InitialBackoff:    1 * time.Second,
					MaxBackoff:        30 * time.Second,
					BackoffMultiplier: 2.0,
				},
			},
			Generation: GenerationConfig{
				MaxTokens:        8192,
				Temperature:      0.5,
				TopP:             0.9,
				PresencePenalty:  0.0,
				FrequencyPenalty: 0.0,
				StopSequences:    []string{},
			},
			Embedding: EmbeddingConfig{
				Model:     "text-embedding-004",
				BatchSize: 32,
				Params:    map[string]interface{}{},
			},
			Capabilities: []string{
				"completion",
				"chat",
				"json",
				"streaming",
				"embedding",
			},
		},
		"ollama": {
			Provider: "ollama",
			ModelID:  "ollama:llama3.2",
			APIKey:   "",
			Endpoint: EndpointConfig{
				BaseURL: "http://localhost:11434",
				Path:    "/api/generate",
				Headers: map[string]string{
					"Content-Type": "application/json",
				},
				Timeout: 600 * time.Second, // 10 minutes for local models
				Retry: RetryConfig{
					MaxRetries:        2,
					InitialBackoff:    2 * time.Second,
					MaxBackoff:        10 * time.Second,
					BackoffMultiplier: 2.0,
				},
			},
			Generation: GenerationConfig{
				MaxTokens:        4096,
				Temperature:      0.7,
				TopP:             0.9,
				PresencePenalty:  0.0,
				FrequencyPenalty: 0.0,
				StopSequences:    []string{},
			},
			Embedding: EmbeddingConfig{
				Model:     "nomic-embed-text",
				BatchSize: 16,
				Params:    map[string]interface{}{},
			},
			Capabilities: []string{
				"completion",
				"chat",
				"streaming",
				"embedding",
			},
		},
		"llamacpp": {
			Provider: "llamacpp",
			ModelID:  "llamacpp:default",
			APIKey:   "",
			Endpoint: EndpointConfig{
				BaseURL: "http://localhost:8080",
				Path:    "/completion",
				Headers: map[string]string{
					"Content-Type": "application/json",
				},
				Timeout: 600 * time.Second, // 10 minutes for local models
				Retry: RetryConfig{
					MaxRetries:        2,
					InitialBackoff:    2 * time.Second,
					MaxBackoff:        10 * time.Second,
					BackoffMultiplier: 2.0,
				},
			},
			Generation: GenerationConfig{
				MaxTokens:        4096,
				Temperature:      0.7,
				TopP:             0.9,
				PresencePenalty:  0.0,
				FrequencyPenalty: 0.0,
				StopSequences:    []string{},
			},
			Embedding: EmbeddingConfig{
				Model:     "",
				BatchSize: 16,
				Params:    map[string]interface{}{},
			},
			Capabilities: []string{
				"completion",
				"chat",
				"streaming",
			},
		},
	}
}

// getDefaultLoggingConfig returns default logging configuration.
func getDefaultLoggingConfig() LoggingConfig {
	return LoggingConfig{
		Level:      "INFO",
		SampleRate: 1, // Log all events by default
		Outputs: []LogOutputConfig{
			{
				Type:   "console",
				Format: "text",
				Colors: true,
			},
		},
		DefaultFields: map[string]interface{}{
			"service": "dspy-go",
			"version": "1.0.0",
		},
	}
}

// getDefaultExecutionConfig returns default execution configuration.
func getDefaultExecutionConfig() ExecutionConfig {
	return ExecutionConfig{
		DefaultTimeout: 5 * time.Minute,
		MaxConcurrency: 10,
		Context: ContextConfig{
			DefaultTimeout: 2 * time.Minute,
			BufferSize:     1000,
		},
		Tracing: TracingConfig{
			Enabled:      false,
			SamplingRate: 0.1,
			Exporter: TracingExporterConfig{
				Type:     "jaeger",
				Endpoint: "http://localhost:14268/api/traces",
				Headers:  map[string]string{},
			},
		},
	}
}

// getDefaultModulesConfig returns default modules configuration.
func getDefaultModulesConfig() ModulesConfig {
	return ModulesConfig{
		ChainOfThought: ChainOfThoughtConfig{
			MaxSteps:         10,
			IncludeReasoning: true,
			StepDelimiter:    "\n---\n",
		},
		MultiChainComparison: MultiChainComparisonConfig{
			NumChains:          3,
			ComparisonStrategy: "majority_vote",
			ParallelExecution:  true,
		},
		ReAct: ReActConfig{
			MaxCycles:                5,
			ActionTimeout:            30 * time.Second,
			IncludeIntermediateSteps: true,
		},
		Refine: RefineConfig{
			MaxIterations:        3,
			ConvergenceThreshold: 0.95,
			RefinementStrategy:   "iterative_improvement",
		},
		Predict: PredictConfig{
			DefaultSettings: PredictSettings{
				IncludeConfidence: true,
				Temperature:       0.5,
				TopK:              50,
			},
			Caching: CachingConfig{
				Enabled: true,
				TTL:     1 * time.Hour,
				MaxSize: 1000,
				Type:    "memory",
				Config:  map[string]interface{}{},
			},
		},
	}
}

// getDefaultAgentsConfig returns default agents configuration.
func getDefaultAgentsConfig() AgentsConfig {
	return AgentsConfig{
		Default: AgentConfig{
			MaxHistory: 100,
			Timeout:    5 * time.Minute,
			ToolUse: ToolUseConfig{
				MaxTools:          10,
				Timeout:           30 * time.Second,
				ParallelExecution: false,
			},
		},
		Memory: AgentMemoryConfig{
			Type:     "buffered",
			Capacity: 1000,
			Persistence: MemoryPersistenceConfig{
				Enabled:      false,
				Path:         "./data/agent_memory",
				SyncInterval: 5 * time.Minute,
			},
		},
		Workflows: WorkflowsConfig{
			DefaultTimeout: 10 * time.Minute,
			MaxParallel:    5,
			Persistence: WorkflowPersistenceConfig{
				Enabled: false,
				Backend: "file",
				Config: map[string]interface{}{
					"path": "./data/workflows",
				},
			},
		},
	}
}

// getDefaultToolsConfig returns default tools configuration.
func getDefaultToolsConfig() ToolsConfig {
	return ToolsConfig{
		Registry: ToolRegistryConfig{
			MaxTools:       100,
			DiscoveryPaths: []string{"./tools", "./plugins"},
			AutoDiscovery:  true,
		},
		MCP: MCPConfig{
			Servers:        []MCPServerConfig{},
			DefaultTimeout: 30 * time.Second,
			ConnectionPool: MCPConnectionPoolConfig{
				MaxConnections:    10,
				ConnectionTimeout: 10 * time.Second,
				IdleTimeout:       5 * time.Minute,
			},
		},
		Functions: FunctionToolsConfig{
			MaxExecutionTime: 30 * time.Second,
		},
	}
}

// getDefaultOptimizersConfig returns default optimizers configuration.
func getDefaultOptimizersConfig() OptimizersConfig {
	return OptimizersConfig{
		BootstrapFewShot: BootstrapFewShotConfig{
			MaxExamples:         50,
			TeacherModel:        "claude-3-opus-20240229",
			StudentModel:        "claude-3-sonnet-20240229",
			BootstrapIterations: 3,
		},
		MIPRO: MIPROConfig{
			PopulationSize: 20,
			NumGenerations: 10,
			MutationRate:   0.1,
			CrossoverRate:  0.8,
		},
		COPRO: COPROConfig{
			MaxIterations:        10,
			ConvergenceThreshold: 0.01,
			LearningRate:         0.1,
		},
		SIMBA: SIMBAConfig{
			NumCandidates:     10,
			SelectionStrategy: "tournament",
			EvaluationMetric:  "accuracy",
		},
		TPE: TPEConfig{
			NumTrials:        100,
			NumStartupTrials: 10,
			Percentile:       0.15,
			RandomSeed:       42,
		},
	}
}


// GetDefaultLLMProviderConfig returns a default LLM provider configuration.
func GetDefaultLLMProviderConfig(provider string) *LLMProviderConfig {
	providers := getDefaultProviders()
	if config, exists := providers[provider]; exists {
		return &config
	}

	// Return anthropic as fallback
	anthropic := providers["anthropic"]
	return &anthropic
}

// GetDefaultGenerationConfig returns default generation configuration.
func GetDefaultGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxTokens:        8192,
		Temperature:      0.5,
		TopP:             0.9,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		StopSequences:    []string{},
	}
}

// GetDefaultEndpointConfig returns default endpoint configuration.
func GetDefaultEndpointConfig() *EndpointConfig {
	return &EndpointConfig{
		BaseURL: "",
		Path:    "",
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		Timeout: 30 * time.Second,
		Retry: RetryConfig{
			MaxRetries:        3,
			InitialBackoff:    1 * time.Second,
			MaxBackoff:        30 * time.Second,
			BackoffMultiplier: 2.0,
		},
	}
}

// GetDefaultRetryConfig returns default retry configuration.
func GetDefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:        3,
		InitialBackoff:    1 * time.Second,
		MaxBackoff:        30 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// GetDefaultEmbeddingConfig returns default embedding configuration.
func GetDefaultEmbeddingConfig() *EmbeddingConfig {
	return &EmbeddingConfig{
		Model:     "",
		BatchSize: 32,
		Params:    map[string]interface{}{},
	}
}

// MergeWithDefaults performs a deep merge of a partial configuration with defaults.
func MergeWithDefaults(partial *Config) *Config {
	if partial == nil {
		return GetDefaultConfig()
	}

	defaults := GetDefaultConfig()
	result := *partial // Start with a copy of partial

	// Deep merge LLM configuration
	result.LLM = mergeLLMConfig(partial.LLM, defaults.LLM)

	// Deep merge other sections
	result.Logging = mergeLoggingConfig(partial.Logging, defaults.Logging)
	result.Execution = mergeExecutionConfig(partial.Execution, defaults.Execution)
	result.Modules = mergeModulesConfig(partial.Modules, defaults.Modules)
	result.Agents = mergeAgentsConfig(partial.Agents, defaults.Agents)
	result.Tools = mergeToolsConfig(partial.Tools, defaults.Tools)
	result.Optimizers = mergeOptimizersConfig(partial.Optimizers, defaults.Optimizers)

	return &result
}

// mergeLLMConfig performs deep merge of LLM configuration.
func mergeLLMConfig(partial LLMConfig, defaults LLMConfig) LLMConfig {
	result := partial

	// Merge Default provider config
	result.Default = mergeLLMProviderConfig(partial.Default, defaults.Default)

	// Merge Teacher provider config if empty
	if partial.Teacher.Provider == "" {
		result.Teacher = defaults.Teacher
	} else {
		result.Teacher = mergeLLMProviderConfig(partial.Teacher, defaults.Teacher)
	}

	// Merge providers map - only add defaults if no providers are provided
	if len(partial.Providers) == 0 {
		result.Providers = defaults.Providers
	} else {
		// If user provided providers, only merge the ones they provided
		if result.Providers == nil {
			result.Providers = make(map[string]LLMProviderConfig)
		}
		for name, partialProvider := range partial.Providers {
			if defaultProvider, exists := defaults.Providers[name]; exists {
				result.Providers[name] = mergeLLMProviderConfig(partialProvider, defaultProvider)
			} else {
				result.Providers[name] = partialProvider
			}
		}
	}

	// Merge global settings
	result.GlobalSettings = mergeLLMGlobalSettings(partial.GlobalSettings, defaults.GlobalSettings)

	return result
}

// mergeLLMProviderConfig performs deep merge of LLM provider configuration.
func mergeLLMProviderConfig(partial LLMProviderConfig, defaults LLMProviderConfig) LLMProviderConfig {
	result := partial

	if result.Provider == "" {
		result.Provider = defaults.Provider
	}
	if result.ModelID == "" {
		result.ModelID = defaults.ModelID
	}
	if result.APIKey == "" {
		result.APIKey = defaults.APIKey
	}

	// Merge endpoint config
	result.Endpoint = mergeEndpointConfig(partial.Endpoint, defaults.Endpoint)

	// Merge generation config
	result.Generation = mergeGenerationConfig(partial.Generation, defaults.Generation)

	// Merge embedding config
	result.Embedding = mergeEmbeddingConfig(partial.Embedding, defaults.Embedding)

	// Merge capabilities
	if len(result.Capabilities) == 0 {
		result.Capabilities = defaults.Capabilities
	}

	return result
}

// mergeEndpointConfig performs deep merge of endpoint configuration.
func mergeEndpointConfig(partial EndpointConfig, defaults EndpointConfig) EndpointConfig {
	result := partial

	if result.BaseURL == "" {
		result.BaseURL = defaults.BaseURL
	}
	if result.Path == "" {
		result.Path = defaults.Path
	}
	if result.Headers == nil {
		result.Headers = make(map[string]string)
	}
	for k, v := range defaults.Headers {
		if _, exists := result.Headers[k]; !exists {
			result.Headers[k] = v
		}
	}
	if result.Timeout == 0 {
		result.Timeout = defaults.Timeout
	}

	// Merge retry config
	result.Retry = mergeRetryConfig(partial.Retry, defaults.Retry)

	return result
}

// mergeRetryConfig performs deep merge of retry configuration.
func mergeRetryConfig(partial RetryConfig, defaults RetryConfig) RetryConfig {
	result := partial

	if result.MaxRetries == 0 {
		result.MaxRetries = defaults.MaxRetries
	}
	if result.InitialBackoff == 0 {
		result.InitialBackoff = defaults.InitialBackoff
	}
	if result.MaxBackoff == 0 {
		result.MaxBackoff = defaults.MaxBackoff
	}
	if result.BackoffMultiplier == 0 {
		result.BackoffMultiplier = defaults.BackoffMultiplier
	}

	return result
}

// mergeGenerationConfig performs deep merge of generation configuration.
func mergeGenerationConfig(partial GenerationConfig, defaults GenerationConfig) GenerationConfig {
	result := partial

	if result.MaxTokens == 0 {
		result.MaxTokens = defaults.MaxTokens
	}
	if result.Temperature == 0 {
		result.Temperature = defaults.Temperature
	}
	if result.TopP == 0 {
		result.TopP = defaults.TopP
	}
	if result.PresencePenalty == 0 {
		result.PresencePenalty = defaults.PresencePenalty
	}
	if result.FrequencyPenalty == 0 {
		result.FrequencyPenalty = defaults.FrequencyPenalty
	}
	if len(result.StopSequences) == 0 {
		result.StopSequences = defaults.StopSequences
	}

	return result
}

// mergeEmbeddingConfig performs deep merge of embedding configuration.
func mergeEmbeddingConfig(partial EmbeddingConfig, defaults EmbeddingConfig) EmbeddingConfig {
	result := partial

	if result.Model == "" {
		result.Model = defaults.Model
	}
	if result.BatchSize == 0 {
		result.BatchSize = defaults.BatchSize
	}
	if result.Params == nil {
		result.Params = make(map[string]interface{})
	}
	for k, v := range defaults.Params {
		if _, exists := result.Params[k]; !exists {
			result.Params[k] = v
		}
	}

	return result
}

// mergeLLMGlobalSettings performs deep merge of LLM global settings.
func mergeLLMGlobalSettings(partial LLMGlobalSettings, defaults LLMGlobalSettings) LLMGlobalSettings {
	result := partial

	if result.ConcurrencyLevel == 0 {
		result.ConcurrencyLevel = defaults.ConcurrencyLevel
	}
	// Note: boolean fields keep their explicit values (including false)

	return result
}


// ValidateDefaults validates that the default configuration is valid.
func ValidateDefaults() error {
	defaults := GetDefaultConfig()
	return defaults.Validate()
}

// mergeLoggingConfig performs deep merge of logging configuration.
func mergeLoggingConfig(partial LoggingConfig, defaults LoggingConfig) LoggingConfig {
	result := partial

	// Merge basic fields
	if result.Level == "" {
		result.Level = defaults.Level
	}

	// Merge outputs slice
	if len(result.Outputs) == 0 {
		result.Outputs = defaults.Outputs
	}

	// Merge sample rate
	if result.SampleRate == 0 {
		result.SampleRate = defaults.SampleRate
	}

	// Merge default fields map
	if result.DefaultFields == nil {
		result.DefaultFields = defaults.DefaultFields
	} else {
		for key, value := range defaults.DefaultFields {
			if _, exists := result.DefaultFields[key]; !exists {
				result.DefaultFields[key] = value
			}
		}
	}

	return result
}

// mergeExecutionConfig performs deep merge of execution configuration.
func mergeExecutionConfig(partial ExecutionConfig, defaults ExecutionConfig) ExecutionConfig {
	result := partial

	if result.DefaultTimeout == 0 {
		result.DefaultTimeout = defaults.DefaultTimeout
	}
	if result.MaxConcurrency == 0 {
		result.MaxConcurrency = defaults.MaxConcurrency
	}

	// Merge context configuration
	result.Context = mergeContextConfig(partial.Context, defaults.Context)

	// Merge tracing configuration
	result.Tracing = mergeTracingConfig(partial.Tracing, defaults.Tracing)

	return result
}

// mergeContextConfig performs deep merge of context configuration.
func mergeContextConfig(partial ContextConfig, defaults ContextConfig) ContextConfig {
	result := partial

	if result.DefaultTimeout == 0 {
		result.DefaultTimeout = defaults.DefaultTimeout
	}
	if result.BufferSize == 0 {
		result.BufferSize = defaults.BufferSize
	}

	return result
}

// mergeTracingConfig performs deep merge of tracing configuration.
func mergeTracingConfig(partial TracingConfig, defaults TracingConfig) TracingConfig {
	result := partial

	// Note: Enabled is a boolean, so we don't override false values
	if result.SamplingRate == 0 {
		result.SamplingRate = defaults.SamplingRate
	}

	// Merge exporter configuration
	result.Exporter = mergeTracingExporterConfig(partial.Exporter, defaults.Exporter)

	return result
}

// mergeTracingExporterConfig performs deep merge of tracing exporter configuration.
func mergeTracingExporterConfig(partial TracingExporterConfig, defaults TracingExporterConfig) TracingExporterConfig {
	result := partial

	if result.Type == "" {
		result.Type = defaults.Type
	}
	if result.Endpoint == "" {
		result.Endpoint = defaults.Endpoint
	}

	// Merge headers map
	if result.Headers == nil {
		result.Headers = defaults.Headers
	} else {
		for key, value := range defaults.Headers {
			if _, exists := result.Headers[key]; !exists {
				result.Headers[key] = value
			}
		}
	}

	return result
}

// mergeModulesConfig performs deep merge of modules configuration.
func mergeModulesConfig(partial ModulesConfig, defaults ModulesConfig) ModulesConfig {
	result := partial

	// Merge ChainOfThought
	result.ChainOfThought = mergeChainOfThoughtConfig(partial.ChainOfThought, defaults.ChainOfThought)

	// Merge MultiChainComparison
	result.MultiChainComparison = mergeMultiChainComparisonConfig(partial.MultiChainComparison, defaults.MultiChainComparison)

	// Merge ReAct
	result.ReAct = mergeReActConfig(partial.ReAct, defaults.ReAct)

	// Merge Refine
	result.Refine = mergeRefineConfig(partial.Refine, defaults.Refine)

	// Merge Predict
	result.Predict = mergePredictConfig(partial.Predict, defaults.Predict)

	return result
}

// mergeChainOfThoughtConfig performs deep merge of chain of thought configuration.
func mergeChainOfThoughtConfig(partial ChainOfThoughtConfig, defaults ChainOfThoughtConfig) ChainOfThoughtConfig {
	result := partial

	if result.MaxSteps == 0 {
		result.MaxSteps = defaults.MaxSteps
	}
	// IncludeReasoning is a boolean, so we don't override false values
	if result.StepDelimiter == "" {
		result.StepDelimiter = defaults.StepDelimiter
	}

	return result
}

// mergeMultiChainComparisonConfig performs deep merge of multi-chain comparison configuration.
func mergeMultiChainComparisonConfig(partial MultiChainComparisonConfig, defaults MultiChainComparisonConfig) MultiChainComparisonConfig {
	result := partial

	if result.NumChains == 0 {
		result.NumChains = defaults.NumChains
	}
	if result.ComparisonStrategy == "" {
		result.ComparisonStrategy = defaults.ComparisonStrategy
	}
	// ParallelExecution is a boolean, so we don't override false values

	return result
}

// mergeReActConfig performs deep merge of ReAct configuration.
func mergeReActConfig(partial ReActConfig, defaults ReActConfig) ReActConfig {
	result := partial

	if result.MaxCycles == 0 {
		result.MaxCycles = defaults.MaxCycles
	}
	if result.ActionTimeout == 0 {
		result.ActionTimeout = defaults.ActionTimeout
	}
	// IncludeIntermediateSteps is a boolean, so we don't override false values

	return result
}

// mergeRefineConfig performs deep merge of refine configuration.
func mergeRefineConfig(partial RefineConfig, defaults RefineConfig) RefineConfig {
	result := partial

	if result.MaxIterations == 0 {
		result.MaxIterations = defaults.MaxIterations
	}
	if result.ConvergenceThreshold == 0 {
		result.ConvergenceThreshold = defaults.ConvergenceThreshold
	}
	if result.RefinementStrategy == "" {
		result.RefinementStrategy = defaults.RefinementStrategy
	}

	return result
}

// mergePredictConfig performs deep merge of predict configuration.
func mergePredictConfig(partial PredictConfig, defaults PredictConfig) PredictConfig {
	result := partial

	// Merge default settings
	result.DefaultSettings = mergePredictSettings(partial.DefaultSettings, defaults.DefaultSettings)

	// Merge caching configuration
	result.Caching = mergeCachingConfig(partial.Caching, defaults.Caching)

	return result
}

// mergePredictSettings performs deep merge of predict settings.
func mergePredictSettings(partial PredictSettings, defaults PredictSettings) PredictSettings {
	result := partial

	// IncludeConfidence is a boolean, so we don't override false values
	if result.Temperature == 0 {
		result.Temperature = defaults.Temperature
	}
	if result.TopK == 0 {
		result.TopK = defaults.TopK
	}

	return result
}

// mergeCachingConfig performs deep merge of caching configuration.
func mergeCachingConfig(partial CachingConfig, defaults CachingConfig) CachingConfig {
	result := partial

	// Enabled is a boolean, so we don't override false values
	if result.TTL == 0 {
		result.TTL = defaults.TTL
	}
	if result.MaxSize == 0 {
		result.MaxSize = defaults.MaxSize
	}
	if result.Type == "" {
		result.Type = defaults.Type
	}

	// Merge config map
	if result.Config == nil {
		result.Config = defaults.Config
	} else {
		for key, value := range defaults.Config {
			if _, exists := result.Config[key]; !exists {
				result.Config[key] = value
			}
		}
	}

	return result
}

// mergeAgentsConfig performs deep merge of agents configuration.
func mergeAgentsConfig(partial AgentsConfig, defaults AgentsConfig) AgentsConfig {
	result := partial

	// Merge Default agent settings
	result.Default = mergeAgentConfig(partial.Default, defaults.Default)

	// Merge Memory settings
	result.Memory = mergeAgentMemoryConfig(partial.Memory, defaults.Memory)

	// Merge Workflows settings
	result.Workflows = mergeWorkflowsConfig(partial.Workflows, defaults.Workflows)

	return result
}

// mergeAgentConfig performs deep merge of agent configuration.
func mergeAgentConfig(partial AgentConfig, defaults AgentConfig) AgentConfig {
	result := partial

	if result.MaxHistory == 0 {
		result.MaxHistory = defaults.MaxHistory
	}
	if result.Timeout == 0 {
		result.Timeout = defaults.Timeout
	}

	// Merge tool use configuration
	result.ToolUse = mergeToolUseConfig(partial.ToolUse, defaults.ToolUse)

	return result
}

// mergeAgentMemoryConfig performs deep merge of agent memory configuration.
func mergeAgentMemoryConfig(partial AgentMemoryConfig, defaults AgentMemoryConfig) AgentMemoryConfig {
	result := partial

	if result.Type == "" {
		result.Type = defaults.Type
	}
	if result.Capacity == 0 {
		result.Capacity = defaults.Capacity
	}

	// Merge persistence configuration
	result.Persistence = mergeMemoryPersistenceConfig(partial.Persistence, defaults.Persistence)

	return result
}

// mergeWorkflowsConfig performs deep merge of workflows configuration.
func mergeWorkflowsConfig(partial WorkflowsConfig, defaults WorkflowsConfig) WorkflowsConfig {
	result := partial

	if result.DefaultTimeout == 0 {
		result.DefaultTimeout = defaults.DefaultTimeout
	}
	if result.MaxParallel == 0 {
		result.MaxParallel = defaults.MaxParallel
	}

	// Merge persistence configuration
	result.Persistence = mergeWorkflowPersistenceConfig(partial.Persistence, defaults.Persistence)

	return result
}

// mergeToolUseConfig performs deep merge of tool use configuration.
func mergeToolUseConfig(partial ToolUseConfig, defaults ToolUseConfig) ToolUseConfig {
	result := partial

	if result.MaxTools == 0 {
		result.MaxTools = defaults.MaxTools
	}
	if result.Timeout == 0 {
		result.Timeout = defaults.Timeout
	}
	// ParallelExecution is a boolean, so we don't override false values

	return result
}

// mergeMemoryPersistenceConfig performs deep merge of memory persistence configuration.
func mergeMemoryPersistenceConfig(partial MemoryPersistenceConfig, defaults MemoryPersistenceConfig) MemoryPersistenceConfig {
	result := partial

	// Enabled is a boolean, so we don't override false values
	if result.Path == "" {
		result.Path = defaults.Path
	}
	if result.SyncInterval == 0 {
		result.SyncInterval = defaults.SyncInterval
	}

	return result
}

// mergeWorkflowPersistenceConfig performs deep merge of workflow persistence configuration.
func mergeWorkflowPersistenceConfig(partial WorkflowPersistenceConfig, defaults WorkflowPersistenceConfig) WorkflowPersistenceConfig {
	result := partial

	// Enabled is a boolean, so we don't override false values
	if result.Backend == "" {
		result.Backend = defaults.Backend
	}

	// Merge config map
	if result.Config == nil {
		result.Config = defaults.Config
	} else {
		for key, value := range defaults.Config {
			if _, exists := result.Config[key]; !exists {
				result.Config[key] = value
			}
		}
	}

	return result
}

// mergeToolsConfig performs deep merge of tools configuration.
func mergeToolsConfig(partial ToolsConfig, defaults ToolsConfig) ToolsConfig {
	result := partial

	// Merge Registry
	result.Registry = mergeToolRegistryConfig(partial.Registry, defaults.Registry)

	// Merge MCP
	result.MCP = mergeMCPConfig(partial.MCP, defaults.MCP)

	// Merge Functions
	result.Functions = mergeFunctionToolsConfig(partial.Functions, defaults.Functions)

	return result
}

// mergeToolRegistryConfig performs deep merge of tool registry configuration.
func mergeToolRegistryConfig(partial ToolRegistryConfig, defaults ToolRegistryConfig) ToolRegistryConfig {
	result := partial

	if result.MaxTools == 0 {
		result.MaxTools = defaults.MaxTools
	}

	// Merge discovery paths slice
	if len(result.DiscoveryPaths) == 0 {
		result.DiscoveryPaths = defaults.DiscoveryPaths
	}

	// AutoDiscovery is a boolean, so we don't override false values

	return result
}

// mergeMCPConfig performs deep merge of MCP configuration.
func mergeMCPConfig(partial MCPConfig, defaults MCPConfig) MCPConfig {
	result := partial

	// Merge servers slice
	if len(result.Servers) == 0 {
		result.Servers = defaults.Servers
	}

	if result.DefaultTimeout == 0 {
		result.DefaultTimeout = defaults.DefaultTimeout
	}

	// Merge connection pool configuration
	result.ConnectionPool = mergeMCPConnectionPoolConfig(partial.ConnectionPool, defaults.ConnectionPool)

	return result
}

// mergeFunctionToolsConfig performs deep merge of function tools configuration.
func mergeFunctionToolsConfig(partial FunctionToolsConfig, defaults FunctionToolsConfig) FunctionToolsConfig {
	result := partial

	if result.MaxExecutionTime == 0 {
		result.MaxExecutionTime = defaults.MaxExecutionTime
	}

	return result
}

// mergeMCPConnectionPoolConfig performs deep merge of MCP connection pool configuration.
func mergeMCPConnectionPoolConfig(partial MCPConnectionPoolConfig, defaults MCPConnectionPoolConfig) MCPConnectionPoolConfig {
	result := partial

	if result.MaxConnections == 0 {
		result.MaxConnections = defaults.MaxConnections
	}
	if result.ConnectionTimeout == 0 {
		result.ConnectionTimeout = defaults.ConnectionTimeout
	}
	if result.IdleTimeout == 0 {
		result.IdleTimeout = defaults.IdleTimeout
	}

	return result
}


// mergeOptimizersConfig performs deep merge of optimizers configuration.
func mergeOptimizersConfig(partial OptimizersConfig, defaults OptimizersConfig) OptimizersConfig {
	result := partial

	// Merge BootstrapFewShot
	result.BootstrapFewShot = mergeBootstrapFewShotConfig(partial.BootstrapFewShot, defaults.BootstrapFewShot)

	// Merge MIPRO
	result.MIPRO = mergeMIPROConfig(partial.MIPRO, defaults.MIPRO)

	// Merge COPRO
	result.COPRO = mergeCOPROConfig(partial.COPRO, defaults.COPRO)

	// Merge SIMBA
	result.SIMBA = mergeSIMBAConfig(partial.SIMBA, defaults.SIMBA)

	// Merge TPE
	result.TPE = mergeTPEConfig(partial.TPE, defaults.TPE)

	return result
}

// mergeBootstrapFewShotConfig performs deep merge of bootstrap few-shot configuration.
func mergeBootstrapFewShotConfig(partial BootstrapFewShotConfig, defaults BootstrapFewShotConfig) BootstrapFewShotConfig {
	result := partial

	if result.MaxExamples == 0 {
		result.MaxExamples = defaults.MaxExamples
	}
	if result.TeacherModel == "" {
		result.TeacherModel = defaults.TeacherModel
	}
	if result.StudentModel == "" {
		result.StudentModel = defaults.StudentModel
	}
	if result.BootstrapIterations == 0 {
		result.BootstrapIterations = defaults.BootstrapIterations
	}

	return result
}

// mergeMIPROConfig performs deep merge of MIPRO configuration.
func mergeMIPROConfig(partial MIPROConfig, defaults MIPROConfig) MIPROConfig {
	result := partial

	if result.PopulationSize == 0 {
		result.PopulationSize = defaults.PopulationSize
	}
	if result.NumGenerations == 0 {
		result.NumGenerations = defaults.NumGenerations
	}
	if result.MutationRate == 0 {
		result.MutationRate = defaults.MutationRate
	}
	if result.CrossoverRate == 0 {
		result.CrossoverRate = defaults.CrossoverRate
	}

	return result
}

// mergeCOPROConfig performs deep merge of COPRO configuration.
func mergeCOPROConfig(partial COPROConfig, defaults COPROConfig) COPROConfig {
	result := partial

	if result.MaxIterations == 0 {
		result.MaxIterations = defaults.MaxIterations
	}
	if result.ConvergenceThreshold == 0 {
		result.ConvergenceThreshold = defaults.ConvergenceThreshold
	}
	if result.LearningRate == 0 {
		result.LearningRate = defaults.LearningRate
	}

	return result
}

// mergeSIMBAConfig performs deep merge of SIMBA configuration.
func mergeSIMBAConfig(partial SIMBAConfig, defaults SIMBAConfig) SIMBAConfig {
	result := partial

	if result.NumCandidates == 0 {
		result.NumCandidates = defaults.NumCandidates
	}
	if result.SelectionStrategy == "" {
		result.SelectionStrategy = defaults.SelectionStrategy
	}
	if result.EvaluationMetric == "" {
		result.EvaluationMetric = defaults.EvaluationMetric
	}

	return result
}

// mergeTPEConfig performs deep merge of TPE configuration.
func mergeTPEConfig(partial TPEConfig, defaults TPEConfig) TPEConfig {
	result := partial

	if result.NumTrials == 0 {
		result.NumTrials = defaults.NumTrials
	}
	if result.NumStartupTrials == 0 {
		result.NumStartupTrials = defaults.NumStartupTrials
	}
	if result.Percentile == 0 {
		result.Percentile = defaults.Percentile
	}
	if result.RandomSeed == 0 {
		result.RandomSeed = defaults.RandomSeed
	}

	return result
}

