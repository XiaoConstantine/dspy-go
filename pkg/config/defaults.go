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
		Storage:    getDefaultStorageConfig(),
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
			EnableSandbox:    true,
			Sandbox: SandboxConfig{
				Type: "native",
				ResourceLimits: ResourceLimitsConfig{
					MemoryMB:         512,
					CPUCores:         1.0,
					ExecutionTimeout: 30 * time.Second,
				},
				AllowedCapabilities: []string{"read", "write", "execute"},
			},
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

// getDefaultStorageConfig returns default storage configuration.
func getDefaultStorageConfig() StorageConfig {
	return StorageConfig{
		DefaultBackend: "file",
		Backends: map[string]StorageBackendConfig{
			"file": {
				Type: "file",
				Config: map[string]interface{}{
					"base_path": "./data",
				},
				ConnectionPool: StorageConnectionPoolConfig{
					MaxConnections:    10,
					ConnectionTimeout: 5 * time.Second,
					IdleTimeout:       1 * time.Minute,
				},
			},
			"sqlite": {
				Type: "sqlite",
				Config: map[string]interface{}{
					"database": "./data/dspy.db",
				},
				ConnectionPool: StorageConnectionPoolConfig{
					MaxConnections:    5,
					ConnectionTimeout: 5 * time.Second,
					IdleTimeout:       1 * time.Minute,
				},
			},
			"memory": {
				Type: "memory",
				Config: map[string]interface{}{
					"max_size": 1000,
				},
				ConnectionPool: StorageConnectionPoolConfig{
					MaxConnections:    1,
					ConnectionTimeout: 1 * time.Second,
					IdleTimeout:       1 * time.Minute,
				},
			},
		},
		Compression: CompressionConfig{
			Enabled:   false,
			Algorithm: "gzip",
			Level:     6,
		},
		Encryption: EncryptionConfig{
			Enabled:       false,
			Algorithm:     "aes256",
			KeyDerivation: "pbkdf2",
			Key: EncryptionKeyConfig{
				Source:     "env",
				Identifier: "DSPY_ENCRYPTION_KEY",
				Rotation: KeyRotationConfig{
					Enabled:       false,
					Interval:      24 * time.Hour,
					BackupOldKeys: true,
				},
			},
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

	// Deep merge other sections only if they are completely empty
	if isEmptyLoggingConfig(partial.Logging) {
		result.Logging = defaults.Logging
	}
	if isEmptyExecutionConfig(partial.Execution) {
		result.Execution = defaults.Execution
	}
	if isEmptyModulesConfig(partial.Modules) {
		result.Modules = defaults.Modules
	}
	if isEmptyAgentsConfig(partial.Agents) {
		result.Agents = defaults.Agents
	}
	if isEmptyToolsConfig(partial.Tools) {
		result.Tools = defaults.Tools
	}
	if isEmptyOptimizersConfig(partial.Optimizers) {
		result.Optimizers = defaults.Optimizers
	}
	if isEmptyStorageConfig(partial.Storage) {
		result.Storage = defaults.Storage
	}

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

// Helper functions to check if configs are empty.
func isEmptyLoggingConfig(config LoggingConfig) bool {
	return config.Level == "" && len(config.Outputs) == 0
}

func isEmptyExecutionConfig(config ExecutionConfig) bool {
	return config.DefaultTimeout == 0 && config.MaxConcurrency == 0
}

func isEmptyModulesConfig(config ModulesConfig) bool {
	return config.ChainOfThought.MaxSteps == 0 &&
		config.MultiChainComparison.NumChains == 0 &&
		config.ReAct.MaxCycles == 0 &&
		config.Refine.MaxIterations == 0
}

func isEmptyAgentsConfig(config AgentsConfig) bool {
	return config.Default.MaxHistory == 0 &&
		config.Memory.Type == "" &&
		config.Workflows.DefaultTimeout == 0
}

func isEmptyToolsConfig(config ToolsConfig) bool {
	return config.Registry.MaxTools == 0 &&
		len(config.MCP.Servers) == 0 &&
		config.Functions.MaxExecutionTime == 0
}

func isEmptyOptimizersConfig(config OptimizersConfig) bool {
	return config.BootstrapFewShot.MaxExamples == 0 &&
		config.MIPRO.PopulationSize == 0 &&
		config.COPRO.MaxIterations == 0 &&
		config.SIMBA.NumCandidates == 0 &&
		config.TPE.NumTrials == 0
}

func isEmptyStorageConfig(config StorageConfig) bool {
	return config.DefaultBackend == "" && len(config.Backends) == 0
}

// ValidateDefaults validates that the default configuration is valid.
func ValidateDefaults() error {
	defaults := GetDefaultConfig()
	return defaults.Validate()
}
