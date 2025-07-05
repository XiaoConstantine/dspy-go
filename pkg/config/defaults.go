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

// MergeWithDefaults merges a partial configuration with defaults.
func MergeWithDefaults(partial *Config) *Config {
	if partial == nil {
		return GetDefaultConfig()
	}

	defaults := GetDefaultConfig()

	// This is a simple merge - for production use, you might want
	// more sophisticated merging logic using reflection or a library
	// like mergo

	if partial.LLM.Default.Provider == "" {
		partial.LLM.Default = defaults.LLM.Default
	}
	if partial.LLM.Teacher.Provider == "" {
		partial.LLM.Teacher = defaults.LLM.Teacher
	}
	if len(partial.LLM.Providers) == 0 {
		partial.LLM.Providers = defaults.LLM.Providers
	}
	if partial.LLM.GlobalSettings.ConcurrencyLevel == 0 {
		partial.LLM.GlobalSettings = defaults.LLM.GlobalSettings
	}

	if partial.Logging.Level == "" {
		partial.Logging = defaults.Logging
	}

	if partial.Execution.DefaultTimeout == 0 {
		partial.Execution = defaults.Execution
	}

	return partial
}

// ValidateDefaults validates that the default configuration is valid.
func ValidateDefaults() error {
	defaults := GetDefaultConfig()
	return defaults.Validate()
}
