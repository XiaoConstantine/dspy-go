package config

import (
	"time"
)

// Config represents the complete configuration for the dspy-go system.
type Config struct {
	// LLM configuration
	LLM LLMConfig `yaml:"llm" validate:"required"`

	// Logging configuration
	Logging LoggingConfig `yaml:"logging,omitempty" validate:"omitempty"`

	// Execution configuration
	Execution ExecutionConfig `yaml:"execution,omitempty" validate:"omitempty"`

	// Module configuration
	Modules ModulesConfig `yaml:"modules,omitempty" validate:"omitempty"`

	// Agents configuration
	Agents AgentsConfig `yaml:"agents,omitempty" validate:"omitempty"`

	// Tools configuration
	Tools ToolsConfig `yaml:"tools,omitempty" validate:"omitempty"`

	// Optimizer configuration
	Optimizers OptimizersConfig `yaml:"optimizers,omitempty" validate:"omitempty"`

	// Interceptors configuration
	Interceptors InterceptorsConfig `yaml:"interceptors,omitempty" validate:"omitempty"`

}

// LLMConfig holds configuration for Language Learning Models.
type LLMConfig struct {
	// Default LLM configuration
	Default LLMProviderConfig `yaml:"default" validate:"required"`

	// Teacher LLM configuration (for optimization)
	Teacher LLMProviderConfig `yaml:"teacher,omitempty" validate:"omitempty"`

	// Available providers and their configurations
	Providers map[string]LLMProviderConfig `yaml:"providers,omitempty" validate:"omitempty"`

	// Global LLM settings
	GlobalSettings LLMGlobalSettings `yaml:"global_settings,omitempty" validate:"omitempty"`
}

// LLMProviderConfig represents configuration for a specific LLM provider.
type LLMProviderConfig struct {
	// Provider name (anthropic, google, ollama, llamacpp)
	Provider string `yaml:"provider" validate:"required,oneof=anthropic google ollama llamacpp"`

	// Model ID (e.g., claude-3-sonnet-20240229)
	ModelID string `yaml:"model_id" validate:"required"`

	// API key for the provider
	APIKey string `yaml:"api_key,omitempty"`

	// Endpoint configuration
	Endpoint EndpointConfig `yaml:"endpoint,omitempty"`

	// Generation parameters
	Generation GenerationConfig `yaml:"generation,omitempty"`

	// Embedding configuration
	Embedding EmbeddingConfig `yaml:"embedding,omitempty"`

	// Capabilities this provider supports
	Capabilities []string `yaml:"capabilities,omitempty"`
}

// EndpointConfig holds endpoint-specific configuration.
type EndpointConfig struct {
	// Base URL for the API
	BaseURL string `yaml:"base_url"`

	// API path (if different from default)
	Path string `yaml:"path"`

	// HTTP headers to include
	Headers map[string]string `yaml:"headers"`

	// Request timeout
	Timeout time.Duration `yaml:"timeout"`

	// Retry configuration
	Retry RetryConfig `yaml:"retry"`
}

// RetryConfig holds retry-specific configuration.
type RetryConfig struct {
	// Maximum number of retries
	MaxRetries int `yaml:"max_retries" validate:"min=0"`

	// Initial backoff duration
	InitialBackoff time.Duration `yaml:"initial_backoff" validate:"min=0"`

	// Maximum backoff duration
	MaxBackoff time.Duration `yaml:"max_backoff" validate:"min=0"`

	// Backoff multiplier
	BackoffMultiplier float64 `yaml:"backoff_multiplier" validate:"min=1.0"`
}

// GenerationConfig holds text generation parameters.
type GenerationConfig struct {
	// Maximum tokens to generate
	MaxTokens int `yaml:"max_tokens" validate:"min=1"`

	// Sampling temperature
	Temperature float64 `yaml:"temperature" validate:"min=0,max=2"`

	// Top-p sampling
	TopP float64 `yaml:"top_p" validate:"min=0,max=1"`

	// Presence penalty
	PresencePenalty float64 `yaml:"presence_penalty" validate:"min=-2,max=2"`

	// Frequency penalty
	FrequencyPenalty float64 `yaml:"frequency_penalty" validate:"min=-2,max=2"`

	// Stop sequences
	StopSequences []string `yaml:"stop_sequences"`
}

// EmbeddingConfig holds embedding generation parameters.
type EmbeddingConfig struct {
	// Model to use for embeddings
	Model string `yaml:"model,omitempty"`

	// Batch size for bulk operations
	BatchSize int `yaml:"batch_size" validate:"min=1"`

	// Additional parameters
	Params map[string]interface{} `yaml:"params"`
}

// LLMGlobalSettings holds global LLM configuration.
type LLMGlobalSettings struct {
	// Default concurrency level
	ConcurrencyLevel int `yaml:"concurrency_level" validate:"min=1"`

	// Enable request/response logging
	LogRequests bool `yaml:"log_requests"`

	// Enable token usage tracking
	TrackTokenUsage bool `yaml:"track_token_usage"`

	// Enable metrics collection
	EnableMetrics bool `yaml:"enable_metrics"`
}

// LoggingConfig holds logging configuration.
type LoggingConfig struct {
	// Log level (DEBUG, INFO, WARN, ERROR, FATAL)
	Level string `yaml:"level" validate:"oneof=DEBUG INFO WARN ERROR FATAL"`

	// Output configurations
	Outputs []LogOutputConfig `yaml:"outputs"`

	// Sampling rate for high-frequency events
	SampleRate uint32 `yaml:"sample_rate"`

	// Default fields to include in all logs
	DefaultFields map[string]interface{} `yaml:"default_fields"`
}

// LogOutputConfig represents a logging output destination.
type LogOutputConfig struct {
	// Type of output (console, file, syslog)
	Type string `yaml:"type" validate:"required,oneof=console file syslog"`

	// Output format (json, text)
	Format string `yaml:"format" validate:"oneof=json text"`

	// File path (for file outputs)
	FilePath string `yaml:"file_path"`

	// Whether to use colors (for console outputs)
	Colors bool `yaml:"colors"`

	// Log rotation configuration
	Rotation LogRotationConfig `yaml:"rotation"`
}

// LogRotationConfig holds log rotation settings.
type LogRotationConfig struct {
	// Maximum file size before rotation
	MaxSize int `yaml:"max_size" validate:"min=1"`

	// Maximum number of old files to retain
	MaxFiles int `yaml:"max_files" validate:"min=1"`

	// Maximum age of files in days
	MaxAge int `yaml:"max_age" validate:"min=1"`

	// Whether to compress old files
	Compress bool `yaml:"compress"`
}

// ExecutionConfig holds execution-related configuration.
type ExecutionConfig struct {
	// Default timeout for operations
	DefaultTimeout time.Duration `yaml:"default_timeout" validate:"min=1s"`

	// Maximum number of concurrent operations
	MaxConcurrency int `yaml:"max_concurrency" validate:"min=1"`

	// Context configuration
	Context ContextConfig `yaml:"context"`

	// Tracing configuration
	Tracing TracingConfig `yaml:"tracing"`
}

// ContextConfig holds context-related configuration.
type ContextConfig struct {
	// Default context timeout
	DefaultTimeout time.Duration `yaml:"default_timeout" validate:"min=1s"`

	// Context buffer size
	BufferSize int `yaml:"buffer_size" validate:"min=1"`
}

// TracingConfig holds tracing configuration.
type TracingConfig struct {
	// Enable tracing
	Enabled bool `yaml:"enabled"`

	// Sampling rate for traces
	SamplingRate float64 `yaml:"sampling_rate" validate:"min=0,max=1"`

	// Trace export configuration
	Exporter TracingExporterConfig `yaml:"exporter"`
}

// TracingExporterConfig holds trace export configuration.
type TracingExporterConfig struct {
	// Exporter type (jaeger, zipkin, otlp)
	Type string `yaml:"type" validate:"oneof=jaeger zipkin otlp"`

	// Endpoint for the exporter
	Endpoint string `yaml:"endpoint"`

	// Additional headers
	Headers map[string]string `yaml:"headers"`
}

// ModulesConfig holds module-specific configuration.
type ModulesConfig struct {
	// Chain of Thought configuration
	ChainOfThought ChainOfThoughtConfig `yaml:"chain_of_thought"`

	// Multi-Chain Comparison configuration
	MultiChainComparison MultiChainComparisonConfig `yaml:"multi_chain_comparison"`

	// ReAct configuration
	ReAct ReActConfig `yaml:"react"`

	// Refine configuration
	Refine RefineConfig `yaml:"refine"`

	// Predict configuration
	Predict PredictConfig `yaml:"predict"`
}

// ChainOfThoughtConfig holds Chain of Thought module configuration.
type ChainOfThoughtConfig struct {
	// Maximum reasoning steps
	MaxSteps int `yaml:"max_steps" validate:"min=1"`

	// Whether to include reasoning in output
	IncludeReasoning bool `yaml:"include_reasoning"`

	// Step delimiter
	StepDelimiter string `yaml:"step_delimiter"`
}

// MultiChainComparisonConfig holds Multi-Chain Comparison configuration.
type MultiChainComparisonConfig struct {
	// Number of chains to run
	NumChains int `yaml:"num_chains" validate:"min=2"`

	// Comparison strategy
	ComparisonStrategy string `yaml:"comparison_strategy" validate:"oneof=majority_vote highest_confidence best_score"`

	// Enable parallel execution
	ParallelExecution bool `yaml:"parallel_execution"`
}

// ReActConfig holds ReAct module configuration.
type ReActConfig struct {
	// Maximum number of action/observation cycles
	MaxCycles int `yaml:"max_cycles" validate:"min=1"`

	// Action timeout
	ActionTimeout time.Duration `yaml:"action_timeout" validate:"min=1s"`

	// Whether to include intermediate steps
	IncludeIntermediateSteps bool `yaml:"include_intermediate_steps"`
}

// RefineConfig holds Refine module configuration.
type RefineConfig struct {
	// Maximum refinement iterations
	MaxIterations int `yaml:"max_iterations" validate:"min=1"`

	// Convergence threshold
	ConvergenceThreshold float64 `yaml:"convergence_threshold" validate:"min=0,max=1"`

	// Refinement strategy
	RefinementStrategy string `yaml:"refinement_strategy" validate:"oneof=iterative_improvement feedback_based"`
}

// PredictConfig holds Predict module configuration.
type PredictConfig struct {
	// Default prediction settings
	DefaultSettings PredictSettings `yaml:"default_settings"`

	// Caching configuration
	Caching CachingConfig `yaml:"caching"`
}

// PredictSettings holds prediction settings.
type PredictSettings struct {
	// Include confidence scores
	IncludeConfidence bool `yaml:"include_confidence"`

	// Temperature for prediction
	Temperature float64 `yaml:"temperature" validate:"min=0,max=2"`

	// Top-k for prediction
	TopK int `yaml:"top_k" validate:"min=1"`
}

// CachingConfig holds caching configuration.
type CachingConfig struct {
	// Enable caching
	Enabled bool `yaml:"enabled"`

	// Cache TTL
	TTL time.Duration `yaml:"ttl" validate:"min=1s"`

	// Maximum cache size (in bytes)
	MaxSize int64 `yaml:"max_size" validate:"min=1"`

	// Cache type (memory, sqlite)
	Type string `yaml:"type" validate:"oneof=memory sqlite"`

	// SQLite specific configuration
	SQLiteConfig SQLiteCacheConfig `yaml:"sqlite_config,omitempty"`

	// Memory cache specific configuration
	MemoryConfig MemoryCacheConfig `yaml:"memory_config,omitempty"`
}

// SQLiteCacheConfig holds SQLite-specific cache configuration.
type SQLiteCacheConfig struct {
	// Path to SQLite database file
	Path string `yaml:"path"`

	// Enable WAL mode for better concurrent performance
	EnableWAL bool `yaml:"enable_wal"`

	// Vacuum interval for database maintenance
	VacuumInterval time.Duration `yaml:"vacuum_interval"`

	// Maximum number of connections
	MaxConnections int `yaml:"max_connections"`
}

// MemoryCacheConfig holds memory cache specific configuration.
type MemoryCacheConfig struct {
	// Cleanup interval for expired entries
	CleanupInterval time.Duration `yaml:"cleanup_interval"`

	// Number of shards for concurrent access
	ShardCount int `yaml:"shard_count"`
}

// AgentsConfig holds agent-specific configuration.
type AgentsConfig struct {
	// Default agent configuration
	Default AgentConfig `yaml:"default"`

	// Memory configuration
	Memory AgentMemoryConfig `yaml:"memory"`

	// Workflow configuration
	Workflows WorkflowsConfig `yaml:"workflows"`
}

// AgentConfig holds individual agent configuration.
type AgentConfig struct {
	// Maximum conversation history
	MaxHistory int `yaml:"max_history" validate:"min=1"`

	// Agent timeout
	Timeout time.Duration `yaml:"timeout" validate:"min=1s"`

	// Tool use configuration
	ToolUse ToolUseConfig `yaml:"tool_use"`
}

// AgentMemoryConfig holds agent memory configuration.
type AgentMemoryConfig struct {
	// Memory type (buffered, sqlite, redis)
	Type string `yaml:"type" validate:"oneof=buffered sqlite redis"`

	// Memory capacity
	Capacity int `yaml:"capacity" validate:"min=1"`

	// Memory persistence settings
	Persistence MemoryPersistenceConfig `yaml:"persistence"`
}

// MemoryPersistenceConfig holds memory persistence configuration.
type MemoryPersistenceConfig struct {
	// Enable persistence
	Enabled bool `yaml:"enabled"`

	// Storage path
	Path string `yaml:"path"`

	// Sync interval
	SyncInterval time.Duration `yaml:"sync_interval" validate:"min=1s"`
}

// WorkflowsConfig holds workflow configuration.
type WorkflowsConfig struct {
	// Default workflow timeout
	DefaultTimeout time.Duration `yaml:"default_timeout" validate:"min=1s"`

	// Maximum parallel workflows
	MaxParallel int `yaml:"max_parallel" validate:"min=1"`

	// Workflow persistence
	Persistence WorkflowPersistenceConfig `yaml:"persistence"`
}

// WorkflowPersistenceConfig holds workflow persistence configuration.
type WorkflowPersistenceConfig struct {
	// Enable persistence
	Enabled bool `yaml:"enabled"`

	// Storage backend
	Backend string `yaml:"backend" validate:"oneof=file sqlite redis"`

	// Storage configuration
	Config map[string]interface{} `yaml:"config"`
}

// ToolUseConfig holds tool usage configuration.
type ToolUseConfig struct {
	// Maximum tools per request
	MaxTools int `yaml:"max_tools" validate:"min=1"`

	// Tool timeout
	Timeout time.Duration `yaml:"timeout" validate:"min=1s"`

	// Enable parallel tool execution
	ParallelExecution bool `yaml:"parallel_execution"`
}

// ToolsConfig holds tool configuration.
type ToolsConfig struct {
	// Registry configuration
	Registry ToolRegistryConfig `yaml:"registry"`

	// MCP configuration
	MCP MCPConfig `yaml:"mcp"`

	// Function tools configuration
	Functions FunctionToolsConfig `yaml:"functions"`
}

// ToolRegistryConfig holds tool registry configuration.
type ToolRegistryConfig struct {
	// Maximum registered tools
	MaxTools int `yaml:"max_tools,omitempty" validate:"omitempty,min=1"`

	// Tool discovery paths
	DiscoveryPaths []string `yaml:"discovery_paths,omitempty"`

	// Auto-discovery enabled
	AutoDiscovery bool `yaml:"auto_discovery,omitempty"`
}

// MCPConfig holds MCP (Model Context Protocol) configuration.
type MCPConfig struct {
	// MCP servers configuration
	Servers []MCPServerConfig `yaml:"servers,omitempty"`

	// Default timeout for MCP operations
	DefaultTimeout time.Duration `yaml:"default_timeout,omitempty" validate:"omitempty,min=1s"`

	// Connection pool settings
	ConnectionPool MCPConnectionPoolConfig `yaml:"connection_pool,omitempty"`
}

// MCPServerConfig holds individual MCP server configuration.
type MCPServerConfig struct {
	// Server name
	Name string `yaml:"name" validate:"required"`

	// Server command
	Command string `yaml:"command" validate:"required"`

	// Server arguments
	Args []string `yaml:"args"`

	// Environment variables
	Env map[string]string `yaml:"env"`

	// Working directory
	WorkingDir string `yaml:"working_dir"`

	// Server timeout
	Timeout time.Duration `yaml:"timeout" validate:"min=1s"`
}

// MCPConnectionPoolConfig holds MCP connection pool configuration.
type MCPConnectionPoolConfig struct {
	// Maximum connections per server
	MaxConnections int `yaml:"max_connections,omitempty" validate:"omitempty,min=1"`

	// Connection timeout
	ConnectionTimeout time.Duration `yaml:"connection_timeout,omitempty" validate:"omitempty,min=1s"`

	// Idle timeout
	IdleTimeout time.Duration `yaml:"idle_timeout,omitempty" validate:"omitempty,min=1s"`
}

// FunctionToolsConfig holds function tools configuration.
type FunctionToolsConfig struct {
	// Maximum execution time
	MaxExecutionTime time.Duration `yaml:"max_execution_time,omitempty" validate:"omitempty,min=1s"`
}

// OptimizersConfig holds optimizer configuration.
type OptimizersConfig struct {
	// Bootstrap Few-Shot optimizer
	BootstrapFewShot BootstrapFewShotConfig `yaml:"bootstrap_few_shot"`

	// MIPRO optimizer
	MIPRO MIPROConfig `yaml:"mipro"`

	// COPRO optimizer
	COPRO COPROConfig `yaml:"copro"`

	// SIMBA optimizer
	SIMBA SIMBAConfig `yaml:"simba"`

	// TPE optimizer
	TPE TPEConfig `yaml:"tpe"`
}

// BootstrapFewShotConfig holds Bootstrap Few-Shot optimizer configuration.
type BootstrapFewShotConfig struct {
	// Maximum examples to use
	MaxExamples int `yaml:"max_examples" validate:"min=1"`

	// Teacher model configuration
	TeacherModel string `yaml:"teacher_model"`

	// Student model configuration
	StudentModel string `yaml:"student_model"`

	// Bootstrap iterations
	BootstrapIterations int `yaml:"bootstrap_iterations" validate:"min=1"`
}

// MIPROConfig holds MIPRO optimizer configuration.
type MIPROConfig struct {
	// Population size
	PopulationSize int `yaml:"population_size" validate:"min=1"`

	// Number of generations
	NumGenerations int `yaml:"num_generations" validate:"min=1"`

	// Mutation rate
	MutationRate float64 `yaml:"mutation_rate" validate:"min=0,max=1"`

	// Crossover rate
	CrossoverRate float64 `yaml:"crossover_rate" validate:"min=0,max=1"`
}

// COPROConfig holds COPRO optimizer configuration.
type COPROConfig struct {
	// Maximum iterations
	MaxIterations int `yaml:"max_iterations" validate:"min=1"`

	// Convergence threshold
	ConvergenceThreshold float64 `yaml:"convergence_threshold" validate:"min=0,max=1"`

	// Learning rate
	LearningRate float64 `yaml:"learning_rate" validate:"min=0,max=1"`
}

// SIMBAConfig holds SIMBA optimizer configuration.
type SIMBAConfig struct {
	// Number of candidates
	NumCandidates int `yaml:"num_candidates" validate:"min=1"`

	// Selection strategy
	SelectionStrategy string `yaml:"selection_strategy" validate:"oneof=random tournament roulette"`

	// Evaluation metric
	EvaluationMetric string `yaml:"evaluation_metric"`
}

// TPEConfig holds TPE optimizer configuration.
type TPEConfig struct {
	// Number of trials
	NumTrials int `yaml:"num_trials" validate:"min=1"`

	// Number of startup trials
	NumStartupTrials int `yaml:"num_startup_trials" validate:"min=1"`

	// Percentile for good/bad split
	Percentile float64 `yaml:"percentile" validate:"min=0,max=1"`

	// Random seed
	RandomSeed int64 `yaml:"random_seed"`
}

// InterceptorsConfig holds interceptor configuration.
type InterceptorsConfig struct {
	// Module interceptors configuration
	Module ModuleInterceptorsConfig `yaml:"module,omitempty" validate:"omitempty"`

	// Agent interceptors configuration
	Agent AgentInterceptorsConfig `yaml:"agent,omitempty" validate:"omitempty"`

	// Tool interceptors configuration
	Tool ToolInterceptorsConfig `yaml:"tool,omitempty" validate:"omitempty"`

	// Global interceptor settings
	Global GlobalInterceptorConfig `yaml:"global,omitempty" validate:"omitempty"`
}

// ModuleInterceptorsConfig holds module-specific interceptor configuration.
type ModuleInterceptorsConfig struct {
	// Standard interceptors
	Logging   InterceptorToggle     `yaml:"logging,omitempty"`
	Metrics   InterceptorToggle     `yaml:"metrics,omitempty"`
	Tracing   InterceptorToggle     `yaml:"tracing,omitempty"`

	// Performance interceptors
	Caching       CachingInterceptorConfig       `yaml:"caching,omitempty"`
	Timeout       TimeoutInterceptorConfig       `yaml:"timeout,omitempty"`
	CircuitBreaker CircuitBreakerInterceptorConfig `yaml:"circuit_breaker,omitempty"`
	Retry         RetryInterceptorConfig         `yaml:"retry,omitempty"`

	// Security interceptors
	Validation    ValidationInterceptorConfig    `yaml:"validation,omitempty"`
	Authorization AuthorizationInterceptorConfig `yaml:"authorization,omitempty"`
	Sanitization  SanitizationInterceptorConfig  `yaml:"sanitization,omitempty"`
}

// AgentInterceptorsConfig holds agent-specific interceptor configuration.
type AgentInterceptorsConfig struct {
	// Standard interceptors
	Logging   InterceptorToggle `yaml:"logging,omitempty"`
	Metrics   InterceptorToggle `yaml:"metrics,omitempty"`
	Tracing   InterceptorToggle `yaml:"tracing,omitempty"`

	// Performance interceptors
	RateLimit RateLimitInterceptorConfig `yaml:"rate_limit,omitempty"`
	Timeout   TimeoutInterceptorConfig   `yaml:"timeout,omitempty"`

	// Security interceptors
	Authorization AuthorizationInterceptorConfig `yaml:"authorization,omitempty"`
	Audit         AuditInterceptorConfig         `yaml:"audit,omitempty"`
}

// ToolInterceptorsConfig holds tool-specific interceptor configuration.
type ToolInterceptorsConfig struct {
	// Standard interceptors
	Logging   InterceptorToggle `yaml:"logging,omitempty"`
	Metrics   InterceptorToggle `yaml:"metrics,omitempty"`
	Tracing   InterceptorToggle `yaml:"tracing,omitempty"`

	// Performance interceptors
	Caching CachingInterceptorConfig `yaml:"caching,omitempty"`
	Timeout TimeoutInterceptorConfig `yaml:"timeout,omitempty"`

	// Security interceptors
	Validation   ValidationInterceptorConfig   `yaml:"validation,omitempty"`
	Authorization AuthorizationInterceptorConfig `yaml:"authorization,omitempty"`
	Sanitization SanitizationInterceptorConfig  `yaml:"sanitization,omitempty"`
}

// GlobalInterceptorConfig holds global interceptor settings.
type GlobalInterceptorConfig struct {
	// Enable interceptors globally
	Enabled bool `yaml:"enabled"`

	// Default timeout for all interceptors
	DefaultTimeout time.Duration `yaml:"default_timeout,omitempty" validate:"omitempty,min=1s"`

	// Maximum interceptor chain length
	MaxChainLength int `yaml:"max_chain_length,omitempty" validate:"omitempty,min=1"`

	// Enable performance monitoring for interceptors
	MonitorPerformance bool `yaml:"monitor_performance,omitempty"`
}

// InterceptorToggle represents a simple on/off toggle for interceptors.
type InterceptorToggle struct {
	Enabled bool `yaml:"enabled"`
}

// CachingInterceptorConfig holds caching interceptor configuration.
type CachingInterceptorConfig struct {
	Enabled bool          `yaml:"enabled"`
	TTL     time.Duration `yaml:"ttl,omitempty" validate:"omitempty,min=1s"`
	MaxSize int64         `yaml:"max_size,omitempty" validate:"omitempty,min=1"`
	Type    string        `yaml:"type,omitempty" validate:"omitempty,oneof=memory sqlite"`
}

// TimeoutInterceptorConfig holds timeout interceptor configuration.
type TimeoutInterceptorConfig struct {
	Enabled bool          `yaml:"enabled"`
	Timeout time.Duration `yaml:"timeout,omitempty" validate:"omitempty,min=1s"`
}

// CircuitBreakerInterceptorConfig holds circuit breaker interceptor configuration.
type CircuitBreakerInterceptorConfig struct {
	Enabled           bool          `yaml:"enabled"`
	FailureThreshold  int           `yaml:"failure_threshold,omitempty" validate:"omitempty,min=1"`
	RecoveryTimeout   time.Duration `yaml:"recovery_timeout,omitempty" validate:"omitempty,min=1s"`
	HalfOpenRequests  int           `yaml:"half_open_requests,omitempty" validate:"omitempty,min=1"`
}

// RetryInterceptorConfig holds retry interceptor configuration.
type RetryInterceptorConfig struct {
	Enabled        bool          `yaml:"enabled"`
	MaxRetries     int           `yaml:"max_retries,omitempty" validate:"omitempty,min=1"`
	InitialBackoff time.Duration `yaml:"initial_backoff,omitempty" validate:"omitempty,min=1ms"`
	MaxBackoff     time.Duration `yaml:"max_backoff,omitempty" validate:"omitempty,min=1ms"`
	BackoffFactor  float64       `yaml:"backoff_factor,omitempty" validate:"omitempty,min=1.0"`
}

// ValidationInterceptorConfig holds validation interceptor configuration.
type ValidationInterceptorConfig struct {
	Enabled              bool     `yaml:"enabled"`
	StrictMode           bool     `yaml:"strict_mode,omitempty"`
	RequiredFields       []string `yaml:"required_fields,omitempty"`
	MaxInputSize         int64    `yaml:"max_input_size,omitempty" validate:"omitempty,min=1"`
	MaxStringLength      int      `yaml:"max_string_length,omitempty" validate:"omitempty,min=1"`
	AllowedContentTypes  []string `yaml:"allowed_content_types,omitempty"`
}

// AuthorizationInterceptorConfig holds authorization interceptor configuration.
type AuthorizationInterceptorConfig struct {
	Enabled         bool              `yaml:"enabled"`
	RequireAuth     bool              `yaml:"require_auth,omitempty"`
	AllowedRoles    []string          `yaml:"allowed_roles,omitempty"`
	RequiredScopes  []string          `yaml:"required_scopes,omitempty"`
	CustomRules     map[string]string `yaml:"custom_rules,omitempty"`
}

// SanitizationInterceptorConfig holds sanitization interceptor configuration.
type SanitizationInterceptorConfig struct {
	Enabled         bool     `yaml:"enabled"`
	RemoveHTML      bool     `yaml:"remove_html,omitempty"`
	RemoveSQL       bool     `yaml:"remove_sql,omitempty"`
	RemoveScript    bool     `yaml:"remove_script,omitempty"`
	CustomPatterns  []string `yaml:"custom_patterns,omitempty"`
	MaxStringLength int      `yaml:"max_string_length,omitempty" validate:"omitempty,min=1"`
}

// RateLimitInterceptorConfig holds rate limiting interceptor configuration.
type RateLimitInterceptorConfig struct {
	Enabled       bool          `yaml:"enabled"`
	RequestsPerMinute int       `yaml:"requests_per_minute,omitempty" validate:"omitempty,min=1"`
	BurstSize     int           `yaml:"burst_size,omitempty" validate:"omitempty,min=1"`
	WindowSize    time.Duration `yaml:"window_size,omitempty" validate:"omitempty,min=1s"`
}

// AuditInterceptorConfig holds audit interceptor configuration.
type AuditInterceptorConfig struct {
	Enabled      bool   `yaml:"enabled"`
	LogLevel     string `yaml:"log_level,omitempty" validate:"omitempty,oneof=DEBUG INFO WARN ERROR"`
	IncludeInput bool   `yaml:"include_input,omitempty"`
	IncludeOutput bool  `yaml:"include_output,omitempty"`
	AuditPath    string `yaml:"audit_path,omitempty"`
}

// Validate validates the configuration using the singleton validator.
func (c *Config) Validate() error {
	return ValidateConfiguration(c)
}
