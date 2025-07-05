package config

import (
	"time"

	"github.com/go-playground/validator/v10"
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

	// Storage configuration
	Storage StorageConfig `yaml:"storage,omitempty" validate:"omitempty"`
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

	// Maximum cache size
	MaxSize int `yaml:"max_size" validate:"min=1"`

	// Cache type (memory, redis, file)
	Type string `yaml:"type" validate:"oneof=memory redis file"`

	// Cache configuration
	Config map[string]interface{} `yaml:"config"`
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
	MaxTools int `yaml:"max_tools" validate:"min=1"`

	// Tool discovery paths
	DiscoveryPaths []string `yaml:"discovery_paths"`

	// Auto-discovery enabled
	AutoDiscovery bool `yaml:"auto_discovery"`
}

// MCPConfig holds MCP (Model Context Protocol) configuration.
type MCPConfig struct {
	// MCP servers configuration
	Servers []MCPServerConfig `yaml:"servers"`

	// Default timeout for MCP operations
	DefaultTimeout time.Duration `yaml:"default_timeout" validate:"min=1s"`

	// Connection pool settings
	ConnectionPool MCPConnectionPoolConfig `yaml:"connection_pool"`
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
	MaxConnections int `yaml:"max_connections" validate:"min=1"`

	// Connection timeout
	ConnectionTimeout time.Duration `yaml:"connection_timeout" validate:"min=1s"`

	// Idle timeout
	IdleTimeout time.Duration `yaml:"idle_timeout" validate:"min=1s"`
}

// FunctionToolsConfig holds function tools configuration.
type FunctionToolsConfig struct {
	// Maximum execution time
	MaxExecutionTime time.Duration `yaml:"max_execution_time" validate:"min=1s"`

	// Enable sandboxing
	EnableSandbox bool `yaml:"enable_sandbox"`

	// Sandbox configuration
	Sandbox SandboxConfig `yaml:"sandbox"`
}

// SandboxConfig holds sandbox configuration.
type SandboxConfig struct {
	// Sandbox type (docker, wasm, native)
	Type string `yaml:"type" validate:"oneof=docker wasm native"`

	// Resource limits
	ResourceLimits ResourceLimitsConfig `yaml:"resource_limits"`

	// Allowed capabilities
	AllowedCapabilities []string `yaml:"allowed_capabilities"`
}

// ResourceLimitsConfig holds resource limits configuration.
type ResourceLimitsConfig struct {
	// Memory limit in MB
	MemoryMB int `yaml:"memory_mb" validate:"min=1"`

	// CPU limit (number of cores)
	CPUCores float64 `yaml:"cpu_cores" validate:"min=0.1"`

	// Execution timeout
	ExecutionTimeout time.Duration `yaml:"execution_timeout" validate:"min=1s"`
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

// StorageConfig holds storage configuration.
type StorageConfig struct {
	// Default storage backend
	DefaultBackend string `yaml:"default_backend" validate:"oneof=file sqlite redis"`

	// Storage backends configuration
	Backends map[string]StorageBackendConfig `yaml:"backends"`

	// Compression settings
	Compression CompressionConfig `yaml:"compression"`

	// Encryption settings
	Encryption EncryptionConfig `yaml:"encryption"`
}

// StorageBackendConfig holds storage backend configuration.
type StorageBackendConfig struct {
	// Backend type
	Type string `yaml:"type" validate:"required"`

	// Connection string or configuration
	Config map[string]interface{} `yaml:"config"`

	// Connection pool settings
	ConnectionPool StorageConnectionPoolConfig `yaml:"connection_pool"`
}

// StorageConnectionPoolConfig holds storage connection pool configuration.
type StorageConnectionPoolConfig struct {
	// Maximum connections
	MaxConnections int `yaml:"max_connections" validate:"min=1"`

	// Connection timeout
	ConnectionTimeout time.Duration `yaml:"connection_timeout" validate:"min=1s"`

	// Idle timeout
	IdleTimeout time.Duration `yaml:"idle_timeout" validate:"min=1s"`
}

// CompressionConfig holds compression configuration.
type CompressionConfig struct {
	// Enable compression
	Enabled bool `yaml:"enabled"`

	// Compression algorithm (gzip, lz4, zstd)
	Algorithm string `yaml:"algorithm" validate:"oneof=gzip lz4 zstd"`

	// Compression level
	Level int `yaml:"level" validate:"min=1"`
}

// EncryptionConfig holds encryption configuration.
type EncryptionConfig struct {
	// Enable encryption
	Enabled bool `yaml:"enabled"`

	// Encryption algorithm (aes256, chacha20poly1305)
	Algorithm string `yaml:"algorithm" validate:"oneof=aes256 chacha20poly1305"`

	// Key derivation function
	KeyDerivation string `yaml:"key_derivation" validate:"oneof=pbkdf2 scrypt argon2"`

	// Key configuration
	Key EncryptionKeyConfig `yaml:"key"`
}

// EncryptionKeyConfig holds encryption key configuration.
type EncryptionKeyConfig struct {
	// Key source (env, file, kms)
	Source string `yaml:"source" validate:"oneof=env file kms"`

	// Key identifier (env var name, file path, KMS key ID)
	Identifier string `yaml:"identifier" validate:"required"`

	// Key rotation settings
	Rotation KeyRotationConfig `yaml:"rotation"`
}

// KeyRotationConfig holds key rotation configuration.
type KeyRotationConfig struct {
	// Enable key rotation
	Enabled bool `yaml:"enabled"`

	// Rotation interval
	Interval time.Duration `yaml:"interval" validate:"min=1h"`

	// Backup old keys
	BackupOldKeys bool `yaml:"backup_old_keys"`
}

// Validate validates the configuration using the validator.
func (c *Config) Validate() error {
	validate := validator.New()

	// Register custom validators
	if err := registerCustomValidators(validate); err != nil {
		return err
	}

	return validate.Struct(c)
}

// registerCustomValidators registers custom validation functions.
func registerCustomValidators(validate *validator.Validate) error {
	// Use the validators from validation.go
	return registerAllValidators(validate)
}
