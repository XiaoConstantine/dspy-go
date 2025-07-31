package config

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Source represents a configuration source.
type Source interface {
	// Load loads configuration from the source into the provided config
	Load(config *Config, paths []string) error

	// Name returns the name of the source
	Name() string

	// Priority returns the priority of the source (higher priority overrides lower)
	Priority() int
}

// FileSource loads configuration from YAML files.
type FileSource struct {
	priority int
}

// NewFileSource creates a new file source.
func NewFileSource() *FileSource {
	return &FileSource{priority: 100}
}

// NewFileSourceWithPriority creates a new file source with custom priority.
func NewFileSourceWithPriority(priority int) *FileSource {
	return &FileSource{priority: priority}
}

// Name returns the name of the file source.
func (fs *FileSource) Name() string {
	return "file"
}

// Priority returns the priority of the file source.
func (fs *FileSource) Priority() int {
	return fs.priority
}

// Load loads configuration from YAML files.
func (fs *FileSource) Load(config *Config, paths []string) error {
	for _, path := range paths {
		if !fileExists(path) {
			continue
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read config file %s: %w", path, err)
		}

		// Parse YAML and merge into config
		var fileConfig Config
		if err := yaml.Unmarshal(data, &fileConfig); err != nil {
			return fmt.Errorf("failed to parse YAML from %s: %w", path, err)
		}

		// Merge the file config into the main config
		if err := fs.mergeConfig(config, &fileConfig); err != nil {
			return fmt.Errorf("failed to merge config from %s: %w", path, err)
		}
	}

	return nil
}

// mergeConfig merges source config into target config.
func (fs *FileSource) mergeConfig(target, source *Config) error {
	// Use YAML marshaling for deep merge
	sourceData, err := yaml.Marshal(source)
	if err != nil {
		return fmt.Errorf("failed to marshal source config: %w", err)
	}

	// Unmarshal into target to override fields
	if err := yaml.Unmarshal(sourceData, target); err != nil {
		return fmt.Errorf("failed to unmarshal into target config: %w", err)
	}

	return nil
}

// EnvironmentSource loads configuration from environment variables.
type EnvironmentSource struct {
	priority int
	prefix   string
}

// NewEnvironmentSource creates a new environment source.
func NewEnvironmentSource() *EnvironmentSource {
	return &EnvironmentSource{
		priority: 200, // Higher priority than file source
		prefix:   "DSPY_",
	}
}

// NewEnvironmentSourceWithPrefix creates a new environment source with custom prefix.
func NewEnvironmentSourceWithPrefix(prefix string) *EnvironmentSource {
	return &EnvironmentSource{
		priority: 200,
		prefix:   prefix,
	}
}

// NewEnvironmentSourceWithOptions creates a new environment source with custom options.
func NewEnvironmentSourceWithOptions(priority int, prefix string) *EnvironmentSource {
	return &EnvironmentSource{
		priority: priority,
		prefix:   prefix,
	}
}

// Name returns the name of the environment source.
func (es *EnvironmentSource) Name() string {
	return "environment"
}

// Priority returns the priority of the environment source.
func (es *EnvironmentSource) Priority() int {
	return es.priority
}

// Load loads configuration from environment variables.
func (es *EnvironmentSource) Load(config *Config, paths []string) error {
	envVars := es.getEnvironmentVariables()

	// Sort keys to ensure consistent processing order
	// Process longer keys first, then shorter ones (so shorter/abbreviated forms take precedence)
	keys := make([]string, 0, len(envVars))
	for key := range envVars {
		keys = append(keys, key)
	}

	// Sort by length (descending) then alphabetically for consistent ordering
	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) != len(keys[j]) {
			return len(keys[i]) > len(keys[j])
		}
		return keys[i] < keys[j]
	})

	// Apply environment variable overrides in sorted order
	for _, key := range keys {
		value := envVars[key]
		if err := es.setConfigValue(config, key, value); err != nil {
			return fmt.Errorf("failed to set config value %s=%s: %w", key, value, err)
		}
	}

	return nil
}

// getEnvironmentVariables gets all environment variables with the configured prefix.
func (es *EnvironmentSource) getEnvironmentVariables() map[string]string {
	envVars := make(map[string]string)

	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key, value := parts[0], parts[1]

		// Only process environment variables with our specific prefix
		if strings.HasPrefix(key, es.prefix) {
			// Convert environment variable to config key
			configKey := strings.ToLower(strings.TrimPrefix(key, es.prefix))
			configKey = strings.ReplaceAll(configKey, "_", ".")
			envVars[configKey] = value
		}
	}

	return envVars
}

// setConfigValue sets a configuration value using dot notation.
func (es *EnvironmentSource) setConfigValue(config *Config, key, value string) error {
	// Handle common configuration paths
	switch {
	case strings.HasPrefix(key, "llm.default."):
		return es.setLLMProviderValue(&config.LLM.Default, strings.TrimPrefix(key, "llm.default."), value)
	case strings.HasPrefix(key, "llm.teacher."):
		return es.setLLMProviderValue(&config.LLM.Teacher, strings.TrimPrefix(key, "llm.teacher."), value)
	case strings.HasPrefix(key, "llm.global."):
		return es.setLLMGlobalValue(&config.LLM.GlobalSettings, strings.TrimPrefix(key, "llm.global."), value)
	case strings.HasPrefix(key, "logging."):
		return es.setLoggingValue(&config.Logging, strings.TrimPrefix(key, "logging."), value)
	case strings.HasPrefix(key, "execution."):
		return es.setExecutionValue(&config.Execution, strings.TrimPrefix(key, "execution."), value)
	case strings.HasPrefix(key, "storage."):
		// Storage configuration removed
		return nil
	case strings.HasPrefix(key, "modules."):
		return es.setModulesValue(&config.Modules, strings.TrimPrefix(key, "modules."), value)
	case strings.HasPrefix(key, "agents."):
		return es.setAgentsValue(&config.Agents, strings.TrimPrefix(key, "agents."), value)
	case strings.HasPrefix(key, "tools."):
		return es.setToolsValue(&config.Tools, strings.TrimPrefix(key, "tools."), value)
	case strings.HasPrefix(key, "optimizers."):
		return es.setOptimizersValue(&config.Optimizers, strings.TrimPrefix(key, "optimizers."), value)
	default:
		// For unhandled paths, simply ignore them rather than failing
		// This allows for more flexible environment variable usage
		return nil
	}
}

// setLLMProviderValue sets LLM provider configuration values.
func (es *EnvironmentSource) setLLMProviderValue(provider *LLMProviderConfig, key, value string) error {
	switch key {
	case "provider":
		provider.Provider = value
	case "model.id", "modelid":
		provider.ModelID = value
	case "api.key", "apikey":
		provider.APIKey = value
	case "endpoint.baseurl", "endpoint.base.url":
		provider.Endpoint.BaseURL = value
	case "endpoint.timeout":
		timeout, err := es.parseDuration(value)
		if err != nil {
			return fmt.Errorf("invalid timeout duration: %s", value)
		}
		provider.Endpoint.Timeout = timeout
	case "generation.maxTokens", "generation.max.tokens":
		if maxTokens, err := strconv.Atoi(value); err == nil {
			provider.Generation.MaxTokens = maxTokens
		} else {
			return fmt.Errorf("invalid max tokens: %s", value)
		}
	case "generation.temperature":
		if temperature, err := strconv.ParseFloat(value, 64); err == nil {
			provider.Generation.Temperature = temperature
		} else {
			return fmt.Errorf("invalid temperature: %s", value)
		}
	case "generation.topp", "generation.top.p":
		if topP, err := strconv.ParseFloat(value, 64); err == nil {
			provider.Generation.TopP = topP
		} else {
			return fmt.Errorf("invalid top-p: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setLLMGlobalValue sets LLM global configuration values.
func (es *EnvironmentSource) setLLMGlobalValue(global *LLMGlobalSettings, key, value string) error {
	switch key {
	case "concurrency.level", "concurrencyLevel":
		if level, err := strconv.Atoi(value); err == nil {
			global.ConcurrencyLevel = level
		} else {
			return fmt.Errorf("invalid concurrency level: %s", value)
		}
	case "log.requests", "logRequests":
		if logRequests, err := strconv.ParseBool(value); err == nil {
			global.LogRequests = logRequests
		} else {
			return fmt.Errorf("invalid log requests flag: %s", value)
		}
	case "track.token.usage", "trackTokenUsage":
		if trackTokenUsage, err := strconv.ParseBool(value); err == nil {
			global.TrackTokenUsage = trackTokenUsage
		} else {
			return fmt.Errorf("invalid track token usage flag: %s", value)
		}
	case "enable.metrics", "enableMetrics":
		if enableMetrics, err := strconv.ParseBool(value); err == nil {
			global.EnableMetrics = enableMetrics
		} else {
			return fmt.Errorf("invalid enable metrics flag: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setLoggingValue sets logging configuration values.
func (es *EnvironmentSource) setLoggingValue(logging *LoggingConfig, key, value string) error {
	switch key {
	case "level":
		logging.Level = value
	case "sample.rate", "sampleRate":
		if sampleRate, err := strconv.ParseUint(value, 10, 32); err == nil {
			logging.SampleRate = uint32(sampleRate)
		} else {
			return fmt.Errorf("invalid sample rate: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setExecutionValue sets execution configuration values.
func (es *EnvironmentSource) setExecutionValue(execution *ExecutionConfig, key, value string) error {
	switch key {
	case "default.timeout", "defaultTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			execution.DefaultTimeout = timeout
		} else {
			return fmt.Errorf("invalid default timeout: %s", value)
		}
	case "max.concurrency", "maxConcurrency":
		if maxConcurrency, err := strconv.Atoi(value); err == nil {
			execution.MaxConcurrency = maxConcurrency
		} else {
			return fmt.Errorf("invalid max concurrency: %s", value)
		}
	case "tracing.enabled":
		if enabled, err := strconv.ParseBool(value); err == nil {
			execution.Tracing.Enabled = enabled
		} else {
			return fmt.Errorf("invalid tracing enabled flag: %s", value)
		}
	case "tracing.sampling.rate":
		if rate, err := strconv.ParseFloat(value, 64); err == nil {
			execution.Tracing.SamplingRate = rate
		} else {
			return fmt.Errorf("invalid tracing sampling rate: %s", value)
		}
	default:
		return nil
	}
	return nil
}


// setModulesValue sets modules configuration values.
func (es *EnvironmentSource) setModulesValue(modules *ModulesConfig, key, value string) error {
	switch {
	case strings.HasPrefix(key, "chainofthought.") || strings.HasPrefix(key, "cot."):
		subkey := strings.TrimPrefix(key, "chainofthought.")
		subkey = strings.TrimPrefix(subkey, "cot.")
		return es.setChainOfThoughtValue(&modules.ChainOfThought, subkey, value)
	case strings.HasPrefix(key, "multichaincomparison.") || strings.HasPrefix(key, "mcc."):
		subkey := strings.TrimPrefix(key, "multichaincomparison.")
		subkey = strings.TrimPrefix(subkey, "mcc.")
		return es.setMultiChainComparisonValue(&modules.MultiChainComparison, subkey, value)
	case strings.HasPrefix(key, "react."):
		return es.setReActValue(&modules.ReAct, strings.TrimPrefix(key, "react."), value)
	case strings.HasPrefix(key, "refine."):
		return es.setRefineValue(&modules.Refine, strings.TrimPrefix(key, "refine."), value)
	case strings.HasPrefix(key, "predict."):
		return es.setPredictValue(&modules.Predict, strings.TrimPrefix(key, "predict."), value)
	default:
		return nil
	}
}

// setChainOfThoughtValue sets Chain of Thought configuration values.
func (es *EnvironmentSource) setChainOfThoughtValue(cot *ChainOfThoughtConfig, key, value string) error {
	switch key {
	case "max.steps", "maxSteps", "maxsteps":
		if maxSteps, err := strconv.Atoi(value); err == nil {
			cot.MaxSteps = maxSteps
		} else {
			return fmt.Errorf("invalid max steps: %s", value)
		}
	case "include.reasoning", "includeReasoning":
		if include, err := strconv.ParseBool(value); err == nil {
			cot.IncludeReasoning = include
		} else {
			return fmt.Errorf("invalid include reasoning flag: %s", value)
		}
	case "step.delimiter", "stepDelimiter":
		cot.StepDelimiter = value
	default:
		return nil
	}
	return nil
}

// setMultiChainComparisonValue sets Multi-Chain Comparison configuration values.
func (es *EnvironmentSource) setMultiChainComparisonValue(mcc *MultiChainComparisonConfig, key, value string) error {
	switch key {
	case "num.chains", "numChains", "numchains":
		if numChains, err := strconv.Atoi(value); err == nil {
			mcc.NumChains = numChains
		} else {
			return fmt.Errorf("invalid num chains: %s", value)
		}
	case "comparison.strategy", "comparisonStrategy":
		mcc.ComparisonStrategy = value
	case "parallel.execution", "parallelExecution":
		if parallel, err := strconv.ParseBool(value); err == nil {
			mcc.ParallelExecution = parallel
		} else {
			return fmt.Errorf("invalid parallel execution flag: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setReActValue sets ReAct configuration values.
func (es *EnvironmentSource) setReActValue(react *ReActConfig, key, value string) error {
	switch key {
	case "max.cycles", "maxCycles", "max.steps", "maxSteps", "maxsteps":
		if maxCycles, err := strconv.Atoi(value); err == nil {
			react.MaxCycles = maxCycles
		} else {
			return fmt.Errorf("invalid max cycles: %s", value)
		}
	case "action.timeout", "actionTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			react.ActionTimeout = timeout
		} else {
			return fmt.Errorf("invalid action timeout: %s", value)
		}
	case "include.intermediate.steps", "includeIntermediateSteps":
		if include, err := strconv.ParseBool(value); err == nil {
			react.IncludeIntermediateSteps = include
		} else {
			return fmt.Errorf("invalid include intermediate steps flag: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setRefineValue sets Refine configuration values.
func (es *EnvironmentSource) setRefineValue(refine *RefineConfig, key, value string) error {
	switch key {
	case "max.iterations", "maxIterations", "max.steps", "maxSteps", "maxsteps":
		if maxIterations, err := strconv.Atoi(value); err == nil {
			refine.MaxIterations = maxIterations
		} else {
			return fmt.Errorf("invalid max iterations: %s", value)
		}
	case "convergence.threshold", "convergenceThreshold":
		if threshold, err := strconv.ParseFloat(value, 64); err == nil {
			refine.ConvergenceThreshold = threshold
		} else {
			return fmt.Errorf("invalid convergence threshold: %s", value)
		}
	case "refinement.strategy", "refinementStrategy":
		refine.RefinementStrategy = value
	default:
		return nil
	}
	return nil
}

// setPredictValue sets Predict configuration values.
func (es *EnvironmentSource) setPredictValue(predict *PredictConfig, key, value string) error {
	switch {
	case strings.HasPrefix(key, "default.settings.") || strings.HasPrefix(key, "defaults."):
		subkey := strings.TrimPrefix(key, "default.settings.")
		subkey = strings.TrimPrefix(subkey, "defaults.")
		return es.setPredictSettingsValue(&predict.DefaultSettings, subkey, value)
	case strings.HasPrefix(key, "caching."):
		return es.setCachingValue(&predict.Caching, strings.TrimPrefix(key, "caching."), value)
	default:
		return nil
	}
}

// setPredictSettingsValue sets Predict settings values.
func (es *EnvironmentSource) setPredictSettingsValue(settings *PredictSettings, key, value string) error {
	switch key {
	case "include.confidence", "includeConfidence":
		if include, err := strconv.ParseBool(value); err == nil {
			settings.IncludeConfidence = include
		} else {
			return fmt.Errorf("invalid include confidence flag: %s", value)
		}
	case "temperature":
		if temp, err := strconv.ParseFloat(value, 64); err == nil {
			settings.Temperature = temp
		} else {
			return fmt.Errorf("invalid temperature: %s", value)
		}
	case "top.k", "topK":
		if topK, err := strconv.Atoi(value); err == nil {
			settings.TopK = topK
		} else {
			return fmt.Errorf("invalid top K: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setCachingValue sets Caching configuration values.
func (es *EnvironmentSource) setCachingValue(caching *CachingConfig, key, value string) error {
	switch key {
	case "enabled":
		if enabled, err := strconv.ParseBool(value); err == nil {
			caching.Enabled = enabled
		} else {
			return fmt.Errorf("invalid enabled flag: %s", value)
		}
	case "ttl":
		if ttl, err := time.ParseDuration(value); err == nil {
			caching.TTL = ttl
		} else {
			return fmt.Errorf("invalid TTL: %s", value)
		}
	case "max.size", "maxSize":
		if maxSize, err := strconv.ParseInt(value, 10, 64); err == nil {
			caching.MaxSize = maxSize
		} else {
			return fmt.Errorf("invalid max size: %s", value)
		}
	case "type":
		caching.Type = value
	default:
		return nil
	}
	return nil
}

// setAgentsValue sets agents configuration values.
func (es *EnvironmentSource) setAgentsValue(agents *AgentsConfig, key, value string) error {
	switch {
	case strings.HasPrefix(key, "default."):
		return es.setAgentValue(&agents.Default, strings.TrimPrefix(key, "default."), value)
	case strings.HasPrefix(key, "memory."):
		return es.setAgentMemoryValue(&agents.Memory, strings.TrimPrefix(key, "memory."), value)
	case strings.HasPrefix(key, "workflows."):
		return es.setWorkflowsValue(&agents.Workflows, strings.TrimPrefix(key, "workflows."), value)
	default:
		return nil
	}
}

// setAgentValue sets Agent configuration values.
func (es *EnvironmentSource) setAgentValue(agent *AgentConfig, key, value string) error {
	switch key {
	case "max.history", "maxHistory":
		if maxHistory, err := strconv.Atoi(value); err == nil {
			agent.MaxHistory = maxHistory
		} else {
			return fmt.Errorf("invalid max history: %s", value)
		}
	case "timeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			agent.Timeout = timeout
		} else {
			return fmt.Errorf("invalid timeout: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setAgentMemoryValue sets Agent Memory configuration values.
func (es *EnvironmentSource) setAgentMemoryValue(memory *AgentMemoryConfig, key, value string) error {
	switch key {
	case "type":
		memory.Type = value
	case "capacity":
		if capacity, err := strconv.Atoi(value); err == nil {
			memory.Capacity = capacity
		} else {
			return fmt.Errorf("invalid capacity: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setWorkflowsValue sets Workflows configuration values.
func (es *EnvironmentSource) setWorkflowsValue(workflows *WorkflowsConfig, key, value string) error {
	switch key {
	case "default.timeout", "defaultTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			workflows.DefaultTimeout = timeout
		} else {
			return fmt.Errorf("invalid default timeout: %s", value)
		}
	case "max.parallel", "maxParallel":
		if maxParallel, err := strconv.Atoi(value); err == nil {
			workflows.MaxParallel = maxParallel
		} else {
			return fmt.Errorf("invalid max parallel: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setToolsValue sets tools configuration values.
func (es *EnvironmentSource) setToolsValue(tools *ToolsConfig, key, value string) error {
	switch {
	case strings.HasPrefix(key, "registry."):
		return es.setToolRegistryValue(&tools.Registry, strings.TrimPrefix(key, "registry."), value)
	case strings.HasPrefix(key, "toolregistry."):
		return es.setToolRegistryValue(&tools.Registry, strings.TrimPrefix(key, "toolregistry."), value)
	case strings.HasPrefix(key, "mcp."):
		return es.setMCPValue(&tools.MCP, strings.TrimPrefix(key, "mcp."), value)
	case strings.HasPrefix(key, "functions."):
		return es.setFunctionToolsValue(&tools.Functions, strings.TrimPrefix(key, "functions."), value)
	case strings.HasPrefix(key, "functiontools."):
		return es.setFunctionToolsValue(&tools.Functions, strings.TrimPrefix(key, "functiontools."), value)
	default:
		return nil
	}
}

// setToolRegistryValue sets Tool Registry configuration values.
func (es *EnvironmentSource) setToolRegistryValue(registry *ToolRegistryConfig, key, value string) error {
	switch key {
	case "max.tools", "maxTools", "max.cached.tools", "maxCachedTools", "maxcachedtools":
		if maxTools, err := strconv.Atoi(value); err == nil {
			registry.MaxTools = maxTools
		} else {
			return fmt.Errorf("invalid max tools: %s", value)
		}
	case "auto.discovery", "autoDiscovery":
		if autoDiscovery, err := strconv.ParseBool(value); err == nil {
			registry.AutoDiscovery = autoDiscovery
		} else {
			return fmt.Errorf("invalid auto discovery flag: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setMCPValue sets MCP configuration values.
func (es *EnvironmentSource) setMCPValue(mcp *MCPConfig, key, value string) error {
	switch key {
	case "default.timeout", "defaultTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			mcp.DefaultTimeout = timeout
		} else {
			return fmt.Errorf("invalid default timeout: %s", value)
		}
	case "max.connections", "maxConnections":
		if maxConnections, err := strconv.Atoi(value); err == nil {
			mcp.ConnectionPool.MaxConnections = maxConnections
		} else {
			return fmt.Errorf("invalid max connections: %s", value)
		}
	case "connection.timeout", "connectionTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			mcp.ConnectionPool.ConnectionTimeout = timeout
		} else {
			return fmt.Errorf("invalid connection timeout: %s", value)
		}
	case "idle.timeout", "idleTimeout":
		if timeout, err := time.ParseDuration(value); err == nil {
			mcp.ConnectionPool.IdleTimeout = timeout
		} else {
			return fmt.Errorf("invalid idle timeout: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setFunctionToolsValue sets Function Tools configuration values.
func (es *EnvironmentSource) setFunctionToolsValue(functions *FunctionToolsConfig, key, value string) error {
	switch key {
	case "max.execution.time", "maxExecutionTime":
		if maxTime, err := time.ParseDuration(value); err == nil {
			functions.MaxExecutionTime = maxTime
		} else {
			return fmt.Errorf("invalid max execution time: %s", value)
		}
	// Note: sandbox configuration removed as it's not implemented
	default:
		return nil
	}
	return nil
}

// setOptimizersValue sets optimizers configuration values.
func (es *EnvironmentSource) setOptimizersValue(optimizers *OptimizersConfig, key, value string) error {
	switch {
	case strings.HasPrefix(key, "bootstrap.") || strings.HasPrefix(key, "bootstrapfewshot."):
		subkey := strings.TrimPrefix(key, "bootstrap.")
		subkey = strings.TrimPrefix(subkey, "bootstrapfewshot.")
		return es.setBootstrapFewShotValue(&optimizers.BootstrapFewShot, subkey, value)
	case strings.HasPrefix(key, "mipro."):
		return es.setMIPROValue(&optimizers.MIPRO, strings.TrimPrefix(key, "mipro."), value)
	case strings.HasPrefix(key, "copro."):
		return es.setCOPROValue(&optimizers.COPRO, strings.TrimPrefix(key, "copro."), value)
	case strings.HasPrefix(key, "simba."):
		return es.setSIMBAValue(&optimizers.SIMBA, strings.TrimPrefix(key, "simba."), value)
	case strings.HasPrefix(key, "tpe."):
		return es.setTPEValue(&optimizers.TPE, strings.TrimPrefix(key, "tpe."), value)
	default:
		return nil
	}
}

// setBootstrapFewShotValue sets Bootstrap Few-Shot configuration values.
func (es *EnvironmentSource) setBootstrapFewShotValue(bootstrap *BootstrapFewShotConfig, key, value string) error {
	switch key {
	case "max.examples", "maxExamples", "max.bootstrap.samples", "maxBootstrapSamples", "maxbootstrapsamples":
		if maxExamples, err := strconv.Atoi(value); err == nil {
			bootstrap.MaxExamples = maxExamples
		} else {
			return fmt.Errorf("invalid max examples: %s", value)
		}
	case "teacher.model", "teacherModel":
		bootstrap.TeacherModel = value
	case "student.model", "studentModel":
		bootstrap.StudentModel = value
	case "bootstrap.iterations", "bootstrapIterations":
		if iterations, err := strconv.Atoi(value); err == nil {
			bootstrap.BootstrapIterations = iterations
		} else {
			return fmt.Errorf("invalid bootstrap iterations: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setMIPROValue sets MIPRO configuration values.
func (es *EnvironmentSource) setMIPROValue(mipro *MIPROConfig, key, value string) error {
	switch key {
	case "population.size", "populationSize":
		if size, err := strconv.Atoi(value); err == nil {
			mipro.PopulationSize = size
		} else {
			return fmt.Errorf("invalid population size: %s", value)
		}
	case "num.generations", "numGenerations", "max.iterations", "maxIterations", "maxiterations":
		if generations, err := strconv.Atoi(value); err == nil {
			mipro.NumGenerations = generations
		} else {
			return fmt.Errorf("invalid num generations: %s", value)
		}
	case "mutation.rate", "mutationRate":
		if rate, err := strconv.ParseFloat(value, 64); err == nil {
			mipro.MutationRate = rate
		} else {
			return fmt.Errorf("invalid mutation rate: %s", value)
		}
	case "crossover.rate", "crossoverRate":
		if rate, err := strconv.ParseFloat(value, 64); err == nil {
			mipro.CrossoverRate = rate
		} else {
			return fmt.Errorf("invalid crossover rate: %s", value)
		}
	case "max.candidates", "maxCandidates", "maxcandidates":
		// Map maxCandidates to PopulationSize for compatibility
		if size, err := strconv.Atoi(value); err == nil {
			mipro.PopulationSize = size
		} else {
			return fmt.Errorf("invalid population size: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// setCOPROValue sets COPRO configuration values.
func (es *EnvironmentSource) setCOPROValue(copro *COPROConfig, key, value string) error {
	switch key {
	case "max.iterations", "maxIterations", "maxiterations":
		if iterations, err := strconv.Atoi(value); err == nil {
			copro.MaxIterations = iterations
		} else {
			return fmt.Errorf("invalid max iterations: %s", value)
		}
	case "convergence.threshold", "convergenceThreshold":
		if threshold, err := strconv.ParseFloat(value, 64); err == nil {
			copro.ConvergenceThreshold = threshold
		} else {
			return fmt.Errorf("invalid convergence threshold: %s", value)
		}
	case "learning.rate", "learningRate":
		if rate, err := strconv.ParseFloat(value, 64); err == nil {
			copro.LearningRate = rate
		} else {
			return fmt.Errorf("invalid learning rate: %s", value)
		}
	case "max.candidates", "maxCandidates":
		// COPRO doesn't have a candidates field, so we ignore this for compatibility
		return nil
	default:
		return nil
	}
	return nil
}

// setSIMBAValue sets SIMBA configuration values.
func (es *EnvironmentSource) setSIMBAValue(simba *SIMBAConfig, key, value string) error {
	switch key {
	case "num.candidates", "numCandidates", "max.candidates", "maxCandidates", "maxcandidates":
		if candidates, err := strconv.Atoi(value); err == nil {
			simba.NumCandidates = candidates
		} else {
			return fmt.Errorf("invalid num candidates: %s", value)
		}
	case "max.iterations", "maxIterations", "maxiterations":
		// SIMBA doesn't have a MaxIterations field, so ignore this for compatibility
		return nil
	case "selection.strategy", "selectionStrategy":
		simba.SelectionStrategy = value
	case "evaluation.metric", "evaluationMetric":
		simba.EvaluationMetric = value
	default:
		return nil
	}
	return nil
}

// setTPEValue sets TPE configuration values.
func (es *EnvironmentSource) setTPEValue(tpe *TPEConfig, key, value string) error {
	switch key {
	case "num.trials", "numTrials", "max.trials", "maxTrials", "maxtrials":
		if trials, err := strconv.Atoi(value); err == nil {
			tpe.NumTrials = trials
		} else {
			return fmt.Errorf("invalid num trials: %s", value)
		}
	case "max.iterations", "maxIterations", "maxiterations":
		// TPE doesn't have a MaxIterations field, ignore for compatibility
		return nil
	case "num.startup.trials", "numStartupTrials":
		if trials, err := strconv.Atoi(value); err == nil {
			tpe.NumStartupTrials = trials
		} else {
			return fmt.Errorf("invalid num startup trials: %s", value)
		}
	case "percentile":
		if percentile, err := strconv.ParseFloat(value, 64); err == nil {
			tpe.Percentile = percentile
		} else {
			return fmt.Errorf("invalid percentile: %s", value)
		}
	case "random.seed", "randomSeed":
		if seed, err := strconv.ParseInt(value, 10, 64); err == nil {
			tpe.RandomSeed = seed
		} else {
			return fmt.Errorf("invalid random seed: %s", value)
		}
	default:
		return nil
	}
	return nil
}

// CommandLineSource loads configuration from command line arguments.
type CommandLineSource struct {
	priority int
	args     []string
}

// NewCommandLineSource creates a new command line source.
func NewCommandLineSource(args []string) *CommandLineSource {
	return &CommandLineSource{
		priority: 300, // Highest priority
		args:     args,
	}
}

// NewCommandLineSourceWithPriority creates a new command line source with custom priority.
func NewCommandLineSourceWithPriority(priority int, args []string) *CommandLineSource {
	return &CommandLineSource{
		priority: priority,
		args:     args,
	}
}

// Name returns the name of the command line source.
func (cls *CommandLineSource) Name() string {
	return "command_line"
}

// Priority returns the priority of the command line source.
func (cls *CommandLineSource) Priority() int {
	return cls.priority
}

// Load loads configuration from command line arguments.
func (cls *CommandLineSource) Load(config *Config, paths []string) error {
	// Parse command line arguments
	configArgs := cls.parseConfigArgs()

	// Apply command line overrides
	for key, value := range configArgs {
		es := &EnvironmentSource{} // Reuse environment source logic
		if err := es.setConfigValue(config, key, value); err != nil {
			return fmt.Errorf("failed to set config value from command line %s=%s: %w", key, value, err)
		}
	}

	return nil
}

// parseConfigArgs parses configuration arguments from command line.
func (cls *CommandLineSource) parseConfigArgs() map[string]string {
	configArgs := make(map[string]string)

	for i, arg := range cls.args {
		// Handle --config-key=value format
		if strings.HasPrefix(arg, "--config.") || strings.HasPrefix(arg, "--config-") {
			parts := strings.SplitN(arg, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimPrefix(parts[0], "--config.")
				key = strings.TrimPrefix(key, "--config-")
				key = strings.ReplaceAll(key, "-", ".")
				configArgs[key] = parts[1]
			} else if i+1 < len(cls.args) && !strings.HasPrefix(cls.args[i+1], "--") {
				// Handle --config-key value format
				key := strings.TrimPrefix(arg, "--config.")
				key = strings.TrimPrefix(key, "--config-")
				key = strings.ReplaceAll(key, "-", ".")
				configArgs[key] = cls.args[i+1]
			}
		}

		// Handle -c key=value format
		if arg == "-c" && i+1 < len(cls.args) {
			parts := strings.SplitN(cls.args[i+1], "=", 2)
			if len(parts) == 2 {
				configArgs[parts[0]] = parts[1]
			}
		}
	}

	return configArgs
}

// MultiSource combines multiple configuration sources.
type MultiSource struct {
	sources []Source
}

// NewMultiSource creates a new multi-source configuration loader.
func NewMultiSource(sources ...Source) *MultiSource {
	return &MultiSource{sources: sources}
}

// Name returns the name of the multi-source.
func (ms *MultiSource) Name() string {
	return "multi_source"
}

// Priority returns the highest priority among all sources.
func (ms *MultiSource) Priority() int {
	maxPriority := 0
	for _, source := range ms.sources {
		if priority := source.Priority(); priority > maxPriority {
			maxPriority = priority
		}
	}
	return maxPriority
}

// Load loads configuration from all sources in priority order.
func (ms *MultiSource) Load(config *Config, paths []string) error {
	// Sort sources by priority (lowest first, so higher priority overrides)
	sources := ms.sortSourcesByPriority()

	// Load from each source
	for _, source := range sources {
		if err := source.Load(config, paths); err != nil {
			return fmt.Errorf("failed to load from source %s: %w", source.Name(), err)
		}
	}

	return nil
}

// sortSourcesByPriority sorts sources by priority (ascending).
func (ms *MultiSource) sortSourcesByPriority() []Source {
	sources := make([]Source, len(ms.sources))
	copy(sources, ms.sources)

	// Simple bubble sort by priority
	for i := 0; i < len(sources)-1; i++ {
		for j := 0; j < len(sources)-i-1; j++ {
			if sources[j].Priority() > sources[j+1].Priority() {
				sources[j], sources[j+1] = sources[j+1], sources[j]
			}
		}
	}

	return sources
}

// AddSource adds a source to the multi-source.
func (ms *MultiSource) AddSource(source Source) {
	ms.sources = append(ms.sources, source)
}

// RemoveSource removes a source from the multi-source.
func (ms *MultiSource) RemoveSource(sourceName string) {
	for i, source := range ms.sources {
		if source.Name() == sourceName {
			ms.sources = append(ms.sources[:i], ms.sources[i+1:]...)
			break
		}
	}
}

// GetSources returns all sources.
func (ms *MultiSource) GetSources() []Source {
	return ms.sources
}

// RemoteSource loads configuration from a remote URL (placeholder for future implementation).
type RemoteSource struct {
	priority int
	url      string
	headers  map[string]string
	timeout  time.Duration
}

// NewRemoteSource creates a new remote source (placeholder).
func NewRemoteSource(url string) *RemoteSource {
	return &RemoteSource{
		priority: 50, // Lower priority than file source
		url:      url,
		headers:  make(map[string]string),
		timeout:  30 * time.Second,
	}
}

// Name returns the name of the remote source.
func (rs *RemoteSource) Name() string {
	return "remote"
}

// Priority returns the priority of the remote source.
func (rs *RemoteSource) Priority() int {
	return rs.priority
}

// Load loads configuration from a remote URL (placeholder implementation).
func (rs *RemoteSource) Load(config *Config, paths []string) error {
	// This would implement HTTP(S) fetching of configuration
	// For now, it's a placeholder
	return fmt.Errorf("failed to fetch remote config from %s: remote source not implemented", rs.url)
}

// Convenience functions

// CreateDefaultSources creates the default set of configuration sources.
func CreateDefaultSources() []Source {
	return []Source{
		NewFileSource(),
		NewEnvironmentSource(),
	}
}

// CreateAllSources creates all available configuration sources.
func CreateAllSources(args []string) []Source {
	return []Source{
		NewFileSource(),
		NewEnvironmentSource(),
		NewCommandLineSource(args),
	}
}

// LoadFromSources loads configuration from multiple sources.
func LoadFromSources(config *Config, sources []Source, paths []string) error {
	multiSource := NewMultiSource(sources...)
	return multiSource.Load(config, paths)
}

// parseDuration parses a duration from string, supporting both duration format and plain numbers (as seconds).
func (es *EnvironmentSource) parseDuration(value string) (time.Duration, error) {
	// First try parsing as standard duration
	if duration, err := time.ParseDuration(value); err == nil {
		return duration, nil
	}

	// If that fails, try parsing as seconds (plain number)
	if seconds, err := strconv.Atoi(value); err == nil {
		return time.Duration(seconds) * time.Second, nil
	}

	// If both fail, try parsing as float seconds
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), nil
	}

	return 0, fmt.Errorf("invalid duration format: %s", value)
}
