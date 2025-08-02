package config

import (
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

// Default values for interceptor configurations.
const (
	// Default timeouts.
	DefaultModuleTimeout = 30 * time.Second
	DefaultAgentTimeout  = 60 * time.Second
	DefaultToolTimeout   = 30 * time.Second

	// Default cache TTL.
	DefaultCacheTTL = 5 * time.Minute

	// Default circuit breaker settings.
	DefaultFailureThreshold = 5
	DefaultRecoveryTimeout  = 30 * time.Second
	DefaultHalfOpenRequests = 3

	// Default retry settings.
	DefaultMaxRetries     = 3
	DefaultInitialBackoff = 100 * time.Millisecond
	DefaultMaxBackoff     = 10 * time.Second
	DefaultBackoffFactor  = 2.0

	// Default rate limiting.
	DefaultRequestsPerMinute = 60
	DefaultBurstSize         = 10
	DefaultRateWindow        = 1 * time.Minute

	// Default validation settings.
	DefaultMaxStringLength = 1000
	DefaultMaxInputSize    = 1024 * 1024 // 1MB

	// Default sanitization settings.
	DefaultSanitizationMaxLength = 10240 // 10KB

	// Default audit settings.
	DefaultAuditLogLevel = "INFO"
)

// InterceptorBuilder provides a fluent interface for building interceptor chains from configuration.
type InterceptorBuilder struct {
	config *InterceptorsConfig
}

// NewInterceptorBuilder creates a new InterceptorBuilder from configuration.
func NewInterceptorBuilder(config *InterceptorsConfig) *InterceptorBuilder {
	return &InterceptorBuilder{config: config}
}

// createCache creates a cache instance based on the configuration.
func (b *InterceptorBuilder) createCache(config CachingInterceptorConfig) (interceptors.Cache, error) {
	switch config.Type {
	case "memory", "":
		// Default to memory cache
		// TODO: Apply MaxSize configuration when cache interface supports it
		// Currently MaxSize field in CachingInterceptorConfig is ignored as MemoryCache doesn't support size limits
		// Users should be aware that cache size limiting is not yet implemented
		return interceptors.NewMemoryCache(), nil
	case "sqlite":
		// Return error instead of silent fallback to prevent security issues
		return nil, fmt.Errorf("sqlite cache is not yet implemented")
	default:
		return nil, fmt.Errorf("unsupported cache type: %s", config.Type)
	}
}

// 3. Default timeout for the component type.
func (b *InterceptorBuilder) resolveTimeout(specificTimeout, defaultTimeout time.Duration) time.Duration {
	if specificTimeout > 0 {
		return specificTimeout
	}
	if b.config.Global.DefaultTimeout > 0 {
		return b.config.Global.DefaultTimeout
	}
	return defaultTimeout
}

// BuildModuleInterceptors builds a chain of module interceptors from configuration.
func (b *InterceptorBuilder) BuildModuleInterceptors() ([]core.ModuleInterceptor, error) {
	if b.config == nil || !b.config.Global.Enabled {
		return nil, nil
	}

	var moduleInterceptors []core.ModuleInterceptor
	moduleConfig := b.config.Module

	// Standard interceptors
	if moduleConfig.Logging.Enabled {
		moduleInterceptors = append(moduleInterceptors, interceptors.LoggingModuleInterceptor())
	}

	if moduleConfig.Metrics.Enabled {
		moduleInterceptors = append(moduleInterceptors, interceptors.MetricsModuleInterceptor())
	}

	if moduleConfig.Tracing.Enabled {
		moduleInterceptors = append(moduleInterceptors, interceptors.TracingModuleInterceptor())
	}

	// Performance interceptors
	if moduleConfig.Caching.Enabled {
		cache, err := b.createCache(moduleConfig.Caching)
		if err != nil {
			return nil, fmt.Errorf("failed to create cache: %w", err)
		}
		ttl := DefaultCacheTTL
		if moduleConfig.Caching.TTL > 0 {
			ttl = moduleConfig.Caching.TTL
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.CachingModuleInterceptor(cache, ttl))
	}

	if moduleConfig.Timeout.Enabled {
		timeout := b.resolveTimeout(moduleConfig.Timeout.Timeout, DefaultModuleTimeout)
		moduleInterceptors = append(moduleInterceptors, interceptors.TimeoutModuleInterceptor(timeout))
	}

	if moduleConfig.CircuitBreaker.Enabled {
		failureThreshold := DefaultFailureThreshold
		recoveryTimeout := DefaultRecoveryTimeout
		halfOpenRequests := DefaultHalfOpenRequests

		if moduleConfig.CircuitBreaker.FailureThreshold > 0 {
			failureThreshold = moduleConfig.CircuitBreaker.FailureThreshold
		}
		if moduleConfig.CircuitBreaker.RecoveryTimeout > 0 {
			recoveryTimeout = moduleConfig.CircuitBreaker.RecoveryTimeout
		}
		if moduleConfig.CircuitBreaker.HalfOpenRequests > 0 {
			halfOpenRequests = moduleConfig.CircuitBreaker.HalfOpenRequests
		}

		cb := interceptors.NewCircuitBreaker(failureThreshold, recoveryTimeout, halfOpenRequests)
		moduleInterceptors = append(moduleInterceptors, interceptors.CircuitBreakerModuleInterceptor(cb))
	}

	if moduleConfig.Retry.Enabled {
		config := interceptors.RetryConfig{
			MaxAttempts: DefaultMaxRetries,
			Delay:       DefaultInitialBackoff,
			MaxBackoff:  DefaultMaxBackoff,
			Backoff:     DefaultBackoffFactor,
		}
		if moduleConfig.Retry.MaxRetries > 0 {
			config.MaxAttempts = moduleConfig.Retry.MaxRetries
		}
		if moduleConfig.Retry.InitialBackoff > 0 {
			config.Delay = moduleConfig.Retry.InitialBackoff
		}
		if moduleConfig.Retry.MaxBackoff > 0 {
			config.MaxBackoff = moduleConfig.Retry.MaxBackoff
		}
		if moduleConfig.Retry.BackoffFactor > 0 {
			config.Backoff = moduleConfig.Retry.BackoffFactor
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.RetryModuleInterceptor(config))
	}

	// Security interceptors
	if moduleConfig.Validation.Enabled {
		// Start with the secure default configuration to ensure all security patterns are included
		config := interceptors.DefaultValidationConfig()

		// Override defaults with values from the user's configuration
		if moduleConfig.Validation.MaxInputSize > 0 {
			config.MaxInputSize = int(moduleConfig.Validation.MaxInputSize)
		}
		if moduleConfig.Validation.MaxStringLength > 0 {
			config.MaxStringLength = moduleConfig.Validation.MaxStringLength
		}
		if len(moduleConfig.Validation.RequiredFields) > 0 {
			config.RequiredFields = moduleConfig.Validation.RequiredFields
		}
		config.AllowHTML = !moduleConfig.Validation.StrictMode

		// In strict mode, ensure HTML is disallowed. The default patterns are already strict.
		if moduleConfig.Validation.StrictMode {
			config.AllowHTML = false
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.ValidationModuleInterceptor(config))
	}

	if moduleConfig.Authorization.Enabled {
		// Create authorization interceptor instance with configuration
		authInterceptor := interceptors.NewAuthorizationInterceptor()

		// Configure authorization policy based on config
		policy := interceptors.AuthorizationPolicy{
			RequiredRoles:  moduleConfig.Authorization.AllowedRoles,
			RequiredScopes: moduleConfig.Authorization.RequiredScopes,
			RequireAuth:    moduleConfig.Authorization.RequireAuth,
			CustomRules:    moduleConfig.Authorization.CustomRules,
		}

		// Apply policy to all modules (using "*" as wildcard)
		// TODO: In future, support per-module policies from config
		authInterceptor.SetPolicy("*", policy)

		moduleInterceptors = append(moduleInterceptors, authInterceptor.ModuleAuthorizationInterceptor())
	}

	if moduleConfig.Sanitization.Enabled {
		// TODO: Current SanitizingModuleInterceptor doesn't accept configuration.
		// Configuration values (RemoveHTML, RemoveSQL, RemoveScript, CustomPatterns, MaxStringLength)
		// will be used when the interceptor is enhanced to accept SanitizationConfig
		moduleInterceptors = append(moduleInterceptors, interceptors.SanitizingModuleInterceptor())
	}

	// Validate chain length
	if b.config.Global.MaxChainLength > 0 && len(moduleInterceptors) > b.config.Global.MaxChainLength {
		return nil, fmt.Errorf("module interceptor chain length (%d) exceeds maximum allowed (%d)",
			len(moduleInterceptors), b.config.Global.MaxChainLength)
	}

	return moduleInterceptors, nil
}

// BuildAgentInterceptors builds a chain of agent interceptors from configuration.
func (b *InterceptorBuilder) BuildAgentInterceptors() ([]core.AgentInterceptor, error) {
	if b.config == nil || !b.config.Global.Enabled {
		return nil, nil
	}

	var agentInterceptors []core.AgentInterceptor
	agentConfig := b.config.Agent

	// Standard interceptors
	if agentConfig.Logging.Enabled {
		agentInterceptors = append(agentInterceptors, interceptors.LoggingAgentInterceptor())
	}

	if agentConfig.Metrics.Enabled {
		agentInterceptors = append(agentInterceptors, interceptors.MetricsAgentInterceptor())
	}

	if agentConfig.Tracing.Enabled {
		agentInterceptors = append(agentInterceptors, interceptors.TracingAgentInterceptor())
	}

	// Performance interceptors
	if agentConfig.RateLimit.Enabled {
		limit := DefaultRequestsPerMinute
		window := DefaultRateWindow

		if agentConfig.RateLimit.RequestsPerMinute > 0 {
			limit = agentConfig.RateLimit.RequestsPerMinute
		}
		if agentConfig.RateLimit.WindowSize > 0 {
			window = agentConfig.RateLimit.WindowSize
		}
		// TODO: BurstSize configuration is not yet supported by RateLimitingAgentInterceptor
		// The current implementation only accepts limit and window parameters
		// if agentConfig.RateLimit.BurstSize > 0 { burstSize = agentConfig.RateLimit.BurstSize }

		agentInterceptors = append(agentInterceptors, interceptors.RateLimitingAgentInterceptor(limit, window))
	}

	if agentConfig.Timeout.Enabled {
		timeout := b.resolveTimeout(agentConfig.Timeout.Timeout, DefaultAgentTimeout)
		agentInterceptors = append(agentInterceptors, interceptors.TimeoutAgentInterceptor(timeout))
	}

	// Security interceptors
	if agentConfig.Authorization.Enabled {
		// Create authorization interceptor instance with configuration
		authInterceptor := interceptors.NewAuthorizationInterceptor()

		// Configure authorization policy based on config
		policy := interceptors.AuthorizationPolicy{
			RequiredRoles:  agentConfig.Authorization.AllowedRoles,
			RequiredScopes: agentConfig.Authorization.RequiredScopes,
			RequireAuth:    agentConfig.Authorization.RequireAuth,
			CustomRules:    agentConfig.Authorization.CustomRules,
		}

		// Apply policy to all agents (using "*" as wildcard)
		authInterceptor.SetPolicy("*", policy)

		agentInterceptors = append(agentInterceptors, authInterceptor.AgentAuthorizationInterceptor())
	}

	if agentConfig.Audit.Enabled {
		// TODO: Audit interceptor is not yet implemented in the interceptors package
		// Configuration values (LogLevel, IncludeInput, IncludeOutput, AuditPath)
		// will be used when the audit interceptor is implemented
		return nil, fmt.Errorf("audit interceptor is not yet implemented")
	}

	// Validate chain length
	if b.config.Global.MaxChainLength > 0 && len(agentInterceptors) > b.config.Global.MaxChainLength {
		return nil, fmt.Errorf("agent interceptor chain length (%d) exceeds maximum allowed (%d)",
			len(agentInterceptors), b.config.Global.MaxChainLength)
	}

	return agentInterceptors, nil
}

// BuildToolInterceptors builds a chain of tool interceptors from configuration.
func (b *InterceptorBuilder) BuildToolInterceptors() ([]core.ToolInterceptor, error) {
	if b.config == nil || !b.config.Global.Enabled {
		return nil, nil
	}

	var toolInterceptors []core.ToolInterceptor
	toolConfig := b.config.Tool

	// Standard interceptors
	if toolConfig.Logging.Enabled {
		toolInterceptors = append(toolInterceptors, interceptors.LoggingToolInterceptor())
	}

	if toolConfig.Metrics.Enabled {
		toolInterceptors = append(toolInterceptors, interceptors.MetricsToolInterceptor())
	}

	if toolConfig.Tracing.Enabled {
		toolInterceptors = append(toolInterceptors, interceptors.TracingToolInterceptor())
	}

	// Performance interceptors
	if toolConfig.Caching.Enabled {
		cache, err := b.createCache(toolConfig.Caching)
		if err != nil {
			return nil, fmt.Errorf("failed to create cache: %w", err)
		}
		ttl := DefaultCacheTTL
		if toolConfig.Caching.TTL > 0 {
			ttl = toolConfig.Caching.TTL
		}
		toolInterceptors = append(toolInterceptors, interceptors.CachingToolInterceptor(cache, ttl))
	}

	if toolConfig.Timeout.Enabled {
		timeout := b.resolveTimeout(toolConfig.Timeout.Timeout, DefaultToolTimeout)
		toolInterceptors = append(toolInterceptors, interceptors.TimeoutToolInterceptor(timeout))
	}

	// Security interceptors
	if toolConfig.Validation.Enabled {
		// Start with the secure default configuration to ensure all security patterns are included
		config := interceptors.DefaultValidationConfig()

		// Override defaults with values from the user's configuration
		if toolConfig.Validation.MaxInputSize > 0 {
			config.MaxInputSize = int(toolConfig.Validation.MaxInputSize)
		}
		if toolConfig.Validation.MaxStringLength > 0 {
			config.MaxStringLength = toolConfig.Validation.MaxStringLength
		}
		if len(toolConfig.Validation.RequiredFields) > 0 {
			config.RequiredFields = toolConfig.Validation.RequiredFields
		}
		config.AllowHTML = !toolConfig.Validation.StrictMode

		// In strict mode, ensure HTML is disallowed. The default patterns are already strict.
		if toolConfig.Validation.StrictMode {
			config.AllowHTML = false
		}
		toolInterceptors = append(toolInterceptors, interceptors.ValidationToolInterceptor(config))
	}

	if toolConfig.Authorization.Enabled {
		// Create authorization interceptor instance with configuration
		authInterceptor := interceptors.NewAuthorizationInterceptor()

		// Configure authorization policy based on config
		policy := interceptors.AuthorizationPolicy{
			RequiredRoles:  toolConfig.Authorization.AllowedRoles,
			RequiredScopes: toolConfig.Authorization.RequiredScopes,
			RequireAuth:    toolConfig.Authorization.RequireAuth,
			CustomRules:    toolConfig.Authorization.CustomRules,
		}

		// Apply policy to all tools (using "*" as wildcard)
		authInterceptor.SetPolicy("*", policy)

		toolInterceptors = append(toolInterceptors, authInterceptor.ToolAuthorizationInterceptor())
	}

	if toolConfig.Sanitization.Enabled {
		// TODO: Current SanitizingToolInterceptor doesn't accept configuration.
		// Configuration values (RemoveHTML, RemoveSQL, RemoveScript, CustomPatterns, MaxStringLength)
		// will be used when the interceptor is enhanced to accept SanitizationConfig
		toolInterceptors = append(toolInterceptors, interceptors.SanitizingToolInterceptor())
	}

	// Validate chain length
	if b.config.Global.MaxChainLength > 0 && len(toolInterceptors) > b.config.Global.MaxChainLength {
		return nil, fmt.Errorf("tool interceptor chain length (%d) exceeds maximum allowed (%d)",
			len(toolInterceptors), b.config.Global.MaxChainLength)
	}

	return toolInterceptors, nil
}

// InterceptorSet holds all configured interceptor chains.
type InterceptorSet struct {
	ModuleInterceptors []core.ModuleInterceptor
	AgentInterceptors  []core.AgentInterceptor
	ToolInterceptors   []core.ToolInterceptor
}

// BuildAll builds all interceptor chains from configuration.
func (b *InterceptorBuilder) BuildAll() (*InterceptorSet, error) {
	moduleInterceptors, err := b.BuildModuleInterceptors()
	if err != nil {
		return nil, fmt.Errorf("failed to build module interceptors: %w", err)
	}

	agentInterceptors, err := b.BuildAgentInterceptors()
	if err != nil {
		return nil, fmt.Errorf("failed to build agent interceptors: %w", err)
	}

	toolInterceptors, err := b.BuildToolInterceptors()
	if err != nil {
		return nil, fmt.Errorf("failed to build tool interceptors: %w", err)
	}

	return &InterceptorSet{
		ModuleInterceptors: moduleInterceptors,
		AgentInterceptors:  agentInterceptors,
		ToolInterceptors:   toolInterceptors,
	}, nil
}

// SetupStandardInterceptors creates a standard interceptor configuration with common settings.
func SetupStandardInterceptors() *InterceptorsConfig {
	return &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:            true,
			DefaultTimeout:     DefaultModuleTimeout,
			MaxChainLength:     10,
			MonitorPerformance: true,
		},
		Module: ModuleInterceptorsConfig{
			Logging:    InterceptorToggle{Enabled: true},
			Metrics:    InterceptorToggle{Enabled: true},
			Tracing:    InterceptorToggle{Enabled: true},
			Caching:    CachingInterceptorConfig{Enabled: true, TTL: DefaultCacheTTL, Type: "memory"},
			Timeout:    TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultModuleTimeout},
			Validation: ValidationInterceptorConfig{Enabled: true, StrictMode: false},
		},
		Agent: AgentInterceptorsConfig{
			Logging:   InterceptorToggle{Enabled: true},
			Metrics:   InterceptorToggle{Enabled: true},
			Tracing:   InterceptorToggle{Enabled: true},
			RateLimit: RateLimitInterceptorConfig{Enabled: true, RequestsPerMinute: DefaultRequestsPerMinute, BurstSize: DefaultBurstSize},
			Timeout:   TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultAgentTimeout},
		},
		Tool: ToolInterceptorsConfig{
			Logging:    InterceptorToggle{Enabled: true},
			Metrics:    InterceptorToggle{Enabled: true},
			Tracing:    InterceptorToggle{Enabled: true},
			Caching:    CachingInterceptorConfig{Enabled: true, TTL: DefaultCacheTTL, Type: "memory"},
			Timeout:    TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultToolTimeout},
			Validation: ValidationInterceptorConfig{Enabled: true, StrictMode: false},
		},
	}
}

// SetupSecurityInterceptors creates an interceptor configuration focused on security.
func SetupSecurityInterceptors() *InterceptorsConfig {
	return &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:            true,
			DefaultTimeout:     DefaultModuleTimeout,
			MaxChainLength:     15,
			MonitorPerformance: true,
		},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultModuleTimeout},
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				MaxInputSize:        1024 * 1024, // 1MB
				MaxStringLength:     10000,       // 10K characters
				AllowedContentTypes: []string{"text/plain", "application/json"},
			},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:     true,
				RequireAuth: true,
			},
			Sanitization: SanitizationInterceptorConfig{
				Enabled:         true,
				RemoveHTML:      true,
				RemoveSQL:       true,
				RemoveScript:    true,
				MaxStringLength: 10000,
			},
		},
		Agent: AgentInterceptorsConfig{
			Logging:   InterceptorToggle{Enabled: true},
			Tracing:   InterceptorToggle{Enabled: true},
			RateLimit: RateLimitInterceptorConfig{Enabled: true, RequestsPerMinute: 30, BurstSize: 5},
			Timeout:   TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultAgentTimeout},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:     true,
				RequireAuth: true,
			},
			// Audit: Disabled because audit interceptor is not yet implemented
			// Audit: AuditInterceptorConfig{
			//	Enabled:       true,
			//	LogLevel:      DefaultAuditLogLevel,
			//	IncludeInput:  true,
			//	IncludeOutput: true,
			// },
		},
		Tool: ToolInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: DefaultToolTimeout},
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				MaxInputSize:        512 * 1024, // 512KB
				MaxStringLength:     5000,       // 5K characters
				AllowedContentTypes: []string{"text/plain", "application/json"},
			},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:     true,
				RequireAuth: true,
			},
			Sanitization: SanitizationInterceptorConfig{
				Enabled:         true,
				RemoveHTML:      true,
				RemoveSQL:       true,
				RemoveScript:    true,
				MaxStringLength: 5000,
			},
		},
	}
}

// SetupPerformanceInterceptors creates an interceptor configuration focused on performance.
func SetupPerformanceInterceptors() *InterceptorsConfig {
	return &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:            true,
			DefaultTimeout:     10 * time.Second,
			MaxChainLength:     8,
			MonitorPerformance: true,
		},
		Module: ModuleInterceptorsConfig{
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Caching: CachingInterceptorConfig{
				Enabled: true,
				TTL:     10 * time.Minute,
				MaxSize: 100 * 1024 * 1024, // 100MB
				Type:    "memory",
			},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 10 * time.Second},
			CircuitBreaker: CircuitBreakerInterceptorConfig{
				Enabled:          true,
				FailureThreshold: 3,
				RecoveryTimeout:  10 * time.Second,
				HalfOpenRequests: 2,
			},
			Retry: RetryInterceptorConfig{
				Enabled:        true,
				MaxRetries:     2,
				InitialBackoff: 50 * time.Millisecond,
				MaxBackoff:     2 * time.Second,
				BackoffFactor:  1.5,
			},
		},
		Agent: AgentInterceptorsConfig{
			Metrics:   InterceptorToggle{Enabled: true},
			Tracing:   InterceptorToggle{Enabled: true},
			RateLimit: RateLimitInterceptorConfig{Enabled: true, RequestsPerMinute: 120, BurstSize: 20},
			Timeout:   TimeoutInterceptorConfig{Enabled: true, Timeout: 30 * time.Second},
		},
		Tool: ToolInterceptorsConfig{
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Caching: CachingInterceptorConfig{
				Enabled: true,
				TTL:     15 * time.Minute,
				MaxSize: 50 * 1024 * 1024, // 50MB
				Type:    "memory",
			},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 10 * time.Second},
		},
	}
}

// SetupInterceptorsFromConfig creates interceptor chains from a loaded configuration.
func SetupInterceptorsFromConfig(config *Config) (*InterceptorSet, error) {
	if config == nil || !config.Interceptors.Global.Enabled {
		return &InterceptorSet{}, nil
	}

	builder := NewInterceptorBuilder(&config.Interceptors)
	return builder.BuildAll()
}
