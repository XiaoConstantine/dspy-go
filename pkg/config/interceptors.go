package config

import (
	"fmt"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

// InterceptorBuilder provides a fluent interface for building interceptor chains from configuration.
type InterceptorBuilder struct {
	config *InterceptorsConfig
}

// NewInterceptorBuilder creates a new InterceptorBuilder from configuration.
func NewInterceptorBuilder(config *InterceptorsConfig) *InterceptorBuilder {
	return &InterceptorBuilder{config: config}
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
		cache := interceptors.NewMemoryCache() // Default cache
		ttl := 5 * time.Minute                 // Default TTL
		if moduleConfig.Caching.TTL > 0 {
			ttl = moduleConfig.Caching.TTL
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.CachingModuleInterceptor(cache, ttl))
	}

	if moduleConfig.Timeout.Enabled {
		timeout := 30 * time.Second // Default timeout
		if moduleConfig.Timeout.Timeout > 0 {
			timeout = moduleConfig.Timeout.Timeout
		} else if b.config.Global.DefaultTimeout > 0 {
			timeout = b.config.Global.DefaultTimeout
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.TimeoutModuleInterceptor(timeout))
	}

	if moduleConfig.CircuitBreaker.Enabled {
		failureThreshold := 5
		recoveryTimeout := 30 * time.Second
		halfOpenRequests := 3

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
			MaxAttempts: 3,
			Delay:       100 * time.Millisecond,
			Backoff:     2.0,
		}
		if moduleConfig.Retry.MaxRetries > 0 {
			config.MaxAttempts = moduleConfig.Retry.MaxRetries
		}
		if moduleConfig.Retry.InitialBackoff > 0 {
			config.Delay = moduleConfig.Retry.InitialBackoff
		}
		if moduleConfig.Retry.BackoffFactor > 0 {
			config.Backoff = moduleConfig.Retry.BackoffFactor
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.RetryModuleInterceptor(config))
	}

	// Security interceptors
	if moduleConfig.Validation.Enabled {
		config := interceptors.ValidationConfig{
			MaxInputSize:    int(moduleConfig.Validation.MaxInputSize),
			MaxStringLength: int(moduleConfig.Validation.MaxInputSize), // Use same value for simplicity
			RequiredFields:  moduleConfig.Validation.RequiredFields,
			AllowHTML:       !moduleConfig.Validation.StrictMode, // Invert strict mode
		}
		// Add forbidden patterns for strict mode
		if moduleConfig.Validation.StrictMode {
			config.ForbiddenPatterns = []string{"<script", "javascript:", "data:"}
		}
		moduleInterceptors = append(moduleInterceptors, interceptors.ValidationModuleInterceptor(config))
	}

	if moduleConfig.Authorization.Enabled {
		// Create authorization interceptor instance
		authInterceptor := interceptors.NewAuthorizationInterceptor()
		// Note: In a real implementation, you would set policies based on config
		// For now, we'll use default behavior
		moduleInterceptors = append(moduleInterceptors, authInterceptor.ModuleAuthorizationInterceptor())
	}

	if moduleConfig.Sanitization.Enabled {
		// Use the sanitizing interceptor
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
		limit := 60 // requests per minute
		window := 1 * time.Minute

		if agentConfig.RateLimit.RequestsPerMinute > 0 {
			limit = agentConfig.RateLimit.RequestsPerMinute
		}
		if agentConfig.RateLimit.WindowSize > 0 {
			window = agentConfig.RateLimit.WindowSize
		}

		agentInterceptors = append(agentInterceptors, interceptors.RateLimitingAgentInterceptor(limit, window))
	}

	if agentConfig.Timeout.Enabled {
		timeout := 60 * time.Second // Default timeout for agents
		if agentConfig.Timeout.Timeout > 0 {
			timeout = agentConfig.Timeout.Timeout
		} else if b.config.Global.DefaultTimeout > 0 {
			timeout = b.config.Global.DefaultTimeout
		}
		agentInterceptors = append(agentInterceptors, interceptors.TimeoutAgentInterceptor(timeout))
	}

	// Security interceptors
	if agentConfig.Authorization.Enabled {
		authInterceptor := interceptors.NewAuthorizationInterceptor()
		agentInterceptors = append(agentInterceptors, authInterceptor.AgentAuthorizationInterceptor())
	}

	// Note: Audit functionality would be implemented as a logging interceptor variant
	// For now, we'll use the standard logging interceptor if audit is enabled
	if agentConfig.Audit.Enabled {
		agentInterceptors = append(agentInterceptors, interceptors.LoggingAgentInterceptor())
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
		cache := interceptors.NewMemoryCache() // Default cache
		ttl := 5 * time.Minute                 // Default TTL
		if toolConfig.Caching.TTL > 0 {
			ttl = toolConfig.Caching.TTL
		}
		toolInterceptors = append(toolInterceptors, interceptors.CachingToolInterceptor(cache, ttl))
	}

	if toolConfig.Timeout.Enabled {
		timeout := 30 * time.Second // Default timeout
		if toolConfig.Timeout.Timeout > 0 {
			timeout = toolConfig.Timeout.Timeout
		} else if b.config.Global.DefaultTimeout > 0 {
			timeout = b.config.Global.DefaultTimeout
		}
		toolInterceptors = append(toolInterceptors, interceptors.TimeoutToolInterceptor(timeout))
	}

	// Security interceptors
	if toolConfig.Validation.Enabled {
		config := interceptors.ValidationConfig{
			MaxInputSize:    int(toolConfig.Validation.MaxInputSize),
			MaxStringLength: int(toolConfig.Validation.MaxInputSize),
			RequiredFields:  toolConfig.Validation.RequiredFields,
			AllowHTML:       !toolConfig.Validation.StrictMode,
		}
		if toolConfig.Validation.StrictMode {
			config.ForbiddenPatterns = []string{"<script", "javascript:", "data:"}
		}
		toolInterceptors = append(toolInterceptors, interceptors.ValidationToolInterceptor(config))
	}

	if toolConfig.Authorization.Enabled {
		authInterceptor := interceptors.NewAuthorizationInterceptor()
		toolInterceptors = append(toolInterceptors, authInterceptor.ToolAuthorizationInterceptor())
	}

	if toolConfig.Sanitization.Enabled {
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
			DefaultTimeout:     30 * time.Second,
			MaxChainLength:     10,
			MonitorPerformance: true,
		},
		Module: ModuleInterceptorsConfig{
			Logging:    InterceptorToggle{Enabled: true},
			Metrics:    InterceptorToggle{Enabled: true},
			Tracing:    InterceptorToggle{Enabled: true},
			Caching:    CachingInterceptorConfig{Enabled: true, TTL: 5 * time.Minute, Type: "memory"},
			Timeout:    TimeoutInterceptorConfig{Enabled: true, Timeout: 30 * time.Second},
			Validation: ValidationInterceptorConfig{Enabled: true, StrictMode: false},
		},
		Agent: AgentInterceptorsConfig{
			Logging:   InterceptorToggle{Enabled: true},
			Metrics:   InterceptorToggle{Enabled: true},
			Tracing:   InterceptorToggle{Enabled: true},
			RateLimit: RateLimitInterceptorConfig{Enabled: true, RequestsPerMinute: 60, BurstSize: 10},
			Timeout:   TimeoutInterceptorConfig{Enabled: true, Timeout: 60 * time.Second},
		},
		Tool: ToolInterceptorsConfig{
			Logging:    InterceptorToggle{Enabled: true},
			Metrics:    InterceptorToggle{Enabled: true},
			Tracing:    InterceptorToggle{Enabled: true},
			Caching:    CachingInterceptorConfig{Enabled: true, TTL: 5 * time.Minute, Type: "memory"},
			Timeout:    TimeoutInterceptorConfig{Enabled: true, Timeout: 30 * time.Second},
			Validation: ValidationInterceptorConfig{Enabled: true, StrictMode: false},
		},
	}
}

// SetupSecurityInterceptors creates an interceptor configuration focused on security.
func SetupSecurityInterceptors() *InterceptorsConfig {
	return &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:            true,
			DefaultTimeout:     30 * time.Second,
			MaxChainLength:     15,
			MonitorPerformance: true,
		},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 30 * time.Second},
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				MaxInputSize:        1024 * 1024, // 1MB
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
			Timeout:   TimeoutInterceptorConfig{Enabled: true, Timeout: 60 * time.Second},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:     true,
				RequireAuth: true,
			},
			Audit: AuditInterceptorConfig{
				Enabled:       true,
				LogLevel:      "INFO",
				IncludeInput:  true,
				IncludeOutput: true,
			},
		},
		Tool: ToolInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 30 * time.Second},
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				MaxInputSize:        512 * 1024, // 512KB
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
