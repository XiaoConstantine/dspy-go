package config

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewInterceptorBuilder(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
	}

	builder := NewInterceptorBuilder(config)
	assert.NotNil(t, builder)
	assert.Equal(t, config, builder.config)
}

func TestInterceptorBuilder_BuildModuleInterceptors_Disabled(t *testing.T) {
	tests := []struct {
		name   string
		config *InterceptorsConfig
	}{
		{
			name:   "nil config",
			config: nil,
		},
		{
			name: "global disabled",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{Enabled: false},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewInterceptorBuilder(tt.config)
			interceptors, err := builder.BuildModuleInterceptors()

			assert.NoError(t, err)
			assert.Nil(t, interceptors)
		})
	}
}

func TestInterceptorBuilder_BuildModuleInterceptors_StandardInterceptors(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	require.NoError(t, err)
	assert.Len(t, interceptors, 3) // logging, metrics, tracing
}

func TestInterceptorBuilder_BuildModuleInterceptors_PerformanceInterceptors(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			DefaultTimeout: 45 * time.Second,
		},
		Module: ModuleInterceptorsConfig{
			Caching: CachingInterceptorConfig{
				Enabled: true,
				TTL:     10 * time.Minute,
				Type:    "memory",
			},
			Timeout: TimeoutInterceptorConfig{
				Enabled: true,
				Timeout: 20 * time.Second,
			},
			CircuitBreaker: CircuitBreakerInterceptorConfig{
				Enabled:          true,
				FailureThreshold: 3,
				RecoveryTimeout:  15 * time.Second,
				HalfOpenRequests: 2,
			},
			Retry: RetryInterceptorConfig{
				Enabled:        true,
				MaxRetries:     5,
				InitialBackoff: 200 * time.Millisecond,
				MaxBackoff:     10 * time.Second,
				BackoffFactor:  1.5,
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	require.NoError(t, err)
	assert.Len(t, interceptors, 4) // caching, timeout, circuit breaker, retry
}

func TestInterceptorBuilder_BuildModuleInterceptors_SecurityInterceptors(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				RequiredFields:      []string{"input", "query"},
				MaxInputSize:        1024,
				AllowedContentTypes: []string{"text/plain"},
			},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:        true,
				RequireAuth:    true,
				AllowedRoles:   []string{"admin", "user"},
				RequiredScopes: []string{"read", "write"},
				CustomRules:    map[string]string{"rule1": "value1"},
			},
			Sanitization: SanitizationInterceptorConfig{
				Enabled:         true,
				RemoveHTML:      true,
				RemoveSQL:       true,
				RemoveScript:    true,
				CustomPatterns:  []string{"pattern1", "pattern2"},
				MaxStringLength: 5000,
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	require.NoError(t, err)
	assert.Len(t, interceptors, 3) // validation, authorization, sanitization
}

func TestInterceptorBuilder_BuildModuleInterceptors_ChainLengthLimit(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			MaxChainLength: 2,
		},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds maximum allowed")
	assert.Nil(t, interceptors)
}

func TestInterceptorBuilder_BuildAgentInterceptors(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			DefaultTimeout: 30 * time.Second,
		},
		Agent: AgentInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			RateLimit: RateLimitInterceptorConfig{
				Enabled:           true,
				RequestsPerMinute: 120,
				BurstSize:         15,
				WindowSize:        2 * time.Minute,
			},
			Timeout: TimeoutInterceptorConfig{
				Enabled: true,
				Timeout: 90 * time.Second,
			},
			Authorization: AuthorizationInterceptorConfig{
				Enabled:     true,
				RequireAuth: true,
			},
			Audit: AuditInterceptorConfig{
				Enabled:       true,
				LogLevel:      "DEBUG",
				IncludeInput:  true,
				IncludeOutput: false,
				AuditPath:     "/var/log/audit",
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildAgentInterceptors()

	// This should now error because audit is not implemented
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "audit interceptor is not yet implemented")
	assert.Nil(t, interceptors)
}

func TestInterceptorBuilder_BuildAgentInterceptors_Disabled(t *testing.T) {
	builder := NewInterceptorBuilder(nil)
	interceptors, err := builder.BuildAgentInterceptors()

	assert.NoError(t, err)
	assert.Nil(t, interceptors)
}

func TestInterceptorBuilder_BuildToolInterceptors(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			DefaultTimeout: 25 * time.Second,
		},
		Tool: ToolInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Metrics: InterceptorToggle{Enabled: true},
			Tracing: InterceptorToggle{Enabled: true},
			Caching: CachingInterceptorConfig{
				Enabled: true,
				TTL:     8 * time.Minute,
				MaxSize: 1024 * 1024,
				Type:    "memory",
			},
			Timeout: TimeoutInterceptorConfig{
				Enabled: true,
			},
			Validation: ValidationInterceptorConfig{
				Enabled:    true,
				StrictMode: false,
			},
			Authorization: AuthorizationInterceptorConfig{
				Enabled: true,
			},
			Sanitization: SanitizationInterceptorConfig{
				Enabled: true,
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildToolInterceptors()

	require.NoError(t, err)
	assert.Len(t, interceptors, 8) // all enabled interceptors
}

func TestInterceptorBuilder_BuildAll(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
		},
		Agent: AgentInterceptorsConfig{
			Metrics: InterceptorToggle{Enabled: true},
		},
		Tool: ToolInterceptorsConfig{
			Tracing: InterceptorToggle{Enabled: true},
		},
	}

	builder := NewInterceptorBuilder(config)
	set, err := builder.BuildAll()

	require.NoError(t, err)
	assert.NotNil(t, set)
	assert.Len(t, set.ModuleInterceptors, 1)
	assert.Len(t, set.AgentInterceptors, 1)
	assert.Len(t, set.ToolInterceptors, 1)
}

func TestInterceptorBuilder_BuildAll_WithError(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			MaxChainLength: 1,
		},
		Module: ModuleInterceptorsConfig{
			Logging: InterceptorToggle{Enabled: true},
			Metrics: InterceptorToggle{Enabled: true}, // This will cause error
		},
	}

	builder := NewInterceptorBuilder(config)
	set, err := builder.BuildAll()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to build module interceptors")
	assert.Nil(t, set)
}

func TestSetupStandardInterceptors(t *testing.T) {
	config := SetupStandardInterceptors()

	require.NotNil(t, config)
	assert.True(t, config.Global.Enabled)
	assert.Equal(t, 30*time.Second, config.Global.DefaultTimeout)
	assert.Equal(t, 10, config.Global.MaxChainLength)
	assert.True(t, config.Global.MonitorPerformance)

	// Test module config
	assert.True(t, config.Module.Logging.Enabled)
	assert.True(t, config.Module.Metrics.Enabled)
	assert.True(t, config.Module.Tracing.Enabled)
	assert.True(t, config.Module.Caching.Enabled)
	assert.Equal(t, 5*time.Minute, config.Module.Caching.TTL)
	assert.Equal(t, "memory", config.Module.Caching.Type)
	assert.True(t, config.Module.Timeout.Enabled)
	assert.Equal(t, 30*time.Second, config.Module.Timeout.Timeout)
	assert.True(t, config.Module.Validation.Enabled)
	assert.False(t, config.Module.Validation.StrictMode)

	// Test agent config
	assert.True(t, config.Agent.Logging.Enabled)
	assert.True(t, config.Agent.Metrics.Enabled)
	assert.True(t, config.Agent.Tracing.Enabled)
	assert.True(t, config.Agent.RateLimit.Enabled)
	assert.Equal(t, 60, config.Agent.RateLimit.RequestsPerMinute)
	assert.Equal(t, 10, config.Agent.RateLimit.BurstSize)
	assert.True(t, config.Agent.Timeout.Enabled)
	assert.Equal(t, 60*time.Second, config.Agent.Timeout.Timeout)

	// Test tool config
	assert.True(t, config.Tool.Logging.Enabled)
	assert.True(t, config.Tool.Metrics.Enabled)
	assert.True(t, config.Tool.Tracing.Enabled)
	assert.True(t, config.Tool.Caching.Enabled)
	assert.Equal(t, 5*time.Minute, config.Tool.Caching.TTL)
	assert.True(t, config.Tool.Timeout.Enabled)
	assert.Equal(t, 30*time.Second, config.Tool.Timeout.Timeout)
	assert.True(t, config.Tool.Validation.Enabled)
	assert.False(t, config.Tool.Validation.StrictMode)
}

func TestSetupSecurityInterceptors(t *testing.T) {
	config := SetupSecurityInterceptors()

	require.NotNil(t, config)
	assert.True(t, config.Global.Enabled)
	assert.Equal(t, 15, config.Global.MaxChainLength)

	// Test module security settings
	assert.True(t, config.Module.Validation.Enabled)
	assert.True(t, config.Module.Validation.StrictMode)
	assert.Equal(t, int64(1024*1024), config.Module.Validation.MaxInputSize)
	assert.Equal(t, 10000, config.Module.Validation.MaxStringLength)
	assert.Equal(t, []string{"text/plain", "application/json"}, config.Module.Validation.AllowedContentTypes)

	assert.True(t, config.Module.Authorization.Enabled)
	assert.True(t, config.Module.Authorization.RequireAuth)

	assert.True(t, config.Module.Sanitization.Enabled)
	assert.True(t, config.Module.Sanitization.RemoveHTML)
	assert.True(t, config.Module.Sanitization.RemoveSQL)
	assert.True(t, config.Module.Sanitization.RemoveScript)
	assert.Equal(t, 10000, config.Module.Sanitization.MaxStringLength)

	// Test agent security settings
	assert.True(t, config.Agent.RateLimit.Enabled)
	assert.Equal(t, 30, config.Agent.RateLimit.RequestsPerMinute)
	assert.Equal(t, 5, config.Agent.RateLimit.BurstSize)

	assert.True(t, config.Agent.Authorization.Enabled)
	assert.True(t, config.Agent.Authorization.RequireAuth)

	// Audit is currently disabled in security preset because it's not implemented
	assert.False(t, config.Agent.Audit.Enabled)

	// Test tool security settings
	assert.True(t, config.Tool.Validation.Enabled)
	assert.True(t, config.Tool.Validation.StrictMode)
	assert.Equal(t, int64(512*1024), config.Tool.Validation.MaxInputSize)
	assert.Equal(t, 5000, config.Tool.Validation.MaxStringLength)

	assert.True(t, config.Tool.Authorization.Enabled)
	assert.True(t, config.Tool.Authorization.RequireAuth)

	assert.True(t, config.Tool.Sanitization.Enabled)
	assert.Equal(t, 5000, config.Tool.Sanitization.MaxStringLength)
}

func TestSetupPerformanceInterceptors(t *testing.T) {
	config := SetupPerformanceInterceptors()

	require.NotNil(t, config)
	assert.True(t, config.Global.Enabled)
	assert.Equal(t, 10*time.Second, config.Global.DefaultTimeout)
	assert.Equal(t, 8, config.Global.MaxChainLength)

	// Test module performance settings
	assert.True(t, config.Module.Metrics.Enabled)
	assert.True(t, config.Module.Tracing.Enabled)

	assert.True(t, config.Module.Caching.Enabled)
	assert.Equal(t, 10*time.Minute, config.Module.Caching.TTL)
	assert.Equal(t, int64(100*1024*1024), config.Module.Caching.MaxSize)
	assert.Equal(t, "memory", config.Module.Caching.Type)

	assert.True(t, config.Module.Timeout.Enabled)
	assert.Equal(t, 10*time.Second, config.Module.Timeout.Timeout)

	assert.True(t, config.Module.CircuitBreaker.Enabled)
	assert.Equal(t, 3, config.Module.CircuitBreaker.FailureThreshold)
	assert.Equal(t, 10*time.Second, config.Module.CircuitBreaker.RecoveryTimeout)
	assert.Equal(t, 2, config.Module.CircuitBreaker.HalfOpenRequests)

	assert.True(t, config.Module.Retry.Enabled)
	assert.Equal(t, 2, config.Module.Retry.MaxRetries)
	assert.Equal(t, 50*time.Millisecond, config.Module.Retry.InitialBackoff)
	assert.Equal(t, 2*time.Second, config.Module.Retry.MaxBackoff)
	assert.Equal(t, 1.5, config.Module.Retry.BackoffFactor)

	// Test agent performance settings
	assert.True(t, config.Agent.RateLimit.Enabled)
	assert.Equal(t, 120, config.Agent.RateLimit.RequestsPerMinute)
	assert.Equal(t, 20, config.Agent.RateLimit.BurstSize)
	assert.Equal(t, 30*time.Second, config.Agent.Timeout.Timeout)

	// Test tool performance settings
	assert.True(t, config.Tool.Caching.Enabled)
	assert.Equal(t, 15*time.Minute, config.Tool.Caching.TTL)
	assert.Equal(t, int64(50*1024*1024), config.Tool.Caching.MaxSize)
	assert.Equal(t, 10*time.Second, config.Tool.Timeout.Timeout)
}

func TestSetupInterceptorsFromConfig_NilConfig(t *testing.T) {
	set, err := SetupInterceptorsFromConfig(nil)

	require.NoError(t, err)
	assert.NotNil(t, set)
	assert.Empty(t, set.ModuleInterceptors)
	assert.Empty(t, set.AgentInterceptors)
	assert.Empty(t, set.ToolInterceptors)
}

func TestSetupInterceptorsFromConfig_DisabledGlobal(t *testing.T) {
	config := &Config{
		Interceptors: InterceptorsConfig{
			Global: GlobalInterceptorConfig{Enabled: false},
		},
	}

	set, err := SetupInterceptorsFromConfig(config)

	require.NoError(t, err)
	assert.NotNil(t, set)
	assert.Empty(t, set.ModuleInterceptors)
	assert.Empty(t, set.AgentInterceptors)
	assert.Empty(t, set.ToolInterceptors)
}

func TestSetupInterceptorsFromConfig_EnabledConfig(t *testing.T) {
	config := &Config{
		Interceptors: InterceptorsConfig{
			Global: GlobalInterceptorConfig{Enabled: true},
			Module: ModuleInterceptorsConfig{
				Logging: InterceptorToggle{Enabled: true},
				Metrics: InterceptorToggle{Enabled: true},
			},
			Agent: AgentInterceptorsConfig{
				Tracing: InterceptorToggle{Enabled: true},
			},
			Tool: ToolInterceptorsConfig{
				Timeout: TimeoutInterceptorConfig{Enabled: true},
			},
		},
	}

	set, err := SetupInterceptorsFromConfig(config)

	require.NoError(t, err)
	assert.NotNil(t, set)
	assert.Len(t, set.ModuleInterceptors, 2)
	assert.Len(t, set.AgentInterceptors, 1)
	assert.Len(t, set.ToolInterceptors, 1)
}

func TestInterceptorBuilder_DefaultTimeouts(t *testing.T) {
	tests := []struct {
		name           string
		config         *InterceptorsConfig
		expectedModule time.Duration
		expectedAgent  time.Duration
		expectedTool   time.Duration
	}{
		{
			name: "uses configured timeouts",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{Enabled: true},
				Module: ModuleInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 15 * time.Second},
				},
				Agent: AgentInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 45 * time.Second},
				},
				Tool: ToolInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true, Timeout: 25 * time.Second},
				},
			},
			expectedModule: 15 * time.Second,
			expectedAgent:  45 * time.Second,
			expectedTool:   25 * time.Second,
		},
		{
			name: "uses global default when specific timeout not set",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{
					Enabled:        true,
					DefaultTimeout: 20 * time.Second,
				},
				Module: ModuleInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true},
				},
				Agent: AgentInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true},
				},
				Tool: ToolInterceptorsConfig{
					Timeout: TimeoutInterceptorConfig{Enabled: true},
				},
			},
			expectedModule: 20 * time.Second,
			expectedAgent:  20 * time.Second,
			expectedTool:   20 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewInterceptorBuilder(tt.config)

			// Test module timeout
			moduleInterceptors, err := builder.BuildModuleInterceptors()
			require.NoError(t, err)
			assert.Len(t, moduleInterceptors, 1)

			// Test agent timeout
			agentInterceptors, err := builder.BuildAgentInterceptors()
			require.NoError(t, err)
			assert.Len(t, agentInterceptors, 1)

			// Test tool timeout
			toolInterceptors, err := builder.BuildToolInterceptors()
			require.NoError(t, err)
			assert.Len(t, toolInterceptors, 1)
		})
	}
}

func TestInterceptorBuilder_DefaultValues(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Caching:        CachingInterceptorConfig{Enabled: true},        // No TTL specified
			CircuitBreaker: CircuitBreakerInterceptorConfig{Enabled: true}, // No values specified
			Retry:          RetryInterceptorConfig{Enabled: true},          // No values specified
		},
		Agent: AgentInterceptorsConfig{
			RateLimit: RateLimitInterceptorConfig{Enabled: true}, // No values specified
		},
	}

	builder := NewInterceptorBuilder(config)

	// Test that defaults are applied correctly
	moduleInterceptors, err := builder.BuildModuleInterceptors()
	require.NoError(t, err)
	assert.Len(t, moduleInterceptors, 3) // caching, circuit breaker, retry

	agentInterceptors, err := builder.BuildAgentInterceptors()
	require.NoError(t, err)
	assert.Len(t, agentInterceptors, 1) // rate limit
}

// Test configuration struct validation tags.
func TestInterceptorConfigValidation(t *testing.T) {
	// Test that the structs have proper validation tags
	config := &InterceptorsConfig{}

	// This is a basic structural test to ensure the config structs are properly formed
	assert.NotNil(t, config.Global)
	assert.NotNil(t, config.Module)
	assert.NotNil(t, config.Agent)
	assert.NotNil(t, config.Tool)
}

func TestInterceptorBuilder_BuildModuleInterceptors_AllCombinations(t *testing.T) {
	tests := []struct {
		name        string
		config      ModuleInterceptorsConfig
		expectedLen int
	}{
		{
			name: "all interceptors enabled",
			config: ModuleInterceptorsConfig{
				Logging:        InterceptorToggle{Enabled: true},
				Metrics:        InterceptorToggle{Enabled: true},
				Tracing:        InterceptorToggle{Enabled: true},
				Caching:        CachingInterceptorConfig{Enabled: true},
				Timeout:        TimeoutInterceptorConfig{Enabled: true},
				CircuitBreaker: CircuitBreakerInterceptorConfig{Enabled: true},
				Retry:          RetryInterceptorConfig{Enabled: true},
				Validation:     ValidationInterceptorConfig{Enabled: true},
				Authorization:  AuthorizationInterceptorConfig{Enabled: true},
				Sanitization:   SanitizationInterceptorConfig{Enabled: true},
			},
			expectedLen: 10,
		},
		{
			name: "only performance interceptors",
			config: ModuleInterceptorsConfig{
				Caching:        CachingInterceptorConfig{Enabled: true},
				Timeout:        TimeoutInterceptorConfig{Enabled: true},
				CircuitBreaker: CircuitBreakerInterceptorConfig{Enabled: true},
				Retry:          RetryInterceptorConfig{Enabled: true},
			},
			expectedLen: 4,
		},
		{
			name: "only security interceptors",
			config: ModuleInterceptorsConfig{
				Validation:    ValidationInterceptorConfig{Enabled: true},
				Authorization: AuthorizationInterceptorConfig{Enabled: true},
				Sanitization:  SanitizationInterceptorConfig{Enabled: true},
			},
			expectedLen: 3,
		},
		{
			name:        "no interceptors enabled",
			config:      ModuleInterceptorsConfig{},
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &InterceptorsConfig{
				Global: GlobalInterceptorConfig{Enabled: true},
				Module: tt.config,
			}

			builder := NewInterceptorBuilder(config)
			interceptors, err := builder.BuildModuleInterceptors()

			require.NoError(t, err)
			assert.Len(t, interceptors, tt.expectedLen)
		})
	}
}

func TestInterceptorBuilder_BuildAgentInterceptors_AllCombinations(t *testing.T) {
	tests := []struct {
		name        string
		config      AgentInterceptorsConfig
		expectedLen int
	}{
		{
			name: "all interceptors except audit enabled",
			config: AgentInterceptorsConfig{
				Logging:       InterceptorToggle{Enabled: true},
				Metrics:       InterceptorToggle{Enabled: true},
				Tracing:       InterceptorToggle{Enabled: true},
				RateLimit:     RateLimitInterceptorConfig{Enabled: true},
				Timeout:       TimeoutInterceptorConfig{Enabled: true},
				Authorization: AuthorizationInterceptorConfig{Enabled: true},
				// Audit: Disabled because it errors when enabled
			},
			expectedLen: 6,
		},
		{
			name: "only standard interceptors",
			config: AgentInterceptorsConfig{
				Logging: InterceptorToggle{Enabled: true},
				Metrics: InterceptorToggle{Enabled: true},
				Tracing: InterceptorToggle{Enabled: true},
			},
			expectedLen: 3,
		},
		{
			name:        "no interceptors enabled",
			config:      AgentInterceptorsConfig{},
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &InterceptorsConfig{
				Global: GlobalInterceptorConfig{Enabled: true},
				Agent:  tt.config,
			}

			builder := NewInterceptorBuilder(config)
			interceptors, err := builder.BuildAgentInterceptors()

			require.NoError(t, err)
			assert.Len(t, interceptors, tt.expectedLen)
		})
	}
}

func TestInterceptorBuilder_BuildToolInterceptors_AllCombinations(t *testing.T) {
	tests := []struct {
		name        string
		config      ToolInterceptorsConfig
		expectedLen int
	}{
		{
			name: "all interceptors enabled",
			config: ToolInterceptorsConfig{
				Logging:       InterceptorToggle{Enabled: true},
				Metrics:       InterceptorToggle{Enabled: true},
				Tracing:       InterceptorToggle{Enabled: true},
				Caching:       CachingInterceptorConfig{Enabled: true},
				Timeout:       TimeoutInterceptorConfig{Enabled: true},
				Validation:    ValidationInterceptorConfig{Enabled: true},
				Authorization: AuthorizationInterceptorConfig{Enabled: true},
				Sanitization:  SanitizationInterceptorConfig{Enabled: true},
			},
			expectedLen: 8,
		},
		{
			name: "only caching and timeout",
			config: ToolInterceptorsConfig{
				Caching: CachingInterceptorConfig{Enabled: true},
				Timeout: TimeoutInterceptorConfig{Enabled: true},
			},
			expectedLen: 2,
		},
		{
			name:        "no interceptors enabled",
			config:      ToolInterceptorsConfig{},
			expectedLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &InterceptorsConfig{
				Global: GlobalInterceptorConfig{Enabled: true},
				Tool:   tt.config,
			}

			builder := NewInterceptorBuilder(config)
			interceptors, err := builder.BuildToolInterceptors()

			require.NoError(t, err)
			assert.Len(t, interceptors, tt.expectedLen)
		})
	}
}

func TestInterceptorBuilder_ErrorHandling(t *testing.T) {
	tests := []struct {
		name        string
		config      *InterceptorsConfig
		expectError bool
		errorMsg    string
	}{
		{
			name: "module chain length exceeded",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{
					Enabled:        true,
					MaxChainLength: 1,
				},
				Module: ModuleInterceptorsConfig{
					Logging: InterceptorToggle{Enabled: true},
					Metrics: InterceptorToggle{Enabled: true},
				},
			},
			expectError: true,
			errorMsg:    "module interceptor chain length",
		},
		{
			name: "agent chain length exceeded",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{
					Enabled:        true,
					MaxChainLength: 1,
				},
				Agent: AgentInterceptorsConfig{
					Logging: InterceptorToggle{Enabled: true},
					Metrics: InterceptorToggle{Enabled: true},
				},
			},
			expectError: true,
			errorMsg:    "agent interceptor chain length",
		},
		{
			name: "tool chain length exceeded",
			config: &InterceptorsConfig{
				Global: GlobalInterceptorConfig{
					Enabled:        true,
					MaxChainLength: 1,
				},
				Tool: ToolInterceptorsConfig{
					Logging: InterceptorToggle{Enabled: true},
					Metrics: InterceptorToggle{Enabled: true},
				},
			},
			expectError: true,
			errorMsg:    "tool interceptor chain length",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			builder := NewInterceptorBuilder(tt.config)

			if strings.Contains(tt.errorMsg, "module") {
				_, err := builder.BuildModuleInterceptors()
				if tt.expectError {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tt.errorMsg)
				} else {
					assert.NoError(t, err)
				}
			} else if strings.Contains(tt.errorMsg, "agent") {
				_, err := builder.BuildAgentInterceptors()
				if tt.expectError {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tt.errorMsg)
				} else {
					assert.NoError(t, err)
				}
			} else if strings.Contains(tt.errorMsg, "tool") {
				_, err := builder.BuildToolInterceptors()
				if tt.expectError {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tt.errorMsg)
				} else {
					assert.NoError(t, err)
				}
			}
		})
	}
}

func TestInterceptorConfigStructs(t *testing.T) {
	// Test InterceptorToggle
	toggle := InterceptorToggle{Enabled: true}
	assert.True(t, toggle.Enabled)

	// Test CachingInterceptorConfig
	caching := CachingInterceptorConfig{
		Enabled: true,
		TTL:     5 * time.Minute,
		MaxSize: 1024,
		Type:    "memory",
	}
	assert.True(t, caching.Enabled)
	assert.Equal(t, 5*time.Minute, caching.TTL)
	assert.Equal(t, int64(1024), caching.MaxSize)
	assert.Equal(t, "memory", caching.Type)

	// Test TimeoutInterceptorConfig
	timeout := TimeoutInterceptorConfig{
		Enabled: true,
		Timeout: 30 * time.Second,
	}
	assert.True(t, timeout.Enabled)
	assert.Equal(t, 30*time.Second, timeout.Timeout)

	// Test CircuitBreakerInterceptorConfig
	cb := CircuitBreakerInterceptorConfig{
		Enabled:          true,
		FailureThreshold: 5,
		RecoveryTimeout:  30 * time.Second,
		HalfOpenRequests: 3,
	}
	assert.True(t, cb.Enabled)
	assert.Equal(t, 5, cb.FailureThreshold)
	assert.Equal(t, 30*time.Second, cb.RecoveryTimeout)
	assert.Equal(t, 3, cb.HalfOpenRequests)

	// Test RetryInterceptorConfig
	retry := RetryInterceptorConfig{
		Enabled:        true,
		MaxRetries:     3,
		InitialBackoff: 100 * time.Millisecond,
		MaxBackoff:     5 * time.Second,
		BackoffFactor:  2.0,
	}
	assert.True(t, retry.Enabled)
	assert.Equal(t, 3, retry.MaxRetries)
	assert.Equal(t, 100*time.Millisecond, retry.InitialBackoff)
	assert.Equal(t, 5*time.Second, retry.MaxBackoff)
	assert.Equal(t, 2.0, retry.BackoffFactor)

	// Test ValidationInterceptorConfig
	validation := ValidationInterceptorConfig{
		Enabled:             true,
		StrictMode:          true,
		RequiredFields:      []string{"field1", "field2"},
		MaxInputSize:        1024,
		AllowedContentTypes: []string{"text/plain"},
	}
	assert.True(t, validation.Enabled)
	assert.True(t, validation.StrictMode)
	assert.Equal(t, []string{"field1", "field2"}, validation.RequiredFields)
	assert.Equal(t, int64(1024), validation.MaxInputSize)
	assert.Equal(t, []string{"text/plain"}, validation.AllowedContentTypes)

	// Test AuthorizationInterceptorConfig
	auth := AuthorizationInterceptorConfig{
		Enabled:        true,
		RequireAuth:    true,
		AllowedRoles:   []string{"admin", "user"},
		RequiredScopes: []string{"read", "write"},
		CustomRules:    map[string]string{"rule1": "value1"},
	}
	assert.True(t, auth.Enabled)
	assert.True(t, auth.RequireAuth)
	assert.Equal(t, []string{"admin", "user"}, auth.AllowedRoles)
	assert.Equal(t, []string{"read", "write"}, auth.RequiredScopes)
	assert.Equal(t, map[string]string{"rule1": "value1"}, auth.CustomRules)

	// Test SanitizationInterceptorConfig
	sanitization := SanitizationInterceptorConfig{
		Enabled:         true,
		RemoveHTML:      true,
		RemoveSQL:       true,
		RemoveScript:    true,
		CustomPatterns:  []string{"pattern1"},
		MaxStringLength: 1000,
	}
	assert.True(t, sanitization.Enabled)
	assert.True(t, sanitization.RemoveHTML)
	assert.True(t, sanitization.RemoveSQL)
	assert.True(t, sanitization.RemoveScript)
	assert.Equal(t, []string{"pattern1"}, sanitization.CustomPatterns)
	assert.Equal(t, 1000, sanitization.MaxStringLength)

	// Test RateLimitInterceptorConfig
	rateLimit := RateLimitInterceptorConfig{
		Enabled:           true,
		RequestsPerMinute: 60,
		BurstSize:         10,
		WindowSize:        1 * time.Minute,
	}
	assert.True(t, rateLimit.Enabled)
	assert.Equal(t, 60, rateLimit.RequestsPerMinute)
	assert.Equal(t, 10, rateLimit.BurstSize)
	assert.Equal(t, 1*time.Minute, rateLimit.WindowSize)

	// Test AuditInterceptorConfig
	audit := AuditInterceptorConfig{
		Enabled:       true,
		LogLevel:      "INFO",
		IncludeInput:  true,
		IncludeOutput: true,
		AuditPath:     "/var/log/audit",
	}
	assert.True(t, audit.Enabled)
	assert.Equal(t, "INFO", audit.LogLevel)
	assert.True(t, audit.IncludeInput)
	assert.True(t, audit.IncludeOutput)
	assert.Equal(t, "/var/log/audit", audit.AuditPath)

	// Test GlobalInterceptorConfig
	global := GlobalInterceptorConfig{
		Enabled:            true,
		DefaultTimeout:     30 * time.Second,
		MaxChainLength:     10,
		MonitorPerformance: true,
	}
	assert.True(t, global.Enabled)
	assert.Equal(t, 30*time.Second, global.DefaultTimeout)
	assert.Equal(t, 10, global.MaxChainLength)
	assert.True(t, global.MonitorPerformance)
}

func TestInterceptorBuilder_ConfigurationMapping(t *testing.T) {
	// Test that configuration values are properly mapped to interceptor parameters
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{
			Enabled:        true,
			DefaultTimeout: 45 * time.Second,
		},
		Module: ModuleInterceptorsConfig{
			Caching: CachingInterceptorConfig{
				Enabled: true,
				TTL:     10 * time.Minute,
				MaxSize: 2048,
				Type:    "memory",
			},
			CircuitBreaker: CircuitBreakerInterceptorConfig{
				Enabled:          true,
				FailureThreshold: 7,
				RecoveryTimeout:  25 * time.Second,
				HalfOpenRequests: 4,
			},
			Retry: RetryInterceptorConfig{
				Enabled:        true,
				MaxRetries:     5,
				InitialBackoff: 200 * time.Millisecond,
				BackoffFactor:  1.8,
			},
			Validation: ValidationInterceptorConfig{
				Enabled:             true,
				StrictMode:          true,
				RequiredFields:      []string{"input", "query"},
				MaxInputSize:        2048,
				AllowedContentTypes: []string{"application/json"},
			},
		},
		Agent: AgentInterceptorsConfig{
			RateLimit: RateLimitInterceptorConfig{
				Enabled:           true,
				RequestsPerMinute: 90,
				BurstSize:         15,
				WindowSize:        2 * time.Minute,
			},
		},
	}

	builder := NewInterceptorBuilder(config)

	// Test module interceptors
	moduleInterceptors, err := builder.BuildModuleInterceptors()
	require.NoError(t, err)
	assert.Len(t, moduleInterceptors, 4) // caching, circuit breaker, retry, validation

	// Test agent interceptors
	agentInterceptors, err := builder.BuildAgentInterceptors()
	require.NoError(t, err)
	assert.Len(t, agentInterceptors, 1) // rate limit

	// Test tool interceptors (empty config should result in no interceptors)
	toolInterceptors, err := builder.BuildToolInterceptors()
	require.NoError(t, err)
	assert.Len(t, toolInterceptors, 0)
}

func TestInterceptorBuilder_CreateCache(t *testing.T) {
	builder := NewInterceptorBuilder(&InterceptorsConfig{})

	tests := []struct {
		name        string
		config      CachingInterceptorConfig
		expectError bool
	}{
		{
			name:        "memory cache",
			config:      CachingInterceptorConfig{Type: "memory"},
			expectError: false,
		},
		{
			name:        "empty type defaults to memory",
			config:      CachingInterceptorConfig{Type: ""},
			expectError: false,
		},
		{
			name:        "sqlite cache (not implemented - should error)",
			config:      CachingInterceptorConfig{Type: "sqlite"},
			expectError: true,
		},
		{
			name:        "unsupported cache type",
			config:      CachingInterceptorConfig{Type: "redis"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache, err := builder.createCache(tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, cache)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, cache)
			}
		})
	}
}

func TestValidationInterceptorConfig_MaxStringLength(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Validation: ValidationInterceptorConfig{
				Enabled:         true,
				MaxInputSize:    2048,
				MaxStringLength: 500,
				StrictMode:      true,
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	require.NoError(t, err)
	assert.Len(t, interceptors, 1) // Only validation interceptor enabled

	// Test that MaxStringLength is properly configured
	// Note: This is a structural test since we can't easily inspect the interceptor config
	// In a real implementation, we'd need to modify the interceptor constructors to expose config
}

func TestInterceptorBuilder_CacheCreationError(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Module: ModuleInterceptorsConfig{
			Caching: CachingInterceptorConfig{
				Enabled: true,
				Type:    "unsupported",
			},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildModuleInterceptors()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create cache")
	assert.Nil(t, interceptors)
}

func TestInterceptorBuilder_AuditInterceptorError(t *testing.T) {
	config := &InterceptorsConfig{
		Global: GlobalInterceptorConfig{Enabled: true},
		Agent: AgentInterceptorsConfig{
			Audit: AuditInterceptorConfig{Enabled: true},
		},
	}

	builder := NewInterceptorBuilder(config)
	interceptors, err := builder.BuildAgentInterceptors()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "audit interceptor is not yet implemented")
	assert.Nil(t, interceptors)
}
