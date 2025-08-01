package interceptors

import (
	"context"
	"errors"
	"fmt"
	"html"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// RateLimiter tracks request rates per key (agent, tool, etc.).
type RateLimiter struct {
	mu         sync.RWMutex
	requests   map[string][]time.Time
	lastAccess map[string]time.Time // Track last access time for cleanup
	limit      int
	window     time.Duration
}

// NewRateLimiter creates a new rate limiter with specified limit and time window.
func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	rl := &RateLimiter{
		requests:   make(map[string][]time.Time),
		lastAccess: make(map[string]time.Time),
		limit:      limit,
		window:     window,
	}

	// Start cleanup goroutine to prevent memory leaks
	go rl.periodicCleanup()

	return rl
}

// Allow checks if a request should be allowed for the given key.
func (rl *RateLimiter) Allow(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-rl.window)

	// Update last access time for cleanup
	rl.lastAccess[key] = now

	// Get existing requests for this key
	requests := rl.requests[key]

	// Remove expired requests
	validRequests := make([]time.Time, 0, len(requests))
	for _, reqTime := range requests {
		if reqTime.After(cutoff) {
			validRequests = append(validRequests, reqTime)
		}
	}

	// Check if we're under the limit
	if len(validRequests) >= rl.limit {
		rl.requests[key] = validRequests
		return false
	}

	// Add current request
	validRequests = append(validRequests, now)
	rl.requests[key] = validRequests
	return true
}

// ValidationConfig holds configuration for input validation.
type ValidationConfig struct {
	MaxInputSize     int      // Maximum size of input data in bytes
	MaxStringLength  int      // Maximum length of string values
	ForbiddenPatterns []string // Regex patterns that should not be present
	RequiredFields   []string // Fields that must be present
	AllowHTML        bool     // Whether HTML is allowed in strings
}

// DefaultValidationConfig returns a secure default validation configuration.
func DefaultValidationConfig() ValidationConfig {
	return ValidationConfig{
		MaxInputSize:    10 * 1024 * 1024, // 10MB
		MaxStringLength: 100000,           // 100KB per string
		ForbiddenPatterns: []string{
			`(?i)<script[^>]*>.*?</script>`,          // Script tags (case insensitive)
			`(?i)javascript:`,                        // JavaScript URLs
			`(?i)data:text/html`,                     // HTML data URLs
			`(?i)on\w+\s*=`,                         // Event handlers
			`(?i)eval\s*\(`,                         // eval() calls
			`(?i)exec\s*\(`,                         // exec() calls
			`(?i)system\s*\(`,                       // system() calls
			`(?i)cmd\s*\(`,                          // cmd() calls
			`(?i)shell_exec\s*\(`,                   // shell_exec() calls
			`\$\{.*\}`,                              // Template injection
			`<%.*%>`,                                // Template tags
			`\{\{.*\}\}`,                            // Mustache/Handlebars templates
			`(?i)<iframe[^>]*>`,                     // iframes
			`(?i)<object[^>]*>`,                     // object tags
			`(?i)<embed[^>]*>`,                      // embed tags
			`(?i)<form[^>]*>`,                       // form tags
			`(?i)vbscript:`,                         // VBScript URLs
			`(?i)expression\s*\(`,                   // CSS expressions
			`(?i)import\s+['\"]`,                    // Import statements
			`\.\.\/`,                                // Path traversal
			`\\\\`,                                  // Windows path traversal
			`(?i)union\s+select`,                    // SQL injection
			`(?i)drop\s+table`,                      // SQL injection
			`(?i)delete\s+from`,                     // SQL injection
			`--\s*$`,                                // SQL comments
			`(?i)/\*.*\*/`,                          // SQL block comments
		},
		RequiredFields: []string{},
		AllowHTML:      false,
	}
}

// RateLimitingAgentInterceptor creates an interceptor that enforces rate limiting on agent execution.
// It limits the number of agent executions per time window per agent ID.
func RateLimitingAgentInterceptor(limit int, window time.Duration) core.AgentInterceptor {
	limiter := NewRateLimiter(limit, window)

	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		// Use agent ID as the rate limiting key
		key := fmt.Sprintf("agent:%s", info.AgentID)

		if !limiter.Allow(key) {
			return nil, fmt.Errorf("rate limit exceeded for agent %s: %d requests per %v",
				info.AgentID, limit, window)
		}

		return handler(ctx, input)
	}
}

// RateLimitingToolInterceptor creates an interceptor that enforces rate limiting on tool execution.
// It limits the number of tool calls per time window per tool name.
func RateLimitingToolInterceptor(limit int, window time.Duration) core.ToolInterceptor {
	limiter := NewRateLimiter(limit, window)

	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Use tool name as the rate limiting key
		key := fmt.Sprintf("tool:%s", info.Name)

		if !limiter.Allow(key) {
			return core.ToolResult{}, fmt.Errorf("rate limit exceeded for tool %s: %d requests per %v",
				info.Name, limit, window)
		}

		return handler(ctx, args)
	}
}

// ValidationModuleInterceptor creates an interceptor that validates module inputs.
// It checks input size, string lengths, and forbidden patterns.
func ValidationModuleInterceptor(config ValidationConfig) core.ModuleInterceptor {
	compiledPatterns := make([]*regexp.Regexp, len(config.ForbiddenPatterns))
	for i, pattern := range config.ForbiddenPatterns {
		compiledPatterns[i] = regexp.MustCompile(pattern)
	}

	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Validate inputs
		if err := validateInputs(inputs, config, compiledPatterns); err != nil {
			return nil, fmt.Errorf("module input validation failed for %s: %w", info.ModuleName, err)
		}

		return handler(ctx, inputs, opts...)
	}
}

// ValidationAgentInterceptor creates an interceptor that validates agent inputs.
func ValidationAgentInterceptor(config ValidationConfig) core.AgentInterceptor {
	compiledPatterns := make([]*regexp.Regexp, len(config.ForbiddenPatterns))
	for i, pattern := range config.ForbiddenPatterns {
		compiledPatterns[i] = regexp.MustCompile(pattern)
	}

	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		// Validate inputs
		if err := validateInputsGeneric(input, config, compiledPatterns); err != nil {
			return nil, fmt.Errorf("agent input validation failed for %s: %w", info.AgentID, err)
		}

		return handler(ctx, input)
	}
}

// ValidationToolInterceptor creates an interceptor that validates tool arguments.
func ValidationToolInterceptor(config ValidationConfig) core.ToolInterceptor {
	compiledPatterns := make([]*regexp.Regexp, len(config.ForbiddenPatterns))
	for i, pattern := range config.ForbiddenPatterns {
		compiledPatterns[i] = regexp.MustCompile(pattern)
	}

	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Validate arguments
		if err := validateInputsGeneric(args, config, compiledPatterns); err != nil {
			return core.ToolResult{}, fmt.Errorf("tool argument validation failed for %s: %w", info.Name, err)
		}

		return handler(ctx, args)
	}
}

// AuthorizationContext holds authorization information.
type AuthorizationContext struct {
	UserID      string
	Roles       []string
	Permissions []string
	Scopes      []string
	APIKey      string
}

// AuthorizationPolicy defines what operations are allowed.
type AuthorizationPolicy struct {
	RequiredRoles       []string
	RequiredPermissions []string
	RequiredScopes      []string
	RequireAuth         bool
	CustomRules         map[string]string
	AllowedModules      []string
	AllowedAgents       []string
	AllowedTools        []string
}

// AuthorizationInterceptor creates authorization interceptors.
type AuthorizationInterceptor struct {
	policies map[string]AuthorizationPolicy // key -> policy mapping
}

// NewAuthorizationInterceptor creates a new authorization interceptor.
func NewAuthorizationInterceptor() *AuthorizationInterceptor {
	return &AuthorizationInterceptor{
		policies: make(map[string]AuthorizationPolicy),
	}
}

// SetPolicy sets an authorization policy for a specific resource.
func (ai *AuthorizationInterceptor) SetPolicy(resource string, policy AuthorizationPolicy) {
	ai.policies[resource] = policy
}

// ModuleAuthorizationInterceptor creates an interceptor that enforces authorization on modules.
func (ai *AuthorizationInterceptor) ModuleAuthorizationInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Get authorization context from request context
		authCtx := getAuthorizationContext(ctx)
		if authCtx == nil {
			return nil, errors.New("authorization context required")
		}

		// Check policy for this module
		policy, exists := ai.policies[info.ModuleName]
		if exists {
			if !ai.checkAuthorization(authCtx, policy, info.ModuleName) {
				return nil, fmt.Errorf("access denied to module %s for user %s", info.ModuleName, authCtx.UserID)
			}
		}

		return handler(ctx, inputs, opts...)
	}
}

// AgentAuthorizationInterceptor creates an interceptor that enforces authorization on agents.
func (ai *AuthorizationInterceptor) AgentAuthorizationInterceptor() core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		// Get authorization context from request context
		authCtx := getAuthorizationContext(ctx)
		if authCtx == nil {
			return nil, errors.New("authorization context required")
		}

		// Check policy for this agent
		policy, exists := ai.policies[info.AgentID]
		if exists {
			if !ai.checkAuthorization(authCtx, policy, info.AgentID) {
				return nil, fmt.Errorf("access denied to agent %s for user %s", info.AgentID, authCtx.UserID)
			}
		}

		return handler(ctx, input)
	}
}

// ToolAuthorizationInterceptor creates an interceptor that enforces authorization on tools.
func (ai *AuthorizationInterceptor) ToolAuthorizationInterceptor() core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Get authorization context from request context
		authCtx := getAuthorizationContext(ctx)
		if authCtx == nil {
			return core.ToolResult{}, errors.New("authorization context required")
		}

		// Check policy for this tool
		policy, exists := ai.policies[info.Name]
		if exists {
			if !ai.checkAuthorization(authCtx, policy, info.Name) {
				return core.ToolResult{}, fmt.Errorf("access denied to tool %s for user %s", info.Name, authCtx.UserID)
			}
		}

		return handler(ctx, args)
	}
}

// InputSanitizationInterceptor creates interceptors that sanitize inputs to prevent injection attacks.

// SanitizingModuleInterceptor creates an interceptor that sanitizes module inputs.
func SanitizingModuleInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		sanitizedInputs := sanitizeInputs(inputs)
		return handler(ctx, sanitizedInputs, opts...)
	}
}

// SanitizingAgentInterceptor creates an interceptor that sanitizes agent inputs.
func SanitizingAgentInterceptor() core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		sanitizedInput := sanitizeInputsGeneric(input)
		return handler(ctx, sanitizedInput)
	}
}

// SanitizingToolInterceptor creates an interceptor that sanitizes tool arguments.
func SanitizingToolInterceptor() core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		sanitizedArgs := sanitizeInputsGeneric(args)
		return handler(ctx, sanitizedArgs)
	}
}

// Helper functions

// validateInputs validates module inputs against the configuration.
func validateInputs(inputs map[string]any, config ValidationConfig, patterns []*regexp.Regexp) error {
	// Convert to generic map for validation
	genericInputs := make(map[string]interface{})
	for k, v := range inputs {
		genericInputs[k] = v
	}
	return validateInputsGeneric(genericInputs, config, patterns)
}

// validateInputsGeneric validates generic inputs against the configuration.
func validateInputsGeneric(inputs map[string]interface{}, config ValidationConfig, patterns []*regexp.Regexp) error {
	// Calculate total input size
	totalSize := calculateMapSize(inputs)
	if totalSize > config.MaxInputSize {
		return fmt.Errorf("input size %d exceeds maximum %d bytes", totalSize, config.MaxInputSize)
	}

	// Check required fields
	for _, field := range config.RequiredFields {
		if _, exists := inputs[field]; !exists {
			return fmt.Errorf("required field '%s' is missing", field)
		}
	}

	// Validate each input recursively
	return validateValue(inputs, config, patterns)
}

// validateValue validates a single value recursively.
func validateValue(value interface{}, config ValidationConfig, patterns []*regexp.Regexp) error {
	switch v := value.(type) {
	case string:
		if len(v) > config.MaxStringLength {
			return fmt.Errorf("string length %d exceeds maximum %d", len(v), config.MaxStringLength)
		}

		// Check forbidden patterns
		for _, pattern := range patterns {
			if pattern.MatchString(v) {
				return fmt.Errorf("input contains forbidden pattern: %s", pattern.String())
			}
		}

		// Check HTML content if not allowed
		if !config.AllowHTML && containsHTML(v) {
			return fmt.Errorf("HTML content not allowed in input")
		}

	case map[string]interface{}:
		for _, subValue := range v {
			if err := validateValue(subValue, config, patterns); err != nil {
				return err
			}
		}

	case []interface{}:
		for _, subValue := range v {
			if err := validateValue(subValue, config, patterns); err != nil {
				return err
			}
		}
	}

	return nil
}

// sanitizeInputs sanitizes module inputs.
func sanitizeInputs(inputs map[string]any) map[string]any {
	result := make(map[string]any)
	for k, v := range inputs {
		result[k] = sanitizeValue(v)
	}
	return result
}

// sanitizeInputsGeneric sanitizes generic inputs.
func sanitizeInputsGeneric(inputs map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range inputs {
		result[k] = sanitizeValue(v)
	}
	return result
}

// sanitizeValue sanitizes a single value recursively.
func sanitizeValue(value interface{}) interface{} {
	switch v := value.(type) {
	case string:
		// Multi-layered sanitization for strings
		sanitized := v

		// HTML escape to prevent XSS
		sanitized = html.EscapeString(sanitized)

		// Remove or replace dangerous characters
		sanitized = strings.ReplaceAll(sanitized, "\x00", "") // Remove null bytes
		sanitized = strings.ReplaceAll(sanitized, "\r\n", " ") // Normalize line endings
		sanitized = strings.ReplaceAll(sanitized, "\n", " ")
		sanitized = strings.ReplaceAll(sanitized, "\r", " ")

		// Limit string length to prevent DoS
		if len(sanitized) > 10000 {
			sanitized = sanitized[:10000]
		}

		return sanitized

	case map[string]interface{}:
		result := make(map[string]interface{})
		for k, subValue := range v {
			// Sanitize both key and value
			sanitizedKey := html.EscapeString(k)
			if len(sanitizedKey) > 1000 { // Limit key length
				sanitizedKey = sanitizedKey[:1000]
			}
			result[sanitizedKey] = sanitizeValue(subValue)
		}
		return result

	case []interface{}:
		// Limit array size to prevent DoS
		maxSize := 1000
		if len(v) > maxSize {
			v = v[:maxSize]
		}

		result := make([]interface{}, len(v))
		for i, subValue := range v {
			result[i] = sanitizeValue(subValue)
		}
		return result

	default:
		return value
	}
}

// calculateMapSize estimates the size of a map in bytes.
func calculateMapSize(m map[string]interface{}) int {
	size := 0
	for k, v := range m {
		size += len(k)
		size += calculateValueSize(v)
	}
	return size
}

// calculateValueSize estimates the size of a value in bytes.
func calculateValueSize(value interface{}) int {
	switch v := value.(type) {
	case string:
		return len(v)
	case map[string]interface{}:
		return calculateMapSize(v)
	case []interface{}:
		size := 0
		for _, item := range v {
			size += calculateValueSize(item)
		}
		return size
	case int, int32, int64:
		return 8
	case float32, float64:
		return 8
	case bool:
		return 1
	default:
		return 0
	}
}

// Context helpers

type authContextKey struct{}

// WithAuthorizationContext adds authorization context to the request context.
func WithAuthorizationContext(ctx context.Context, authCtx *AuthorizationContext) context.Context {
	return context.WithValue(ctx, authContextKey{}, authCtx)
}

// getAuthorizationContext retrieves authorization context from the request context.
func getAuthorizationContext(ctx context.Context) *AuthorizationContext {
	if authCtx, ok := ctx.Value(authContextKey{}).(*AuthorizationContext); ok {
		return authCtx
	}
	return nil
}

// checkAuthorization verifies if the authorization context satisfies the policy.
func (ai *AuthorizationInterceptor) checkAuthorization(authCtx *AuthorizationContext, policy AuthorizationPolicy, resource string) bool {
	// Check if authentication is required but not present
	if policy.RequireAuth && (authCtx.UserID == "" && authCtx.APIKey == "") {
		return false
	}

	// Check required roles
	if len(policy.RequiredRoles) > 0 {
		if !hasAnyRole(authCtx.Roles, policy.RequiredRoles) {
			return false
		}
	}

	// Check required permissions
	if len(policy.RequiredPermissions) > 0 {
		if !hasAnyPermission(authCtx.Permissions, policy.RequiredPermissions) {
			return false
		}
	}

	// Check required scopes
	if len(policy.RequiredScopes) > 0 {
		if !hasAnyScope(authCtx.Scopes, policy.RequiredScopes) {
			return false
		}
	}

	// Check custom rules (simple key-value matching for now)
	// TODO: Implement more sophisticated rule evaluation if needed
	if len(policy.CustomRules) > 0 {
		// For now, this is a placeholder - custom rules would need specific implementation
		// depending on the business logic requirements
		for ruleKey, ruleValue := range policy.CustomRules {
			// This is a basic implementation - extend as needed
			_ = ruleKey
			_ = ruleValue
			// Custom rule evaluation would go here
		}
	}

	// Check resource-specific allowlists
	if len(policy.AllowedModules) > 0 {
		if !contains(policy.AllowedModules, resource) {
			return false
		}
	}

	if len(policy.AllowedAgents) > 0 {
		if !contains(policy.AllowedAgents, resource) {
			return false
		}
	}

	if len(policy.AllowedTools) > 0 {
		if !contains(policy.AllowedTools, resource) {
			return false
		}
	}

	return true
}

// hasAnyRole checks if any of the user's roles match the required roles.
func hasAnyRole(userRoles, requiredRoles []string) bool {
	for _, userRole := range userRoles {
		for _, requiredRole := range requiredRoles {
			if userRole == requiredRole {
				return true
			}
		}
	}
	return false
}

// hasAnyPermission checks if any of the user's permissions match the required permissions.
func hasAnyPermission(userPermissions, requiredPermissions []string) bool {
	for _, userPerm := range userPermissions {
		for _, requiredPerm := range requiredPermissions {
			if matchesPermission(userPerm, requiredPerm) {
				return true
			}
		}
	}
	return false
}

// hasAnyScope checks if any of the user's scopes match the required scopes.
func hasAnyScope(userScopes, requiredScopes []string) bool {
	for _, userScope := range userScopes {
		for _, requiredScope := range requiredScopes {
			if userScope == requiredScope {
				return true
			}
		}
	}
	return false
}

// matchesPermission checks if a user permission matches a required permission.
// Supports wildcard matching where user permissions ending with "*" match
// required permissions that start with the prefix (excluding the "*").
func matchesPermission(userPerm, requiredPerm string) bool {
	// Exact match
	if userPerm == requiredPerm {
		return true
	}

	// Wildcard matching - user permission ends with "*"
	if strings.HasSuffix(userPerm, "*") {
		prefix := strings.TrimSuffix(userPerm, "*")
		// Only match if required permission starts with the prefix
		// and is longer than the prefix (prevents prefix matching itself and empty prefixes)
		return len(prefix) > 0 && len(requiredPerm) > len(prefix) && strings.HasPrefix(requiredPerm, prefix)
	}

	return false
}

// contains checks if a slice contains a specific string.
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// containsHTML checks if a string contains HTML-like content.
// This is a simple heuristic check for common HTML patterns.
func containsHTML(s string) bool {
	// Check for HTML tags (opening, closing, or self-closing)
	htmlTagPattern := regexp.MustCompile(`<\s*/?[a-zA-Z][^>]*>`)
	// Check for HTML comments
	htmlCommentPattern := regexp.MustCompile(`<!--.*?-->`)

	return htmlTagPattern.MatchString(s) || htmlCommentPattern.MatchString(s)
}

// periodicCleanup removes old unused keys from the rate limiter to prevent memory leaks.
// Keys that haven't been accessed for more than 2x the rate limiting window are removed.
func (rl *RateLimiter) periodicCleanup() {
	ticker := time.NewTicker(rl.window * 2) // Cleanup every 2x window duration
	defer ticker.Stop()

	for range ticker.C {
		rl.mu.Lock()
		now := time.Now()
		cleanupThreshold := now.Add(-rl.window * 2) // Remove keys not accessed for 2x window

		for key, lastAccess := range rl.lastAccess {
			if lastAccess.Before(cleanupThreshold) {
				delete(rl.requests, key)
				delete(rl.lastAccess, key)
			}
		}
		rl.mu.Unlock()
	}
}
