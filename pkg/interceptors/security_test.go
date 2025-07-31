package interceptors

import (
	"context"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

func TestRateLimiter(t *testing.T) {
	limiter := NewRateLimiter(2, time.Second)

	// Should allow first two requests
	if !limiter.Allow("test") {
		t.Error("Expected first request to be allowed")
	}
	if !limiter.Allow("test") {
		t.Error("Expected second request to be allowed")
	}

	// Should block third request
	if limiter.Allow("test") {
		t.Error("Expected third request to be blocked")
	}

	// Different key should be allowed
	if !limiter.Allow("other") {
		t.Error("Expected request with different key to be allowed")
	}
}

func TestRateLimiterTimeWindow(t *testing.T) {
	limiter := NewRateLimiter(1, 100*time.Millisecond)

	// Should allow first request
	if !limiter.Allow("test") {
		t.Error("Expected first request to be allowed")
	}

	// Should block second request
	if limiter.Allow("test") {
		t.Error("Expected second request to be blocked")
	}

	// Wait for window to reset
	time.Sleep(150 * time.Millisecond)

	// Should allow request after window reset
	if !limiter.Allow("test") {
		t.Error("Expected request after window reset to be allowed")
	}
}

func TestRateLimitingAgentInterceptor(t *testing.T) {
	interceptor := RateLimitingAgentInterceptor(2, time.Second)

	ctx := context.Background()
	input := map[string]interface{}{"test": "value"}
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		return map[string]interface{}{"result": "success"}, nil
	}

	// First two requests should succeed
	for i := 0; i < 2; i++ {
		result, err := interceptor(ctx, input, info, handler)
		if err != nil {
			t.Errorf("Request %d should have succeeded, got error: %v", i+1, err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	}

	// Third request should fail
	_, err := interceptor(ctx, input, info, handler)
	if err == nil {
		t.Error("Expected third request to be rate limited")
	}
	if !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("Expected rate limit error, got: %v", err)
	}
}

func TestMatchesPermission(t *testing.T) {
	tests := []struct {
		userPerm     string
		requiredPerm string
		expected     bool
		description  string
	}{
		{"read", "read", true, "exact match"},
		{"read", "write", false, "different permissions"},
		{"admin*", "admin.users", true, "wildcard match"},
		{"admin*", "admin", false, "wildcard should not match prefix itself"},
		{"*", "anything", false, "empty prefix should not match"},
		{"", "read", false, "empty user permission"},
		{"read*", "readonly", true, "prefix match"},
		{"read*", "writeonly", false, "no prefix match"},
	}

	for _, test := range tests {
		result := matchesPermission(test.userPerm, test.requiredPerm)
		if result != test.expected {
			t.Errorf("%s: matchesPermission(%q, %q) = %v, want %v", 
				test.description, test.userPerm, test.requiredPerm, result, test.expected)
		}
	}
}

func TestSanitizeValue(t *testing.T) {
	tests := []struct {
		input       interface{}
		expectedStr string
		description string
	}{
		{"<script>alert('xss')</script>", "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;", "XSS script tag"},
		{"Hello\x00World", "HelloWorld", "null byte removal"},
		{"Line1\r\nLine2\nLine3\r", "Line1 Line2 Line3 ", "line ending normalization"},
		{strings.Repeat("A", 15000), strings.Repeat("A", 10000), "string length limit"},
		{map[string]interface{}{"<script>": "alert('xss')"}, "", "map key/value sanitization"},
	}

	for _, test := range tests {
		result := sanitizeValue(test.input)
		
		switch test.input.(type) {
		case string:
			if str, ok := result.(string); ok {
				if str != test.expectedStr {
					t.Errorf("%s: got %q, want %q", test.description, str, test.expectedStr)
				}
			} else {
				t.Errorf("%s: result is not a string", test.description)
			}
		case map[string]interface{}:
			if m, ok := result.(map[string]interface{}); ok {
				// Check that keys and values are sanitized
				for k, v := range m {
					if strings.Contains(k, "<script>") || strings.Contains(v.(string), "<script>") {
						t.Errorf("%s: map not properly sanitized: %v", test.description, m)
					}
				}
			}
		}
	}
}

func TestDefaultValidationConfig(t *testing.T) {
	config := DefaultValidationConfig()
	
	// Test that dangerous patterns are caught
	dangerousInputs := []string{
		"<script>alert('xss')</script>",
		"javascript:alert('xss')",
		"JAVASCRIPT:alert('xss')", // Case insensitive
		"data:text/html,<script>alert('xss')</script>",
		"onload=alert('xss')",
		"eval('malicious')",
		"UNION SELECT * FROM users",
		"DROP TABLE users",
		"../../../etc/passwd",
		"${jndi:ldap://malicious.com}",
		"<%=system('rm -rf /')%>",
	}
	
	// Compile patterns
	compiledPatterns := make([]*regexp.Regexp, len(config.ForbiddenPatterns))
	for i, pattern := range config.ForbiddenPatterns {
		compiledPatterns[i] = regexp.MustCompile(pattern)
	}
	
	for _, input := range dangerousInputs {
		found := false
		for _, pattern := range compiledPatterns {
			if pattern.MatchString(input) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Dangerous input not caught by validation patterns: %s", input)
		}
	}
}

func TestRateLimitingToolInterceptor(t *testing.T) {
	interceptor := RateLimitingToolInterceptor(1, time.Second)

	ctx := context.Background()
	args := map[string]interface{}{"test": "value"}
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return core.ToolResult{Data: "success"}, nil
	}

	// First request should succeed
	result, err := interceptor(ctx, args, info, handler)
	if err != nil {
		t.Errorf("First request should have succeeded, got error: %v", err)
	}
	if result.Data != "success" {
		t.Errorf("Expected success result, got %v", result.Data)
	}

	// Second request should fail
	_, err = interceptor(ctx, args, info, handler)
	if err == nil {
		t.Error("Expected second request to be rate limited")
	}
}

func TestAllowHTMLValidation(t *testing.T) {
	tests := []struct {
		name        string
		allowHTML   bool
		input       string
		shouldError bool
	}{
		{"HTML allowed - valid", true, "<div>content</div>", false},
		{"HTML not allowed - invalid", false, "<div>content</div>", true},
		{"HTML not allowed - plain text", false, "plain text", false},
		{"HTML allowed - plain text", true, "plain text", false},
		{"HTML not allowed - self-closing tag", false, "<img src='test'/>", true},
		{"HTML not allowed - comment", false, "<!-- comment -->", true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := ValidationConfig{
				MaxInputSize:      1000,
				MaxStringLength:   100,
				ForbiddenPatterns: []string{},
				RequiredFields:    []string{},
				AllowHTML:         test.allowHTML,
			}

			compiledPatterns := make([]*regexp.Regexp, 0)
			inputs := map[string]interface{}{"test": test.input}
			
			err := validateInputsGeneric(inputs, config, compiledPatterns)
			
			if test.shouldError && err == nil {
				t.Errorf("Expected error for input %q with AllowHTML=%v", test.input, test.allowHTML)
			}
			if !test.shouldError && err != nil {
				t.Errorf("Unexpected error for input %q with AllowHTML=%v: %v", test.input, test.allowHTML, err)
			}
		})
	}
}

func TestDeterministicCacheKeys(t *testing.T) {
	// Test that cache keys are deterministic even when JSON marshaling fails
	inputs1 := map[string]any{"b": "value2", "a": "value1"}
	inputs2 := map[string]any{"a": "value1", "b": "value2"}
	
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})
	
	key1 := generateModuleCacheKey(inputs1, info)
	key2 := generateModuleCacheKey(inputs2, info)
	
	if key1 != key2 {
		t.Errorf("Expected same cache keys for equivalent maps, got %s and %s", key1, key2)
	}
	
	// Test with tool cache keys too
	args1 := map[string]interface{}{"b": "value2", "a": "value1"}
	args2 := map[string]interface{}{"a": "value1", "b": "value2"}
	
	toolInfo := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})
	
	toolKey1 := generateToolCacheKey(args1, toolInfo)
	toolKey2 := generateToolCacheKey(args2, toolInfo)
	
	if toolKey1 != toolKey2 {
		t.Errorf("Expected same tool cache keys for equivalent maps, got %s and %s", toolKey1, toolKey2)
	}
}

func TestDefaultValidationConfigBasic(t *testing.T) {
	config := DefaultValidationConfig()

	if config.MaxInputSize <= 0 {
		t.Error("Expected positive MaxInputSize")
	}
	if config.MaxStringLength <= 0 {
		t.Error("Expected positive MaxStringLength")
	}
	if len(config.ForbiddenPatterns) == 0 {
		t.Error("Expected forbidden patterns to be configured")
	}
}

func TestValidationModuleInterceptor(t *testing.T) {
	config := ValidationConfig{
		MaxInputSize:      1000,
		MaxStringLength:   100,
		ForbiddenPatterns: []string{"<script>", "javascript:"},
		RequiredFields:    []string{"required_field"},
		AllowHTML:         false,
	}

	interceptor := ValidationModuleInterceptor(config)

	ctx := context.Background()
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	t.Run("Valid input", func(t *testing.T) {
		inputs := map[string]any{
			"required_field": "safe value",
			"other_field":    "another safe value",
		}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error for valid input, got: %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	})

	t.Run("Missing required field", func(t *testing.T) {
		inputs := map[string]any{
			"other_field": "safe value",
		}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error for missing required field")
		}
		if !strings.Contains(err.Error(), "required field") {
			t.Errorf("Expected required field error, got: %v", err)
		}
	})

	t.Run("Forbidden pattern", func(t *testing.T) {
		inputs := map[string]any{
			"required_field": "safe value",
			"malicious":      "<script>alert('xss')</script>",
		}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error for forbidden pattern")
		}
		if !strings.Contains(err.Error(), "forbidden pattern") {
			t.Errorf("Expected forbidden pattern error, got: %v", err)
		}
	})

	t.Run("String too long", func(t *testing.T) {
		longString := strings.Repeat("a", 200) // Exceeds MaxStringLength of 100
		inputs := map[string]any{
			"required_field": longString,
		}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error for string too long")
		}
		if !strings.Contains(err.Error(), "length") && !strings.Contains(err.Error(), "exceeds") {
			t.Errorf("Expected string length error, got: %v", err)
		}
	})
}

func TestValidationAgentInterceptor(t *testing.T) {
	config := ValidationConfig{
		MaxInputSize:      1000,
		MaxStringLength:   100,
		ForbiddenPatterns: []string{"<script>"},
		RequiredFields:    []string{},
		AllowHTML:         false,
	}

	interceptor := ValidationAgentInterceptor(config)

	ctx := context.Background()
	info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

	handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
		return map[string]interface{}{"result": "success"}, nil
	}

	// Valid input
	input := map[string]interface{}{"safe_field": "safe value"}
	result, err := interceptor(ctx, input, info, handler)
	if err != nil {
		t.Errorf("Expected no error for valid input, got: %v", err)
	}
	if result["result"] != "success" {
		t.Errorf("Expected success result, got %v", result["result"])
	}

	// Malicious input
	maliciousInput := map[string]interface{}{"malicious": "<script>alert('xss')</script>"}
	_, err = interceptor(ctx, maliciousInput, info, handler)
	if err == nil {
		t.Error("Expected error for malicious input")
	}
}

func TestValidationToolInterceptor(t *testing.T) {
	config := ValidationConfig{
		MaxInputSize:      1000,
		MaxStringLength:   100,
		ForbiddenPatterns: []string{"javascript:"},
		RequiredFields:    []string{},
		AllowHTML:         false,
	}

	interceptor := ValidationToolInterceptor(config)

	ctx := context.Background()
	info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

	handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
		return core.ToolResult{Data: "success"}, nil
	}

	// Valid input
	args := map[string]interface{}{"safe_arg": "safe value"}
	result, err := interceptor(ctx, args, info, handler)
	if err != nil {
		t.Errorf("Expected no error for valid input, got: %v", err)
	}
	if result.Data != "success" {
		t.Errorf("Expected success result, got %v", result.Data)
	}

	// Malicious input
	maliciousArgs := map[string]interface{}{"malicious": "javascript:alert('xss')"}
	_, err = interceptor(ctx, maliciousArgs, info, handler)
	if err == nil {
		t.Error("Expected error for malicious input")
	}
}

func TestAuthorizationInterceptor(t *testing.T) {
	authInterceptor := NewAuthorizationInterceptor()

	// Set up policy
	policy := AuthorizationPolicy{
		RequiredRoles:       []string{"admin", "user"},
		RequiredPermissions: []string{"read", "write"},
		AllowedModules:      []string{"TestModule"},
	}
	authInterceptor.SetPolicy("TestModule", policy)

	interceptor := authInterceptor.ModuleAuthorizationInterceptor()

	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	t.Run("No authorization context", func(t *testing.T) {
		ctx := context.Background()
		inputs := map[string]any{"test": "value"}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error for missing authorization context")
		}
		if !strings.Contains(err.Error(), "authorization context required") {
			t.Errorf("Expected authorization context error, got: %v", err)
		}
	})

	t.Run("Valid authorization", func(t *testing.T) {
		authCtx := &AuthorizationContext{
			UserID:      "user123",
			Roles:       []string{"admin"},
			Permissions: []string{"read", "write"},
		}
		ctx := WithAuthorizationContext(context.Background(), authCtx)
		inputs := map[string]any{"test": "value"}

		result, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			t.Errorf("Expected no error for valid authorization, got: %v", err)
		}
		if result["result"] != "success" {
			t.Errorf("Expected success result, got %v", result["result"])
		}
	})

	t.Run("Insufficient permissions", func(t *testing.T) {
		authCtx := &AuthorizationContext{
			UserID:      "user123",
			Roles:       []string{"guest"}, // Not in allowed roles
			Permissions: []string{"read"},  // Missing write permission
		}
		ctx := WithAuthorizationContext(context.Background(), authCtx)
		inputs := map[string]any{"test": "value"}

		_, err := interceptor(ctx, inputs, info, handler)
		if err == nil {
			t.Error("Expected error for insufficient permissions")
		}
		if !strings.Contains(err.Error(), "access denied") {
			t.Errorf("Expected access denied error, got: %v", err)
		}
	})
}

func TestSanitizingInterceptors(t *testing.T) {
	t.Run("Module sanitization", func(t *testing.T) {
		interceptor := SanitizingModuleInterceptor()

		ctx := context.Background()
		inputs := map[string]any{
			"safe":      "normal text",
			"malicious": "<script>alert('xss')</script>",
		}
		info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

		handlerCalled := false
		var sanitizedInputs map[string]any
		handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
			handlerCalled = true
			sanitizedInputs = inputs
			return map[string]any{"result": "success"}, nil
		}

		_, err := interceptor(ctx, inputs, info, handler)

		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}
		if !handlerCalled {
			t.Error("Handler should have been called")
		}

		// Check that malicious content was sanitized
		sanitizedMalicious := sanitizedInputs["malicious"].(string)
		if strings.Contains(sanitizedMalicious, "<script>") {
			t.Error("Expected script tags to be escaped")
		}
		if !strings.Contains(sanitizedMalicious, "&lt;script&gt;") {
			t.Error("Expected HTML to be escaped")
		}
	})

	t.Run("Agent sanitization", func(t *testing.T) {
		interceptor := SanitizingAgentInterceptor()

		ctx := context.Background()
		input := map[string]interface{}{
			"user_input": "<img src=x onerror=alert('xss')>",
		}
		info := core.NewAgentInfo("TestAgent", "TestType", []core.Tool{})

		handlerCalled := false
		var sanitizedInput map[string]interface{}
		handler := func(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
			handlerCalled = true
			sanitizedInput = input
			return map[string]interface{}{"result": "success"}, nil
		}

		_, err := interceptor(ctx, input, info, handler)

		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}
		if !handlerCalled {
			t.Error("Handler should have been called")
		}

		// Check sanitization - HTML escaping converts < and > to &lt; and &gt;
		sanitizedUserInput := sanitizedInput["user_input"].(string)
		if !strings.Contains(sanitizedUserInput, "&lt;img") {
			t.Error("Expected HTML tags to be escaped")
		}
	})

	t.Run("Tool sanitization", func(t *testing.T) {
		interceptor := SanitizingToolInterceptor()

		ctx := context.Background()
		args := map[string]interface{}{
			"query": "SELECT * FROM users; DROP TABLE users;--",
		}
		info := core.NewToolInfo("TestTool", "Test tool", "TestType", models.InputSchema{})

		handlerCalled := false
		handler := func(ctx context.Context, args map[string]interface{}) (core.ToolResult, error) {
			handlerCalled = true
			return core.ToolResult{Data: "success"}, nil
		}

		_, err := interceptor(ctx, args, info, handler)

		if err != nil {
			t.Errorf("Expected no error, got: %v", err)
		}
		if !handlerCalled {
			t.Error("Handler should have been called")
		}
	})
}

func TestSecurityHelperFunctions(t *testing.T) {
	t.Run("validateValue", func(t *testing.T) {
		config := ValidationConfig{
			MaxInputSize:      1000,
			MaxStringLength:   10,
			ForbiddenPatterns: []string{"<script>"},
		}
		patterns := []*regexp.Regexp{regexp.MustCompile("<script>")}

		// Valid string
		err := validateValue("safe", config, patterns)
		if err != nil {
			t.Errorf("Expected no error for valid string, got: %v", err)
		}

		// String too long
		err = validateValue("this string is too long", config, patterns)
		if err == nil {
			t.Error("Expected error for string too long")
		}

		// Forbidden pattern
		err = validateValue("<script>alert('xss')</script>", config, patterns)
		if err == nil {
			t.Error("Expected error for forbidden pattern")
		}

		// Nested map
		nestedMap := map[string]interface{}{
			"nested": "<script>",
		}
		err = validateValue(nestedMap, config, patterns)
		if err == nil {
			t.Error("Expected error for forbidden pattern in nested map")
		}

		// Array
		array := []interface{}{"safe", "<script>"}
		err = validateValue(array, config, patterns)
		if err == nil {
			t.Error("Expected error for forbidden pattern in array")
		}
	})

	t.Run("sanitizeValue", func(t *testing.T) {
		// String sanitization
		result := sanitizeValue("<script>alert('xss')</script>")
		if !strings.Contains(result.(string), "&lt;script&gt;") {
			t.Error("Expected HTML to be escaped")
		}

		// Map sanitization
		input := map[string]interface{}{
			"safe":      "normal",
			"malicious": "<img src=x onerror=alert('xss')>",
		}
		result = sanitizeValue(input)
		resultMap := result.(map[string]interface{})
		if !strings.Contains(resultMap["malicious"].(string), "&lt;img") {
			t.Error("Expected HTML content to be escaped")
		}

		// Array sanitization
		inputArray := []interface{}{"safe", "<script>"}
		result = sanitizeValue(inputArray)
		resultArray := result.([]interface{})
		if strings.Contains(resultArray[1].(string), "<script>") {
			t.Error("Expected script tag to be escaped in array")
		}
	})

	t.Run("calculateMapSize", func(t *testing.T) {
		m := map[string]interface{}{
			"short": "hi",
			"long":  "this is a longer string",
			"nested": map[string]interface{}{
				"inner": "value",
			},
		}
		size := calculateMapSize(m)
		if size <= 0 {
			t.Error("Expected positive size calculation")
		}
	})

	t.Run("authorization helpers", func(t *testing.T) {
		// Test hasAnyRole
		userRoles := []string{"admin", "editor"}
		requiredRoles := []string{"editor", "viewer"}
		if !hasAnyRole(userRoles, requiredRoles) {
			t.Error("Expected to find matching role")
		}

		requiredRoles = []string{"viewer", "guest"}
		if hasAnyRole(userRoles, requiredRoles) {
			t.Error("Expected no matching role")
		}

		// Test hasAnyPermission
		userPerms := []string{"read", "write"}
		requiredPerms := []string{"write", "delete"}
		if !hasAnyPermission(userPerms, requiredPerms) {
			t.Error("Expected to find matching permission")
		}

		// Test wildcard permission
		userPerms = []string{"admin.*"}
		requiredPerms = []string{"admin.users"}
		if !hasAnyPermission(userPerms, requiredPerms) {
			t.Error("Expected wildcard permission to match")
		}

		// Test contains
		slice := []string{"apple", "banana", "cherry"}
		if !contains(slice, "banana") {
			t.Error("Expected to find banana in slice")
		}
		if contains(slice, "grape") {
			t.Error("Expected not to find grape in slice")
		}
	})
}

// Benchmark tests.
func BenchmarkRateLimiter(b *testing.B) {
	limiter := NewRateLimiter(1000, time.Second)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		limiter.Allow("test")
	}
}

func BenchmarkValidationModuleInterceptor(b *testing.B) {
	config := DefaultValidationConfig()
	interceptor := ValidationModuleInterceptor(config)

	ctx := context.Background()
	inputs := map[string]any{"test": "safe value"}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSanitizingModuleInterceptor(b *testing.B) {
	interceptor := SanitizingModuleInterceptor()

	ctx := context.Background()
	inputs := map[string]any{
		"normal":    "safe value",
		"malicious": "<script>alert('xss')</script>",
	}
	info := core.NewModuleInfo("TestModule", "TestType", core.Signature{})

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{"result": "success"}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(ctx, inputs, info, handler)
		if err != nil {
			b.Fatal(err)
		}
	}
}
