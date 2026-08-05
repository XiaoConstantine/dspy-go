package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

var (
	globalCacheInstance Cache
	globalCacheConfig   CacheConfig
	globalCacheMu       sync.RWMutex
)

// CachedLLM wraps an LLM with transparent caching functionality.
// It implements core.LLM interface and can be used as a drop-in replacement.
type CachedLLM struct {
	core.LLM
	cache        Cache
	keyGenerator *KeyGenerator
	ttl          time.Duration
	enabled      bool
}

// WrapWithCache wraps an LLM with caching using the global cache configuration.
// This is called automatically by the LLM factory if caching is enabled.
func WrapWithCache(llm core.LLM, fileConfig *config.CachingConfig) core.LLM {
	cacheConfig := LoadCacheConfig(fileConfig)

	// If caching is disabled, return the original LLM
	if !IsEnabled(cacheConfig) {
		return llm
	}

	cache := getOrCreateGlobalCache(cacheConfig)
	if cache == nil {
		// If cache creation failed, return original LLM
		return llm
	}

	return &CachedLLM{
		LLM:          llm,
		cache:        cache,
		keyGenerator: NewKeyGenerator("dspy_"),
		ttl:          cacheConfig.DefaultTTL,
		enabled:      true,
	}
}

// getOrCreateGlobalCache returns the global cache instance, creating it if necessary.
func getOrCreateGlobalCache(cacheConfig CacheConfig) Cache {
	globalCacheMu.Lock()
	defer globalCacheMu.Unlock()

	// If config changed, recreate cache
	if globalCacheInstance == nil || !configEqual(globalCacheConfig, cacheConfig) {
		if globalCacheInstance != nil {
			globalCacheInstance.Close()
		}

		var err error
		globalCacheInstance, err = NewCache(cacheConfig)
		if err != nil {
			// Log error but don't fail - return nil to disable caching
			return nil
		}
		globalCacheConfig = cacheConfig
	}

	return globalCacheInstance
}

// configEqual checks if two cache configurations are equal.
func configEqual(a, b CacheConfig) bool {
	if a.Type != b.Type ||
		a.DefaultTTL != b.DefaultTTL ||
		a.MaxSize != b.MaxSize {
		return false
	}

	// Compare SQLite-specific configuration
	if a.SQLiteConfig.Path != b.SQLiteConfig.Path ||
		a.SQLiteConfig.EnableWAL != b.SQLiteConfig.EnableWAL ||
		a.SQLiteConfig.VacuumInterval != b.SQLiteConfig.VacuumInterval ||
		a.SQLiteConfig.MaxConnections != b.SQLiteConfig.MaxConnections {
		return false
	}

	// Compare Memory-specific configuration
	if a.MemoryConfig.EvictionPolicy != b.MemoryConfig.EvictionPolicy ||
		a.MemoryConfig.CleanupInterval != b.MemoryConfig.CleanupInterval ||
		a.MemoryConfig.ShardCount != b.MemoryConfig.ShardCount {
		return false
	}

	return true
}

// Generate implements core.LLM with caching.
func (c *CachedLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	if !c.enabled || c.cache == nil {
		return c.LLM.Generate(ctx, prompt, options...)
	}

	namespace := c.cacheNamespace()
	if namespace == "" {
		return c.LLM.Generate(ctx, prompt, options...)
	}
	key := c.keyGenerator.GenerateKey(namespace, prompt, options)
	if key == "" {
		return c.LLM.Generate(ctx, prompt, options...)
	}
	return c.withCache(ctx, key, func() (*core.LLMResponse, error) {
		return c.LLM.Generate(ctx, prompt, options...)
	})
}

// GenerateWithJSON implements core.LLM with caching for JSON responses.
func (c *CachedLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	if !c.enabled || c.cache == nil {
		return c.LLM.GenerateWithJSON(ctx, prompt, options...)
	}

	namespace := c.cacheNamespace()
	if namespace == "" {
		return c.LLM.GenerateWithJSON(ctx, prompt, options...)
	}
	key := c.keyGenerator.GenerateJSONKey(namespace, prompt, nil, options)
	if key == "" {
		return c.LLM.GenerateWithJSON(ctx, prompt, options...)
	}

	// Try cache first
	if cached, found, err := c.cache.Get(ctx, key); found && err == nil {
		var result map[string]any
		if err := json.Unmarshal(cached, &result); err == nil {
			return result, nil
		}
	}

	// Call underlying LLM
	result, err := c.LLM.GenerateWithJSON(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	// Cache the result
	if data, err := json.Marshal(result); err == nil {
		_ = c.cache.Set(ctx, key, data, c.ttl)
	}

	return result, nil
}

// GenerateWithFunctions implements core.LLM with caching for function calls.
func (c *CachedLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	if !c.enabled || c.cache == nil {
		return c.LLM.GenerateWithFunctions(ctx, prompt, functions, options...)
	}

	// Create a special key that includes function definitions
	namespace := c.cacheNamespace()
	if namespace == "" {
		return c.LLM.GenerateWithFunctions(ctx, prompt, functions, options...)
	}
	key := c.keyGenerator.GenerateJSONKey(namespace, prompt, functions, options)
	if key == "" {
		return c.LLM.GenerateWithFunctions(ctx, prompt, functions, options...)
	}

	// Try cache first
	if cached, found, err := c.cache.Get(ctx, key); found && err == nil {
		var result map[string]any
		if err := json.Unmarshal(cached, &result); err == nil {
			return result, nil
		}
	}

	// Call underlying LLM
	result, err := c.LLM.GenerateWithFunctions(ctx, prompt, functions, options...)
	if err != nil {
		return nil, err
	}

	// Cache the result
	if data, err := json.Marshal(result); err == nil {
		_ = c.cache.Set(ctx, key, data, c.ttl)
	}

	return result, nil
}

// GenerateWithTools preserves provider-native tool calling when the wrapped
// LLM supports it. Native chat/tool sessions are stateful, so this path is
// currently pass-through rather than cache-backed.
func (c *CachedLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	chatLLM, ok := c.LLM.(core.ToolCallingChatLLM)
	if !ok {
		return nil, fmt.Errorf("wrapped llm %s does not support native tool calling", c.ModelID())
	}
	return chatLLM.GenerateWithTools(ctx, messages, tools, options...)
}

// GenerateWithContent implements core.LLM with caching for multimodal content.
func (c *CachedLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	if !c.enabled || c.cache == nil {
		return c.LLM.GenerateWithContent(ctx, content, options...)
	}

	// Convert to internal format for key generation
	var contents []Content
	for _, block := range content {
		contents = append(contents, Content{
			Type:     string(block.Type),
			Text:     block.Text,
			Data:     string(block.Data),
			MimeType: block.MimeType,
			Metadata: block.Metadata,
		})
	}

	namespace := c.cacheNamespace()
	if namespace == "" {
		return c.LLM.GenerateWithContent(ctx, content, options...)
	}
	key := c.keyGenerator.GenerateContentKey(namespace, contents, options)
	if key == "" {
		return c.LLM.GenerateWithContent(ctx, content, options...)
	}
	return c.withCache(ctx, key, func() (*core.LLMResponse, error) {
		return c.LLM.GenerateWithContent(ctx, content, options...)
	})
}

type endpointConfigurer interface {
	GetEndpointConfig() *core.EndpointConfig
}

type llmUnwrapper interface {
	Unwrap() core.LLM
}

// cacheNamespace separates otherwise identical model names across providers,
// deployments, and credential scopes. Header values participate only in the
// namespace digest and are never included verbatim in the returned key.
func (c *CachedLLM) cacheNamespace() string {
	identity := struct {
		Provider    string `json:"provider"`
		Model       string `json:"model"`
		BaseURL     string `json:"base_url,omitempty"`
		Path        string `json:"path,omitempty"`
		HeadersHash string `json:"headers_hash,omitempty"`
	}{Provider: c.LLM.ProviderName(), Model: c.LLM.ModelID()}
	if identity.Provider == "" || identity.Model == "" {
		return ""
	}

	llm := c.LLM
	seen := make(map[core.LLM]struct{})
	for llm != nil {
		// Decorator implementations are normally pointers and therefore
		// comparable. Track comparable values so a malformed Unwrap cycle cannot
		// make cache-key generation loop forever, without imposing an arbitrary
		// limit on valid decorator depth.
		value := reflect.ValueOf(llm)
		if value.Comparable() {
			if _, exists := seen[llm]; exists {
				return ""
			}
			seen[llm] = struct{}{}
		}
		if configured, ok := llm.(endpointConfigurer); ok {
			if endpoint := configured.GetEndpointConfig(); endpoint != nil {
				identity.BaseURL = endpoint.BaseURL
				identity.Path = endpoint.Path
				identity.HeadersHash = cacheHeadersHash(endpoint.Headers)
			}
			break
		}
		// Interface values with non-comparable concrete types cannot be tracked
		// safely. Disable caching rather than invoking an untrackable wrapper that
		// could return itself forever or hiding routing identity from the key.
		if !value.Comparable() {
			return ""
		}
		unwrapper, ok := llm.(llmUnwrapper)
		if !ok {
			break
		}
		next := unwrapper.Unwrap()
		if next == nil {
			return ""
		}
		llm = next
	}

	data, err := json.Marshal(identity)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s:%s:%x", identity.Provider, identity.Model, sha256.Sum256(data))
}

func cacheHeadersHash(headers map[string]string) string {
	if len(headers) == 0 {
		return ""
	}
	data, err := json.Marshal(headers)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

// Streaming methods are not cached - pass through directly.
func (c *CachedLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return c.LLM.StreamGenerate(ctx, prompt, options...)
}

func (c *CachedLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return c.LLM.StreamGenerateWithContent(ctx, content, options...)
}

// Embedding methods are passed through (could be cached in the future).
func (c *CachedLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return c.LLM.CreateEmbedding(ctx, input, options...)
}

func (c *CachedLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return c.LLM.CreateEmbeddings(ctx, inputs, options...)
}

// withCache is a helper for caching LLMResponse.
func (c *CachedLLM) withCache(ctx context.Context, key string, fn func() (*core.LLMResponse, error)) (*core.LLMResponse, error) {
	// Try cache first
	if cached, found, err := c.cache.Get(ctx, key); found && err == nil {
		var response core.LLMResponse
		if err := json.Unmarshal(cached, &response); err == nil {
			// Mark as cache hit
			if response.Metadata == nil {
				response.Metadata = make(map[string]any)
			}
			response.Metadata["cache_hit"] = true
			return &response, nil
		}
	}

	// Call underlying function
	response, err := fn()
	if err != nil {
		return nil, err
	}

	// Cache successful response
	if response != nil {
		// Create a copy of metadata to avoid race conditions
		metadata := make(map[string]any)
		if response.Metadata != nil {
			for k, v := range response.Metadata {
				metadata[k] = v
			}
		}
		metadata["cache_hit"] = false

		// Create a copy of the response for caching and return
		responseCopy := *response
		responseCopy.Metadata = metadata

		if data, err := json.Marshal(responseCopy); err == nil {
			_ = c.cache.Set(ctx, key, data, c.ttl)
		}

		// Return the copied response (not the original) to avoid race conditions
		return &responseCopy, nil
	}

	return response, nil
}

// Cache management methods

// Unwrap returns the underlying LLM instance.
func (c *CachedLLM) Unwrap() core.LLM {
	return c.LLM
}

// SetCacheEnabled enables/disables caching for this LLM instance.
func (c *CachedLLM) SetCacheEnabled(enabled bool) {
	c.enabled = enabled
}

// IsCacheEnabled returns whether caching is enabled for this LLM instance.
func (c *CachedLLM) IsCacheEnabled() bool {
	return c.enabled && c.cache != nil
}

// ClearCache clears the cache.
func (c *CachedLLM) ClearCache(ctx context.Context) error {
	if c.cache == nil {
		return nil
	}
	return c.cache.Clear(ctx)
}

// CacheStats returns cache statistics.
func (c *CachedLLM) CacheStats() CacheStats {
	if c.cache == nil {
		return CacheStats{}
	}
	return c.cache.Stats()
}

// Global cache management functions

// ClearGlobalCache clears the global cache used by all LLMs.
func ClearGlobalCache(ctx context.Context) error {
	globalCacheMu.RLock()
	defer globalCacheMu.RUnlock()

	if globalCacheInstance == nil {
		return nil
	}
	return globalCacheInstance.Clear(ctx)
}

// GetGlobalCacheStats returns statistics for the global cache.
func GetGlobalCacheStats() CacheStats {
	globalCacheMu.RLock()
	defer globalCacheMu.RUnlock()

	if globalCacheInstance == nil {
		return CacheStats{}
	}
	return globalCacheInstance.Stats()
}

// This affects all new LLM instances created after this call.
func SetGlobalCacheEnabled(enabled bool) {
	globalCacheMu.Lock()
	defer globalCacheMu.Unlock()

	if !enabled && globalCacheInstance != nil {
		globalCacheInstance.Close()
		globalCacheInstance = nil
	}
}
