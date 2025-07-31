package cache

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// Middleware provides caching functionality for LLM requests.
// It can be embedded into LLM providers or used as a standalone component.
type Middleware struct {
	cache        Cache
	keyGenerator *KeyGenerator
	ttl          time.Duration
	enabled      atomic.Bool
}

// NewMiddleware creates a new cache middleware.
func NewMiddleware(cache Cache, ttl time.Duration) *Middleware {
	m := &Middleware{
		cache:        cache,
		keyGenerator: NewKeyGenerator("dspy_"),
		ttl:          ttl,
	}
	m.enabled.Store(true)
	return m
}

// WithCache wraps an LLM request with caching logic.
// This is meant to be called by LLM providers internally.
func (m *Middleware) WithCache(
	ctx context.Context,
	cacheKey string,
	ttl time.Duration,
	fn func() (*core.LLMResponse, error),
) (*core.LLMResponse, error) {
	if !m.enabled.Load() || m.cache == nil {
		return fn()
	}

	// Use provided TTL or fall back to middleware default
	if ttl == 0 {
		ttl = m.ttl
	}

	// Try to get from cache
	if cached, found, err := m.cache.Get(ctx, cacheKey); found && err == nil {
		var response core.LLMResponse
		if err := json.Unmarshal(cached, &response); err == nil {
			// Mark as cache hit
			if response.Metadata == nil {
				response.Metadata = make(map[string]interface{})
			}
			response.Metadata["cache_hit"] = true
			response.Metadata["cache_key"] = cacheKey
			return &response, nil
		}
	}

	// Execute the actual LLM call
	response, err := fn()
	if err != nil {
		return nil, err
	}

	// Cache the successful response
	if response != nil {
		if response.Metadata == nil {
			response.Metadata = make(map[string]interface{})
		}
		response.Metadata["cache_hit"] = false
		response.Metadata["cache_key"] = cacheKey

		if data, err := json.Marshal(response); err == nil {
			_ = m.cache.Set(ctx, cacheKey, data, ttl)
		}
	}

	return response, nil
}

// GenerateCacheKey creates a cache key for a standard generation request.
func (m *Middleware) GenerateCacheKey(modelID string, prompt string, options []core.GenerateOption) string {
	return m.keyGenerator.GenerateKey(modelID, prompt, options)
}

// GenerateJSONCacheKey creates a cache key for JSON-structured generation.
func (m *Middleware) GenerateJSONCacheKey(modelID string, prompt string, schema interface{}, options []core.GenerateOption) string {
	return m.keyGenerator.GenerateJSONKey(modelID, prompt, schema, options)
}

// GenerateContentCacheKey creates a cache key for multimodal content.
func (m *Middleware) GenerateContentCacheKey(modelID string, contents []Content, options []core.GenerateOption) string {
	return m.keyGenerator.GenerateContentKey(modelID, contents, options)
}

// SetEnabled enables or disables caching.
func (m *Middleware) SetEnabled(enabled bool) {
	m.enabled.Store(enabled)
}

// IsEnabled returns whether caching is enabled.
func (m *Middleware) IsEnabled() bool {
	return m.enabled.Load()
}

// Stats returns cache statistics.
func (m *Middleware) Stats() CacheStats {
	if m.cache == nil {
		return CacheStats{}
	}
	return m.cache.Stats()
}

// Clear clears all cached entries.
func (m *Middleware) Clear(ctx context.Context) error {
	if m.cache == nil {
		return nil
	}
	return m.cache.Clear(ctx)
}

// Close closes the cache.
func (m *Middleware) Close() error {
	if m.cache == nil {
		return nil
	}
	return m.cache.Close()
}

// Option is a functional option for configuring cache middleware.
type Option func(*Middleware)

// WithTTL sets the default TTL for cache entries.
func WithTTL(ttl time.Duration) Option {
	return func(m *Middleware) {
		m.ttl = ttl
	}
}

// WithKeyPrefix sets a custom key prefix.
func WithKeyPrefix(prefix string) Option {
	return func(m *Middleware) {
		m.keyGenerator = NewKeyGenerator(prefix)
	}
}

// WithEnabled sets the initial enabled state.
func WithEnabled(enabled bool) Option {
	return func(m *Middleware) {
		m.enabled.Store(enabled)
	}
}
