package cache

import (
	"context"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// ProviderCache is a helper struct that can be embedded in LLM providers
// to add caching functionality in a Go-idiomatic way.
type ProviderCache struct {
	middleware *Middleware
}

// NewProviderCache creates a new provider cache helper.
func NewProviderCache(cacheConfig *CacheConfig) (*ProviderCache, error) {
	if cacheConfig == nil || cacheConfig.Type == "" {
		// No cache configured
		return &ProviderCache{}, nil
	}

	cache, err := NewCache(*cacheConfig)
	if err != nil {
		return nil, err
	}

	middleware := NewMiddleware(cache, cacheConfig.DefaultTTL)
	return &ProviderCache{
		middleware: middleware,
	}, nil
}

// CacheGenerate wraps a Generate call with caching.
func (pc *ProviderCache) CacheGenerate(
	ctx context.Context,
	modelID string,
	prompt string,
	options []core.GenerateOption,
	generateFn func() (*core.LLMResponse, error),
) (*core.LLMResponse, error) {
	if pc.middleware == nil {
		return generateFn()
	}

	key := pc.middleware.GenerateCacheKey(modelID, prompt, options)
	return pc.middleware.WithCache(ctx, key, 0, generateFn)
}

// CacheGenerateJSON wraps a GenerateWithJSON call with caching.
func (pc *ProviderCache) CacheGenerateJSON(
	ctx context.Context,
	modelID string,
	prompt string,
	schema interface{},
	options []core.GenerateOption,
	generateFn func() (*core.LLMResponse, error),
) (*core.LLMResponse, error) {
	if pc.middleware == nil {
		return generateFn()
	}

	key := pc.middleware.GenerateJSONCacheKey(modelID, prompt, schema, options)
	return pc.middleware.WithCache(ctx, key, 0, generateFn)
}

// CacheGenerateContent wraps a GenerateWithContent call with caching.
func (pc *ProviderCache) CacheGenerateContent(
	ctx context.Context,
	modelID string,
	contents []Content,
	options []core.GenerateOption,
	generateFn func() (*core.LLMResponse, error),
) (*core.LLMResponse, error) {
	if pc.middleware == nil {
		return generateFn()
	}

	key := pc.middleware.GenerateContentCacheKey(modelID, contents, options)
	return pc.middleware.WithCache(ctx, key, 0, generateFn)
}

// SetCacheEnabled enables or disables caching.
func (pc *ProviderCache) SetCacheEnabled(enabled bool) {
	if pc.middleware != nil {
		pc.middleware.SetEnabled(enabled)
	}
}

// CacheStats returns cache statistics.
func (pc *ProviderCache) CacheStats() CacheStats {
	if pc.middleware == nil {
		return CacheStats{}
	}
	return pc.middleware.Stats()
}

// ClearCache clears the cache.
func (pc *ProviderCache) ClearCache(ctx context.Context) error {
	if pc.middleware == nil {
		return nil
	}
	return pc.middleware.Clear(ctx)
}

// Close closes the cache.
func (pc *ProviderCache) Close() error {
	if pc.middleware == nil {
		return nil
	}
	return pc.middleware.Close()
}

// ProviderOption is a functional option for configuring providers with caching.
type ProviderOption func(*ProviderCache)

// WithCache enables caching with the given configuration.
func WithCache(config CacheConfig) ProviderOption {
	return func(pc *ProviderCache) {
		cache, err := NewProviderCache(&config)
		if err == nil {
			*pc = *cache
		}
	}
}

// WithCacheMiddleware uses an existing cache middleware.
func WithCacheMiddleware(middleware *Middleware) ProviderOption {
	return func(pc *ProviderCache) {
		pc.middleware = middleware
	}
}

// Example of how a provider would use this:
//
// type GeminiProvider struct {
//     apiKey string
//     model  string
//     cache  *cache.ProviderCache
// }
//
// func NewGeminiProvider(apiKey, model string, opts ...cache.ProviderOption) *GeminiProvider {
//     p := &GeminiProvider{
//         apiKey: apiKey,
//         model:  model,
//         cache:  &cache.ProviderCache{},
//     }
//     
//     for _, opt := range opts {
//         opt(p.cache)
//     }
//     
//     return p
// }
//
// func (p *GeminiProvider) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
//     return p.cache.CacheGenerate(ctx, p.model, prompt, options, func() (*core.LLMResponse, error) {
//         // Actual Gemini API call here
//         return p.doGenerate(ctx, prompt, options...)
//     })
// }

// CacheContext allows passing cache configuration through context.
type cacheContextKey struct{}

// WithCacheContext adds cache configuration to context.
func WithCacheContext(ctx context.Context, cache Cache, ttl time.Duration) context.Context {
	return context.WithValue(ctx, cacheContextKey{}, &Middleware{
		cache:        cache,
		keyGenerator: NewKeyGenerator("dspy_"),
		ttl:          ttl,
	})
}

// CacheFromContext retrieves cache middleware from context.
func CacheFromContext(ctx context.Context) *Middleware {
	if v := ctx.Value(cacheContextKey{}); v != nil {
		return v.(*Middleware)
	}
	return nil
}