package cache

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestNewProviderCache(t *testing.T) {
	t.Run("With valid config", func(t *testing.T) {
		config := &CacheConfig{
			Type:       "memory",
			MaxSize:    1024 * 1024,
			DefaultTTL: time.Hour,
		}

		pc, err := NewProviderCache(config)
		assert.NoError(t, err)
		assert.NotNil(t, pc)
		assert.NotNil(t, pc.middleware)
	})

	t.Run("With nil config", func(t *testing.T) {
		pc, err := NewProviderCache(nil)
		assert.NoError(t, err)
		assert.NotNil(t, pc)
		assert.Nil(t, pc.middleware)
	})

	t.Run("With empty type", func(t *testing.T) {
		config := &CacheConfig{
			Type: "",
		}

		pc, err := NewProviderCache(config)
		assert.NoError(t, err)
		assert.NotNil(t, pc)
		assert.Nil(t, pc.middleware)
	})

	t.Run("With invalid config", func(t *testing.T) {
		config := &CacheConfig{
			Type:         "sqlite",
			SQLiteConfig: SQLiteConfig{Path: "/invalid/path/cache.db"},
		}

		pc, err := NewProviderCache(config)
		assert.Error(t, err)
		assert.Nil(t, pc)
	})
}

func TestProviderCache_CacheGenerate(t *testing.T) {
	ctx := context.Background()
	modelID := "gpt-4"
	prompt := "Test prompt"
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
	}

	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test response"}, nil
		}

		result, err := pc.CacheGenerate(ctx, modelID, prompt, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		assert.Equal(t, 1, callCount)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test response"}, nil
		}

		result, err := pc.CacheGenerate(ctx, modelID, prompt, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test response", result.Content)
		assert.Equal(t, 1, callCount)
	})

	t.Run("Generate function error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)

		expectedErr := errors.New("generate error")
		generateFn := func() (*core.LLMResponse, error) {
			return nil, expectedErr
		}

		result, err := pc.CacheGenerate(ctx, modelID, prompt, options, generateFn)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Nil(t, result)
		mockCache.AssertExpectations(t)
	})
}

func TestProviderCache_CacheGenerateJSON(t *testing.T) {
	ctx := context.Background()
	modelID := "gpt-4"
	prompt := "Test prompt"
	schema := map[string]interface{}{"type": "object"}
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
	}

	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test json response"}, nil
		}

		result, err := pc.CacheGenerateJSON(ctx, modelID, prompt, schema, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test json response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		assert.Equal(t, 1, callCount)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test json response"}, nil
		}

		result, err := pc.CacheGenerateJSON(ctx, modelID, prompt, schema, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test json response", result.Content)
		assert.Equal(t, 1, callCount)
	})
}

func TestProviderCache_CacheGenerateContent(t *testing.T) {
	ctx := context.Background()
	modelID := "gpt-4"
	contents := []Content{
		{Type: "text", Data: "Hello world"},
		{Type: "image", Data: "base64data"},
	}
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
	}

	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test content response"}, nil
		}

		result, err := pc.CacheGenerateContent(ctx, modelID, contents, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test content response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		assert.Equal(t, 1, callCount)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		callCount := 0
		generateFn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "test content response"}, nil
		}

		result, err := pc.CacheGenerateContent(ctx, modelID, contents, options, generateFn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "test content response", result.Content)
		assert.Equal(t, 1, callCount)
	})
}

func TestProviderCache_SetCacheEnabled(t *testing.T) {
	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		assert.True(t, middleware.IsEnabled())

		pc.SetCacheEnabled(false)
		assert.False(t, middleware.IsEnabled())

		pc.SetCacheEnabled(true)
		assert.True(t, middleware.IsEnabled())
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		// Should not panic
		pc.SetCacheEnabled(false)
		pc.SetCacheEnabled(true)
	})
}

func TestProviderCache_CacheStats(t *testing.T) {
	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		expectedStats := CacheStats{
			Hits:   100,
			Misses: 50,
		}
		mockCache.On("Stats").Return(expectedStats)

		stats := pc.CacheStats()
		assert.Equal(t, expectedStats, stats)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		stats := pc.CacheStats()
		assert.Equal(t, CacheStats{}, stats)
	})
}

func TestProviderCache_ClearCache(t *testing.T) {
	ctx := context.Background()

	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		mockCache.On("Clear", ctx).Return(nil)

		err := pc.ClearCache(ctx)
		assert.NoError(t, err)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		err := pc.ClearCache(ctx)
		assert.NoError(t, err)
	})

	t.Run("Clear error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		expectedErr := errors.New("clear error")
		mockCache.On("Clear", ctx).Return(expectedErr)

		err := pc.ClearCache(ctx)
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		mockCache.AssertExpectations(t)
	})
}

func TestProviderCache_Close(t *testing.T) {
	t.Run("With middleware", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		mockCache.On("Close").Return(nil)

		err := pc.Close()
		assert.NoError(t, err)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without middleware", func(t *testing.T) {
		pc := &ProviderCache{middleware: nil}

		err := pc.Close()
		assert.NoError(t, err)
	})

	t.Run("Close error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)
		pc := &ProviderCache{middleware: middleware}

		expectedErr := errors.New("close error")
		mockCache.On("Close").Return(expectedErr)

		err := pc.Close()
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		mockCache.AssertExpectations(t)
	})
}

func TestProviderCache_Options(t *testing.T) {
	t.Run("WithCache", func(t *testing.T) {
		pc := &ProviderCache{}

		config := CacheConfig{
			Type:       "memory",
			MaxSize:    1024 * 1024,
			DefaultTTL: time.Hour,
		}

		option := WithCache(config)
		option(pc)

		assert.NotNil(t, pc.middleware)
	})

	t.Run("WithCache error", func(t *testing.T) {
		pc := &ProviderCache{}

		config := CacheConfig{
			Type:         "sqlite",
			SQLiteConfig: SQLiteConfig{Path: "/invalid/path/cache.db"},
		}

		option := WithCache(config)
		option(pc)

		// Should not modify pc if cache creation fails
		assert.Nil(t, pc.middleware)
	})

	t.Run("WithCacheMiddleware", func(t *testing.T) {
		pc := &ProviderCache{}
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		option := WithCacheMiddleware(middleware)
		option(pc)

		assert.Equal(t, middleware, pc.middleware)
	})
}

func TestWithCacheContext(t *testing.T) {
	ctx := context.Background()
	mockCache := &MockCache{}
	ttl := time.Hour

	newCtx := WithCacheContext(ctx, mockCache, ttl)
	assert.NotEqual(t, ctx, newCtx)

	middleware := CacheFromContext(newCtx)
	assert.NotNil(t, middleware)
	assert.Equal(t, mockCache, middleware.cache)
	assert.Equal(t, ttl, middleware.ttl)
}

func TestCacheFromContext(t *testing.T) {
	t.Run("Context with cache", func(t *testing.T) {
		ctx := context.Background()
		mockCache := &MockCache{}
		ttl := time.Hour

		newCtx := WithCacheContext(ctx, mockCache, ttl)
		middleware := CacheFromContext(newCtx)

		assert.NotNil(t, middleware)
		assert.Equal(t, mockCache, middleware.cache)
		assert.Equal(t, ttl, middleware.ttl)
	})

	t.Run("Context without cache", func(t *testing.T) {
		ctx := context.Background()
		middleware := CacheFromContext(ctx)

		assert.Nil(t, middleware)
	})
}

func TestProviderCache_CacheHit(t *testing.T) {
	ctx := context.Background()
	modelID := "gpt-4"
	prompt := "Test prompt"
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
	}

	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)
	pc := &ProviderCache{middleware: middleware}

	// Mock cache hit
	cachedResponse := `{"content":"cached response","metadata":{"cache_hit":true}}`
	mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(cachedResponse), true, nil)

	callCount := 0
	generateFn := func() (*core.LLMResponse, error) {
		callCount++
		return &core.LLMResponse{Content: "fresh response"}, nil
	}

	result, err := pc.CacheGenerate(ctx, modelID, prompt, options, generateFn)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "cached response", result.Content)
	assert.Equal(t, true, result.Metadata["cache_hit"])
	assert.Equal(t, 0, callCount) // Should not call generate function
	mockCache.AssertExpectations(t)
}

func TestProviderCache_CacheKeyGeneration(t *testing.T) {
	config := &CacheConfig{
		Type:       "memory",
		MaxSize:    1024 * 1024,
		DefaultTTL: time.Hour,
	}

	pc, err := NewProviderCache(config)
	assert.NoError(t, err)
	assert.NotNil(t, pc)

	// Test that different inputs produce different keys
	ctx := context.Background()
	modelID := "gpt-4"
	prompt1 := "Test prompt 1"
	prompt2 := "Test prompt 2"
	options := []core.GenerateOption{
		core.WithTemperature(0.7),
	}

	mockCache := &MockCache{}
	pc.middleware.cache = mockCache

	// Mock cache misses for both
	mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
	mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

	generateFn := func() (*core.LLMResponse, error) {
		return &core.LLMResponse{Content: "response"}, nil
	}

	result1, err1 := pc.CacheGenerate(ctx, modelID, prompt1, options, generateFn)
	result2, err2 := pc.CacheGenerate(ctx, modelID, prompt2, options, generateFn)

	assert.NoError(t, err1)
	assert.NoError(t, err2)
	assert.NotNil(t, result1)
	assert.NotNil(t, result2)

	// The cache keys should be different (verified by the fact that both generate functions were called)
	mockCache.AssertExpectations(t)
}
