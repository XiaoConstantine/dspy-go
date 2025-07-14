package cache

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockCache is a mock implementation of Cache interface.
type MockCache struct {
	mock.Mock
}

func (m *MockCache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	args := m.Called(ctx, key)
	return args.Get(0).([]byte), args.Bool(1), args.Error(2)
}

func (m *MockCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	args := m.Called(ctx, key, value, ttl)
	return args.Error(0)
}

func (m *MockCache) Delete(ctx context.Context, key string) error {
	args := m.Called(ctx, key)
	return args.Error(0)
}

func (m *MockCache) Clear(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockCache) Stats() CacheStats {
	args := m.Called()
	return args.Get(0).(CacheStats)
}

func (m *MockCache) Close() error {
	args := m.Called()
	return args.Error(0)
}

func TestNewMiddleware(t *testing.T) {
	mockCache := &MockCache{}
	ttl := time.Hour

	middleware := NewMiddleware(mockCache, ttl)

	assert.NotNil(t, middleware)
	assert.Equal(t, mockCache, middleware.cache)
	assert.Equal(t, ttl, middleware.ttl)
	assert.NotNil(t, middleware.keyGenerator)
	assert.True(t, middleware.enabled.Load())
}

func TestMiddleware_WithCache(t *testing.T) {
	ctx := context.Background()
	cacheKey := "test_key"
	ttl := time.Hour

	t.Run("Cache hit", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache hit
		cachedData := `{"content":"cached response","metadata":{"model":"gpt-4"}}`
		mockCache.On("Get", ctx, cacheKey).Return([]byte(cachedData), true, nil)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "cached response", result.Content)
		assert.Equal(t, true, result.Metadata["cache_hit"])
		assert.Equal(t, cacheKey, result.Metadata["cache_key"])
		assert.Equal(t, 0, callCount) // Function should not be called
		mockCache.AssertExpectations(t)
	})

	t.Run("Cache miss", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache miss
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, cacheKey, mock.AnythingOfType("[]uint8"), ttl).Return(nil)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		assert.Equal(t, cacheKey, result.Metadata["cache_key"])
		assert.Equal(t, 1, callCount) // Function should be called
		mockCache.AssertExpectations(t)
	})

	t.Run("Cache disabled", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)
		middleware.SetEnabled(false)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, 1, callCount) // Function should be called
		// No cache interactions should happen
		mockCache.AssertNotCalled(t, "Get")
		mockCache.AssertNotCalled(t, "Set")
	})

	t.Run("Nil cache", func(t *testing.T) {
		middleware := NewMiddleware(nil, ttl)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, 1, callCount) // Function should be called
	})

	t.Run("Function error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache miss
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, nil)

		expectedErr := errors.New("function error")
		fn := func() (*core.LLMResponse, error) {
			return nil, expectedErr
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Nil(t, result)
		mockCache.AssertExpectations(t)
		mockCache.AssertNotCalled(t, "Set") // Should not cache errors
	})

	t.Run("Cache get error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache error
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, errors.New("cache error"))
		mockCache.On("Set", ctx, cacheKey, mock.AnythingOfType("[]uint8"), ttl).Return(nil)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, 1, callCount) // Function should be called
		mockCache.AssertExpectations(t)
	})

	t.Run("Invalid cached data", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache hit with invalid JSON
		mockCache.On("Get", ctx, cacheKey).Return([]byte("invalid json"), true, nil)
		mockCache.On("Set", ctx, cacheKey, mock.AnythingOfType("[]uint8"), ttl).Return(nil)

		callCount := 0
		fn := func() (*core.LLMResponse, error) {
			callCount++
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, 1, callCount) // Function should be called
		mockCache.AssertExpectations(t)
	})

	t.Run("Zero TTL uses default", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache miss
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, cacheKey, mock.AnythingOfType("[]uint8"), ttl).Return(nil) // Should use default TTL

		fn := func() (*core.LLMResponse, error) {
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, 0, fn) // Zero TTL

		assert.NoError(t, err)
		assert.NotNil(t, result)
		mockCache.AssertExpectations(t)
	})

	t.Run("Nil response", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache miss
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, nil)

		fn := func() (*core.LLMResponse, error) {
			return nil, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.Nil(t, result)
		mockCache.AssertExpectations(t)
		mockCache.AssertNotCalled(t, "Set") // Should not cache nil response
	})
}

func TestMiddleware_GenerateCacheKey(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	key := middleware.GenerateCacheKey("gpt-4", "test prompt", nil)
	assert.NotEmpty(t, key)
	assert.Contains(t, key, "gpt-4")
}

func TestMiddleware_GenerateJSONCacheKey(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	schema := map[string]interface{}{"type": "object"}
	key := middleware.GenerateJSONCacheKey("gpt-4", "test prompt", schema, nil)
	assert.NotEmpty(t, key)
	assert.Contains(t, key, "json_gpt-4")
}

func TestMiddleware_GenerateContentCacheKey(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	contents := []Content{
		{Type: "text", Data: "Hello world"},
	}
	key := middleware.GenerateContentCacheKey("gpt-4", contents, nil)
	assert.NotEmpty(t, key)
	assert.Contains(t, key, "content_gpt-4")
}

func TestMiddleware_SetEnabled(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	assert.True(t, middleware.IsEnabled())

	middleware.SetEnabled(false)
	assert.False(t, middleware.IsEnabled())

	middleware.SetEnabled(true)
	assert.True(t, middleware.IsEnabled())
}

func TestMiddleware_Stats(t *testing.T) {
	t.Run("With cache", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		expectedStats := CacheStats{
			Hits:   100,
			Misses: 50,
		}
		mockCache.On("Stats").Return(expectedStats)

		stats := middleware.Stats()
		assert.Equal(t, expectedStats, stats)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without cache", func(t *testing.T) {
		middleware := NewMiddleware(nil, time.Hour)

		stats := middleware.Stats()
		assert.Equal(t, CacheStats{}, stats)
	})
}

func TestMiddleware_Clear(t *testing.T) {
	ctx := context.Background()

	t.Run("With cache", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		mockCache.On("Clear", ctx).Return(nil)

		err := middleware.Clear(ctx)
		assert.NoError(t, err)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without cache", func(t *testing.T) {
		middleware := NewMiddleware(nil, time.Hour)

		err := middleware.Clear(ctx)
		assert.NoError(t, err)
	})

	t.Run("Clear error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		expectedErr := errors.New("clear error")
		mockCache.On("Clear", ctx).Return(expectedErr)

		err := middleware.Clear(ctx)
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		mockCache.AssertExpectations(t)
	})
}

func TestMiddleware_Close(t *testing.T) {
	t.Run("With cache", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		mockCache.On("Close").Return(nil)

		err := middleware.Close()
		assert.NoError(t, err)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without cache", func(t *testing.T) {
		middleware := NewMiddleware(nil, time.Hour)

		err := middleware.Close()
		assert.NoError(t, err)
	})

	t.Run("Close error", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		expectedErr := errors.New("close error")
		mockCache.On("Close").Return(expectedErr)

		err := middleware.Close()
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		mockCache.AssertExpectations(t)
	})
}

func TestMiddleware_Options(t *testing.T) {
	t.Run("WithTTL", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		newTTL := 2 * time.Hour
		option := WithTTL(newTTL)
		option(middleware)

		assert.Equal(t, newTTL, middleware.ttl)
	})

	t.Run("WithKeyPrefix", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, time.Hour)

		option := WithKeyPrefix("custom_")
		option(middleware)

		assert.Equal(t, "custom_", middleware.keyGenerator.prefix)
	})
}

func TestMiddleware_ConcurrentAccess(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	// Test concurrent enable/disable
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			middleware.SetEnabled(i%2 == 0)
			middleware.IsEnabled()
		}(i)
	}
	wg.Wait()

	// Should not panic and should have some final state
	assert.IsType(t, true, middleware.IsEnabled())
}

func TestMiddleware_WithCacheResponseMetadata(t *testing.T) {
	ctx := context.Background()
	cacheKey := "test_key"
	ttl := time.Hour

	t.Run("Cache hit preserves existing metadata", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache hit with existing metadata
		cachedData := `{"content":"cached response","metadata":{"existing":"value"}}`
		mockCache.On("Get", ctx, cacheKey).Return([]byte(cachedData), true, nil)

		fn := func() (*core.LLMResponse, error) {
			return &core.LLMResponse{Content: "fresh response"}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "cached response", result.Content)
		assert.Equal(t, true, result.Metadata["cache_hit"])
		assert.Equal(t, cacheKey, result.Metadata["cache_key"])
		assert.Equal(t, "value", result.Metadata["existing"])
		mockCache.AssertExpectations(t)
	})

	t.Run("Cache miss preserves existing metadata", func(t *testing.T) {
		mockCache := &MockCache{}
		middleware := NewMiddleware(mockCache, ttl)

		// Mock cache miss
		mockCache.On("Get", ctx, cacheKey).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, cacheKey, mock.AnythingOfType("[]uint8"), ttl).Return(nil)

		fn := func() (*core.LLMResponse, error) {
			return &core.LLMResponse{
				Content: "fresh response",
				Metadata: map[string]interface{}{
					"existing": "value",
				},
			}, nil
		}

		result, err := middleware.WithCache(ctx, cacheKey, ttl, fn)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		assert.Equal(t, cacheKey, result.Metadata["cache_key"])
		assert.Equal(t, "value", result.Metadata["existing"])
		mockCache.AssertExpectations(t)
	})
}

func TestMiddleware_WithEnabled(t *testing.T) {
	mockCache := &MockCache{}
	middleware := NewMiddleware(mockCache, time.Hour)

	t.Run("Enable middleware", func(t *testing.T) {
		option := WithEnabled(true)
		option(middleware)
		assert.True(t, middleware.IsEnabled())
	})

	t.Run("Disable middleware", func(t *testing.T) {
		option := WithEnabled(false)
		option(middleware)
		assert.False(t, middleware.IsEnabled())
	})
}
