package cache

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockLLM is a mock implementation of core.LLM for testing.
type MockLLM struct {
	mock.Mock
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

func (m *MockLLM) CreateEmbedding(ctx context.Context, input string, options ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	args := m.Called(ctx, input, options)
	return args.Get(0).(*core.EmbeddingResult), args.Error(1)
}

func (m *MockLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	args := m.Called(ctx, inputs, options)
	return args.Get(0).(*core.BatchEmbeddingResult), args.Error(1)
}

func (m *MockLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(*core.StreamResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	args := m.Called(ctx, content, options)
	return args.Get(0).(*core.LLMResponse), args.Error(1)
}

func (m *MockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...core.GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, functions, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

func (m *MockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	args := m.Called(ctx, content, options)
	return args.Get(0).(*core.StreamResponse), args.Error(1)
}

func (m *MockLLM) ProviderName() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLLM) ModelID() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockLLM) GetMaxTokens() int {
	args := m.Called()
	return args.Int(0)
}

func (m *MockLLM) GetContextWindow() int {
	args := m.Called()
	return args.Int(0)
}

func (m *MockLLM) Capabilities() []core.Capability {
	args := m.Called()
	return args.Get(0).([]core.Capability)
}

func (m *MockLLM) String() string {
	args := m.Called()
	return args.String(0)
}

func TestWrapWithCache(t *testing.T) {
	// Clean up global state
	resetGlobalCache()

	t.Run("Caching enabled", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")

		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "memory",
			TTL:     time.Hour,
			MaxSize: 1024 * 1024,
		}

		cachedLLM := WrapWithCache(mockLLM, fileConfig)
		assert.NotNil(t, cachedLLM)
		assert.IsType(t, &CachedLLM{}, cachedLLM)

		// Verify the underlying LLM is wrapped
		cached := cachedLLM.(*CachedLLM)
		assert.Equal(t, mockLLM, cached.LLM)
		assert.True(t, cached.enabled)
		assert.NotNil(t, cached.cache)
		assert.NotNil(t, cached.keyGenerator)
	})

	t.Run("Caching disabled", func(t *testing.T) {
		mockLLM := &MockLLM{}

		fileConfig := &config.CachingConfig{
			Enabled: false,
		}

		cachedLLM := WrapWithCache(mockLLM, fileConfig)
		assert.Equal(t, mockLLM, cachedLLM) // Should return original LLM
	})

	t.Run("Nil config", func(t *testing.T) {
		mockLLM := &MockLLM{}

		cachedLLM := WrapWithCache(mockLLM, nil)
		assert.IsType(t, &CachedLLM{}, cachedLLM) // Should use default config
	})

	t.Run("Invalid cache type defaults to memory", func(t *testing.T) {
		mockLLM := &MockLLM{}

		fileConfig := &config.CachingConfig{
			Enabled: true,
			Type:    "invalid",
		}

		// Reset global cache to ensure clean state
		resetGlobalCache()

		cachedLLM := WrapWithCache(mockLLM, fileConfig)
		// Should return cached LLM since invalid types default to memory
		assert.IsType(t, &CachedLLM{}, cachedLLM)
	})
}

func TestCachedLLM_Generate(t *testing.T) {
	ctx := context.Background()
	prompt := "Test prompt"

	t.Run("Cache hit", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache hit
		cachedResponse := `{"content":"cached response","metadata":{"cache_hit":true}}`
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(cachedResponse), true, nil)

		result, err := cachedLLM.Generate(ctx, prompt)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "cached response", result.Content)
		assert.Equal(t, true, result.Metadata["cache_hit"])
		mockCache.AssertExpectations(t)
		mockLLM.AssertNotCalled(t, "Generate") // Should not call underlying LLM
	})

	t.Run("Cache miss", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockLLM.On("Generate", ctx, prompt, mock.AnythingOfType("[]core.GenerateOption")).Return(
			&core.LLMResponse{Content: "fresh response"}, nil)
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		result, err := cachedLLM.Generate(ctx, prompt)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		assert.Equal(t, false, result.Metadata["cache_hit"])
		mockCache.AssertExpectations(t)
		mockLLM.AssertExpectations(t)
	})

	t.Run("Cache disabled", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("Generate", ctx, prompt, mock.AnythingOfType("[]core.GenerateOption")).Return(
			&core.LLMResponse{Content: "fresh response"}, nil)

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        nil,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      false,
		}

		result, err := cachedLLM.Generate(ctx, prompt)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh response", result.Content)
		mockLLM.AssertExpectations(t)
	})
}

func TestCachedLLM_GenerateWithJSON(t *testing.T) {
	ctx := context.Background()
	prompt := "Test prompt"

	t.Run("Cache hit", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache hit
		cachedResponse := `{"result":"cached json"}`
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(cachedResponse), true, nil)

		result, err := cachedLLM.GenerateWithJSON(ctx, prompt)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "cached json", result["result"])
		mockCache.AssertExpectations(t)
		mockLLM.AssertNotCalled(t, "GenerateWithJSON")
	})

	t.Run("Cache miss", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockLLM.On("GenerateWithJSON", ctx, prompt, mock.AnythingOfType("[]core.GenerateOption")).Return(
			map[string]interface{}{"result": "fresh json"}, nil)
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		result, err := cachedLLM.GenerateWithJSON(ctx, prompt)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh json", result["result"])
		mockCache.AssertExpectations(t)
		mockLLM.AssertExpectations(t)
	})
}

func TestCachedLLM_GenerateWithContent(t *testing.T) {
	ctx := context.Background()
	content := []core.ContentBlock{
		{Type: "text", Text: "Hello world"},
	}

	t.Run("Cache hit", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache hit
		cachedResponse := `{"content":"cached content response"}`
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(cachedResponse), true, nil)

		result, err := cachedLLM.GenerateWithContent(ctx, content)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "cached content response", result.Content)
		mockCache.AssertExpectations(t)
		mockLLM.AssertNotCalled(t, "GenerateWithContent")
	})

	t.Run("Cache miss", func(t *testing.T) {
		mockLLM := &MockLLM{}
		mockLLM.On("ModelID").Return("gpt-4")
		mockLLM.On("GenerateWithContent", ctx, content, mock.AnythingOfType("[]core.GenerateOption")).Return(
			&core.LLMResponse{Content: "fresh content response"}, nil)
		mockCache := &MockCache{}

		cachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        mockCache,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      true,
		}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		result, err := cachedLLM.GenerateWithContent(ctx, content)

		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, "fresh content response", result.Content)
		mockCache.AssertExpectations(t)
		mockLLM.AssertExpectations(t)
	})
}

func TestCachedLLM_Unwrap(t *testing.T) {
	mockLLM := &MockLLM{}
	cachedLLM := &CachedLLM{
		LLM: mockLLM,
	}

	unwrapped := cachedLLM.Unwrap()
	assert.Equal(t, mockLLM, unwrapped)
}

func TestCachedLLM_SetCacheEnabled(t *testing.T) {
	cachedLLM := &CachedLLM{}

	cachedLLM.SetCacheEnabled(true)
	assert.True(t, cachedLLM.enabled)

	cachedLLM.SetCacheEnabled(false)
	assert.False(t, cachedLLM.enabled)
}

func TestCachedLLM_IsCacheEnabled(t *testing.T) {
	t.Run("Enabled with cache", func(t *testing.T) {
		mockCache := &MockCache{}
		cachedLLM := &CachedLLM{
			cache:   mockCache,
			enabled: true,
		}

		assert.True(t, cachedLLM.IsCacheEnabled())
	})

	t.Run("Disabled", func(t *testing.T) {
		mockCache := &MockCache{}
		cachedLLM := &CachedLLM{
			cache:   mockCache,
			enabled: false,
		}

		assert.False(t, cachedLLM.IsCacheEnabled())
	})

	t.Run("Enabled without cache", func(t *testing.T) {
		cachedLLM := &CachedLLM{
			cache:   nil,
			enabled: true,
		}

		assert.False(t, cachedLLM.IsCacheEnabled())
	})
}

func TestCachedLLM_ClearCache(t *testing.T) {
	ctx := context.Background()

	t.Run("With cache", func(t *testing.T) {
		mockCache := &MockCache{}
		cachedLLM := &CachedLLM{
			cache: mockCache,
		}

		mockCache.On("Clear", ctx).Return(nil)

		err := cachedLLM.ClearCache(ctx)
		assert.NoError(t, err)
		mockCache.AssertExpectations(t)
	})

	t.Run("Without cache", func(t *testing.T) {
		cachedLLM := &CachedLLM{
			cache: nil,
		}

		err := cachedLLM.ClearCache(ctx)
		assert.NoError(t, err)
	})
}

func TestCachedLLM_PassthroughMethods(t *testing.T) {
	mockLLM := &MockLLM{}
	cachedLLM := &CachedLLM{
		LLM: mockLLM,
	}

	// Test passthrough methods
	mockLLM.On("ProviderName").Return("test-provider")
	mockLLM.On("ModelID").Return("test-model")
	mockLLM.On("Capabilities").Return([]core.Capability{core.CapabilityCompletion})

	assert.Equal(t, "test-provider", cachedLLM.ProviderName())
	assert.Equal(t, "test-model", cachedLLM.ModelID())
	assert.Equal(t, []core.Capability{core.CapabilityCompletion}, cachedLLM.Capabilities())

	mockLLM.AssertExpectations(t)
}

func TestGetOrCreateGlobalCache(t *testing.T) {
	// Clean up global state
	resetGlobalCache()

	t.Run("Create new cache", func(t *testing.T) {
		config := CacheConfig{
			Type:       "memory",
			MaxSize:    1024 * 1024,
			DefaultTTL: time.Hour,
		}

		cache := getOrCreateGlobalCache(config)
		assert.NotNil(t, cache)
		assert.IsType(t, &MemoryCache{}, cache)
	})

	t.Run("Reuse existing cache", func(t *testing.T) {
		config := CacheConfig{
			Type:       "memory",
			MaxSize:    1024 * 1024,
			DefaultTTL: time.Hour,
		}

		cache1 := getOrCreateGlobalCache(config)
		cache2 := getOrCreateGlobalCache(config)
		assert.Same(t, cache1, cache2)
	})

	t.Run("Recreate cache on config change", func(t *testing.T) {
		config1 := CacheConfig{
			Type:       "memory",
			MaxSize:    1024 * 1024,
			DefaultTTL: time.Hour,
		}

		config2 := CacheConfig{
			Type:       "memory",
			MaxSize:    2048 * 1024,
			DefaultTTL: time.Hour,
		}

		cache1 := getOrCreateGlobalCache(config1)
		cache2 := getOrCreateGlobalCache(config2)
		assert.NotSame(t, cache1, cache2)
	})

	t.Run("Returns nil on cache creation error", func(t *testing.T) {
		config := CacheConfig{
			Type:         "sqlite",
			SQLiteConfig: SQLiteConfig{Path: "/invalid/path/cache.db"},
		}

		cache := getOrCreateGlobalCache(config)
		assert.Nil(t, cache)
	})
}

func TestConfigEqual(t *testing.T) {
	config1 := CacheConfig{
		Type:       "memory",
		MaxSize:    1024,
		DefaultTTL: time.Hour,
		SQLiteConfig: SQLiteConfig{
			Path: "/tmp/cache.db",
		},
	}

	config2 := CacheConfig{
		Type:       "memory",
		MaxSize:    1024,
		DefaultTTL: time.Hour,
		SQLiteConfig: SQLiteConfig{
			Path: "/tmp/cache.db",
		},
	}

	config3 := CacheConfig{
		Type:       "sqlite",
		MaxSize:    1024,
		DefaultTTL: time.Hour,
		SQLiteConfig: SQLiteConfig{
			Path: "/tmp/cache.db",
		},
	}

	assert.True(t, configEqual(config1, config2))
	assert.False(t, configEqual(config1, config3))
}

func TestCachedLLM_WithCache_ConcurrentAccess(t *testing.T) {
	mockLLM := &MockLLM{}
	mockLLM.On("ModelID").Return("gpt-4")
	mockCache := &MockCache{}

	cachedLLM := &CachedLLM{
		LLM:          mockLLM,
		cache:        mockCache,
		keyGenerator: NewKeyGenerator("test_"),
		ttl:          time.Hour,
		enabled:      true,
	}

	// Mock cache operations for concurrent access
	mockCache.On("Get", mock.Anything, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
	mockCache.On("Set", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)
	mockLLM.On("Generate", mock.Anything, mock.AnythingOfType("string"), mock.AnythingOfType("[]core.GenerateOption")).Return(
		&core.LLMResponse{Content: "response"}, nil)

	// Test concurrent access using Go 1.25's WaitGroup.Go()
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Go(func() {
			ctx := context.Background()
			result, err := cachedLLM.Generate(ctx, "test prompt")
			assert.NoError(t, err)
			assert.NotNil(t, result)
		})
	}
	wg.Wait()

	// Should not panic and should have called the underlying methods
	mockLLM.AssertExpectations(t)
	mockCache.AssertExpectations(t)
}

// resetGlobalCache clears the global cache state for testing.
func resetGlobalCache() {
	globalCacheMu.Lock()
	defer globalCacheMu.Unlock()

	if globalCacheInstance != nil {
		globalCacheInstance.Close()
		globalCacheInstance = nil
	}
	globalCacheConfig = CacheConfig{}
}
