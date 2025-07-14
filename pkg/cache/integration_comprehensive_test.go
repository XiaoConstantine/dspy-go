package cache

import (
	"context"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestCachedLLM_StreamingMethods(t *testing.T) {
	mockLLM := &MockLLM{}
	mockCache := &MockCache{}

	cachedLLM := &CachedLLM{
		LLM:          mockLLM,
		cache:        mockCache,
		keyGenerator: NewKeyGenerator("test_"),
		ttl:          time.Hour,
		enabled:      true,
	}

	ctx := context.Background()

	t.Run("StreamGenerate", func(t *testing.T) {
		prompt := "test prompt"
		ch := make(chan core.StreamChunk, 1)
		ch <- core.StreamChunk{Content: "test", Done: true}
		close(ch)
		expectedResponse := &core.StreamResponse{
			ChunkChannel: ch,
			Cancel:       func() {},
		}

		mockLLM.On("StreamGenerate", ctx, prompt, mock.AnythingOfType("[]core.GenerateOption")).Return(expectedResponse, nil)

		result, err := cachedLLM.StreamGenerate(ctx, prompt)
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, result)

		mockLLM.AssertExpectations(t)
		// Streaming methods should not interact with cache
		mockCache.AssertNotCalled(t, "Get")
		mockCache.AssertNotCalled(t, "Set")
	})

	t.Run("StreamGenerateWithContent", func(t *testing.T) {
		content := []core.ContentBlock{
			{Type: "text", Text: "Hello world"},
		}
		ch := make(chan core.StreamChunk, 1)
		ch <- core.StreamChunk{Content: "test", Done: true}
		close(ch)
		expectedResponse := &core.StreamResponse{
			ChunkChannel: ch,
			Cancel:       func() {},
		}

		mockLLM.On("StreamGenerateWithContent", ctx, content, mock.AnythingOfType("[]core.GenerateOption")).Return(expectedResponse, nil)

		result, err := cachedLLM.StreamGenerateWithContent(ctx, content)
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, result)

		mockLLM.AssertExpectations(t)
		// Streaming methods should not interact with cache
		mockCache.AssertNotCalled(t, "Get")
		mockCache.AssertNotCalled(t, "Set")
	})
}

func TestCachedLLM_EmbeddingMethods(t *testing.T) {
	mockLLM := &MockLLM{}

	cachedLLM := &CachedLLM{
		LLM:          mockLLM,
		cache:        nil, // No cache for embedding methods
		keyGenerator: NewKeyGenerator("test_"),
		ttl:          time.Hour,
		enabled:      true,
	}

	ctx := context.Background()

	t.Run("CreateEmbedding", func(t *testing.T) {
		input := "test input"
		expectedResult := &core.EmbeddingResult{
			Vector:     []float32{0.1, 0.2, 0.3},
			TokenCount: 10,
		}

		mockLLM.On("CreateEmbedding", ctx, input, mock.AnythingOfType("[]core.EmbeddingOption")).Return(expectedResult, nil)

		result, err := cachedLLM.CreateEmbedding(ctx, input)
		assert.NoError(t, err)
		assert.Equal(t, expectedResult, result)

		mockLLM.AssertExpectations(t)
	})

	t.Run("CreateEmbeddings", func(t *testing.T) {
		inputs := []string{"input1", "input2"}
		expectedResult := &core.BatchEmbeddingResult{
			Embeddings: []core.EmbeddingResult{
				{Vector: []float32{0.1, 0.2, 0.3}, TokenCount: 10},
				{Vector: []float32{0.4, 0.5, 0.6}, TokenCount: 10},
			},
		}

		mockLLM.On("CreateEmbeddings", ctx, inputs, mock.AnythingOfType("[]core.EmbeddingOption")).Return(expectedResult, nil)

		result, err := cachedLLM.CreateEmbeddings(ctx, inputs)
		assert.NoError(t, err)
		assert.Equal(t, expectedResult, result)

		mockLLM.AssertExpectations(t)
	})
}

func TestCachedLLM_GenerateWithFunctions(t *testing.T) {
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

	ctx := context.Background()
	prompt := "test prompt"
	functions := []map[string]interface{}{
		{
			"name": "test_function",
			"description": "A test function",
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"param1": map[string]interface{}{
						"type": "string",
					},
				},
			},
		},
	}

	t.Run("Cache miss", func(t *testing.T) {
		expectedResult := map[string]interface{}{
			"function_call": map[string]interface{}{
				"name": "test_function",
				"arguments": `{"param1": "value1"}`,
			},
		}

		// Mock cache miss
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(nil), false, nil)
		mockCache.On("Set", ctx, mock.AnythingOfType("string"), mock.AnythingOfType("[]uint8"), time.Hour).Return(nil)

		mockLLM.On("GenerateWithFunctions", ctx, prompt, functions, mock.AnythingOfType("[]core.GenerateOption")).Return(expectedResult, nil)

		result, err := cachedLLM.GenerateWithFunctions(ctx, prompt, functions)
		assert.NoError(t, err)
		assert.Equal(t, expectedResult, result)

		mockCache.AssertExpectations(t)
		mockLLM.AssertExpectations(t)
	})

	t.Run("Cache hit", func(t *testing.T) {
		cachedResult := `{"function_call":{"name":"test_function","arguments":"{\"param1\":\"cached_value\"}"}}`

		// Mock cache hit
		mockCache.On("Get", ctx, mock.AnythingOfType("string")).Return([]byte(cachedResult), true, nil)

		result, err := cachedLLM.GenerateWithFunctions(ctx, prompt, functions)
		assert.NoError(t, err)
		
		// Verify the result structure
		assert.Contains(t, result, "function_call")

		mockCache.AssertExpectations(t)
		mockLLM.AssertNotCalled(t, "GenerateWithFunctions")
	})

	t.Run("Cache disabled", func(t *testing.T) {
		disabledCachedLLM := &CachedLLM{
			LLM:          mockLLM,
			cache:        nil,
			keyGenerator: NewKeyGenerator("test_"),
			ttl:          time.Hour,
			enabled:      false,
		}

		expectedResult := map[string]interface{}{
			"function_call": map[string]interface{}{
				"name": "test_function",
				"arguments": `{"param1": "value1"}`,
			},
		}

		mockLLM.On("GenerateWithFunctions", ctx, prompt, functions, mock.AnythingOfType("[]core.GenerateOption")).Return(expectedResult, nil)

		result, err := disabledCachedLLM.GenerateWithFunctions(ctx, prompt, functions)
		assert.NoError(t, err)
		assert.Equal(t, expectedResult, result)

		mockLLM.AssertExpectations(t)
	})
}

func TestCachedLLM_CacheStats(t *testing.T) {
	t.Run("With cache", func(t *testing.T) {
		mockCache := &MockCache{}
		expectedStats := CacheStats{
			Hits:    100,
			Misses:  50,
			Sets:    75,
			Deletes: 10,
			Size:    1024,
			MaxSize: 2048,
		}

		cachedLLM := &CachedLLM{
			cache: mockCache,
		}

		mockCache.On("Stats").Return(expectedStats)

		stats := cachedLLM.CacheStats()
		assert.Equal(t, expectedStats, stats)

		mockCache.AssertExpectations(t)
	})

	t.Run("Without cache", func(t *testing.T) {
		cachedLLM := &CachedLLM{
			cache: nil,
		}

		stats := cachedLLM.CacheStats()
		assert.Equal(t, CacheStats{}, stats)
	})
}

func TestGlobalCacheFunctions(t *testing.T) {
	// Clean up any existing global cache
	resetGlobalCache()

	ctx := context.Background()

	t.Run("ClearGlobalCache with no cache", func(t *testing.T) {
		err := ClearGlobalCache(ctx)
		assert.NoError(t, err)
	})

	t.Run("GetGlobalCacheStats with no cache", func(t *testing.T) {
		stats := GetGlobalCacheStats()
		assert.Equal(t, CacheStats{}, stats)
	})

	t.Run("SetGlobalCacheEnabled false with no cache", func(t *testing.T) {
		SetGlobalCacheEnabled(false)
		// Should not panic
	})

	t.Run("Global cache operations with cache", func(t *testing.T) {
		// Create a global cache
		config := CacheConfig{
			Type:    "memory",
			MaxSize: 1024,
			MemoryConfig: MemoryConfig{
				CleanupInterval: time.Minute,
			},
		}

		cache := getOrCreateGlobalCache(config)
		assert.NotNil(t, cache)

		// Set some data
		err := cache.Set(ctx, "global-test", []byte("global-value"), 0)
		assert.NoError(t, err)

		// Test ClearGlobalCache
		err = ClearGlobalCache(ctx)
		assert.NoError(t, err)

		// Verify cleared
		_, found, err := cache.Get(ctx, "global-test")
		assert.NoError(t, err)
		assert.False(t, found)

		// Test GetGlobalCacheStats
		stats := GetGlobalCacheStats()
		assert.Equal(t, int64(1024), stats.MaxSize)

		// Test SetGlobalCacheEnabled false
		SetGlobalCacheEnabled(false)
		
		// Clean up
		resetGlobalCache()
	})
}

func TestWrapWithCache_EdgeCases(t *testing.T) {
	mockLLM := &MockLLM{}
	mockLLM.On("ModelID").Return("test-model")

	t.Run("WrapWithCache with nil config", func(t *testing.T) {
		// Should create cached LLM with default config
		result := WrapWithCache(mockLLM, nil)
		assert.IsType(t, &CachedLLM{}, result)
	})
}