package core

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockStreamConfig struct {
	Content      string
	ChunkSize    int
	ChunkDelay   time.Duration
	Error        error
	ErrorAfter   int
	TokenCounts  *TokenInfo
	ShouldCancel bool
}

// MockBaseLLM implements the LLM interface for testing.
type MockBaseLLM struct {
	mock.Mock
}

func (m *MockBaseLLM) Generate(ctx context.Context, prompt string, options ...GenerateOption) (*LLMResponse, error) {
	args := m.Called(ctx, prompt, options)
	if resp, ok := args.Get(0).(*LLMResponse); ok {
		return resp, args.Error(1)
	}
	return nil, args.Error(1)
}

func (m *MockBaseLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, options ...GenerateOption) (map[string]interface{}, error) {
	return nil, nil
}

func (m *MockBaseLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...GenerateOption) (map[string]interface{}, error) {
	args := m.Called(ctx, prompt, options)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

// CreateEmbedding mocks the single embedding creation following the same pattern as Generate.
func (m *MockBaseLLM) CreateEmbedding(ctx context.Context, input string, options ...EmbeddingOption) (*EmbeddingResult, error) {
	// Record the method call and get the mock results
	args := m.Called(ctx, input, options)

	// Handle nil case first - if first argument is nil, return error
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	// Check if we got a properly structured EmbeddingResult
	if result, ok := args.Get(0).(*EmbeddingResult); ok {
		return result, args.Error(1)
	}

	// Fallback case: create a simple embedding result with basic values
	// This is similar to how Generate falls back to string conversion
	return &EmbeddingResult{
		Vector:     []float32{0.1, 0.2, 0.3}, // Default vector
		TokenCount: len(input),
		Metadata: map[string]interface{}{
			"fallback": true,
		},
	}, args.Error(1)
}

// CreateEmbeddings mocks the batch embedding creation.
func (m *MockBaseLLM) CreateEmbeddings(ctx context.Context, inputs []string, options ...EmbeddingOption) (*BatchEmbeddingResult, error) {
	// Record the method call and get the mock results
	args := m.Called(ctx, inputs, options)

	// Handle nil case
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	// Check if we got a properly structured BatchEmbeddingResult
	if result, ok := args.Get(0).(*BatchEmbeddingResult); ok {
		return result, args.Error(1)
	}

	// Similar to the single embedding case, provide a fallback
	embeddings := make([]EmbeddingResult, len(inputs))
	for i := range inputs {
		embeddings[i] = EmbeddingResult{
			Vector:     []float32{0.1, 0.2, 0.3},
			TokenCount: len(inputs[i]),
			Metadata: map[string]interface{}{
				"fallback": true,
				"index":    i,
			},
		}
	}

	return &BatchEmbeddingResult{
		Embeddings: embeddings,
		Error:      nil,
		ErrorIndex: -1,
	}, args.Error(1)
}

func (m *MockBaseLLM) StreamGenerate(ctx context.Context, prompt string, opts ...GenerateOption) (*StreamResponse, error) {
	args := m.Called(ctx, prompt, opts)

	// Handle the case where we return an error directly
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}

	// Create cancellable context - ONLY after we pass the early return check
	streamCtx, cancelFunc := context.WithCancel(ctx)

	// Extract mock configuration from args
	mockConfig, ok := args.Get(0).(*MockStreamConfig)
	if !ok {
		// Fall back to string conversion for simple cases
		content := args.String(0)
		mockConfig = &MockStreamConfig{
			Content:   content,
			ChunkSize: 10, // Default chunk size
			TokenCounts: &TokenInfo{
				PromptTokens: 3, // Default token count
			},
		}
	}

	// Create the actual stream response
	chunkChan := make(chan StreamChunk)

	// Start goroutine to send chunks according to the config
	go func() {
		defer close(chunkChan)
		defer cancelFunc() // Ensure context is cancelled when goroutine exits

		// Handle immediate error case
		if mockConfig.Error != nil && mockConfig.ErrorAfter <= 0 {
			chunkChan <- StreamChunk{Error: mockConfig.Error}
			return
		}

		// Split content into chunks
		content := mockConfig.Content
		chunkSize := mockConfig.ChunkSize
		if chunkSize <= 0 {
			chunkSize = 10 // Default chunk size if not specified
		}

		var chunksSent int
		for i := 0; i < len(content); i += chunkSize {
			// Check for cancellation
			select {
			case <-streamCtx.Done():
				return
			default:
				// Continue processing
			}

			// Simulate cancel request
			if mockConfig.ShouldCancel {
				return
			}

			end := i + chunkSize
			if end > len(content) {
				end = len(content)
			}

			chunk := content[i:end]
			chunksSent++

			// Check if we should inject an error
			if mockConfig.Error != nil && chunksSent >= mockConfig.ErrorAfter {
				chunkChan <- StreamChunk{Error: mockConfig.Error}
				return
			}

			// Create token info for this chunk
			var tokenInfo *TokenInfo
			if mockConfig.TokenCounts != nil {
				tokenInfo = &TokenInfo{
					PromptTokens:     mockConfig.TokenCounts.PromptTokens,
					CompletionTokens: len(chunk) / 4, // Rough approximation for testing
					TotalTokens:      mockConfig.TokenCounts.PromptTokens + (len(chunk) / 4),
				}
			}

			// Send the chunk
			chunkChan <- StreamChunk{
				Content: chunk,
				Usage:   tokenInfo,
			}

			// Apply delay if specified
			if mockConfig.ChunkDelay > 0 {
				time.Sleep(mockConfig.ChunkDelay)
			}
		}

		// Signal completion
		chunkChan <- StreamChunk{Done: true}

	}()

	return &StreamResponse{
		ChunkChannel: chunkChan,
		Cancel:       cancelFunc,
	}, args.Error(1)
}

func (m *MockBaseLLM) ModelID() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBaseLLM) ProviderName() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBaseLLM) Capabilities() []Capability {
	args := m.Called()
	return args.Get(0).([]Capability)
}

func (m *MockBaseLLM) GenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*LLMResponse, error) {
	args := m.Called(ctx, content, options)
	if resp, ok := args.Get(0).(*LLMResponse); ok {
		return resp, args.Error(1)
	}
	return nil, args.Error(1)
}

func (m *MockBaseLLM) StreamGenerateWithContent(ctx context.Context, content []ContentBlock, options ...GenerateOption) (*StreamResponse, error) {
	args := m.Called(ctx, content, options)
	if resp, ok := args.Get(0).(*StreamResponse); ok {
		return resp, args.Error(1)
	}
	return nil, args.Error(1)
}

func TestBaseDecorator_Unwrap(t *testing.T) {
	// Test that Unwrap correctly returns the wrapped LLM
	baseLLM := new(MockBaseLLM)
	decorator := BaseDecorator{LLM: baseLLM}

	unwrapped := decorator.Unwrap()
	assert.Equal(t, baseLLM, unwrapped, "Unwrap should return the original LLM")
}

func TestModelContextDecorator_Generate(t *testing.T) {
	t.Run("with execution state", func(t *testing.T) {
		// Set up the mock base LLM
		baseLLM := new(MockBaseLLM)
		expectedModelID := "test-model"
		expectedResponse := &LLMResponse{Content: "test response"}

		// Set up mock expectations
		baseLLM.On("ModelID").Return(expectedModelID)
		baseLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).Return(expectedResponse, nil)

		// Create the decorator
		decorator := NewModelContextDecorator(baseLLM)

		// Create context with execution state
		ctx := WithExecutionState(context.Background())

		// Execute the decorated Generate method
		response, err := decorator.Generate(ctx, "test prompt")

		// Verify results
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, response)

		// Verify the model ID was set in the execution state
		state := GetExecutionState(ctx)
		assert.Equal(t, expectedModelID, state.GetModelID())

		// Verify mock expectations were met
		baseLLM.AssertExpectations(t)
	})

	t.Run("without execution state", func(t *testing.T) {
		// Set up the mock base LLM
		baseLLM := new(MockBaseLLM)
		expectedResponse := &LLMResponse{Content: "test response"}

		// Set up mock expectations
		baseLLM.On("Generate", mock.Anything, "test prompt", mock.Anything).Return(expectedResponse, nil)

		// Create the decorator
		decorator := NewModelContextDecorator(baseLLM)

		// Execute with context that has no execution state
		ctx := context.Background()
		response, err := decorator.Generate(ctx, "test prompt")

		// Verify results
		assert.NoError(t, err)
		assert.Equal(t, expectedResponse, response)

		// Verify mock expectations were met
		baseLLM.AssertExpectations(t)
	})
}

func TestChain(t *testing.T) {
	// Create mock LLM
	baseLLM := new(MockBaseLLM)

	// Create test decorators
	decorator1 := func(base LLM) LLM {
		return &ModelContextDecorator{BaseDecorator{LLM: base}}
	}

	decorator2 := func(base LLM) LLM {
		return &ModelContextDecorator{BaseDecorator{LLM: base}}
	}

	t.Run("single decorator", func(t *testing.T) {
		result := Chain(baseLLM, decorator1)

		// Verify the type and structure
		decorated, ok := result.(*ModelContextDecorator)
		assert.True(t, ok)
		assert.Equal(t, baseLLM, decorated.Unwrap())
	})

	t.Run("multiple decorators", func(t *testing.T) {
		result := Chain(baseLLM, decorator1, decorator2)

		// Verify the decorator chain structure
		decorated1, ok := result.(*ModelContextDecorator)
		assert.True(t, ok)

		decorated2, ok := decorated1.Unwrap().(*ModelContextDecorator)
		assert.True(t, ok)

		assert.Equal(t, baseLLM, decorated2.Unwrap())
	})

	t.Run("no decorators", func(t *testing.T) {
		result := Chain(baseLLM)
		assert.Equal(t, baseLLM, result)
	})
}
