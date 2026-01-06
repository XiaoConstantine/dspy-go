package rlm

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockLLM implements core.LLM for testing.
type mockLLM struct {
	responses []string
	callCount int
}

func (m *mockLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	if m.callCount >= len(m.responses) {
		return nil, fmt.Errorf("no more mock responses")
	}
	resp := m.responses[m.callCount]
	m.callCount++
	return &core.LLMResponse{
		Content: resp,
		Usage: &core.TokenInfo{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}, nil
}

func (m *mockLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("JSON generation not implemented in mock")
}

func (m *mockLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("function calling not implemented in mock")
}

func (m *mockLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	return m.Generate(ctx, "", opts...)
}

func (m *mockLLM) StreamGenerate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in mock")
}

func (m *mockLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in mock")
}

func (m *mockLLM) CreateEmbedding(ctx context.Context, input string, opts ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("embeddings not implemented in mock")
}

func (m *mockLLM) CreateEmbeddings(ctx context.Context, inputs []string, opts ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("batch embeddings not implemented in mock")
}

func (m *mockLLM) ProviderName() string { return "mock" }
func (m *mockLLM) ModelID() string      { return "mock-model" }
func (m *mockLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

// mockSubLLMClient implements SubLLMClient for testing.
type mockSubLLMClient struct {
	queryResponse        string
	batchedResponses     []string
	queryPromptTokens    int
	queryCompletionTokens int
}

func (m *mockSubLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	return QueryResponse{
		Response:         m.queryResponse,
		PromptTokens:     m.queryPromptTokens,
		CompletionTokens: m.queryCompletionTokens,
	}, nil
}

func (m *mockSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	responses := make([]QueryResponse, len(prompts))
	for i := range prompts {
		resp := m.queryResponse
		if i < len(m.batchedResponses) {
			resp = m.batchedResponses[i]
		}
		responses[i] = QueryResponse{
			Response:         resp,
			PromptTokens:     m.queryPromptTokens,
			CompletionTokens: m.queryCompletionTokens,
		}
	}
	return responses, nil
}

// TestFindCodeBlocks tests the code block extraction.
func TestFindCodeBlocks(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "single go block",
			input:    "Some text\n```go\nfmt.Println(\"hello\")\n```\nMore text",
			expected: []string{`fmt.Println("hello")`},
		},
		{
			name:     "single repl block",
			input:    "Some text\n```repl\nx := 42\n```\nMore text",
			expected: []string{"x := 42"},
		},
		{
			name:     "multiple blocks",
			input:    "```go\ncode1\n```\n\n```go\ncode2\n```",
			expected: []string{"code1", "code2"},
		},
		{
			name:     "no blocks",
			input:    "Just plain text without code blocks",
			expected: []string{},
		},
		{
			name:     "empty block",
			input:    "```go\n\n```",
			expected: []string{},
		},
		{
			name:     "multiline code",
			input:    "```go\nfmt.Println(\"line1\")\nfmt.Println(\"line2\")\n```",
			expected: []string{"fmt.Println(\"line1\")\nfmt.Println(\"line2\")"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindCodeBlocks(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestFindFinalAnswer tests the final answer detection.
func TestFindFinalAnswer(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected *FinalAnswer
	}{
		{
			name:  "FINAL_VAR",
			input: "Some text\nFINAL_VAR(answer)\nMore text",
			expected: &FinalAnswer{
				Type:    FinalTypeVariable,
				Content: "answer",
			},
		},
		{
			name:  "FINAL direct",
			input: "FINAL(42)",
			expected: &FinalAnswer{
				Type:    FinalTypeDirect,
				Content: "42",
			},
		},
		{
			name:  "FINAL with quotes",
			input: `FINAL("hello world")`,
			expected: &FinalAnswer{
				Type:    FinalTypeDirect,
				Content: "hello world",
			},
		},
		{
			name:     "no final",
			input:    "Just regular text",
			expected: nil,
		},
		{
			name:  "FINAL_VAR with spaces",
			input: "  FINAL_VAR( result )",
			expected: &FinalAnswer{
				Type:    FinalTypeVariable,
				Content: "result",
			},
		},
		{
			name:  "FINAL_VAR with underscore",
			input: "FINAL_VAR(final_answer)",
			expected: &FinalAnswer{
				Type:    FinalTypeVariable,
				Content: "final_answer",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindFinalAnswer(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestTokenTracker tests the token tracking functionality.
func TestTokenTracker(t *testing.T) {
	tracker := NewTokenTracker()

	// Add root usage
	tracker.AddRootUsage(100, 50)
	tracker.AddRootUsage(200, 100)

	// Add sub calls
	tracker.AddSubCall(LLMCall{
		Prompt:           "test prompt 1",
		Response:         "test response 1",
		PromptTokens:     50,
		CompletionTokens: 25,
	})
	tracker.AddSubCall(LLMCall{
		Prompt:           "test prompt 2",
		Response:         "test response 2",
		PromptTokens:     75,
		CompletionTokens: 35,
	})

	// Test root usage
	rootUsage := tracker.GetRootUsage()
	assert.Equal(t, 300, rootUsage.PromptTokens)
	assert.Equal(t, 150, rootUsage.CompletionTokens)
	assert.Equal(t, 450, rootUsage.TotalTokens)

	// Test sub usage
	subUsage := tracker.GetSubUsage()
	assert.Equal(t, 125, subUsage.PromptTokens)
	assert.Equal(t, 60, subUsage.CompletionTokens)
	assert.Equal(t, 185, subUsage.TotalTokens)

	// Test total usage
	totalUsage := tracker.GetTotalUsage()
	assert.Equal(t, 425, totalUsage.PromptTokens)
	assert.Equal(t, 210, totalUsage.CompletionTokens)
	assert.Equal(t, 635, totalUsage.TotalTokens)

	// Test sub calls
	calls := tracker.GetSubCalls()
	assert.Len(t, calls, 2)
	assert.Equal(t, "test prompt 1", calls[0].Prompt)

	// Test reset
	tracker.Reset()
	totalUsage = tracker.GetTotalUsage()
	assert.Equal(t, 0, totalUsage.TotalTokens)
}

// TestConfig tests the configuration options.
func TestConfig(t *testing.T) {
	// Test default config
	cfg := DefaultConfig()
	assert.Equal(t, 30, cfg.MaxIterations)
	assert.False(t, cfg.Verbose)

	// Test options
	cfg = DefaultConfig()
	WithMaxIterations(10)(&cfg)
	assert.Equal(t, 10, cfg.MaxIterations)

	cfg = DefaultConfig()
	WithVerbose(true)(&cfg)
	assert.True(t, cfg.Verbose)

	cfg = DefaultConfig()
	WithTimeout(5 * time.Second)(&cfg)
	assert.Equal(t, 5*time.Second, cfg.Timeout)
}

// TestRLMNew tests the RLM constructor.
func TestRLMNew(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	rlm := New(mockRoot, mockSub)
	assert.NotNil(t, rlm)
	assert.Equal(t, "RLM", rlm.GetModuleType())
	assert.Equal(t, 30, rlm.config.MaxIterations)

	// Test with options
	rlm = New(mockRoot, mockSub, WithMaxIterations(5), WithVerbose(true))
	assert.Equal(t, 5, rlm.config.MaxIterations)
	assert.True(t, rlm.config.Verbose)
}

// TestRLMProcess tests the main Process method.
func TestRLMProcess(t *testing.T) {
	// Create mock LLM that returns structured outputs for the iteration module
	// The Predict module will parse these fields from the response
	mockRoot := &mockLLM{
		responses: []string{
			"Reasoning:\nI'll explore the context and provide a final answer.\n\nAction:\nfinal\n\nCode:\n\n\nAnswer:\ntest answer",
		},
	}
	mockSub := &mockSubLLMClient{
		queryResponse: "sub llm response",
	}

	rlm := New(mockRoot, mockSub)

	inputs := map[string]any{
		"context": "test context data",
		"query":   "what is the answer?",
	}

	result, err := rlm.Process(context.Background(), inputs)
	require.NoError(t, err)
	assert.Equal(t, "test answer", result["answer"])
}

// TestRLMProcessMissingInputs tests error handling for missing inputs.
func TestRLMProcessMissingInputs(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}
	rlm := New(mockRoot, mockSub)

	// Missing context
	_, err := rlm.Process(context.Background(), map[string]any{
		"query": "test query",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context")

	// Missing query
	_, err = rlm.Process(context.Background(), map[string]any{
		"context": "test context",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "query")
}

// TestRLMWithCodeExecution tests the iteration loop with code execution.
func TestRLMWithCodeExecution(t *testing.T) {
	// First response contains code to explore, second contains final answer
	mockRoot := &mockLLM{
		responses: []string{
			"Reasoning:\nLet me check the context length.\n\nAction:\nexplore\n\nCode:\nfmt.Println(len(context))\n\nAnswer:\n",
			"Reasoning:\nBased on the output, the context is 'hello world'.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\nhello world",
		},
	}
	mockSub := &mockSubLLMClient{}

	rlm := New(mockRoot, mockSub, WithMaxIterations(5))

	result, err := rlm.Complete(context.Background(), "hello world", "what is in the context?")
	require.NoError(t, err)
	assert.Equal(t, "hello world", result.Response)
	assert.Equal(t, 2, result.Iterations)
}

// TestRLMMaxIterations tests that max iterations is respected.
func TestRLMMaxIterations(t *testing.T) {
	// Never return a final answer until forced
	mockRoot := &mockLLM{
		responses: []string{
			"Reasoning:\nExploring iteration 1.\n\nAction:\nexplore\n\nCode:\nfmt.Println(\"iteration 1\")\n\nAnswer:\n",
			"Reasoning:\nExploring iteration 2.\n\nAction:\nexplore\n\nCode:\nfmt.Println(\"iteration 2\")\n\nAnswer:\n",
			"Reasoning:\nExploring iteration 3.\n\nAction:\nexplore\n\nCode:\nfmt.Println(\"iteration 3\")\n\nAnswer:\n",
			"Reasoning:\nForced to provide answer.\n\nAction:\nfinal\n\nCode:\n\nAnswer:\nforced answer", // Final response when forced
		},
	}
	mockSub := &mockSubLLMClient{}

	rlm := New(mockRoot, mockSub, WithMaxIterations(3))

	result, err := rlm.Complete(context.Background(), "test", "test query")
	require.NoError(t, err)
	assert.Equal(t, 3, result.Iterations) // Should stop at max
	assert.Equal(t, "forced answer", result.Response)
}

// TestRLMContextCancellation tests that context cancellation works.
func TestRLMContextCancellation(t *testing.T) {
	mockRoot := &mockLLM{
		responses: []string{
			"Reasoning:\nTest reasoning.\n\nAction:\nexplore\n\nCode:\nfmt.Println(\"test\")\n\nAnswer:\n",
		},
	}
	mockSub := &mockSubLLMClient{}

	rlm := New(mockRoot, mockSub)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := rlm.Complete(ctx, "test", "test query")
	assert.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

// TestRLMClone tests the Clone method.
func TestRLMClone(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	original := New(mockRoot, mockSub, WithMaxIterations(10), WithVerbose(true))

	cloned := original.Clone().(*RLM)
	assert.Equal(t, original.config.MaxIterations, cloned.config.MaxIterations)
	assert.Equal(t, original.config.Verbose, cloned.config.Verbose)
	assert.NotSame(t, original.tokenTracker, cloned.tokenTracker)
}

// TestFormatExecutionResult tests the execution result formatting.
func TestFormatExecutionResult(t *testing.T) {
	tests := []struct {
		name     string
		result   *ExecutionResult
		expected string
	}{
		{
			name: "stdout only",
			result: &ExecutionResult{
				Stdout: "hello world",
			},
			expected: "hello world",
		},
		{
			name: "stderr only",
			result: &ExecutionResult{
				Stderr: "error message",
			},
			expected: "error message",
		},
		{
			name: "both stdout and stderr",
			result: &ExecutionResult{
				Stdout: "output",
				Stderr: "error",
			},
			expected: "output\n\nerror",
		},
		{
			name: "empty result",
			result: &ExecutionResult{},
			expected: "No output",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatExecutionResult(tt.result)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestContextMetadata tests the context metadata helper.
func TestContextMetadata(t *testing.T) {
	tests := []struct {
		name     string
		payload  any
		expected string
	}{
		{
			name:     "string",
			payload:  "hello world",
			expected: "string, 11 chars",
		},
		{
			name:     "array",
			payload:  []any{1, 2, 3},
			expected: "array, 3 items",
		},
		{
			name:     "map",
			payload:  map[string]any{"key": "value"},
			expected: "object, 1 keys",
		},
		{
			name:     "other type",
			payload:  42,
			expected: "int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContextMetadata(tt.payload)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestTruncate tests the truncate helper functions.
func TestTruncate(t *testing.T) {
	assert.Equal(t, "hello", truncate("hello", 10))
	assert.Equal(t, "hel...", truncate("hello world", 3))
	assert.Equal(t, "hello...", truncate("hello world", 5))
}

// TestStripQuotes tests the quote stripping helper.
func TestStripQuotes(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"hello"`, "hello"},
		{`'hello'`, "hello"},
		{"`hello`", "hello"},
		{"hello", "hello"},
		{`"`, `"`},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := stripQuotes(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// BenchmarkFindCodeBlocks benchmarks code block extraction.
func BenchmarkFindCodeBlocks(b *testing.B) {
	input := `Some text before

` + "```go" + `
func example() {
    fmt.Println("hello")
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
}
` + "```" + `

More text

` + "```go" + `
x := 42
y := 84
fmt.Println(x + y)
` + "```" + `

Final text`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FindCodeBlocks(input)
	}
}

// BenchmarkFindFinalAnswer benchmarks final answer detection.
func BenchmarkFindFinalAnswer(b *testing.B) {
	input := `Based on my analysis, I found the answer stored in the result variable.

FINAL_VAR(result)

This concludes the analysis.`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FindFinalAnswer(input)
	}
}
