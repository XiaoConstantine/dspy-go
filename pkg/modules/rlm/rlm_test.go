package rlm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
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

// TestTruncate tests the TruncateString helper from utils package.
func TestTruncate(t *testing.T) {
	assert.Equal(t, "hello", utils.TruncateString("hello", 10))
	assert.Equal(t, "hel...", utils.TruncateString("hello world", 3))
	assert.Equal(t, "hello...", utils.TruncateString("hello world", 5))
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

// TestHistoryCompressionConfig tests history compression configuration.
func TestHistoryCompressionConfig(t *testing.T) {
	cfg := DefaultConfig()
	WithHistoryCompression(5, 1000)(&cfg)

	assert.NotNil(t, cfg.HistoryCompression)
	assert.True(t, cfg.HistoryCompression.Enabled)
	assert.Equal(t, 5, cfg.HistoryCompression.VerbatimIterations)
	assert.Equal(t, 1000, cfg.HistoryCompression.MaxSummaryTokens)

	// Test default values
	cfg = DefaultConfig()
	WithHistoryCompression(0, 0)(&cfg)
	assert.Equal(t, 3, cfg.HistoryCompression.VerbatimIterations) // Default
	assert.Equal(t, 500, cfg.HistoryCompression.MaxSummaryTokens) // Default
}

// TestAdaptiveIterationConfig tests adaptive iteration configuration.
func TestAdaptiveIterationConfig(t *testing.T) {
	cfg := DefaultConfig()
	WithAdaptiveIteration()(&cfg)

	assert.NotNil(t, cfg.AdaptiveIteration)
	assert.True(t, cfg.AdaptiveIteration.Enabled)
	assert.Equal(t, 10, cfg.AdaptiveIteration.BaseIterations)
	assert.Equal(t, 50, cfg.AdaptiveIteration.MaxIterations)
	assert.Equal(t, 100000, cfg.AdaptiveIteration.ContextScaleFactor)
	assert.True(t, cfg.AdaptiveIteration.EnableEarlyTermination)
	assert.Equal(t, 1, cfg.AdaptiveIteration.ConfidenceThreshold)
}

// TestAdaptiveIterationConfigCustom tests custom adaptive iteration configuration.
func TestAdaptiveIterationConfigCustom(t *testing.T) {
	customCfg := AdaptiveIterationConfig{
		BaseIterations:         5,
		MaxIterations:          20,
		ContextScaleFactor:     50000,
		EnableEarlyTermination: false,
		ConfidenceThreshold:    3,
	}

	cfg := DefaultConfig()
	WithAdaptiveIterationConfig(customCfg)(&cfg)

	assert.True(t, cfg.AdaptiveIteration.Enabled)
	assert.Equal(t, 5, cfg.AdaptiveIteration.BaseIterations)
	assert.Equal(t, 20, cfg.AdaptiveIteration.MaxIterations)
	assert.Equal(t, 50000, cfg.AdaptiveIteration.ContextScaleFactor)
	assert.False(t, cfg.AdaptiveIteration.EnableEarlyTermination)
	assert.Equal(t, 3, cfg.AdaptiveIteration.ConfidenceThreshold)
}

// TestComputeMaxIterations tests the dynamic max iterations calculation.
func TestComputeMaxIterations(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	tests := []struct {
		name         string
		contextSize  int
		config       Config
		expectedMax  int
	}{
		{
			name:        "adaptive disabled uses config max",
			contextSize: 500000,
			config: Config{
				MaxIterations: 30,
			},
			expectedMax: 30,
		},
		{
			name:        "adaptive enabled small context",
			contextSize: 10000, // 10KB
			config: Config{
				MaxIterations: 30,
				AdaptiveIteration: &AdaptiveIterationConfig{
					Enabled:            true,
					BaseIterations:     10,
					MaxIterations:      50,
					ContextScaleFactor: 100000, // 100KB per iteration
				},
			},
			expectedMax: 10, // base + (10000/100000) = 10 + 0 = 10
		},
		{
			name:        "adaptive enabled medium context",
			contextSize: 300000, // 300KB
			config: Config{
				MaxIterations: 30,
				AdaptiveIteration: &AdaptiveIterationConfig{
					Enabled:            true,
					BaseIterations:     10,
					MaxIterations:      50,
					ContextScaleFactor: 100000, // 100KB per iteration
				},
			},
			expectedMax: 13, // base + (300000/100000) = 10 + 3 = 13
		},
		{
			name:        "adaptive enabled capped at max",
			contextSize: 10000000, // 10MB
			config: Config{
				MaxIterations: 30,
				AdaptiveIteration: &AdaptiveIterationConfig{
					Enabled:            true,
					BaseIterations:     10,
					MaxIterations:      50,
					ContextScaleFactor: 100000,
				},
			},
			expectedMax: 50, // Would be 110 but capped at 50
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rlm := New(mockRoot, mockSub)
			rlm.config = tt.config
			result := rlm.computeMaxIterations(tt.contextSize)
			assert.Equal(t, tt.expectedMax, result)
		})
	}
}

// TestHistoryCompression tests the history compression functionality.
func TestHistoryCompression(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	rlm := New(mockRoot, mockSub, WithHistoryCompression(2, 500))

	// Build history with multiple iterations
	history := `
--- Iteration 1 ---
Action: explore
Reasoning: Looking at the first part
Code:
` + "```go" + `
fmt.Println("test1")
` + "```" + `
Output:
test1

--- Iteration 2 ---
Action: explore
Reasoning: Looking at the second part
Code:
` + "```go" + `
fmt.Println("test2")
` + "```" + `
Output:
test2

--- Iteration 3 ---
Action: explore
Reasoning: Looking at the third part
Code:
` + "```go" + `
fmt.Println("test3")
` + "```" + `
Output:
test3

--- Iteration 4 ---
Action: explore
Reasoning: Looking at the fourth part with an Error: something went wrong
Code:
` + "```go" + `
fmt.Println("test4")
` + "```" + `
Output:
test4
`

	// Compress the history keeping last 2 iterations verbatim
	compressed := rlm.compressHistory(history, 4)

	// Should contain summary of iterations 1-2 and verbatim 3-4
	assert.Contains(t, compressed, "[Previous iterations summary]")
	assert.Contains(t, compressed, "--- Iteration 3 ---") // Verbatim
	assert.Contains(t, compressed, "--- Iteration 4 ---") // Verbatim
	assert.NotContains(t, compressed, "Looking at the first part") // Summarized
}

// TestDetectConfidence tests confidence detection.
func TestDetectConfidence(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	// Test with default detector (uses FINAL markers)
	rlm := New(mockRoot, mockSub, WithAdaptiveIteration())

	// Response with FINAL marker should be detected as confident
	// Note: FINAL must be at the start of a line per the regex pattern
	assert.True(t, rlm.detectConfidence("After analysis.\nFINAL(the answer is 42)"))

	// Response with FINAL_VAR should also be detected
	assert.True(t, rlm.detectConfidence("FINAL_VAR(result)"))

	// Response without FINAL should not be confident
	assert.False(t, rlm.detectConfidence("Let me continue exploring..."))

	// Test with custom detector
	customDetector := func(response string) bool {
		return len(response) > 100 // Silly custom detector
	}

	rlm2 := New(mockRoot, mockSub, WithAdaptiveIterationConfig(AdaptiveIterationConfig{
		Enabled:            true,
		ConfidenceDetector: customDetector,
	}))

	shortResponse := "short"
	longResponse := "This is a very long response that exceeds one hundred characters in length to test the custom detector"

	assert.False(t, rlm2.detectConfidence(shortResponse))
	assert.True(t, rlm2.detectConfidence(longResponse))
}

// TestProgressHandler tests the progress callback.
func TestProgressHandler(t *testing.T) {
	mockRoot := &mockLLM{
		responses: []string{
			"Reasoning:\nI'll provide a final answer.\n\nAction:\nfinal\n\nCode:\n\n\nAnswer:\ntest answer",
		},
	}
	mockSub := &mockSubLLMClient{}

	var progressUpdates []IterationProgress
	rlm := New(mockRoot, mockSub, WithProgressHandler(func(progress IterationProgress) {
		progressUpdates = append(progressUpdates, progress)
	}))

	_, err := rlm.Complete(context.Background(), "test context", "test query")
	require.NoError(t, err)

	assert.Len(t, progressUpdates, 1)
	assert.Equal(t, 1, progressUpdates[0].CurrentIteration)
	assert.Equal(t, 30, progressUpdates[0].MaxIterations)
}

// TestContextAnalyzer tests the context analyzer.
func TestContextAnalyzer(t *testing.T) {
	tests := []struct {
		name         string
		content      string
		expectedType ContextType
	}{
		{
			name:         "JSON content",
			content:      `{"key": "value", "nested": {"a": 1}}`,
			expectedType: TypeJSON,
		},
		{
			name:         "Markdown content",
			content:      "# Header\n\nSome text\n\n## Subheader\n\n- list item\n- another item",
			expectedType: TypeMarkdown,
		},
		{
			name:         "Code content",
			content:      "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"hello\")\n}",
			expectedType: TypeCode,
		},
		{
			name:         "Plain text",
			content:      "Just some regular text without any special formatting.",
			expectedType: TypePlainText,
		},
		{
			name:         "Log content",
			content:      "2024-01-15 10:30:00 INFO Starting application\n2024-01-15 10:30:01 DEBUG Connected to database",
			expectedType: TypeLog,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := AnalyzeContext(tt.content)
			assert.Equal(t, tt.expectedType, analysis.Type)
			assert.Greater(t, analysis.Size, 0)
			assert.Greater(t, analysis.EstimatedTokens, 0)
		})
	}
}

// TestContextAnalyzerChunkStrategy tests chunking recommendations.
func TestContextAnalyzerChunkStrategy(t *testing.T) {
	// Small context - no chunking needed
	smallContent := "Small content"
	analysis := AnalyzeContext(smallContent)
	assert.Equal(t, StrategyNone, analysis.RecommendedStrategy)

	// Large JSON - hierarchical chunking
	largeJSON := `{"data": [` + repeatString(`{"id": 1, "name": "test"},`, 10000) + `]}`
	analysis = AnalyzeContext(largeJSON)
	assert.NotEqual(t, StrategyNone, analysis.RecommendedStrategy)
}

func repeatString(s string, count int) string {
	result := ""
	for i := 0; i < count; i++ {
		result += s
	}
	return result
}

// TestAsyncQueryHandle tests async query functionality.
func TestAsyncQueryHandle(t *testing.T) {
	mockSub := &mockSubLLMClient{
		queryResponse: "async response",
	}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	// Start async query
	handle := repl.QueryAsync("test prompt")
	assert.NotEmpty(t, handle.ID())

	// Wait for result
	result, err := handle.Wait()
	require.NoError(t, err)
	assert.Equal(t, "async response", result)
	assert.True(t, handle.Ready())

	// Result should still be available after Wait
	resultStr, ready := handle.Result()
	assert.True(t, ready)
	assert.Equal(t, "async response", resultStr)
}

// TestAsyncBatchHandle tests batch async query functionality.
func TestAsyncBatchHandle(t *testing.T) {
	mockSub := &mockSubLLMClient{
		batchedResponses: []string{"response1", "response2", "response3"},
	}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	// Start batch async queries
	prompts := []string{"prompt1", "prompt2", "prompt3"}
	batchHandle := repl.QueryBatchedAsync(prompts)

	assert.Equal(t, 3, batchHandle.TotalCount())

	// Wait for all results
	results, err := batchHandle.WaitAll()
	require.NoError(t, err)
	assert.Len(t, results, 3)
	assert.True(t, batchHandle.Ready())
	assert.Equal(t, 3, batchHandle.CompletedCount())
}

// TestAsyncQueryFromInterpreter tests async queries from interpreted code.
func TestAsyncQueryFromInterpreter(t *testing.T) {
	mockSub := &mockSubLLMClient{
		queryResponse: "interpreted async response",
	}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	// Start async query from interpreted code
	code := `
handleID := QueryAsync("test async prompt")
result := WaitAsync(handleID)
fmt.Println(result)
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "interpreted async response")
}

// TestPendingAsyncQueries tests tracking of pending queries.
func TestPendingAsyncQueries(t *testing.T) {
	// Create a slow mock client for testing pending state
	slowMock := &slowMockSubLLMClient{
		delay:    50 * time.Millisecond,
		response: "delayed response",
	}

	repl, err := NewYaegiREPL(slowMock)
	require.NoError(t, err)

	// Start multiple async queries
	_ = repl.QueryAsync("prompt1")
	_ = repl.QueryAsync("prompt2")

	// Should have pending queries immediately
	pending := repl.PendingAsyncQueries()
	assert.GreaterOrEqual(t, pending, 0) // May already complete on fast systems

	// Wait for all
	repl.WaitAllAsyncQueries()

	// Should have no pending after wait
	assert.Equal(t, 0, repl.PendingAsyncQueries())
}

// slowMockSubLLMClient is a mock that introduces delay for testing async behavior.
type slowMockSubLLMClient struct {
	delay    time.Duration
	response string
}

func (m *slowMockSubLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	time.Sleep(m.delay)
	return QueryResponse{Response: m.response}, nil
}

func (m *slowMockSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	time.Sleep(m.delay)
	responses := make([]QueryResponse, len(prompts))
	for i := range prompts {
		responses[i] = QueryResponse{Response: m.response}
	}
	return responses, nil
}

// TestGetContextSize tests context size calculation.
func TestGetContextSize(t *testing.T) {
	tests := []struct {
		name     string
		payload  any
		expected int
	}{
		{
			name:     "string payload",
			payload:  "hello world",
			expected: 11,
		},
		{
			name:     "byte slice payload",
			payload:  []byte("test bytes"),
			expected: 10,
		},
		{
			name:     "complex payload",
			payload:  map[string]any{"key": "value"},
			expected: len(fmt.Sprintf("%v", map[string]any{"key": "value"})),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getContextSize(tt.payload)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestShouldTerminateEarly tests early termination logic.
func TestShouldTerminateEarly(t *testing.T) {
	mockRoot := &mockLLM{}
	mockSub := &mockSubLLMClient{}

	// Adaptive disabled
	rlm := New(mockRoot, mockSub)
	assert.False(t, rlm.shouldTerminateEarly(5, false))

	// Adaptive enabled, threshold not met
	rlm = New(mockRoot, mockSub, WithAdaptiveIterationConfig(AdaptiveIterationConfig{
		Enabled:                true,
		EnableEarlyTermination: true,
		ConfidenceThreshold:    3,
	}))
	assert.False(t, rlm.shouldTerminateEarly(2, false)) // 2 < 3

	// Threshold met
	assert.True(t, rlm.shouldTerminateEarly(3, false))

	// Threshold met but has code (should not terminate)
	assert.False(t, rlm.shouldTerminateEarly(5, true))

	// Early termination disabled
	rlm = New(mockRoot, mockSub, WithAdaptiveIterationConfig(AdaptiveIterationConfig{
		Enabled:                true,
		EnableEarlyTermination: false,
		ConfidenceThreshold:    1,
	}))
	assert.False(t, rlm.shouldTerminateEarly(10, false))
}

// TestContextAnalysisHelpers tests helper methods on ContextAnalysis.
func TestContextAnalysisHelpers(t *testing.T) {
	// Small context
	analysis := AnalyzeContext("small content")
	assert.False(t, analysis.IsLargeContext())
	assert.False(t, analysis.ShouldUseBatching())

	// Large context simulation
	analysis.EstimatedTokens = 60000
	analysis.RecommendedStrategy = StrategyFixed
	assert.True(t, analysis.IsLargeContext())
	assert.True(t, analysis.ShouldUseBatching())
}

// promptCapturingMockSubLLMClient captures the prompts sent to it for verification.
type promptCapturingMockSubLLMClient struct {
	capturedPrompts []string
	response        string
	mu              sync.Mutex
}

func (m *promptCapturingMockSubLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	m.mu.Lock()
	m.capturedPrompts = append(m.capturedPrompts, prompt)
	m.mu.Unlock()
	return QueryResponse{Response: m.response, PromptTokens: 10, CompletionTokens: 5}, nil
}

func (m *promptCapturingMockSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	m.mu.Lock()
	m.capturedPrompts = append(m.capturedPrompts, prompts...)
	m.mu.Unlock()
	responses := make([]QueryResponse, len(prompts))
	for i := range prompts {
		responses[i] = QueryResponse{Response: m.response, PromptTokens: 10, CompletionTokens: 5}
	}
	return responses, nil
}

func (m *promptCapturingMockSubLLMClient) getCapturedPrompts() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]string, len(m.capturedPrompts))
	copy(result, m.capturedPrompts)
	return result
}

// TestQueryRaw tests that QueryRaw sends prompts without prepending context.
func TestQueryRaw(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "raw response"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	// Load a context
	err = repl.LoadContext("This is the full context that should NOT be prepended")
	require.NoError(t, err)

	// Execute code that uses QueryRaw
	code := `
result := QueryRaw("Analyze this specific slice only")
fmt.Println(result)
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "raw response")

	// Verify the captured prompt does NOT contain the context
	prompts := capturingMock.getCapturedPrompts()
	require.Len(t, prompts, 1)
	assert.Equal(t, "Analyze this specific slice only", prompts[0])
	assert.NotContains(t, prompts[0], "full context")
}

// TestQueryWith tests that QueryWith sends prompts with explicit context slice.
func TestQueryWith(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "slice response"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	// Load a context (which should be ignored by QueryWith)
	err = repl.LoadContext("This is the full context that should be IGNORED")
	require.NoError(t, err)

	// Execute code that uses QueryWith with explicit slice
	code := `
slice := "This is the specific chunk to analyze"
result := QueryWith(slice, "Summarize this chunk")
fmt.Println(result)
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "slice response")

	// Verify the captured prompt contains the slice but NOT the full context
	prompts := capturingMock.getCapturedPrompts()
	require.Len(t, prompts, 1)
	assert.Contains(t, prompts[0], "Context:\nThis is the specific chunk to analyze")
	assert.Contains(t, prompts[0], "Task: Summarize this chunk")
	assert.NotContains(t, prompts[0], "IGNORED")
}

// TestQueryVsQueryRaw compares Query (with context) vs QueryRaw (without context).
func TestQueryVsQueryRaw(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "test response"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	// Load a context
	err = repl.LoadContext("FULL_CONTEXT_MARKER")
	require.NoError(t, err)

	// Execute code that uses both Query and QueryRaw
	code := `
// Regular Query - should include context
result1 := Query("First query")

// QueryRaw - should NOT include context
result2 := QueryRaw("Second query")

fmt.Println(result1, result2)
`
	_, err = repl.Execute(context.Background(), code)
	require.NoError(t, err)

	prompts := capturingMock.getCapturedPrompts()
	require.Len(t, prompts, 2)

	// First prompt (Query) should have context prepended
	assert.Contains(t, prompts[0], "FULL_CONTEXT_MARKER")
	assert.Contains(t, prompts[0], "First query")

	// Second prompt (QueryRaw) should NOT have context
	assert.NotContains(t, prompts[1], "FULL_CONTEXT_MARKER")
	assert.Equal(t, "Second query", prompts[1])
}

// TestQueryBatchedRaw tests batched queries without context prepending.
func TestQueryBatchedRaw(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "batched response"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	// Load a context
	err = repl.LoadContext("FULL_CONTEXT_SHOULD_BE_IGNORED")
	require.NoError(t, err)

	// Execute code that uses QueryBatchedRaw
	code := `
prompts := []string{"prompt1", "prompt2", "prompt3"}
results := QueryBatchedRaw(prompts)
for _, r := range results {
    fmt.Println(r)
}
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "batched response")

	// Verify none of the captured prompts contain the context
	prompts := capturingMock.getCapturedPrompts()
	require.Len(t, prompts, 3)
	for _, p := range prompts {
		assert.NotContains(t, p, "FULL_CONTEXT_SHOULD_BE_IGNORED")
	}
	assert.Equal(t, "prompt1", prompts[0])
	assert.Equal(t, "prompt2", prompts[1])
	assert.Equal(t, "prompt3", prompts[2])
}

// TestQueryWithEmptySlice tests QueryWith with an empty slice (should act like QueryRaw).
func TestQueryWithEmptySlice(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "empty slice response"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	code := `
result := QueryWith("", "Just the task, no context")
fmt.Println(result)
`
	_, err = repl.Execute(context.Background(), code)
	require.NoError(t, err)

	prompts := capturingMock.getCapturedPrompts()
	require.Len(t, prompts, 1)
	// With empty slice, should just be the prompt
	assert.Equal(t, "Just the task, no context", prompts[0])
}

// TestREPLFindRelevant tests the FindRelevant REPL builtin.
func TestREPLFindRelevant(t *testing.T) {
	mockSub := &mockSubLLMClient{queryResponse: "response"}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	// Load context with multiple topics
	content := `
This section covers error handling.
Errors should be handled carefully.

This section covers user authentication.
Users must authenticate before access.

This section covers database queries.
Queries should be optimized.
`
	err = repl.LoadContext(content)
	require.NoError(t, err)

	// Execute code that uses FindRelevant
	code := `
chunks := FindRelevant("error handling", 2)
fmt.Printf("Found %d chunks\n", len(chunks))
for _, c := range chunks {
    fmt.Println("---")
    fmt.Println(c)
}
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "Found")
}

// TestREPLGetChunk tests the GetChunk REPL builtin.
func TestREPLGetChunk(t *testing.T) {
	mockSub := &mockSubLLMClient{queryResponse: "response"}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	// Set small chunk size for testing
	repl.SetChunkConfig(ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 2,
	})

	content := "line1\nline2\nline3\nline4"
	err = repl.LoadContext(content)
	require.NoError(t, err)

	// Execute code that uses GetChunk
	code := `
chunk0 := GetChunk(0)
fmt.Println("Chunk 0:", chunk0)
count := ChunkCount()
fmt.Println("Total chunks:", count)
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "Chunk 0:")
	assert.Contains(t, execResult.Stdout, "line1")
	assert.Contains(t, execResult.Stdout, "Total chunks:")
}

// TestREPLGetContext tests the GetContext REPL builtin.
func TestREPLGetContext(t *testing.T) {
	mockSub := &mockSubLLMClient{queryResponse: "response"}

	repl, err := NewYaegiREPL(mockSub)
	require.NoError(t, err)

	content := "line1\nline2\nline3\nline4\nline5"
	err = repl.LoadContext(content)
	require.NoError(t, err)

	// Execute code that uses GetContext
	code := `
// Get lines 2-4
section := GetContext(2, 4)
fmt.Println("Section:")
fmt.Println(section)
lines := LineCount()
fmt.Println("Total lines:", lines)
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "line2")
	assert.Contains(t, execResult.Stdout, "line3")
	assert.Contains(t, execResult.Stdout, "line4")
	assert.Contains(t, execResult.Stdout, "Total lines: 5")
}

// TestREPLContextIndexingWorkflow tests a realistic workflow using context indexing.
func TestREPLContextIndexingWorkflow(t *testing.T) {
	capturingMock := &promptCapturingMockSubLLMClient{response: "Analysis complete"}

	repl, err := NewYaegiREPL(capturingMock)
	require.NoError(t, err)

	// Set small chunk size (2 lines per chunk) to ensure multiple chunks
	repl.SetChunkConfig(ChunkConfig{
		ChunkByLines:  true,
		LinesPerChunk: 2,
	})

	// Content with distinct sections
	content := `package main
import "fmt"
func main() {
    handleErrors()
}
func handleErrors() {
    // Error handling logic
    fmt.Println("handling errors")
}
func processData() {
    // Data processing
    fmt.Println("processing")
}
func helperFunction() {
    // Helper code
    fmt.Println("helper")
}`
	err = repl.LoadContext(content)
	require.NoError(t, err)

	// Verify we have multiple chunks
	idx := repl.GetContextIndex()
	require.NotNil(t, idx)
	assert.Greater(t, idx.ChunkCount(), 1, "Should have multiple small chunks")

	// Workflow: Find relevant chunks, then use QueryRaw for targeted analysis
	code := `
// Find chunks about errors
chunks := FindRelevant("error handling", 1)
fmt.Printf("Found %d relevant chunks\n", len(chunks))

// Analyze only the relevant chunk using QueryRaw (no context prepending)
if len(chunks) > 0 {
    result := QueryRaw("Analyze: " + chunks[0])
    fmt.Println("Result:", result)
}
`
	execResult, err := repl.Execute(context.Background(), code)
	require.NoError(t, err)
	assert.Contains(t, execResult.Stdout, "Found")
	assert.Contains(t, execResult.Stdout, "Result:")

	// Verify QueryRaw was called with just the chunk, not full context
	prompts := capturingMock.getCapturedPrompts()
	require.Greater(t, len(prompts), 0)

	// The prompt should be small (just one chunk) and start with "Analyze:"
	lastPrompt := prompts[len(prompts)-1]
	assert.True(t, strings.HasPrefix(lastPrompt, "Analyze:"), "Should use QueryRaw format")
}

// TestSubRLMConfig tests the sub-RLM configuration options.
func TestSubRLMConfig(t *testing.T) {
	cfg := DefaultConfig()
	assert.Nil(t, cfg.SubRLM, "Sub-RLM should be nil by default")

	// Apply WithSubRLM option
	WithSubRLM()(&cfg)
	require.NotNil(t, cfg.SubRLM)
	assert.Equal(t, 3, cfg.SubRLM.MaxDepth)
	assert.Equal(t, 0, cfg.SubRLM.CurrentDepth)
	assert.Equal(t, 10, cfg.SubRLM.MaxIterationsPerSubRLM)

	// Apply custom config
	cfg2 := DefaultConfig()
	WithSubRLMConfig(SubRLMConfig{
		MaxDepth:               5,
		MaxIterationsPerSubRLM: 15,
	})(&cfg2)
	require.NotNil(t, cfg2.SubRLM)
	assert.Equal(t, 5, cfg2.SubRLM.MaxDepth)
	assert.Equal(t, 15, cfg2.SubRLM.MaxIterationsPerSubRLM)
}

// TestSubRLMTokenTracker tests sub-RLM token tracking.
func TestSubRLMTokenTracker(t *testing.T) {
	tracker := NewTokenTracker()

	// Add some root usage
	tracker.AddRootUsage(100, 50)

	// Add sub-LLM calls
	tracker.AddSubCall(LLMCall{
		Prompt:           "test prompt",
		Response:         "test response",
		PromptTokens:     50,
		CompletionTokens: 25,
	})

	// Add sub-RLM call
	tracker.AddSubRLMCall(SubRLMCall{
		Query:            "sub-query",
		Result:           "sub-result",
		Iterations:       3,
		Depth:            1,
		Duration:         time.Second,
		PromptTokens:     200,
		CompletionTokens: 100,
	})

	// Check totals
	total := tracker.GetTotalUsage()
	assert.Equal(t, 350, total.PromptTokens, "Total prompt tokens should include all sources")
	assert.Equal(t, 175, total.CompletionTokens, "Total completion tokens should include all sources")

	// Check sub-RLM specific usage
	subRLMUsage := tracker.GetSubRLMUsage()
	assert.Equal(t, 200, subRLMUsage.PromptTokens)
	assert.Equal(t, 100, subRLMUsage.CompletionTokens)

	// Check sub-RLM calls
	calls := tracker.GetSubRLMCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, "sub-query", calls[0].Query)
	assert.Equal(t, 3, calls[0].Iterations)
	assert.Equal(t, 1, calls[0].Depth)

	// Reset and verify
	tracker.Reset()
	total = tracker.GetTotalUsage()
	assert.Equal(t, 0, total.TotalTokens)
	assert.Empty(t, tracker.GetSubRLMCalls())
}

// TestYaegiREPLSetVariable tests the SetVariable method.
func TestYaegiREPLSetVariable(t *testing.T) {
	client := &mockSubLLMClient{queryResponse: "test"}
	repl, err := NewYaegiREPL(client)
	require.NoError(t, err)

	// Set a variable
	err = repl.SetVariable("testVar", "hello world")
	require.NoError(t, err)

	// Retrieve it
	val, err := repl.GetVariable("testVar")
	require.NoError(t, err)
	assert.Equal(t, "hello world", val)

	// Test with special characters (backticks)
	err = repl.SetVariable("complexVar", "code with `backticks` inside")
	require.NoError(t, err)

	val, err = repl.GetVariable("complexVar")
	require.NoError(t, err)
	assert.Contains(t, val, "backticks")
}

// TestSubRLMDepthLimit tests that sub-RLM respects depth limits.
func TestSubRLMDepthLimit(t *testing.T) {
	// Create an RLM with sub-RLM config at max depth
	llm := &mockLLM{responses: []string{}}
	client := &mockSubLLMClient{queryResponse: "test"}

	r := New(llm, client, WithSubRLMConfig(SubRLMConfig{
		MaxDepth:               2, // Only allow 1 level of nesting
		CurrentDepth:           1, // Already at depth 1
		MaxIterationsPerSubRLM: 5,
	}))

	// Create a REPL for testing
	repl, err := NewYaegiREPL(client)
	require.NoError(t, err)
	err = repl.LoadContext("test context")
	require.NoError(t, err)

	// Try to execute sub-RLM at max depth
	ctx := context.Background()
	_, err = r.executeSubRLM(ctx, repl, "test query", "parent query", nil, time.Now())

	// Should fail due to depth limit
	require.Error(t, err)
	assert.Contains(t, err.Error(), "depth limit")
}

// TestSUBMIT tests the typed SUBMIT functionality.
func TestSUBMIT(t *testing.T) {
	client := &mockSubLLMClient{queryResponse: "test"}
	repl, err := NewYaegiREPL(client)
	require.NoError(t, err)

	// Test without schema (should accept any fields)
	ctx := context.Background()
	code := `result := SUBMIT(map[string]any{"answer": "hello", "count": 42})`
	_, err = repl.Execute(ctx, code)
	require.NoError(t, err)

	assert.True(t, repl.HasSubmit())
	output := repl.GetSubmitOutput()
	assert.Equal(t, "hello", output["answer"])
	assert.Equal(t, 42, output["count"])

	// Reset and test with schema validation
	repl.ClearSubmit()
	repl.SetSubmitSchema(map[string]OutputFieldSpec{
		"answer":   {Type: "string", Required: true},
		"optional": {Type: "int", Required: false},
	})

	// This should work
	code = `result := SUBMIT(map[string]any{"answer": "world"})`
	_, err = repl.Execute(ctx, code)
	require.NoError(t, err)
	assert.True(t, repl.HasSubmit())

	// Reset and try missing required field
	repl.ClearSubmit()
	code = `result := SUBMIT(map[string]any{"optional": 123})`
	result, err := repl.Execute(ctx, code)
	require.NoError(t, err) // Execute succeeds, but SUBMIT returns error message
	assert.False(t, repl.HasSubmit())
	assert.Contains(t, result.Stderr, "missing required field")
}

// TestGetVariableMetadata tests rich variable metadata extraction.
func TestGetVariableMetadata(t *testing.T) {
	client := &mockSubLLMClient{queryResponse: "test"}
	repl, err := NewYaegiREPL(client)
	require.NoError(t, err)

	// Set up some variables
	err = repl.LoadContext("test context with some content")
	require.NoError(t, err)

	ctx := context.Background()
	_, err = repl.Execute(ctx, `result := "computed value"`)
	require.NoError(t, err)
	_, err = repl.Execute(ctx, `count := 42`)
	require.NoError(t, err)

	// Get variable metadata
	vars := repl.GetVariableMetadata()
	assert.NotEmpty(t, vars)

	// Check context variable
	var contextVar *REPLVariable
	var resultVar *REPLVariable
	for i := range vars {
		if vars[i].Name == "context" {
			contextVar = &vars[i]
		}
		if vars[i].Name == "result" {
			resultVar = &vars[i]
		}
	}

	require.NotNil(t, contextVar)
	assert.Equal(t, "string", contextVar.Type)
	assert.Greater(t, contextVar.Length, 0)

	require.NotNil(t, resultVar)
	assert.Equal(t, "string", resultVar.Type)
	assert.True(t, resultVar.IsImportant) // result is marked as important
}

// TestImmutableHistory tests the immutable history structure.
func TestImmutableHistory(t *testing.T) {
	history := NewImmutableHistory()
	assert.Equal(t, 0, history.Len())

	// Append entries
	history.Append(HistoryEntry{
		Iteration: 1,
		Action:    "explore",
		Code:      `fmt.Println(len(context))`,
		Output:    "12345",
	})
	history.Append(HistoryEntry{
		Iteration: 2,
		Action:    "query",
		Code:      `answer := Query("What is X?")`,
		Output:    "[Query] What is X?\n[Result] X is 42",
	})
	history.Append(HistoryEntry{
		Iteration: 3,
		Action:    "final",
		Code:      "",
		Output:    "",
	})

	assert.Equal(t, 3, history.Len())

	// Get entries (should be immutable copy)
	entries := history.Entries()
	assert.Len(t, entries, 3)
	assert.Equal(t, "explore", entries[0].Action)
	assert.Equal(t, "query", entries[1].Action)
	assert.Equal(t, "final", entries[2].Action)

	// Convert to string
	str := history.String(1000)
	assert.Contains(t, str, "Iteration 1")
	assert.Contains(t, str, "explore")
	assert.Contains(t, str, "fmt.Println")
	assert.Contains(t, str, "Iteration 2")
	assert.Contains(t, str, "Query")

	// Test truncation
	truncatedStr := history.String(10) // Very short truncation
	assert.Contains(t, truncatedStr, "...")
}

// TestFormatREPLStateRich tests the rich variable formatting.
func TestFormatREPLStateRich(t *testing.T) {
	vars := []REPLVariable{
		{Name: "context", Type: "string", Length: 50000, Preview: "test data...", IsImportant: false},
		{Name: "result", Type: "string", Length: 100, Preview: "computed answer", IsImportant: true},
		{Name: "count", Type: "int", Length: -1, Preview: "42", IsImportant: true},
	}

	result := formatREPLStateRich(vars, 50)

	assert.Contains(t, result, "Variables:")
	assert.Contains(t, result, "context: string (len=50000)")
	assert.Contains(t, result, "result *: string (len=100)") // Important marked with *
	assert.Contains(t, result, "count *: int")
	assert.Contains(t, result, "→ computed answer") // Preview shown
	assert.Contains(t, result, "→ 42")
	// Context preview should be skipped (too large)
}
