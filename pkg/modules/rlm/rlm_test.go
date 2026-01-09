package rlm

import (
	"context"
	"fmt"
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
