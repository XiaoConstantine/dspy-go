package tools

import (
	"context"
	"fmt"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBayesianToolSelector_ScoreTools(t *testing.T) {
	selector := NewBayesianToolSelector()

	// Create test tools
	tools := []core.Tool{
		newMockTool("search", "Search for information in databases", []string{"search", "query", "find"}),
		newMockTool("create", "Create new resources and files", []string{"create", "generate", "make"}),
		newMockTool("analyze", "Analyze and process data", []string{"analyze", "process", "examine"}),
	}

	tests := []struct {
		name           string
		intent         string
		expectedBest   string
		minScores      int
	}{
		{
			name:         "search intent should favor search tool",
			intent:       "find user information",
			expectedBest: "search",
			minScores:    3,
		},
		{
			name:         "create intent should favor create tool",
			intent:       "generate new report",
			expectedBest: "create",
			minScores:    3,
		},
		{
			name:         "analyze intent should favor analyze tool",
			intent:       "process the data",
			expectedBest: "analyze",
			minScores:    3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			scores, err := selector.ScoreTools(ctx, tt.intent, tools)

			require.NoError(t, err)
			assert.Len(t, scores, tt.minScores)

			// Find the best scoring tool
			var bestScore ToolScore
			for _, score := range scores {
				if score.FinalScore > bestScore.FinalScore {
					bestScore = score
				}
			}

			assert.Equal(t, tt.expectedBest, bestScore.Tool.Name())
			assert.True(t, bestScore.FinalScore > 0)
			assert.True(t, bestScore.MatchScore >= 0)
			assert.True(t, bestScore.PerformanceScore >= 0)
			assert.True(t, bestScore.CapabilityScore >= 0)
		})
	}
}

func TestBayesianToolSelector_SelectBest(t *testing.T) {
	selector := NewBayesianToolSelector()

	// Create test tool scores
	scores := []ToolScore{
		{
			Tool:             newMockTool("low", "Low scoring tool", []string{}),
			MatchScore:       0.2,
			PerformanceScore: 0.3,
			CapabilityScore:  0.1,
			FinalScore:       0.2,
		},
		{
			Tool:             newMockTool("high", "High scoring tool", []string{}),
			MatchScore:       0.8,
			PerformanceScore: 0.9,
			CapabilityScore:  0.7,
			FinalScore:       0.8,
		},
		{
			Tool:             newMockTool("medium", "Medium scoring tool", []string{}),
			MatchScore:       0.5,
			PerformanceScore: 0.6,
			CapabilityScore:  0.4,
			FinalScore:       0.5,
		},
	}

	ctx := context.Background()
	bestTool, err := selector.SelectBest(ctx, "test intent", scores)

	require.NoError(t, err)
	assert.Equal(t, "high", bestTool.Name())
}

func TestBayesianToolSelector_TokenizeIntent(t *testing.T) {
	selector := NewBayesianToolSelector()

	tests := []struct {
		name     string
		intent   string
		expected []string
	}{
		{
			name:     "simple intent",
			intent:   "search for data",
			expected: []string{"search", "data"},
		},
		{
			name:     "intent with stop words",
			intent:   "I need to find the user information",
			expected: []string{"need", "find", "user", "information"},
		},
		{
			name:     "intent with underscores and dashes",
			intent:   "create_new-resource",
			expected: []string{"create", "new", "resource"},
		},
		{
			name:     "empty intent",
			intent:   "",
			expected: []string{},
		},
		{
			name:     "intent with only stop words",
			intent:   "the and or but",
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens := selector.tokenizeIntent(tt.intent)
			assert.Equal(t, tt.expected, tokens)
		})
	}
}

func TestBayesianToolSelector_TokensMatch(t *testing.T) {
	selector := NewBayesianToolSelector()

	tests := []struct {
		name     string
		token1   string
		token2   string
		expected bool
	}{
		{
			name:     "exact match",
			token1:   "search",
			token2:   "search",
			expected: true,
		},
		{
			name:     "substring match",
			token1:   "searching",
			token2:   "search",
			expected: true,
		},
		{
			name:     "synonym match - search/find",
			token1:   "search",
			token2:   "find",
			expected: true,
		},
		{
			name:     "synonym match - create/generate",
			token1:   "create",
			token2:   "generate",
			expected: true,
		},
		{
			name:     "no match",
			token1:   "search",
			token2:   "delete",
			expected: false,
		},
		{
			name:     "case insensitive",
			token1:   "SEARCH",
			token2:   "search",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			match := selector.tokensMatch(tt.token1, tt.token2)
			assert.Equal(t, tt.expected, match)
		})
	}
}

func TestBayesianToolSelector_CalculateMatchScore(t *testing.T) {
	selector := NewBayesianToolSelector()

	tool := newMockTool("search_engine", "A tool to search for information", []string{"search"})

	tests := []struct {
		name         string
		intentTokens []string
		minScore     float64
		maxScore     float64
	}{
		{
			name:         "perfect name match",
			intentTokens: []string{"search", "engine"},
			minScore:     0.5,
			maxScore:     1.0,
		},
		{
			name:         "description match",
			intentTokens: []string{"information", "tool"},
			minScore:     0.1,
			maxScore:     0.8,
		},
		{
			name:         "no match",
			intentTokens: []string{"unrelated", "words"},
			minScore:     0.0,
			maxScore:     0.1,
		},
		{
			name:         "empty tokens",
			intentTokens: []string{},
			minScore:     0.0,
			maxScore:     0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := selector.calculateMatchScore(tt.intentTokens, tool)
			assert.GreaterOrEqual(t, score, tt.minScore)
			assert.LessOrEqual(t, score, tt.maxScore)
		})
	}
}

func TestBayesianToolSelector_CalculatePerformanceScore(t *testing.T) {
	selector := NewBayesianToolSelector()

	tests := []struct {
		name     string
		tool     core.Tool
		minScore float64
		maxScore float64
	}{
		{
			name:     "tool with version and capabilities",
			tool:     newMockTool("versioned", "Tool with version", []string{"cap1", "cap2", "cap3"}),
			minScore: 0.7,
			maxScore: 1.0,
		},
		{
			name:     "basic tool",
			tool:     newMockTool("basic", "Basic tool", []string{}),
			minScore: 0.5,
			maxScore: 0.7,
		},
		{
			name:     "tool with nil metadata",
			tool:     &mockToolWithNilMetadata{name: "nil_meta"},
			minScore: 0.5,
			maxScore: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := selector.calculatePerformanceScore(tt.tool)
			assert.GreaterOrEqual(t, score, tt.minScore)
			assert.LessOrEqual(t, score, tt.maxScore)
		})
	}
}

func TestBayesianToolSelector_CalculateCapabilityScore(t *testing.T) {
	selector := NewBayesianToolSelector()

	tests := []struct {
		name         string
		intentTokens []string
		tool         core.Tool
		minScore     float64
	}{
		{
			name:         "matching capabilities",
			intentTokens: []string{"search", "query"},
			tool:         newMockTool("search_tool", "Search tool", []string{"search", "query", "find"}),
			minScore:     0.5,
		},
		{
			name:         "partial capability match",
			intentTokens: []string{"search", "create"},
			tool:         newMockTool("search_tool", "Search tool", []string{"search", "query"}),
			minScore:     0.3,
		},
		{
			name:         "no capability match",
			intentTokens: []string{"delete", "remove"},
			tool:         newMockTool("search_tool", "Search tool", []string{"search", "query"}),
			minScore:     0.0,
		},
		{
			name:         "tool without capabilities",
			intentTokens: []string{"search"},
			tool:         newMockTool("no_caps", "Tool without capabilities", []string{}),
			minScore:     0.3, // Should get default score
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := selector.calculateCapabilityScore(tt.intentTokens, tt.tool)
			assert.GreaterOrEqual(t, score, tt.minScore)
			assert.LessOrEqual(t, score, 1.0)
		})
	}
}

func TestBayesianToolSelector_UpdatePriorProbabilities(t *testing.T) {
	selector := NewBayesianToolSelector()

	usageStats := map[string]int{
		"search": 50,
		"create": 30,
		"delete": 20,
	}

	selector.UpdatePriorProbabilities(usageStats)

	expectedSearch := 50.0 / 100.0
	expectedCreate := 30.0 / 100.0
	expectedDelete := 20.0 / 100.0

	assert.Equal(t, expectedSearch, selector.PriorProbabilities["search"])
	assert.Equal(t, expectedCreate, selector.PriorProbabilities["create"])
	assert.Equal(t, expectedDelete, selector.PriorProbabilities["delete"])

	// Test with empty stats
	selector.UpdatePriorProbabilities(map[string]int{})
	// Should not modify existing probabilities
	assert.Equal(t, expectedSearch, selector.PriorProbabilities["search"])
}

func TestBayesianToolSelector_CalculateFinalScore(t *testing.T) {
	selector := NewBayesianToolSelector()

	// Set up prior probabilities
	selector.PriorProbabilities["test_tool"] = 0.8

	score := ToolScore{
		Tool:             newMockTool("test_tool", "Test tool", []string{}),
		MatchScore:       0.6,
		PerformanceScore: 0.7,
		CapabilityScore:  0.5,
	}

	finalScore := selector.calculateFinalScore(score)

	// Should be a weighted combination of the scores, potentially modified by prior
	assert.True(t, finalScore > 0)
	assert.True(t, finalScore <= 1.0)

	// Test without prior probability
	scoreWithoutPrior := ToolScore{
		Tool:             newMockTool("unknown_tool", "Unknown tool", []string{}),
		MatchScore:       0.6,
		PerformanceScore: 0.7,
		CapabilityScore:  0.5,
	}

	finalScoreWithoutPrior := selector.calculateFinalScore(scoreWithoutPrior)
	assert.True(t, finalScoreWithoutPrior > 0)
	assert.True(t, finalScoreWithoutPrior <= 1.0)
}

func TestBayesianToolSelector_EmptyInputs(t *testing.T) {
	selector := NewBayesianToolSelector()
	ctx := context.Background()

	// Test with empty tools list
	scores, err := selector.ScoreTools(ctx, "test intent", []core.Tool{})
	assert.Error(t, err)
	assert.Nil(t, scores)

	// Test with empty candidates list
	tool, err := selector.SelectBest(ctx, "test intent", []ToolScore{})
	assert.Error(t, err)
	assert.Nil(t, tool)
}

func BenchmarkBayesianToolSelector_ScoreTools(b *testing.B) {
	selector := NewBayesianToolSelector()

	// Create a larger set of tools
	tools := make([]core.Tool, 100)
	for i := 0; i < 100; i++ {
		tools[i] = newMockTool(
			fmt.Sprintf("tool_%d", i),
			fmt.Sprintf("Tool number %d for testing", i),
			[]string{fmt.Sprintf("capability_%d", i%10)},
		)
	}

	ctx := context.Background()
	intent := "search for information using capability_5"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := selector.ScoreTools(ctx, intent, tools)
		if err != nil {
			b.Fatalf("ScoreTools failed: %v", err)
		}
	}
}
