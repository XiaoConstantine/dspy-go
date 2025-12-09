package ace

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockPatternSource struct {
	patterns map[string]*PatternInfo
}

func (m *mockPatternSource) GetPatterns() map[string]*PatternInfo {
	return m.patterns
}

type mockMetricsSource struct {
	errorPatterns map[string]int
}

func (m *mockMetricsSource) GetErrorPatterns() map[string]int {
	return m.errorPatterns
}

func TestSelfReflectorAdapter(t *testing.T) {
	t.Run("extracts high-success patterns", func(t *testing.T) {
		patterns := &mockPatternSource{
			patterns: map[string]*PatternInfo{
				"[search, read]": {Name: "[search, read]", Occurrences: 5, SuccessRate: 0.9},
				"[write]":        {Name: "[write]", Occurrences: 2, SuccessRate: 0.5}, // Too few occurrences
			},
		}

		adapter := NewSelfReflectorAdapter(patterns, nil)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Len(t, insights, 1)
		assert.Equal(t, "strategies", insights[0].Category)
		assert.Contains(t, insights[0].Content, "[search, read]")
	})

	t.Run("extracts error patterns", func(t *testing.T) {
		metrics := &mockMetricsSource{
			errorPatterns: map[string]int{
				"timeout error":    3,
				"single occurrence": 1, // Not enough occurrences
			},
		}

		adapter := NewSelfReflectorAdapter(nil, metrics)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Len(t, insights, 1)
		assert.Equal(t, "mistakes", insights[0].Category)
		assert.Contains(t, insights[0].Content, "timeout error")
	})

	t.Run("nil sources", func(t *testing.T) {
		adapter := NewSelfReflectorAdapter(nil, nil)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Empty(t, insights)
	})
}

type mockErrorSource struct {
	errors    []ErrorInfo
	successes []SuccessInfo
}

func (m *mockErrorSource) GetErrors() []ErrorInfo {
	return m.errors
}

func (m *mockErrorSource) GetSuccesses() []SuccessInfo {
	return m.successes
}

func TestErrorRetainerAdapter(t *testing.T) {
	t.Run("extracts errors as mistakes", func(t *testing.T) {
		source := &mockErrorSource{
			errors: []ErrorInfo{
				{ErrorType: "timeout", Message: "Connection timed out", Count: 3},
			},
		}

		adapter := NewErrorRetainerAdapter(source)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Len(t, insights, 1)
		assert.Equal(t, "mistakes", insights[0].Category)
		assert.Contains(t, insights[0].Content, "Connection timed out")
	})

	t.Run("extracts successes as strategies", func(t *testing.T) {
		source := &mockErrorSource{
			successes: []SuccessInfo{
				{SuccessType: "retry", Description: "Retry with backoff works", Confidence: 0.85},
			},
		}

		adapter := NewErrorRetainerAdapter(source)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Len(t, insights, 1)
		assert.Equal(t, "strategies", insights[0].Category)
		assert.Equal(t, 0.85, insights[0].Confidence)
	})

	t.Run("includes pattern in content", func(t *testing.T) {
		source := &mockErrorSource{
			errors: []ErrorInfo{
				{ErrorType: "api", Message: "Rate limited", Pattern: "429_response", Count: 1},
			},
		}

		adapter := NewErrorRetainerAdapter(source)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Contains(t, insights[0].Content, "429_response")
	})

	t.Run("nil source", func(t *testing.T) {
		adapter := NewErrorRetainerAdapter(nil)
		insights, err := adapter.Extract(context.Background())

		require.NoError(t, err)
		assert.Empty(t, insights)
	})
}

func TestStaticAdapter(t *testing.T) {
	insights := []InsightCandidate{
		{Content: "Test 1", Category: "strategies"},
		{Content: "Test 2", Category: "mistakes"},
	}

	adapter := NewStaticAdapter(insights)
	result, err := adapter.Extract(context.Background())

	require.NoError(t, err)
	assert.Equal(t, insights, result)
}
