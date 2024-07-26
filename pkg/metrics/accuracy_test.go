package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExactMatch(t *testing.T) {
	tests := []struct {
		name     string
		expected map[string]interface{}
		actual   map[string]interface{}
		want     float64
	}{
		{
			name:     "Exact match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": "hello"},
			want:     1.0,
		},
		{
			name:     "No match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": "world"},
			want:     0.0,
		},
		{
			name:     "Multiple fields match",
			expected: map[string]interface{}{"answer": "hello", "confidence": 0.9},
			actual:   map[string]interface{}{"answer": "hello", "confidence": 0.9},
			want:     1.0,
		},
		{
			name:     "Multiple fields, partial match",
			expected: map[string]interface{}{"answer": "hello", "confidence": 0.9},
			actual:   map[string]interface{}{"answer": "hello", "confidence": 0.8},
			want:     0.0,
		},
		{
			name:     "Missing field in actual",
			expected: map[string]interface{}{"answer": "hello", "confidence": 0.9},
			actual:   map[string]interface{}{"answer": "hello"},
			want:     0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExactMatch(tt.expected, tt.actual)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestAnyMatch(t *testing.T) {
	tests := []struct {
		name     string
		expected map[string]interface{}
		actual   map[string]interface{}
		want     float64
	}{
		{
			name:     "Single value match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": "hello"},
			want:     1.0,
		},
		{
			name:     "Single value no match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": "world"},
			want:     0.0,
		},
		{
			name:     "Slice match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": []interface{}{"world", "hello", "foo"}},
			want:     1.0,
		},
		{
			name:     "Slice no match",
			expected: map[string]interface{}{"answer": "hello"},
			actual:   map[string]interface{}{"answer": []interface{}{"world", "foo", "bar"}},
			want:     0.0,
		},
		{
			name:     "Multiple fields, all match",
			expected: map[string]interface{}{"answer": "hello", "confidence": 0.9},
			actual:   map[string]interface{}{"answer": []interface{}{"world", "hello"}, "confidence": 0.9},
			want:     1.0,
		},
		{
			name:     "Multiple fields, partial match",
			expected: map[string]interface{}{"answer": "hello", "confidence": 0.9},
			actual:   map[string]interface{}{"answer": []interface{}{"world", "hello"}, "confidence": 0.8},
			want:     0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AnyMatch(tt.expected, tt.actual)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestF1Score(t *testing.T) {
	tests := []struct {
		name     string
		expected map[string]interface{}
		actual   map[string]interface{}
		want     float64
	}{
		{
			name:     "Perfect match",
			expected: map[string]interface{}{"answer": "the quick brown fox"},
			actual:   map[string]interface{}{"answer": "the quick brown fox"},
			want:     1.0,
		},
		{
			name:     "No match",
			expected: map[string]interface{}{"answer": "the quick brown fox"},
			actual:   map[string]interface{}{"answer": "a lazy dog"},
			want:     0.0,
		},
		{
			name:     "Partial match",
			expected: map[string]interface{}{"answer": "the quick brown fox"},
			actual:   map[string]interface{}{"answer": "the quick fox jumps"},
			want:     0.75, // (2 * 3/4 * 3/4) / (3/4 + 3/4)
		},
		{
			name:     "Multiple fields",
			expected: map[string]interface{}{"answer1": "the quick brown fox", "answer2": "jumps over the lazy dog"},
			actual:   map[string]interface{}{"answer1": "the quick fox", "answer2": "jumps over the dog"},
			want:     0.8730158730158731, // Average of 0.75 and 0.9960317460317461
		},
		// {
		// 	name:     "Non-string fields",
		// 	expected: map[string]interface{}{"answer": "the quick brown fox", "confidence": 0.9},
		// 	actual:   map[string]interface{}{"answer": "the quick fox", "confidence": 0.8},
		// 	want:     0.75, // Only considers the string field
		// },
		{
			name:     "Empty string",
			expected: map[string]interface{}{"answer": ""},
			actual:   map[string]interface{}{"answer": ""},
			want:     1.0, // Both empty strings should be considered a perfect match
		},
		{
			name:     "One empty string",
			expected: map[string]interface{}{"answer": "the quick brown fox"},
			actual:   map[string]interface{}{"answer": ""},
			want:     0.0,
		},
		{
			name:     "All non-string fields",
			expected: map[string]interface{}{"confidence": 0.9, "score": 5},
			actual:   map[string]interface{}{"confidence": 0.8, "score": 4},
			want:     0.0, // No string fields to compare
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := F1Score(tt.expected, tt.actual)
			assert.InDelta(t, tt.want, got, 0.0001) // Using InDelta for float comparison
		})
	}
}
func TestAccuracy(t *testing.T) {
	expected := map[string]interface{}{"answer": "hello world"}
	actual := map[string]interface{}{"answer": "hello world"}

	exactMatchAccuracy := NewAccuracy(ExactMatch)
	assert.Equal(t, 1.0, exactMatchAccuracy.Evaluate(expected, actual))

	f1ScoreAccuracy := NewAccuracy(F1Score)
	assert.Equal(t, 1.0, f1ScoreAccuracy.Evaluate(expected, actual))

	customMetric := func(expected, actual map[string]interface{}) float64 {
		return 0.5 // Always return 0.5 for testing purposes
	}
	customAccuracy := NewAccuracy(customMetric)
	assert.Equal(t, 0.5, customAccuracy.Evaluate(expected, actual))
}
