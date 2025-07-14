package cache

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCacheStats_HitRate(t *testing.T) {
	tests := []struct {
		name     string
		hits     int64
		misses   int64
		expected float64
	}{
		{
			name:     "No hits or misses",
			hits:     0,
			misses:   0,
			expected: 0.0,
		},
		{
			name:     "Only hits",
			hits:     10,
			misses:   0,
			expected: 1.0,
		},
		{
			name:     "Only misses",
			hits:     0,
			misses:   10,
			expected: 0.0,
		},
		{
			name:     "50% hit rate",
			hits:     5,
			misses:   5,
			expected: 0.5,
		},
		{
			name:     "75% hit rate",
			hits:     75,
			misses:   25,
			expected: 0.75,
		},
		{
			name:     "25% hit rate",
			hits:     25,
			misses:   75,
			expected: 0.25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats := CacheStats{
				Hits:   tt.hits,
				Misses: tt.misses,
			}
			assert.Equal(t, tt.expected, stats.HitRate())
		})
	}
}

func TestCacheStats_MissRate(t *testing.T) {
	tests := []struct {
		name     string
		hits     int64
		misses   int64
		expected float64
	}{
		{
			name:     "No hits or misses",
			hits:     0,
			misses:   0,
			expected: 0.0,
		},
		{
			name:     "Only hits",
			hits:     10,
			misses:   0,
			expected: 0.0,
		},
		{
			name:     "Only misses",
			hits:     0,
			misses:   10,
			expected: 1.0,
		},
		{
			name:     "50% miss rate",
			hits:     5,
			misses:   5,
			expected: 0.5,
		},
		{
			name:     "25% miss rate",
			hits:     75,
			misses:   25,
			expected: 0.25,
		},
		{
			name:     "75% miss rate",
			hits:     25,
			misses:   75,
			expected: 0.75,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats := CacheStats{
				Hits:   tt.hits,
				Misses: tt.misses,
			}
			assert.Equal(t, tt.expected, stats.MissRate())
		})
	}
}

func TestCacheStats_HitRateAndMissRateSum(t *testing.T) {
	// Test that HitRate + MissRate = 1.0 (except when both are 0)
	testCases := []struct {
		hits   int64
		misses int64
	}{
		{10, 0},
		{0, 10},
		{50, 50},
		{75, 25},
		{1, 99},
		{999, 1},
	}

	for _, tc := range testCases {
		stats := CacheStats{
			Hits:   tc.hits,
			Misses: tc.misses,
		}
		
		hitRate := stats.HitRate()
		missRate := stats.MissRate()
		
		// Should sum to 1.0 when there are hits or misses
		assert.InDelta(t, 1.0, hitRate+missRate, 0.001, 
			"HitRate (%f) + MissRate (%f) should equal 1.0", hitRate, missRate)
	}
}

func TestCacheStats_ZeroDivisionSafety(t *testing.T) {
	// Test that division by zero is handled safely
	stats := CacheStats{
		Hits:       0,
		Misses:     0,
		Sets:       0,
		Deletes:    0,
		Size:       0,
		MaxSize:    1000,
		LastAccess: time.Now(),
	}
	
	// These should not panic and should return 0.0
	assert.Equal(t, 0.0, stats.HitRate())
	assert.Equal(t, 0.0, stats.MissRate())
}