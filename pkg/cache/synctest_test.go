// Package cache provides synctest-based tests for concurrent cache operations.
// These tests use Go 1.25's testing/synctest package for deterministic concurrent testing.
package cache

import (
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestCacheConcurrencyWithSynctest demonstrates deterministic cache concurrency testing.
// Using synctest eliminates timing-dependent test flakiness in concurrent scenarios.
func TestCacheConcurrencyWithSynctest(t *testing.T) {
	t.Run("Concurrent key generation is deterministic", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			keyGen := NewKeyGenerator("synctest_")

			// Test concurrent key generation - in synctest, all goroutines complete deterministically
			var wg sync.WaitGroup
			keys := make([]string, 10)

			for i := 0; i < 10; i++ {
				idx := i
				wg.Go(func() {
					// Simulate some processing time
					time.Sleep(10 * time.Millisecond)
					keys[idx] = keyGen.GenerateKey("test-model", "prompt", nil)
				})
			}
			wg.Wait()

			// synctest.Wait ensures all background work is complete
			synctest.Wait()

			// All keys for the same input should be identical (deterministic)
			for i := 1; i < 10; i++ {
				assert.Equal(t, keys[0], keys[i], "Key %d should match key 0", i)
			}
		})
	})

	t.Run("Virtual time advances with concurrent operations", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			start := time.Now()
			var completionCount atomic.Int32

			var wg sync.WaitGroup
			for i := 0; i < 5; i++ {
				wg.Go(func() {
					// Each goroutine sleeps for 100ms
					time.Sleep(100 * time.Millisecond)
					completionCount.Add(1)
				})
			}
			wg.Wait()

			elapsed := time.Since(start)

			// All goroutines complete
			assert.Equal(t, int32(5), completionCount.Load())

			// Virtual time should have advanced by exactly 100ms (parallel execution)
			// In real time this would be ~100ms, in synctest it's deterministic
			t.Logf("Virtual elapsed: %v", elapsed)
			assert.Equal(t, 100*time.Millisecond, elapsed, "Virtual time should advance exactly 100ms for parallel sleeps")
		})
	})

	t.Run("TTL simulation with virtual time", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			// Simulate a cache entry with TTL
			type cacheEntry struct {
				value     string
				expiresAt time.Time
			}

			ttl := 500 * time.Millisecond
			entry := cacheEntry{
				value:     "cached_value",
				expiresAt: time.Now().Add(ttl),
			}

			// Entry should not be expired yet
			assert.False(t, time.Now().After(entry.expiresAt), "Entry should not be expired initially")

			// Advance virtual time past TTL
			time.Sleep(600 * time.Millisecond)

			// Entry should now be expired
			assert.True(t, time.Now().After(entry.expiresAt), "Entry should be expired after TTL")

			t.Logf("TTL expiration test completed in virtual time (no real waiting)")
		})
	})
}
