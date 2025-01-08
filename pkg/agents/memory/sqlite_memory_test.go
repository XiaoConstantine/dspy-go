package memory

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLiteStore(t *testing.T) {
	// Create an in-memory database for testing
	store, err := NewSQLiteStore(":memory:")
	require.NoError(t, err)
	defer store.Close()

	t.Run("Basic Store and Retrieve", func(t *testing.T) {
		testData := map[string]interface{}{
			"string": "test value",
			"number": 42,
			"bool":   true,
			"array":  []string{"a", "b", "c"},
			"map":    map[string]int{"one": 1, "two": 2},
		}

		for key, value := range testData {
			err := store.Store(key, value)
			assert.NoError(t, err)

			retrieved, err := store.Retrieve(key)
			assert.NoError(t, err)
			assert.Equal(t, value, retrieved)
		}
	})

	t.Run("List Keys", func(t *testing.T) {
		// Clear existing data
		err := store.Clear()
		require.NoError(t, err)

		// Store some test data
		testKeys := []string{"key1", "key2", "key3"}
		for _, key := range testKeys {
			err := store.Store(key, "value")
			require.NoError(t, err)
		}

		// List keys
		keys, err := store.List()
		assert.NoError(t, err)
		assert.ElementsMatch(t, testKeys, keys)
	})

	t.Run("Clear Store", func(t *testing.T) {
		// Store some data
		err := store.Store("test", "value")
		require.NoError(t, err)

		// Clear the store
		err = store.Clear()
		assert.NoError(t, err)

		// Verify store is empty
		keys, err := store.List()
		assert.NoError(t, err)
		assert.Empty(t, keys)
	})

	t.Run("Non-existent Key", func(t *testing.T) {
		_, err := store.Retrieve("nonexistent")
		assert.Error(t, err)
	})

	t.Run("Concurrent Access", func(t *testing.T) {
		t.Skip()
		const numGoroutines = 10
		done := make(chan bool, numGoroutines)
		var wg sync.WaitGroup

		// Clear any existing data
		err := store.Clear()
		require.NoError(t, err)

		wg.Add(numGoroutines)
		for i := 0; i < numGoroutines; i++ {
			go func(n int) {
				defer wg.Done()

				key := fmt.Sprintf("concurrent_key_%d", n)
				err := store.Store(key, n)
				if !assert.NoError(t, err, "Store failed for key: %s", key) {
					done <- false
					return
				}

				// Small delay to increase chance of concurrent access
				time.Sleep(time.Millisecond * time.Duration(rand.Intn(10)))

				retrieved, err := store.Retrieve(key)
				if !assert.NoError(t, err, "Retrieve failed for key: %s", key) {
					done <- false
					return
				}

				if !assert.Equal(t, n, retrieved, "Value mismatch for key: %s", key) {
					done <- false
					return
				}

				done <- true
			}(i)
		}

		// Wait for all goroutines to finish
		wg.Wait()
		close(done)

		// Check if any goroutine failed
		for success := range done {
			assert.True(t, success, "One or more goroutines failed")
		}
	})

	t.Run("TTL Storage", func(t *testing.T) {
		ctx := context.Background()

		// Store with short TTL
		err := store.StoreWithTTL(ctx, "ttl_key", "ttl_value", 100*time.Millisecond)
		require.NoError(t, err)

		// Verify value exists
		value, err := store.Retrieve("ttl_key")
		assert.NoError(t, err)
		assert.Equal(t, "ttl_value", value)
		currentTime := time.Now().Format(time.RFC3339)
		t.Logf("Current time (UTC): %s", currentTime)
		// Wait for TTL to expire
		time.Sleep(500 * time.Millisecond)
		cleaned, err := store.CleanExpired(ctx)
		assert.NoError(t, err)
		assert.Equal(t, int64(1), cleaned, "Expected one entry to be cleaned")
		// Verify value is gone
		_, err = store.Retrieve("ttl_key")
		assert.Error(t, err)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		// Try to store a channel (cannot be marshaled to JSON)
		ch := make(chan bool)
		err := store.Store("invalid", ch)
		assert.Error(t, err)
	})

	t.Run("Database Connection", func(t *testing.T) {
		// Test invalid database path
		_, err := NewSQLiteStore("/root/forbidden/db.sqlite")
		assert.Error(t, err)

		// Test closing database
		tempStore, err := NewSQLiteStore(":memory:")
		require.NoError(t, err)
		assert.NoError(t, tempStore.Close())

		// Try operations after closing
		err = tempStore.Store("key", "value")
		assert.Error(t, err)
	})
}
