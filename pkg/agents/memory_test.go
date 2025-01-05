package agents

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInMemoryStore(t *testing.T) {
	t.Run("Basic Operations", func(t *testing.T) {
		store := NewInMemoryStore()

		// Test Store
		err := store.Store("key1", "value1")
		require.NoError(t, err)

		// Test Retrieve
		value, err := store.Retrieve("key1")
		require.NoError(t, err)
		assert.Equal(t, "value1", value)

		// Test non-existent key
		_, err = store.Retrieve("nonexistent")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not found")

		// Test List
		keys, err := store.List()
		require.NoError(t, err)
		assert.Contains(t, keys, "key1")

		// Test Clear
		err = store.Clear()
		require.NoError(t, err)
		keys, err = store.List()
		require.NoError(t, err)
		assert.Empty(t, keys)
	})

	t.Run("Concurrent Operations", func(t *testing.T) {
		store := NewInMemoryStore()
		var wg sync.WaitGroup
		iterations := 100

		// Concurrent writes
		for i := 0; i < iterations; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				err := store.Store(fmt.Sprintf("key%d", i), i)
				assert.NoError(t, err)
			}(i)
		}

		// Concurrent reads
		for i := 0; i < iterations; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				_, err := store.Retrieve(fmt.Sprintf("key%d", i))
				// Error is acceptable since we're reading concurrently with writes
				if err != nil {
					assert.Contains(t, err.Error(), "not found")
				}
			}(i)
		}

		wg.Wait()

		// Verify final state
		keys, err := store.List()
		require.NoError(t, err)
		assert.Len(t, keys, iterations)
	})

	t.Run("Store Different Types", func(t *testing.T) {
		store := NewInMemoryStore()

		testCases := []struct {
			key   string
			value interface{}
		}{
			{"string", "test"},
			{"int", 42},
			{"float", 3.14},
			{"bool", true},
			{"slice", []string{"a", "b", "c"}},
			{"map", map[string]int{"a": 1, "b": 2}},
			{"struct", struct{ Name string }{"test"}},
		}

		for _, tc := range testCases {
			t.Run(tc.key, func(t *testing.T) {
				err := store.Store(tc.key, tc.value)
				require.NoError(t, err)

				retrieved, err := store.Retrieve(tc.key)
				require.NoError(t, err)
				assert.Equal(t, tc.value, retrieved)
			})
		}
	})
}
