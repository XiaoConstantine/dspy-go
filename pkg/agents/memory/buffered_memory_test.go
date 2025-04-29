package memory

import (
	"context"
	goerrors "errors" // Alias standard errors package
	"fmt"
	"sync"
	"testing"

	pkgErrors "github.com/XiaoConstantine/dspy-go/pkg/errors" // Alias custom errors package
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func assertHistoryEquals(t *testing.T, mem *BufferedMemory, expected []Message) {
	t.Helper()
	ctx := context.Background()
	history, err := mem.Get(ctx)
	require.NoError(t, err)
	assert.Equal(t, expected, history)
}

func TestNewBufferedMemory(t *testing.T) {
	t.Run("ValidMaxSize", func(t *testing.T) {
		mem := NewBufferedMemory(5)
		require.NotNil(t, mem)
		assert.Equal(t, 5, mem.maxSize)
		assert.Equal(t, defaultHistoryKey, mem.historyKey)
		require.NotNil(t, mem.store)
		// Check underlying store is initialized (should be empty)
		_, err := mem.store.Retrieve(mem.historyKey)
		require.Error(t, err)
		var dspyErr *pkgErrors.Error // Use alias
		require.ErrorAs(t, err, &dspyErr)
		assert.Equal(t, pkgErrors.ResourceNotFound, dspyErr.Code()) // Use alias
	})

	t.Run("ZeroMaxSizeDefaultsToOne", func(t *testing.T) {
		mem := NewBufferedMemory(0)
		require.NotNil(t, mem)
		assert.Equal(t, 1, mem.maxSize) // Should default to 1
	})

	t.Run("NegativeMaxSizeDefaultsToOne", func(t *testing.T) {
		mem := NewBufferedMemory(-5)
		require.NotNil(t, mem)
		assert.Equal(t, 1, mem.maxSize) // Should default to 1
	})
}

func TestBufferedMemory_AddAndGet(t *testing.T) {
	t.Run("AddWithinLimit", func(t *testing.T) {
		mem := NewBufferedMemory(3)
		ctx := context.Background()

		err := mem.Add(ctx, "user", "msg1")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{{Role: "user", Content: "msg1"}})

		err = mem.Add(ctx, "assistant", "msg2")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{
			{Role: "user", Content: "msg1"},
			{Role: "assistant", Content: "msg2"},
		})

		err = mem.Add(ctx, "user", "msg3")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{
			{Role: "user", Content: "msg1"},
			{Role: "assistant", Content: "msg2"},
			{Role: "user", Content: "msg3"},
		})
	})

	t.Run("AddExceedingLimit", func(t *testing.T) {
		mem := NewBufferedMemory(2)
		ctx := context.Background()

		_ = mem.Add(ctx, "user", "msg1")
		_ = mem.Add(ctx, "assistant", "msg2")
		assertHistoryEquals(t, mem, []Message{
			{Role: "user", Content: "msg1"},
			{Role: "assistant", Content: "msg2"},
		})

		// Add third message, msg1 should be dropped
		err := mem.Add(ctx, "user", "msg3")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{
			{Role: "assistant", Content: "msg2"},
			{Role: "user", Content: "msg3"},
		})

		// Add fourth message, msg2 should be dropped
		err = mem.Add(ctx, "assistant", "msg4")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{
			{Role: "user", Content: "msg3"},
			{Role: "assistant", Content: "msg4"},
		})
	})

	t.Run("MaxSizeOne", func(t *testing.T) {
		mem := NewBufferedMemory(1)
		ctx := context.Background()

		_ = mem.Add(ctx, "user", "msg1")
		assertHistoryEquals(t, mem, []Message{{Role: "user", Content: "msg1"}})

		_ = mem.Add(ctx, "assistant", "msg2")
		assertHistoryEquals(t, mem, []Message{{Role: "assistant", Content: "msg2"}})
	})
}

func TestBufferedMemory_GetEmpty(t *testing.T) {
	mem := NewBufferedMemory(5)
	ctx := context.Background()

	history, err := mem.Get(ctx)
	require.NoError(t, err)   // Expect no error when getting empty history
	assert.Empty(t, history)  // Expect an empty slice
	assert.NotNil(t, history) // Ensure it's an empty slice, not nil
}

func TestBufferedMemory_Clear(t *testing.T) {
	t.Run("ClearEmpty", func(t *testing.T) {
		mem := NewBufferedMemory(5)
		ctx := context.Background()
		err := mem.Clear(ctx)
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{})
	})

	t.Run("ClearPopulated", func(t *testing.T) {
		mem := NewBufferedMemory(3)
		ctx := context.Background()
		_ = mem.Add(ctx, "user", "msg1")
		_ = mem.Add(ctx, "assistant", "msg2")

		err := mem.Clear(ctx)
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{})

		// Add again after clearing
		err = mem.Add(ctx, "user", "msg3")
		require.NoError(t, err)
		assertHistoryEquals(t, mem, []Message{{Role: "user", Content: "msg3"}})
	})
}

func TestBufferedMemory_CorruptedData(t *testing.T) {
	// These tests simulate cases where the underlying store might hold invalid data.
	t.Run("NonByteSliceData", func(t *testing.T) {
		mem := NewBufferedMemory(5)
		ctx := context.Background() // Define ctx here

		// Manually store invalid data type (integer) in the underlying store
		err := mem.store.Store(mem.historyKey, 12345)
		require.NoError(t, err)

		_, getErr := mem.Get(ctx)
		require.Error(t, getErr)

		var dspyErr *pkgErrors.Error // Use alias
		require.ErrorAs(t, getErr, &dspyErr)
		assert.Equal(t, pkgErrors.InvalidResponse, dspyErr.Code()) // Use alias
		assert.Contains(t, dspyErr.Error(), "stored history is not []byte or string")
	})

	t.Run("InvalidJSONData", func(t *testing.T) {
		mem := NewBufferedMemory(5)
		ctx := context.Background() // Define ctx here

		// Manually store invalid JSON bytes in the underlying store
		invalidJSON := []byte("this is not json{")
		err := mem.store.Store(mem.historyKey, invalidJSON)
		require.NoError(t, err)

		_, getErr := mem.Get(ctx)
		require.Error(t, getErr)

		var dspyErr *pkgErrors.Error // Use alias
		require.ErrorAs(t, getErr, &dspyErr)
		assert.Equal(t, pkgErrors.InvalidResponse, dspyErr.Code()) // Use alias
		assert.Contains(t, dspyErr.Error(), "failed to unmarshal history")
	})

	t.Run("EmptyByteSliceData", func(t *testing.T) {
		mem := NewBufferedMemory(5)
		// ctx is not needed here as we only call assertHistoryEquals which defines its own ctx

		// Manually store empty byte slice (different from key not found)
		emptyBytes := []byte{}
		err := mem.store.Store(mem.historyKey, emptyBytes)
		require.NoError(t, err)

		// Get should return empty slice, not error
		assertHistoryEquals(t, mem, []Message{})
	})
}

func TestBufferedMemory_Concurrency(t *testing.T) {
	mem := NewBufferedMemory(100) // Use a larger buffer to avoid eviction races
	ctx := context.Background()
	numGoroutines := 50
	addsPerG := 10

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2) // For Add and Get goroutines

	// Concurrent Adds
	for i := 0; i < numGoroutines; i++ {
		go func(gID int) {
			defer wg.Done()
			for j := 0; j < addsPerG; j++ {
				msg := fmt.Sprintf("msg-%d-%d", gID, j)
				role := "user"
				if j%2 != 0 {
					role = "assistant"
				}
				err := mem.Add(ctx, role, msg)
				assert.NoError(t, err) // Assert no error during concurrent adds
			}
		}(i)
	}

	// Concurrent Gets
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, err := mem.Get(ctx)
			// We might get ResourceNotFound initially if Get runs before Add, which is okay.
			if err != nil {
				var dspyErr *pkgErrors.Error // Use alias for custom error type
				// Use aliased standard errors package for As check
				if !goerrors.As(err, &dspyErr) || dspyErr.Code() != pkgErrors.ResourceNotFound {
					assert.NoError(t, err, "Unexpected error during concurrent Get")
				}
			}
		}()
	}

	wg.Wait()

	// Final check on history length
	history, err := mem.Get(ctx)
	require.NoError(t, err)
	assert.Len(t, history, mem.maxSize) // Should contain maxSize items
}

// Mock Underlying Store to test Add error paths (Optional, as InMemoryStore is simple)
// If the underlying store had complex error conditions (e.g., network issues for a remote store),
// you would mock agents.Memory and inject it into BufferedMemory to test those paths.
// For now, the InMemoryStore used is simple enough that direct testing covers most paths.
