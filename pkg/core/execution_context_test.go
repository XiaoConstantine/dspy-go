package core

import (
	"encoding/hex"
	"sync"
	"testing"
	"time"
)

func TestGenerateSpanID(t *testing.T) {
	// Reset generator state
	resetSpanIDGenerator()

	// Test basic functionality
	t.Run("Basic Generation", func(t *testing.T) {
		id := generateSpanID()

		// Verify length (16 characters for 8 bytes hex-encoded)
		if len(id) != 16 {
			t.Errorf("Expected ID length of 16, got %d", len(id))
		}

		// Verify it's valid hex
		_, err := hex.DecodeString(id)
		if err != nil {
			t.Errorf("Invalid hex string: %v", err)
		}
	})

	// Test uniqueness
	t.Run("Uniqueness", func(t *testing.T) {
		const iterations = 10000
		ids := make(map[string]bool)

		for i := 0; i < iterations; i++ {
			id := generateSpanID()
			if ids[id] {
				t.Errorf("Duplicate ID generated: %s", id)
			}
			ids[id] = true
		}
	})

	// Test concurrent generation
	t.Run("Concurrent Generation", func(t *testing.T) {
		const goroutines = 10
		const idsPerRoutine = 1000

		var wg sync.WaitGroup
		ids := make(chan string, goroutines*idsPerRoutine)

		for i := 0; i < goroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < idsPerRoutine; j++ {
					ids <- generateSpanID()
				}
			}()
		}

		wg.Wait()
		close(ids)

		// Check for duplicates
		seen := make(map[string]bool)
		for id := range ids {
			if seen[id] {
				t.Errorf("Duplicate ID generated in concurrent test: %s", id)
			}
			seen[id] = true
		}
	})

	// Test timestamp component
	t.Run("Timestamp Component", func(t *testing.T) {
		// Generate two IDs with a time gap
		id1 := generateSpanID()
		time.Sleep(2 * time.Second)
		id2 := generateSpanID()

		// Convert hex to bytes
		bytes1, _ := hex.DecodeString(id1)
		bytes2, _ := hex.DecodeString(id2)

		// Extract timestamps (first 4 bytes)
		timestamp1 := uint32(bytes1[0])<<24 | uint32(bytes1[1])<<16 | uint32(bytes1[2])<<8 | uint32(bytes1[3])
		timestamp2 := uint32(bytes2[0])<<24 | uint32(bytes2[1])<<16 | uint32(bytes2[2])<<8 | uint32(bytes2[3])

		if timestamp2 <= timestamp1 {
			t.Errorf("Second timestamp not greater than first: %d <= %d", timestamp2, timestamp1)
		}
	})
}

func BenchmarkGenerateSpanID(b *testing.B) {
	for i := 0; i < b.N; i++ {
		generateSpanID()
	}
}
