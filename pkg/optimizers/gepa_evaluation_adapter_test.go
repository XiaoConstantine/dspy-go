package optimizers

import (
	"strings"
	"testing"
	"time"
)

func TestStableHashStringAnyMapIsIndependentOfMapInsertionOrder(t *testing.T) {
	first := map[string]any{
		"inputs": map[string]any{"question": "Why?", "context": "Because."},
		"score":  1.0,
	}
	second := map[string]any{
		"score":  1.0,
		"inputs": map[string]any{"context": "Because.", "question": "Why?"},
	}

	firstHash := stableHashStringAnyMap(first)
	secondHash := stableHashStringAnyMap(second)
	if firstHash != secondHash {
		t.Fatalf("stable hashes differ by map insertion order: %q != %q", firstHash, secondHash)
	}
}

func TestStableHashStringAnyMapPreservesV1ValueEncoding(t *testing.T) {
	hash := stableHashStringAnyMap(map[string]any{
		"duration":  time.Second,
		"nil_map":   map[string]any(nil),
		"nil_slice": []string(nil),
	})

	if strings.HasPrefix(hash, "fallback:") {
		t.Fatalf("stable hash unexpectedly used fallback encoding: %q", hash)
	}
}
