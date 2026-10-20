package main

import (
	"context"
	"io"
	"path/filepath"
	"testing"
)

func TestInvalidArguments(t *testing.T) {
	for _, args := range [][2]string{{"", "parse_config"}, {t.TempDir(), ""}, {filepath.Join(t.TempDir(), "missing"), "parse_config"}} {
		if err := run(context.Background(), args[0], args[1], io.Discard); err == nil {
			t.Fatalf("expected error for %v", args)
		}
	}
}

func TestLookupContract(t *testing.T) {
	tool := lookupTool(nil)
	if tool.Name() != "sem_find" || len(tool.InputSchema().Properties) != 3 {
		t.Fatal("expected the narrow sem_find contract")
	}
	if err := tool.Validate(map[string]any{}); err == nil {
		t.Fatal("query is required")
	}
	if err := tool.Validate(map[string]any{"query": "parse_config"}); err != nil {
		t.Fatal(err)
	}
}
