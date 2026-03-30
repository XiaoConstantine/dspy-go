package rlm

import (
	"context"
	"testing"
)

func TestExecuteFinalVarUnquoted(t *testing.T) {
	repl, err := NewYaegiREPL(nil)
	if err != nil {
		t.Fatalf("NewYaegiREPL() error = %v", err)
	}

	result, err := repl.Execute(context.Background(), `
res := "repo overview"
FINAL_VAR(res)
`)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("Execute() stderr = %q, want empty", result.Stderr)
	}
	if !repl.HasFinal() {
		t.Fatal("HasFinal() = false, want true")
	}
	if repl.Final() != "repo overview" {
		t.Fatalf("Final() = %q, want %q", repl.Final(), "repo overview")
	}
}

func TestExecuteNormalizesQuotedFinalVarIdentifier(t *testing.T) {
	repl, err := NewYaegiREPL(nil)
	if err != nil {
		t.Fatalf("NewYaegiREPL() error = %v", err)
	}

	result, err := repl.Execute(context.Background(), `
res := "repo overview"
FINAL_VAR("res")
`)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("Execute() stderr = %q, want empty", result.Stderr)
	}
	if !repl.HasFinal() {
		t.Fatal("HasFinal() = false, want true")
	}
	if repl.Final() != "repo overview" {
		t.Fatalf("Final() = %q, want %q", repl.Final(), "repo overview")
	}
}
