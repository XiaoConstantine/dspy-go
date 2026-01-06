package logging

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func TestNewTraceSession(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	session, err := NewTraceSession(tracePath)
	if err != nil {
		t.Fatalf("Failed to create trace session: %v", err)
	}
	defer session.Close()

	if session.TraceID() == "" {
		t.Error("Expected non-empty trace ID")
	}

	if session.startTime.IsZero() {
		t.Error("Expected non-zero start time")
	}
}

func TestTraceSessionEmitEvents(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	session, err := NewTraceSession(tracePath)
	if err != nil {
		t.Fatalf("Failed to create trace session: %v", err)
	}

	err = session.EmitSpanStart("span-1", "", "test_operation", map[string]any{"input": "value"})
	if err != nil {
		t.Errorf("EmitSpanStart failed: %v", err)
	}

	err = session.EmitLLMCall("span-1", "openai", "gpt-4", "prompt", "response", &core.TokenUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}, 100)
	if err != nil {
		t.Errorf("EmitLLMCall failed: %v", err)
	}

	err = session.EmitModule("span-1", "Predict", "qa", "question -> answer", map[string]any{"question": "test"}, map[string]any{"answer": "result"}, 200, 1, 30, true)
	if err != nil {
		t.Errorf("EmitModule failed: %v", err)
	}

	err = session.EmitCodeExec("span-1", 1, "print('hello')", "hello\n", "", nil, 50, nil)
	if err != nil {
		t.Errorf("EmitCodeExec failed: %v", err)
	}

	err = session.EmitToolCall("span-1", "calculator", map[string]any{"expr": "2+2"}, 4, 10, true, nil)
	if err != nil {
		t.Errorf("EmitToolCall failed: %v", err)
	}

	err = session.EmitError("span-1", "ValidationError", "invalid input", true)
	if err != nil {
		t.Errorf("EmitError failed: %v", err)
	}

	err = session.EmitSpanEnd("span-1", map[string]any{"output": "result"}, nil, 500)
	if err != nil {
		t.Errorf("EmitSpanEnd failed: %v", err)
	}

	session.Close()

	content, err := os.ReadFile(tracePath)
	if err != nil {
		t.Fatalf("Failed to read trace file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 8 {
		t.Errorf("Expected 8 events, got %d", len(lines))
	}

	var sessionEvent TraceEvent
	if err := json.Unmarshal([]byte(lines[0]), &sessionEvent); err != nil {
		t.Fatalf("Failed to parse session event: %v", err)
	}
	if sessionEvent.Type != TraceEventSession {
		t.Errorf("Expected session event, got %s", sessionEvent.Type)
	}

	var llmEvent TraceEvent
	if err := json.Unmarshal([]byte(lines[2]), &llmEvent); err != nil {
		t.Fatalf("Failed to parse LLM event: %v", err)
	}
	if llmEvent.Type != TraceEventLLMCall {
		t.Errorf("Expected llm_call event, got %s", llmEvent.Type)
	}
	if llmEvent.Data["model"] != "gpt-4" {
		t.Errorf("Expected model gpt-4, got %v", llmEvent.Data["model"])
	}
}

func TestTraceOutputJSONLFormat(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	output, err := NewTraceOutput(tracePath)
	if err != nil {
		t.Fatalf("Failed to create trace output: %v", err)
	}

	events := []TraceEvent{
		{
			Type:      TraceEventSession,
			Timestamp: time.Now(),
			TraceID:   "trace-123",
			Data:      map[string]interface{}{"key": "value"},
		},
		{
			Type:      TraceEventSpan,
			Timestamp: time.Now(),
			TraceID:   "trace-123",
			SpanID:    "span-1",
			Data:      map[string]interface{}{"operation": "test"},
		},
	}

	for _, event := range events {
		if err := output.Write(event); err != nil {
			t.Errorf("Failed to write event: %v", err)
		}
	}

	output.Close()

	content, err := os.ReadFile(tracePath)
	if err != nil {
		t.Fatalf("Failed to read trace file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 2 {
		t.Errorf("Expected 2 lines, got %d", len(lines))
	}

	for i, line := range lines {
		var parsed TraceEvent
		if err := json.Unmarshal([]byte(line), &parsed); err != nil {
			t.Errorf("Line %d is not valid JSON: %v", i+1, err)
		}
	}
}

func TestTraceOutputRotation(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	output, err := NewTraceOutput(tracePath, WithTraceRotation(500, 2))
	if err != nil {
		t.Fatalf("Failed to create trace output: %v", err)
	}

	for i := 0; i < 20; i++ {
		event := TraceEvent{
			Type:      TraceEventSpan,
			Timestamp: time.Now(),
			TraceID:   "trace-123",
			SpanID:    "span-1",
			Data:      map[string]interface{}{"iteration": i, "padding": "some longer data to fill up the file"},
		}
		if err := output.Write(event); err != nil {
			t.Errorf("Failed to write event: %v", err)
		}
	}

	output.Close()

	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read dir: %v", err)
	}

	if len(files) < 2 {
		t.Errorf("Expected at least 2 files after rotation, got %d", len(files))
	}
}

func TestStartTraceSessionWithContext(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	ctx := core.WithExecutionState(context.Background())
	state := core.GetExecutionState(ctx)
	expectedTraceID := state.GetTraceID()

	session, err := StartTraceSession(ctx, tracePath, map[string]any{"app": "test"})
	if err != nil {
		t.Fatalf("Failed to start trace session: %v", err)
	}
	defer session.Close()

	if session.TraceID() != expectedTraceID {
		t.Errorf("Expected trace ID %s, got %s", expectedTraceID, session.TraceID())
	}
}

func TestTraceSessionConcurrency(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	session, err := NewTraceSession(tracePath)
	if err != nil {
		t.Fatalf("Failed to create trace session: %v", err)
	}
	defer session.Close()

	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(idx int) {
			for j := 0; j < 10; j++ {
				_ = session.EmitSpanStart("span", "", "op", nil)
				_ = session.EmitSpanEnd("span", nil, nil, 100)
			}
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestRLMTraceSession(t *testing.T) {
	tmpDir := t.TempDir()

	session, err := NewRLMTraceSession(tmpDir, RLMSessionConfig{
		RootModel:     "gpt-4",
		MaxIterations: 10,
		Backend:       "openai",
		BackendKwargs: map[string]any{"temperature": 0.7},
		Context:       "This is test context for RLM.",
		Query:         "What is the answer?",
	})
	if err != nil {
		t.Fatalf("Failed to create RLM trace session: %v", err)
	}
	defer session.Close()

	if session.Path() == "" {
		t.Error("Expected non-empty path")
	}

	// Log an iteration with code execution
	err = session.LogIteration(
		[]RLMMessage{{Role: "user", Content: "What is 2+2?"}},
		"Let me calculate that using code.",
		[]RLMCodeBlock{{
			Code: "result = 2 + 2",
			Result: RLMCodeResult{
				Stdout:        "4",
				Stderr:        "",
				Locals:        map[string]any{"result": 4},
				ExecutionTime: 0.01,
				RLMCalls:      nil,
			},
		}},
		nil,
		100*time.Millisecond,
	)
	if err != nil {
		t.Errorf("LogIteration failed: %v", err)
	}

	// Log a final iteration
	err = session.LogIteration(
		[]RLMMessage{{Role: "user", Content: "What is 2+2?"}},
		"The answer is 4.",
		[]RLMCodeBlock{},
		"4",
		50*time.Millisecond,
	)
	if err != nil {
		t.Errorf("LogIteration with final answer failed: %v", err)
	}

	session.Close()

	// Verify the file content is valid JSONL
	content, err := os.ReadFile(session.Path())
	if err != nil {
		t.Fatalf("Failed to read trace file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 3 { // metadata + 2 iterations
		t.Errorf("Expected 3 lines, got %d", len(lines))
	}

	// Verify metadata entry
	var metadata RLMMetadataEntry
	if err := json.Unmarshal([]byte(lines[0]), &metadata); err != nil {
		t.Errorf("Failed to parse metadata: %v", err)
	}
	if metadata.Type != "metadata" {
		t.Errorf("Expected type 'metadata', got '%s'", metadata.Type)
	}
	if metadata.RootModel != "gpt-4" {
		t.Errorf("Expected root_model 'gpt-4', got '%s'", metadata.RootModel)
	}

	// Verify iteration entries
	var iter1 RLMIterationEntry
	if err := json.Unmarshal([]byte(lines[1]), &iter1); err != nil {
		t.Errorf("Failed to parse iteration 1: %v", err)
	}
	if iter1.Type != "iteration" {
		t.Errorf("Expected type 'iteration', got '%s'", iter1.Type)
	}
	if iter1.Iteration != 1 {
		t.Errorf("Expected iteration 1, got %d", iter1.Iteration)
	}

	var iter2 RLMIterationEntry
	if err := json.Unmarshal([]byte(lines[2]), &iter2); err != nil {
		t.Errorf("Failed to parse iteration 2: %v", err)
	}
	if iter2.FinalAnswer != "4" {
		t.Errorf("Expected final answer '4', got '%v'", iter2.FinalAnswer)
	}
}

func TestRLMTraceSessionWithSubLLMCalls(t *testing.T) {
	tmpDir := t.TempDir()

	session, err := NewRLMTraceSession(tmpDir, RLMSessionConfig{
		RootModel:     "claude-3",
		MaxIterations: 5,
		Backend:       "anthropic",
		Query:         "Summarize the document",
	})
	if err != nil {
		t.Fatalf("Failed to create RLM trace session: %v", err)
	}
	defer session.Close()

	// Log iteration with sub-LLM calls
	err = session.LogIteration(
		[]RLMMessage{
			{Role: "system", Content: "You are an assistant."},
			{Role: "user", Content: "Summarize the document"},
		},
		"I will query sections of the document.",
		[]RLMCodeBlock{{
			Code: `summary = Query("Summarize section 1")`,
			Result: RLMCodeResult{
				Stdout:        "Section 1 discusses...",
				Stderr:        "",
				Locals:        map[string]any{"summary": "Section 1 discusses..."},
				ExecutionTime: 1.5,
				RLMCalls: []RLMCallEntry{{
					Prompt:           "Summarize section 1",
					Response:         "Section 1 discusses the main topic.",
					PromptTokens:     100,
					CompletionTokens: 50,
					ExecutionTime:    1.2,
				}},
			},
		}},
		nil,
		2*time.Second,
	)
	if err != nil {
		t.Errorf("LogIteration failed: %v", err)
	}

	session.Close()

	// Verify the file can be parsed
	content, err := os.ReadFile(session.Path())
	if err != nil {
		t.Fatalf("Failed to read trace file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) != 2 { // metadata + 1 iteration
		t.Errorf("Expected 2 lines, got %d", len(lines))
	}

	// Verify iteration has RLM calls
	var iter RLMIterationEntry
	if err := json.Unmarshal([]byte(lines[1]), &iter); err != nil {
		t.Errorf("Failed to parse iteration: %v", err)
	}
	if len(iter.CodeBlocks) != 1 {
		t.Errorf("Expected 1 code block, got %d", len(iter.CodeBlocks))
	}
	if len(iter.CodeBlocks[0].Result.RLMCalls) != 1 {
		t.Errorf("Expected 1 RLM call, got %d", len(iter.CodeBlocks[0].Result.RLMCalls))
	}
}

func TestContextTraceSession(t *testing.T) {
	tmpDir := t.TempDir()
	tracePath := filepath.Join(tmpDir, "test.jsonl")

	session, err := NewTraceSession(tracePath)
	if err != nil {
		t.Fatalf("Failed to create trace session: %v", err)
	}
	defer session.Close()

	// Test context integration
	ctx := context.Background()
	ctx = WithTraceSession(ctx, session)

	retrieved := GetTraceSession(ctx)
	if retrieved != session {
		t.Error("Expected to retrieve same session from context")
	}

	// Test nil case
	nilCtx := context.Background()
	nilSession := GetTraceSession(nilCtx)
	if nilSession != nil {
		t.Error("Expected nil session from context without session")
	}
}
