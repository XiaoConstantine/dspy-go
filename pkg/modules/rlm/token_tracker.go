package rlm

import (
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// LLMCall represents a sub-LLM call made from within the REPL.
type LLMCall struct {
	Prompt           string        `json:"prompt"`
	Response         string        `json:"response"`
	Duration         time.Duration `json:"duration"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
}

// TokenTracker aggregates token usage across root LLM and sub-LLM calls.
type TokenTracker struct {
	mu sync.Mutex

	// Root LLM usage (orchestration)
	rootPromptTokens     int
	rootCompletionTokens int

	// Sub-LLM usage (Query/QueryBatched calls from REPL)
	subPromptTokens     int
	subCompletionTokens int

	// Detailed call history
	subCalls []LLMCall
}

// NewTokenTracker creates a new token tracker.
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{
		subCalls: make([]LLMCall, 0),
	}
}

// AddRootUsage adds token usage from a root LLM call.
func (t *TokenTracker) AddRootUsage(promptTokens, completionTokens int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rootPromptTokens += promptTokens
	t.rootCompletionTokens += completionTokens
}

// AddSubCall adds a sub-LLM call with its token usage.
func (t *TokenTracker) AddSubCall(call LLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.subCalls = append(t.subCalls, call)
	t.subPromptTokens += call.PromptTokens
	t.subCompletionTokens += call.CompletionTokens
}

// AddSubCalls adds multiple sub-LLM calls.
func (t *TokenTracker) AddSubCalls(calls []LLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, call := range calls {
		t.subCalls = append(t.subCalls, call)
		t.subPromptTokens += call.PromptTokens
		t.subCompletionTokens += call.CompletionTokens
	}
}

// GetTotalUsage returns the total aggregated token usage.
func (t *TokenTracker) GetTotalUsage() core.TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	totalPrompt := t.rootPromptTokens + t.subPromptTokens
	totalCompletion := t.rootCompletionTokens + t.subCompletionTokens

	return core.TokenUsage{
		PromptTokens:     totalPrompt,
		CompletionTokens: totalCompletion,
		TotalTokens:      totalPrompt + totalCompletion,
	}
}

// GetRootUsage returns token usage from root LLM calls only.
func (t *TokenTracker) GetRootUsage() core.TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	return core.TokenUsage{
		PromptTokens:     t.rootPromptTokens,
		CompletionTokens: t.rootCompletionTokens,
		TotalTokens:      t.rootPromptTokens + t.rootCompletionTokens,
	}
}

// GetSubUsage returns token usage from sub-LLM calls only.
func (t *TokenTracker) GetSubUsage() core.TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()

	return core.TokenUsage{
		PromptTokens:     t.subPromptTokens,
		CompletionTokens: t.subCompletionTokens,
		TotalTokens:      t.subPromptTokens + t.subCompletionTokens,
	}
}

// GetSubCalls returns a copy of all sub-LLM calls.
func (t *TokenTracker) GetSubCalls() []LLMCall {
	t.mu.Lock()
	defer t.mu.Unlock()

	calls := make([]LLMCall, len(t.subCalls))
	copy(calls, t.subCalls)
	return calls
}

// ClearSubCalls clears the recorded sub-LLM calls but preserves the counts.
func (t *TokenTracker) ClearSubCalls() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.subCalls = make([]LLMCall, 0)
}

// Reset clears all tracked usage.
func (t *TokenTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.rootPromptTokens = 0
	t.rootCompletionTokens = 0
	t.subPromptTokens = 0
	t.subCompletionTokens = 0
	t.subCalls = make([]LLMCall, 0)
}
