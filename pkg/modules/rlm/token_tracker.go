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

// SubRLMCall represents a nested sub-RLM invocation.
type SubRLMCall struct {
	Query            string        `json:"query"`
	Result           string        `json:"result"`
	Iterations       int           `json:"iterations"`
	Depth            int           `json:"depth"`
	Duration         time.Duration `json:"duration"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
}

// TokenTracker aggregates token usage across root LLM and sub-LLM calls.
type TokenTracker struct {
	mu sync.RWMutex

	// Root LLM usage (orchestration)
	rootPromptTokens     int
	rootCompletionTokens int

	// Sub-LLM usage (Query/QueryBatched calls from REPL)
	subPromptTokens     int
	subCompletionTokens int

	// Sub-RLM usage (nested RLM loops)
	subRLMPromptTokens     int
	subRLMCompletionTokens int

	// Detailed call history
	subCalls    []LLMCall
	subRLMCalls []SubRLMCall
}

// NewTokenTracker creates a new token tracker.
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{
		subCalls:    make([]LLMCall, 0),
		subRLMCalls: make([]SubRLMCall, 0),
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

// AddSubRLMCall adds a sub-RLM call with its token usage.
func (t *TokenTracker) AddSubRLMCall(call SubRLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.subRLMCalls = append(t.subRLMCalls, call)
	t.subRLMPromptTokens += call.PromptTokens
	t.subRLMCompletionTokens += call.CompletionTokens
}

// GetTotalUsage returns the total aggregated token usage.
func (t *TokenTracker) GetTotalUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	totalPrompt := t.rootPromptTokens + t.subPromptTokens + t.subRLMPromptTokens
	totalCompletion := t.rootCompletionTokens + t.subCompletionTokens + t.subRLMCompletionTokens

	return core.TokenUsage{
		PromptTokens:     totalPrompt,
		CompletionTokens: totalCompletion,
		TotalTokens:      totalPrompt + totalCompletion,
	}
}

// GetRootUsage returns token usage from root LLM calls only.
func (t *TokenTracker) GetRootUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return core.TokenUsage{
		PromptTokens:     t.rootPromptTokens,
		CompletionTokens: t.rootCompletionTokens,
		TotalTokens:      t.rootPromptTokens + t.rootCompletionTokens,
	}
}

// GetSubUsage returns token usage from sub-LLM calls only.
func (t *TokenTracker) GetSubUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return core.TokenUsage{
		PromptTokens:     t.subPromptTokens,
		CompletionTokens: t.subCompletionTokens,
		TotalTokens:      t.subPromptTokens + t.subCompletionTokens,
	}
}

// GetSubCalls returns a copy of all sub-LLM calls.
func (t *TokenTracker) GetSubCalls() []LLMCall {
	t.mu.RLock()
	defer t.mu.RUnlock()

	calls := make([]LLMCall, len(t.subCalls))
	copy(calls, t.subCalls)
	return calls
}

// GetSubRLMCalls returns a copy of all sub-RLM calls.
func (t *TokenTracker) GetSubRLMCalls() []SubRLMCall {
	t.mu.RLock()
	defer t.mu.RUnlock()

	calls := make([]SubRLMCall, len(t.subRLMCalls))
	copy(calls, t.subRLMCalls)
	return calls
}

// GetSubRLMUsage returns token usage from sub-RLM calls only.
func (t *TokenTracker) GetSubRLMUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return core.TokenUsage{
		PromptTokens:     t.subRLMPromptTokens,
		CompletionTokens: t.subRLMCompletionTokens,
		TotalTokens:      t.subRLMPromptTokens + t.subRLMCompletionTokens,
	}
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
	t.subRLMPromptTokens = 0
	t.subRLMCompletionTokens = 0
	t.subCalls = make([]LLMCall, 0)
	t.subRLMCalls = make([]SubRLMCall, 0)
}
