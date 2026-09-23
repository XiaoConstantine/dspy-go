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
	TotalTokens      int           `json:"total_tokens"`
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
	TotalTokens      int           `json:"total_tokens"`
}

// RootIterationSnapshot captures per-iteration root LLM prompt token counts.
// This is the key data needed to prove that RLM context stays bounded:
// if PromptTokens stays flat as iterations increase, context is not accumulating.
type RootIterationSnapshot struct {
	Iteration        int `json:"iteration"`         // 1-indexed iteration number
	PromptTokens     int `json:"prompt_tokens"`     // Root LLM prompt tokens for this iteration
	CompletionTokens int `json:"completion_tokens"` // Root LLM completion tokens for this iteration
}

// TokenTracker aggregates token usage across root LLM and sub-LLM calls.
type TokenTracker struct {
	mu sync.RWMutex

	// Root LLM usage (orchestration)
	rootPromptTokens     int
	rootCompletionTokens int
	rootTotalTokens      int

	// Sub-LLM usage (Query/QueryBatched calls from REPL)
	subPromptTokens     int
	subCompletionTokens int
	subTotalTokens      int

	// Sub-RLM usage (nested RLM loops)
	subRLMPromptTokens     int
	subRLMCompletionTokens int
	subRLMTotalTokens      int

	// Detailed call history
	subCalls    []LLMCall
	subRLMCalls []SubRLMCall

	// Per-iteration root LLM snapshots for context fill ratio analysis
	rootSnapshots []RootIterationSnapshot
}

// NewTokenTracker creates a new token tracker.
func NewTokenTracker() *TokenTracker {
	return &TokenTracker{
		subCalls:      make([]LLMCall, 0),
		subRLMCalls:   make([]SubRLMCall, 0),
		rootSnapshots: make([]RootIterationSnapshot, 0),
	}
}

func normalizedTokenTotal(promptTokens, completionTokens, totalTokens int) int {
	componentTotal := promptTokens + completionTokens
	if totalTokens < componentTotal {
		return componentTotal
	}
	return totalTokens
}

// AddRootUsage adds token usage from a root LLM call.
func (t *TokenTracker) AddRootUsage(promptTokens, completionTokens int) {
	t.AddRootTokenUsage(core.TokenUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	})
}

// AddRootTokenUsage adds a complete root LLM usage record, preserving provider
// totals that include tokens outside the prompt/completion breakdown.
func (t *TokenTracker) AddRootTokenUsage(usage core.TokenUsage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.addRootTokenUsage(usage)
}

// AddRootUsageForIteration adds token usage from a root LLM call and records
// a per-iteration snapshot. The snapshot captures the exact PromptTokens the
// provider reported for this single root call — not a cumulative delta.
// This is the data needed to compute context_fill_ratio per iteration.
func (t *TokenTracker) AddRootUsageForIteration(iteration, promptTokens, completionTokens int) {
	t.AddRootTokenUsageForIteration(iteration, core.TokenUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	})
}

// AddRootTokenUsageForIteration records complete root usage and its iteration snapshot.
func (t *TokenTracker) AddRootTokenUsageForIteration(iteration int, usage core.TokenUsage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.addRootTokenUsage(usage)
	t.rootSnapshots = append(t.rootSnapshots, RootIterationSnapshot{
		Iteration:        iteration,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
	})
}

func (t *TokenTracker) addRootTokenUsage(usage core.TokenUsage) {
	t.rootPromptTokens += usage.PromptTokens
	t.rootCompletionTokens += usage.CompletionTokens
	t.rootTotalTokens += normalizedTokenTotal(usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
}

// AddSubCall adds a sub-LLM call with its token usage.
func (t *TokenTracker) AddSubCall(call LLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.subCalls = append(t.subCalls, call)
	t.subPromptTokens += call.PromptTokens
	t.subCompletionTokens += call.CompletionTokens
	t.subTotalTokens += normalizedTokenTotal(call.PromptTokens, call.CompletionTokens, call.TotalTokens)
}

// AddSubCalls adds multiple sub-LLM calls.
func (t *TokenTracker) AddSubCalls(calls []LLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, call := range calls {
		t.subCalls = append(t.subCalls, call)
		t.subPromptTokens += call.PromptTokens
		t.subCompletionTokens += call.CompletionTokens
		t.subTotalTokens += normalizedTokenTotal(call.PromptTokens, call.CompletionTokens, call.TotalTokens)
	}
}

// AddSubRLMCall adds a sub-RLM call with its token usage.
func (t *TokenTracker) AddSubRLMCall(call SubRLMCall) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.subRLMCalls = append(t.subRLMCalls, call)
	t.subRLMPromptTokens += call.PromptTokens
	t.subRLMCompletionTokens += call.CompletionTokens
	t.subRLMTotalTokens += normalizedTokenTotal(call.PromptTokens, call.CompletionTokens, call.TotalTokens)
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
		TotalTokens:      t.rootTotalTokens + t.subTotalTokens + t.subRLMTotalTokens,
	}
}

// GetRootUsage returns token usage from root LLM calls only.
func (t *TokenTracker) GetRootUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return core.TokenUsage{
		PromptTokens:     t.rootPromptTokens,
		CompletionTokens: t.rootCompletionTokens,
		TotalTokens:      t.rootTotalTokens,
	}
}

// GetSubUsage returns token usage from sub-LLM calls only.
func (t *TokenTracker) GetSubUsage() core.TokenUsage {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return core.TokenUsage{
		PromptTokens:     t.subPromptTokens,
		CompletionTokens: t.subCompletionTokens,
		TotalTokens:      t.subTotalTokens,
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
		TotalTokens:      t.subRLMTotalTokens,
	}
}

// GetRootSnapshots returns a copy of all per-iteration root LLM snapshots.
func (t *TokenTracker) GetRootSnapshots() []RootIterationSnapshot {
	t.mu.RLock()
	defer t.mu.RUnlock()

	snapshots := make([]RootIterationSnapshot, len(t.rootSnapshots))
	copy(snapshots, t.rootSnapshots)
	return snapshots
}

// GetMaxRootPromptTokens returns the largest single root prompt across all iterations.
// Returns 0 if no snapshots have been recorded.
func (t *TokenTracker) GetMaxRootPromptTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	maxTokens := 0
	for _, s := range t.rootSnapshots {
		if s.PromptTokens > maxTokens {
			maxTokens = s.PromptTokens
		}
	}
	return maxTokens
}

// GetMeanRootPromptTokens returns the mean root prompt tokens across all iterations.
// Returns 0 if no snapshots have been recorded.
func (t *TokenTracker) GetMeanRootPromptTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if len(t.rootSnapshots) == 0 {
		return 0
	}
	total := 0
	for _, s := range t.rootSnapshots {
		total += s.PromptTokens
	}
	return total / len(t.rootSnapshots)
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
	t.rootTotalTokens = 0
	t.subPromptTokens = 0
	t.subCompletionTokens = 0
	t.subTotalTokens = 0
	t.subRLMPromptTokens = 0
	t.subRLMCompletionTokens = 0
	t.subRLMTotalTokens = 0
	t.subCalls = make([]LLMCall, 0)
	t.subRLMCalls = make([]SubRLMCall, 0)
	t.rootSnapshots = make([]RootIterationSnapshot, 0)
}
