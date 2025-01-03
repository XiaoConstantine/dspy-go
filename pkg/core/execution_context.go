package core

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ExecutionState holds the mutable state for an execution context.
type ExecutionState struct {
	mu sync.RWMutex

	// Execution metadata
	traceID    string
	spans      []*Span
	activeSpan *Span

	// LLM-specific state
	modelID    string
	tokenUsage *TokenUsage

	// Custom annotations
	annotations map[string]interface{}
}

// Span represents a single operation within the execution.
type Span struct {
	ID          string
	ParentID    string
	Operation   string
	StartTime   time.Time
	EndTime     time.Time
	Error       error
	Annotations map[string]interface{}
}

// TokenUsage tracks token consumption.
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Cost             float64
}

type spanIDGenerator struct {
	// counter ensures uniqueness even with identical timestamps
	counter uint64
	// lastTimestamp helps detect time backwards movement
	lastTimestamp int64
}

// ExecutionContextKey is the type for context keys specific to dspy-go.
type ExecutionContextKey struct {
	name string
}

var (
	stateKey         = &ExecutionContextKey{"dspy-state"}
	defaultGenerator = &spanIDGenerator{}
)

// WithExecutionState creates a new context with dspy-go execution state.
func WithExecutionState(ctx context.Context) context.Context {
	if GetExecutionState(ctx) != nil {
		return ctx // State already exists
	}
	return context.WithValue(ctx, stateKey, &ExecutionState{
		traceID:     generateTraceID(),
		annotations: make(map[string]interface{}),
		spans:       make([]*Span, 0),
	})
}

// GetExecutionState retrieves the execution state from a context.
func GetExecutionState(ctx context.Context) *ExecutionState {
	if state, ok := ctx.Value(stateKey).(*ExecutionState); ok {
		return state
	}
	return nil
}

// StartSpan begins a new operation span.
func StartSpan(ctx context.Context, operation string) (context.Context, *Span) {
	state := GetExecutionState(ctx)
	if state == nil {
		ctx = WithExecutionState(ctx)
		state = GetExecutionState(ctx)
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	span := &Span{
		ID:          generateSpanID(), // Implementation needed
		Operation:   operation,
		StartTime:   time.Now(),
		Annotations: make(map[string]interface{}),
	}

	if state.activeSpan != nil {
		span.ParentID = state.activeSpan.ID
	}

	state.spans = append(state.spans, span)
	state.activeSpan = span

	return ctx, span
}

// EndSpan completes the current span.
func EndSpan(ctx context.Context) {
	if state := GetExecutionState(ctx); state != nil {
		state.mu.Lock()
		defer state.mu.Unlock()

		if state.activeSpan != nil {
			state.activeSpan.EndTime = time.Now()
			state.activeSpan = nil
		}
	}
}

// State modification methods.
func (s *ExecutionState) WithModelID(modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modelID = modelID
}

func (s *ExecutionState) WithTokenUsage(usage *TokenUsage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tokenUsage = usage
}

// State access methods.
func (s *ExecutionState) GetModelID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.modelID
}

func (s *ExecutionState) GetTokenUsage() *TokenUsage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tokenUsage
}

// Span methods.
func (s *Span) WithError(err error) {
	s.Error = err
}

func (s *Span) WithAnnotation(key string, value interface{}) {
	s.Annotations[key] = value
}

// Helper method to collect all spans.
func CollectSpans(ctx context.Context) []*Span {
	if state := GetExecutionState(ctx); state != nil {
		state.mu.RLock()
		defer state.mu.RUnlock()

		spans := make([]*Span, len(state.spans))
		copy(spans, state.spans)
		return spans
	}
	return nil
}

// generateSpanID creates a new unique span identifier.
// The format is: 8 bytes total
// - 4 bytes: timestamp (seconds since epoch)
// - 2 bytes: counter
// - 2 bytes: random data
// This provides a good balance of:
// - Temporal ordering (timestamp component)
// - Uniqueness guarantee (counter component)
// - Collision resistance (random component)
//
// Example:
// 63f51a2a01ab9c8d
// │        │  └─┴─ Random component (2 bytes)
// │        └─┴─ Counter (2 bytes)
// └─┴─┴─┴─ Timestamp (4 bytes).
func generateSpanID() string {
	// Get current timestamp
	now := time.Now().Unix()

	// Increment counter atomically
	counter := atomic.AddUint64(&defaultGenerator.counter, 1)

	// Create buffer for our ID components
	id := make([]byte, 8)

	// Add timestamp (4 bytes)
	id[0] = byte(now >> 24)
	id[1] = byte(now >> 16)
	id[2] = byte(now >> 8)
	id[3] = byte(now)

	// Add counter (2 bytes)
	id[4] = byte(counter >> 8)
	id[5] = byte(counter)

	// Add random component (2 bytes)
	if _, err := rand.Read(id[6:]); err != nil {
		// Fallback to using more counter bits if random fails
		id[6] = byte(counter >> 16)
		id[7] = byte(counter >> 24)
	}

	// Return hex-encoded string
	return hex.EncodeToString(id)
}

// For testing and debugging.
func resetSpanIDGenerator() {
	atomic.StoreUint64(&defaultGenerator.counter, 0)
	defaultGenerator.lastTimestamp = 0
}

func (s *ExecutionState) GetCurrentSpan() *Span {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.activeSpan
}

func generateTraceID() string {
	// Generate 16 random bytes for trace ID
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp-based ID if random generation fails
		now := time.Now().UnixNano()
		return fmt.Sprintf("trace-%d", now)
	}

	// Format as hex string
	return hex.EncodeToString(b)
}

func (s *ExecutionState) GetTraceID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.traceID
}
