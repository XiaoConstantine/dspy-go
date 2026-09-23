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
	modelID          string
	tokenUsage       *TokenUsage
	tokenUsageEvents uint64

	// Custom annotations
	annotations map[string]any
}

// Span represents a single operation within the execution.
type Span struct {
	mu          sync.Mutex
	ID          string
	ParentID    string
	Operation   string
	StartTime   time.Time
	EndTime     time.Time
	Error       error
	Annotations map[string]any
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
	counter atomic.Uint64
}

// ExecutionContextKey is the type for context keys specific to dspy-go.
type ExecutionContextKey struct {
	name string
}

type spanContextFrame struct {
	mu sync.Mutex

	state  *ExecutionState
	span   *Span
	parent *spanContextFrame

	modelID          string
	tokenUsage       TokenUsage
	lastTokenUsage   TokenUsage
	tokenUsageEvents uint64
	hasTokenUsage    bool
}

var (
	stateKey         = &ExecutionContextKey{"dspy-state"}
	spanKey          = &ExecutionContextKey{"dspy-span"}
	defaultGenerator = &spanIDGenerator{}
)

// WithExecutionState creates a new context with dspy-go execution state.
func WithExecutionState(ctx context.Context) context.Context {
	if GetExecutionState(ctx) != nil {
		return ctx // State already exists
	}
	return context.WithValue(ctx, stateKey, &ExecutionState{
		traceID:     generateTraceID(),
		annotations: make(map[string]any),
		spans:       make([]*Span, 0),
	})
}

// WithFreshExecutionState creates a new context with a fresh ExecutionState,
// even if the parent context already has one. This is useful for parallel workers
// that need isolated execution state to avoid mutex contention while still
// inheriting other context values from the parent while preserving the trace ID.
func WithFreshExecutionState(ctx context.Context) context.Context {
	traceID := generateTraceID()
	if state := GetExecutionState(ctx); state != nil {
		traceID = state.GetTraceID()
	}
	return context.WithValue(ctx, stateKey, &ExecutionState{
		traceID:     traceID,
		annotations: make(map[string]any),
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

// RecordModelCall stores the current model identifier in execution state so
// logs and traces can attribute downstream events to the active model.
func RecordModelCall(ctx context.Context, model ModelIdentifier) {
	if model == nil {
		return
	}
	state := GetExecutionState(ctx)
	if state == nil {
		return
	}

	// This helper is best-effort observability. Avoid letting a misconfigured
	// mock or custom LLM implementation crash execution while reading ModelID.
	modelID := ""
	func() {
		defer func() {
			if recover() != nil {
				modelID = ""
			}
		}()
		modelID = model.ModelID()
	}()
	if modelID == "" {
		return
	}
	state.WithModelID(modelID)
	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return
	}
	for current := frame; current != nil; current = current.parent {
		current.mu.Lock()
		current.modelID = modelID
		current.mu.Unlock()
	}
}

// RecordLLMCall stores the current LLM model identifier in execution state so
// logs and traces can attribute downstream events to the active model.
func RecordLLMCall(ctx context.Context, llm LLM) {
	RecordModelCall(ctx, llm)
}

// RecordTokenUsage records one LLM usage event on the shared legacy execution
// state and aggregates it into this context branch and each of its ancestors.
// TotalTokens is normalized to at least PromptTokens + CompletionTokens.
func RecordTokenUsage(ctx context.Context, usage *TokenUsage) {
	if usage == nil {
		return
	}
	state := GetExecutionState(ctx)
	if state == nil {
		return
	}
	normalized := *usage
	if componentTotal := normalized.PromptTokens + normalized.CompletionTokens; componentTotal > normalized.TotalTokens {
		normalized.TotalTokens = componentTotal
	}
	state.WithTokenUsage(&normalized)

	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return
	}
	for current := frame; current != nil; current = current.parent {
		current.mu.Lock()
		current.tokenUsage.PromptTokens += normalized.PromptTokens
		current.tokenUsage.CompletionTokens += normalized.CompletionTokens
		current.tokenUsage.TotalTokens += normalized.TotalTokens
		current.tokenUsage.Cost += normalized.Cost
		current.lastTokenUsage = normalized
		current.tokenUsageEvents++
		current.hasTokenUsage = true
		current.mu.Unlock()
	}
}

// ModelIDFromContext returns the most recently recorded model for this context
// branch without consulting a concurrent sibling branch.
func ModelIDFromContext(ctx context.Context) string {
	state := GetExecutionState(ctx)
	if state == nil {
		return ""
	}
	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return ""
	}
	frame.mu.Lock()
	defer frame.mu.Unlock()
	return frame.modelID
}

// TokenUsageFromContext returns aggregate token usage for this context branch.
// The returned value is a copy and cannot be mutated to alter execution state.
func TokenUsageFromContext(ctx context.Context) *TokenUsage {
	state := GetExecutionState(ctx)
	if state == nil {
		return nil
	}
	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return nil
	}
	frame.mu.Lock()
	defer frame.mu.Unlock()
	if !frame.hasTokenUsage {
		return nil
	}
	usage := frame.tokenUsage
	return &usage
}

// LatestTokenUsageFromContext returns the latest token-usage event recorded on
// this context branch. The returned value is a copy.
func LatestTokenUsageFromContext(ctx context.Context) *TokenUsage {
	state := GetExecutionState(ctx)
	if state == nil {
		return nil
	}
	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return nil
	}
	frame.mu.Lock()
	defer frame.mu.Unlock()
	if !frame.hasTokenUsage {
		return nil
	}
	usage := frame.lastTokenUsage
	return &usage
}

// TokenUsageEventCount returns the number of usage events recorded on this
// context branch. Contexts without a span frame use the legacy state counter.
func TokenUsageEventCount(ctx context.Context) uint64 {
	state := GetExecutionState(ctx)
	if state == nil {
		return 0
	}
	if frame, ok := ctx.Value(spanKey).(*spanContextFrame); ok && frame.state == state {
		frame.mu.Lock()
		defer frame.mu.Unlock()
		return frame.tokenUsageEvents
	}
	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.tokenUsageEvents
}

// StartSpan begins a new operation span. Callers must propagate the returned
// context to nested work and pass it to EndSpan.
func StartSpan(ctx context.Context, operation string) (context.Context, *Span) {
	return StartSpanWithContext(ctx, operation, "", nil)
}

// StartSpanWithContext begins a new operation span with additional context
// information. Parentage is derived only from the supplied context; callers
// must propagate the returned context to nested work and pass it to EndSpan.
func StartSpanWithContext(ctx context.Context, operation string, moduleName string, metadata map[string]any) (context.Context, *Span) {
	state := GetExecutionState(ctx)
	if state == nil {
		ctx = WithExecutionState(ctx)
		state = GetExecutionState(ctx)
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	// Create display name with module context
	displayName := operation
	if moduleName != "" {
		displayName = fmt.Sprintf("%s (%s)", operation, moduleName)
	}

	// Initialize annotations with provided metadata
	annotations := make(map[string]any)
	for k, v := range metadata {
		annotations[k] = v
	}

	// Add module information to annotations
	if moduleName != "" {
		moduleInfo := map[string]any{
			"name": moduleName,
		}
		if moduleType, ok := metadata["module_type"].(string); ok {
			moduleInfo["type"] = moduleType
		}
		if moduleConfig, ok := metadata["module_config"]; ok {
			moduleInfo["config"] = moduleConfig
		}
		annotations["module"] = moduleInfo
	}

	span := &Span{
		ID:          generateSpanID(),
		Operation:   displayName,
		StartTime:   time.Now(),
		Annotations: annotations,
	}

	// Parentage belongs to the context branch, not to the most recently started
	// span in the shared execution state. This keeps sibling goroutines from
	// becoming accidental parent/child spans.
	var parent *spanContextFrame
	if frame, ok := ctx.Value(spanKey).(*spanContextFrame); ok && frame.state == state {
		parent = frame
		span.ParentID = parent.span.ID
	}

	state.spans = append(state.spans, span)
	state.activeSpan = span
	ctx = context.WithValue(ctx, spanKey, &spanContextFrame{
		state:  state,
		span:   span,
		parent: parent,
	})

	return ctx, span
}

// EndSpan completes the span associated with ctx. The context must be the one
// returned by StartSpan or StartSpanWithContext. When multiple context branches
// share one ExecutionState, ending one branch never closes another branch's
// span.
func EndSpan(ctx context.Context) {
	state := GetExecutionState(ctx)
	if state == nil {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return
	}
	target := frame.span
	var parent *Span
	if frame.parent != nil {
		parent = frame.parent.span
	}
	if target == nil {
		return
	}

	target.mu.Lock()
	if target.EndTime.IsZero() {
		target.EndTime = time.Now()
	}
	target.mu.Unlock()

	if state.activeSpan == target {
		state.activeSpan = parent
	}
}

// State modification methods.
func (s *ExecutionState) WithModelID(modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.modelID = modelID
}

// WithTokenUsage sets the shared legacy usage snapshot. Context-aware callers
// should use RecordTokenUsage so concurrent branches remain distinguishable.
func (s *ExecutionState) WithTokenUsage(usage *TokenUsage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if usage == nil {
		s.tokenUsage = nil
		return
	}
	copy := *usage
	s.tokenUsage = &copy
	s.tokenUsageEvents++
}

// State access methods.
func (s *ExecutionState) GetModelID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.modelID
}

// GetTokenUsage returns a copy of the shared legacy usage snapshot. Use
// TokenUsageFromContext for branch-aware aggregate accounting.
func (s *ExecutionState) GetTokenUsage() *TokenUsage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.tokenUsage == nil {
		return nil
	}
	copy := *s.tokenUsage
	return &copy
}

// Span methods.
func (s *Span) WithError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Error = err
}

func (s *Span) WithAnnotation(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Annotations == nil {
		s.Annotations = make(map[string]any)
	}
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
	counter := defaultGenerator.counter.Add(1)

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
	defaultGenerator.counter.Store(0)
}

// SpanFromContext returns the innermost span associated with this context
// branch. Unlike ExecutionState.GetCurrentSpan, it cannot return a span from a
// concurrent sibling branch.
func SpanFromContext(ctx context.Context) *Span {
	state := GetExecutionState(ctx)
	if state == nil {
		return nil
	}
	frame, ok := ctx.Value(spanKey).(*spanContextFrame)
	if !ok || frame.state != state {
		return nil
	}
	return frame.span
}

// GetCurrentSpan returns the shared legacy active span. It is retained for
// compatibility with serial callers; branch-aware code should use
// SpanFromContext.
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
