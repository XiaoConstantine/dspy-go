package rlm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

const (
	truncateLenShort  = 100
	truncateLenMedium = 200
	truncateLenLong   = 5000
)

var (
	ErrMissingContext      = errors.New("missing required input: context")
	ErrMissingQuery        = errors.New("missing or invalid required input: query")
	ErrTokenBudgetExceeded = errors.New("RLM token budget exceeded")
)

// CodeBlock represents an extracted and executed code block.
type CodeBlock struct {
	Code   string
	Result ExecutionResult
}

// HistoryEntry represents a single iteration in the RLM history.
// This provides immutable, structured history for debugging/checkpointing.
type HistoryEntry struct {
	Iteration int       // 1-indexed iteration number
	Timestamp time.Time // When this iteration started
	Action    string    // Action type: explore, query, compute, final, subrlm
	Code      string    // Code that was executed (if any)
	Output    string    // Execution output (truncated)
	Duration  time.Duration
	Success   bool
	Error     string
	SubRLM    *SubRLMEntry // Non-nil if this was a subrlm action
}

// SubRLMEntry captures sub-RLM invocation details.
type SubRLMEntry struct {
	Query      string        // The sub-RLM query
	Result     string        // The sub-RLM result
	Iterations int           // How many iterations the sub-RLM took
	Duration   time.Duration // How long the sub-RLM ran
}

// ImmutableHistory provides append-only history for RLM iterations.
type ImmutableHistory struct {
	entries []HistoryEntry
}

// NewImmutableHistory creates a new empty history.
func NewImmutableHistory() *ImmutableHistory {
	return &ImmutableHistory{
		entries: make([]HistoryEntry, 0),
	}
}

// Append adds a new entry to the history.
func (h *ImmutableHistory) Append(entry HistoryEntry) {
	h.entries = append(h.entries, entry)
}

// Entries returns all history entries (immutable copy).
func (h *ImmutableHistory) Entries() []HistoryEntry {
	result := make([]HistoryEntry, len(h.entries))
	copy(result, h.entries)
	return result
}

// Len returns the number of entries.
func (h *ImmutableHistory) Len() int {
	return len(h.entries)
}

// CountAction returns the number of entries with the provided action.
func (h *ImmutableHistory) CountAction(action string) int {
	if h == nil {
		return 0
	}
	action = strings.TrimSpace(strings.ToLower(action))
	if action == "" {
		return 0
	}
	count := 0
	for _, entry := range h.entries {
		if strings.ToLower(strings.TrimSpace(entry.Action)) == action {
			count++
		}
	}
	return count
}

// String converts history to a string for LLM prompts.
// Uses configurable truncation settings.
func (h *ImmutableHistory) String(maxEntryLen int) string {
	return h.render(maxEntryLen, h.entries)
}

// RenderCheckpointed summarizes older entries and keeps recent entries verbatim.
func (h *ImmutableHistory) RenderCheckpointed(maxEntryLen, verbatimEntries, maxSummaryTokens int) string {
	if h == nil || len(h.entries) == 0 {
		return ""
	}
	if verbatimEntries <= 0 || len(h.entries) <= verbatimEntries {
		return h.String(maxEntryLen)
	}

	checkpointEntries := h.entries[:len(h.entries)-verbatimEntries]
	verbatimEntriesSlice := h.entries[len(h.entries)-verbatimEntries:]

	var result strings.Builder
	result.WriteString(renderHistoryCheckpoint(checkpointEntries, maxEntryLen, maxSummaryTokens))
	result.WriteString(h.render(maxEntryLen, verbatimEntriesSlice))
	return result.String()
}

func (h *ImmutableHistory) render(maxEntryLen int, entries []HistoryEntry) string {
	if len(entries) == 0 {
		return ""
	}

	var result strings.Builder
	for _, entry := range entries {
		appendHistoryEntry(&result, entry, maxEntryLen)
	}
	return result.String()
}

func appendHistoryEntry(result *strings.Builder, entry HistoryEntry, maxEntryLen int) {
	if result == nil {
		return
	}
	fmt.Fprintf(result, "\n--- Iteration %d ---\n", entry.Iteration)
	fmt.Fprintf(result, "Action: %s\n", entry.Action)
	if entry.Code != "" {
		fmt.Fprintf(result, "Code:\n```go\n%s\n```\n", truncateHistoryText(entry.Code, maxEntryLen))
	}
	if entry.Output != "" {
		fmt.Fprintf(result, "Output:\n%s\n", truncateHistoryText(entry.Output, maxEntryLen))
	}
	if entry.Error != "" {
		fmt.Fprintf(result, "Error: %s\n", truncateHistoryText(entry.Error, maxEntryLen))
	}
	if entry.SubRLM != nil {
		fmt.Fprintf(result, "SubRLM Query: %s\n", truncateHistoryText(entry.SubRLM.Query, maxEntryLen))
		fmt.Fprintf(result, "SubRLM Result: %s\n", truncateHistoryText(entry.SubRLM.Result, maxEntryLen))
	}
}

func renderHistoryCheckpoint(entries []HistoryEntry, maxEntryLen, maxSummaryTokens int) string {
	if len(entries) == 0 {
		return ""
	}

	var result strings.Builder
	result.WriteString("[History checkpoint]\n")
	for _, entry := range entries {
		result.WriteString("- Iteration ")
		result.WriteString(fmt.Sprintf("%d: action=%s", entry.Iteration, entry.Action))
		if entry.SubRLM != nil {
			result.WriteString(", subquery=")
			result.WriteString(truncateHistoryText(entry.SubRLM.Query, maxEntryLen))
			if entry.SubRLM.Result != "" {
				result.WriteString(", result=")
				result.WriteString(truncateHistoryText(entry.SubRLM.Result, maxEntryLen))
			}
		}
		if entry.Error != "" {
			result.WriteString(", error=")
			result.WriteString(truncateHistoryText(entry.Error, maxEntryLen))
		} else if entry.Output != "" {
			result.WriteString(", output=")
			result.WriteString(truncateHistoryText(entry.Output, maxEntryLen))
		}
		result.WriteString("\n")
	}

	checkpoint := result.String()
	if maxSummaryTokens > 0 {
		maxChars := maxSummaryTokens * 4
		if len(checkpoint) > maxChars {
			checkpoint = checkpoint[:maxChars] + "\n[...truncated]\n"
		}
	}
	return checkpoint
}

func truncateHistoryText(text string, maxEntryLen int) string {
	if maxEntryLen > 0 && len(text) > maxEntryLen {
		return text[:maxEntryLen] + "..."
	}
	return text
}

// CompletionResult represents the final result of an RLM completion.
type CompletionResult struct {
	Response   string
	Iterations int
	Duration   time.Duration
	Usage      core.TokenUsage
}

// RLM is the main Recursive Language Model module implementation.
// It enables LLMs to explore large contexts programmatically through a Go REPL,
// making iterative queries to sub-LLMs until a final answer is reached.
type RLM struct {
	core.BaseModule
	config          Config
	rootLLM         core.LLM     // Root LLM for orchestration
	subLLMClient    SubLLMClient // Client for sub-LLM calls from REPL
	tokenTrackerMu  sync.RWMutex
	tokenTracker    *TokenTracker    // Snapshot of the most recently completed request
	iterationModule *modules.Predict // Internal Predict module for iterations
}

// Ensure RLM implements the InterceptableModule interface.
var _ core.InterceptableModule = (*RLM)(nil)

// NewFromLLM creates a new RLM module using a single core.LLM for both
// root orchestration and sub-queries. This is the recommended constructor
// for most use cases.
func NewFromLLM(llm core.LLM, opts ...Option) *RLM {
	return New(llm, NewLLMSubClient(llm), opts...)
}

// New creates a new RLM module instance with separate LLMs.
// rootLLM is used for the main orchestration loop.
// subLLMClient is used for Query/QueryBatched calls from within the REPL.
// For most cases, use NewFromLLM instead which uses the same LLM for both.
func New(rootLLM core.LLM, subLLMClient SubLLMClient, opts ...Option) *RLM {
	cfg := DefaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	cfg = cloneConfig(cfg)

	// Create the main RLM signature using the DSPy-native signature
	signature := RLMSignatureWithInstruction(resolveOuterInstruction(cfg))

	baseModule := core.NewModule(signature)
	baseModule.ModuleType = "RLM"

	iterMod := buildIterationModule(rootLLM, cfg)

	return &RLM{
		BaseModule:      *baseModule,
		config:          cfg,
		rootLLM:         rootLLM,
		subLLMClient:    subLLMClient,
		tokenTracker:    NewTokenTracker(),
		iterationModule: iterMod,
	}
}

// WithOptions applies additional options to the RLM module.
func (r *RLM) WithOptions(opts ...Option) *RLM {
	cfg := cloneConfig(r.config)
	for _, opt := range opts {
		opt(&cfg)
	}
	r.SetConfig(cfg)
	return r
}

// SetLLM sets the root LLM for orchestration and updates the iteration module.
func (r *RLM) SetLLM(llm core.LLM) {
	r.rootLLM = llm
	r.BaseModule.SetLLM(llm)
	r.rebuildIterationModule()
}

// Config returns a defensive copy of the current RLM configuration.
func (r *RLM) Config() Config {
	if r == nil {
		return DefaultConfig()
	}
	return cloneConfig(r.config)
}

// SetConfig applies a new RLM configuration and rebuilds prompt-bearing internals.
func (r *RLM) SetConfig(cfg Config) {
	if r == nil {
		return
	}
	r.config = cloneConfig(cfg)
	r.BaseModule.SetSignature(RLMSignatureWithInstruction(resolveOuterInstruction(r.config)))
	r.rebuildIterationModule()
}

func buildIterationModule(rootLLM core.LLM, cfg Config) *modules.Predict {
	signature := iterationSignatureWithInstruction(resolveIterationInstruction(cfg))

	iterMod := modules.NewPredict(signature).WithTextOutput()
	if cfg.UseIterationDemos {
		iterMod.SetDemos(IterationDemos())
	}
	iterMod.SetLLM(rootLLM)
	return iterMod
}

func (r *RLM) rebuildIterationModule() {
	if r.rootLLM == nil {
		r.iterationModule = nil
		return
	}
	r.iterationModule = buildIterationModule(r.rootLLM, r.config)
}

func resolveOuterInstruction(cfg Config) string {
	if cfg.OuterInstruction != "" {
		return cfg.OuterInstruction
	}
	return DefaultOuterInstruction()
}

func resolveIterationInstruction(cfg Config) string {
	if cfg.IterationInstruction != "" {
		return cfg.IterationInstruction
	}
	return DefaultIterationInstruction(cfg.CompactIterationInstructions)
}

// ProcessWithInterceptors executes the RLM module's logic with interceptor support.
func (r *RLM) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return r.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, r.Process, opts...)
}

// Process implements the core.Module interface.
// It takes inputs with "context" and "query" fields and returns the answer.
func (r *RLM) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, "RLM")
	defer core.EndSpan(ctx)
	span.WithAnnotation("inputs", inputs)

	contextPayload, ok := inputs["context"]
	if !ok {
		return nil, ErrMissingContext
	}

	query, ok := inputs["query"].(string)
	if !ok {
		return nil, ErrMissingQuery
	}

	logger.Debug(ctx, "Starting RLM completion with query: %s", query)

	// Run the completion
	result, _, err := r.CompleteWithTrace(ctx, contextPayload, query)
	if err != nil {
		span.WithError(err)
		return nil, err
	}

	// Update execution state with token usage
	if state := core.GetExecutionState(ctx); state != nil {
		state.WithTokenUsage(&core.TokenUsage{
			PromptTokens:     result.Usage.PromptTokens,
			CompletionTokens: result.Usage.CompletionTokens,
			TotalTokens:      result.Usage.TotalTokens,
		})
	}

	span.WithAnnotation("iterations", result.Iterations)
	span.WithAnnotation("duration", result.Duration.String())

	return map[string]any{
		"answer": result.Response,
	}, nil
}

// Complete runs an RLM completion.
// contextPayload is the context data (string, map, or slice).
// query is the user's question.
func (r *RLM) Complete(ctx context.Context, contextPayload any, query string) (*CompletionResult, error) {
	result, _, err := r.CompleteWithTrace(ctx, contextPayload, query)
	return result, err
}

// CompleteWithTrace runs an RLM completion and returns a structured trace of the execution.
func (r *RLM) CompleteWithTrace(ctx context.Context, contextPayload any, query string) (*CompletionResult, *RLMTrace, error) {
	logger := logging.GetLogger()
	start := time.Now()
	tokenTracker := NewTokenTracker()
	trace := &RLMTrace{
		Input: map[string]any{
			"context": contextPayload,
			"query":   query,
		},
		StartedAt:    start,
		tokenTracker: tokenTracker,
	}
	defer r.setTokenTracker(tokenTracker)

	ctx = core.WithExecutionState(ctx)

	// Apply timeout if configured
	if r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
	}

	// Initialize trace session if tracing is enabled
	var traceSession *logging.RLMTraceSession
	if r.config.TraceDir != "" {
		contextStr := ""
		if s, ok := contextPayload.(string); ok {
			contextStr = s
		} else {
			contextStr = fmt.Sprintf("%v", contextPayload)
		}

		var err error
		traceSession, err = logging.NewRLMTraceSession(r.config.TraceDir, logging.RLMSessionConfig{
			RootModel:     r.rootLLM.ModelID(),
			MaxIterations: r.config.MaxIterations,
			Backend:       r.rootLLM.ProviderName(),
			BackendKwargs: map[string]any{},
			Context:       contextStr,
			Query:         query,
		})
		if err != nil {
			logger.Warn(ctx, "[RLM] Failed to create trace session: %v", err)
		} else {
			defer traceSession.Close()
			logger.Info(ctx, "[RLM] Tracing to: %s", traceSession.Path())
		}
	}

	// Create REPL environment
	replEnv, err := NewYaegiREPL(r.subLLMClient)
	if err != nil {
		finalizeRLMTrace(trace, nil, "repl_create_error", fmt.Errorf("failed to create REPL: %w", err))
		return nil, trace, fmt.Errorf("failed to create REPL: %w", err)
	}
	replEnv.SetContextInfoPreviewChars(r.config.ContextInfoPreviewChars)
	replEnv.SetMaxFullContextQueryChars(r.config.MaxFullContextQueryChars)

	// Load context into REPL
	if err := replEnv.LoadContext(contextPayload); err != nil {
		loadErr := fmt.Errorf("failed to load context: %w", err)
		finalizeRLMTrace(trace, nil, "context_load_error", loadErr)
		return nil, trace, loadErr
	}

	// Allow callers to extend the REPL before the iteration loop starts.
	if r.config.REPLSetup != nil {
		if err := r.config.REPLSetup(replEnv); err != nil {
			setupErr := fmt.Errorf("REPL setup hook failed: %w", err)
			finalizeRLMTrace(trace, nil, "repl_setup_error", setupErr)
			return nil, trace, setupErr
		}
	}

	// Compute max iterations (adaptive or fixed)
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterations(contextSize)

	// Track confidence signals for early termination
	var confidenceSignals int

	// Build structured iteration history.
	history := NewImmutableHistory()

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		select {
		case <-ctx.Done():
			trace.CompletedAt = time.Now()
			trace.ProcessingTime = time.Since(start)
			trace.TerminationCause = "context_canceled"
			trace.Error = ctx.Err().Error()
			return nil, trace, ctx.Err()
		default:
		}
		if err := r.enforceTokenBudget(tokenTracker); err != nil {
			trace.CompletedAt = time.Now()
			trace.ProcessingTime = time.Since(start)
			trace.TerminationCause = "token_budget_exceeded"
			trace.Error = err.Error()
			return nil, trace, err
		}

		if r.config.Verbose {
			logger.Info(ctx, "[RLM] Iteration %d/%d", i+1, maxIterations)
		}

		// Prepare inputs for iteration module
		iterInputs := map[string]any{
			"context_info": replEnv.ContextInfo(),
			"query":        query,
			"history":      r.renderHistoryPrompt(history),
			"repl_state":   r.formatREPLStateInput(replEnv),
		}

		// Call the iteration module
		outputs, err := r.iterationModule.Process(ctx, iterInputs)
		if err != nil {
			processErr := fmt.Errorf("iteration %d: module process failed: %w", i, err)
			finalizeRLMTrace(trace, nil, "iteration_process_error", processErr)
			return nil, trace, processErr
		}

		// Extract structured outputs
		action := extractStringOutput(outputs, "action")
		code := extractStringOutput(outputs, "code")
		answer := extractStringOutput(outputs, "answer")
		reasoning := extractStringOutput(outputs, "reasoning")
		subquery := extractStringOutput(outputs, "subquery")

		// Strip language markers from code (handles LLM outputting "go\n" at start)
		code = stripCodeLanguageMarker(code)

		rootPromptTokens := r.trackTokenUsage(ctx, tokenTracker, i+1)
		if err := r.enforceTokenBudget(tokenTracker); err != nil {
			finalizeRLMTrace(trace, nil, "token_budget_exceeded", err)
			return nil, trace, err
		}

		// Report progress after the LLM call so per-iteration prompt tokens are available
		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: confidenceSignals,
				HasFinalAttempt:   action == "final",
				ContextSize:       contextSize,
				RootPromptTokens:  rootPromptTokens,
			})
		}

		if r.config.Verbose {
			logger.Debug(ctx, "[RLM] Action: %s, Reasoning: %s", action, utils.TruncateString(reasoning, truncateLenShort))
		}

		// Handle the action
		action = strings.ToLower(strings.TrimSpace(action))

		// Check for subrlm action (nested RLM loop)
		if action == "subrlm" {
			subRLMResult, err := r.executeSubRLM(ctx, replEnv, subquery, query, traceSession, start, tokenTracker, history)
			if err != nil {
				logger.Warn(ctx, "[RLM] Sub-RLM execution error: %v", err)
				r.appendHistoryEntry(history, HistoryEntry{
					Iteration: i + 1,
					Timestamp: iterStart,
					Action:    action,
					Output:    fmt.Sprintf("Sub-RLM error: %v", err),
					Duration:  time.Since(iterStart),
					Success:   false,
					Error:     err.Error(),
					SubRLM: &SubRLMEntry{
						Query:    subquery,
						Duration: time.Since(iterStart),
					},
				})
				trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, "", subquery, "", fmt.Sprintf("Sub-RLM error: %v", err), time.Since(iterStart), false, err))
			} else {
				// Store result in REPL variable for access in subsequent iterations
				_ = replEnv.SetVariable("subrlm_result", subRLMResult.Response)
				r.appendHistoryEntry(history, HistoryEntry{
					Iteration: i + 1,
					Timestamp: iterStart,
					Action:    action,
					Output:    fmt.Sprintf("Sub-RLM completed: %s", subRLMResult.Response),
					Duration:  time.Since(iterStart),
					Success:   true,
					SubRLM: &SubRLMEntry{
						Query:      subquery,
						Result:     subRLMResult.Response,
						Iterations: subRLMResult.Iterations,
						Duration:   subRLMResult.Duration,
					},
				})
				trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, "", subquery, fmt.Sprintf("Sub-RLM completed: %s", subRLMResult.Response), "", time.Since(iterStart), true, nil))

				if r.config.Verbose {
					logger.Debug(ctx, "[RLM] Sub-RLM result: %s", utils.TruncateString(subRLMResult.Response, truncateLenMedium))
				}
			}
			continue
		}

		// Check for final answer
		if action == "final" {
			// If action is final, use the answer field
			finalAnswer := answer
			if finalAnswer == "" {
				// Fallback 1: try to find FINAL() in the raw reasoning
				if final := FindFinalAnswer(reasoning); final != nil {
					if final.Type == FinalTypeVariable {
						resolved, err := replEnv.GetVariable(final.Content)
						if err == nil {
							finalAnswer = resolved
						} else {
							finalAnswer = fmt.Sprintf("Error: failed to resolve FINAL_VAR(%s): %v", final.Content, err)
						}
					} else {
						finalAnswer = final.Content
					}
				}
			}

			// Fallback 2: when action is explicitly "final", check common REPL variables
			if finalAnswer == "" && action == "final" {
				for _, varName := range []string{"result", "answer", "final_answer", "output", "total", "count"} {
					if val, err := replEnv.GetVariable(varName); err == nil && val != "" {
						finalAnswer = val
						break
					}
				}
			}

			if finalAnswer != "" {
				// Log final iteration to trace
				if traceSession != nil {
					_ = traceSession.LogIteration(
						[]logging.RLMMessage{{Role: "user", Content: query}},
						reasoning,
						[]logging.RLMCodeBlock{},
						finalAnswer,
						time.Since(iterStart),
					)
				}
				trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, code, subquery, finalAnswer, "", time.Since(iterStart), true, nil))

				result := &CompletionResult{
					Response:   finalAnswer,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}
				finalizeRLMTrace(trace, result, "final_answer", nil)
				return result, trace, nil
			}
		}

		if code != "" {
			if r.config.Verbose {
				logger.Debug(ctx, "[RLM] Executing code:\n%s", utils.TruncateString(code, truncateLenMedium))
			}

			// Clear final state before execution (Nightjar Algorithm 1 Gap 1)
			replEnv.ClearFinal()

			result, execErr := replEnv.Execute(ctx, code)
			if execErr != nil {
				logger.Warn(ctx, "[RLM] Code execution error: %v", execErr)
			}

			if r.config.Verbose && result.Stdout != "" {
				logger.Debug(ctx, "[RLM] Output: %s", utils.TruncateString(result.Stdout, truncateLenMedium))
			}
			if r.config.Verbose && result.Stderr != "" {
				logger.Debug(ctx, "[RLM] Stderr: %s", utils.TruncateString(result.Stderr, truncateLenMedium))
			}

			// Collect sub-LLM calls for trace and build compact history summaries.
			var rlmCalls []logging.RLMCallEntry
			fullExecOutput := FormatExecutionResult(result)
			llmCalls := replEnv.GetLLMCalls()
			for _, call := range llmCalls {
				tokenTracker.AddSubCall(call)
				rlmCalls = append(rlmCalls, logging.RLMCallEntry{
					Prompt:           call.Prompt,
					Response:         call.Response,
					PromptTokens:     call.PromptTokens,
					CompletionTokens: call.CompletionTokens,
					ExecutionTime:    call.Duration.Seconds(),
				})
				fullExecOutput += fmt.Sprintf("\n\n[LLM query]\n%s\n[LLM result]\n%s", call.Prompt, call.Response)
			}
			if err := r.enforceTokenBudget(tokenTracker); err != nil {
				trace.CompletedAt = time.Now()
				trace.ProcessingTime = time.Since(start)
				trace.TerminationCause = "token_budget_exceeded"
				trace.Error = err.Error()
				return nil, trace, err
			}

			callSummary := r.summarizeLLMCalls(llmCalls)

			// State-verified completion check (Nightjar Algorithm 1 Gap 1)
			// This is the PRIMARY completion detection - check REPL state first
			if replEnv.HasFinal() {
				resultResponse := replEnv.Final()

				if r.config.Verbose {
					logger.Debug(ctx, "[RLM] State-verified FINAL detected: %s", utils.TruncateString(resultResponse, truncateLenShort))
				}

				// Log final iteration to trace
				if traceSession != nil {
					_ = traceSession.LogIteration(
						[]logging.RLMMessage{{Role: "user", Content: query}},
						reasoning,
						[]logging.RLMCodeBlock{{
							Code: code,
							Result: logging.RLMCodeResult{
								Stdout:        result.Stdout,
								Stderr:        result.Stderr,
								Locals:        replEnv.GetLocals(),
								ExecutionTime: result.Duration.Seconds(),
								RLMCalls:      rlmCalls,
							},
						}},
						resultResponse,
						time.Since(iterStart),
					)
				}
				trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, code, subquery, resultResponse, "", time.Since(iterStart), execErr == nil, execErr))

				completion := &CompletionResult{
					Response:   resultResponse,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}
				finalizeRLMTrace(trace, completion, "state_final", nil)
				return completion, trace, nil
			}

			// Fallback: Check for FINAL in execution output via regex (deprecated, kept for backward compatibility)
			if final := FindFinalAnswer(fullExecOutput); final != nil {
				resultResponse := final.Content

				if r.config.Verbose {
					logger.Debug(ctx, "[RLM] Regex-based FINAL detected (fallback): %s", utils.TruncateString(resultResponse, truncateLenShort))
				}

				if traceSession != nil {
					_ = traceSession.LogIteration(
						[]logging.RLMMessage{{Role: "user", Content: query}},
						reasoning,
						[]logging.RLMCodeBlock{{
							Code: code,
							Result: logging.RLMCodeResult{
								Stdout:        result.Stdout,
								Stderr:        result.Stderr,
								Locals:        replEnv.GetLocals(),
								ExecutionTime: result.Duration.Seconds(),
								RLMCalls:      rlmCalls,
							},
						}},
						resultResponse,
						time.Since(iterStart),
					)
				}
				trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, code, subquery, resultResponse, "", time.Since(iterStart), execErr == nil, execErr))

				completion := &CompletionResult{
					Response:   resultResponse,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}
				finalizeRLMTrace(trace, completion, "regex_final", nil)
				return completion, trace, nil
			}

			// Log iteration to trace
			if traceSession != nil {
				codeBlocks := []logging.RLMCodeBlock{{
					Code: code,
					Result: logging.RLMCodeResult{
						Stdout:        result.Stdout,
						Stderr:        result.Stderr,
						Locals:        replEnv.GetLocals(),
						ExecutionTime: result.Duration.Seconds(),
						RLMCalls:      rlmCalls,
					},
				}}
				_ = traceSession.LogIteration(
					[]logging.RLMMessage{{Role: "user", Content: query}},
					reasoning,
					codeBlocks,
					nil,
					time.Since(iterStart),
				)
			}

			execOutput := r.formatExecutionOutput(result)
			if callSummary != "" {
				if execOutput == "No output" {
					execOutput = callSummary
				} else {
					execOutput = execOutput + "\n\n" + callSummary
				}
			}
			r.appendHistoryEntry(history, HistoryEntry{
				Iteration: i + 1,
				Timestamp: iterStart,
				Action:    action,
				Code:      code,
				Output:    execOutput,
				Duration:  time.Since(iterStart),
				Success:   execErr == nil,
				Error:     errString(execErr),
			})
			trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, code, subquery, execOutput, "", time.Since(iterStart), execErr == nil, execErr))
		} else if action != "final" {
			// Log non-code iteration to trace
			if traceSession != nil {
				_ = traceSession.LogIteration(
					[]logging.RLMMessage{{Role: "user", Content: query}},
					reasoning,
					[]logging.RLMCodeBlock{},
					nil,
					time.Since(iterStart),
				)
			}
			r.appendHistoryEntry(history, HistoryEntry{
				Iteration: i + 1,
				Timestamp: iterStart,
				Action:    action,
				Duration:  time.Since(iterStart),
				Success:   true,
			})
			trace.Steps = append(trace.Steps, newRLMTraceStep(i+1, reasoning, action, "", subquery, "", "", time.Since(iterStart), true, nil))
		}

		// Detect confidence signals for adaptive early termination
		if r.detectConfidence(reasoning) {
			confidenceSignals++
			trace.ConfidenceSignals = confidenceSignals
			if r.config.Verbose {
				logger.Debug(ctx, "[RLM] Confidence signal detected (total: %d)", confidenceSignals)
			}
		}

		// Check for early termination based on confidence signals
		hasCode := code != ""
		if r.shouldTerminateEarly(confidenceSignals, hasCode) {
			if r.config.Verbose {
				logger.Info(ctx, "[RLM] Early termination triggered (confidence signals: %d)", confidenceSignals)
			}
			result, err := r.forceDefaultAnswer(ctx, replEnv, query, r.renderHistoryPrompt(history), start, maxIterations, tokenTracker)
			finalizeRLMTrace(trace, result, "early_termination", err)
			return result, trace, err
		}
	}

	// Max iterations exhausted - force final answer
	result, err := r.forceDefaultAnswer(ctx, replEnv, query, r.renderHistoryPrompt(history), start, maxIterations, tokenTracker)
	finalizeRLMTrace(trace, result, "max_iterations", err)
	return result, trace, err
}

// executeSubRLM spawns a nested RLM loop that shares REPL state with the parent.
// It respects depth limits to prevent infinite recursion.
func (r *RLM) executeSubRLM(ctx context.Context, replEnv *YaegiREPL, subquery, parentQuery string, traceSession *logging.RLMTraceSession, parentStart time.Time, parentTracker *TokenTracker, history *ImmutableHistory) (*CompletionResult, error) {
	logger := logging.GetLogger()
	start := time.Now()

	// Determine current depth and max depth
	currentDepth := 0
	maxDepth := 3
	maxSubIterations := 10

	if r.config.SubRLM != nil {
		currentDepth = r.config.SubRLM.CurrentDepth
		maxDepth = r.config.SubRLM.MaxDepth
		if r.config.SubRLM.MaxIterationsPerSubRLM > 0 {
			maxSubIterations = r.config.SubRLM.MaxIterationsPerSubRLM
		}
	}

	if err := r.enforceSubRLMBudgets(history, parentTracker); err != nil {
		return nil, err
	}

	// Check depth limit
	if currentDepth+1 >= maxDepth {
		return nil, fmt.Errorf("sub-RLM depth limit reached (max: %d)", maxDepth)
	}

	if subquery == "" {
		return nil, fmt.Errorf("sub-RLM requires a subquery")
	}

	if r.config.Verbose {
		logger.Info(ctx, "[RLM] Spawning sub-RLM at depth %d with query: %s", currentDepth+1, utils.TruncateString(subquery, truncateLenShort))
	}

	// Create sub-RLM config with incremented depth
	subConfig := r.config
	subConfig.SubRLM = &SubRLMConfig{
		MaxDepth:               maxDepth,
		CurrentDepth:           currentDepth + 1,
		MaxIterationsPerSubRLM: maxSubIterations,
	}
	subConfig.MaxIterations = maxSubIterations

	// Create sub-RLM iteration module using the same prompt/demos policy as the parent config.
	subIterMod := buildIterationModule(r.rootLLM, subConfig)

	// Create sub-RLM instance
	subRLM := &RLM{
		BaseModule:      *core.NewModule(RLMSignatureWithInstruction(resolveOuterInstruction(subConfig))),
		config:          subConfig,
		rootLLM:         r.rootLLM,
		subLLMClient:    r.subLLMClient,
		tokenTracker:    NewTokenTracker(),
		iterationModule: subIterMod,
	}
	subRLM.ModuleType = "SubRLM"

	// Execute sub-RLM using the SHARED REPL environment
	subTracker := NewTokenTracker()
	result, err := r.completeWithSharedREPL(ctx, subRLM, replEnv, subquery, subTracker)
	if err != nil {
		return nil, fmt.Errorf("sub-RLM execution failed: %w", err)
	}
	subRLM.setTokenTracker(subTracker)

	// Track sub-RLM token usage
	subUsage := subTracker.GetTotalUsage()
	parentTracker.AddSubRLMCall(SubRLMCall{
		Query:            subquery,
		Result:           result.Response,
		Iterations:       result.Iterations,
		Depth:            currentDepth + 1,
		Duration:         time.Since(start),
		PromptTokens:     subUsage.PromptTokens,
		CompletionTokens: subUsage.CompletionTokens,
	})

	if r.config.Verbose {
		logger.Debug(ctx, "[RLM] Sub-RLM completed in %d iterations, %v", result.Iterations, time.Since(start))
	}

	return result, nil
}

// completeWithSharedREPL runs an RLM completion loop using an existing REPL environment.
// This allows sub-RLMs to share state with their parent.
func (r *RLM) completeWithSharedREPL(ctx context.Context, subRLM *RLM, replEnv *YaegiREPL, query string, tokenTracker *TokenTracker) (*CompletionResult, error) {
	logger := logging.GetLogger()
	start := time.Now()

	// Apply timeout if configured
	if subRLM.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, subRLM.config.Timeout)
		defer cancel()
	}

	// Get context info from shared REPL
	contextSize := 0
	if idx := replEnv.GetContextIndex(); idx != nil {
		contextSize = len(idx.GetRawContent())
	}
	maxIterations := subRLM.computeMaxIterations(contextSize)

	var confidenceSignals int
	history := NewImmutableHistory()

	// Sub-RLM iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		if err := subRLM.enforceTokenBudget(tokenTracker); err != nil {
			return nil, err
		}

		if subRLM.config.Verbose {
			logger.Info(ctx, "[SubRLM] Iteration %d/%d (depth %d)", i+1, maxIterations, subRLM.config.SubRLM.CurrentDepth)
		}

		// Prepare inputs for iteration module
		iterInputs := map[string]any{
			"context_info": replEnv.ContextInfo(),
			"query":        query,
			"history":      subRLM.renderHistoryPrompt(history),
			"repl_state":   subRLM.formatREPLStateInput(replEnv),
		}

		// Call the iteration module
		outputs, err := subRLM.iterationModule.Process(ctx, iterInputs)
		if err != nil {
			return nil, fmt.Errorf("sub-RLM iteration %d: module process failed: %w", i, err)
		}

		// Extract structured outputs
		action := extractStringOutput(outputs, "action")
		code := extractStringOutput(outputs, "code")
		answer := extractStringOutput(outputs, "answer")
		reasoning := extractStringOutput(outputs, "reasoning")
		subquery := extractStringOutput(outputs, "subquery")

		code = stripCodeLanguageMarker(code)
		subRLM.trackTokenUsage(ctx, tokenTracker, i+1)
		if err := subRLM.enforceTokenBudget(tokenTracker); err != nil {
			return nil, err
		}

		action = strings.ToLower(strings.TrimSpace(action))

		// Handle nested subrlm action
		if action == "subrlm" {
			nestedResult, err := subRLM.executeSubRLM(ctx, replEnv, subquery, query, nil, start, tokenTracker, history)
			if err != nil {
				logger.Warn(ctx, "[SubRLM] Nested sub-RLM error: %v", err)
				subRLM.appendHistoryEntry(history, HistoryEntry{
					Iteration: i + 1,
					Timestamp: iterStart,
					Action:    action,
					Output:    fmt.Sprintf("Nested sub-RLM error: %v", err),
					Duration:  time.Since(iterStart),
					Success:   false,
					Error:     err.Error(),
					SubRLM: &SubRLMEntry{
						Query:    subquery,
						Duration: time.Since(iterStart),
					},
				})
			} else {
				_ = replEnv.SetVariable("subrlm_result", nestedResult.Response)
				subRLM.appendHistoryEntry(history, HistoryEntry{
					Iteration: i + 1,
					Timestamp: iterStart,
					Action:    action,
					Output:    fmt.Sprintf("Nested sub-RLM completed: %s", nestedResult.Response),
					Duration:  time.Since(iterStart),
					Success:   true,
					SubRLM: &SubRLMEntry{
						Query:      subquery,
						Result:     nestedResult.Response,
						Iterations: nestedResult.Iterations,
						Duration:   nestedResult.Duration,
					},
				})
			}
			continue
		}

		// Check for final answer
		if action == "final" {
			finalAnswer := answer
			if finalAnswer == "" {
				if final := FindFinalAnswer(reasoning); final != nil {
					if final.Type == FinalTypeVariable {
						resolved, err := replEnv.GetVariable(final.Content)
						if err == nil {
							finalAnswer = resolved
						}
					} else {
						finalAnswer = final.Content
					}
				}
			}
			if finalAnswer == "" {
				for _, varName := range []string{"result", "answer", "final_answer", "output", "subrlm_result"} {
					if val, err := replEnv.GetVariable(varName); err == nil && val != "" {
						finalAnswer = val
						break
					}
				}
			}
			if finalAnswer != "" {
				return &CompletionResult{
					Response:   finalAnswer,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}, nil
			}
		}

		if code != "" {
			replEnv.ClearFinal()
			result, execErr := replEnv.Execute(ctx, code)

			// Collect sub-LLM calls
			fullExecOutput := FormatExecutionResult(result)
			llmCalls := replEnv.GetLLMCalls()
			for _, call := range llmCalls {
				tokenTracker.AddSubCall(call)
				fullExecOutput += fmt.Sprintf("\n\n[LLM query]\n%s\n[LLM result]\n%s", call.Prompt, call.Response)
			}
			if err := subRLM.enforceTokenBudget(tokenTracker); err != nil {
				return nil, err
			}

			callSummary := subRLM.summarizeLLMCalls(llmCalls)

			// Check for FINAL via state
			if replEnv.HasFinal() {
				return &CompletionResult{
					Response:   replEnv.Final(),
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}, nil
			}

			// Fallback regex check
			if final := FindFinalAnswer(fullExecOutput); final != nil {
				return &CompletionResult{
					Response:   final.Content,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      tokenTracker.GetTotalUsage(),
				}, nil
			}

			execOutput := subRLM.formatExecutionOutput(result)
			if callSummary != "" {
				if execOutput == "No output" {
					execOutput = callSummary
				} else {
					execOutput = execOutput + "\n\n" + callSummary
				}
			}
			subRLM.appendHistoryEntry(history, HistoryEntry{
				Iteration: i + 1,
				Timestamp: iterStart,
				Action:    action,
				Code:      code,
				Output:    execOutput,
				Duration:  time.Since(iterStart),
				Success:   execErr == nil,
				Error:     errString(execErr),
			})
		} else if action != "final" {
			subRLM.appendHistoryEntry(history, HistoryEntry{
				Iteration: i + 1,
				Timestamp: iterStart,
				Action:    action,
				Duration:  time.Since(iterStart),
				Success:   true,
			})
		}

		// Detect confidence signals
		if subRLM.detectConfidence(reasoning) {
			confidenceSignals++
		}
		if subRLM.shouldTerminateEarly(confidenceSignals, code != "") {
			break
		}
	}

	// Max iterations exhausted - force final answer
	return subRLM.forceDefaultAnswer(ctx, replEnv, query, subRLM.renderHistoryPrompt(history), start, maxIterations, tokenTracker)
}

// forceDefaultAnswer forces the LLM to provide a final answer when max iterations reached.
func (r *RLM) forceDefaultAnswer(ctx context.Context, replEnv REPLEnvironment, query string, history string, start time.Time, maxIterations int, tokenTracker *TokenTracker) (*CompletionResult, error) {
	if err := r.enforceTokenBudget(tokenTracker); err != nil {
		return nil, err
	}

	// Prepare inputs asking for a final answer
	iterInputs := map[string]any{
		"context_info": replEnv.ContextInfo(),
		"query":        query,
		"history":      history + "\n\nMAX ITERATIONS REACHED. You MUST provide a final answer now.",
		"repl_state":   r.formatREPLStateInput(replEnv),
	}

	outputs, err := r.iterationModule.Process(ctx, iterInputs)
	if err != nil {
		return nil, fmt.Errorf("default answer: module process failed: %w", err)
	}

	r.trackTokenUsage(ctx, tokenTracker, 0)
	if err := r.enforceTokenBudget(tokenTracker); err != nil {
		return nil, err
	}

	// Extract the answer
	answer := extractStringOutput(outputs, "answer")

	// Fallback 1: check REPL state for computed values before using reasoning
	if answer == "" {
		locals := replEnv.GetLocals()
		for _, key := range []string{"result", "answer", "final_answer", "output", "total", "count"} {
			if val, ok := locals[key]; ok && val != "" {
				answer = fmt.Sprintf("%v", val)
				break
			}
		}
	}

	// Fallback 2: use reasoning as last resort
	if answer == "" {
		answer = extractStringOutput(outputs, "reasoning")
	}

	return &CompletionResult{
		Response:   answer,
		Iterations: maxIterations,
		Duration:   time.Since(start),
		Usage:      tokenTracker.GetTotalUsage(),
	}, nil
}

// computeMaxIterations calculates the dynamic max iterations based on context size.
func (r *RLM) computeMaxIterations(contextSize int) int {
	if r.config.AdaptiveIteration == nil || !r.config.AdaptiveIteration.Enabled {
		return r.config.MaxIterations
	}

	cfg := r.config.AdaptiveIteration
	additionalIterations := contextSize / cfg.ContextScaleFactor
	computed := cfg.BaseIterations + additionalIterations

	if computed > cfg.MaxIterations {
		return cfg.MaxIterations
	}
	return computed
}

func (r *RLM) outputTruncationConfig() OutputTruncationConfig {
	defaults := DefaultOutputTruncationConfig()
	if r.config.OutputTruncation == nil {
		return defaults
	}
	if !r.config.OutputTruncation.Enabled {
		return OutputTruncationConfig{}
	}

	cfg := *r.config.OutputTruncation
	if cfg.MaxOutputLen <= 0 {
		cfg.MaxOutputLen = defaults.MaxOutputLen
	}
	if cfg.MaxVarPreviewLen <= 0 {
		cfg.MaxVarPreviewLen = defaults.MaxVarPreviewLen
	}
	if cfg.MaxHistoryEntryLen <= 0 {
		cfg.MaxHistoryEntryLen = defaults.MaxHistoryEntryLen
	}
	cfg.Enabled = true
	return cfg
}

func (r *RLM) formatREPLStateInput(replEnv REPLEnvironment) string {
	cfg := r.outputTruncationConfig()
	if !cfg.Enabled {
		return formatREPLStateRich(replEnv.GetVariableMetadata(), 0)
	}
	return formatREPLStateRich(replEnv.GetVariableMetadata(), cfg.MaxVarPreviewLen)
}

func (r *RLM) truncateHistoryEntryText(text string) string {
	cfg := r.outputTruncationConfig()
	if !cfg.Enabled || cfg.MaxHistoryEntryLen <= 0 {
		return text
	}
	return utils.TruncateString(text, cfg.MaxHistoryEntryLen)
}

func (r *RLM) formatExecutionOutput(result *ExecutionResult) string {
	output := FormatExecutionResult(result)
	cfg := r.outputTruncationConfig()
	if !cfg.Enabled || cfg.MaxOutputLen <= 0 {
		return output
	}
	return utils.TruncateString(output, cfg.MaxOutputLen)
}

func (r *RLM) llmCallPreviewLimit() int {
	cfg := r.outputTruncationConfig()
	if !cfg.Enabled {
		return 0
	}
	if cfg.MaxVarPreviewLen > 0 {
		return cfg.MaxVarPreviewLen * 2
	}
	return truncateLenMedium
}

func (r *RLM) summarizeLLMCalls(calls []LLMCall) string {
	if len(calls) == 0 {
		return ""
	}

	previewLen := r.llmCallPreviewLimit()
	var summary strings.Builder
	for i, call := range calls {
		prompt := call.Prompt
		response := call.Response
		if previewLen > 0 {
			prompt = utils.TruncateString(prompt, previewLen)
			response = utils.TruncateString(response, previewLen)
		}
		fmt.Fprintf(&summary, "[LLM query %d] prompt=%q response=%q tokens=%d/%d\n",
			i+1, prompt, response, call.PromptTokens, call.CompletionTokens)
	}
	return strings.TrimSuffix(summary.String(), "\n")
}

func (r *RLM) enforceTokenBudget(tokenTracker *TokenTracker) error {
	if r.config.MaxTokens <= 0 {
		return nil
	}

	if tokenTracker == nil {
		tokenTracker = r.GetTokenTracker()
	}
	if tokenTracker == nil {
		return nil
	}
	usage := tokenTracker.GetTotalUsage()
	if usage.TotalTokens <= r.config.MaxTokens {
		return nil
	}

	return fmt.Errorf("%w: used %d tokens (limit %d)", ErrTokenBudgetExceeded, usage.TotalTokens, r.config.MaxTokens)
}

// getContextSize returns the size of the context payload in bytes.
func getContextSize(payload any) int {
	switch v := payload.(type) {
	case string:
		return len(v)
	case []byte:
		return len(v)
	default:
		// For complex types, estimate based on string representation
		return len(fmt.Sprintf("%v", v))
	}
}

// detectConfidence checks if the response contains confidence signals.
// Uses the custom detector if configured, otherwise defaults to checking for FINAL markers.
func (r *RLM) detectConfidence(response string) bool {
	if r.config.AdaptiveIteration == nil || !r.config.AdaptiveIteration.Enabled {
		return false
	}

	// Use custom detector if provided
	if r.config.AdaptiveIteration.ConfidenceDetector != nil {
		return r.config.AdaptiveIteration.ConfidenceDetector(response)
	}

	// Default: check for FINAL markers which indicate the model is ready to provide an answer
	return FindFinalAnswer(response) != nil
}

// shouldTerminateEarly checks if we should terminate early based on confidence signals.
// Early termination only happens when:
// 1. Adaptive iteration is enabled with early termination
// 2. Confidence threshold is met
// 3. There are no pending code blocks (the model isn't waiting for execution results).
func (r *RLM) shouldTerminateEarly(confidenceSignals int, hasCode bool) bool {
	if r.config.AdaptiveIteration == nil || !r.config.AdaptiveIteration.Enabled {
		return false
	}
	if !r.config.AdaptiveIteration.EnableEarlyTermination {
		return false
	}
	if hasCode {
		return false // Don't terminate while code was just executed
	}
	return confidenceSignals >= r.config.AdaptiveIteration.ConfidenceThreshold
}

func (r *RLM) appendHistoryEntry(history *ImmutableHistory, entry HistoryEntry) {
	if history == nil {
		return
	}
	history.Append(entry)
}

func (r *RLM) renderHistoryPrompt(history *ImmutableHistory) string {
	if history == nil || history.Len() == 0 {
		return ""
	}

	maxEntryLen := 0
	cfg := r.outputTruncationConfig()
	if cfg.Enabled {
		maxEntryLen = cfg.MaxHistoryEntryLen
	}

	switch normalizeContextPolicyPreset(r.config.ContextPolicy) {
	case ContextPolicyFull:
		return history.String(maxEntryLen)
	case ContextPolicyCheckpointed:
		verbatimEntries, maxSummaryTokens := r.historyCheckpointSettings()
		return history.RenderCheckpointed(maxEntryLen, verbatimEntries, maxSummaryTokens)
	case ContextPolicyAdaptive:
		fallthrough
	default:
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			verbatimEntries, maxSummaryTokens := r.historyCheckpointSettings()
			return history.RenderCheckpointed(maxEntryLen, verbatimEntries, maxSummaryTokens)
		}
		if history.Len() >= r.adaptiveCheckpointThreshold() {
			verbatimEntries, maxSummaryTokens := r.historyCheckpointSettings()
			return history.RenderCheckpointed(maxEntryLen, verbatimEntries, maxSummaryTokens)
		}
		return history.String(maxEntryLen)
	}
}

func (r *RLM) historyCheckpointSettings() (int, int) {
	if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
		return r.config.HistoryCompression.VerbatimIterations, r.config.HistoryCompression.MaxSummaryTokens
	}
	return 3, 500
}

func (r *RLM) adaptiveCheckpointThreshold() int {
	if r == nil || r.config.AdaptiveCheckpointThreshold <= 0 {
		return DefaultConfig().AdaptiveCheckpointThreshold
	}
	return r.config.AdaptiveCheckpointThreshold
}

func normalizeContextPolicyPreset(preset ContextPolicyPreset) ContextPolicyPreset {
	switch preset {
	case ContextPolicyFull, ContextPolicyCheckpointed, ContextPolicyAdaptive:
		return preset
	default:
		return ContextPolicyAdaptive
	}
}

func (r *RLM) enforceSubRLMBudgets(history *ImmutableHistory, parentTracker *TokenTracker) error {
	if r == nil || r.config.SubRLM == nil {
		return nil
	}

	cfg := r.config.SubRLM
	if cfg.MaxDirectSubRLMCalls > 0 && history != nil && history.CountAction("subrlm") >= cfg.MaxDirectSubRLMCalls {
		return fmt.Errorf("sub-RLM direct child budget exceeded (limit %d)", cfg.MaxDirectSubRLMCalls)
	}
	if cfg.MaxTotalSubRLMCalls > 0 && parentTracker != nil && len(parentTracker.GetSubRLMCalls()) >= cfg.MaxTotalSubRLMCalls {
		return fmt.Errorf("sub-RLM total budget exceeded (limit %d)", cfg.MaxTotalSubRLMCalls)
	}
	return nil
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// Clone creates a copy of the RLM module.
func (r *RLM) Clone() core.Module {
	// Clone the iteration module
	var clonedIterMod *modules.Predict
	if r.iterationModule != nil {
		clonedIterMod = r.iterationModule.Clone().(*modules.Predict)
	}

	return &RLM{
		BaseModule:      *r.BaseModule.Clone().(*core.BaseModule),
		config:          cloneConfig(r.config),
		rootLLM:         r.rootLLM,
		subLLMClient:    r.subLLMClient,
		tokenTracker:    NewTokenTracker(),
		iterationModule: clonedIterMod,
	}
}

// GetTokenTracker returns the token tracker for inspecting usage.
func (r *RLM) GetTokenTracker() *TokenTracker {
	if r == nil {
		return nil
	}
	r.tokenTrackerMu.RLock()
	defer r.tokenTrackerMu.RUnlock()
	return r.tokenTracker
}

// extractStringOutput safely extracts a string value from outputs map.
func extractStringOutput(outputs map[string]any, key string) string {
	if v, ok := outputs[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// stripCodeLanguageMarker removes leading language identifiers that LLMs sometimes include
// in code output (e.g., "go\n" at the start when already in a code field).
func stripCodeLanguageMarker(code string) string {
	code = strings.TrimSpace(code)
	if code == "" {
		return code
	}

	// Common language markers that might appear at the start
	markers := []string{"go\n", "golang\n", "repl\n", "Go\n", "GO\n"}
	for _, marker := range markers {
		if strings.HasPrefix(code, marker) {
			return strings.TrimSpace(code[len(marker):])
		}
	}

	// Also handle case where first line is just "go" (no newline yet trimmed)
	lines := strings.SplitN(code, "\n", 2)
	if len(lines) >= 2 {
		firstLine := strings.TrimSpace(lines[0])
		if firstLine == "go" || firstLine == "golang" || firstLine == "repl" ||
			firstLine == "Go" || firstLine == "GO" {
			return strings.TrimSpace(lines[1])
		}
	}
	return code
}

// formatREPLStateRich converts variable metadata to a rich, informative string.
// This provides type info, length, and preview for better LLM context.
func formatREPLStateRich(variables []REPLVariable, maxPreviewLen int) string {
	if len(variables) == 0 {
		return "No variables defined"
	}

	var result strings.Builder
	result.WriteString("Variables:\n")

	for _, v := range variables {
		// Mark important variables
		marker := ""
		if v.IsImportant {
			marker = " *"
		}

		// Format based on type
		if v.Length >= 0 {
			fmt.Fprintf(&result, "  %s%s: %s (len=%d)\n", v.Name, marker, v.Type, v.Length)
		} else {
			fmt.Fprintf(&result, "  %s%s: %s\n", v.Name, marker, v.Type)
		}

		// Show preview for non-trivial values
		preview := v.Preview
		if maxPreviewLen > 0 && len(preview) > maxPreviewLen {
			preview = preview[:maxPreviewLen] + "..."
		}
		if preview != "" && v.Name != "context" { // Skip context preview (too large)
			fmt.Fprintf(&result, "    → %s\n", preview)
		}
	}

	return result.String()
}

func newRLMTraceStep(index int, thought, action, code, subquery, observation, errorText string, duration time.Duration, success bool, err error) RLMTraceStep {
	step := RLMTraceStep{
		Index:       index,
		Thought:     thought,
		Action:      action,
		Code:        code,
		SubQuery:    subquery,
		Observation: observation,
		Duration:    duration,
		Success:     success,
		Error:       errorText,
	}
	if err != nil && step.Error == "" {
		step.Error = err.Error()
	}
	return step
}

func finalizeRLMTrace(trace *RLMTrace, result *CompletionResult, terminationCause string, err error) {
	if trace == nil {
		return
	}

	trace.CompletedAt = time.Now()
	trace.ProcessingTime = trace.CompletedAt.Sub(trace.StartedAt)
	trace.TerminationCause = terminationCause

	if result != nil {
		trace.Output = map[string]any{
			"answer": result.Response,
		}
		trace.Iterations = result.Iterations
		trace.Usage = result.Usage
	}
	if tokenTracker := trace.tokenTracker; tokenTracker != nil {
		trace.RootUsage = tokenTracker.GetRootUsage()
		trace.SubUsage = tokenTracker.GetSubUsage()
		trace.SubRLMUsage = tokenTracker.GetSubRLMUsage()
		trace.RootSnapshots = tokenTracker.GetRootSnapshots()
		trace.SubLLMCallCount = len(tokenTracker.GetSubCalls())
		trace.SubRLMCallCount = len(tokenTracker.GetSubRLMCalls())
	}

	if err != nil {
		trace.Error = err.Error()
	}
}

// trackTokenUsage extracts token usage from the execution state and records it.
// When iteration > 0, a per-iteration snapshot is recorded for context fill ratio analysis.
// Returns the per-call prompt tokens reported by the provider (0 if unavailable).
func (r *RLM) trackTokenUsage(ctx context.Context, tokenTracker *TokenTracker, iteration int) int {
	if tokenTracker == nil {
		tokenTracker = r.GetTokenTracker()
	}
	if tokenTracker == nil {
		return 0
	}
	if state := core.GetExecutionState(ctx); state != nil {
		if usage := state.GetTokenUsage(); usage != nil {
			if iteration > 0 {
				tokenTracker.AddRootUsageForIteration(iteration, usage.PromptTokens, usage.CompletionTokens)
			} else {
				tokenTracker.AddRootUsage(usage.PromptTokens, usage.CompletionTokens)
			}
			return usage.PromptTokens
		}
	}
	return 0
}

func (r *RLM) setTokenTracker(tokenTracker *TokenTracker) {
	if r == nil {
		return
	}
	r.tokenTrackerMu.Lock()
	defer r.tokenTrackerMu.Unlock()
	r.tokenTracker = tokenTracker
}

// appendIterationHistory adds iteration info to the history string.
// Nightjar Algorithm 1 Gap 2: Reasoning is NOT included to save tokens.
// History contains only: action, code, and truncated output.
func (r *RLM) appendIterationHistory(history *strings.Builder, iteration int, action, _, code, output string) {
	fmt.Fprintf(history, "\n--- Iteration %d ---\n", iteration)
	fmt.Fprintf(history, "Action: %s\n", action)
	if code != "" {
		fmt.Fprintf(history, "Code:\n```go\n%s\n```\n", r.truncateHistoryEntryText(code))
	}
	if output != "" {
		fmt.Fprintf(history, "Output:\n%s\n", r.truncateHistoryEntryText(output))
	}
}

// ContextMetadata returns a string describing the context.
func ContextMetadata(payload any) string {
	switch v := payload.(type) {
	case string:
		return fmt.Sprintf("string, %d chars", len(v))
	case []any:
		return fmt.Sprintf("array, %d items", len(v))
	case map[string]any:
		return fmt.Sprintf("object, %d keys", len(v))
	default:
		return fmt.Sprintf("%T", v)
	}
}
