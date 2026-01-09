package rlm

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// iterationPattern matches "--- Iteration N ---" headers in history.
var iterationPattern = regexp.MustCompile(`(?m)^--- Iteration (\d+) ---$`)

const (
	truncateLenShort  = 100
	truncateLenMedium = 200
	truncateLenLong   = 5000
)

var (
	ErrMissingContext = errors.New("missing required input: context")
	ErrMissingQuery   = errors.New("missing or invalid required input: query")
)

// CodeBlock represents an extracted and executed code block.
type CodeBlock struct {
	Code   string
	Result ExecutionResult
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
	rootLLM         core.LLM         // Root LLM for orchestration
	subLLMClient    SubLLMClient     // Client for sub-LLM calls from REPL
	tokenTracker    *TokenTracker    // Tracks token usage across all calls
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

	// Create the main RLM signature using the DSPy-native signature
	signature := RLMSignature()

	baseModule := core.NewModule(signature)
	baseModule.ModuleType = "RLM"

	// Create iteration module with signature and demos
	iterMod := modules.NewPredict(IterationSignature()).WithTextOutput()
	iterMod.SetDemos(IterationDemos())
	iterMod.SetLLM(rootLLM)

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
	for _, opt := range opts {
		opt(&r.config)
	}
	return r
}

// SetLLM sets the root LLM for orchestration and updates the iteration module.
func (r *RLM) SetLLM(llm core.LLM) {
	r.rootLLM = llm
	r.BaseModule.SetLLM(llm)
	if r.iterationModule != nil {
		r.iterationModule.SetLLM(llm)
	}
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
	result, err := r.Complete(ctx, contextPayload, query)
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
	logger := logging.GetLogger()
	start := time.Now()

	// Reset token tracker for this completion
	r.tokenTracker.Reset()

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
		return nil, fmt.Errorf("failed to create REPL: %w", err)
	}

	// Load context into REPL
	if err := replEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}

	// Compute max iterations (adaptive or fixed)
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterations(contextSize)

	// Track confidence signals for early termination
	var confidenceSignals int

	// Build iteration history
	var history strings.Builder

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Report progress if handler is set
		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
			})
		}

		if r.config.Verbose {
			logger.Info(ctx, "[RLM] Iteration %d/%d", i+1, maxIterations)
		}

		// Prepare inputs for iteration module
		iterInputs := map[string]any{
			"context_info": replEnv.ContextInfo(),
			"query":        query,
			"history":      history.String(),
			"repl_state":   formatREPLState(replEnv.GetLocals()),
		}

		// Call the iteration module
		outputs, err := r.iterationModule.Process(ctx, iterInputs)
		if err != nil {
			return nil, fmt.Errorf("iteration %d: module process failed: %w", i, err)
		}

		// Extract structured outputs
		action := extractStringOutput(outputs, "action")
		code := extractStringOutput(outputs, "code")
		answer := extractStringOutput(outputs, "answer")
		reasoning := extractStringOutput(outputs, "reasoning")

		r.trackTokenUsage(ctx)

		if r.config.Verbose {
			logger.Debug(ctx, "[RLM] Action: %s, Reasoning: %s", action, truncate(reasoning, truncateLenShort))
		}

		// Handle the action
		action = strings.ToLower(strings.TrimSpace(action))

		// Check for final answer
		if action == "final" || answer != "" {
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

				return &CompletionResult{
					Response:   finalAnswer,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      r.tokenTracker.GetTotalUsage(),
				}, nil
			}
		}

		if code != "" {
			if r.config.Verbose {
				logger.Debug(ctx, "[RLM] Executing code:\n%s", truncate(code, truncateLenMedium))
			}

			result, execErr := replEnv.Execute(ctx, code)
			if execErr != nil {
				logger.Warn(ctx, "[RLM] Code execution error: %v", execErr)
			}

			if r.config.Verbose && result.Stdout != "" {
				logger.Debug(ctx, "[RLM] Output: %s", truncate(result.Stdout, truncateLenMedium))
			}
			if r.config.Verbose && result.Stderr != "" {
				logger.Debug(ctx, "[RLM] Stderr: %s", truncate(result.Stderr, truncateLenMedium))
			}

			// Collect sub-LLM calls for trace and add Query results to history
			var rlmCalls []logging.RLMCallEntry
			var allOutput strings.Builder
			llmCalls := replEnv.GetLLMCalls()
			for _, call := range llmCalls {
				r.tokenTracker.AddSubCall(call)
				rlmCalls = append(rlmCalls, logging.RLMCallEntry{
					Prompt:           call.Prompt,
					Response:         call.Response,
					PromptTokens:     call.PromptTokens,
					CompletionTokens: call.CompletionTokens,
					ExecutionTime:    call.Duration.Seconds(),
				})

				// CRITICAL FIX: Add Query results to output so the LLM can see them in subsequent iterations.
				// Without this, Query() return values assigned to variables are invisible in history,
				// causing the LLM to guess/hallucinate the answer in later iterations.
				allOutput.WriteString(fmt.Sprintf("\n[Query] %s\n[Result] %s\n",
					truncate(call.Prompt, truncateLenMedium),
					call.Response))
			}

			// Also capture execution stdout in allOutput for FINAL detection
			if result.Stdout != "" {
				allOutput.WriteString(result.Stdout)
			}

			// Check for FINAL in execution output (from code-called FINAL/FINAL_VAR functions)
			outputStr := allOutput.String()
			if final := FindFinalAnswer(outputStr); final != nil {
				// Found FINAL in execution output - the value is already resolved
				resultResponse := final.Content

				if r.config.Verbose {
					logger.Debug(ctx, "[RLM] Found FINAL in execution output: %s", truncate(resultResponse, truncateLenShort))
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

				return &CompletionResult{
					Response:   resultResponse,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      r.tokenTracker.GetTotalUsage(),
				}, nil
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

			// Include Query results in history output
			execOutput := FormatExecutionResult(result)
			if allOutput.Len() > 0 {
				execOutput = execOutput + allOutput.String()
			}
			r.appendIterationHistory(&history, i+1, action, reasoning, code, execOutput)
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
			r.appendIterationHistory(&history, i+1, action, reasoning, "", "")
		}

		// Detect confidence signals for adaptive early termination
		if r.detectConfidence(reasoning) {
			confidenceSignals++
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
			return r.forceDefaultAnswer(ctx, replEnv, query, history.String(), start, maxIterations)
		}

		// Apply history compression if enabled
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			historyStr := history.String()
			compressedHistory := r.compressHistory(historyStr, i+1)
			if compressedHistory != historyStr {
				history.Reset()
				history.WriteString(compressedHistory)
				if r.config.Verbose {
					logger.Debug(ctx, "[RLM] History compressed: %d -> %d chars", len(historyStr), len(compressedHistory))
				}
			}
		}
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, replEnv, query, history.String(), start, maxIterations)
}

// forceDefaultAnswer forces the LLM to provide a final answer when max iterations reached.
func (r *RLM) forceDefaultAnswer(ctx context.Context, replEnv REPLEnvironment, query string, history string, start time.Time, maxIterations int) (*CompletionResult, error) {
	// Prepare inputs asking for a final answer
	iterInputs := map[string]any{
		"context_info": replEnv.ContextInfo(),
		"query":        query,
		"history":      history + "\n\nMAX ITERATIONS REACHED. You MUST provide a final answer now.",
		"repl_state":   formatREPLState(replEnv.GetLocals()),
	}

	outputs, err := r.iterationModule.Process(ctx, iterInputs)
	if err != nil {
		return nil, fmt.Errorf("default answer: module process failed: %w", err)
	}

	r.trackTokenUsage(ctx)

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
		Usage:      r.tokenTracker.GetTotalUsage(),
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

// compressHistory compresses older iterations in the history string.
// It keeps the most recent N iterations verbatim and summarizes older ones.
func (r *RLM) compressHistory(history string, currentIteration int) string {
	cfg := r.config.HistoryCompression
	if cfg == nil || !cfg.Enabled {
		return history
	}

	// Find all iteration boundaries
	matches := iterationPattern.FindAllStringSubmatchIndex(history, -1)
	if len(matches) <= cfg.VerbatimIterations {
		return history // Nothing to compress
	}

	// Calculate the split point
	splitIndex := len(matches) - cfg.VerbatimIterations
	if splitIndex <= 0 {
		return history
	}

	// Get the position where verbatim content starts
	verbatimStartPos := matches[splitIndex][0]

	// Content to summarize
	toSummarize := history[:verbatimStartPos]

	// Content to keep verbatim
	verbatim := history[verbatimStartPos:]

	// Create summary
	summary := r.summarizeIterations(toSummarize, cfg.MaxSummaryTokens)

	return summary + verbatim
}

// summarizeIterations creates a concise summary of older iteration history.
func (r *RLM) summarizeIterations(history string, maxTokens int) string {
	var summary strings.Builder
	summary.WriteString("[Previous iterations summary]\n")

	// Find iterations in the history to summarize
	matches := iterationPattern.FindAllStringSubmatchIndex(history, -1)

	for i, match := range matches {
		iterNum := "?"
		if len(match) >= 4 {
			iterNum = history[match[2]:match[3]]
		}

		// Find the content of this iteration (until next iteration or end)
		startPos := match[1] // End of the header line
		var endPos int
		if i+1 < len(matches) {
			endPos = matches[i+1][0]
		} else {
			endPos = len(history)
		}

		iterContent := history[startPos:endPos]

		// Extract key information
		hasCode := strings.Contains(iterContent, "```go")
		hasFinal := strings.Contains(iterContent, "FINAL")
		hasError := strings.Contains(iterContent, "Error:") || strings.Contains(iterContent, "error")

		summary.WriteString(fmt.Sprintf("- Iteration %s: ", iterNum))
		var parts []string
		if hasCode {
			parts = append(parts, "executed code")
		}
		if hasError {
			parts = append(parts, "had errors")
		}
		if hasFinal {
			parts = append(parts, "mentioned FINAL")
		}
		if len(parts) == 0 {
			parts = append(parts, "reasoning only")
		}
		summary.WriteString(strings.Join(parts, ", "))
		summary.WriteString("\n")
	}

	result := summary.String()

	// Rough token estimation: ~4 chars per token
	maxChars := maxTokens * 4
	if len(result) > maxChars {
		result = result[:maxChars] + "\n[...truncated]"
	}

	return result
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
		config:          r.config,
		rootLLM:         r.rootLLM,
		subLLMClient:    r.subLLMClient,
		tokenTracker:    NewTokenTracker(),
		iterationModule: clonedIterMod,
	}
}

// GetTokenTracker returns the token tracker for inspecting usage.
func (r *RLM) GetTokenTracker() *TokenTracker {
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

// formatREPLState converts the REPL locals map to a readable string.
func formatREPLState(locals map[string]any) string {
	if len(locals) == 0 {
		return "No variables defined"
	}

	var parts []string
	for name, value := range locals {
		valueStr := fmt.Sprintf("%v", value)
		if len(valueStr) > 100 {
			valueStr = valueStr[:100] + "..."
		}
		parts = append(parts, fmt.Sprintf("%s: %s", name, valueStr))
	}
	return strings.Join(parts, "\n")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func (r *RLM) trackTokenUsage(ctx context.Context) {
	if state := core.GetExecutionState(ctx); state != nil {
		if usage := state.GetTokenUsage(); usage != nil {
			r.tokenTracker.AddRootUsage(usage.PromptTokens, usage.CompletionTokens)
		}
	}
}

func (r *RLM) appendIterationHistory(history *strings.Builder, iteration int, action, reasoning, code, output string) {
	fmt.Fprintf(history, "\n--- Iteration %d ---\n", iteration)
	fmt.Fprintf(history, "Action: %s\n", action)
	fmt.Fprintf(history, "Reasoning: %s\n", reasoning)
	if code != "" {
		fmt.Fprintf(history, "Code:\n```go\n%s\n```\n", code)
		fmt.Fprintf(history, "Output:\n%s\n", truncate(output, truncateLenLong))
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
