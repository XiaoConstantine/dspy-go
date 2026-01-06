package rlm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
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
	rootLLM         core.LLM          // Root LLM for orchestration
	subLLMClient    SubLLMClient      // Client for sub-LLM calls from REPL
	tokenTracker    *TokenTracker     // Tracks token usage across all calls
	iterationModule *modules.Predict  // Internal Predict module for iterations
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

	// Extract context and query from inputs
	contextPayload, ok := inputs["context"]
	if !ok {
		return nil, fmt.Errorf("missing required input: context")
	}

	query, ok := inputs["query"].(string)
	if !ok {
		return nil, fmt.Errorf("missing or invalid required input: query")
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

	// Create REPL environment
	replEnv, err := NewYaegiREPL(r.subLLMClient)
	if err != nil {
		return nil, fmt.Errorf("failed to create REPL: %w", err)
	}

	// Load context into REPL
	if err := replEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}

	// Build iteration history
	var history strings.Builder

	// Iteration loop
	for i := 0; i < r.config.MaxIterations; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if r.config.Verbose {
			logger.Info(ctx, "[RLM] Iteration %d/%d", i+1, r.config.MaxIterations)
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

		// Track token usage from iteration module
		if state := core.GetExecutionState(ctx); state != nil {
			usage := state.GetTokenUsage()
			if usage != nil {
				r.tokenTracker.AddRootUsage(usage.PromptTokens, usage.CompletionTokens)
			}
		}

		if r.config.Verbose {
			logger.Debug(ctx, "[RLM] Action: %s, Reasoning: %s", action, truncate(reasoning, 100))
		}

		// Handle the action
		action = strings.ToLower(strings.TrimSpace(action))

		// Check for final answer
		if action == "final" || answer != "" {
			// If action is final, use the answer field
			finalAnswer := answer
			if finalAnswer == "" {
				// Fallback: try to find FINAL() in the raw reasoning
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

			if finalAnswer != "" {
				return &CompletionResult{
					Response:   finalAnswer,
					Iterations: i + 1,
					Duration:   time.Since(start),
					Usage:      r.tokenTracker.GetTotalUsage(),
				}, nil
			}
		}

		// Execute code if provided (for explore, query, compute actions)
		if code != "" {
			if r.config.Verbose {
				logger.Debug(ctx, "[RLM] Executing code:\n%s", truncate(code, 200))
			}

			result, _ := replEnv.Execute(ctx, code)

			if r.config.Verbose && result.Stdout != "" {
				logger.Debug(ctx, "[RLM] Output: %s", truncate(result.Stdout, 200))
			}
			if r.config.Verbose && result.Stderr != "" {
				logger.Debug(ctx, "[RLM] Stderr: %s", truncate(result.Stderr, 200))
			}

			// Get LLM calls made during code execution and track tokens
			for _, call := range replEnv.GetLLMCalls() {
				r.tokenTracker.AddSubCall(call)
			}

			// Append to history
			history.WriteString(fmt.Sprintf("\n--- Iteration %d ---\n", i+1))
			history.WriteString(fmt.Sprintf("Action: %s\n", action))
			history.WriteString(fmt.Sprintf("Reasoning: %s\n", reasoning))
			history.WriteString(fmt.Sprintf("Code:\n```go\n%s\n```\n", code))
			history.WriteString(fmt.Sprintf("Output:\n%s\n", truncateString(FormatExecutionResult(result), 5000)))
		} else if action != "final" {
			// No code and not final - still record the iteration
			history.WriteString(fmt.Sprintf("\n--- Iteration %d ---\n", i+1))
			history.WriteString(fmt.Sprintf("Action: %s\n", action))
			history.WriteString(fmt.Sprintf("Reasoning: %s\n", reasoning))
		}
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, replEnv, query, history.String(), start)
}

// forceDefaultAnswer forces the LLM to provide a final answer when max iterations reached.
func (r *RLM) forceDefaultAnswer(ctx context.Context, replEnv REPLEnvironment, query string, history string, start time.Time) (*CompletionResult, error) {
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

	// Track token usage
	if state := core.GetExecutionState(ctx); state != nil {
		usage := state.GetTokenUsage()
		if usage != nil {
			r.tokenTracker.AddRootUsage(usage.PromptTokens, usage.CompletionTokens)
		}
	}

	// Extract the answer
	answer := extractStringOutput(outputs, "answer")
	if answer == "" {
		// Fallback to reasoning if no answer field
		answer = extractStringOutput(outputs, "reasoning")
	}

	return &CompletionResult{
		Response:   answer,
		Iterations: r.config.MaxIterations,
		Duration:   time.Since(start),
		Usage:      r.tokenTracker.GetTotalUsage(),
	}, nil
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

// truncate shortens a string for logging.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// truncateString shortens a string with a suffix indicator.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
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
