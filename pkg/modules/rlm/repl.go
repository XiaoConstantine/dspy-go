package rlm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// REPLEnvironment defines the interface for a REPL that can execute code
// and make LLM queries.
type REPLEnvironment interface {
	// LoadContext loads the context payload into the REPL environment.
	LoadContext(payload any) error

	// Execute runs Go code in the interpreter and returns the result.
	Execute(ctx context.Context, code string) (*ExecutionResult, error)

	// GetVariable retrieves a variable value from the interpreter.
	GetVariable(name string) (string, error)

	// Reset clears the interpreter state.
	Reset() error

	// ContextInfo returns metadata about the loaded context.
	ContextInfo() string
}

// ExecutionResult represents the result of executing code in the REPL.
type ExecutionResult struct {
	Stdout   string
	Stderr   string
	Duration time.Duration
}

// QueryResponse contains the LLM response with usage metadata.
type QueryResponse struct {
	Response         string
	PromptTokens     int
	CompletionTokens int
}

// SubLLMClient defines the interface for making LLM calls from within the REPL.
type SubLLMClient interface {
	// Query makes a single LLM query.
	Query(ctx context.Context, prompt string) (QueryResponse, error)
	// QueryBatched makes concurrent LLM queries.
	QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error)
}

// LLMSubClient adapts a core.LLM to the SubLLMClient interface.
// This allows any dspy-go LLM to be used for sub-queries in RLM.
type LLMSubClient struct {
	llm core.LLM
}

// NewLLMSubClient creates a SubLLMClient from a core.LLM.
func NewLLMSubClient(llm core.LLM) *LLMSubClient {
	return &LLMSubClient{llm: llm}
}

// Query implements SubLLMClient.
func (c *LLMSubClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	resp, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		return QueryResponse{}, err
	}

	var promptTokens, completionTokens int
	if resp.Usage != nil {
		promptTokens = resp.Usage.PromptTokens
		completionTokens = resp.Usage.CompletionTokens
	}

	return QueryResponse{
		Response:         resp.Content,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, nil
}

// QueryBatched implements SubLLMClient with concurrent queries.
func (c *LLMSubClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	results := make([]QueryResponse, len(prompts))
	var wg sync.WaitGroup

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := c.Query(ctx, p)
			if err != nil {
				results[idx] = QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
			} else {
				results[idx] = result
			}
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}

// YaegiREPL represents a Yaegi-based Go interpreter with RLM capabilities.
type YaegiREPL struct {
	interp    *interp.Interpreter
	stdout    *bytes.Buffer
	stderr    *bytes.Buffer
	llmClient SubLLMClient
	ctx       context.Context
	mu        sync.Mutex
	llmCalls  []LLMCall // Track LLM calls made during execution
}

// NewYaegiREPL creates a new YaegiREPL instance.
func NewYaegiREPL(client SubLLMClient) *YaegiREPL {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)

	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})

	// Load standard library
	if err := i.Use(stdlib.Symbols); err != nil {
		panic(fmt.Sprintf("failed to load stdlib: %v", err))
	}

	r := &YaegiREPL{
		interp:    i,
		stdout:    stdout,
		stderr:    stderr,
		llmClient: client,
		ctx:       context.Background(),
	}

	// Inject RLM functions
	if err := r.injectBuiltins(); err != nil {
		panic(fmt.Sprintf("failed to inject builtins: %v", err))
	}

	return r
}

// injectBuiltins registers Query and QueryBatched functions in the interpreter.
func (r *YaegiREPL) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"Query":        reflect.ValueOf(r.llmQuery),
			"QueryBatched": reflect.ValueOf(r.llmQueryBatched),
		},
	}

	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Pre-import common packages and RLM functions so they're available without qualification
	setupCode := `
import "fmt"
import "strings"
import "regexp"
import "strconv"
import . "rlm/rlm"
`
	_, err := r.interp.Eval(setupCode)
	return err
}

// llmQuery makes a single LLM query. This is called from interpreted code.
func (r *YaegiREPL) llmQuery(prompt string) string {
	start := time.Now()
	result, err := r.llmClient.Query(r.ctx, prompt)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record the call with token usage
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})

	return response
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
func (r *YaegiREPL) llmQueryBatched(prompts []string) []string {
	start := time.Now()
	results, err := r.llmClient.QueryBatched(r.ctx, prompts)
	duration := time.Since(start)

	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		// Record each as a failed call
		avgDuration := duration / time.Duration(len(prompts))
		for i, p := range prompts {
			r.llmCalls = append(r.llmCalls, LLMCall{
				Prompt:   p,
				Response: errResults[i],
				Duration: avgDuration,
			})
		}
		return errResults
	}

	// Record each successful call with token usage
	responses := make([]string, len(results))
	avgDuration := duration / time.Duration(len(prompts))
	for i, p := range prompts {
		responses[i] = results[i].Response
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           p,
			Response:         results[i].Response,
			Duration:         avgDuration,
			PromptTokens:     results[i].PromptTokens,
			CompletionTokens: results[i].CompletionTokens,
		})
	}
	return responses
}

// LoadContext injects the context payload into the interpreter as the `context` variable.
func (r *YaegiREPL) LoadContext(payload any) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch v := payload.(type) {
	case string:
		// String context - inject directly
		_, err := r.interp.Eval(`var context = ` + strconv.Quote(v))
		return err

	case map[string]any:
		// Map context - serialize to JSON, then unmarshal in REPL
		return r.loadStructuredContext(v, "map[string]interface{}")

	case []any:
		// Slice context - serialize to JSON, then unmarshal in REPL
		return r.loadStructuredContext(v, "[]interface{}")

	default:
		// Try JSON marshaling as fallback
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		return r.LoadContext(string(jsonBytes))
	}
}

// loadStructuredContext handles map and slice context types.
func (r *YaegiREPL) loadStructuredContext(v any, typeDecl string) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}

	code := fmt.Sprintf(`
import "encoding/json"
var context %s
func init() {
	json.Unmarshal([]byte(%s), &context)
}
`, typeDecl, strconv.Quote(string(jsonBytes)))

	_, err = r.interp.Eval(code)
	return err
}

// SetContext sets the execution context for LLM calls.
func (r *YaegiREPL) SetContext(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ctx = ctx
}

// Execute runs Go code in the interpreter and returns the result.
func (r *YaegiREPL) Execute(ctx context.Context, code string) (*ExecutionResult, error) {
	r.mu.Lock()
	// Set context for LLM calls
	r.ctx = ctx

	// Reset buffers
	r.stdout.Reset()
	r.stderr.Reset()
	r.mu.Unlock() // Release lock before executing code (allows LLM calls to proceed)

	start := time.Now()

	// Execute the code (may call llmQuery which doesn't need lock)
	_, err := r.interp.Eval(code)

	r.mu.Lock()
	result := &ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}
	r.mu.Unlock()

	if err != nil {
		// Append error to stderr
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += err.Error()
	}

	return result, nil // We don't return error - execution errors go to stderr
}

// GetVariable retrieves a variable value from the interpreter.
// Used for resolving FINAL_VAR references.
func (r *YaegiREPL) GetVariable(name string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	v, err := r.interp.Eval(name)
	if err != nil {
		return "", fmt.Errorf("variable %q not found: %w", name, err)
	}

	if !v.IsValid() {
		return "", fmt.Errorf("variable %q is invalid", name)
	}

	return fmt.Sprintf("%v", v.Interface()), nil
}

// Reset clears the interpreter state.
func (r *YaegiREPL) Reset() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil

	// Create a fresh interpreter
	i := interp.New(interp.Options{
		Stdout: r.stdout,
		Stderr: r.stderr,
	})

	if err := i.Use(stdlib.Symbols); err != nil {
		return fmt.Errorf("failed to load stdlib: %w", err)
	}

	r.interp = i

	return r.injectBuiltins()
}

// GetLLMCalls returns and clears the recorded LLM calls.
func (r *YaegiREPL) GetLLMCalls() []LLMCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := r.llmCalls
	r.llmCalls = nil
	return calls
}

// ClearLLMCalls clears the recorded LLM calls.
func (r *YaegiREPL) ClearLLMCalls() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.llmCalls = nil
}

// GetLocals extracts user-defined variables from the interpreter.
// Returns a map of variable names to their values.
func (r *YaegiREPL) GetLocals() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()

	locals := make(map[string]any)

	// Check for commonly used variable names in RLM code
	varNames := []string{
		"context", "result", "answer", "data", "output", "response",
		"analysis", "summary", "final_answer", "count", "total",
		"items", "records", "values", "results", "findings",
	}

	for _, name := range varNames {
		v, err := r.interp.Eval(name)
		if err != nil || !v.IsValid() {
			continue
		}
		locals[name] = v.Interface()
	}

	return locals
}

// ContextInfo returns metadata about the loaded context.
func (r *YaegiREPL) ContextInfo() string {
	r.mu.Lock()
	defer r.mu.Unlock()

	v, err := r.interp.Eval("context")
	if err != nil {
		return "context not loaded"
	}

	if !v.IsValid() {
		return "context not loaded"
	}

	iface := v.Interface()
	switch ctx := iface.(type) {
	case string:
		return fmt.Sprintf("type=string, len=%d", len(ctx))
	default:
		return fmt.Sprintf("type=%T", ctx)
	}
}

// FormatExecutionResult formats an execution result for display to the LLM.
func FormatExecutionResult(result *ExecutionResult) string {
	var parts []string

	if result.Stdout != "" {
		parts = append(parts, result.Stdout)
	}
	if result.Stderr != "" {
		parts = append(parts, result.Stderr)
	}

	if len(parts) == 0 {
		return "No output"
	}

	return strings.Join(parts, "\n\n")
}
