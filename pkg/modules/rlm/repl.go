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
	"github.com/sourcegraph/conc/pool"
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

	// GetLocals extracts commonly used variables from the interpreter.
	GetLocals() map[string]any
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
	p := pool.New().WithErrors().WithContext(ctx)

	for i, prompt := range prompts {
		i, prompt := i, prompt
		p.Go(func(ctx context.Context) error {
			result, err := c.Query(ctx, prompt)
			if err != nil {
				results[i] = QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
				return err
			}
			results[i] = result
			return nil
		})
	}

	return results, p.Wait()
}

// YaegiREPL is a Yaegi-based Go interpreter with RLM capabilities.
//
// SECURITY NOTE: The interpreter is sandboxed by restricting imports to a safe
// subset of the standard library (no os, net, syscall, etc.). However, it does
// NOT protect against resource exhaustion attacks. LLM-generated code could
// potentially allocate large amounts of memory or create infinite loops that
// exceed the execution timeout. If running untrusted code in production, consider
// additional OS-level resource limits (e.g., cgroups, containers) or running
// the interpreter in a separate process with strict memory limits.
type YaegiREPL struct {
	interp    *interp.Interpreter
	stdout    *bytes.Buffer
	stderr    *bytes.Buffer
	llmClient SubLLMClient
	ctx       context.Context
	mu        sync.Mutex
	llmCalls  []LLMCall
}

// safeStdlibSymbols returns a sandboxed subset of stdlib symbols.
// Excludes dangerous packages: os, os/exec, net, syscall, unsafe, plugin, runtime.
func safeStdlibSymbols() interp.Exports {
	return interp.Exports{
		"fmt/fmt":               stdlib.Symbols["fmt/fmt"],
		"strings/strings":       stdlib.Symbols["strings/strings"],
		"strconv/strconv":       stdlib.Symbols["strconv/strconv"],
		"regexp/regexp":         stdlib.Symbols["regexp/regexp"],
		"math/math":             stdlib.Symbols["math/math"],
		"sort/sort":             stdlib.Symbols["sort/sort"],
		"encoding/json/json":    stdlib.Symbols["encoding/json/json"],
		"encoding/base64/base64": stdlib.Symbols["encoding/base64/base64"],
		"bytes/bytes":           stdlib.Symbols["bytes/bytes"],
		"unicode/unicode":       stdlib.Symbols["unicode/unicode"],
		"unicode/utf8/utf8":     stdlib.Symbols["unicode/utf8/utf8"],
		"time/time":             stdlib.Symbols["time/time"],
	}
}

// NewYaegiREPL creates a new YaegiREPL instance.
// Returns an error if initialization fails (e.g., stdlib loading or builtin injection).
func NewYaegiREPL(client SubLLMClient) (*YaegiREPL, error) {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)

	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})
	if err := i.Use(safeStdlibSymbols()); err != nil {
		return nil, fmt.Errorf("failed to load stdlib: %w", err)
	}

	r := &YaegiREPL{
		interp:    i,
		stdout:    stdout,
		stderr:    stderr,
		llmClient: client,
		ctx:       context.Background(),
	}
	if err := r.injectBuiltins(); err != nil {
		return nil, fmt.Errorf("failed to inject builtins: %w", err)
	}

	return r, nil
}

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

	_, err := r.interp.Eval(`
import "fmt"
import "strings"
import "regexp"
import "strconv"
import . "rlm/rlm"
`)
	return err
}

func (r *YaegiREPL) llmQuery(prompt string) string {
	start := time.Now()
	result, err := r.llmClient.Query(r.ctx, prompt)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	r.mu.Lock()
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
	r.mu.Unlock()

	return response
}

func (r *YaegiREPL) llmQueryBatched(prompts []string) []string {
	if len(prompts) == 0 {
		return []string{}
	}

	start := time.Now()
	results, err := r.llmClient.QueryBatched(r.ctx, prompts)
	duration := time.Since(start)
	avgDuration := duration / time.Duration(len(prompts))

	// If a batch-level error occurs (e.g., context cancellation),
	// the results slice might be incomplete. Return consistent error
	// messages for all prompts to avoid partial results.
	if err != nil {
		responses := make([]string, len(prompts))
		errMsg := fmt.Sprintf("Error: %v", err)
		r.mu.Lock()
		defer r.mu.Unlock()
		for i, prompt := range prompts {
			responses[i] = errMsg
			r.llmCalls = append(r.llmCalls, LLMCall{
				Prompt:   prompt,
				Response: errMsg,
				Duration: avgDuration,
			})
		}
		return responses
	}

	responses := make([]string, len(results))
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, res := range results {
		responses[i] = res.Response
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           prompts[i],
			Response:         res.Response,
			Duration:         avgDuration,
			PromptTokens:     res.PromptTokens,
			CompletionTokens: res.CompletionTokens,
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
		_, err := r.interp.Eval(`var context = ` + strconv.Quote(v))
		return err
	case map[string]any:
		return r.loadStructuredContext(v, "map[string]interface{}")
	case []any:
		return r.loadStructuredContext(v, "[]interface{}")
	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		return r.LoadContext(string(jsonBytes))
	}
}

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

// Execute runs Go code in the interpreter. Execution errors are captured in
// stderr rather than returned, allowing the caller to inspect all output.
func (r *YaegiREPL) Execute(ctx context.Context, code string) (*ExecutionResult, error) {
	r.mu.Lock()
	r.ctx = ctx
	r.stdout.Reset()
	r.stderr.Reset()
	r.mu.Unlock()

	start := time.Now()
	_, err := r.interp.Eval(code)

	r.mu.Lock()
	result := &ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}
	r.mu.Unlock()

	if err != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += err.Error()
	}

	return result, nil
}

// GetVariable retrieves a variable value from the interpreter.
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

// Reset clears the interpreter state and creates a fresh instance.
func (r *YaegiREPL) Reset() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil

	i := interp.New(interp.Options{
		Stdout: r.stdout,
		Stderr: r.stderr,
	})
	if err := i.Use(safeStdlibSymbols()); err != nil {
		return fmt.Errorf("failed to load stdlib: %w", err)
	}
	r.interp = i

	return r.injectBuiltins()
}

// GetLLMCalls returns and clears the recorded LLM calls.
// Returns a copy of the calls slice to prevent external modification.
func (r *YaegiREPL) GetLLMCalls() []LLMCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.llmCalls) == 0 {
		return nil
	}
	calls := make([]LLMCall, len(r.llmCalls))
	copy(calls, r.llmCalls)
	r.llmCalls = nil
	return calls
}

// ClearLLMCalls clears the recorded LLM calls.
func (r *YaegiREPL) ClearLLMCalls() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.llmCalls = nil
}

// GetLocals extracts commonly used variables from the interpreter.
func (r *YaegiREPL) GetLocals() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()

	locals := make(map[string]any)
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
	if err != nil || !v.IsValid() {
		return "context not loaded"
	}

	switch ctx := v.Interface().(type) {
	case string:
		return fmt.Sprintf("type=string, len=%d", len(ctx))
	default:
		return fmt.Sprintf("type=%T", ctx)
	}
}

// FormatExecutionResult formats an execution result for display.
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
