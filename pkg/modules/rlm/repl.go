package rlm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/google/uuid"
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

var errStdlibLoad = errors.New("failed to load stdlib")

// AsyncQueryHandle represents a pending async query.
type AsyncQueryHandle struct {
	id       string
	done     chan struct{}
	result   *QueryResponse
	err      error
	mu       sync.Mutex
	started  time.Time
}

// newAsyncQueryHandle creates a new async query handle.
func newAsyncQueryHandle() *AsyncQueryHandle {
	return &AsyncQueryHandle{
		id:      uuid.New().String(),
		done:    make(chan struct{}),
		started: time.Now(),
	}
}

// ID returns the unique identifier for this async query.
func (h *AsyncQueryHandle) ID() string {
	return h.id
}

// Wait blocks until the query completes and returns the result.
func (h *AsyncQueryHandle) Wait() (string, error) {
	<-h.done

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.err != nil {
		return "", h.err
	}
	if h.result != nil {
		return h.result.Response, nil
	}
	return "", nil
}

// Ready returns true if the result is available.
func (h *AsyncQueryHandle) Ready() bool {
	select {
	case <-h.done:
		return true
	default:
		return false
	}
}

// Result returns the result if ready, or empty string if not.
func (h *AsyncQueryHandle) Result() (string, bool) {
	if !h.Ready() {
		return "", false
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.result != nil {
		return h.result.Response, true
	}
	return "", true
}

// Error returns any error that occurred during the query.
func (h *AsyncQueryHandle) Error() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

// Duration returns the time elapsed since the query was started.
func (h *AsyncQueryHandle) Duration() time.Duration {
	return time.Since(h.started)
}

// complete marks the handle as complete with the given result.
func (h *AsyncQueryHandle) complete(resp QueryResponse, err error) {
	h.mu.Lock()
	if err != nil {
		h.err = err
	} else {
		h.result = &resp
	}
	h.mu.Unlock()

	close(h.done)
}

// AsyncBatchHandle represents a batch of pending async queries.
type AsyncBatchHandle struct {
	handles    []*AsyncQueryHandle
	allDone    chan struct{}
	doneOnce   sync.Once
	completed  int32
	totalCount int32
}

// newAsyncBatchHandle creates a new batch handle for the given handles.
func newAsyncBatchHandle(handles []*AsyncQueryHandle) *AsyncBatchHandle {
	bh := &AsyncBatchHandle{
		handles:    handles,
		allDone:    make(chan struct{}),
		totalCount: int32(len(handles)),
	}

	// Start a goroutine to track completion
	go bh.trackCompletion()

	return bh
}

// trackCompletion monitors all handles and closes allDone when all complete.
func (bh *AsyncBatchHandle) trackCompletion() {
	for _, h := range bh.handles {
		go func(handle *AsyncQueryHandle) {
			<-handle.done
			if atomic.AddInt32(&bh.completed, 1) == bh.totalCount {
				bh.doneOnce.Do(func() {
					close(bh.allDone)
				})
			}
		}(h)
	}
}

// WaitAll blocks until all queries complete and returns all results.
func (bh *AsyncBatchHandle) WaitAll() ([]string, error) {
	<-bh.allDone

	results := make([]string, len(bh.handles))
	var firstErr error

	for i, h := range bh.handles {
		result, err := h.Wait()
		if err != nil && firstErr == nil {
			firstErr = err
		}
		results[i] = result
	}

	return results, firstErr
}

// Ready returns true if all queries have completed.
func (bh *AsyncBatchHandle) Ready() bool {
	select {
	case <-bh.allDone:
		return true
	default:
		return false
	}
}

// Handles returns the individual query handles.
func (bh *AsyncBatchHandle) Handles() []*AsyncQueryHandle {
	return bh.handles
}

// CompletedCount returns the number of completed queries.
func (bh *AsyncBatchHandle) CompletedCount() int {
	return int(atomic.LoadInt32(&bh.completed))
}

// TotalCount returns the total number of queries in the batch.
func (bh *AsyncBatchHandle) TotalCount() int {
	return int(bh.totalCount)
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
	interp       *interp.Interpreter
	stdout       *bytes.Buffer
	stderr       *bytes.Buffer
	llmClient    SubLLMClient
	ctx          context.Context
	mu           sync.Mutex
	llmCalls     []LLMCall
	llmCallsMu   sync.Mutex // Separate mutex for async LLM call recording
	asyncQueries map[string]*AsyncQueryHandle
	asyncMu      sync.RWMutex
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
		return nil, fmt.Errorf("%w: %v", errStdlibLoad, err)
	}

	r := &YaegiREPL{
		interp:       i,
		stdout:       stdout,
		stderr:       stderr,
		llmClient:    client,
		ctx:          context.Background(),
		asyncQueries: make(map[string]*AsyncQueryHandle),
	}
	if err := r.injectBuiltins(); err != nil {
		return nil, fmt.Errorf("failed to inject builtins: %w", err)
	}

	return r, nil
}

func (r *YaegiREPL) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"Query":             reflect.ValueOf(r.llmQuery),
			"QueryBatched":      reflect.ValueOf(r.llmQueryBatched),
			"QueryAsync":        reflect.ValueOf(r.llmQueryAsync),
			"QueryBatchedAsync": reflect.ValueOf(r.llmQueryBatchedAsync),
			"WaitAsync":         reflect.ValueOf(r.waitAsync),
			"AsyncReady":        reflect.ValueOf(r.asyncReady),
			"AsyncResult":       reflect.ValueOf(r.asyncResult),
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

	r.recordLLMCall(prompt, response, duration, result.PromptTokens, result.CompletionTokens)
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

	if err != nil {
		responses := make([]string, len(prompts))
		errMsg := fmt.Sprintf("Error: %v", err)
		for i, prompt := range prompts {
			responses[i] = errMsg
			r.recordLLMCall(prompt, errMsg, avgDuration, 0, 0)
		}
		return responses
	}

	responses := make([]string, len(results))
	for i, res := range results {
		responses[i] = res.Response
		r.recordLLMCall(prompts[i], res.Response, avgDuration, res.PromptTokens, res.CompletionTokens)
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
// The mutex is held for the entire duration to ensure thread safety, as the
// yaegi interpreter is not safe for concurrent use.
// Panics from the Yaegi interpreter are recovered and reported as stderr errors.
func (r *YaegiREPL) Execute(ctx context.Context, code string) (result *ExecutionResult, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.ctx = ctx
	r.stdout.Reset()
	r.stderr.Reset()

	start := time.Now()

	// Recover from Yaegi interpreter panics (e.g., nil pointer dereference on certain code patterns)
	defer func() {
		if rec := recover(); rec != nil {
			result = &ExecutionResult{
				Stdout:   r.stdout.String(),
				Stderr:   fmt.Sprintf("interpreter panic: %v", rec),
				Duration: time.Since(start),
			}
			err = nil // Report as stderr, not error, so iteration can continue
		}
	}()

	_, evalErr := r.interp.Eval(code)

	result = &ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}

	if evalErr != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += evalErr.Error()
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

	// Clear LLM calls with its dedicated mutex
	r.llmCallsMu.Lock()
	r.llmCalls = nil
	r.llmCallsMu.Unlock()

	i := interp.New(interp.Options{
		Stdout: r.stdout,
		Stderr: r.stderr,
	})
	if err := i.Use(safeStdlibSymbols()); err != nil {
		return fmt.Errorf("%w: %v", errStdlibLoad, err)
	}
	r.interp = i

	return r.injectBuiltins()
}

// recordLLMCall appends an LLM call record using the dedicated LLM calls mutex.
// This is called from llmQuery/llmQueryBatched during Execute().
func (r *YaegiREPL) recordLLMCall(prompt, response string, duration time.Duration, promptTokens, completionTokens int) {
	r.llmCallsMu.Lock()
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	})
	r.llmCallsMu.Unlock()
}

// GetLLMCalls returns and clears the recorded LLM calls.
// Returns a copy of the calls slice to prevent external modification.
func (r *YaegiREPL) GetLLMCalls() []LLMCall {
	r.llmCallsMu.Lock()
	defer r.llmCallsMu.Unlock()
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
	r.llmCallsMu.Lock()
	defer r.llmCallsMu.Unlock()
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

// llmQueryAsync starts an async query and returns a handle ID.
// This is called from interpreted code via QueryAsync().
func (r *YaegiREPL) llmQueryAsync(prompt string) string {
	handle := newAsyncQueryHandle()

	// Track the handle
	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	// Start the async query
	go func() {
		start := time.Now()
		result, err := r.llmClient.Query(r.ctx, prompt)
		duration := time.Since(start)

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		// Record the async call using the dedicated LLM calls mutex
		// This avoids deadlock with the main interpreter mutex
		r.llmCallsMu.Lock()
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           prompt,
			Response:         response,
			Duration:         duration,
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens,
		})
		r.llmCallsMu.Unlock()

		// Complete the handle
		handle.complete(result, err)
	}()

	return handle.id
}

// llmQueryBatchedAsync starts batch async queries and returns handle IDs.
// This is called from interpreted code via QueryBatchedAsync().
func (r *YaegiREPL) llmQueryBatchedAsync(prompts []string) []string {
	handleIDs := make([]string, len(prompts))

	for i, prompt := range prompts {
		handleIDs[i] = r.llmQueryAsync(prompt)
	}

	return handleIDs
}

// waitAsync blocks until the async query with the given ID completes.
// This is called from interpreted code via WaitAsync().
func (r *YaegiREPL) waitAsync(handleID string) string {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return fmt.Sprintf("Error: async query %s not found", handleID)
	}

	result, err := handle.Wait()
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return result
}

// asyncReady returns true if the async query with the given ID is complete.
// This is called from interpreted code via AsyncReady().
func (r *YaegiREPL) asyncReady(handleID string) bool {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return true // Non-existent handle is considered "ready" (error case)
	}

	return handle.Ready()
}

// asyncResult returns the result if ready, or empty string if not.
// This is called from interpreted code via AsyncResult().
func (r *YaegiREPL) asyncResult(handleID string) string {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return fmt.Sprintf("Error: async query %s not found", handleID)
	}

	result, ready := handle.Result()
	if !ready {
		return ""
	}
	return result
}

// QueryAsync starts an async query and returns a handle.
// This is the Go API for async queries.
func (r *YaegiREPL) QueryAsync(prompt string) *AsyncQueryHandle {
	handle := newAsyncQueryHandle()

	// Track the handle
	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	// Start the async query
	go func() {
		start := time.Now()
		result, err := r.llmClient.Query(r.ctx, prompt)
		duration := time.Since(start)

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		// Record the async call using the dedicated LLM calls mutex
		// This avoids deadlock with the main interpreter mutex
		r.llmCallsMu.Lock()
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           prompt,
			Response:         response,
			Duration:         duration,
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens,
		})
		r.llmCallsMu.Unlock()

		// Complete the handle
		handle.complete(result, err)
	}()

	return handle
}

// QueryBatchedAsync starts batch async queries and returns a batch handle.
// This is the Go API for batch async queries.
func (r *YaegiREPL) QueryBatchedAsync(prompts []string) *AsyncBatchHandle {
	handles := make([]*AsyncQueryHandle, len(prompts))

	for i, prompt := range prompts {
		handles[i] = r.QueryAsync(prompt)
	}

	return newAsyncBatchHandle(handles)
}

// GetAsyncQuery returns the async query handle by ID.
func (r *YaegiREPL) GetAsyncQuery(handleID string) (*AsyncQueryHandle, bool) {
	r.asyncMu.RLock()
	defer r.asyncMu.RUnlock()
	handle, exists := r.asyncQueries[handleID]
	return handle, exists
}

// PendingAsyncQueries returns the number of pending async queries.
func (r *YaegiREPL) PendingAsyncQueries() int {
	r.asyncMu.RLock()
	defer r.asyncMu.RUnlock()

	count := 0
	for _, h := range r.asyncQueries {
		if !h.Ready() {
			count++
		}
	}
	return count
}

// WaitAllAsyncQueries waits for all pending async queries to complete.
func (r *YaegiREPL) WaitAllAsyncQueries() {
	r.asyncMu.RLock()
	handles := make([]*AsyncQueryHandle, 0, len(r.asyncQueries))
	for _, h := range r.asyncQueries {
		handles = append(handles, h)
	}
	r.asyncMu.RUnlock()

	for _, h := range handles {
		<-h.done
	}
}

// ClearAsyncQueries clears all tracked async queries.
func (r *YaegiREPL) ClearAsyncQueries() {
	r.asyncMu.Lock()
	defer r.asyncMu.Unlock()
	r.asyncQueries = make(map[string]*AsyncQueryHandle)
}
