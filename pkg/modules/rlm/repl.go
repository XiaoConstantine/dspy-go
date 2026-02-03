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

	// SetVariable sets a variable in the interpreter.
	SetVariable(name, value string) error

	// Reset clears the interpreter state.
	Reset() error

	// ContextInfo returns metadata about the loaded context.
	ContextInfo() string

	// GetLocals extracts commonly used variables from the interpreter.
	GetLocals() map[string]any

	// GetVariableMetadata returns rich metadata about REPL variables.
	GetVariableMetadata() []REPLVariable

	// HasFinal returns true if FINAL() or FINAL_VAR() has been called.
	// This provides state-verified completion detection (Nightjar Algorithm 1 Gap 1).
	HasFinal() bool

	// Final returns the value passed to FINAL() or FINAL_VAR().
	// Returns empty string if not set.
	Final() string

	// ClearFinal resets the final state.
	ClearFinal()

	// HasSubmit returns true if SUBMIT() has been called with valid output.
	HasSubmit() bool

	// GetSubmitOutput returns the submitted output fields.
	GetSubmitOutput() map[string]any

	// SetSubmitSchema sets the expected output schema for SUBMIT validation.
	SetSubmitSchema(schema map[string]OutputFieldSpec)

	// GetLLMCalls returns the LLM calls made during code execution.
	GetLLMCalls() []LLMCall
}

// REPLVariable provides rich metadata about a variable in the REPL.
type REPLVariable struct {
	Name       string // Variable name
	Value      any    // The actual value
	Type       string // Type description (string, int, []string, map, etc.)
	Length     int    // Length/size for strings and collections (-1 if N/A)
	Preview    string // Truncated representation for display
	IsImportant bool  // True for key variables (result, answer, etc.)
}

// OutputFieldSpec defines expected type and validation for SUBMIT output fields.
type OutputFieldSpec struct {
	Type        string // Expected type: "string", "int", "float", "bool", "[]string", "map"
	Required    bool   // Whether the field is required
	Description string // Description of what this field should contain
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

	// State-verified completion (Nightjar Algorithm 1 Gap 1)
	// These fields allow programmatic detection of completion via FINAL()/FINAL_VAR()
	// calls in code, rather than relying on regex parsing of text output.
	finalSet   bool
	finalValue string
	finalMu    sync.RWMutex // Protects finalSet and finalValue

	// Typed SUBMIT support (Nightjar Algorithm 1 - typed output validation)
	// SUBMIT allows LLMs to provide structured output with type validation.
	submitSet    bool
	submitOutput map[string]any
	submitSchema map[string]OutputFieldSpec
	submitMu     sync.RWMutex // Protects submitSet, submitOutput, submitSchema

	// Context indexing support
	contextIndex  *ContextIndex
	embeddingFunc EmbeddingFunc
	chunkConfig   ChunkConfig
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
			"QueryRaw":          reflect.ValueOf(r.llmQueryRaw),
			"QueryWith":         reflect.ValueOf(r.llmQueryWith),
			"QueryBatched":      reflect.ValueOf(r.llmQueryBatched),
			"QueryBatchedRaw":   reflect.ValueOf(r.llmQueryBatchedRaw),
			"QueryAsync":        reflect.ValueOf(r.llmQueryAsync),
			"QueryBatchedAsync": reflect.ValueOf(r.llmQueryBatchedAsync),
			"WaitAsync":         reflect.ValueOf(r.waitAsync),
			"AsyncReady":        reflect.ValueOf(r.asyncReady),
			"AsyncResult":       reflect.ValueOf(r.asyncResult),
			// FINAL and FINAL_VAR allow LLMs to signal completion from within code blocks
			"FINAL":     reflect.ValueOf(r.finalAnswer),
			"FINAL_VAR": reflect.ValueOf(r.finalVarAnswer),
			// SUBMIT allows typed output with field validation
			"SUBMIT": reflect.ValueOf(r.submitOutput_),
			// Context indexing builtins for efficient slice-only queries
			"FindRelevant": reflect.ValueOf(r.findRelevant),
			"GetChunk":     reflect.ValueOf(r.getChunk),
			"GetContext":   reflect.ValueOf(r.getContextRange),
			"ChunkCount":   reflect.ValueOf(r.chunkCount),
			"LineCount":    reflect.ValueOf(r.lineCount),
		},
	}
	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Pre-import common packages and RLM functions so they're available without qualification
	// NOTE: All common packages are pre-imported so LLM code should NOT use import statements
	// (mixing imports with statements causes Yaegi to treat code as package-level where statements are invalid)
	_, err := r.interp.Eval(`
import "fmt"
import "strings"
import "regexp"
import "strconv"
import "encoding/json"
import "sort"
import . "rlm/rlm"
`)
	return err
}

// finalAnswer handles FINAL(value) calls from within code blocks.
// It sets the state-verified completion fields and prints a marker.
// This allows LLMs to signal completion from inside code, which is more natural.
// The state fields (finalSet, finalValue) provide reliable programmatic detection
// instead of regex parsing (Nightjar Algorithm 1 Gap 1).
func (r *YaegiREPL) finalAnswer(value string) string {
	// Set state-verified completion fields
	r.finalMu.Lock()
	r.finalSet = true
	r.finalValue = value
	r.finalMu.Unlock()

	// Also print to stdout for backward compatibility and debugging
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", value)
	return value
}

// finalVarAnswer handles FINAL_VAR(varName) calls from within code blocks.
// The varName is treated as the value directly (the variable was already evaluated by Go).
// This is called when LLM writes FINAL_VAR(answer) where answer is a Go variable.
// Sets state-verified completion fields (Nightjar Algorithm 1 Gap 1).
func (r *YaegiREPL) finalVarAnswer(value string) string {
	// Set state-verified completion fields
	r.finalMu.Lock()
	r.finalSet = true
	r.finalValue = value
	r.finalMu.Unlock()

	// Also print to stdout for backward compatibility and debugging
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", value)
	return value
}

// HasFinal returns true if FINAL() or FINAL_VAR() has been called.
// This provides state-verified completion detection (Nightjar Algorithm 1 Gap 1).
func (r *YaegiREPL) HasFinal() bool {
	r.finalMu.RLock()
	defer r.finalMu.RUnlock()
	return r.finalSet
}

// Final returns the value passed to FINAL() or FINAL_VAR().
// Returns empty string if not set.
func (r *YaegiREPL) Final() string {
	r.finalMu.RLock()
	defer r.finalMu.RUnlock()
	return r.finalValue
}

// ClearFinal resets the final state.
// Call this at the start of each iteration or when reusing the REPL.
func (r *YaegiREPL) ClearFinal() {
	r.finalMu.Lock()
	r.finalSet = false
	r.finalValue = ""
	r.finalMu.Unlock()
}

// =============================================================================
// SUBMIT - Typed Output Validation (Nightjar Algorithm 1)
// =============================================================================

// submitOutput_ handles SUBMIT(fields) calls from within code blocks.
// It validates the output against the schema (if set) and stores it.
// This provides typed output validation similar to DSPy Python.
func (r *YaegiREPL) submitOutput_(fields map[string]any) string {
	r.submitMu.Lock()
	defer r.submitMu.Unlock()

	// Validate against schema if set
	if r.submitSchema != nil {
		for name, spec := range r.submitSchema {
			value, exists := fields[name]
			if spec.Required && !exists {
				errMsg := fmt.Sprintf("SUBMIT error: missing required field '%s'", name)
				fmt.Fprintln(r.stderr, errMsg)
				return errMsg
			}
			if exists {
				if err := validateFieldType(value, spec.Type); err != nil {
					errMsg := fmt.Sprintf("SUBMIT error: field '%s' %v", name, err)
					fmt.Fprintln(r.stderr, errMsg)
					return errMsg
				}
			}
		}
	}

	// Store the validated output
	r.submitSet = true
	r.submitOutput = fields

	// Print confirmation
	fmt.Fprintf(r.stdout, "\nSUBMIT(%v)\n", fields)
	return "OK"
}

// validateFieldType checks if a value matches the expected type.
func validateFieldType(value any, expectedType string) error {
	if value == nil {
		return fmt.Errorf("is nil")
	}

	switch expectedType {
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("expected string, got %T", value)
		}
	case "int":
		switch value.(type) {
		case int, int64, int32, float64: // float64 covers JSON numbers
		default:
			return fmt.Errorf("expected int, got %T", value)
		}
	case "float":
		switch value.(type) {
		case float64, float32, int, int64:
		default:
			return fmt.Errorf("expected float, got %T", value)
		}
	case "bool":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected bool, got %T", value)
		}
	case "[]string":
		slice, ok := value.([]string)
		if !ok {
			// Try to convert []any to []string
			if anySlice, ok := value.([]any); ok {
				for _, v := range anySlice {
					if _, ok := v.(string); !ok {
						return fmt.Errorf("expected []string, got []any with non-string element")
					}
				}
			} else {
				return fmt.Errorf("expected []string, got %T", value)
			}
		}
		_ = slice
	case "map":
		switch value.(type) {
		case map[string]any, map[string]string:
		default:
			return fmt.Errorf("expected map, got %T", value)
		}
	}
	return nil
}

// HasSubmit returns true if SUBMIT() has been called with valid output.
func (r *YaegiREPL) HasSubmit() bool {
	r.submitMu.RLock()
	defer r.submitMu.RUnlock()
	return r.submitSet
}

// GetSubmitOutput returns the submitted output fields.
func (r *YaegiREPL) GetSubmitOutput() map[string]any {
	r.submitMu.RLock()
	defer r.submitMu.RUnlock()
	return r.submitOutput
}

// SetSubmitSchema sets the expected output schema for SUBMIT validation.
func (r *YaegiREPL) SetSubmitSchema(schema map[string]OutputFieldSpec) {
	r.submitMu.Lock()
	defer r.submitMu.Unlock()
	r.submitSchema = schema
}

// ClearSubmit resets the submit state.
func (r *YaegiREPL) ClearSubmit() {
	r.submitMu.Lock()
	r.submitSet = false
	r.submitOutput = nil
	r.submitMu.Unlock()
}

// =============================================================================
// Variable Metadata (Nightjar Algorithm 1 - rich variable info)
// =============================================================================

// importantVarNames lists variables that should be marked as important.
var importantVarNames = map[string]bool{
	"result": true, "answer": true, "output": true, "final_answer": true,
	"total": true, "count": true, "response": true, "summary": true,
}

// GetVariableMetadata returns rich metadata about REPL variables.
// This provides type info, length, and preview for better LLM context.
func (r *YaegiREPL) GetVariableMetadata() []REPLVariable {
	r.mu.Lock()
	defer r.mu.Unlock()

	varNames := []string{
		"context", "result", "answer", "data", "output", "response",
		"analysis", "summary", "final_answer", "count", "total",
		"items", "records", "values", "results", "findings",
		"subrlm_result", // Added for sub-RLM support
	}

	var variables []REPLVariable
	for _, name := range varNames {
		v, err := r.interp.Eval(name)
		if err != nil || !v.IsValid() {
			continue
		}

		val := v.Interface()
		meta := REPLVariable{
			Name:        name,
			Value:       val,
			Type:        getTypeName(val),
			Length:      getValueLength(val),
			Preview:     getValuePreview(val, 100),
			IsImportant: importantVarNames[name],
		}
		variables = append(variables, meta)
	}

	return variables
}

// getTypeName returns a human-readable type name.
func getTypeName(val any) string {
	if val == nil {
		return "nil"
	}
	t := reflect.TypeOf(val)
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Int, reflect.Int64, reflect.Int32:
		return "int"
	case reflect.Float64, reflect.Float32:
		return "float"
	case reflect.Bool:
		return "bool"
	case reflect.Slice:
		return fmt.Sprintf("[]%s", t.Elem().Name())
	case reflect.Map:
		return fmt.Sprintf("map[%s]%s", t.Key().Name(), t.Elem().Name())
	default:
		return t.String()
	}
}

// getValueLength returns the length/size of a value, or -1 if not applicable.
func getValueLength(val any) int {
	if val == nil {
		return -1
	}
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.String:
		return v.Len()
	case reflect.Slice, reflect.Array, reflect.Map:
		return v.Len()
	default:
		return -1
	}
}

// getValuePreview returns a truncated preview of the value.
func getValuePreview(val any, maxLen int) string {
	if val == nil {
		return "<nil>"
	}

	var preview string
	switch v := val.(type) {
	case string:
		preview = v
	default:
		preview = fmt.Sprintf("%v", val)
	}

	if len(preview) > maxLen {
		return preview[:maxLen] + "..."
	}
	return preview
}

// SetVariable sets a variable in the interpreter that can be accessed in subsequent code executions.
// This is used by sub-RLM to store results that the parent RLM can access.
func (r *YaegiREPL) SetVariable(name, value string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	escapedValue := strings.ReplaceAll(value, "`", "` + \"`\" + `")
	code := fmt.Sprintf("%s := `%s`", name, escapedValue)
	_, err := r.interp.Eval(code)
	return err
}

// buildPromptWithContext retrieves the context variable and prepends it to the prompt.
// This ensures sub-LLM queries have access to the loaded context data.
func (r *YaegiREPL) buildPromptWithContext(prompt string) string {
	// Try to get the context variable from the interpreter
	v, err := r.interp.Eval("context")
	if err != nil || !v.IsValid() {
		// No context loaded, use prompt as-is
		return prompt
	}

	contextStr := ""
	switch ctx := v.Interface().(type) {
	case string:
		contextStr = ctx
	default:
		// For non-string context, try JSON marshaling
		jsonBytes, err := json.Marshal(ctx)
		if err != nil {
			return prompt
		}
		contextStr = string(jsonBytes)
	}

	if contextStr == "" {
		return prompt
	}

	// Prepend context to prompt
	return fmt.Sprintf("Context data:\n%s\n\nTask: %s", contextStr, prompt)
}

// llmQuery makes a single LLM query. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in the prompt.
func (r *YaegiREPL) llmQuery(prompt string) string {
	start := time.Now()

	// Get the context variable from the interpreter and include it in the prompt
	fullPrompt := r.buildPromptWithContext(prompt)

	result, err := r.llmClient.Query(r.ctx, fullPrompt)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record the call (store original prompt for clarity)
	r.recordLLMCall(prompt, response, duration, result.PromptTokens, result.CompletionTokens)
	return response
}

// llmQueryRaw makes a single LLM query WITHOUT prepending context.
// Use this for slice-only queries where you want to control exactly what context is sent.
// This is essential for RLM token efficiency - send only the slice needed.
func (r *YaegiREPL) llmQueryRaw(prompt string) string {
	start := time.Now()

	// Do NOT prepend context - send prompt as-is
	result, err := r.llmClient.Query(r.ctx, prompt)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	r.recordLLMCall(prompt, response, duration, result.PromptTokens, result.CompletionTokens)
	return response
}

// llmQueryWith makes a single LLM query with an explicit context slice.
// Use this when you want to provide a specific context (e.g., a chunk) instead of the full context.
// Format: "Context:\n{slice}\n\nTask: {prompt}"
func (r *YaegiREPL) llmQueryWith(contextSlice, prompt string) string {
	start := time.Now()

	// Build prompt with the provided slice instead of full context
	var fullPrompt string
	if contextSlice != "" {
		fullPrompt = fmt.Sprintf("Context:\n%s\n\nTask: %s", contextSlice, prompt)
	} else {
		fullPrompt = prompt
	}

	result, err := r.llmClient.Query(r.ctx, fullPrompt)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record with the original prompt for clarity in logs
	r.recordLLMCall(prompt, response, duration, result.PromptTokens, result.CompletionTokens)
	return response
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in each prompt.
func (r *YaegiREPL) llmQueryBatched(prompts []string) []string {
	if len(prompts) == 0 {
		return []string{}
	}

	start := time.Now()

	// Build full prompts with context included
	fullPrompts := make([]string, len(prompts))
	for i, p := range prompts {
		fullPrompts[i] = r.buildPromptWithContext(p)
	}

	results, err := r.llmClient.QueryBatched(r.ctx, fullPrompts)
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

// llmQueryBatchedRaw makes concurrent LLM queries WITHOUT prepending context.
// Use this for parallel slice-only queries where you control the context per-prompt.
func (r *YaegiREPL) llmQueryBatchedRaw(prompts []string) []string {
	if len(prompts) == 0 {
		return []string{}
	}

	start := time.Now()

	// Do NOT prepend context - send prompts as-is
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

	var contentStr string

	switch v := payload.(type) {
	case string:
		contentStr = v
		_, err := r.interp.Eval(`var context = ` + strconv.Quote(v))
		if err != nil {
			return err
		}
	case map[string]any:
		if err := r.loadStructuredContext(v, "map[string]interface{}"); err != nil {
			return err
		}
		// Convert to string for indexing
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to marshal map context for indexing: %w", err)
		}
		contentStr = string(jsonBytes)
	case []any:
		if err := r.loadStructuredContext(v, "[]interface{}"); err != nil {
			return err
		}
		// Convert to string for indexing
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to marshal slice context for indexing: %w", err)
		}
		contentStr = string(jsonBytes)
	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		// Recurse with string
		r.mu.Unlock()
		err = r.LoadContext(string(jsonBytes))
		r.mu.Lock()
		return err
	}

	// Build context index for efficient slicing
	config := r.chunkConfig
	if config.MaxChunkSize == 0 {
		config = DefaultChunkConfig()
	}
	r.contextIndex = NewContextIndex(contentStr, config)

	// Set embedding function if available
	if r.embeddingFunc != nil {
		r.contextIndex.SetEmbeddingFunc(r.embeddingFunc)
	}

	return nil
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
// It automatically includes the context variable (if loaded) in the prompt.
func (r *YaegiREPL) llmQueryAsync(prompt string) string {
	handle := newAsyncQueryHandle()

	// Build full prompt with context included (capture before goroutine)
	fullPrompt := r.buildPromptWithContext(prompt)

	// Track the handle
	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	// Start the async query
	go func() {
		start := time.Now()
		result, err := r.llmClient.Query(r.ctx, fullPrompt)
		duration := time.Since(start)

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		// Record the async call using the dedicated LLM calls mutex (store original prompt for clarity)
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

// =============================================================================
// Context Indexing Support
// =============================================================================

// SetEmbeddingFunc sets the function used for semantic search.
// If not set, FindRelevant falls back to keyword matching.
func (r *YaegiREPL) SetEmbeddingFunc(fn EmbeddingFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.embeddingFunc = fn
}

// SetChunkConfig sets the chunking configuration for context indexing.
func (r *YaegiREPL) SetChunkConfig(config ChunkConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.chunkConfig = config
}

// GetContextIndex returns the current context index (for external access).
func (r *YaegiREPL) GetContextIndex() *ContextIndex {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.contextIndex
}

// IndexContext eagerly indexes the loaded context for semantic search.
// Call this after LoadContext if you want embedding-based FindRelevant.
func (r *YaegiREPL) IndexContext(ctx context.Context) error {
	r.mu.Lock()
	idx := r.contextIndex
	r.mu.Unlock()

	if idx == nil {
		return fmt.Errorf("no context loaded")
	}

	return idx.IndexEagerly(ctx)
}

// findRelevant is the REPL builtin for finding relevant chunks.
// It returns chunk contents as a slice of strings.
// Note: Does NOT acquire r.mu since it's called from within Execute which already holds the lock.
// The contextIndex has its own internal mutex for thread safety.
func (r *YaegiREPL) findRelevant(query string, topK int) []string {
	idx := r.contextIndex
	if idx == nil {
		return []string{}
	}

	chunks, err := idx.FindRelevant(r.ctx, query, topK)
	if err != nil {
		return []string{fmt.Sprintf("Error: %v", err)}
	}

	results := make([]string, len(chunks))
	for i, chunk := range chunks {
		results[i] = chunk.Content
	}
	return results
}

// getChunk is the REPL builtin for getting a chunk by ID.
// Note: Does NOT acquire r.mu since it's called from within Execute which already holds the lock.
func (r *YaegiREPL) getChunk(id int) string {
	idx := r.contextIndex
	if idx == nil {
		return ""
	}

	chunk, ok := idx.GetChunk(id)
	if !ok {
		return ""
	}
	return chunk.Content
}

// getContextRange is the REPL builtin for getting context by line range.
// Note: Does NOT acquire r.mu since it's called from within Execute which already holds the lock.
func (r *YaegiREPL) getContextRange(startLine, endLine int) string {
	idx := r.contextIndex
	if idx == nil {
		return ""
	}

	return idx.GetContext(startLine, endLine)
}

// chunkCount is the REPL builtin for getting the number of chunks.
// Note: Does NOT acquire r.mu since it's called from within Execute which already holds the lock.
func (r *YaegiREPL) chunkCount() int {
	idx := r.contextIndex
	if idx == nil {
		return 0
	}
	return idx.ChunkCount()
}

// lineCount is the REPL builtin for getting the number of lines.
// Note: Does NOT acquire r.mu since it's called from within Execute which already holds the lock.
func (r *YaegiREPL) lineCount() int {
	idx := r.contextIndex
	if idx == nil {
		return 0
	}
	return idx.LineCount()
}
