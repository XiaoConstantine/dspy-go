package logging

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type TraceEventType string

const (
	TraceEventSession  TraceEventType = "session"
	TraceEventSpan     TraceEventType = "span"
	TraceEventLLMCall  TraceEventType = "llm_call"
	TraceEventModule   TraceEventType = "module"
	TraceEventCodeExec TraceEventType = "code_exec"
	TraceEventToolCall TraceEventType = "tool_call"
	TraceEventError    TraceEventType = "error"
)

// TraceFormat specifies the output format for trace files.
type TraceFormat int

const (
	// TraceFormatDSPy is the native dspy-go format with rich span/event data.
	TraceFormatDSPy TraceFormat = iota
	// TraceFormatRLM is compatible with rlm-go's viewer (metadata + iteration entries).
	TraceFormatRLM
)

// Context key for trace session.
type traceSessionKeyType struct{}

var traceSessionKey = traceSessionKeyType{}

// WithTraceSession adds a TraceSession to the context.
func WithTraceSession(ctx context.Context, session *TraceSession) context.Context {
	return context.WithValue(ctx, traceSessionKey, session)
}

// GetTraceSession retrieves the TraceSession from context.
func GetTraceSession(ctx context.Context) *TraceSession {
	if session, ok := ctx.Value(traceSessionKey).(*TraceSession); ok {
		return session
	}
	return nil
}

type TraceEvent struct {
	Type      TraceEventType         `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	TraceID   string                 `json:"trace_id"`
	SpanID    string                 `json:"span_id,omitempty"`
	ParentID  string                 `json:"parent_id,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

type TraceOutput struct {
	mu         sync.Mutex
	file       fileWriter
	path       string
	rotateSize int64
	curSize    int64
	maxFiles   int
	bufferSize int
	buffer     []byte
}

type TraceOutputOption func(*TraceOutput)

func WithTraceRotation(maxSize int64, maxFiles int) TraceOutputOption {
	return func(t *TraceOutput) {
		t.rotateSize = maxSize
		t.maxFiles = maxFiles
	}
}

func WithTraceBufferSize(size int) TraceOutputOption {
	return func(t *TraceOutput) {
		t.bufferSize = size
		t.buffer = make([]byte, 0, size)
	}
}

func NewTraceOutput(path string, opts ...TraceOutputOption) (*TraceOutput, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create trace directory: %w", err)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open trace file: %w", err)
	}

	info, err := file.Stat()
	var curSize int64 = 0
	if err == nil {
		curSize = info.Size()
	}

	output := &TraceOutput{
		file:       file,
		path:       path,
		curSize:    curSize,
		bufferSize: 4096,
		buffer:     make([]byte, 0, 4096),
	}

	for _, opt := range opts {
		opt(output)
	}

	return output, nil
}

func (t *TraceOutput) Write(event TraceEvent) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal trace event: %w", err)
	}
	data = append(data, '\n')

	entrySize := int64(len(data))
	if t.rotateSize > 0 && (t.curSize+entrySize) >= t.rotateSize {
		if err := t.rotate(); err != nil {
			return fmt.Errorf("failed to rotate trace file: %w", err)
		}
		t.curSize = 0
	}

	n, err := t.file.Write(data)
	if err != nil {
		return fmt.Errorf("failed to write trace event: %w", err)
	}

	t.curSize += int64(n)
	return nil
}

func (t *TraceOutput) Flush() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.file.Sync()
}

func (t *TraceOutput) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.file.Close()
}

func (t *TraceOutput) rotate() error {
	if err := t.file.Close(); err != nil {
		return err
	}

	backupPath := fmt.Sprintf("%s.%s", t.path, time.Now().Format("20060102-150405"))
	if err := os.Rename(t.path, backupPath); err != nil {
		return err
	}

	file, err := os.OpenFile(t.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}

	t.file = file
	t.curSize = 0

	if t.maxFiles > 0 {
		if err := t.cleanOldTraceFiles(); err != nil {
			fmt.Fprintf(os.Stderr, "Error cleaning old trace files: %v\n", err)
		}
	}

	return nil
}

func (t *TraceOutput) cleanOldTraceFiles() error {
	dir := filepath.Dir(t.path)
	base := filepath.Base(t.path)

	files, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	var traceFiles []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		name := file.Name()
		if filepath.Base(t.path) != name && len(name) > len(base) && name[:len(base)] == base {
			traceFiles = append(traceFiles, filepath.Join(dir, name))
		}
	}

	if len(traceFiles) > t.maxFiles {
		for i := 0; i < len(traceFiles)-t.maxFiles; i++ {
			if err := os.Remove(traceFiles[i]); err != nil {
				return err
			}
		}
	}

	return nil
}

type TraceSession struct {
	output    *TraceOutput
	traceID   string
	startTime time.Time
	mu        sync.Mutex
}

func NewTraceSession(path string, opts ...TraceOutputOption) (*TraceSession, error) {
	output, err := NewTraceOutput(path, opts...)
	if err != nil {
		return nil, err
	}

	traceID := generateTraceID()
	session := &TraceSession{
		output:    output,
		traceID:   traceID,
		startTime: time.Now(),
	}

	err = session.emitSessionStart(nil)
	if err != nil {
		output.Close()
		return nil, err
	}

	return session, nil
}

func (s *TraceSession) TraceID() string {
	return s.traceID
}

func (s *TraceSession) emitSessionStart(metadata map[string]any) error {
	data := map[string]interface{}{
		"start_time": s.startTime,
	}
	for k, v := range metadata {
		data[k] = v
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventSession,
		Timestamp: s.startTime,
		TraceID:   s.traceID,
		Data:      data,
	})
}

func (s *TraceSession) EmitSpanStart(spanID, parentID, operation string, inputs map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"event":     "start",
		"operation": operation,
	}
	if inputs != nil {
		data["inputs"] = inputs
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventSpan,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		ParentID:  parentID,
		Data:      data,
	})
}

func (s *TraceSession) EmitSpanEnd(spanID string, outputs map[string]any, err error, durationMs int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"event":       "end",
		"duration_ms": durationMs,
	}
	if outputs != nil {
		data["outputs"] = outputs
	}
	if err != nil {
		data["error"] = err.Error()
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventSpan,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) EmitLLMCall(spanID, provider, model, prompt, response string, usage *core.TokenUsage, latencyMs int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"provider":   provider,
		"model":      model,
		"prompt":     prompt,
		"response":   response,
		"latency_ms": latencyMs,
	}

	if usage != nil {
		data["usage"] = map[string]interface{}{
			"prompt_tokens":     usage.PromptTokens,
			"completion_tokens": usage.CompletionTokens,
			"total_tokens":      usage.TotalTokens,
			"cost":              usage.Cost,
		}
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventLLMCall,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) EmitModule(spanID, moduleType, moduleName, signature string, inputs, outputs map[string]any, durationMs int64, llmCalls int, totalTokens int, success bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"module_type":  moduleType,
		"module_name":  moduleName,
		"signature":    signature,
		"duration_ms":  durationMs,
		"llm_calls":    llmCalls,
		"total_tokens": totalTokens,
		"success":      success,
	}
	if inputs != nil {
		data["inputs"] = inputs
	}
	if outputs != nil {
		data["outputs"] = outputs
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventModule,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) EmitCodeExec(spanID string, iteration int, code, stdout, stderr string, locals map[string]any, durationMs int64, subLLMCalls []map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"iteration":   iteration,
		"code":        code,
		"stdout":      stdout,
		"stderr":      stderr,
		"duration_ms": durationMs,
	}
	if locals != nil {
		data["locals"] = locals
	}
	if subLLMCalls != nil {
		data["sub_llm_calls"] = subLLMCalls
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventCodeExec,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) EmitToolCall(spanID, toolName string, input, output any, durationMs int64, success bool, err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"tool_name":   toolName,
		"input":       input,
		"output":      output,
		"duration_ms": durationMs,
		"success":     success,
	}
	if err != nil {
		data["error"] = err.Error()
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventToolCall,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) EmitError(spanID, errorType, message string, recoverable bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data := map[string]interface{}{
		"error_type":  errorType,
		"message":     message,
		"recoverable": recoverable,
	}

	return s.output.Write(TraceEvent{
		Type:      TraceEventError,
		Timestamp: time.Now(),
		TraceID:   s.traceID,
		SpanID:    spanID,
		Data:      data,
	})
}

func (s *TraceSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.output.Flush(); err != nil {
		return err
	}
	return s.output.Close()
}

func StartTraceSession(ctx context.Context, path string, metadata map[string]any) (*TraceSession, error) {
	output, err := NewTraceOutput(path)
	if err != nil {
		return nil, err
	}

	traceID := ""
	if state := core.GetExecutionState(ctx); state != nil {
		traceID = state.GetTraceID()
	}
	if traceID == "" {
		traceID = generateTraceID()
	}

	session := &TraceSession{
		output:    output,
		traceID:   traceID,
		startTime: time.Now(),
	}

	err = session.emitSessionStart(metadata)
	if err != nil {
		output.Close()
		return nil, err
	}

	return session, nil
}

func generateTraceID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// ============================================================================
// RLM-Compatible Format Types (for rlm-go viewer compatibility)
// ============================================================================

// RLMMetadataEntry is compatible with rlm-go's MetadataEntry for viewer support.
type RLMMetadataEntry struct {
	Type              string         `json:"type"`
	Timestamp         string         `json:"timestamp"`
	RootModel         string         `json:"root_model"`
	MaxDepth          int            `json:"max_depth"`
	MaxIterations     int            `json:"max_iterations"`
	Backend           string         `json:"backend"`
	BackendKwargs     map[string]any `json:"backend_kwargs"`
	EnvironmentType   string         `json:"environment_type"`
	EnvironmentKwargs map[string]any `json:"environment_kwargs"`
	OtherBackends     any            `json:"other_backends"`
	Context           string         `json:"context,omitempty"`
	Query             string         `json:"query,omitempty"`
}

// RLMIterationEntry is compatible with rlm-go's IterationEntry for viewer support.
type RLMIterationEntry struct {
	Type          string           `json:"type"`
	Iteration     int              `json:"iteration"`
	Timestamp     string           `json:"timestamp"`
	Prompt        []RLMMessage     `json:"prompt"`
	Response      string           `json:"response"`
	CodeBlocks    []RLMCodeBlock   `json:"code_blocks"`
	FinalAnswer   any              `json:"final_answer"`
	IterationTime float64          `json:"iteration_time"`
}

// RLMMessage represents a chat message in RLM format.
type RLMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RLMCodeBlock represents an executed code block in RLM format.
type RLMCodeBlock struct {
	Code   string          `json:"code"`
	Result RLMCodeResult   `json:"result"`
}

// RLMCodeResult represents code execution results in RLM format.
type RLMCodeResult struct {
	Stdout        string         `json:"stdout"`
	Stderr        string         `json:"stderr"`
	Locals        map[string]any `json:"locals"`
	ExecutionTime float64        `json:"execution_time"`
	RLMCalls      []RLMCallEntry `json:"rlm_calls"`
}

// RLMCallEntry represents a sub-LLM call in RLM format.
type RLMCallEntry struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	ExecutionTime    float64 `json:"execution_time"`
}

// RLMTraceSession provides rlm-go viewer compatible logging.
type RLMTraceSession struct {
	mu             sync.Mutex
	file           *os.File
	path           string
	startTime      time.Time
	iterationCount int
}

// RLMSessionConfig holds configuration for an RLM trace session.
type RLMSessionConfig struct {
	RootModel     string
	MaxIterations int
	Backend       string
	BackendKwargs map[string]any
	Context       string
	Query         string
}

// NewRLMTraceSession creates a new RLM-compatible trace session.
func NewRLMTraceSession(logDir string, cfg RLMSessionConfig) (*RLMTraceSession, error) {
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, fmt.Errorf("create log directory: %w", err)
	}

	now := time.Now()
	filename := fmt.Sprintf("rlm_%s_%s.jsonl",
		now.Format("2006-01-02_15-04-05"),
		generateRLMSessionID(),
	)
	path := filepath.Join(logDir, filename)

	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create log file: %w", err)
	}

	session := &RLMTraceSession{
		file:      file,
		path:      path,
		startTime: now,
	}

	// Truncate context for display
	contextPreview := cfg.Context
	if len(contextPreview) > 500 {
		contextPreview = contextPreview[:500] + "..."
	}

	// Write metadata entry
	metadata := RLMMetadataEntry{
		Type:              "metadata",
		Timestamp:         now.Format(time.RFC3339Nano),
		RootModel:         cfg.RootModel,
		MaxDepth:          1,
		MaxIterations:     cfg.MaxIterations,
		Backend:           cfg.Backend,
		BackendKwargs:     cfg.BackendKwargs,
		EnvironmentType:   "dspy-go",
		EnvironmentKwargs: map[string]any{},
		OtherBackends:     nil,
		Context:           contextPreview,
		Query:             cfg.Query,
	}

	if err := session.writeEntry(metadata); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("write metadata: %w", err)
	}

	return session, nil
}

// LogIteration logs a single iteration in RLM-compatible format.
func (s *RLMTraceSession) LogIteration(
	prompt []RLMMessage,
	response string,
	codeBlocks []RLMCodeBlock,
	finalAnswer any,
	iterationTime time.Duration,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.iterationCount++

	entry := RLMIterationEntry{
		Type:          "iteration",
		Iteration:     s.iterationCount,
		Timestamp:     time.Now().Format(time.RFC3339Nano),
		Prompt:        prompt,
		Response:      response,
		CodeBlocks:    codeBlocks,
		FinalAnswer:   finalAnswer,
		IterationTime: iterationTime.Seconds(),
	}

	return s.writeEntry(entry)
}

// LogLLMCall logs an LLM call as a simple iteration (for non-RLM modules).
func (s *RLMTraceSession) LogLLMCall(
	prompt string,
	response string,
	promptTokens int,
	completionTokens int,
	latency time.Duration,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.iterationCount++

	// Convert to iteration format for viewer compatibility
	entry := RLMIterationEntry{
		Type:      "iteration",
		Iteration: s.iterationCount,
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Prompt: []RLMMessage{
			{Role: "user", Content: prompt},
		},
		Response:      response,
		CodeBlocks:    []RLMCodeBlock{},
		FinalAnswer:   nil,
		IterationTime: latency.Seconds(),
	}

	return s.writeEntry(entry)
}

// Path returns the path to the log file.
func (s *RLMTraceSession) Path() string {
	return s.path
}

// Close closes the log file.
func (s *RLMTraceSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.file != nil {
		return s.file.Close()
	}
	return nil
}

func (s *RLMTraceSession) writeEntry(entry any) error {
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	_, err = s.file.Write(append(data, '\n'))
	return err
}

func generateRLMSessionID() string {
	return fmt.Sprintf("%08x", time.Now().UnixNano()&0xFFFFFFFF)
}

// ============================================================================
// Tracing Interceptor for automatic LLM call capture
// ============================================================================

// TracingInterceptor creates a module interceptor that logs LLM calls.
func TracingInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, next core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		session := GetTraceSession(ctx)
		if session == nil {
			return next(ctx, inputs, opts...)
		}

		spanID := generateTraceID()[:16]
		startTime := time.Now()

		// Log span start
		_ = session.EmitSpanStart(spanID, "", info.ModuleType+":"+info.ModuleName, inputs)

		// Execute the module
		outputs, err := next(ctx, inputs, opts...)

		// Log span end
		durationMs := time.Since(startTime).Milliseconds()
		_ = session.EmitSpanEnd(spanID, outputs, err, durationMs)

		return outputs, err
	}
}

// TracingInterceptorWithRLM creates an interceptor that logs in RLM-compatible format.
func TracingInterceptorWithRLM(rlmSession *RLMTraceSession) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, next core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		startTime := time.Now()

		// Build prompt from inputs
		promptStr := ""
		if p, ok := inputs["prompt"].(string); ok {
			promptStr = p
		} else if q, ok := inputs["query"].(string); ok {
			promptStr = q
		}

		// Execute the module
		outputs, err := next(ctx, inputs, opts...)

		// Extract response
		responseStr := ""
		if r, ok := outputs["response"].(string); ok {
			responseStr = r
		} else if a, ok := outputs["answer"].(string); ok {
			responseStr = a
		}

		// Log as iteration
		_ = rlmSession.LogLLMCall(promptStr, responseStr, 0, 0, time.Since(startTime))

		return outputs, err
	}
}
