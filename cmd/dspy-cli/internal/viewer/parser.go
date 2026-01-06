package viewer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// ParseLog parses a JSONL log file into LogData, auto-detecting the format.
func ParseLog(filename string) (*LogData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	data := &LogData{
		Filename: filename,
		Format:   FormatUnknown,
		Spans:    make(map[string]*SpanData),
	}

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// First, detect the format from the type field
		var entry struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}

		// Detect and parse based on format
		switch entry.Type {
		// RLM format types
		case "metadata":
			data.Format = FormatRLM
			var m Metadata
			if err := json.Unmarshal([]byte(line), &m); err == nil {
				data.Metadata = &m
			}
		case "iteration":
			data.Format = FormatRLM
			var iter Iteration
			if err := json.Unmarshal([]byte(line), &iter); err == nil {
				data.Iterations = append(data.Iterations, iter)
			}

		// Native DSPy format types
		case "session":
			data.Format = FormatDSPy
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				data.Session = parseSessionEvent(event)
			}
		case "span":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				parseSpanEvent(data, event)
			}
		case "llm_call":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				if call := parseLLMCallEvent(event); call != nil {
					data.LLMCalls = append(data.LLMCalls, *call)
				}
			}
		case "module":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				if mod := parseModuleEvent(event); mod != nil {
					data.Modules = append(data.Modules, *mod)
				}
			}
		case "code_exec":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				if exec := parseCodeExecEvent(event); exec != nil {
					data.CodeExecs = append(data.CodeExecs, *exec)
				}
			}
		case "tool_call":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				if tool := parseToolCallEvent(event); tool != nil {
					data.ToolCalls = append(data.ToolCalls, *tool)
				}
			}
		case "error":
			if data.Format == FormatUnknown {
				data.Format = FormatDSPy
			}
			var event TraceEvent
			if err := json.Unmarshal([]byte(line), &event); err == nil {
				data.Events = append(data.Events, event)
				if errData := parseErrorEvent(event); errData != nil {
					data.Errors = append(data.Errors, *errData)
				}
			}
		}
	}

	return data, scanner.Err()
}

// parseSessionEvent extracts session data from a session event.
func parseSessionEvent(event TraceEvent) *SessionData {
	session := &SessionData{
		TraceID:   event.TraceID,
		StartTime: event.Timestamp,
		Metadata:  event.Data,
	}
	return session
}

// parseSpanEvent extracts span data and updates the spans map.
func parseSpanEvent(data *LogData, event TraceEvent) {
	spanID := event.SpanID
	if spanID == "" {
		return
	}

	span, exists := data.Spans[spanID]
	if !exists {
		span = &SpanData{
			SpanID:   spanID,
			ParentID: event.ParentID,
		}
		data.Spans[spanID] = span
	}

	// Check if this is a start or end event
	if eventType, ok := event.Data["event"].(string); ok {
		switch eventType {
		case "start":
			span.StartTime = event.Timestamp
			if op, ok := event.Data["operation"].(string); ok {
				span.Operation = op
			}
			if inputs, ok := event.Data["inputs"].(map[string]interface{}); ok {
				span.Inputs = inputs
			}
		case "end":
			span.EndTime = event.Timestamp
			if duration, ok := event.Data["duration_ms"].(float64); ok {
				span.DurationMs = int64(duration)
			}
			if outputs, ok := event.Data["outputs"].(map[string]interface{}); ok {
				span.Outputs = outputs
			}
			if errMsg, ok := event.Data["error"].(string); ok {
				span.Error = errMsg
			}
		}
	}
}

// parseLLMCallEvent extracts LLM call data from an event.
func parseLLMCallEvent(event TraceEvent) *LLMCallData {
	call := &LLMCallData{
		SpanID:    event.SpanID,
		Timestamp: event.Timestamp,
	}

	if provider, ok := event.Data["provider"].(string); ok {
		call.Provider = provider
	}
	if model, ok := event.Data["model"].(string); ok {
		call.Model = model
	}
	if prompt, ok := event.Data["prompt"].(string); ok {
		call.Prompt = prompt
	}
	if response, ok := event.Data["response"].(string); ok {
		call.Response = response
	}
	if latency, ok := event.Data["latency_ms"].(float64); ok {
		call.LatencyMs = int64(latency)
	}

	// Parse usage data
	if usage, ok := event.Data["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			call.PromptTokens = int(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			call.CompletionTokens = int(ct)
		}
		if tt, ok := usage["total_tokens"].(float64); ok {
			call.TotalTokens = int(tt)
		}
		if cost, ok := usage["cost"].(float64); ok {
			call.Cost = cost
		}
	}

	return call
}

// parseModuleEvent extracts module execution data from an event.
func parseModuleEvent(event TraceEvent) *ModuleData {
	mod := &ModuleData{
		SpanID:    event.SpanID,
		Timestamp: event.Timestamp,
	}

	if mt, ok := event.Data["module_type"].(string); ok {
		mod.ModuleType = mt
	}
	if mn, ok := event.Data["module_name"].(string); ok {
		mod.ModuleName = mn
	}
	if sig, ok := event.Data["signature"].(string); ok {
		mod.Signature = sig
	}
	if duration, ok := event.Data["duration_ms"].(float64); ok {
		mod.DurationMs = int64(duration)
	}
	if calls, ok := event.Data["llm_calls"].(float64); ok {
		mod.LLMCalls = int(calls)
	}
	if tokens, ok := event.Data["total_tokens"].(float64); ok {
		mod.TotalTokens = int(tokens)
	}
	if success, ok := event.Data["success"].(bool); ok {
		mod.Success = success
	}
	if inputs, ok := event.Data["inputs"].(map[string]interface{}); ok {
		mod.Inputs = inputs
	}
	if outputs, ok := event.Data["outputs"].(map[string]interface{}); ok {
		mod.Outputs = outputs
	}

	return mod
}

// parseCodeExecEvent extracts code execution data from an event.
func parseCodeExecEvent(event TraceEvent) *CodeExecData {
	exec := &CodeExecData{
		SpanID:    event.SpanID,
		Timestamp: event.Timestamp,
	}

	if iter, ok := event.Data["iteration"].(float64); ok {
		exec.Iteration = int(iter)
	}
	if code, ok := event.Data["code"].(string); ok {
		exec.Code = code
	}
	if stdout, ok := event.Data["stdout"].(string); ok {
		exec.Stdout = stdout
	}
	if stderr, ok := event.Data["stderr"].(string); ok {
		exec.Stderr = stderr
	}
	if duration, ok := event.Data["duration_ms"].(float64); ok {
		exec.DurationMs = int64(duration)
	}
	if locals, ok := event.Data["locals"].(map[string]interface{}); ok {
		exec.Locals = locals
	}
	if calls, ok := event.Data["sub_llm_calls"].([]interface{}); ok {
		for _, c := range calls {
			if callMap, ok := c.(map[string]interface{}); ok {
				exec.SubLLMCalls = append(exec.SubLLMCalls, callMap)
			}
		}
	}

	return exec
}

// parseToolCallEvent extracts tool call data from an event.
func parseToolCallEvent(event TraceEvent) *ToolCallData {
	tool := &ToolCallData{
		SpanID:    event.SpanID,
		Timestamp: event.Timestamp,
	}

	if name, ok := event.Data["tool_name"].(string); ok {
		tool.ToolName = name
	}
	tool.Input = event.Data["input"]
	tool.Output = event.Data["output"]
	if duration, ok := event.Data["duration_ms"].(float64); ok {
		tool.DurationMs = int64(duration)
	}
	if success, ok := event.Data["success"].(bool); ok {
		tool.Success = success
	}
	if errMsg, ok := event.Data["error"].(string); ok {
		tool.Error = errMsg
	}

	return tool
}

// parseErrorEvent extracts error data from an event.
func parseErrorEvent(event TraceEvent) *ErrorData {
	errData := &ErrorData{
		SpanID:    event.SpanID,
		Timestamp: event.Timestamp,
	}

	if et, ok := event.Data["error_type"].(string); ok {
		errData.ErrorType = et
	}
	if msg, ok := event.Data["message"].(string); ok {
		errData.Message = msg
	}
	if rec, ok := event.Data["recoverable"].(bool); ok {
		errData.Recoverable = rec
	}

	return errData
}

// FilterIterations applies filters to iterations based on config (RLM format only).
func FilterIterations(iterations []Iteration, cfg Config) []Iteration {
	var result []Iteration

	for _, iter := range iterations {
		// Specific iteration filter
		if cfg.Iteration > 0 && iter.Iteration != cfg.Iteration {
			continue
		}

		// Errors only filter
		if cfg.ErrorsOnly {
			hasError := false
			for _, block := range iter.CodeBlocks {
				if block.Result.Stderr != "" {
					hasError = true
					break
				}
			}
			if !hasError {
				continue
			}
		}

		// Search filter
		if cfg.Search != "" {
			if !SearchInIteration(iter, cfg.Search) {
				continue
			}
		}

		result = append(result, iter)
	}

	return result
}

// FilterEvents applies filters to events based on config (DSPy format).
func FilterEvents(data *LogData, cfg Config) *LogData {
	if cfg.Search == "" && !cfg.ErrorsOnly {
		return data
	}

	filtered := &LogData{
		Filename: data.Filename,
		Format:   data.Format,
		Session:  data.Session,
		Spans:    data.Spans,
	}

	// Filter LLM calls
	for _, call := range data.LLMCalls {
		if cfg.Search != "" {
			if !searchInLLMCall(call, cfg.Search) {
				continue
			}
		}
		filtered.LLMCalls = append(filtered.LLMCalls, call)
	}

	// Filter modules
	for _, mod := range data.Modules {
		if cfg.ErrorsOnly && mod.Success {
			continue
		}
		if cfg.Search != "" {
			if !searchInModule(mod, cfg.Search) {
				continue
			}
		}
		filtered.Modules = append(filtered.Modules, mod)
	}

	// Filter code execs
	for _, exec := range data.CodeExecs {
		if cfg.ErrorsOnly && exec.Stderr == "" {
			continue
		}
		if cfg.Search != "" {
			if !searchInCodeExec(exec, cfg.Search) {
				continue
			}
		}
		filtered.CodeExecs = append(filtered.CodeExecs, exec)
	}

	// Filter tool calls
	for _, tool := range data.ToolCalls {
		if cfg.ErrorsOnly && tool.Success {
			continue
		}
		if cfg.Search != "" {
			if !searchInToolCall(tool, cfg.Search) {
				continue
			}
		}
		filtered.ToolCalls = append(filtered.ToolCalls, tool)
	}

	// Always include errors
	filtered.Errors = data.Errors

	return filtered
}

// SearchInIteration checks if a query matches content in an iteration (RLM format).
func SearchInIteration(iter Iteration, query string) bool {
	query = strings.ToLower(query)

	// Search in response
	if strings.Contains(strings.ToLower(iter.Response), query) {
		return true
	}

	// Search in code blocks
	for _, block := range iter.CodeBlocks {
		if strings.Contains(strings.ToLower(block.Code), query) {
			return true
		}
		if strings.Contains(strings.ToLower(block.Result.Stdout), query) {
			return true
		}
		if strings.Contains(strings.ToLower(block.Result.Stderr), query) {
			return true
		}
	}

	return false
}

func searchInLLMCall(call LLMCallData, query string) bool {
	query = strings.ToLower(query)
	return strings.Contains(strings.ToLower(call.Prompt), query) ||
		strings.Contains(strings.ToLower(call.Response), query) ||
		strings.Contains(strings.ToLower(call.Model), query) ||
		strings.Contains(strings.ToLower(call.Provider), query)
}

func searchInModule(mod ModuleData, query string) bool {
	query = strings.ToLower(query)
	return strings.Contains(strings.ToLower(mod.ModuleName), query) ||
		strings.Contains(strings.ToLower(mod.ModuleType), query) ||
		strings.Contains(strings.ToLower(mod.Signature), query)
}

func searchInCodeExec(exec CodeExecData, query string) bool {
	query = strings.ToLower(query)
	return strings.Contains(strings.ToLower(exec.Code), query) ||
		strings.Contains(strings.ToLower(exec.Stdout), query) ||
		strings.Contains(strings.ToLower(exec.Stderr), query)
}

func searchInToolCall(tool ToolCallData, query string) bool {
	query = strings.ToLower(query)
	return strings.Contains(strings.ToLower(tool.ToolName), query)
}

// FindSearchMatches finds all iteration indices that match the search query (RLM format).
func FindSearchMatches(iterations []Iteration, query string) []int {
	var matches []int
	for i, iter := range iterations {
		if SearchInIteration(iter, query) {
			matches = append(matches, i)
		}
	}
	return matches
}

// FindEventSearchMatches finds indices of events matching search (DSPy format).
func FindEventSearchMatches(data *LogData, query string) []int {
	var matches []int
	for i, event := range data.Events {
		if searchInEvent(event, query) {
			matches = append(matches, i)
		}
	}
	return matches
}

func searchInEvent(event TraceEvent, query string) bool {
	query = strings.ToLower(query)

	// Search in data fields
	for _, v := range event.Data {
		if str, ok := v.(string); ok {
			if strings.Contains(strings.ToLower(str), query) {
				return true
			}
		}
	}
	return false
}

// GetEventTimestamp returns the timestamp of an event for sorting/display.
func GetEventTimestamp(event TraceEvent) time.Time {
	return event.Timestamp
}
