package adapters

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/internal/schema"
	llm "github.com/XiaoConstantine/llm-go"
)

// JSONAdapter formats and parses structured outputs as a JSON object in exact
// signature field order, stepping down cleanly across provider capabilities:
// native json_schema -> json_object -> prompt-only JSON. It never falls back
// to another adapter when used directly.
type JSONAdapter struct {
	Cache *AdapterCache
}

var _ core.Adapter = (*JSONAdapter)(nil)

// NewJSONAdapter creates a new JSONAdapter.
func NewJSONAdapter() *JSONAdapter {
	return &JSONAdapter{}
}

// WithCache attaches a validated-output cache to the adapter.
func (a *JSONAdapter) WithCache(c *AdapterCache) *JSONAdapter {
	a.Cache = c
	return a
}

func (a *JSONAdapter) Name() string    { return "json" }
func (a *JSONAdapter) Version() string { return "v1" }

// Format renders signature, demos, conversation history, and current inputs
// into canonical chat messages with JSON assistant demo turns in exact
// signature order.
func (a *JSONAdapter) Format(signature core.Signature, demos []core.Example, inputs map[string]any) ([]core.ChatMessage, error) {
	sigNoHist, historyTurns, hasHistory := extractHistoryTurns(signature, inputs)

	messages := make([]core.ChatMessage, 0, 2+len(demos)*2+len(historyTurns)*2)
	messages = append(messages, core.ChatMessage{
		Role:    "system",
		Content: []core.ContentBlock{core.NewTextBlock(a.formatSystemMessage(sigNoHist))},
	})

	for _, demo := range demos {
		messages = append(messages, a.formatDemoMessages(sigNoHist, demo)...)
	}

	if hasHistory {
		for _, turn := range historyTurns {
			turnDemo := core.Example{Inputs: turn, Outputs: turn}
			messages = append(messages, a.formatDemoMessages(sigNoHist, turnDemo)...)
		}
	}

	userBlocks := buildInputContentBlocks(
		sigNoHist,
		inputs,
		func(f core.InputField, val string) string {
			return fmt.Sprintf("<%s>\n%s\n</%s>", f.Name, val, f.Name)
		},
		func(f core.InputField) string {
			return fmt.Sprintf("<%s>\n", f.Name)
		},
		a.userOutputReminder(sigNoHist),
	)
	messages = append(messages, core.ChatMessage{
		Role:    "user",
		Content: userBlocks,
	})
	return messages, nil
}

func (a *JSONAdapter) formatSystemMessage(sig core.Signature) string {
	var sb strings.Builder
	if len(sig.Inputs) > 0 {
		sb.WriteString("Your input fields are:\n")
		for i, f := range sig.Inputs {
			fmt.Fprintf(&sb, "%d. `%s`", i+1, f.Name)
			if f.Description != "" {
				fmt.Fprintf(&sb, ": %s", f.Description)
			}
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
	}

	if len(sig.Outputs) > 0 {
		sb.WriteString("Your output fields are:\n")
		for i, f := range sig.Outputs {
			fmt.Fprintf(&sb, "%d. `%s`%s", i+1, f.Name, fieldTypeHint(f))
			if f.Description != "" {
				fmt.Fprintf(&sb, ": %s", f.Description)
			}
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
		sb.WriteString("Respond with a single valid JSON object containing these keys in order:\n")
		sb.WriteString(a.formatJSONTemplate(sig))
		sb.WriteString("\n\n")
	}

	if sig.Instruction != "" {
		sb.WriteString("Task instructions:\n")
		sb.WriteString(sig.Instruction)
	}
	return strings.TrimSpace(sb.String())
}

func (a *JSONAdapter) formatJSONTemplate(sig core.Signature) string {
	var sb strings.Builder
	sb.WriteString("{\n")
	for i, f := range sig.Outputs {
		keyJSON, _ := json.Marshal(f.Name)
		fmt.Fprintf(&sb, "  %s: \"<%s>\"", string(keyJSON), f.Name)
		if i < len(sig.Outputs)-1 {
			sb.WriteByte(',')
		}
		sb.WriteByte('\n')
	}
	sb.WriteByte('}')
	return sb.String()
}

func (a *JSONAdapter) userOutputReminder(sig core.Signature) string {
	if len(sig.Outputs) == 0 {
		return ""
	}
	names := make([]string, len(sig.Outputs))
	for i, f := range sig.Outputs {
		names[i] = "`" + f.Name + "`" + fieldTypeHint(f)
	}
	return "Respond with a valid JSON object with fields in this exact order: " + strings.Join(names, ", then ") + "."
}

func (a *JSONAdapter) formatDemoMessages(sig core.Signature, demo core.Example) []core.ChatMessage {
	userBlocks := buildInputContentBlocks(
		sig,
		demo.Inputs,
		func(f core.InputField, val string) string {
			return fmt.Sprintf("<%s>\n%s\n</%s>", f.Name, val, f.Name)
		},
		func(f core.InputField) string {
			return fmt.Sprintf("<%s>\n", f.Name)
		},
		"",
	)
	order := make([]string, len(sig.Outputs))
	outMap := make(map[string]any, len(sig.Outputs))
	for i, f := range sig.Outputs {
		order[i] = f.Name
		if val, ok := demo.Outputs[f.Name]; ok {
			outMap[f.Name] = val
		} else if f.Optional {
			outMap[f.Name] = nil
		}
	}
	raw, err := schema.MarshalOrderedObject(order, outMap)
	if err != nil {
		raw = json.RawMessage("{}")
	}
	return []core.ChatMessage{
		{Role: "user", Content: userBlocks},
		{Role: "assistant", Content: []core.ContentBlock{core.NewTextBlock(string(raw))}},
	}
}

// Parse strips optional markdown fences/surrounding prose, validates the JSON
// strictly against the compiled signature schema (rejecting duplicate keys and
// invalid UTF-8), and coerces values into declared Go types.
func (a *JSONAdapter) Parse(signature core.Signature, content string) (map[string]any, error) {
	if len(signature.Outputs) == 0 {
		return map[string]any{}, nil
	}
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: "model returned an empty response",
			Raw:     content,
		}
	}

	candidate, problem := extractJSONObjectBytes(trimmed)
	if problem != "" {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: problem,
			Raw:     content,
		}
	}

	compiled, err := schema.CompileSignatureSchema(signature)
	if err != nil {
		return nil, fmt.Errorf("compile signature schema: %w", err)
	}

	if err := schema.ValidateJSON(compiled.Raw, candidate); err != nil {
		return nil, a.diagnoseValidationFailure(signature, candidate, content, err)
	}

	// Decode with UseNumber to preserve integer precision.
	dec := json.NewDecoder(strings.NewReader(string(candidate)))
	dec.UseNumber()
	var decoded map[string]any
	if err := dec.Decode(&decoded); err != nil {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: "failed to decode JSON object",
			Raw:     content,
			Err:     err,
		}
	}

	outputs := make(map[string]any, len(signature.Outputs))
	invalidFields := make(map[string]string)
	var missingFields []string
	for _, field := range signature.Outputs {
		rawVal, exists := decoded[field.Name]
		if !exists {
			if field.IsRequired() {
				missingFields = append(missingFields, field.Name)
			} else {
				outputs[field.Name] = nil
			}
			continue
		}
		coerced, coerceErr := schema.CoerceFieldValue(field, rawVal)
		if coerceErr != nil {
			invalidFields[field.Name] = coerceErr.Error()
			continue
		}
		outputs[field.Name] = coerced
	}
	if len(missingFields) > 0 || len(invalidFields) > 0 {
		return nil, &dspyerrors.ParseError{
			Adapter:       a.Name(),
			MissingFields: missingFields,
			InvalidFields: invalidFields,
			Raw:           content,
		}
	}
	return outputs, nil
}

func (a *JSONAdapter) diagnoseValidationFailure(
	signature core.Signature,
	candidate json.RawMessage,
	rawContent string,
	validationErr error,
) error {
	var missing []string
	invalid := map[string]string{}

	var obj map[string]any
	if json.Unmarshal(candidate, &obj) == nil && obj != nil {
		for _, field := range signature.Outputs {
			val, exists := obj[field.Name]
			if !exists {
				missing = append(missing, field.Name)
				continue
			}
			if _, err := schema.CoerceFieldValue(field, val); err != nil {
				invalid[field.Name] = err.Error()
			}
		}
	}
	if len(invalid) == 0 {
		invalid = nil
	}
	return &dspyerrors.ParseError{
		Adapter:       a.Name(),
		MissingFields: missing,
		InvalidFields: invalid,
		Raw:           rawContent,
		Message:       "JSON schema validation failed",
		Err:           validationErr,
	}
}

// Call runs the full JSONAdapter pipeline: native reasoning resolution,
// capability step-down, cache lookup, sync or streaming model call, strict
// local validation, and cache storage.
func (a *JSONAdapter) Call(
	ctx context.Context,
	model core.LLM,
	signature core.Signature,
	demos []core.Example,
	inputs map[string]any,
	opts core.AdapterCallOptions,
) (*core.AdapterResult, error) {
	if a.Cache == nil {
		if cache := modelCache(model); cache != nil {
			withCache := *a
			withCache.Cache = cache
			return withCache.Call(ctx, model, signature, demos, inputs, opts)
		}
	}

	plan, err := resolveNativeReasoning(model, signature, opts)
	if err != nil {
		return nil, err
	}

	messages, err := a.Format(plan.wireSignature, demos, inputs)
	if err != nil {
		return nil, err
	}

	compiled, err := schema.CompileSignatureSchema(plan.wireSignature)
	if err != nil {
		return nil, err
	}

	req := core.ChatRequest{
		Messages:              messages,
		Tools:                 opts.Tools,
		ToolChoice:            opts.ToolChoice,
		ParallelToolCalls:     opts.ParallelToolCalls,
		ReasoningEffort:       plan.reasoningEffort,
		ReasoningBudgetTokens: opts.ReasoningBudgetTokens,
		Options:               opts.GenerateOptions,
	}

	// Step down cleanly BEFORE the call based on honest capabilities.
	enforcement := "prompt_only"
	if len(plan.wireSignature.Outputs) > 0 {
		if core.HasCapability(model, core.CapabilityJSONSchema) && compiled.StrictCompatible {
			req.ResponseFormat = core.ResponseFormatJSONSchema
			req.ResponseSchema = &core.ResponseSchema{
				Name:       "response",
				Schema:     compiled.Raw,
				Strictness: core.ToolStrictPrefer,
			}
			enforcement = "native"
		} else if core.HasCapability(model, core.CapabilityJSON) {
			req.ResponseFormat = core.ResponseFormatJSONObject
			enforcement = "json_object"
		}
	}

	var cacheKey string
	if a.Cache != nil && !opts.DisableCache {
		if key, keyErr := buildAdapterCacheKey(a.Name(), a.Version(), model, req); keyErr == nil {
			cacheKey = key
			if entry, hit := a.Cache.get(ctx, cacheKey, signature); hit {
				if opts.StreamHandler != nil {
					if err := replayCachedStream(signature, entry, opts.StreamHandler); err != nil {
						return nil, err
					}
				}
				return entry.result(a), nil
			}
		}
	}

	if opts.StreamHandler != nil {
		var spent *core.TokenInfo
		res, err := a.callStreaming(ctx, model, signature, plan, req, enforcement, cacheKey, opts.StreamHandler, &spent)
		if err != nil {
			recordRejectedUsage(ctx, spent)
		}
		return res, err
	}

	resp, err := executeChat(ctx, model, req)
	if err != nil {
		return nil, classifyModelError(a.Name(), err)
	}
	if err := checkResponseStatus(resp, func(raw string) map[string]any {
		return partialJSONFields(plan.wireSignature, raw)
	}); err != nil {
		recordRejectedUsage(ctx, resp.Usage)
		return nil, err
	}

	outputs, err := a.Parse(plan.wireSignature, resp.Content)
	if err != nil {
		recordRejectedUsage(ctx, resp.Usage)
		return nil, err
	}
	if plan.usingNative {
		outputs[plan.nativeFieldName] = resp.ReasoningSummary
	}

	if a.Cache != nil && cacheKey != "" && (resp.FinishReason == "" || resp.FinishReason == core.FinishReasonStop || resp.FinishReason == core.FinishReasonToolCall) {
		a.Cache.set(ctx, cacheKey, cachedAdapterEntry{
			Outputs:           outputs,
			ToolCalls:         resp.ToolCalls,
			ReasoningSummary:  resp.ReasoningSummary,
			FinishReason:      resp.FinishReason,
			SchemaEnforcement: enforcement,
		})
	}

	return &core.AdapterResult{
		Outputs:           outputs,
		ToolCalls:         resp.ToolCalls,
		ReasoningSummary:  resp.ReasoningSummary,
		FinishReason:      resp.FinishReason,
		Usage:             resp.Usage,
		AdapterName:       a.Name(),
		AdapterVersion:    a.Version(),
		SchemaEnforcement: enforcement,
		RawContent:        resp.Content,
	}, nil
}

func (a *JSONAdapter) callStreaming(
	ctx context.Context,
	model core.LLM,
	originalSig core.Signature,
	plan reasoningPlan,
	req core.ChatRequest,
	enforcement string,
	cacheKey string,
	handler core.StreamHandler,
	spent **core.TokenInfo,
) (*core.AdapterResult, error) {
	stream, err := executeStreamChat(ctx, model, req)
	if err != nil {
		return nil, classifyModelError(a.Name(), err)
	}
	defer stream.Cancel()

	var fullRaw strings.Builder
	var reasoningBuf strings.Builder
	var lastUsage *core.TokenInfo
	defer func() { *spent = lastUsage }()
	var lastFinish core.FinishReason
	var lastRefusal *string
	var toolCalls []core.ToolCall

	emitted := make(map[string]int, len(plan.wireSignature.Outputs))
	anyEmitted := false
	fieldOrder := make([]string, len(plan.wireSignature.Outputs))
	for i, f := range plan.wireSignature.Outputs {
		fieldOrder[i] = f.Name
	}

	for chunk := range stream.ChunkChannel {
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
		if chunk.FinishReason != "" {
			lastFinish = chunk.FinishReason
		}
		if chunk.Refusal != nil {
			lastRefusal = chunk.Refusal
		}
		toolCalls = append(toolCalls, chunk.StreamToolCalls()...)
		if chunk.Error != nil {
			err := classifyModelError(a.Name(), chunk.Error)
			if anyEmitted {
				return nil, &dspyerrors.StreamParseError{
					PartialFields: partialJSONFields(plan.wireSignature, fullRaw.String()),
					FinishReason:  string(lastFinish),
					Raw:           fullRaw.String(),
					Err:           err,
				}
			}
			return nil, err
		}
		if chunk.ReasoningSummary != "" {
			reasoningBuf.WriteString(chunk.ReasoningSummary)
			if plan.usingNative {
				anyEmitted = true
				if err := handler(core.StreamChunk{Field: plan.nativeFieldName, Content: chunk.ReasoningSummary, ReasoningSummary: chunk.ReasoningSummary}); err != nil {
					return nil, err
				}
			}
		}
		if chunk.Content != "" {
			fullRaw.WriteString(chunk.Content)
			if obj, ok := llm.ParseStreamingJSON(fullRaw.String()).(map[string]any); ok {
				for _, name := range fieldOrder {
					strVal, ok := obj[name].(string)
					if !ok {
						continue
					}
					prevLen := emitted[name]
					if len(strVal) > prevLen {
						delta := strVal[prevLen:]
						emitted[name] = len(strVal)
						anyEmitted = true
						if err := handler(core.StreamChunk{Field: name, Content: delta}); err != nil {
							return nil, err
						}
					}
				}
			}
		}
		if chunk.Done {
			break
		}
	}

	resp := &core.ChatResponse{
		Content:          fullRaw.String(),
		ReasoningSummary: reasoningBuf.String(),
		FinishReason:     lastFinish,
		Refusal:          lastRefusal,
		ToolCalls:        toolCalls,
		Usage:            lastUsage,
	}
	if statusErr := checkResponseStatus(resp, func(raw string) map[string]any {
		return partialJSONFields(plan.wireSignature, raw)
	}); statusErr != nil {
		if anyEmitted {
			return nil, &dspyerrors.StreamParseError{
				PartialFields: partialJSONFields(plan.wireSignature, resp.Content),
				FinishReason:  string(lastFinish),
				Raw:           resp.Content,
				Err:           statusErr,
			}
		}
		return nil, statusErr
	}

	outputs, parseErr := a.Parse(plan.wireSignature, resp.Content)
	if parseErr != nil {
		if anyEmitted {
			return nil, &dspyerrors.StreamParseError{
				PartialFields: partialJSONFields(plan.wireSignature, resp.Content),
				FinishReason:  string(lastFinish),
				Raw:           resp.Content,
				Err:           parseErr,
			}
		}
		return nil, parseErr
	}
	// The incremental parse reads nothing from a reply that wraps its object
	// in a code fence or prose, so send whatever field text is still unsent.
	for _, name := range fieldOrder {
		text, ok := outputs[name].(string)
		if !ok || len(text) <= emitted[name] {
			continue
		}
		if err := handler(core.StreamChunk{Field: name, Content: text[emitted[name]:]}); err != nil {
			return nil, err
		}
	}
	if plan.usingNative {
		outputs[plan.nativeFieldName] = resp.ReasoningSummary
	}

	finish := lastFinish
	if finish == "" {
		finish = core.FinishReasonStop
	}
	if err := handler(core.StreamChunk{Done: true, FinishReason: finish, Usage: lastUsage}); err != nil {
		return nil, err
	}

	if a.Cache != nil && cacheKey != "" && (finish == core.FinishReasonStop || finish == core.FinishReasonToolCall) {
		a.Cache.set(ctx, cacheKey, cachedAdapterEntry{
			Outputs:           outputs,
			ToolCalls:         toolCalls,
			ReasoningSummary:  resp.ReasoningSummary,
			FinishReason:      finish,
			SchemaEnforcement: enforcement,
		})
	}

	return &core.AdapterResult{
		Outputs:           outputs,
		ToolCalls:         toolCalls,
		ReasoningSummary:  resp.ReasoningSummary,
		FinishReason:      finish,
		Usage:             lastUsage,
		AdapterName:       a.Name(),
		AdapterVersion:    a.Version(),
		SchemaEnforcement: enforcement,
		RawContent:        resp.Content,
	}, nil
}

func partialJSONFields(sig core.Signature, raw string) map[string]any {
	obj, ok := llm.ParseStreamingJSON(raw).(map[string]any)
	if !ok || len(obj) == 0 {
		return nil
	}
	partial := make(map[string]any)
	for _, f := range sig.Outputs {
		if val, exists := obj[f.Name]; exists {
			partial[f.Name] = val
		}
	}
	if len(partial) == 0 {
		return nil
	}
	return partial
}

// extractJSONObjectBytes strips an optional markdown code fence and locates a
// top-level JSON object slice without mutating valid JSON bytes.
// extractJSONObjectBytes returns the reply's JSON object without its code
// fence and surrounding prose, or the reason there is no single object to
// return. A reply with a second object, such as an example ahead of the real
// one, is rejected instead of guessed at.
func extractJSONObjectBytes(s string) (json.RawMessage, string) {
	cleaned := stripOptionalCodeFence(s, "json")
	trimmed := strings.TrimSpace(cleaned)
	if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
		return json.RawMessage(trimmed), ""
	}

	start, end, found := nextJSONObject(trimmed, 0)
	if !found {
		return nil, "no JSON object found in response"
	}
	if _, _, second := nextJSONObject(trimmed, end); second {
		return nil, "more than one JSON object found in response"
	}
	return json.RawMessage(trimmed[start:end]), ""
}

// nextJSONObject finds the first balanced {...} at or after from and returns
// its bounds.
func nextJSONObject(s string, from int) (start, end int, found bool) {
	offset := strings.IndexByte(s[from:], '{')
	if offset < 0 {
		return 0, 0, false
	}
	start = from + offset
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return start, i + 1, true
			}
		}
	}
	return 0, 0, false
}

func stripOptionalCodeFence(s string, lang string) string {
	trimmed := strings.TrimSpace(s)
	if !strings.HasPrefix(trimmed, "```") {
		return trimmed
	}
	nl := strings.IndexByte(trimmed, '\n')
	if nl < 0 {
		return trimmed
	}
	header := strings.TrimSpace(strings.TrimPrefix(trimmed[:nl], "```"))
	if header != "" && !strings.EqualFold(header, lang) {
		return trimmed
	}
	body := trimmed[nl+1:]
	if end := strings.LastIndex(body, "```"); end >= 0 {
		return strings.TrimSpace(body[:end])
	}
	return trimmed
}

// isParseError reports whether err is a ParseError eligible for fallback.
func isParseError(err error) bool {
	var pe *dspyerrors.ParseError
	return stderrors.As(err, &pe)
}
