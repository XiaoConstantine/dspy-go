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
)

// XMLAdapter is a structured-output adapter that asks for each output field
// between <field> tags. Choose it with Predict.WithAdapter.
// It uses a literal tag scanner (not encoding/xml), so string field values may
// contain arbitrary Go code, HTML, or operators like `<`, `>`, `&`, `<-`
// without escaping. On ParseError (before any streaming output is emitted), it
// falls back once to JSONAdapter.
type XMLAdapter struct {
	UseJSONFallback bool
	Cache           *AdapterCache
}

var _ core.Adapter = (*XMLAdapter)(nil)

// NewXMLAdapter creates an XMLAdapter with the one-shot JSON fallback enabled.
func NewXMLAdapter() *XMLAdapter {
	return &XMLAdapter{UseJSONFallback: true}
}

// WithCache attaches a validated-output cache to the adapter.
func (a *XMLAdapter) WithCache(c *AdapterCache) *XMLAdapter {
	a.Cache = c
	return a
}

// WithJSONFallback enables or disables the one-shot fallback to JSONAdapter on ParseError.
func (a *XMLAdapter) WithJSONFallback(enabled bool) *XMLAdapter {
	a.UseJSONFallback = enabled
	return a
}

func (a *XMLAdapter) Name() string    { return "xml" }
func (a *XMLAdapter) Version() string { return "v1" }

// Format renders signature, demos, conversation history, and inputs into
// canonical chat messages with XML-tag delimiters.
func (a *XMLAdapter) Format(signature core.Signature, demos []core.Example, inputs map[string]any) ([]core.ChatMessage, error) {
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

func (a *XMLAdapter) formatSystemMessage(sig core.Signature) string {
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
		hasOptional := false
		hasNonScalar := false
		for i, f := range sig.Outputs {
			if f.Optional {
				hasOptional = true
			}
			if schema.IsNonScalarField(f) {
				hasNonScalar = true
			}
			fmt.Fprintf(&sb, "%d. `%s`%s", i+1, f.Name, fieldTypeHint(f))
			if f.Description != "" {
				fmt.Fprintf(&sb, ": %s", f.Description)
			}
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
		sb.WriteString("Output format (replace placeholders):\n")
		for i, f := range sig.Outputs {
			if i > 0 {
				sb.WriteString("\n\n")
			}
			fmt.Fprintf(&sb, "<%s>\n{%s}\n</%s>", f.Name, f.Name, f.Name)
		}
		sb.WriteString("\n\nPut field tags on separate lines. Write string values literally, without XML or JSON escaping.")
		if hasOptional {
			sb.WriteString(" For null on an optional field, self-close the tag as <field />. For an empty string, use <field></field>.")
		}
		if hasNonScalar {
			sb.WriteString(" For structured object or array fields, write valid JSON inside the field's tags.")
		}
		sb.WriteString("\n\n")
	}

	if sig.Instruction != "" {
		sb.WriteString("Task instructions:\n")
		sb.WriteString(sig.Instruction)
	}
	return strings.TrimSpace(sb.String())
}

func (a *XMLAdapter) userOutputReminder(sig core.Signature) string {
	if len(sig.Outputs) == 0 {
		return ""
	}
	tags := make([]string, len(sig.Outputs))
	hasOptional := false
	for i, f := range sig.Outputs {
		tags[i] = fmt.Sprintf("<%s>", f.Name)
		if f.Optional {
			hasOptional = true
		}
	}
	msg := "Respond using these XML tags in order: " + strings.Join(tags, ", ") +
		". Put field tags on separate lines and write string values literally without XML escaping."
	if hasOptional {
		msg += " Use <field /> for null and <field></field> for an empty string."
	}
	return msg
}

func (a *XMLAdapter) formatDemoMessages(sig core.Signature, demo core.Example) []core.ChatMessage {
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

	var assistantBuf strings.Builder
	first := true
	for _, f := range sig.Outputs {
		val, exists := demo.Outputs[f.Name]
		if !exists {
			if f.Optional {
				val = nil
			} else {
				continue
			}
		}
		if !first {
			assistantBuf.WriteString("\n\n")
		}
		first = false
		if val == nil {
			fmt.Fprintf(&assistantBuf, "<%s />", f.Name)
			continue
		}
		var formatted string
		if schema.IsNonScalarField(f) {
			if encoded, err := json.Marshal(val); err == nil {
				formatted = string(encoded)
			} else {
				formatted = fmt.Sprint(val)
			}
		} else {
			formatted = fmt.Sprint(val)
		}
		fmt.Fprintf(&assistantBuf, "<%s>\n%s\n</%s>", f.Name, formatted, f.Name)
	}

	return []core.ChatMessage{
		{Role: "user", Content: userBlocks},
		{Role: "assistant", Content: []core.ContentBlock{core.NewTextBlock(assistantBuf.String())}},
	}
}

// Parse parses and validates XML-tagged fields using a literal tag scanner
// (never encoding/xml), then validates the resulting object against the
// compiled JSON Schema.
func (a *XMLAdapter) Parse(signature core.Signature, content string) (map[string]any, error) {
	if len(signature.Outputs) == 0 {
		return map[string]any{}, nil
	}
	if strings.TrimSpace(content) == "" {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: "model returned an empty response",
			Raw:     content,
		}
	}

	rawFields, err := scanXMLFields(signature, content)
	if err != nil {
		var pe *dspyerrors.ParseError
		if errAsParseError(err, &pe) {
			pe.Adapter = a.Name()
			pe.Raw = content
			return nil, pe
		}
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: err.Error(),
			Raw:     content,
		}
	}

	return a.coerceAndValidate(signature, rawFields, content)
}

func (a *XMLAdapter) coerceAndValidate(
	signature core.Signature,
	rawFields map[string]xmlFieldCapture,
	rawContent string,
) (map[string]any, error) {
	compiled, err := schema.CompileSignatureSchema(signature)
	if err != nil {
		return nil, fmt.Errorf("compile signature schema: %w", err)
	}

	outputs := make(map[string]any, len(signature.Outputs))
	jsonValues := make(map[string]any, len(signature.Outputs))
	var missing []string
	invalid := make(map[string]string)

	for _, field := range signature.Outputs {
		capVal, found := rawFields[field.Name]
		if !found {
			if field.IsRequired() {
				missing = append(missing, field.Name)
			} else {
				outputs[field.Name] = nil
				jsonValues[field.Name] = nil
			}
			continue
		}

		if capVal.isNull {
			if field.IsRequired() {
				invalid[field.Name] = "required field cannot be null (<" + field.Name + " />)"
				continue
			}
			outputs[field.Name] = nil
			jsonValues[field.Name] = nil
			continue
		}

		if schema.IsNonScalarField(field) {
			inner := strings.TrimSpace(stripOptionalCodeFence(capVal.text, "json"))
			fieldSchema, _, sErr := schema.CompileFieldSchema(field)
			if sErr != nil {
				invalid[field.Name] = sErr.Error()
				continue
			}
			if vErr := schema.ValidateJSON(fieldSchema, json.RawMessage(inner)); vErr != nil {
				invalid[field.Name] = fmt.Sprintf("invalid JSON value: %v", vErr)
				continue
			}
			coerced, cErr := schema.CoerceFieldValue(field, inner)
			if cErr != nil {
				invalid[field.Name] = cErr.Error()
				continue
			}
			var decodedForSchema any
			_ = json.Unmarshal([]byte(inner), &decodedForSchema)
			outputs[field.Name] = coerced
			jsonValues[field.Name] = decodedForSchema
			continue
		}

		coerced, cErr := schema.CoerceFieldValue(field, capVal.text)
		if cErr != nil {
			invalid[field.Name] = cErr.Error()
			continue
		}
		outputs[field.Name] = coerced
		jsonValues[field.Name] = coerced
	}

	if len(missing) > 0 || len(invalid) > 0 {
		if len(invalid) == 0 {
			invalid = nil
		}
		return nil, &dspyerrors.ParseError{
			Adapter:       a.Name(),
			MissingFields: missing,
			InvalidFields: invalid,
			Raw:           rawContent,
		}
	}

	// Validate complete output object locally against the compiled JSON Schema.
	orderedRaw, err := schema.MarshalOrderedObject(compiled.PropertyOrder, jsonValues)
	if err != nil {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: "failed to serialize parsed fields for schema validation",
			Raw:     rawContent,
			Err:     err,
		}
	}
	if err := schema.ValidateJSON(compiled.Raw, orderedRaw); err != nil {
		return nil, &dspyerrors.ParseError{
			Adapter: a.Name(),
			Message: "parsed output failed schema validation",
			Raw:     rawContent,
			Err:     err,
		}
	}

	return outputs, nil
}

// Call runs the XMLAdapter pipeline: native reasoning resolution, cache lookup,
// sync or streaming model call, strict literal XML parsing and schema validation,
// and optional one-shot fallback to JSONAdapter on ParseError (never after
// emitting streaming output, and never on refusal/truncation/content filter).
func (a *XMLAdapter) Call(
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

	req := core.ChatRequest{
		Messages:              messages,
		Tools:                 opts.Tools,
		ToolChoice:            opts.ToolChoice,
		ParallelToolCalls:     opts.ParallelToolCalls,
		ResponseFormat:        core.ResponseFormatText,
		ReasoningEffort:       plan.reasoningEffort,
		ReasoningBudgetTokens: opts.ReasoningBudgetTokens,
		Options:               opts.GenerateOptions,
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

	// spent is the usage of the XML call, also when its reply is rejected.
	var res *core.AdapterResult
	var spent *core.TokenInfo
	if opts.StreamHandler != nil {
		res, err = a.callStreaming(ctx, model, signature, plan, req, cacheKey, opts.StreamHandler, &spent)
	} else {
		res, err = a.callSync(ctx, model, plan, req, &spent)
	}

	if err != nil {
		// A StreamParseError means the handler already received output, so a
		// second reply must not follow it.
		var streamed *dspyerrors.StreamParseError
		if !a.UseJSONFallback || !isParseError(err) || stderrors.As(err, &streamed) {
			recordRejectedUsage(ctx, spent)
			return nil, err
		}
		jsonAdapter := NewJSONAdapter().WithCache(a.Cache)
		fallbackOpts := opts
		fallbackOpts.DisableCache = true
		res, fallbackErr := jsonAdapter.Call(ctx, model, signature, demos, inputs, fallbackOpts)
		if fallbackErr != nil {
			recordRejectedUsage(ctx, spent)
			return nil, fallbackFailure(err, fallbackErr)
		}
		res.FallbackUsed = true
		res.Usage = addUsage(spent, res.Usage)
		a.store(ctx, cacheKey, res)
		return res, nil
	}

	a.store(ctx, cacheKey, res)
	return res, nil
}

// store caches a validated result that ended normally.
func (a *XMLAdapter) store(ctx context.Context, cacheKey string, res *core.AdapterResult) {
	if a.Cache == nil || cacheKey == "" {
		return
	}
	switch res.FinishReason {
	case "", core.FinishReasonStop, core.FinishReasonToolCall:
	default:
		return
	}
	a.Cache.set(ctx, cacheKey, cachedAdapterEntry{
		Outputs:           res.Outputs,
		ToolCalls:         res.ToolCalls,
		ReasoningSummary:  res.ReasoningSummary,
		FinishReason:      res.FinishReason,
		SchemaEnforcement: res.SchemaEnforcement,
		FallbackUsed:      res.FallbackUsed,
	})
}

// fallbackFailure is the error returned when the JSON fallback fails too. When
// both replies failed to parse, it keeps the first failure, which names the
// fields the XML reply was missing, and attaches the fallback's. Any other
// fallback failure, such as a refusal or a transport error, is returned as is.
func fallbackFailure(primary, fallback error) error {
	var primaryParse, fallbackParse *dspyerrors.ParseError
	if !stderrors.As(primary, &primaryParse) || !stderrors.As(fallback, &fallbackParse) {
		return fallback
	}
	combined := *primaryParse
	combined.Err = fmt.Errorf("JSON fallback also failed: %w", fallback)
	if primaryParse.Err != nil {
		combined.Err = fmt.Errorf("%w; JSON fallback also failed: %w", primaryParse.Err, fallback)
	}
	return &combined
}

func (a *XMLAdapter) callSync(
	ctx context.Context,
	model core.LLM,
	plan reasoningPlan,
	req core.ChatRequest,
	spent **core.TokenInfo,
) (*core.AdapterResult, error) {
	resp, err := executeChat(ctx, model, req)
	if err != nil {
		return nil, classifyModelError(a.Name(), err)
	}
	*spent = resp.Usage
	if err := checkResponseStatus(resp, func(raw string) map[string]any {
		return partialXMLFields(plan.wireSignature, raw)
	}); err != nil {
		return nil, err
	}

	outputs, err := a.Parse(plan.wireSignature, resp.Content)
	if err != nil {
		return nil, err
	}
	if plan.usingNative {
		outputs[plan.nativeFieldName] = resp.ReasoningSummary
	}

	return &core.AdapterResult{
		Outputs:          outputs,
		ToolCalls:        resp.ToolCalls,
		ReasoningSummary: resp.ReasoningSummary,
		FinishReason:     resp.FinishReason,
		Usage:            resp.Usage,
		AdapterName:      a.Name(),
		AdapterVersion:   a.Version(),
		RawContent:       resp.Content,
	}, nil
}

func (a *XMLAdapter) callStreaming(
	ctx context.Context,
	model core.LLM,
	originalSig core.Signature,
	plan reasoningPlan,
	req core.ChatRequest,
	cacheKey string,
	handler core.StreamHandler,
	spent **core.TokenInfo,
) (*core.AdapterResult, error) {
	stream, err := executeStreamChat(ctx, model, req)
	if err != nil {
		return nil, classifyModelError(a.Name(), err)
	}
	defer stream.Cancel()

	scanner := newXMLStreamScanner(plan.wireSignature)
	var fullRaw strings.Builder
	var reasoningBuf strings.Builder
	var lastUsage *core.TokenInfo
	defer func() { *spent = lastUsage }()
	var lastFinish core.FinishReason
	var lastRefusal *string
	var toolCalls []core.ToolCall
	anyEmitted := false

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
					PartialFields: partialXMLFields(plan.wireSignature, fullRaw.String()),
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
			deltas := scanner.feed(chunk.Content, false)
			for _, d := range deltas {
				if d.content == "" {
					continue
				}
				anyEmitted = true
				if err := handler(core.StreamChunk{Field: d.field, Content: d.content}); err != nil {
					return nil, err
				}
			}
		}
		if chunk.Done {
			break
		}
	}

	// Flush remaining buffered field deltas only if stream completed cleanly.
	resp := &core.ChatResponse{
		Content:          fullRaw.String(),
		ReasoningSummary: reasoningBuf.String(),
		FinishReason:     lastFinish,
		Refusal:          lastRefusal,
		ToolCalls:        toolCalls,
		Usage:            lastUsage,
	}
	if statusErr := checkResponseStatus(resp, func(raw string) map[string]any {
		return partialXMLFields(plan.wireSignature, raw)
	}); statusErr != nil {
		if anyEmitted {
			return nil, &dspyerrors.StreamParseError{
				PartialFields: partialXMLFields(plan.wireSignature, resp.Content),
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
				PartialFields: partialXMLFields(plan.wireSignature, resp.Content),
				FinishReason:  string(lastFinish),
				Raw:           resp.Content,
				Err:           parseErr,
			}
		}
		return nil, parseErr
	}

	for _, d := range scanner.feed("", true) {
		if d.content == "" {
			continue
		}
		if err := handler(core.StreamChunk{Field: d.field, Content: d.content}); err != nil {
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

	return &core.AdapterResult{
		Outputs:          outputs,
		ToolCalls:        toolCalls,
		ReasoningSummary: resp.ReasoningSummary,
		FinishReason:     finish,
		Usage:            lastUsage,
		AdapterName:      a.Name(),
		AdapterVersion:   a.Version(),
		RawContent:       resp.Content,
	}, nil
}

// --- Literal XML tag scanner ---

type xmlFieldCapture struct {
	text   string
	isNull bool
}

func scanXMLFields(sig core.Signature, content string) (map[string]xmlFieldCapture, error) {
	declared := make(map[string]core.OutputField, len(sig.Outputs))
	for _, f := range sig.Outputs {
		declared[f.Name] = f
	}

	text, err := stripXMLPrefixAndFence(content, declared)
	if err != nil {
		return nil, err
	}

	// Skip introductory prose before the first declared opening tag.
	startIdx := findFirstDeclaredOpeningTag(text, declared)
	if startIdx < 0 {
		return nil, &dspyerrors.ParseError{Message: "no declared XML output tags found"}
	}
	// Prose before the first tag is accepted only while it leaves no doubt
	// about where the first field starts.
	afterProse := strings.TrimSpace(text[:startIdx]) != ""
	text = text[startIdx:]

	results := make(map[string]xmlFieldCapture, len(sig.Outputs))
	pos := 0

	for pos < len(text) {
		pos = skipASCIIWhitespace(text, pos)
		if pos >= len(text) {
			break
		}
		if text[pos] != '<' {
			// Trailing sign-off prose after all fields are parsed is ignored.
			break
		}

		tagName, selfClosing, tagEnd, ok := parseOpeningTagAt(text, pos)
		if !ok {
			return nil, &dspyerrors.ParseError{
				Message: fmt.Sprintf("malformed tag at offset %d", pos),
			}
		}
		if _, known := declared[tagName]; !known {
			return nil, &dspyerrors.ParseError{
				Message: fmt.Sprintf("unexpected output tag <%s>", tagName),
			}
		}
		if _, dup := results[tagName]; dup {
			return nil, &dspyerrors.ParseError{
				Message: fmt.Sprintf("duplicate output tag <%s>", tagName),
			}
		}

		pos = tagEnd
		if selfClosing {
			results[tagName] = xmlFieldCapture{isNull: true}
			continue
		}

		// Find matching closing tag </tagName> using boundary lookahead so
		// literal `</tagName>` inside code/HTML values does not prematurely close.
		closeStart, closeEnd, found := findClosingTag(text, pos, tagName, declared)
		if !found {
			return nil, &dspyerrors.ParseError{
				Message: fmt.Sprintf("missing closing tag </%s>", tagName),
			}
		}

		body := trimSingleBoundaryNewlines(text[pos:closeStart])
		if afterProse && len(results) == 0 && containsOpeningTag(body, tagName) {
			// "I'll use <answer> tags. <answer>42</answer>": the prose named
			// the tag, and the value would swallow the real one.
			return nil, &dspyerrors.ParseError{
				Message: fmt.Sprintf("ambiguous output: <%s> appears again inside its value after introductory text", tagName),
			}
		}
		results[tagName] = xmlFieldCapture{text: body}
		pos = closeEnd
	}

	return results, nil
}

// containsOpeningTag reports whether s holds an opening tag named name.
func containsOpeningTag(s, name string) bool {
	for pos := strings.IndexByte(s, '<'); pos >= 0 && pos < len(s); {
		if tagName, _, _, ok := parseOpeningTagAt(s, pos); ok && tagName == name {
			return true
		}
		next := strings.IndexByte(s[pos+1:], '<')
		if next < 0 {
			return false
		}
		pos += 1 + next
	}
	return false
}

// partialXMLFields extracts any complete or in-progress declared tags from a
// truncated reply for TruncatedError.PartialFields / StreamParseError.
func partialXMLFields(sig core.Signature, content string) map[string]any {
	declared := make(map[string]core.OutputField, len(sig.Outputs))
	for _, f := range sig.Outputs {
		declared[f.Name] = f
	}
	text, err := stripXMLPrefixAndFence(content, declared)
	if err != nil {
		text = content
	}
	startIdx := findFirstDeclaredOpeningTag(text, declared)
	if startIdx < 0 {
		return nil
	}
	text = text[startIdx:]
	out := make(map[string]any)
	pos := 0
	for pos < len(text) {
		pos = skipASCIIWhitespace(text, pos)
		if pos >= len(text) || text[pos] != '<' {
			break
		}
		tagName, selfClosing, tagEnd, ok := parseOpeningTagAt(text, pos)
		if !ok {
			break
		}
		if _, known := declared[tagName]; !known {
			break
		}
		pos = tagEnd
		if selfClosing {
			out[tagName] = nil
			continue
		}
		closeStart, closeEnd, found := findClosingTag(text, pos, tagName, declared)
		if !found {
			out[tagName] = trimSingleBoundaryNewlines(text[pos:])
			break
		}
		out[tagName] = trimSingleBoundaryNewlines(text[pos:closeStart])
		pos = closeEnd
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func stripXMLPrefixAndFence(content string, declared map[string]core.OutputField) (string, error) {
	s := strings.TrimSpace(content)
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		// Strip leading <think>...</think> or <thinking>...</thinking> unless declared.
		strippedThought := false
		for _, thoughtTag := range []string{"think", "thinking"} {
			if _, isField := declared[thoughtTag]; isField {
				continue
			}
			open := "<" + thoughtTag + ">"
			if strings.HasPrefix(s, open) {
				closeTag := "</" + thoughtTag + ">"
				idx := strings.Index(s[len(open):], closeTag)
				if idx < 0 {
					return "", &dspyerrors.ParseError{Message: "unclosed leading <" + thoughtTag + "> block"}
				}
				s = s[len(open)+idx+len(closeTag):]
				strippedThought = true
				break
			}
		}
		if strippedThought {
			continue
		}

		// Strip optional matching <response>...</response> root wrapper when "response" is not a declared field.
		if _, isField := declared["response"]; !isField {
			if strings.HasPrefix(s, "<response>") && strings.HasSuffix(s, "</response>") {
				s = strings.TrimSpace(s[len("<response>") : len(s)-len("</response>")])
				continue
			}
		}

		// Strip optional ```xml or ``` fence around the whole response.
		if strings.HasPrefix(s, "```") {
			nl := strings.IndexByte(s, '\n')
			if nl >= 0 {
				header := strings.TrimSpace(strings.TrimPrefix(s[:nl], "```"))
				if header == "" || strings.EqualFold(header, "xml") {
					body := s[nl+1:]
					end := strings.LastIndex(body, "```")
					if end < 0 {
						return "", &dspyerrors.ParseError{Message: "missing closing XML code fence"}
					}
					s = strings.TrimSpace(body[:end])
					continue
				}
			}
		}
		break
	}
	return s, nil
}

func findFirstDeclaredOpeningTag(text string, declared map[string]core.OutputField) int {
	for i := 0; i < len(text); i++ {
		if text[i] != '<' {
			continue
		}
		name, _, _, ok := parseOpeningTagAt(text, i)
		if ok {
			if _, known := declared[name]; known {
				return i
			}
		}
	}
	return -1
}

// parseOpeningTagAt parses `<name>` or `<name />` starting at pos.
func parseOpeningTagAt(s string, pos int) (name string, selfClosing bool, end int, ok bool) {
	if pos >= len(s) || s[pos] != '<' {
		return "", false, 0, false
	}
	i := pos + 1
	if i >= len(s) || s[i] == '/' || !isTagNameStart(s[i]) {
		return "", false, 0, false
	}
	nameStart := i
	for i < len(s) && isTagNameChar(s[i]) {
		i++
	}
	name = s[nameStart:i]
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i < len(s) && s[i] == '/' {
		selfClosing = true
		i++
	}
	if i >= len(s) || s[i] != '>' {
		return "", false, 0, false
	}
	return name, selfClosing, i + 1, true
}

func parseClosingTagAt(s string, pos int, wantName string) (end int, ok bool) {
	if pos+len(wantName)+3 > len(s) || s[pos] != '<' || s[pos+1] != '/' {
		return 0, false
	}
	i := pos + 2
	if !strings.HasPrefix(s[i:], wantName) {
		return 0, false
	}
	i += len(wantName)
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) || s[i] != '>' {
		return 0, false
	}
	return i + 1, true
}

// findClosingTag locates the closing `</tagName>` for a literal field body.
// If multiple `</tagName>` occurrences exist (e.g., code discussing XML tags),
// it prefers the candidate followed by whitespace + a declared opening tag or
// EOF; otherwise it takes the last valid closing tag before EOF.
func findClosingTag(text string, from int, tagName string, declared map[string]core.OutputField) (int, int, bool) {
	needle := "</" + tagName
	searchFrom := from
	lastStart, lastEnd := -1, -1

	for searchFrom < len(text) {
		rel := strings.Index(text[searchFrom:], needle)
		if rel < 0 {
			break
		}
		candStart := searchFrom + rel
		candEnd, ok := parseClosingTagAt(text, candStart, tagName)
		if !ok {
			searchFrom = candStart + 2
			continue
		}
		lastStart, lastEnd = candStart, candEnd

		after := skipASCIIWhitespace(text, candEnd)
		if after >= len(text) {
			return candStart, candEnd, true
		}
		if nextName, _, _, nextOK := parseOpeningTagAt(text, after); nextOK {
			if _, known := declared[nextName]; known {
				return candStart, candEnd, true
			}
		}
		searchFrom = candEnd
	}

	if lastStart >= 0 {
		return lastStart, lastEnd, true
	}
	return 0, 0, false
}

func trimSingleBoundaryNewlines(s string) string {
	if strings.HasPrefix(s, "\r\n") {
		s = s[2:]
	} else if strings.HasPrefix(s, "\n") {
		s = s[1:]
	}
	if strings.HasSuffix(s, "\r\n") {
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "\n") {
		s = s[:len(s)-1]
	}
	return s
}

func skipASCIIWhitespace(s string, pos int) int {
	for pos < len(s) {
		switch s[pos] {
		case ' ', '\t', '\r', '\n':
			pos++
		default:
			return pos
		}
	}
	return pos
}

func isTagNameStart(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_'
}

func isTagNameChar(b byte) bool {
	return isTagNameStart(b) || (b >= '0' && b <= '9') || b == '-' || b == '.'
}

func errAsParseError(err error, target **dspyerrors.ParseError) bool {
	pe, ok := err.(*dspyerrors.ParseError)
	if ok {
		*target = pe
	}
	return ok
}

// --- Incremental literal XML stream scanner ---

type xmlStreamDelta struct {
	field   string
	content string
}

type xmlStreamScanner struct {
	declared     map[string]core.OutputField
	buf          string
	inField      string
	strippedInit bool
	inThought    string
	fenced       bool
}

func newXMLStreamScanner(sig core.Signature) *xmlStreamScanner {
	declared := make(map[string]core.OutputField, len(sig.Outputs))
	for _, f := range sig.Outputs {
		declared[f.Name] = f
	}
	return &xmlStreamScanner{declared: declared}
}

func (s *xmlStreamScanner) feed(chunk string, final bool) []xmlStreamDelta {
	s.buf += chunk
	var deltas []xmlStreamDelta

	for {
		if s.inField == "" {
			// Skip leading <think>...</think> if currently inside one.
			if s.inThought != "" {
				closeTag := "</" + s.inThought + ">"
				idx := strings.Index(s.buf, closeTag)
				if idx < 0 {
					if !final && len(s.buf) > len(closeTag) {
						s.buf = s.buf[len(s.buf)-(len(closeTag)-1):]
					}
					break
				}
				s.buf = s.buf[idx+len(closeTag):]
				s.inThought = ""
				continue
			}

			trimmed := strings.TrimLeft(s.buf, " \t\r\n")
			if trimmed == "" {
				s.buf = ""
				break
			}

			// Detect leading <think> / <thinking>.
			startedThought := false
			for _, thoughtTag := range []string{"think", "thinking"} {
				if _, isField := s.declared[thoughtTag]; isField {
					continue
				}
				open := "<" + thoughtTag + ">"
				if strings.HasPrefix(trimmed, open) {
					s.buf = trimmed[len(open):]
					s.inThought = thoughtTag
					startedThought = true
					break
				}
				if !final && strings.HasPrefix(open, trimmed) {
					return deltas
				}
			}
			if startedThought {
				continue
			}

			// Strip optional opening ```xml\n fence.
			if !s.fenced && strings.HasPrefix(trimmed, "```") {
				nl := strings.IndexByte(trimmed, '\n')
				if nl < 0 {
					if !final {
						return deltas
					}
				} else {
					s.fenced = true
					s.buf = trimmed[nl+1:]
					continue
				}
			}

			// Look for the next declared opening tag.
			idx := findFirstDeclaredOpeningTag(s.buf, s.declared)
			if idx < 0 {
				// Keep only a trailing suffix that could start a tag (`<...`)
				// so prose before the first tag is never emitted.
				if !final {
					if lt := strings.LastIndexByte(s.buf, '<'); lt >= 0 {
						s.buf = s.buf[lt:]
					} else {
						s.buf = ""
					}
				}
				break
			}

			name, selfClosing, end, ok := parseOpeningTagAt(s.buf, idx)
			if !ok {
				break
			}
			s.buf = s.buf[end:]
			if selfClosing {
				continue
			}
			s.inField = name
			s.strippedInit = false
			continue
		}

		// Inside a field: strip the single leading newline right after <field>.
		if !s.strippedInit {
			if s.buf == "" {
				break
			}
			if s.buf == "\r" && !final {
				break
			}
			if strings.HasPrefix(s.buf, "\r\n") {
				s.buf = s.buf[2:]
			} else if strings.HasPrefix(s.buf, "\n") {
				s.buf = s.buf[1:]
			}
			s.strippedInit = true
		}

		// Look for closing tag `</inField>`.
		closeNeedle := "</" + s.inField
		if rel := strings.Index(s.buf, closeNeedle); rel >= 0 {
			if closeEnd, ok := parseClosingTagAt(s.buf, rel, s.inField); ok {
				emitText := s.buf[:rel]
				if strings.HasSuffix(emitText, "\r\n") {
					emitText = emitText[:len(emitText)-2]
				} else if strings.HasSuffix(emitText, "\n") {
					emitText = emitText[:len(emitText)-1]
				}
				if emitText != "" {
					deltas = append(deltas, xmlStreamDelta{field: s.inField, content: emitText})
				}
				s.buf = s.buf[closeEnd:]
				s.inField = ""
				continue
			}
		}

		if final {
			break
		}

		// Hold back up to len("</field>") + 2 bytes (for optional `\r\n` + partial closing tag).
		holdBack := len(s.inField) + 5
		if len(s.buf) <= holdBack {
			break
		}
		emitLen := len(s.buf) - holdBack
		// Do not split a multi-byte UTF-8 rune.
		for emitLen > 0 && (s.buf[emitLen]&0xC0) == 0x80 {
			emitLen--
		}
		if emitLen == 0 {
			break
		}
		deltas = append(deltas, xmlStreamDelta{field: s.inField, content: s.buf[:emitLen]})
		s.buf = s.buf[emitLen:]
		break
	}

	return deltas
}
