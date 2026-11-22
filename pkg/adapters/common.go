package adapters

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"maps"
	"reflect"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	llm "github.com/XiaoConstantine/llm-go"
)

// reasoningPlan describes call-time native reasoning resolution for a signature.
type reasoningPlan struct {
	wireSignature   core.Signature
	nativeFieldName string
	reasoningEffort core.ReasoningEffort
	usingNative     bool
}

// resolveNativeReasoning decides at call time whether a NativeReasoning output
// field is satisfied by model-native reasoning or emitted explicitly first.
func resolveNativeReasoning(
	model core.LLM,
	sig core.Signature,
	opts core.AdapterCallOptions,
) (reasoningPlan, error) {
	plan := reasoningPlan{
		wireSignature:   sig,
		reasoningEffort: opts.ReasoningEffort,
	}

	nativeIdx := -1
	for i, field := range sig.Outputs {
		if field.NativeReasoning {
			nativeIdx = i
			break
		}
	}
	if nativeIdx == -1 {
		return plan, nil
	}

	// Respect explicit disabling or models without native reasoning support.
	if opts.ReasoningEffort == core.ReasoningEffortNone || !core.HasCapability(model, core.CapabilityReasoning) {
		return plan, nil
	}

	// Anthropic rejects temperature combined with extended thinking.
	if strings.EqualFold(model.ProviderName(), "anthropic") && hasExplicitTemperature(opts.GenerateOptions) {
		return reasoningPlan{}, dspyerrors.New(
			dspyerrors.ConfigurationError,
			"temperature cannot be combined with native reasoning on anthropic; omit temperature or disable reasoning effort",
		)
	}

	// Strip the native reasoning field from the wire signature and enable reasoning.
	wireOutputs := make([]core.OutputField, 0, len(sig.Outputs)-1)
	wireOutputs = append(wireOutputs, sig.Outputs[:nativeIdx]...)
	wireOutputs = append(wireOutputs, sig.Outputs[nativeIdx+1:]...)

	plan.wireSignature = sig
	plan.wireSignature.Outputs = wireOutputs
	plan.nativeFieldName = sig.Outputs[nativeIdx].Name
	plan.usingNative = true
	if plan.reasoningEffort == core.ReasoningEffortDefault && opts.ReasoningBudgetTokens == 0 {
		plan.reasoningEffort = core.ReasoningEffortLow
	}
	return plan, nil
}

func hasExplicitTemperature(opts []core.GenerateOption) bool {
	cfg := core.NewGenerateOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
	return cfg.TemperatureSet || cfg.Temperature != 0
}

// extractHistoryTurns checks for a core.History input field and returns the
// signature without that field plus the extracted history turns.
func extractHistoryTurns(sig core.Signature, inputs map[string]any) (core.Signature, []map[string]any, bool) {
	historyIdx := -1
	var history core.History
	for i, field := range sig.Inputs {
		val, exists := inputs[field.Name]
		if !exists {
			continue
		}
		switch h := val.(type) {
		case core.History:
			historyIdx = i
			history = h
		case *core.History:
			if h != nil {
				historyIdx = i
				history = *h
			}
		}
		if historyIdx != -1 {
			break
		}
	}
	if historyIdx == -1 {
		return sig, nil, false
	}
	stripped := sig
	stripped.Inputs = make([]core.InputField, 0, len(sig.Inputs)-1)
	stripped.Inputs = append(stripped.Inputs, sig.Inputs[:historyIdx]...)
	stripped.Inputs = append(stripped.Inputs, sig.Inputs[historyIdx+1:]...)
	return stripped, history.Messages, true
}

// buildInputContentBlocks renders input fields into multimodal ContentBlocks,
// interleaving text tags and image/audio blocks cleanly.
func buildInputContentBlocks(
	sig core.Signature,
	inputs map[string]any,
	formatTextField func(field core.InputField, value string) string,
	formatMediaLabel func(field core.InputField) string,
	suffix string,
) []core.ContentBlock {
	var blocks []core.ContentBlock
	var textBuf strings.Builder

	flushText := func() {
		if textBuf.Len() > 0 {
			blocks = append(blocks, core.NewTextBlock(textBuf.String()))
			textBuf.Reset()
		}
	}

	first := true
	for _, field := range sig.Inputs {
		val, exists := inputs[field.Name]
		if !exists {
			continue
		}
		if !first {
			textBuf.WriteString("\n\n")
		}
		first = false

		if cb, ok := val.(core.ContentBlock); ok && cb.Type != core.FieldTypeText {
			textBuf.WriteString(formatMediaLabel(field))
			flushText()
			blocks = append(blocks, cb)
			continue
		}
		if cbPtr, ok := val.(*core.ContentBlock); ok && cbPtr != nil && cbPtr.Type != core.FieldTypeText {
			textBuf.WriteString(formatMediaLabel(field))
			flushText()
			blocks = append(blocks, *cbPtr)
			continue
		}

		textBuf.WriteString(formatTextField(field, stringifyInputValue(val)))
	}

	if suffix != "" {
		if textBuf.Len() > 0 || len(blocks) > 0 {
			textBuf.WriteString("\n\n")
		}
		textBuf.WriteString(suffix)
	}
	flushText()
	if len(blocks) == 0 {
		blocks = append(blocks, core.NewTextBlock(""))
	}
	return blocks
}

func stringifyInputValue(val any) string {
	if val == nil {
		return ""
	}
	if s, ok := val.(string); ok {
		return s
	}
	if cb, ok := val.(core.ContentBlock); ok {
		return cb.String()
	}
	return fmt.Sprintf("%v", val)
}

func fieldTypeHint(field core.OutputField) string {
	var hints []string
	if field.GoType != nil {
		t := field.GoType
		for t.Kind() == reflect.Pointer {
			t = t.Elem()
		}
		switch t.Kind() {
		case reflect.String:
		case reflect.Bool:
			hints = append(hints, "boolean: true or false")
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			hints = append(hints, "integer")
		case reflect.Float32, reflect.Float64:
			hints = append(hints, "number")
		case reflect.Slice, reflect.Array:
			if t.Elem().Kind() != reflect.Uint8 {
				hints = append(hints, "JSON array")
			}
		case reflect.Struct, reflect.Map:
			hints = append(hints, "JSON object")
		}
	}
	if enum := field.EnumValues(); len(enum) > 0 {
		hints = append(hints, "one of: "+strings.Join(enum, ", "))
	}
	if field.Optional || (field.GoType != nil && field.GoType.Kind() == reflect.Pointer) {
		hints = append(hints, "optional")
	}
	if len(hints) == 0 {
		return ""
	}
	return " (" + strings.Join(hints, "; ") + ")"
}

// executeChat invokes ChatLLM when available, or flattens messages for custom
// legacy core.LLM implementations so everything still flows through the same
// adapter parsing and validation.
func executeChat(ctx context.Context, model core.LLM, req core.ChatRequest) (*core.ChatResponse, error) {
	ctx = core.WithoutReplyCache(ctx)
	if chat, ok := core.ResolveChatLLM(model); ok {
		core.RecordLLMCall(ctx, model)
		return chat.Chat(ctx, req)
	}

	// Fallback for custom core.LLM implementations without ChatLLM.
	core.RecordLLMCall(ctx, model)
	blocks, isMultimodal := flattenChatMessages(req.Messages)
	if isMultimodal {
		resp, err := model.GenerateWithContent(ctx, blocks, req.Options...)
		if err != nil {
			return nil, err
		}
		return legacyLLMResponseToChat(resp), nil
	}
	prompt := concatTextBlocks(blocks)
	if req.ResponseFormat == core.ResponseFormatJSONObject && core.HasCapability(model, core.CapabilityJSON) {
		jsonMap, err := model.GenerateWithJSON(ctx, prompt, req.Options...)
		if err != nil {
			return nil, err
		}
		raw, err := json.Marshal(jsonMap)
		if err != nil {
			return nil, err
		}
		return &core.ChatResponse{
			Content:      string(raw),
			FinishReason: core.FinishReasonStop,
		}, nil
	}
	resp, err := model.Generate(ctx, prompt, req.Options...)
	if err != nil {
		return nil, err
	}
	return legacyLLMResponseToChat(resp), nil
}

func executeStreamChat(ctx context.Context, model core.LLM, req core.ChatRequest) (*core.StreamResponse, error) {
	ctx = core.WithoutReplyCache(ctx)
	if chat, ok := core.ResolveChatLLM(model); ok {
		core.RecordLLMCall(ctx, model)
		return chat.StreamChat(ctx, req)
	}

	core.RecordLLMCall(ctx, model)
	blocks, isMultimodal := flattenChatMessages(req.Messages)
	if isMultimodal {
		return model.StreamGenerateWithContent(ctx, blocks, req.Options...)
	}
	return model.StreamGenerate(ctx, concatTextBlocks(blocks), req.Options...)
}

func legacyLLMResponseToChat(resp *core.LLMResponse) *core.ChatResponse {
	if resp == nil {
		return &core.ChatResponse{FinishReason: core.FinishReasonStop}
	}
	finish := core.FinishReasonStop
	var reasoning string
	if resp.Metadata != nil {
		if fr, ok := resp.Metadata["finish_reason"].(string); ok && fr != "" {
			finish = core.FinishReason(fr)
		}
		if rs, ok := resp.Metadata["reasoning_summary"].(string); ok {
			reasoning = rs
		}
	}
	return &core.ChatResponse{
		Content:          resp.Content,
		FinishReason:     finish,
		ReasoningSummary: reasoning,
		Usage:            resp.Usage,
		Metadata:         resp.Metadata,
	}
}

func flattenChatMessages(messages []core.ChatMessage) ([]core.ContentBlock, bool) {
	var blocks []core.ContentBlock
	isMultimodal := false
	for i, msg := range messages {
		if i > 0 {
			blocks = append(blocks, core.NewTextBlock("\n\n"))
		}
		switch msg.Role {
		case "system":
		case "user":
			blocks = append(blocks, core.NewTextBlock("---\n"))
		case "assistant":
			blocks = append(blocks, core.NewTextBlock("Assistant:\n"))
		}
		for _, b := range msg.Content {
			if b.Type != core.FieldTypeText {
				isMultimodal = true
			}
			blocks = append(blocks, b)
		}
	}
	return blocks, isMultimodal
}

func concatTextBlocks(blocks []core.ContentBlock) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == core.FieldTypeText {
			sb.WriteString(b.Text)
		} else {
			sb.WriteString(b.String())
		}
	}
	return sb.String()
}

// checkResponseStatus enforces refusal, content_filter, and truncation errors
// before parsing. It returns nil when parsing should proceed.
func checkResponseStatus(
	resp *core.ChatResponse,
	partialExtractor func(content string) map[string]any,
) error {
	if resp == nil {
		return &dspyerrors.ParseError{Message: "empty nil response from model"}
	}
	if resp.Refusal != nil || resp.FinishReason == core.FinishReasonRefusal {
		reason := ""
		if resp.Refusal != nil {
			reason = *resp.Refusal
		}
		return &dspyerrors.RefusalError{Reason: reason}
	}
	if resp.FinishReason == core.FinishReasonContentFilter {
		return &dspyerrors.ContentFilterError{Raw: resp.Content}
	}
	if resp.FinishReason == core.FinishReasonLength {
		var partial map[string]any
		if partialExtractor != nil {
			partial = partialExtractor(resp.Content)
		}
		return &dspyerrors.TruncatedError{
			PartialFields: dspyerrors.ClonePartialFields(partial),
			Raw:           resp.Content,
		}
	}
	return nil
}

// classifyModelError converts llm-go's KindMalformedResponse into a ParseError
// so XMLAdapter fallback and non-retryable classification handle it cleanly.
func classifyModelError(adapterName string, err error) error {
	if err == nil {
		return nil
	}
	var llmErr *llm.Error
	if stderrors.As(err, &llmErr) && llmErr.Kind == llm.KindMalformedResponse {
		return &dspyerrors.ParseError{
			Adapter: adapterName,
			Message: "provider returned malformed response",
			Err:     err,
		}
	}
	return err
}

func replayCachedStream(
	sig core.Signature,
	entry *cachedAdapterEntry,
	handler core.StreamHandler,
) error {
	if handler == nil {
		return nil
	}
	if entry.ReasoningSummary != "" {
		if err := handler(core.StreamChunk{ReasoningSummary: entry.ReasoningSummary}); err != nil {
			return err
		}
	}
	for _, field := range sig.Outputs {
		val, ok := entry.Outputs[field.Name]
		if !ok || val == nil {
			continue
		}
		text := fmt.Sprint(val)
		if err := handler(core.StreamChunk{Field: field.Name, Content: text}); err != nil {
			return err
		}
	}
	finish := entry.FinishReason
	if finish == "" {
		finish = core.FinishReasonStop
	}
	return handler(core.StreamChunk{Done: true, FinishReason: finish})
}

// addUsage returns the sum of two usages; either may be nil.
func addUsage(a, b *core.TokenInfo) *core.TokenInfo {
	if a == nil || b == nil {
		if a == nil {
			a = b
		}
		if a == nil {
			return nil
		}
		sum := *a
		return &sum
	}
	return &core.TokenInfo{
		PromptTokens:     a.PromptTokens + b.PromptTokens,
		CompletionTokens: a.CompletionTokens + b.CompletionTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
	}
}

// recordRejectedUsage records the tokens of a model call whose reply the
// adapter rejected. The caller records a successful call's tokens from
// AdapterResult.Usage, and a failed call returns no result to carry them.
func recordRejectedUsage(ctx context.Context, usage *core.TokenInfo) {
	if usage == nil {
		return
	}
	core.RecordTokenUsage(ctx, &core.TokenUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
	})
}

func cloneOutputMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	return maps.Clone(in)
}
