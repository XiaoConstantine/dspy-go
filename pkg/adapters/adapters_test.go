package adapters

import (
	"context"
	stderrors "errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	llm "github.com/XiaoConstantine/llm-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubChatLLM struct {
	*core.BaseLLM
	chatCalls    []core.ChatRequest
	responses    []*core.ChatResponse
	chatErrs     []error
	streamChunks [][]core.StreamChunk
	callIdx      int
}

func newStubChatLLM(provider, model string, caps ...core.Capability) *stubChatLLM {
	return &stubChatLLM{
		BaseLLM: core.NewBaseLLM(provider, core.ModelID(model), caps, nil),
	}
}

func (s *stubChatLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return &core.LLMResponse{}, nil
}
func (s *stubChatLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}
func (s *stubChatLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}
func (s *stubChatLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}
func (s *stubChatLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}
func (s *stubChatLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}

func (s *stubChatLLM) Chat(_ context.Context, req core.ChatRequest) (*core.ChatResponse, error) {
	s.chatCalls = append(s.chatCalls, req)
	idx := s.callIdx
	s.callIdx++
	if idx < len(s.chatErrs) && s.chatErrs[idx] != nil {
		return nil, s.chatErrs[idx]
	}
	if idx < len(s.responses) {
		return s.responses[idx], nil
	}
	return &core.ChatResponse{FinishReason: core.FinishReasonStop}, nil
}

func (s *stubChatLLM) StreamChat(_ context.Context, req core.ChatRequest) (*core.StreamResponse, error) {
	s.chatCalls = append(s.chatCalls, req)
	idx := s.callIdx
	s.callIdx++
	if idx < len(s.chatErrs) && s.chatErrs[idx] != nil {
		return nil, s.chatErrs[idx]
	}
	ch := make(chan core.StreamChunk, 32)
	var chunks []core.StreamChunk
	if idx < len(s.streamChunks) {
		chunks = s.streamChunks[idx]
	}
	go func() {
		defer close(ch)
		for _, c := range chunks {
			ch <- c
		}
	}()
	return &core.StreamResponse{ChunkChannel: ch, Cancel: func() {}}, nil
}

func TestXMLAdapter_LiteralCodeHTMLThinkFenceNullAndBlank(t *testing.T) {
	type Item struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{
			{Field: core.NewField("code")},
			{Field: core.NewField("note")},
			{Field: core.NewField("optional_note", core.WithOptional())},
			{Field: core.NewField("count", core.WithGoType(reflect.TypeOf(0)))},
			{Field: core.NewField("items", core.WithGoType(reflect.TypeOf([]Item{})))},
		},
	)

	raw := `<think>Let me analyze a < b && c > d</think>
Sure, here are the outputs:
` + "```xml\n" + `<code >
func f[T any](ch chan T, a, b int) bool {
    if a < b && b > 0 {
        ch <- *new(T)
        return true
    }
    return false
}
</code>

<note></note>

<optional_note />

<count>
0
</count>

<items>
[{"name":"alpha","count":2}]
</items>
` + "```"

	xmlAdapter := NewXMLAdapter()
	out, err := xmlAdapter.Parse(sig, raw)
	require.NoError(t, err)
	assert.Contains(t, out["code"], "if a < b && b > 0")
	assert.Contains(t, out["code"], "ch <- *new(T)")
	assert.Equal(t, "", out["note"], "blank string must be valid for required string")
	assert.Nil(t, out["optional_note"], "self-closing tag must decode to nil for optional field")
	assert.Equal(t, 0, out["count"])
	items, ok := out["items"].([]Item)
	require.True(t, ok)
	require.Len(t, items, 1)
	assert.Equal(t, "alpha", items[0].Name)
	assert.Equal(t, 2, items[0].Count)
}

func TestXMLAdapter_LiteralInnerClosingTag(t *testing.T) {
	sig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("answer")},
		{Field: core.NewField("confidence")},
	})
	raw := "<answer>\nUse </answer> to close the tag.\n</answer>\n\n<confidence>\nhigh\n</confidence>"
	out, err := NewXMLAdapter().Parse(sig, raw)
	require.NoError(t, err)
	assert.Equal(t, "Use </answer> to close the tag.", out["answer"])
	assert.Equal(t, "high", out["confidence"])
}

func TestXMLAdapter_ErrorsAndFallbackMatrix(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("q")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	t.Run("ParseError triggers single JSON fallback", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON, core.CapabilityJSONSchema)
		model.responses = []*core.ChatResponse{
			{Content: "plain text with no tags at all", FinishReason: core.FinishReasonStop},
			{Content: `{"answer":"42"}`, FinishReason: core.FinishReasonStop},
		}
		res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "6*7"}, core.AdapterCallOptions{})
		require.NoError(t, err)
		assert.Equal(t, "42", res.Outputs["answer"])
		assert.True(t, res.FallbackUsed)
		assert.Equal(t, "native", res.SchemaEnforcement)
		require.Len(t, model.chatCalls, 2)
		assert.Equal(t, core.ResponseFormatJSONSchema, model.chatCalls[1].ResponseFormat)
	})

	t.Run("a failed fallback still reports the fields the XML reply was missing", func(t *testing.T) {
		twoFields := core.NewSignature(
			[]core.InputField{{Field: core.NewField("q")}},
			[]core.OutputField{{Field: core.NewField("answer")}, {Field: core.NewField("confidence")}},
		)
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat)
		model.responses = []*core.ChatResponse{
			{Content: "<answer>42</answer>", FinishReason: core.FinishReasonStop},
			{Content: "still not JSON", FinishReason: core.FinishReasonStop},
		}
		_, err := NewXMLAdapter().Call(context.Background(), model, twoFields, nil, map[string]any{"q": "6*7"}, core.AdapterCallOptions{})
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
		var pe *dspyerrors.ParseError
		require.True(t, stderrors.As(err, &pe))
		assert.Equal(t, "xml", pe.Adapter)
		assert.Equal(t, []string{"confidence"}, pe.MissingFields)
		assert.Equal(t, "<answer>42</answer>", pe.Raw)
		assert.ErrorContains(t, err, "JSON fallback also failed")
		assert.ErrorContains(t, err, "no JSON object found")
		assert.Len(t, model.chatCalls, 2)
	})

	t.Run("Refusal returns ErrRefused with NO fallback", func(t *testing.T) {
		reason := "safety policy"
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON)
		model.responses = []*core.ChatResponse{
			{Refusal: &reason, FinishReason: core.FinishReasonRefusal},
		}
		_, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
		require.ErrorIs(t, err, dspyerrors.ErrRefused)
		assert.False(t, dspyerrors.Retryable(err))
		assert.Len(t, model.chatCalls, 1)
	})

	t.Run("ContentFilter returns ErrContentFiltered with NO fallback", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON)
		model.responses = []*core.ChatResponse{
			{Content: "partial", FinishReason: core.FinishReasonContentFilter},
		}
		_, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
		require.ErrorIs(t, err, dspyerrors.ErrContentFiltered)
		assert.Len(t, model.chatCalls, 1)
	})

	t.Run("Length truncation returns ErrTruncated with partial fields and NO fallback", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON)
		model.responses = []*core.ChatResponse{
			{Content: "<answer>\npartial ans", FinishReason: core.FinishReasonLength},
		}
		_, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
		require.ErrorIs(t, err, dspyerrors.ErrTruncated)
		var te *dspyerrors.TruncatedError
		require.True(t, stderrors.As(err, &te))
		assert.Equal(t, "partial ans", te.PartialFields["answer"])
		assert.Len(t, model.chatCalls, 1)
	})

	t.Run("llm.KindMalformedResponse maps to ParseError and triggers XML fallback", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON)
		model.chatErrs = []error{&llm.Error{Kind: llm.KindMalformedResponse, Err: stderrors.New("bad json")}, nil}
		model.responses = []*core.ChatResponse{nil, {Content: `{"answer":"ok"}`, FinishReason: core.FinishReasonStop}}
		res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
		require.NoError(t, err)
		assert.True(t, res.FallbackUsed)
		assert.Equal(t, "ok", res.Outputs["answer"])
	})

	t.Run("JSONAdapter never falls back", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON)
		model.responses = []*core.ChatResponse{
			{Content: "not json", FinishReason: core.FinishReasonStop},
		}
		_, err := NewJSONAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
		assert.Len(t, model.chatCalls, 1)
	})
}

func TestJSONAdapter_CapabilityStepDownAndOpenMap(t *testing.T) {
	openSig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("meta", core.WithGoType(reflect.TypeOf(map[string]any{})))},
	})
	model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityJSON, core.CapabilityJSONSchema)
	model.responses = []*core.ChatResponse{
		{Content: `{"meta":{"k":"v"}}`, FinishReason: core.FinishReasonStop},
	}
	res, err := NewJSONAdapter().Call(context.Background(), model, openSig, nil, nil, core.AdapterCallOptions{})
	require.NoError(t, err)
	assert.Equal(t, "json_object", res.SchemaEnforcement)
	assert.Equal(t, core.ResponseFormatJSONObject, model.chatCalls[0].ResponseFormat)

	// Model with neither schema nor JSON steps down to prompt_only.
	plainSig := core.NewSignature(nil, []core.OutputField{{Field: core.NewField("answer")}})
	plainModel := newStubChatLLM("local", "llama3", core.CapabilityChat)
	plainModel.responses = []*core.ChatResponse{
		{Content: "```json\n{\"answer\":\"hi\"}\n```", FinishReason: core.FinishReasonStop},
	}
	res2, err := NewJSONAdapter().Call(context.Background(), plainModel, plainSig, nil, nil, core.AdapterCallOptions{})
	require.NoError(t, err)
	assert.Equal(t, "prompt_only", res2.SchemaEnforcement)
	assert.Equal(t, core.ResponseFormatText, plainModel.chatCalls[0].ResponseFormat)
	assert.Equal(t, "hi", res2.Outputs["answer"])
}

func TestChainOfThoughtNativeReasoningResolution(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{
			{Field: core.NewField("reasoning", core.WithNativeReasoning())},
			{Field: core.NewField("answer")},
		},
	)

	t.Run("reasoning-capable model uses native reasoning and allows empty summary", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityReasoning)
		model.responses = []*core.ChatResponse{
			{Content: "<answer>\nParis\n</answer>", ReasoningSummary: "", FinishReason: core.FinishReasonStop},
		}
		res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"question": "capital of France"}, core.AdapterCallOptions{})
		require.NoError(t, err)
		assert.Equal(t, "Paris", res.Outputs["answer"])
		assert.Equal(t, "", res.Outputs["reasoning"])
		assert.Equal(t, core.ReasoningEffortLow, model.chatCalls[0].ReasoningEffort)
		assert.NotContains(t, model.chatCalls[0].Messages[0].Content[0].Text, "<reasoning>")
	})

	t.Run("non-reasoning model keeps reasoning first in ordered schema", func(t *testing.T) {
		model := newStubChatLLM("openai", "gpt-4o", core.CapabilityChat, core.CapabilityJSON, core.CapabilityJSONSchema)
		model.responses = []*core.ChatResponse{
			{Content: `{"reasoning":"France's capital is Paris","answer":"Paris"}`, FinishReason: core.FinishReasonStop},
		}
		res, err := NewJSONAdapter().Call(context.Background(), model, sig, nil, map[string]any{"question": "capital of France"}, core.AdapterCallOptions{})
		require.NoError(t, err)
		assert.Equal(t, "Paris", res.Outputs["answer"])
		assert.Equal(t, "France's capital is Paris", res.Outputs["reasoning"])
		schema := string(model.chatCalls[0].ResponseSchema.Schema)
		assert.Less(t, strings.Index(schema, `"reasoning"`), strings.Index(schema, `"answer"`))
	})

	t.Run("anthropic rejects temperature combined with native reasoning", func(t *testing.T) {
		model := newStubChatLLM("anthropic", "claude-sonnet-5", core.CapabilityChat, core.CapabilityReasoning)
		_, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"question": "hi"}, core.AdapterCallOptions{
			GenerateOptions: []core.GenerateOption{core.WithTemperature(0.7)},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "temperature cannot be combined with native reasoning")
		assert.Empty(t, model.chatCalls)
	})
}

func TestXMLStreamingChunkInvarianceAndErrors(t *testing.T) {
	sig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("reasoning")},
		{Field: core.NewField("answer")},
	})
	full := "<think>hidden</think>\n<reasoning>\nStep 1: a < b\nStep 2: done\n</reasoning>\n\n<answer>\n42\n</answer>"
	expected, err := NewXMLAdapter().Parse(sig, full)
	require.NoError(t, err)

	// Split the exact same reply at every chunk size 1..15 and verify identical field streams.
	for chunkSize := 1; chunkSize <= 15; chunkSize++ {
		var chunks []core.StreamChunk
		for i := 0; i < len(full); i += chunkSize {
			end := min(i+chunkSize, len(full))
			chunks = append(chunks, core.StreamChunk{Content: full[i:end]})
		}
		chunks = append(chunks, core.StreamChunk{Done: true, FinishReason: core.FinishReasonStop})

		model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityStreaming)
		model.streamChunks = [][]core.StreamChunk{chunks}

		gotFields := map[string]*strings.Builder{
			"reasoning": new(strings.Builder),
			"answer":    new(strings.Builder),
		}
		doneCount := 0
		res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, nil, core.AdapterCallOptions{
			StreamHandler: func(c core.StreamChunk) error {
				if c.Done {
					doneCount++
					return nil
				}
				assert.NotContains(t, c.Content, "<reasoning>")
				assert.NotContains(t, c.Content, "</reasoning>")
				assert.NotContains(t, c.Content, "<answer>")
				assert.NotContains(t, c.Content, "</answer>")
				require.Contains(t, gotFields, c.Field)
				gotFields[c.Field].WriteString(c.Content)
				return nil
			},
		})
		require.NoError(t, err, "chunkSize=%d", chunkSize)
		assert.Equal(t, 1, doneCount, "must emit exactly one Done chunk")
		assert.Equal(t, expected["reasoning"], gotFields["reasoning"].String(), "chunkSize=%d", chunkSize)
		assert.Equal(t, expected["answer"], gotFields["answer"].String(), "chunkSize=%d", chunkSize)
		assert.Equal(t, expected, res.Outputs)
	}

	// Failure after emitting returns StreamParseError with no silent fallback.
	truncModel := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityStreaming, core.CapabilityJSON)
	truncModel.streamChunks = [][]core.StreamChunk{{
		{Content: "<reasoning>\nlong enough text to flush past holdback buffer now\n"},
		{Done: true, FinishReason: core.FinishReasonLength},
	}}
	_, err = NewXMLAdapter().Call(context.Background(), truncModel, sig, nil, nil, core.AdapterCallOptions{
		StreamHandler: func(c core.StreamChunk) error { return nil },
	})
	var streamErr *dspyerrors.StreamParseError
	require.True(t, stderrors.As(err, &streamErr))
	assert.ErrorIs(t, err, dspyerrors.ErrTruncated)
	assert.Len(t, truncModel.chatCalls, 1)
}

func TestJSONStreamingFieldScopedDeltas(t *testing.T) {
	sig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("reasoning")},
		{Field: core.NewField("answer")},
	})
	model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityStreaming, core.CapabilityJSON)
	model.streamChunks = [][]core.StreamChunk{{
		{Content: `{"reasoning":"first `},
		{Content: `step","answer":"hel`},
		{Content: `lo\nworld"}`},
		{Done: true, FinishReason: core.FinishReasonStop},
	}}

	collected := map[string]*strings.Builder{
		"reasoning": new(strings.Builder),
		"answer":    new(strings.Builder),
	}
	res, err := NewJSONAdapter().Call(context.Background(), model, sig, nil, nil, core.AdapterCallOptions{
		StreamHandler: func(c core.StreamChunk) error {
			if c.Done {
				return nil
			}
			require.Contains(t, collected, c.Field)
			collected[c.Field].WriteString(c.Content)
			return nil
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "first step", collected["reasoning"].String())
	assert.Equal(t, "hello\nworld", collected["answer"].String())
	assert.Equal(t, "hello\nworld", res.Outputs["answer"])
}

func TestAdapterCache_ValidOnlyAndStreamingReplay(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("q")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	store, err := cache.NewMemoryCache(cache.CacheConfig{
		Type:       "memory",
		MaxSize:    1024 * 1024,
		DefaultTTL: time.Minute,
	})
	require.NoError(t, err)
	defer store.Close()
	ac := NewAdapterCache(store, time.Minute)

	// Invalid output is NOT cached.
	model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat)
	model.responses = []*core.ChatResponse{
		{Content: "garbage", FinishReason: core.FinishReasonStop},
		{Content: "<answer>42</answer>", FinishReason: core.FinishReasonStop},
	}
	adapter := NewXMLAdapter().WithJSONFallback(false).WithCache(ac)
	_, err = adapter.Call(context.Background(), model, sig, nil, map[string]any{"q": "life"}, core.AdapterCallOptions{})
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)

	// Next call hits model again and caches valid result.
	res1, err := adapter.Call(context.Background(), model, sig, nil, map[string]any{"q": "life"}, core.AdapterCallOptions{})
	require.NoError(t, err)
	assert.False(t, res1.CacheHit)
	assert.Equal(t, "42", res1.Outputs["answer"])
	assert.Len(t, model.chatCalls, 2)

	// Third call with streaming replays cached field chunks without calling model!
	var streamed strings.Builder
	doneCount := 0
	res2, err := adapter.Call(context.Background(), model, sig, nil, map[string]any{"q": "life"}, core.AdapterCallOptions{
		StreamHandler: func(c core.StreamChunk) error {
			if c.Done {
				doneCount++
				return nil
			}
			assert.Equal(t, "answer", c.Field)
			streamed.WriteString(c.Content)
			return nil
		},
	})
	require.NoError(t, err)
	assert.True(t, res2.CacheHit)
	assert.Equal(t, "42", streamed.String())
	assert.Equal(t, 1, doneCount)
	assert.Len(t, model.chatCalls, 2, "cached streaming call must not invoke model")
}

func TestFormatMultimodalDemosAndHistory(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{
			{Field: core.NewImageField("photo")},
			{Field: core.NewField("history")},
			{Field: core.NewField("question")},
		},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	img := core.NewImageBlock([]byte{1, 2, 3}, "image/png")
	demos := []core.Example{{
		Inputs:  map[string]any{"photo": img, "question": "what is this?"},
		Outputs: map[string]any{"answer": "a cat"},
	}}
	history := core.History{
		Messages: []map[string]any{{
			"question": "earlier?",
			"answer":   "yes",
		}},
	}
	msgs, err := NewXMLAdapter().Format(sig, demos, map[string]any{
		"photo":    img,
		"history":  history,
		"question": "now?",
	})
	require.NoError(t, err)
	// system + demo(user, assistant) + history(user, assistant) + current user = 6 messages
	require.Len(t, msgs, 6)
	assert.Equal(t, "system", msgs[0].Role)
	assert.Equal(t, "user", msgs[1].Role)
	// Demo user turn retains binary image block!
	foundDemoImg := false
	for _, b := range msgs[1].Content {
		if b.Type == core.FieldTypeImage && len(b.Data) == 3 {
			foundDemoImg = true
		}
	}
	assert.True(t, foundDemoImg, "multimodal demo must preserve image ContentBlock")
	assert.Equal(t, "assistant", msgs[2].Role)
	assert.Equal(t, "user", msgs[3].Role)
	assert.Equal(t, "assistant", msgs[4].Role)
	assert.Equal(t, "user", msgs[5].Role)
}

func FuzzXMLStreamScannerMatchesParse(f *testing.F) {
	f.Add("hello world", "42", uint8(3))
	f.Add("if a < b && c > d { ch <- 1 }", "", uint8(1))
	f.Add("line1\nline2\n\nline3", "done", uint8(5))
	sig := core.NewSignature(nil, []core.OutputField{
		{Field: core.NewField("reasoning")},
		{Field: core.NewField("answer")},
	})

	f.Fuzz(func(t *testing.T, r, a string, step uint8) {
		// Avoid raw closing tag injections inside fuzzed bodies without a trailing boundary.
		if strings.Contains(r, "</reasoning>") || strings.Contains(a, "</answer>") ||
			strings.Contains(r, "\r") || strings.Contains(a, "\r") ||
			strings.HasPrefix(r, "\n") || strings.HasSuffix(r, "\n") ||
			strings.HasPrefix(a, "\n") || strings.HasSuffix(a, "\n") {
			return
		}
		if step == 0 {
			step = 1
		}
		wire := "<reasoning>\n" + r + "\n</reasoning>\n\n<answer>\n" + a + "\n</answer>"
		parsed, err := NewXMLAdapter().Parse(sig, wire)
		if err != nil {
			t.Fatalf("Parse failed for valid formatted wire %q: %v", wire, err)
		}
		if parsed["reasoning"] != r || parsed["answer"] != a {
			t.Fatalf("round-trip mismatch: got (%q, %q), want (%q, %q)", parsed["reasoning"], parsed["answer"], r, a)
		}

		scanner := newXMLStreamScanner(sig)
		got := map[string]*strings.Builder{"reasoning": new(strings.Builder), "answer": new(strings.Builder)}
		sz := int(step)
		for i := 0; i < len(wire); i += sz {
			end := min(i+sz, len(wire))
			for _, d := range scanner.feed(wire[i:end], false) {
				got[d.field].WriteString(d.content)
			}
		}
		for _, d := range scanner.feed("", true) {
			got[d.field].WriteString(d.content)
		}
		if got["reasoning"].String() != r || got["answer"].String() != a {
			t.Fatalf("stream mismatch (step=%d): got (%q, %q), want (%q, %q)", sz, got["reasoning"].String(), got["answer"].String(), r, a)
		}
	})
}

func newTestAdapterCache(t *testing.T) *AdapterCache {
	t.Helper()
	store, err := cache.NewMemoryCache(cache.CacheConfig{Type: "memory", MaxSize: 1024 * 1024, DefaultTTL: time.Minute})
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	return NewAdapterCache(store, time.Minute)
}

func TestAdapterCache_HitReturnsTheTypesTheFirstCallReturned(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("q")}},
		[]core.OutputField{
			{Field: core.NewField("count", core.WithGoType(reflect.TypeOf(0)))},
			{Field: core.NewField("tags", core.WithGoType(reflect.TypeOf([]string(nil))))},
			{Field: core.NewField("note")},
		},
	)
	usage := &core.TokenInfo{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	replies := map[string]string{
		"xml":  "<count>3</count>\n<tags>[\"a\", \"b\"]</tags>\n<note>n</note>",
		"json": `{"count": 3, "tags": ["a", "b"], "note": "n"}`,
	}
	adapters := map[string]core.Adapter{
		"xml":  NewXMLAdapter().WithCache(newTestAdapterCache(t)),
		"json": NewJSONAdapter().WithCache(newTestAdapterCache(t)),
	}
	for name, adapter := range adapters {
		t.Run(name, func(t *testing.T) {
			model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat)
			model.responses = []*core.ChatResponse{{Content: replies[name], FinishReason: core.FinishReasonStop, Usage: usage}}
			want := map[string]any{"count": 3, "tags": []string{"a", "b"}, "note": "n"}

			first, err := adapter.Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
			require.NoError(t, err)
			assert.Equal(t, want, first.Outputs)
			assert.Equal(t, usage, first.Usage)

			second, err := adapter.Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
			require.NoError(t, err)
			require.True(t, second.CacheHit)
			assert.Len(t, model.chatCalls, 1)
			assert.Equal(t, want, second.Outputs, "a cache hit returns the same Go types")
			assert.Nil(t, second.Usage, "a cache hit spent no tokens")
		})
	}
}

func TestXMLAdapter_NoFallbackAfterStreamedOutput(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("q")}},
		[]core.OutputField{{Field: core.NewField("answer")}, {Field: core.NewField("confidence")}},
	)
	model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat, core.CapabilityStreaming)
	model.streamChunks = [][]core.StreamChunk{
		// The answer streams to the handler, then the reply ends without confidence.
		{{Content: "<answer>Paris is the capital</answer>"}, {Done: true, FinishReason: core.FinishReasonStop}},
		{{Content: `{"answer": "Lyon", "confidence": "low"}`}, {Done: true, FinishReason: core.FinishReasonStop}},
	}

	var streamed strings.Builder
	res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{
		StreamHandler: func(c core.StreamChunk) error {
			streamed.WriteString(c.Content)
			return nil
		},
	})

	assert.Nil(t, res)
	var streamErr *dspyerrors.StreamParseError
	require.ErrorAs(t, err, &streamErr)
	assert.Equal(t, "Paris is the capital", streamErr.PartialFields["answer"])
	assert.Equal(t, "Paris is the capital", streamed.String(), "the handler never sees a second reply")
	assert.Len(t, model.chatCalls, 1, "no fallback call after output was streamed")
}

func TestXMLAdapter_FallbackUsageCoversBothCalls(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("q")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	model := newStubChatLLM("openai", "gpt-5", core.CapabilityChat)
	model.responses = []*core.ChatResponse{
		{Content: "Paris", FinishReason: core.FinishReasonStop, Usage: &core.TokenInfo{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107}},
		{Content: `{"answer": "Paris"}`, FinishReason: core.FinishReasonStop, Usage: &core.TokenInfo{PromptTokens: 90, CompletionTokens: 9, TotalTokens: 99}},
	}

	res, err := NewXMLAdapter().Call(context.Background(), model, sig, nil, map[string]any{"q": "x"}, core.AdapterCallOptions{})
	require.NoError(t, err)
	require.True(t, res.FallbackUsed)
	assert.Equal(t, &core.TokenInfo{PromptTokens: 190, CompletionTokens: 16, TotalTokens: 206}, res.Usage)
}

func TestAdapters_AmbiguousRepliesAreErrorsNotGuesses(t *testing.T) {
	sig := core.NewSignature(nil, []core.OutputField{{Field: core.NewField("answer")}})

	t.Run("xml: prose that mentions the tag before the real one", func(t *testing.T) {
		raw := "I'll put it in <answer> tags.\n<answer>42</answer>"
		out, err := NewXMLAdapter().Parse(sig, raw)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed, "outputs: %v", out)
	})

	t.Run("xml: a value that mentions its own tag still parses when the reply starts with the tag", func(t *testing.T) {
		out, err := NewXMLAdapter().Parse(sig, "<answer>\nUse <answer> to open the field.\n</answer>")
		require.NoError(t, err)
		assert.Equal(t, "Use <answer> to open the field.", out["answer"])
	})

	t.Run("xml: prose before a single tag is still accepted", func(t *testing.T) {
		out, err := NewXMLAdapter().Parse(sig, "Sure, here it is.\n<answer>42</answer>")
		require.NoError(t, err)
		assert.Equal(t, "42", out["answer"])
	})

	t.Run("json: an example object before the real one", func(t *testing.T) {
		raw := `Example: {"answer": "<your answer>"}. Real: {"answer": "b"}`
		out, err := NewJSONAdapter().Parse(sig, raw)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed, "outputs: %v", out)
	})

	t.Run("json: prose around one object is still accepted", func(t *testing.T) {
		out, err := NewJSONAdapter().Parse(sig, "Sure.\n{\"answer\": \"a {b} c\"}\nDone.")
		require.NoError(t, err)
		assert.Equal(t, "a {b} c", out["answer"])
	})
}
