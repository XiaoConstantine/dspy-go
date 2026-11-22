package modules

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/adapters"
	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// replyLLM is a text-only model that answers each call with the next reply,
// repeating the last one, and records the prompts it was sent.
type replyLLM struct {
	core.LLM
	id      string
	mu      sync.Mutex
	replies []string
	prompts []string
}

func newReplyLLM(t *testing.T, replies ...string) *replyLLM {
	// The model ID is unique per test, so tests that wrap the model in the
	// process-wide reply cache never share entries.
	return &replyLLM{id: t.Name(), replies: replies}
}

func (m *replyLLM) ModelID() string      { return m.id }
func (m *replyLLM) ProviderName() string { return "test" }
func (m *replyLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

func (m *replyLLM) Generate(_ context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	reply := m.replies[min(len(m.prompts), len(m.replies)-1)]
	m.prompts = append(m.prompts, prompt)
	return &core.LLMResponse{Content: reply}, nil
}

func (m *replyLLM) calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.prompts)
}

func qaSignature(outputs ...string) core.Signature {
	fields := make([]core.OutputField, len(outputs))
	for i, name := range outputs {
		fields[i] = core.OutputField{Field: core.NewField(name)}
	}
	return core.NewSignature([]core.InputField{{Field: core.NewField("question")}}, fields)
}

var question = map[string]any{"question": "Capital of France?"}

func TestPredictWithAdapter_ParsesTaggedReplyAndReportsUnparseableOnes(t *testing.T) {
	t.Run("a tagged reply fills every field and nothing else", func(t *testing.T) {
		llm := newReplyLLM(t, "<answer>Paris</answer>\n<confidence>high</confidence>")
		predict := NewPredict(qaSignature("answer", "confidence")).WithAdapter(adapters.NewXMLAdapter())
		predict.SetLLM(llm)

		outputs, err := predict.Process(context.Background(), question)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"answer": "Paris", "confidence": "high"}, outputs, "no raw response key")
		require.Equal(t, 1, llm.calls())
		assert.True(t, strings.HasPrefix(llm.prompts[0], "Your input fields are:"))
	})

	t.Run("a missing field is an error naming it, after one JSON fallback call", func(t *testing.T) {
		llm := newReplyLLM(t, "<answer>Paris</answer>")
		predict := NewPredict(qaSignature("answer", "confidence")).WithAdapter(adapters.NewXMLAdapter())
		predict.SetLLM(llm)

		outputs, err := predict.Process(context.Background(), question)
		assert.Nil(t, outputs)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
		var parseErr *dspyerrors.ParseError
		require.True(t, stderrors.As(err, &parseErr))
		assert.Equal(t, []string{"confidence"}, parseErr.MissingFields)
		assert.Equal(t, 2, llm.calls())
	})

	t.Run("the JSON adapter never falls back", func(t *testing.T) {
		llm := newReplyLLM(t, "I do not know.")
		predict := NewPredict(qaSignature("answer")).WithAdapter(adapters.NewJSONAdapter())
		predict.SetLLM(llm)

		_, err := predict.Process(context.Background(), question)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
		assert.Equal(t, 1, llm.calls())
	})
}

func TestPredictWithAdapter_ReplacesAndRestoresTheOutputMode(t *testing.T) {
	// Two outputs make NewPredict choose its XML interceptor mode.
	predict := NewPredict(qaSignature("answer", "confidence"))
	require.True(t, predict.IsXMLModeEnabled())
	require.Nil(t, predict.GetAdapter())

	predict.WithAdapter(adapters.NewJSONAdapter())
	assert.False(t, predict.IsXMLModeEnabled())
	assert.Empty(t, predict.GetInterceptors(), "the XML interceptor must not parse the adapter's outputs")
	llm := newReplyLLM(t, `{"answer":"Paris","confidence":"high"}`, "answer: Lyon\nconfidence: low")
	predict.SetLLM(llm)
	outputs, err := predict.Process(context.Background(), question)
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"answer": "Paris", "confidence": "high"}, outputs)

	cloned := predict.Clone().(*Predict)
	require.NotNil(t, cloned.GetAdapter())
	assert.Equal(t, "json", cloned.GetAdapter().Name())

	// Without an adapter the same module parses prefixed text again.
	predict.WithAdapter(nil)
	outputs, err = predict.Process(context.Background(), question)
	require.NoError(t, err)
	assert.Equal(t, "Lyon", outputs["answer"])
	assert.Equal(t, "low", outputs["confidence"])
	assert.True(t, strings.HasPrefix(llm.prompts[1], "Given the fields"))
}

func TestPredictWithAdapter_BypassesTheRawReplyCache(t *testing.T) {
	t.Setenv("DSPY_CACHE_ENABLED", "true")
	t.Setenv("DSPY_CACHE_TYPE", "memory")

	// The first reply cannot be parsed. A raw-reply cache would replay it.
	llm := newReplyLLM(t, "no tags", "no tags", "<answer>Paris</answer>")
	cached := cache.WrapWithCache(llm, nil)
	require.IsType(t, &cache.CachedLLM{}, cached)

	withAdapter := NewPredict(qaSignature("answer")).WithAdapter(adapters.NewXMLAdapter())
	withAdapter.SetLLM(cached)
	_, err := withAdapter.Process(context.Background(), question)
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	require.Equal(t, 2, llm.calls(), "the XML call and its JSON fallback")

	outputs, err := withAdapter.Process(context.Background(), question)
	require.NoError(t, err, "the same request reaches the model again instead of the cached failure")
	assert.Equal(t, "Paris", outputs["answer"])
	assert.Equal(t, 3, llm.calls())

	// The validated output is cached in the model's cache in its place.
	outputs, err = withAdapter.Process(context.Background(), question)
	require.NoError(t, err)
	assert.Equal(t, "Paris", outputs["answer"])
	assert.Equal(t, 3, llm.calls(), "a validated output is served from the cache")

	// A Predict without an adapter still uses the cache, as before.
	plain := NewPredict(qaSignature("answer"))
	plain.SetLLM(cached)
	for range 2 {
		_, err := plain.Process(context.Background(), question)
		require.NoError(t, err)
	}
	assert.Equal(t, 4, llm.calls(), "the second identical call is served from the cache")
}

func TestPredictAdapter_IsSavedOnlyWhenChosen(t *testing.T) {
	newProgram := func(withAdapter bool) core.Program {
		chosen := NewPredict(qaSignature("answer"))
		if withAdapter {
			chosen.WithAdapter(adapters.NewXMLAdapter())
		}
		return core.NewProgram(map[string]core.Module{
			"chosen":  chosen,
			"default": NewPredict(qaSignature("answer", "confidence")),
		}, nil)
	}
	saved := newProgram(true)
	path := filepath.Join(t.TempDir(), "program.json")
	require.NoError(t, core.SaveProgram(&saved, path))

	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	var file struct {
		Modules map[string]map[string]any `json:"modules"`
	}
	require.NoError(t, json.Unmarshal(raw, &file))
	assert.Equal(t, "xml", file.Modules["chosen"]["adapter"])
	assert.NotContains(t, file.Modules["default"], "adapter", "a module without an adapter is saved as before")

	loaded := newProgram(false)
	var warnings []string
	require.NoError(t, core.LoadProgramWithOptions(&loaded, path, core.LoadProgramOptions{
		WarningHandler: func(message string) { warnings = append(warnings, message) },
	}))
	assert.Empty(t, warnings)
	require.NotNil(t, loaded.Modules["chosen"].(*Predict).GetAdapter())
	assert.Equal(t, "xml", loaded.Modules["chosen"].(*Predict).GetAdapter().Name())
	assert.Nil(t, loaded.Modules["default"].(*Predict).GetAdapter())
	assert.True(t, loaded.Modules["default"].(*Predict).IsXMLModeEnabled())
}

func TestTypedPredictWithAdapter_ValidatesTypesAndPresence(t *testing.T) {
	type inputs struct {
		Question string `dspy:"question"`
	}
	type outputs struct {
		Answer string  `dspy:"answer"`
		Score  int     `dspy:"score"`
		Note   *string `dspy:"note"`
	}
	newPredict := func(reply string) (*Predict, *replyLLM) {
		llm := newReplyLLM(t, reply)
		predict := NewTypedPredict[inputs, outputs]().WithAdapter(adapters.NewXMLAdapter().WithJSONFallback(false))
		predict.SetLLM(llm)
		return predict, llm
	}

	predict, _ := newPredict("<answer></answer>\n<score>7</score>\n<note />")
	got, err := ProcessTyped[inputs, outputs](context.Background(), predict, inputs{Question: "q"})
	require.NoError(t, err)
	assert.Equal(t, outputs{Answer: "", Score: 7, Note: nil}, got, "a blank string and an omitted pointer field are valid")

	predict, _ = newPredict("<answer>a</answer>\n<score>seven</score>")
	_, err = ProcessTyped[inputs, outputs](context.Background(), predict, inputs{Question: "q"})
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	assert.ErrorContains(t, err, "score")

	predict, _ = newPredict("<score>7</score>")
	_, err = ProcessTyped[inputs, outputs](context.Background(), predict, inputs{Question: "q"})
	require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	assert.ErrorContains(t, err, "missing fields: answer")
}

func TestChainOfThought_AdapterOutputsAlsoCarryRationale(t *testing.T) {
	t.Run("without an adapter the field is rationale", func(t *testing.T) {
		llm := newReplyLLM(t, "<response><rationale>because</rationale><answer>Paris</answer></response>")
		cot := NewChainOfThought(qaSignature("answer"))
		cot.SetLLM(llm)
		assert.Equal(t, "rationale", cot.GetSignature().Outputs[0].Name)

		var recorded map[string]any
		ctx := WithPredictionObserver(context.Background(), func(_ *Predict, _, outputs map[string]any) {
			recorded = outputs
		})
		outputs, err := cot.Process(ctx, question)
		require.NoError(t, err)
		assert.Equal(t, "because", outputs["rationale"])
		assert.Equal(t, "Paris", outputs["answer"])
		assert.NotContains(t, outputs, "reasoning", "the default path gains no key")
		assert.Equal(t, outputs, recorded)
		assert.Contains(t, llm.prompts[0], "rationale")
		assert.NotContains(t, llm.prompts[0], "reasoning:")
	})

	t.Run("with an adapter the field is reasoning", func(t *testing.T) {
		llm := newReplyLLM(t, "<reasoning>because</reasoning>\n<answer>Paris</answer>")
		cot := NewChainOfThought(qaSignature("answer")).WithAdapter(adapters.NewXMLAdapter())
		cot.SetLLM(llm)
		require.Equal(t, "reasoning", cot.GetSignature().Outputs[0].Name)

		var recorded map[string]any
		ctx := WithPredictionObserver(context.Background(), func(_ *Predict, _, outputs map[string]any) {
			recorded = outputs
		})
		outputs, err := cot.Process(ctx, question)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"reasoning": "because", "rationale": "because", "answer": "Paris"}, outputs)
		assert.NotContains(t, recorded, "rationale", "recorded demonstrations keep only the declared field")

		cot.SetSignature(qaSignature("answer", "confidence"))
		assert.Equal(t, "reasoning", cot.GetSignature().Outputs[0].Name)
		cot.WithAdapter(nil)
		assert.Equal(t, "rationale", cot.GetSignature().Outputs[0].Name)
		assert.Equal(t, "rationale:", cot.GetSignature().Outputs[0].Prefix)
	})

	t.Run("a field the caller declared is never overwritten", func(t *testing.T) {
		llm := newReplyLLM(t, "<rationale>why</rationale>\n<reasoning>mine</reasoning>\n<answer>Paris</answer>")
		cot := NewChainOfThought(qaSignature("reasoning", "answer")).WithAdapter(adapters.NewXMLAdapter())
		cot.SetLLM(llm)

		outputs, err := cot.Process(context.Background(), question)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"rationale": "why", "reasoning": "mine", "answer": "Paris"}, outputs)
	})

	t.Run("a field the caller declared stays absent when the adapter omits it", func(t *testing.T) {
		signature := qaSignature("reasoning", "answer")
		signature.Outputs = append([]core.OutputField{{Field: core.NewField("rationale", core.WithOptional())}}, signature.Outputs...)
		llm := newReplyLLM(t, "<reasoning>mine</reasoning>\n<answer>Paris</answer>")
		cot := NewChainOfThought(signature).WithAdapter(omitsNilOutputs{adapters.NewXMLAdapter()})
		cot.SetLLM(llm)

		outputs, err := cot.Process(context.Background(), question)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"reasoning": "mine", "answer": "Paris"}, outputs)
	})
}

// omitsNilOutputs is a caller's adapter that leaves an absent optional output
// out of the result, where the built-in adapters return it as nil.
type omitsNilOutputs struct{ core.Adapter }

func (a omitsNilOutputs) Call(ctx context.Context, model core.LLM, signature core.Signature, demos []core.Example, inputs map[string]any, opts core.AdapterCallOptions) (*core.AdapterResult, error) {
	result, err := a.Adapter.Call(ctx, model, signature, demos, inputs, opts)
	if err != nil {
		return nil, err
	}
	for name, value := range result.Outputs {
		if value == nil {
			delete(result.Outputs, name)
		}
	}
	return result, nil
}

// toolChatLLM is a chat model with native tool calling that answers each Chat
// call with the next scripted response.
type toolChatLLM struct {
	*core.BaseLLM
	requests  []core.ChatRequest
	responses []*core.ChatResponse
}

func (m *toolChatLLM) Chat(_ context.Context, request core.ChatRequest) (*core.ChatResponse, error) {
	response := m.responses[min(len(m.requests), len(m.responses)-1)]
	m.requests = append(m.requests, request)
	return response, nil
}

func (m *toolChatLLM) StreamChat(context.Context, core.ChatRequest) (*core.StreamResponse, error) {
	return nil, stderrors.New("not used")
}

func (m *toolChatLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, stderrors.New("the text loop must not run")
}

func (m *toolChatLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, stderrors.New("not used")
}

func (m *toolChatLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, stderrors.New("not used")
}

func (m *toolChatLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, stderrors.New("not used")
}

func (m *toolChatLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, stderrors.New("not used")
}

func (m *toolChatLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, stderrors.New("not used")
}

func TestReActWithAdapter_SubmitsEveryOutputFieldThroughTheToolLoop(t *testing.T) {
	registry := tools.NewInMemoryToolRegistry()
	var searched []any
	require.NoError(t, registry.Register(tools.NewFuncTool("search", "Search the web",
		models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{
			"query": {Type: "string", Description: "query", Required: true},
		}},
		func(_ context.Context, args map[string]any) (*models.CallToolResult, error) {
			searched = append(searched, args["query"])
			return &models.CallToolResult{Content: []models.Content{models.TextContent{Type: "text", Text: "Paris is the capital."}}}, nil
		})))
	newModel := func(responses ...*core.ChatResponse) *toolChatLLM {
		return &toolChatLLM{
			BaseLLM:   core.NewBaseLLM("test", "tools", []core.Capability{core.CapabilityChat, core.CapabilityToolCalling}, nil),
			responses: responses,
		}
	}
	call := func(name string, arguments map[string]any) *core.ChatResponse {
		return &core.ChatResponse{
			ToolCalls:    []core.ToolCall{{ID: "call-" + name, Name: name, Arguments: arguments}},
			FinishReason: core.FinishReasonToolCall,
		}
	}

	t.Run("a tool call, then a submit with both fields", func(t *testing.T) {
		model := newModel(
			call("search", map[string]any{"query": "capital of France"}),
			call("submit", map[string]any{"answer": "Paris", "confidence": "high"}),
		)
		react := NewReAct(qaSignature("answer", "confidence"), registry, 4).WithAdapter(adapters.NewXMLAdapter())
		react.SetLLM(model)

		outputs, trace, err := react.ProcessWithTrace(context.Background(), question)
		require.NoError(t, err)
		assert.Equal(t, map[string]any{"answer": "Paris", "confidence": "high"}, outputs)
		assert.Equal(t, []any{"capital of France"}, searched)
		assert.Equal(t, "submit", trace.TerminationCause)
		assert.Len(t, model.requests, 2)
	})

	t.Run("a submit that omits a field is an error, not an empty field", func(t *testing.T) {
		model := newModel(call("submit", map[string]any{"answer": "Paris"}))
		react := NewReAct(qaSignature("answer", "confidence"), registry, 1).WithAdapter(adapters.NewXMLAdapter())
		react.SetLLM(model)

		outputs, err := react.Process(context.Background(), question)
		assert.Nil(t, outputs)
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	})
}
