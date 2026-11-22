package agentloop

import (
	"context"
	"reflect"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubTool struct {
	name   string
	result core.ToolResult
	err    error
	calls  int
}

func (s *stubTool) Name() string                           { return s.name }
func (s *stubTool) Description() string                    { return "stub " + s.name }
func (s *stubTool) Metadata() *core.ToolMetadata           { return &core.ToolMetadata{Name: s.name} }
func (s *stubTool) CanHandle(context.Context, string) bool { return true }
func (s *stubTool) Validate(map[string]any) error          { return nil }
func (s *stubTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {Type: "string", Description: "query", Required: true},
		},
	}
}
func (s *stubTool) Execute(context.Context, map[string]any) (core.ToolResult, error) {
	s.calls++
	return s.result, s.err
}

type stubLoopLLM struct {
	*core.BaseLLM
	reqs  []core.ChatRequest
	resps []*core.ChatResponse
	idx   int
}

func newStubLoopLLM(resps ...*core.ChatResponse) *stubLoopLLM {
	return &stubLoopLLM{
		BaseLLM: core.NewBaseLLM("openai", "gpt-5", []core.Capability{core.CapabilityChat, core.CapabilityToolCalling}, nil),
		resps:   resps,
	}
}

func (s *stubLoopLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, nil
}
func (s *stubLoopLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}
func (s *stubLoopLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, nil
}
func (s *stubLoopLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, nil
}
func (s *stubLoopLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, nil
}
func (s *stubLoopLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, nil
}
func (s *stubLoopLLM) StreamChat(context.Context, core.ChatRequest) (*core.StreamResponse, error) {
	return nil, nil
}
func (s *stubLoopLLM) Chat(_ context.Context, req core.ChatRequest) (*core.ChatResponse, error) {
	s.reqs = append(s.reqs, req)
	r := s.resps[s.idx]
	s.idx++
	return r, nil
}

func TestRun_MultiFieldSubmitUnknownToolAndForcedSubmit(t *testing.T) {
	sig := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question")}},
		[]core.OutputField{
			{Field: core.NewField("answer")},
			{Field: core.NewField("confidence", core.WithGoType(reflect.TypeOf(0)))},
		},
	)

	t.Run("rejects user tool named submit", func(t *testing.T) {
		model := newStubLoopLLM()
		_, err := Run(context.Background(), model, nil, Config{
			Signature: sig,
			Tools:     []core.Tool{&stubTool{name: "submit"}},
			MaxTurns:  3,
		})
		require.ErrorContains(t, err, "reserved")
	})

	t.Run("executes all calls in turn, turns unknown tool into error observation, validates multi-field submit", func(t *testing.T) {
		search := &stubTool{name: "search", result: core.ToolResult{Data: "Paris"}}
		model := newStubLoopLLM(
			&core.ChatResponse{
				FinishReason: core.FinishReasonToolCall,
				ToolCalls: []core.ToolCall{
					{ID: "c1", Name: "search", Arguments: map[string]any{"query": "France"}},
					{ID: "c2", Name: "missing_tool", Arguments: map[string]any{}},
				},
			},
			&core.ChatResponse{
				FinishReason: core.FinishReasonToolCall,
				ToolCalls: []core.ToolCall{
					{ID: "c3", Name: "submit", Arguments: map[string]any{"answer": "Paris", "confidence": 95}},
				},
			},
		)
		res, err := Run(context.Background(), model, []core.ChatMessage{
			{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("capital?")}},
		}, Config{
			Signature:        sig,
			Tools:            []core.Tool{search},
			MaxTurns:         4,
			ForceFinalSubmit: true,
		})
		require.NoError(t, err)
		assert.Equal(t, 1, search.calls)
		assert.Equal(t, "Paris", res.Outputs["answer"])
		assert.Equal(t, 95, res.Outputs["confidence"])
		assert.Equal(t, "submit", res.TerminationCause)
		assert.Equal(t, 2, res.Turns)
	})

	t.Run("budget counts forced submit and turns off reasoning effort", func(t *testing.T) {
		search := &stubTool{name: "search", result: core.ToolResult{Data: "fact"}}
		model := newStubLoopLLM(
			&core.ChatResponse{
				FinishReason: core.FinishReasonToolCall,
				ToolCalls:    []core.ToolCall{{ID: "c1", Name: "search", Arguments: map[string]any{"query": "q"}}},
			},
			&core.ChatResponse{
				FinishReason: core.FinishReasonToolCall,
				ToolCalls:    []core.ToolCall{{ID: "c2", Name: "submit", Arguments: map[string]any{"answer": "final", "confidence": 80}}},
			},
		)
		res, err := Run(context.Background(), model, []core.ChatMessage{
			{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("q")}},
		}, Config{
			Signature:        sig,
			Tools:            []core.Tool{search},
			MaxTurns:         2,
			ForceFinalSubmit: true,
			ReasoningEffort:  core.ReasoningEffortMedium,
		})
		require.NoError(t, err)
		assert.Equal(t, 2, res.Turns, "forced submit counts against MaxTurns budget")
		assert.Equal(t, "forced_submit", res.TerminationCause)
		assert.Equal(t, core.ToolChoiceNamed, model.reqs[1].ToolChoice.Mode)
		assert.Equal(t, SubmitToolName, model.reqs[1].ToolChoice.Name)
		assert.Equal(t, core.ReasoningEffortNone, model.reqs[1].ReasoningEffort)
	})

	t.Run("missing submit on forced turn returns ParseError, never empty success", func(t *testing.T) {
		model := newStubLoopLLM(
			&core.ChatResponse{FinishReason: core.FinishReasonStop, Content: "no tool"},
			&core.ChatResponse{FinishReason: core.FinishReasonStop, Content: "still no submit"},
		)
		_, err := Run(context.Background(), model, []core.ChatMessage{
			{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("q")}},
		}, Config{
			Signature:        sig,
			MaxTurns:         2,
			ForceFinalSubmit: true,
		})
		require.ErrorIs(t, err, dspyerrors.ErrParseFailed)
	})
}
