package llms

import (
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	llm "github.com/XiaoConstantine/llm-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type generatorStub struct {
	info     llm.ModelInfo
	request  llm.Request
	response *llm.Response
	stream   llm.Stream
	err      error
}

func (g *generatorStub) Info() llm.ModelInfo {
	return g.info
}

func (g *generatorStub) Generate(_ context.Context, request llm.Request) (*llm.Response, error) {
	g.request = request
	return g.response, g.err
}

func (g *generatorStub) Stream(_ context.Context, request llm.Request) (llm.Stream, error) {
	g.request = request
	return g.stream, g.err
}

type sliceStream struct {
	chunks []llm.Chunk
	next   int
	closed bool
}

type embeddingStub struct {
	input       string
	inputs      []string
	single      *core.EmbeddingResult
	batch       *core.BatchEmbeddingResult
	singleError error
	batchError  error
}

func (e *embeddingStub) CreateEmbedding(
	_ context.Context,
	input string,
	_ ...core.EmbeddingOption,
) (*core.EmbeddingResult, error) {
	e.input = input
	return e.single, e.singleError
}

func (e *embeddingStub) CreateEmbeddings(
	_ context.Context,
	inputs []string,
	_ ...core.EmbeddingOption,
) (*core.BatchEmbeddingResult, error) {
	e.inputs = inputs
	return e.batch, e.batchError
}

func (s *sliceStream) Recv() (llm.Chunk, error) {
	if s.next == len(s.chunks) {
		return llm.Chunk{}, io.EOF
	}
	chunk := s.chunks[s.next]
	s.next++
	return chunk, nil
}

func (s *sliceStream) Close() error {
	s.closed = true
	return nil
}

func TestAdaptGeneratePreservesOptionsUsageAndMetadata(t *testing.T) {
	cost := llm.UsageCost{Input: 0.1, Output: 0.2, Total: 0.3}
	generator := &generatorStub{
		info: llm.ModelInfo{
			Provider: "test", Model: "model",
			Capabilities: []llm.Capability{
				llm.CapabilityGeneration,
				llm.CapabilityStreaming,
				llm.CapabilityTools,
				llm.CapabilityJSON,
				llm.CapabilityVision,
				llm.CapabilityAudio,
			},
		},
		response: &llm.Response{
			ID: "response-1", Model: "resolved-model",
			Message:      llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "hello"}}},
			FinishReason: llm.FinishReasonStop,
			Usage: &llm.Usage{
				InputTokens: 3, CacheReadTokens: 2, OutputTokens: 4, TotalTokens: 9, Cost: &cost,
			},
		},
	}

	adapted, err := Adapt(generator)
	require.NoError(t, err)
	response, err := adapted.Generate(
		context.Background(),
		"hi",
		core.WithMaxTokens(123),
		core.WithTemperature(0.25),
		core.WithTopP(0.8),
		core.WithPresencePenalty(0.1),
		core.WithFrequencyPenalty(0.2),
		core.WithStopSequences("END"),
	)
	require.NoError(t, err)

	assert.Equal(t, "test", adapted.ProviderName())
	assert.Equal(t, "model", adapted.ModelID())
	assert.Same(t, generator, adapted.Generator())
	assert.ElementsMatch(t, []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityStreaming,
		core.CapabilityToolCalling,
		core.CapabilityJSON,
		core.CapabilityMultimodal,
		core.CapabilityVision,
		core.CapabilityAudio,
	}, adapted.Capabilities())

	require.Len(t, generator.request.Messages, 1)
	assert.Equal(t, llm.RoleUser, generator.request.Messages[0].Role)
	assert.Equal(t, "hi", generator.request.Messages[0].Text())
	assert.Equal(t, 123, generator.request.MaxOutputTokens)
	require.NotNil(t, generator.request.Temperature)
	assert.InDelta(t, 0.25, *generator.request.Temperature, 0.0001)
	require.NotNil(t, generator.request.TopP)
	assert.InDelta(t, 0.8, *generator.request.TopP, 0.0001)
	assert.Equal(t, []string{"END"}, generator.request.Stop)

	assert.Equal(t, "hello", response.Content)
	assert.Equal(t, &core.TokenInfo{PromptTokens: 5, CompletionTokens: 4, TotalTokens: 9}, response.Usage)
	assert.Equal(t, "response-1", response.Metadata["response_id"])
	assert.Equal(t, "resolved-model", response.Metadata["model"])
	assert.Equal(t, "stop", response.Metadata["finish_reason"])
	assert.Equal(t, cost, response.Metadata["usage_cost"])
}

func TestAdaptOmitsUnsupportedImplicitDefaults(t *testing.T) {
	tests := []struct {
		name  string
		info  llm.ModelInfo
		check func(*testing.T, llm.Request)
	}{
		{
			name: "Codex max output tokens",
			info: llm.ModelInfo{Provider: "openai-codex", Model: "gpt-5.4"},
			check: func(t *testing.T, request llm.Request) {
				assert.Zero(t, request.MaxOutputTokens)
				require.NotNil(t, request.Temperature)
				assert.InDelta(t, 0.5, *request.Temperature, 0.0001)
			},
		},
		{
			name: "Anthropic temperature",
			info: llm.ModelInfo{
				Provider: "anthropic", Model: "claude-opus-5",
				Compatibility: &llm.ModelCompatibility{Anthropic: &llm.AnthropicCompatibility{
					Temperature: llm.CompatibilityDisabled,
				}},
			},
			check: func(t *testing.T, request llm.Request) {
				assert.Equal(t, 8192, request.MaxOutputTokens)
				assert.Nil(t, request.Temperature)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := &generatorStub{
				info: test.info,
				response: &llm.Response{Message: llm.Message{
					Role: llm.RoleAssistant, Content: []llm.Part{{Text: "ok"}},
				}},
			}
			adapted, err := Adapt(generator)
			require.NoError(t, err)
			_, err = adapted.Generate(context.Background(), "hi")
			require.NoError(t, err)
			test.check(t, generator.request)
		})
	}
}

func TestAdaptPreservesExplicitUnsupportedOptions(t *testing.T) {
	tests := []struct {
		name    string
		info    llm.ModelInfo
		options []core.GenerateOption
		check   func(*testing.T, llm.Request)
	}{
		{
			name:    "Codex max output tokens",
			info:    llm.ModelInfo{Provider: "openai-codex", Model: "gpt-5.4"},
			options: []core.GenerateOption{core.WithMaxTokens(123)},
			check: func(t *testing.T, request llm.Request) {
				assert.Equal(t, 123, request.MaxOutputTokens)
			},
		},
		{
			name: "Anthropic temperature",
			info: llm.ModelInfo{
				Provider: "anthropic", Model: "claude-opus-5",
				Compatibility: &llm.ModelCompatibility{Anthropic: &llm.AnthropicCompatibility{
					Temperature: llm.CompatibilityDisabled,
				}},
			},
			options: []core.GenerateOption{core.WithTemperature(0.25)},
			check: func(t *testing.T, request llm.Request) {
				require.NotNil(t, request.Temperature)
				assert.InDelta(t, 0.25, *request.Temperature, 0.0001)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := &generatorStub{
				info: test.info,
				response: &llm.Response{Message: llm.Message{
					Role: llm.RoleAssistant, Content: []llm.Part{{Text: "ok"}},
				}},
			}
			adapted, err := Adapt(generator)
			require.NoError(t, err)
			_, err = adapted.Generate(context.Background(), "hi", test.options...)
			require.NoError(t, err)
			test.check(t, generator.request)
		})
	}
}

func TestAdaptGenerateWithJSONFallsBackToPromptInstruction(t *testing.T) {
	generator := &generatorStub{
		info: llm.ModelInfo{
			Provider: "anthropic", Model: "model",
			Capabilities: []llm.Capability{llm.CapabilityGeneration},
		},
		response: &llm.Response{Message: llm.Message{
			Role: llm.RoleAssistant, Content: []llm.Part{{Text: `{"answer":42}`}},
		}},
	}
	adapted, err := Adapt(generator)
	require.NoError(t, err)

	result, err := adapted.GenerateWithJSON(context.Background(), "answer")
	require.NoError(t, err)
	assert.Equal(t, float64(42), result["answer"])
	assert.Equal(t, llm.ResponseFormatText, generator.request.ResponseFormat)
	assert.Contains(t, generator.request.Messages[0].Text(), "Return only a valid JSON object")
}

func TestAdaptGenerateWithToolsPreservesConversationState(t *testing.T) {
	providerData := json.RawMessage(`{"provider":"test","data":{"opaque":"state"}}`)
	generator := &generatorStub{
		info: llm.ModelInfo{
			Provider: "test", Model: "model",
			Capabilities: []llm.Capability{llm.CapabilityGeneration, llm.CapabilityTools},
		},
		response: &llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []llm.Part{{Text: "checking"}},
				ToolCalls: []llm.ToolCall{{
					ID: "call-2", Name: "lookup", Arguments: json.RawMessage(`{"query":"next"}`),
				}},
				ProviderData: providerData,
			},
			Usage: &llm.Usage{InputTokens: 8, OutputTokens: 3, TotalTokens: 11},
		},
	}
	adapted, err := Adapt(generator)
	require.NoError(t, err)

	result, err := adapted.GenerateWithTools(
		context.Background(),
		[]core.ChatMessage{
			{Role: "system", Content: []core.ContentBlock{core.NewTextBlock("be useful")}},
			{
				Role: "assistant", ToolCalls: []core.ToolCall{{
					ID: "call-1", Name: "lookup", Arguments: map[string]any{"query": "first"},
				}},
				ProviderData: map[string]any{"provider": "test", "turn": float64(1)},
			},
			{
				Role: "tool", ToolResult: &core.ChatToolResult{
					ToolCallID: "call-1", Name: "lookup",
					Content: []core.ContentBlock{core.NewTextBlock("found")},
				},
			},
		},
		[]map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name": "lookup", "description": "look something up", "strict": true,
				"parameters": map[string]any{
					"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}},
				},
			},
		}},
	)
	require.NoError(t, err)

	require.Len(t, generator.request.Messages, 3)
	assert.Equal(t, llm.RoleSystem, generator.request.Messages[0].Role)
	require.Len(t, generator.request.Messages[1].ToolCalls, 1)
	assert.JSONEq(t, `{"query":"first"}`, string(generator.request.Messages[1].ToolCalls[0].Arguments))
	assert.JSONEq(t, `{"provider":"test","turn":1}`, string(generator.request.Messages[1].ProviderData))
	require.Len(t, generator.request.Messages[2].ToolResults, 1)
	assert.Equal(t, "found", generator.request.Messages[2].ToolResults[0].Content[0].Text)

	require.Len(t, generator.request.Tools, 1)
	assert.Equal(t, "lookup", generator.request.Tools[0].Name)
	assert.True(t, generator.request.Tools[0].Strict)
	assert.JSONEq(t, `{"type":"object","properties":{"query":{"type":"string"}}}`, string(generator.request.Tools[0].InputSchema))

	calls, ok := result["tool_calls"].([]core.ToolCall)
	require.True(t, ok)
	require.Len(t, calls, 1)
	assert.Equal(t, core.ToolCall{ID: "call-2", Name: "lookup", Arguments: map[string]any{"query": "next"}}, calls[0])
	assert.Equal(t, "checking", result["content"])
	assert.Equal(t, map[string]any{
		"provider": "test", "data": map[string]any{"opaque": "state"},
	}, result["provider_data"])
}

func TestLegacyProviderDataRejectsDuplicateJSONNames(t *testing.T) {
	providerData, err := legacyProviderData(json.RawMessage(`{"provider":"first","provider":"second"}`))
	require.Error(t, err)
	assert.Nil(t, providerData)
}

func TestCanonicalMessagesPreserveLegacyJSONWireSemantics(t *testing.T) {
	messages, err := canonicalMessages([]core.ChatMessage{{
		Role: "assistant",
		ToolCalls: []core.ToolCall{{
			ID: "call-1", Name: "lookup", Arguments: nil,
		}},
		ProviderData: map[string]any{"z": 1, "a": 2},
	}})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	require.Len(t, messages[0].ToolCalls, 1)
	assert.Equal(t, "null", string(messages[0].ToolCalls[0].Arguments))
	assert.Equal(t, `{"a":2,"z":1}`, string(messages[0].ProviderData))
}

func TestAdaptStreamsTextAndFinalUsage(t *testing.T) {
	stream := &sliceStream{chunks: []llm.Chunk{
		{Content: []llm.Part{{Text: "hel"}}},
		{Content: []llm.Part{{Text: "lo"}}, Usage: &llm.Usage{InputTokens: 2, OutputTokens: 1, TotalTokens: 3}},
	}}
	generator := &generatorStub{
		info: llm.ModelInfo{
			Provider: "test", Model: "model",
			Capabilities: []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming},
		},
		stream: stream,
	}
	adapted, err := Adapt(generator)
	require.NoError(t, err)

	response, err := adapted.StreamGenerate(context.Background(), "hi")
	require.NoError(t, err)
	var chunks []core.StreamChunk
	for chunk := range response.ChunkChannel {
		chunks = append(chunks, chunk)
	}

	require.Len(t, chunks, 3)
	assert.Equal(t, "hel", chunks[0].Content)
	assert.Equal(t, "lo", chunks[1].Content)
	assert.True(t, chunks[2].Done)
	assert.NoError(t, chunks[2].Error)
	assert.Equal(t, &core.TokenInfo{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}, chunks[2].Usage)
	assert.True(t, stream.closed)
}

func TestAdapterDelegatesEmbeddingsWhenProviderSuppliesThem(t *testing.T) {
	embedder := &embeddingStub{
		single: &core.EmbeddingResult{Vector: []float32{1, 2}},
		batch: &core.BatchEmbeddingResult{Embeddings: []core.EmbeddingResult{
			{Vector: []float32{3, 4}},
		}},
	}
	adapted, err := adapt(&generatorStub{info: llm.ModelInfo{
		Provider: "test", Model: "model", Capabilities: []llm.Capability{llm.CapabilityGeneration},
	}}, nil, nil, embedder)
	require.NoError(t, err)
	assert.Contains(t, adapted.Capabilities(), core.CapabilityEmbedding)

	single, err := adapted.CreateEmbedding(context.Background(), "single")
	require.NoError(t, err)
	assert.Equal(t, "single", embedder.input)
	assert.Equal(t, []float32{1, 2}, single.Vector)

	batch, err := adapted.CreateEmbeddings(context.Background(), []string{"one", "two"})
	require.NoError(t, err)
	assert.Equal(t, []string{"one", "two"}, embedder.inputs)
	require.Len(t, batch.Embeddings, 1)
	assert.Equal(t, []float32{3, 4}, batch.Embeddings[0].Vector)
}

func TestAdaptRejectsNilGeneratorAndEmbeddings(t *testing.T) {
	adapted, err := Adapt(nil)
	require.Error(t, err)
	assert.Nil(t, adapted)

	adapted, err = Adapt(&generatorStub{info: llm.ModelInfo{
		Provider: "test", Model: "model", Capabilities: []llm.Capability{llm.CapabilityGeneration},
	}})
	require.NoError(t, err)
	_, err = adapted.CreateEmbedding(context.Background(), "text")
	require.Error(t, err)
	_, err = adapted.CreateEmbeddings(context.Background(), []string{"text"})
	require.Error(t, err)
}
