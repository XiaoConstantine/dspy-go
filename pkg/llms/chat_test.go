package llms

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	llm "github.com/XiaoConstantine/llm-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type chatStubGenerator struct {
	info         llm.ModelInfo
	lastRequest  llm.Request
	response     *llm.Response
	streamChunks []llm.Chunk
}

func (g *chatStubGenerator) Info() llm.ModelInfo { return g.info }

func (g *chatStubGenerator) Generate(_ context.Context, req llm.Request) (*llm.Response, error) {
	g.lastRequest = req
	return g.response, nil
}

func (g *chatStubGenerator) Stream(_ context.Context, req llm.Request) (llm.Stream, error) {
	g.lastRequest = req
	return &stubLLMStream{chunks: g.streamChunks}, nil
}

type stubLLMStream struct {
	chunks []llm.Chunk
	idx    int
}

func (s *stubLLMStream) Recv() (llm.Chunk, error) {
	if s.idx >= len(s.chunks) {
		return llm.Chunk{}, io.EOF
	}
	c := s.chunks[s.idx]
	s.idx++
	return c, nil
}

func (s *stubLLMStream) Close() error { return nil }

func TestGeneratorLLM_ChatCapabilitiesReasoningOrderAndRefusal(t *testing.T) {
	stub := &chatStubGenerator{
		info: llm.ModelInfo{
			Provider:     "openai",
			Model:        "gpt-5",
			Capabilities: []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityJSON},
			Reasoning:    true,
		},
		response: &llm.Response{
			ID:               "resp-1",
			Model:            "gpt-5",
			Message:          llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "<answer>42</answer>"}}},
			ReasoningSummary: "thought carefully",
			FinishReason:     llm.FinishReasonStop,
			Usage:            &llm.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
		},
	}

	adapted, err := Adapt(stub)
	require.NoError(t, err)
	assert.True(t, core.HasCapability(adapted, core.CapabilityReasoning), "ModelInfo.Reasoning must map to CapabilityReasoning")

	orderedSchema := json.RawMessage(`{"type":"object","properties":{"reasoning":{"type":"string"},"answer":{"type":"string"}},"required":["reasoning","answer"],"additionalProperties":false}`)
	resp, err := adapted.Chat(context.Background(), core.ChatRequest{
		Messages: []core.ChatMessage{
			{Role: "system", Content: []core.ContentBlock{core.NewTextBlock("sys")}},
			{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("hi")}},
		},
		Tools: []core.ChatTool{{
			Name:        "submit",
			Description: "Submit final output",
			InputSchema: orderedSchema,
			Strict:      true,
		}},
		ToolChoice:      core.ToolChoice{Mode: core.ToolChoiceNamed, Name: "submit"},
		ReasoningEffort: core.ReasoningEffortLow,
	})
	require.NoError(t, err)
	assert.Equal(t, "<answer>42</answer>", resp.Content)
	assert.Equal(t, "thought carefully", resp.ReasoningSummary)
	assert.Equal(t, core.FinishReasonStop, resp.FinishReason)
	require.Len(t, stub.lastRequest.Tools, 1)
	assert.Equal(t, string(orderedSchema), string(stub.lastRequest.Tools[0].InputSchema), "tool schema order must be preserved verbatim")
	assert.Equal(t, llm.ToolChoiceNamed, stub.lastRequest.ToolChoice.Mode)
	assert.Equal(t, "submit", stub.lastRequest.ToolChoice.Name)
	assert.Equal(t, llm.ReasoningEffortLow, stub.lastRequest.ReasoningEffort)

	// OpenAI refusal in ProviderData maps to Refusal + FinishReasonRefusal.
	stub.response = &llm.Response{
		ID:           "resp-2",
		Model:        "gpt-5",
		FinishReason: llm.FinishReasonStop,
		Message: llm.Message{
			Role:         llm.RoleAssistant,
			ProviderData: json.RawMessage(`{"provider":"openai","version":1,"data":{"refusal":"cannot comply"}}`),
		},
	}
	refResp, err := adapted.Chat(context.Background(), core.ChatRequest{
		Messages: []core.ChatMessage{{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("bad")}}},
	})
	require.NoError(t, err)
	require.NotNil(t, refResp.Refusal)
	assert.Equal(t, "cannot comply", *refResp.Refusal)
	assert.Equal(t, core.FinishReasonRefusal, refResp.FinishReason)
}

func TestGeneratorLLM_StreamChatForwardsFinishReasonAndReasoning(t *testing.T) {
	stub := &chatStubGenerator{
		info: llm.ModelInfo{
			Provider:     "anthropic",
			Model:        "claude-sonnet-5",
			Capabilities: []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming},
			Reasoning:    true,
		},
		streamChunks: []llm.Chunk{
			{ReasoningSummary: "step 1"},
			{Content: []llm.Part{{Text: "<answer>hel"}}},
			{Content: []llm.Part{{Text: "lo</answer>"}}},
			{FinishReason: llm.FinishReasonLength, Usage: &llm.Usage{InputTokens: 4, OutputTokens: 6, TotalTokens: 10}},
		},
	}

	adapted, err := Adapt(stub)
	require.NoError(t, err)

	stream, err := adapted.StreamChat(context.Background(), core.ChatRequest{
		Messages:        []core.ChatMessage{{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("hi")}}},
		ReasoningEffort: core.ReasoningEffortMedium,
	})
	require.NoError(t, err)

	var text strings.Builder
	var reasoning strings.Builder
	var terminal core.StreamChunk
	doneCount := 0
	for chunk := range stream.ChunkChannel {
		if chunk.Done {
			doneCount++
			terminal = chunk
			continue
		}
		text.WriteString(chunk.Content)
		reasoning.WriteString(chunk.ReasoningSummary)
	}
	assert.Equal(t, 1, doneCount)
	assert.Equal(t, "<answer>hello</answer>", text.String())
	assert.Equal(t, "step 1", reasoning.String())
	assert.Equal(t, core.FinishReasonLength, terminal.FinishReason)
	require.NotNil(t, terminal.Usage)
	assert.Equal(t, 10, terminal.Usage.TotalTokens)
}
