package llms

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAILLM_GenerateWithContent(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	expectedURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.True(t, strings.HasPrefix(r.Header.Get("Authorization"), "Bearer "))

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var req openai.ChatCompletionRequest
		require.NoError(t, json.Unmarshal(body, &req))
		require.Len(t, req.Messages, 1)
		require.True(t, req.Messages[0].Content.IsMultimodal())

		parts := req.Messages[0].Content.Parts()
		require.Len(t, parts, 2)
		assert.Equal(t, "text", parts[0].Type)
		assert.Equal(t, "describe", parts[0].Text)
		assert.Equal(t, "image_url", parts[1].Type)
		require.NotNil(t, parts[1].ImageURL)
		assert.Equal(t, expectedURL, parts[1].ImageURL.URL)

		resp := openai.ChatCompletionResponse{
			ID:    "vision-test",
			Model: "gpt-4o-mini",
			Choices: []openai.ChatChoice{{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: openai.TextContent("a small PNG"),
				},
				FinishReason: "stop",
			}},
			Usage: openai.CompletionUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		}
		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(resp))
	}))
	defer server.Close()

	llm, err := NewOpenAILLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}, core.ModelOpenAIGPT4oMini)
	require.NoError(t, err)
	assert.NotContains(t, llm.Capabilities(), core.CapabilityVision)
	assert.NotContains(t, llm.Capabilities(), core.CapabilityMultimodal)

	response, err := llm.GenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock("describe"),
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Equal(t, "a small PNG", response.Content)
	require.NotNil(t, response.Usage)
	assert.Equal(t, 8, response.Usage.TotalTokens)
}

func TestOpenAILLM_GenerateWithContent_EmptyImage(t *testing.T) {
	llm, err := NewOpenAILLM(core.ModelOpenAIGPT4oMini, WithAPIKey("test-api-key"))
	require.NoError(t, err)

	_, err = llm.GenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock("describe"),
		core.NewImageBlock(nil, "image/png"),
	})
	require.Error(t, err)
	dsyErr, ok := err.(*errors.Error)
	require.True(t, ok)
	assert.Equal(t, errors.InvalidInput, dsyErr.Code())
}

func TestOpenAILLM_GenerateWithContent_AudioUnsupported(t *testing.T) {
	llm, err := NewOpenAILLM(core.ModelOpenAIGPT4oMini, WithAPIKey("test-api-key"))
	require.NoError(t, err)

	_, err = llm.GenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock("transcribe"),
		{Type: core.FieldTypeAudio, Data: []byte("fake"), MimeType: "audio/wav"},
	})
	require.Error(t, err)
	dsyErr, ok := err.(*errors.Error)
	require.True(t, ok)
	assert.Equal(t, errors.UnsupportedOperation, dsyErr.Code())
}

func TestOpenAILLM_GenerateWithContent_EmptyTextRejected(t *testing.T) {
	llm, err := NewOpenAILLM(core.ModelOpenAIGPT4oMini, WithAPIKey("test-api-key"))
	require.NoError(t, err)

	_, err = llm.GenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock(""),
	})
	require.Error(t, err)
	dsyErr, ok := err.(*errors.Error)
	require.True(t, ok)
	assert.Equal(t, errors.InvalidInput, dsyErr.Code())
	assert.Contains(t, err.Error(), "no content provided")
}

func TestOpenAILLM_GenerateWithContent_PreservesTextOnlyWhitespace(t *testing.T) {
	assertGenerateWithContentSendsString(t, "  prompt  ", "  prompt  ")
}

func TestOpenAILLM_GenerateWithContent_KeepsWhitespaceOnlyText(t *testing.T) {
	assertGenerateWithContentSendsString(t, "   \t", "   \t")
}

func assertGenerateWithContentSendsString(t *testing.T, blockText, wantContent string) {
	t.Helper()
	var gotContent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var raw map[string]any
		require.NoError(t, json.Unmarshal(body, &raw))
		messages, ok := raw["messages"].([]any)
		require.True(t, ok)
		require.Len(t, messages, 1)
		msg, ok := messages[0].(map[string]any)
		require.True(t, ok)
		content, isString := msg["content"].(string)
		assert.True(t, isString, "text-only content must marshal as a JSON string")
		gotContent = content

		resp := openai.ChatCompletionResponse{
			ID:    "whitespace",
			Model: "gpt-4o-mini",
			Choices: []openai.ChatChoice{{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: openai.TextContent("ok"),
				},
				FinishReason: "stop",
			}},
		}
		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(resp))
	}))
	defer server.Close()

	llm, err := NewOpenAILLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}, core.ModelOpenAIGPT4oMini)
	require.NoError(t, err)

	response, err := llm.GenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock(blockText),
	})
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Content)
	assert.Equal(t, wantContent, gotContent)
}

func TestOpenAILLM_Generate_StillSendsStringContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var raw map[string]any
		require.NoError(t, json.Unmarshal(body, &raw))
		messages, ok := raw["messages"].([]any)
		require.True(t, ok)
		require.Len(t, messages, 1)
		msg, ok := messages[0].(map[string]any)
		require.True(t, ok)
		_, isString := msg["content"].(string)
		assert.True(t, isString, "text-only content must marshal as a JSON string")

		resp := openai.ChatCompletionResponse{
			ID:    "text-only",
			Model: "gpt-4",
			Choices: []openai.ChatChoice{{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: openai.TextContent("ok"),
				},
				FinishReason: "stop",
			}},
		}
		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(resp))
	}))
	defer server.Close()

	llm, err := NewOpenAILLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}, core.ModelOpenAIGPT4)
	require.NoError(t, err)

	response, err := llm.Generate(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Content)
}

func TestOpenAILLM_StreamGenerateWithContent(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var req openai.ChatCompletionRequest
		require.NoError(t, json.Unmarshal(body, &req))
		assert.True(t, req.Stream)
		require.Len(t, req.Messages, 1)
		assert.True(t, req.Messages[0].Content.IsMultimodal())

		w.Header().Set("Content-Type", "text/plain")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		chunks := []string{
			`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"seen"},"finish_reason":null}]}`,
			`data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" image"},"finish_reason":"stop"}]}`,
			"data: [DONE]",
		}
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		for _, chunk := range chunks {
			fmt.Fprintf(w, "%s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer server.Close()

	llm, err := NewOpenAILLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}, core.ModelOpenAIGPT4oMini)
	require.NoError(t, err)

	stream, err := llm.StreamGenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock("what is this?"),
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)

	var parts []string
	for chunk := range stream.ChunkChannel {
		require.NoError(t, chunk.Error)
		if chunk.Done {
			break
		}
		if chunk.Content != "" {
			parts = append(parts, chunk.Content)
		}
	}
	assert.Equal(t, []string{"seen", " image"}, parts)
}

func TestOpenAILLM_StreamGenerateWithContent_KeepsWhitespaceOnlyText(t *testing.T) {
	var gotContent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		var raw map[string]any
		require.NoError(t, json.Unmarshal(body, &raw))
		messages, ok := raw["messages"].([]any)
		require.True(t, ok)
		require.Len(t, messages, 1)
		msg, ok := messages[0].(map[string]any)
		require.True(t, ok)
		content, isString := msg["content"].(string)
		assert.True(t, isString, "text-only content must marshal as a JSON string")
		gotContent = content

		w.Header().Set("Content-Type", "text/plain")
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		fmt.Fprintf(w, "%s\n\n", `data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`)
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	llm, err := NewOpenAILLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai",
		APIKey: "test-api-key",
		Endpoint: &core.EndpointConfig{
			BaseURL:    server.URL,
			TimeoutSec: 30,
		},
	}, core.ModelOpenAIGPT4oMini)
	require.NoError(t, err)

	stream, err := llm.StreamGenerateWithContent(context.Background(), []core.ContentBlock{
		core.NewTextBlock("   \t"),
	})
	require.NoError(t, err)
	for chunk := range stream.ChunkChannel {
		require.NoError(t, chunk.Error)
	}
	assert.Equal(t, "   \t", gotContent)
}

func assertFlattenedImageWire(t *testing.T, content openai.MessageContent, wantTextFragments ...string) {
	t.Helper()
	assert.False(t, content.IsMultimodal())
	for _, fragment := range wantTextFragments {
		assert.Contains(t, content.Text(), fragment)
	}
	raw, err := json.Marshal(content)
	require.NoError(t, err)
	require.NotEmpty(t, raw)
	assert.Equal(t, byte('"'), raw[0], "flattened content must marshal as a JSON string")
	assert.NotContains(t, string(raw), "image_url")
}

func TestCoreContentBlocksToMessageContent_PreservesTextOnlyWhitespace(t *testing.T) {
	content, err := coreContentBlocksToMessageContent([]core.ContentBlock{
		core.NewTextBlock("  prompt  "),
	})
	require.NoError(t, err)
	assert.False(t, content.IsMultimodal())

	raw, err := json.Marshal(content)
	require.NoError(t, err)
	require.NotEmpty(t, raw)
	assert.Equal(t, byte('"'), raw[0], "text-only content must marshal as a JSON string")

	var got string
	require.NoError(t, json.Unmarshal(raw, &got))
	assert.Equal(t, "  prompt  ", got)
}

func TestCoreContentBlocksToMessageContent_PreservesWhitespaceWithImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	content, err := coreContentBlocksToMessageContent([]core.ContentBlock{
		core.NewTextBlock("  prompt  "),
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)
	require.True(t, content.IsMultimodal())
	parts := content.Parts()
	require.GreaterOrEqual(t, len(parts), 2)
	assert.Equal(t, "text", parts[0].Type)
	assert.Equal(t, "  prompt  ", parts[0].Text)
	assert.Equal(t, "image_url", parts[1].Type)
}

func TestCoreContentBlocksToMessageContent_KeepsWhitespaceOnlyText(t *testing.T) {
	content, err := coreContentBlocksToMessageContent([]core.ContentBlock{
		core.NewTextBlock("   \t"),
	})
	require.NoError(t, err)
	assert.False(t, content.IsMultimodal())

	raw, err := json.Marshal(content)
	require.NoError(t, err)
	require.NotEmpty(t, raw)
	assert.Equal(t, byte('"'), raw[0], "text-only content must marshal as a JSON string")

	var got string
	require.NoError(t, json.Unmarshal(raw, &got))
	assert.Equal(t, "   \t", got)
}

func TestConvertCoreChatMessagesToOpenAI_PreservesUserImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "user",
			Content: []core.ContentBlock{
				core.NewTextBlock("what is this?"),
				core.NewImageBlock(pngBytes, "image/png"),
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	require.True(t, messages[0].Content.IsMultimodal())
	parts := messages[0].Content.Parts()
	require.Len(t, parts, 2)
	assert.Equal(t, "text", parts[0].Type)
	assert.Equal(t, "image_url", parts[1].Type)
	require.NotNil(t, parts[1].ImageURL)
	assert.True(t, strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,"))
}

func TestConvertCoreChatMessagesToOpenAI_UserRoleCaseInsensitive(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "User",
			Content: []core.ContentBlock{
				core.NewTextBlock("what is this?"),
				core.NewImageBlock(pngBytes, "image/png"),
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "user", messages[0].Role)
	require.True(t, messages[0].Content.IsMultimodal())
	parts := messages[0].Content.Parts()
	require.Len(t, parts, 2)
	assert.Equal(t, "image_url", parts[1].Type)
}

func TestConvertCoreChatMessagesToOpenAI_NonUserRolesFlattenImages(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	tests := []struct {
		name string
		role string
		text string
	}{
		{name: "system", role: "system", text: "use these brand refs"},
		{name: "assistant", role: "assistant", text: "Looks like a tabby."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
				{
					Role: tt.role,
					Content: []core.ContentBlock{
						core.NewTextBlock(tt.text),
						core.NewImageBlock(pngBytes, "image/png"),
					},
				},
			})
			require.NoError(t, err)
			require.Len(t, messages, 1)
			assert.Equal(t, tt.role, messages[0].Role)
			assertFlattenedImageWire(t, messages[0].Content, tt.text, "[Image: image/png, 4 bytes]")
		})
	}
}

func TestConvertCoreChatMessagesToOpenAI_ToolRoleFlattensImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "tool",
			ToolResult: &core.ChatToolResult{
				ToolCallID: "call_1",
				Name:       "vision_tool",
				Content: []core.ContentBlock{
					core.NewTextBlock("tool says"),
					core.NewImageBlock(pngBytes, "image/png"),
				},
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "tool", messages[0].Role)
	assertFlattenedImageWire(t, messages[0].Content, "tool says", "[Image: image/png, 4 bytes]")
	assert.Equal(t, "call_1", messages[0].ToolCallID)
}

func TestConvertCoreChatMessagesToOpenAI_ToolRoleWithoutToolResultFlattensImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "tool",
			Content: []core.ContentBlock{
				core.NewTextBlock("orphan tool payload"),
				core.NewImageBlock(pngBytes, "image/png"),
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "tool", messages[0].Role)
	assertFlattenedImageWire(t, messages[0].Content, "orphan tool payload", "[Image: image/png, 4 bytes]")
	assert.Empty(t, messages[0].ToolCallID)
}

func TestConvertCoreChatMessagesToOpenAI_ToolRoleWithoutToolResultUsesMessageContent(t *testing.T) {
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "tool",
			Content: []core.ContentBlock{
				core.NewTextBlock("plain tool payload"),
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "tool", messages[0].Role)
	assert.False(t, messages[0].Content.IsMultimodal())
	assert.Equal(t, "plain tool payload", messages[0].Content.Text())
	assert.Empty(t, messages[0].ToolCallID)
}

func TestConvertCoreChatMessagesToOpenAI_ToolResultWinsOverMessageContent(t *testing.T) {
	messages, err := convertCoreChatMessagesToOpenAI([]core.ChatMessage{
		{
			Role: "tool",
			Content: []core.ContentBlock{
				core.NewTextBlock("ignored message content"),
			},
			ToolResult: &core.ChatToolResult{
				ToolCallID: "call_2",
				Name:       "vision_tool",
				Content: []core.ContentBlock{
					core.NewTextBlock("from tool result"),
				},
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, messages, 1)
	assert.Equal(t, "from tool result", messages[0].Content.Text())
	assert.Equal(t, "call_2", messages[0].ToolCallID)
}

func TestContentBlocksToOpenAIParts_ImageOnlyKeepsImageURL(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	parts, err := contentBlocksToOpenAIParts([]core.ContentBlock{
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, "image_url", parts[0].Type)
	require.NotNil(t, parts[0].ImageURL)
	assert.True(t, strings.HasPrefix(parts[0].ImageURL.URL, "data:image/png;base64,"))
}

func TestContentBlocksToOpenAIParts_WhitespaceTextWithImagePreserved(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	parts, err := contentBlocksToOpenAIParts([]core.ContentBlock{
		core.NewTextBlock("   \t"),
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)
	require.Len(t, parts, 2)
	assert.Equal(t, "text", parts[0].Type)
	assert.Equal(t, "   \t", parts[0].Text)
	assert.Equal(t, "image_url", parts[1].Type)
}

func TestContentBlocksToOpenAIParts_EmptyTextWithImageDropsEmptyText(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	parts, err := contentBlocksToOpenAIParts([]core.ContentBlock{
		core.NewTextBlock(""),
		core.NewImageBlock(pngBytes, "image/png"),
	})
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, "image_url", parts[0].Type)
}

func TestOpenAIGenerateWithContentLive(t *testing.T) {
	if os.Getenv("DSPY_GO_OPENAI_VISION_LIVE") != "1" {
		t.Skip("set DSPY_GO_OPENAI_VISION_LIVE=1 to run live OpenAI vision test")
	}

	apiKey := os.Getenv("DSPY_GO_OPENAI_VISION_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		t.Skip("set OPENAI_API_KEY or DSPY_GO_OPENAI_VISION_API_KEY")
	}

	// BaseURL is the API host only; OpenAILLM appends Path "/v1/chat/completions".
	// Example Polypus: http://127.0.0.1:1320 (not .../v1).
	baseURL := os.Getenv("DSPY_GO_OPENAI_VISION_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	model := os.Getenv("DSPY_GO_OPENAI_VISION_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}

	opts := []OpenAIOption{WithAPIKey(apiKey), WithOpenAIBaseURL(baseURL)}
	llm, err := NewOpenAILLM(core.ModelID(model), opts...)
	require.NoError(t, err)

	// Default fixture is a small synthetic PNG (>=10px for Cloudflare Workers AI).
	imageData, mimeType := loadOpenAIVisionFixtureImage(t)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	response, err := llm.GenerateWithContent(ctx, []core.ContentBlock{
		core.NewTextBlock("In one short sentence, describe what you see in this image."),
		core.NewImageBlock(imageData, mimeType),
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.NotEmpty(t, strings.TrimSpace(response.Content))
}

func loadOpenAIVisionFixtureImage(t *testing.T) ([]byte, string) {
	t.Helper()
	// Default: compact synthetic PNG (>=10px for Cloudflare Workers AI).
	// Use examples/multimodal/cat.jpeg only when DSPY_GO_OPENAI_VISION_USE_CAT=1.
	if os.Getenv("DSPY_GO_OPENAI_VISION_USE_CAT") == "1" {
		_, thisFile, _, ok := runtime.Caller(0)
		if ok {
			path := filepath.Join(filepath.Dir(thisFile), "..", "..", "examples", "multimodal", "cat.jpeg")
			if data, err := os.ReadFile(path); err == nil && len(data) > 0 {
				return data, "image/jpeg"
			}
		}
	}
	return mustMakeSolidPNG(t, 16, 16), "image/png"
}

func mustMakeSolidPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, img))
	return buf.Bytes()
}

func TestContentBlocksToOpenAIParts_ImageMIME(t *testing.T) {
	jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46}
	garbageData := []byte("not-an-image")

	t.Run("infer jpeg from empty mime", func(t *testing.T) {
		parts, err := contentBlocksToOpenAIParts([]core.ContentBlock{
			core.NewImageBlock(jpegData, ""),
		})
		require.NoError(t, err)
		require.Len(t, parts, 1)
		assert.Equal(t, "image_url", parts[0].Type)
		require.NotNil(t, parts[0].ImageURL)
		assert.True(t, strings.HasPrefix(parts[0].ImageURL.URL, "data:image/jpeg;base64,"))
	})

	t.Run("reject jpeg bytes with png mime", func(t *testing.T) {
		_, err := contentBlocksToOpenAIParts([]core.ContentBlock{
			core.NewImageBlock(jpegData, "image/png"),
		})
		require.Error(t, err)
		dsyErr, ok := err.(*errors.Error)
		require.True(t, ok)
		assert.Equal(t, errors.InvalidInput, dsyErr.Code())
	})

	t.Run("reject garbage with empty mime", func(t *testing.T) {
		_, err := contentBlocksToOpenAIParts([]core.ContentBlock{
			core.NewImageBlock(garbageData, ""),
		})
		require.Error(t, err)
		dsyErr, ok := err.(*errors.Error)
		require.True(t, ok)
		assert.Equal(t, errors.InvalidInput, dsyErr.Code())
	})
}
