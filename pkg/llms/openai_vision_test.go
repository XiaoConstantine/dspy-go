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

	// Prefer a small >=10px PNG (Cloudflare Workers AI rejects tiny images).
	// Fall back to examples/multimodal/cat.jpeg when present.
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
	// Prefer a compact synthetic PNG (>=10px for Cloudflare Workers AI).
	// Override with examples/multimodal/cat.jpeg when DSPY_GO_OPENAI_VISION_USE_CAT=1.
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
