package llms

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamsReportMalformedJSONFrames(t *testing.T) {
	openAIFrames := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"partial"}}]}`,
		`data: {"duplicate":1,"duplicate":2}`,
	}, "\n") + "\n"
	nativeFrames := strings.Join([]string{
		`{"response":"partial","done":false}`,
		`{"duplicate":1,"duplicate":2}`,
	}, "\n") + "\n"

	tests := []struct {
		name      string
		provider  string
		model     string
		body      string
		newStream func(*testing.T, string) *core.StreamResponse
	}{
		{
			name:     "OpenAI",
			provider: "openai",
			model:    string(core.ModelOpenAIGPT4),
			body:     openAIFrames,
			newStream: func(t *testing.T, baseURL string) *core.StreamResponse {
				llm, err := NewOpenAILLM(
					core.ModelOpenAIGPT4,
					WithAPIKey("test-key"),
					WithOpenAIBaseURL(baseURL),
				)
				require.NoError(t, err)
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
		{
			name:     "Gemini",
			provider: "google",
			model:    string(core.ModelGoogleGeminiFlash),
			body: strings.Join([]string{
				`data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`,
				`data: {"duplicate":1,"duplicate":2}`,
			}, "\n") + "\n",
			newStream: func(t *testing.T, baseURL string) *core.StreamResponse {
				llm := &GeminiLLM{
					apiKey: "test-key",
					BaseLLM: core.NewBaseLLM(
						"google",
						core.ModelGoogleGeminiFlash,
						[]core.Capability{core.CapabilityStreaming},
						&core.EndpointConfig{
							BaseURL:    baseURL,
							Path:       "/models/gemini-2.5-flash:generateContent",
							Headers:    map[string]string{"Content-Type": "application/json"},
							TimeoutSec: 30,
						},
					),
				}
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
		{
			name:     "LlamaCPP",
			provider: "llamacpp",
			body: strings.Join([]string{
				`data: {"content":"partial","stop":false}`,
				`data: {"duplicate":1,"duplicate":2}`,
			}, "\n") + "\n",
			newStream: func(t *testing.T, baseURL string) *core.StreamResponse {
				llm, err := NewLlamacppLLM(baseURL)
				require.NoError(t, err)
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
		{
			name:     "Ollama native",
			provider: "ollama",
			model:    "llama3:8b",
			body:     nativeFrames,
			newStream: func(t *testing.T, baseURL string) *core.StreamResponse {
				llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithNativeAPI())
				require.NoError(t, err)
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
		{
			name:     "Ollama OpenAI compatible",
			provider: "ollama",
			model:    "llama3:8b",
			body:     openAIFrames,
			newStream: func(t *testing.T, baseURL string) *core.StreamResponse {
				llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithOpenAIAPI())
				require.NoError(t, err)
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = fmt.Fprint(w, test.body)
			}))
			defer server.Close()

			stream := test.newStream(t, server.URL)
			var contents []string
			var streamErrors []error
			doneChunks := 0
			for chunk := range stream.ChunkChannel {
				if chunk.Content != "" {
					contents = append(contents, chunk.Content)
				}
				if chunk.Error != nil {
					streamErrors = append(streamErrors, chunk.Error)
				}
				if chunk.Done {
					doneChunks++
				}
			}

			assert.Equal(t, []string{"partial"}, contents)
			assert.Zero(t, doneChunks)
			require.Len(t, streamErrors, 1)
			var typedErr *dspyerrors.Error
			require.ErrorAs(t, streamErrors[0], &typedErr)
			assert.Equal(t, dspyerrors.InvalidResponse, typedErr.Code())
			assert.Equal(t, test.provider, typedErr.Fields()["provider"])
			assert.Equal(t, test.model, typedErr.Fields()["model"])
		})
	}
}
