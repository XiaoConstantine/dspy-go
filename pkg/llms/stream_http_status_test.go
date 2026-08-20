package llms

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNativeStreamsReportHTTPStatusErrors(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		statusField string
		newStream   func(*testing.T, string, *http.Client) *core.StreamResponse
	}{
		{
			name:        "Ollama",
			model:       "llama3:8b",
			statusField: "status_code",
			newStream: func(t *testing.T, baseURL string, client *http.Client) *core.StreamResponse {
				llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithNativeAPI())
				require.NoError(t, err)
				llm.GetHTTPClient().Transport = client.Transport
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
		{
			name:        "Gemini",
			model:       string(core.ModelGoogleGeminiFlash),
			statusField: "statusCode",
			newStream: func(t *testing.T, baseURL string, client *http.Client) *core.StreamResponse {
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
						core.WithHTTPClient(client),
					),
				}
				stream, err := llm.StreamGenerate(context.Background(), "hello")
				require.NoError(t, err)
				return stream
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "rate limited", http.StatusTooManyRequests)
			}))
			serverClient := server.Client()

			stream := test.newStream(t, server.URL, serverClient)
			var chunks []core.StreamChunk
			for chunk := range stream.ChunkChannel {
				chunks = append(chunks, chunk)
			}

			require.Len(t, chunks, 1)
			streamErr := chunks[0].Error
			require.Error(t, streamErr)
			assert.Contains(t, streamErr.Error(), "status code 429")
			var typedErr *dspyerrors.Error
			require.ErrorAs(t, streamErr, &typedErr)
			assert.Equal(t, dspyerrors.LLMGenerationFailed, typedErr.Code())
			assert.Equal(t, test.model, typedErr.Fields()["model"])
			assert.Equal(t, http.StatusTooManyRequests, typedErr.Fields()[test.statusField])
		})
	}
}
