package llms

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil/jsonv2test"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type streamJSONContractResult struct {
	contents   []string
	doneChunks int
}

func TestStreamsFollowJSONV2Contract(t *testing.T) {
	sseFrame := func(payload string) []byte {
		return []byte("data: " + payload + "\n")
	}
	nativeFrame := func(payload string) []byte {
		return []byte(payload + "\n")
	}

	providers := []struct {
		name              string
		provider          string
		model             string
		validFrame        string
		caseMismatchFrame string
		frame             func(string) []byte
		newStream         func(string, *http.Client) (*core.StreamResponse, error)
		doneOnEOF         bool
	}{
		{
			name:              "OpenAI",
			provider:          "openai",
			model:             string(core.ModelOpenAIGPT4),
			validFrame:        `{"choices":[{"delta":{"content":"partial"}}]}`,
			caseMismatchFrame: `{"CHOICES":[{"delta":{"content":"wrong"}}]}`,
			frame:             sseFrame,
			newStream: func(baseURL string, client *http.Client) (*core.StreamResponse, error) {
				llm, err := NewOpenAILLM(
					core.ModelOpenAIGPT4,
					WithAPIKey("test-key"),
					WithOpenAIBaseURL(baseURL),
					WithHTTPClient(client),
				)
				if err != nil {
					return nil, err
				}
				return llm.StreamGenerate(context.Background(), "hello")
			},
		},
		{
			name:              "Gemini",
			provider:          "google",
			model:             string(core.ModelGoogleGeminiFlash),
			validFrame:        `{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`,
			caseMismatchFrame: `{"CANDIDATES":[{"content":{"parts":[{"text":"wrong"}]}}]}`,
			frame:             sseFrame,
			newStream: func(baseURL string, client *http.Client) (*core.StreamResponse, error) {
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
				return llm.StreamGenerate(context.Background(), "hello")
			},
		},
		{
			name:              "LlamaCPP",
			provider:          "llamacpp",
			validFrame:        `{"content":"partial","stop":false}`,
			caseMismatchFrame: `{"CONTENT":"wrong","stop":false}`,
			frame:             sseFrame,
			doneOnEOF:         true,
			newStream: func(baseURL string, client *http.Client) (*core.StreamResponse, error) {
				llm, err := NewLlamacppLLM(baseURL)
				if err != nil {
					return nil, err
				}
				llm.GetHTTPClient().Transport = client.Transport
				return llm.StreamGenerate(context.Background(), "hello")
			},
		},
		{
			name:              "Ollama native",
			provider:          "ollama",
			model:             "llama3:8b",
			validFrame:        `{"response":"partial","done":false}`,
			caseMismatchFrame: `{"RESPONSE":"wrong","done":false}`,
			frame:             nativeFrame,
			newStream: func(baseURL string, client *http.Client) (*core.StreamResponse, error) {
				llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithNativeAPI())
				if err != nil {
					return nil, err
				}
				llm.GetHTTPClient().Transport = client.Transport
				return llm.StreamGenerate(context.Background(), "hello")
			},
		},
		{
			name:              "Ollama OpenAI compatible",
			provider:          "ollama",
			model:             "llama3:8b",
			validFrame:        `{"choices":[{"delta":{"content":"partial"}}]}`,
			caseMismatchFrame: `{"CHOICES":[{"delta":{"content":"wrong"}}]}`,
			frame:             sseFrame,
			newStream: func(baseURL string, client *http.Client) (*core.StreamResponse, error) {
				llm, err := NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithOpenAIAPI())
				if err != nil {
					return nil, err
				}
				llm.GetHTTPClient().Transport = client.Transport
				return llm.StreamGenerate(context.Background(), "hello")
			},
		},
	}

	for _, provider := range providers {
		t.Run(provider.name, func(t *testing.T) {
			body := func(frames ...[]byte) []byte {
				var joined []byte
				for _, frame := range frames {
					joined = append(joined, frame...)
				}
				return joined
			}
			valid := provider.frame(provider.validFrame)
			contract := jsonv2test.Contract[streamJSONContractResult]{
				Valid:           valid,
				DuplicateMember: body(valid, provider.frame(`{"contract_duplicate":1,"contract_duplicate":2}`)),
				InvalidUTF8: body(valid, provider.frame(string(
					jsonv2test.InvalidUTF8(`{"contract_invalid":"`, `"}`)))),
				CaseMismatch:  body(valid, provider.frame(provider.caseMismatchFrame)),
				UnknownMember: body(valid, provider.frame(`{"contract_unknown":true}`)),
			}

			checkPartial := func(t testing.TB, result streamJSONContractResult, wantDone int) {
				t.Helper()
				assert.Equal(t, []string{"partial"}, result.contents)
				assert.Equal(t, wantDone, result.doneChunks)
			}
			checkCompatible := func(t testing.TB, result streamJSONContractResult) {
				t.Helper()
				wantDone := 0
				if provider.doneOnEOF {
					wantDone = 1
				}
				checkPartial(t, result, wantDone)
			}
			checkStrictError := func(t testing.TB, result streamJSONContractResult, err error) {
				t.Helper()
				checkPartial(t, result, 0)
				var typedErr *dspyerrors.Error
				require.ErrorAs(t, err, &typedErr)
				assert.Equal(t, dspyerrors.InvalidResponse, typedErr.Code())
				assert.Equal(t, provider.provider, typedErr.Fields()["provider"])
				assert.Equal(t, provider.model, typedErr.Fields()["model"])
			}
			contract.CheckValid = checkCompatible
			contract.CheckDuplicateError = checkStrictError
			contract.CheckInvalidUTF8Error = checkStrictError
			contract.CheckCaseMismatch = checkCompatible
			contract.CheckUnknownMember = checkCompatible

			jsonv2test.Check(t, func(payload []byte) (streamJSONContractResult, error) {
				server := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write(payload)
				}))
				serverClient := server.Client()

				stream, err := provider.newStream(server.URL, serverClient)
				if err != nil {
					return streamJSONContractResult{}, err
				}

				var result streamJSONContractResult
				var streamErrors []error
				for chunk := range stream.ChunkChannel {
					if chunk.Content != "" {
						result.contents = append(result.contents, chunk.Content)
					}
					if chunk.Error != nil {
						streamErrors = append(streamErrors, chunk.Error)
					}
					if chunk.Done {
						result.doneChunks++
					}
				}

				switch len(streamErrors) {
				case 0:
					return result, nil
				case 1:
					return result, streamErrors[0]
				default:
					return result, fmt.Errorf("stream returned %d errors: %v", len(streamErrors), streamErrors)
				}
			}, contract)
		})
	}
}
