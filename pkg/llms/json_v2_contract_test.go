package llms

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/XiaoConstantine/dspy-go/internal/testutil/jsonv2test"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func TestCompleteResponsesFollowJSONV2Contract(t *testing.T) {
	providers := []struct {
		name             string
		valid            []byte
		caseMismatchName string
		newLLM           func(string) (core.LLM, error)
	}{
		{
			name: "OpenAI",
			valid: []byte(`{
				"id":"response-id",
				"model":"gpt-4",
				"choices":[{"index":0,"message":{"role":"assistant","content":"expected"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
			}`),
			caseMismatchName: "CHOICES",
			newLLM: func(baseURL string) (core.LLM, error) {
				return NewOpenAILLM(
					core.ModelOpenAIGPT4,
					WithAPIKey("test-key"),
					WithOpenAIBaseURL(baseURL),
				)
			},
		},
		{
			name: "Gemini",
			valid: []byte(`{
				"candidates":[{"content":{"parts":[{"text":"expected"}]},"finishReason":"STOP"}],
				"usageMetadata":{}
			}`),
			caseMismatchName: "CANDIDATES",
			newLLM: func(baseURL string) (core.LLM, error) {
				return &GeminiLLM{
					apiKey: "test-key",
					BaseLLM: core.NewBaseLLM(
						"google",
						core.ModelGoogleGeminiFlash,
						[]core.Capability{core.CapabilityCompletion},
						&core.EndpointConfig{
							BaseURL:    baseURL,
							Path:       "/models/gemini-2.5-flash:generateContent",
							Headers:    map[string]string{"Content-Type": "application/json"},
							TimeoutSec: 30,
						},
					),
				}, nil
			},
		},
		{
			name:             "LlamaCPP",
			valid:            []byte(`{"content":"expected"}`),
			caseMismatchName: "CONTENT",
			newLLM: func(baseURL string) (core.LLM, error) {
				return NewLlamacppLLM(baseURL)
			},
		},
		{
			name:             "Ollama native",
			valid:            []byte(`{"model":"llama3:8b","response":"expected"}`),
			caseMismatchName: "RESPONSE",
			newLLM: func(baseURL string) (core.LLM, error) {
				return NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithNativeAPI())
			},
		},
		{
			name: "Ollama OpenAI compatible",
			valid: []byte(`{
				"id":"response-id",
				"model":"llama3:8b",
				"choices":[{"index":0,"message":{"role":"assistant","content":"expected"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
			}`),
			caseMismatchName: "CHOICES",
			newLLM: func(baseURL string) (core.LLM, error) {
				return NewOllamaLLM("llama3:8b", WithBaseURL(baseURL), WithOpenAIAPI())
			},
		},
	}

	for _, provider := range providers {
		t.Run(provider.name, func(t *testing.T) {
			contract := completeResponseJSONV2Contract(provider.valid, provider.caseMismatchName)
			jsonv2test.Check(t, func(payload []byte) (*core.LLMResponse, error) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write(payload)
				}))
				defer server.Close()

				llm, err := provider.newLLM(server.URL)
				if err != nil {
					return nil, err
				}
				return llm.Generate(context.Background(), "hello")
			}, contract)
		})
	}
}

func completeResponseJSONV2Contract(valid []byte, caseMismatchName string) jsonv2test.Contract[*core.LLMResponse] {
	checkExpected := func(t testing.TB, response *core.LLMResponse) {
		t.Helper()
		if response == nil {
			t.Fatal("provider returned a nil response")
		}
		if response.Content != "expected" {
			t.Fatalf("response content = %q, want expected", response.Content)
		}
	}

	return jsonv2test.Contract[*core.LLMResponse]{
		Valid:           valid,
		DuplicateMember: jsonv2test.WithObjectMembers(valid, []byte(`"contract_duplicate":1,"contract_duplicate":2`)),
		InvalidUTF8: jsonv2test.WithObjectMembers(valid,
			jsonv2test.InvalidUTF8(`"contract_invalid":"`, `"`)),
		CaseMismatch: jsonv2test.WithObjectMembers(valid,
			[]byte(fmt.Sprintf("%q:%q", caseMismatchName, "wrong"))),
		UnknownMember:      jsonv2test.WithObjectMembers(valid, []byte(`"contract_unknown":true`)),
		CheckValid:         checkExpected,
		CheckCaseMismatch:  checkExpected,
		CheckUnknownMember: checkExpected,
	}
}
