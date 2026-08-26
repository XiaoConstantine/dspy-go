package llms

import (
	"context"
	jsonv2 "encoding/json/v2"
	stderrors "errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAICompatibleEmbeddingsRemainInDspy(t *testing.T) {
	var requests []struct {
		Authorization string
		Body          map[string]any
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := jsonv2.UnmarshalRead(request.Body, &body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requests = append(requests, struct {
			Authorization string
			Body          map[string]any
		}{request.Header.Get("Authorization"), body})
		w.Header().Set("Content-Type", "application/json")
		if _, ok := body["input"].([]any); ok {
			fmt.Fprint(w, `{"data":[{"embedding":[3,4],"index":1},{"embedding":[1,2],"index":0}],"model":"custom-embedding","usage":{"total_tokens":6}}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small","usage":{"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)

	model, err := NewOpenAICompatible("openai", core.ModelOpenAIGPT4o, server.URL, WithAPIKey("test-key"))
	require.NoError(t, err)
	assert.Contains(t, model.Capabilities(), core.CapabilityEmbedding)

	single, err := model.CreateEmbedding(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2}, single.Vector)
	assert.Equal(t, 2, single.TokenCount)

	batch, err := model.CreateEmbeddings(
		context.Background(),
		[]string{"one", "two"},
		core.WithModel("custom-embedding"),
	)
	require.NoError(t, err)
	require.NoError(t, batch.Error)
	require.Len(t, batch.Embeddings, 2)
	assert.Equal(t, []float32{1, 2}, batch.Embeddings[0].Vector)
	assert.Equal(t, 0, batch.Embeddings[0].Metadata["index"])
	assert.Equal(t, []float32{3, 4}, batch.Embeddings[1].Vector)
	assert.Equal(t, 1, batch.Embeddings[1].Metadata["index"])
	assert.Equal(t, 3, batch.Embeddings[1].TokenCount)

	require.Len(t, requests, 2)
	for _, request := range requests {
		assert.Equal(t, "Bearer test-key", request.Authorization)
	}
	assert.Equal(t, "text-embedding-3-small", requests[0].Body["model"])
	assert.Equal(t, "custom-embedding", requests[1].Body["model"])
}

func TestOpenAICompatibleBatchEmbeddingsRejectMalformedIndexes(t *testing.T) {
	tests := []struct {
		name     string
		response string
	}{
		{
			name:     "count mismatch",
			response: `{"data":[{"embedding":[1,2],"index":0}]}`,
		},
		{
			name:     "out of range index",
			response: `{"data":[{"embedding":[1,2],"index":0},{"embedding":[3,4],"index":2}]}`,
		},
		{
			name:     "duplicate index",
			response: `{"data":[{"embedding":[1,2],"index":0},{"embedding":[3,4],"index":0}]}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, test.response)
			}))
			t.Cleanup(server.Close)

			model, err := NewOpenAICompatible("localai", core.ModelLocalAICodeLlama, server.URL)
			require.NoError(t, err)
			batch, err := model.CreateEmbeddings(context.Background(), []string{"one", "two"})
			require.Error(t, err)
			assert.Nil(t, batch)
			requireDSPyErrorCode(t, err, dspyerrors.InvalidResponse)
		})
	}
}

func TestOpenAICompatibleEmbeddingsRejectDuplicateJSONNames(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"data":[{"embedding":[1,2],"index":0,"index":0}]}`)
	}))
	t.Cleanup(server.Close)

	model, err := NewOpenAICompatible("localai", core.ModelLocalAICodeLlama, server.URL)
	require.NoError(t, err)
	batch, err := model.CreateEmbeddings(context.Background(), []string{"one"})
	require.NoError(t, err)
	require.NotNil(t, batch)
	requireDSPyErrorCode(t, batch.Error, dspyerrors.InvalidResponse)
}

func TestOllamaEmbeddingUsesConfiguredModelAndOpenAIEndpoint(t *testing.T) {
	var model string
	var path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		path = request.URL.Path
		var body struct {
			Model string `json:"model"`
		}
		if err := jsonv2.UnmarshalRead(request.Body, &body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		model = body.Model
		fmt.Fprint(w, `{"data":[{"embedding":[1,2],"index":0}],"model":"nomic-embed-text"}`)
	}))
	t.Cleanup(server.Close)

	llm, err := NewOllamaLLM("nomic-embed-text", WithBaseURL(server.URL))
	require.NoError(t, err)
	result, err := llm.CreateEmbedding(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 2}, result.Vector)
	assert.Equal(t, "/v1/embeddings", path)
	assert.Equal(t, "nomic-embed-text", model)
}

func TestGeminiEmbeddingsRemainInDspy(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")

	type capturedRequest struct {
		Path  string
		Key   string
		Body  map[string]any
		Error error
	}
	var (
		mu       sync.Mutex
		requests []capturedRequest
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var body map[string]any
		err := jsonv2.UnmarshalRead(request.Body, &body)
		mu.Lock()
		requests = append(requests, capturedRequest{
			Path: request.URL.Path, Key: request.Header.Get("x-goog-api-key"), Body: body, Error: err,
		})
		requestIndex := len(requests)
		mu.Unlock()
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"embeddings":[{"values":[%d,%d]}]}`, requestIndex, requestIndex+1)
	}))
	t.Cleanup(server.Close)

	model, err := NewGeminiLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "google", APIKey: "test-key", BaseURL: server.URL,
	}, core.ModelGoogleGeminiFlash)
	require.NoError(t, err)
	assert.Contains(t, model.Capabilities(), core.CapabilityEmbedding)

	batch, err := model.CreateEmbeddings(
		context.Background(),
		[]string{"one", "two"},
		core.WithBatchSize(1),
		core.WithParams(map[string]any{
			"task_type":             "RETRIEVAL_DOCUMENT",
			"title":                 "document",
			"output_dimensionality": 128,
		}),
	)
	require.NoError(t, err)
	require.NoError(t, batch.Error)
	require.Len(t, batch.Embeddings, 2)
	assert.Equal(t, []float32{1, 2}, batch.Embeddings[0].Vector)
	assert.Equal(t, []float32{2, 3}, batch.Embeddings[1].Vector)

	mu.Lock()
	captured := append([]capturedRequest(nil), requests...)
	mu.Unlock()
	require.Len(t, captured, 2)
	for _, request := range captured {
		require.NoError(t, request.Error)
		assert.Equal(t, "/v1beta/models/gemini-embedding-2:batchEmbedContents", request.Path)
		assert.Equal(t, "test-key", request.Key)
		require.Len(t, request.Body["requests"], 1)
		entry := request.Body["requests"].([]any)[0].(map[string]any)
		assert.Equal(t, "models/gemini-embedding-2", entry["model"])
		assert.Equal(t, "RETRIEVAL_DOCUMENT", entry["taskType"])
		assert.Equal(t, "document", entry["title"])
		assert.Equal(t, float64(128), entry["outputDimensionality"])
	}
}

func requireDSPyErrorCode(t *testing.T, err error, expected dspyerrors.ErrorCode) {
	t.Helper()
	actual, ok := stderrors.AsType[*dspyerrors.Error](err)
	require.True(t, ok, "error %v does not contain a dspy error", err)
	assert.Equal(t, expected, actual.Code())
}
