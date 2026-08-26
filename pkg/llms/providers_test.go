package llms

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiProviderPreservesLongRequestTimeout(t *testing.T) {
	model, err := NewGeminiLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "google", APIKey: "test-key",
	}, core.ModelGoogleGeminiFlash)
	require.NoError(t, err)

	assert.Equal(t, 10*time.Minute, model.GetHTTPClient().Timeout)
	require.NotNil(t, model.GetEndpointConfig())
	assert.Equal(t, 600, model.GetEndpointConfig().TimeoutSec)
}

func TestOfficialOpenAIEndpointWithTrailingSlashesUsesResponses(t *testing.T) {
	model, err := OpenAIProviderFactory(context.Background(), core.ProviderConfig{
		Name: "openai", APIKey: "test-key", BaseURL: "https://api.openai.com/v1/",
		Endpoint: &core.EndpointConfig{Path: "/v1/responses/"},
	}, core.ModelOpenAIGPT4o)
	require.NoError(t, err)

	adapted, ok := model.(*GeneratorLLM)
	require.True(t, ok)
	assert.Contains(t, adapted.Capabilities(), core.CapabilityEmbedding)
	compatibility := adapted.Generator().Info().Compatibility
	require.NotNil(t, compatibility)
	assert.NotNil(t, compatibility.OpenAIResponses)
}

func TestOpenAICodexConfigDerivesAccountIDFromIDToken(t *testing.T) {
	var accountID string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		accountID = request.Header.Get("ChatGPT-Account-ID")
		http.Error(w, "stop after capturing headers", http.StatusBadRequest)
	}))
	t.Cleanup(server.Close)

	model, err := NewOpenAICodexLLMFromConfig(context.Background(), core.ProviderConfig{
		Name:   "openai-codex",
		APIKey: "opaque-access-token",
		Params: map[string]any{"id_token": codexTestToken("account-from-id-token")},
		Endpoint: &core.EndpointConfig{
			BaseURL: server.URL,
		},
	}, "openai-codex:gpt-5.4")
	require.NoError(t, err)

	_, err = model.Generate(context.Background(), "hello")
	require.Error(t, err)
	assert.Equal(t, "account-from-id-token", accountID)
}

func TestOllamaNativeProtocolIsExplicitlyUnsupported(t *testing.T) {
	model, err := NewOllamaLLM("llama3.2", WithNativeAPI())
	require.Error(t, err)
	assert.Nil(t, model)
}

func codexTestToken(accountID string) string {
	payload := `{"https://api.openai.com/auth":{"chatgpt_account_id":"` + accountID + `"}}`
	return "e30." + base64.RawURLEncoding.EncodeToString([]byte(payload)) + ".signature"
}
