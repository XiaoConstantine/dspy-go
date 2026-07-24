package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAICodexLive(t *testing.T) {
	accessToken := os.Getenv("DSPY_GO_CODEX_LIVE_ACCESS_TOKEN")
	accountID := os.Getenv("DSPY_GO_CODEX_LIVE_ACCOUNT_ID")
	if accessToken == "" || accountID == "" {
		t.Skip("set DSPY_GO_CODEX_LIVE_ACCESS_TOKEN and DSPY_GO_CODEX_LIVE_ACCOUNT_ID")
	}
	model := os.Getenv("DSPY_GO_CODEX_LIVE_MODEL")
	if model == "" {
		model = "gpt-5.4"
	}
	llm, err := NewOpenAICodexLLM(core.ModelID(model), WithOpenAICodexCredentials(
		func(context.Context, string) (OpenAICodexCredentials, error) {
			return OpenAICodexCredentials{AccessToken: accessToken, AccountID: accountID}, nil
		},
	))
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	tools := []map[string]any{{
		"name": "echo", "description": "Return the supplied value",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}, "required": []string{"value"}},
	}}
	first, err := llm.GenerateWithTools(ctx, []core.ChatMessage{
		{Role: "system", Content: []core.ContentBlock{core.NewTextBlock("Call echo exactly once with value pong. After its result, answer with that result without calling another tool.")}},
		{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("Begin")}},
	}, tools)
	require.NoError(t, err)
	calls, ok := first["tool_calls"].([]core.ToolCall)
	require.True(t, ok)
	require.Len(t, calls, 1)
	require.Equal(t, "echo", calls[0].Name)

	second, err := llm.GenerateWithTools(ctx, []core.ChatMessage{
		{Role: "system", Content: []core.ContentBlock{core.NewTextBlock("Call echo exactly once with value pong. After its result, answer with that result without calling another tool.")}},
		{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("Begin")}},
		{Role: "assistant", ToolCalls: calls, ProviderData: first["provider_data"].(map[string]any)},
		{Role: "tool", ToolResult: &core.ChatToolResult{ToolCallID: calls[0].ID, Name: "echo", Content: []core.ContentBlock{core.NewTextBlock("pong")}}},
	}, tools)
	require.NoError(t, err)
	assert.Contains(t, strings.ToLower(second["content"].(string)), "pong")
}

func TestOpenAICodexGenerateWithToolsUsesResponsesProtocol(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		assert.Equal(t, "/codex/responses", r.URL.Path)
		assert.Equal(t, "Bearer access-token", r.Header.Get("Authorization"))
		assert.Equal(t, "account-123", r.Header.Get("ChatGPT-Account-ID"))
		assert.Equal(t, "test-suite", r.Header.Get("Originator"))
		assert.Equal(t, "text/event-stream", r.Header.Get("Accept"))

		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		assert.Equal(t, "gpt-5.2-codex", payload["model"])
		assert.Equal(t, true, payload["stream"])
		assert.Equal(t, false, payload["store"])
		input := payload["input"].([]any)
		if requests.Load() == 1 {
			assert.Equal(t, "coding instructions", payload["instructions"])
			require.Len(t, input, 3)
			assert.Equal(t, "function_call", input[0].(map[string]any)["type"])
			assert.Equal(t, "function_call_output", input[1].(map[string]any)["type"])
			tools := payload["tools"].([]any)
			require.Len(t, tools, 1)
			assert.Equal(t, "read", tools[0].(map[string]any)["name"])
			assert.Equal(t, false, tools[0].(map[string]any)["strict"])
		} else {
			require.GreaterOrEqual(t, len(input), 2)
			assert.Equal(t, "reasoning", input[0].(map[string]any)["type"])
			assert.Equal(t, "encrypted-value", input[0].(map[string]any)["encrypted_content"])
			assert.Equal(t, "function_call", input[1].(map[string]any)["type"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"encrypted-value\",\"summary\":[]}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"path\\\":\\\"README.md\\\"}\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"total_tokens\":14}}}\n\n")
	}))
	defer server.Close()

	var resolves atomic.Int32
	llm, err := NewOpenAICodexLLM("gpt-5.2-codex",
		WithOpenAICodexBaseURL(server.URL),
		WithOpenAICodexOriginator("test-suite"),
		WithOpenAICodexHTTPClient(server.Client()),
		WithOpenAICodexCredentials(func(context.Context, string) (OpenAICodexCredentials, error) {
			resolves.Add(1)
			return OpenAICodexCredentials{AccessToken: "access-token", AccountID: "account-123"}, nil
		}),
	)
	require.NoError(t, err)

	messages := []core.ChatMessage{
		{Role: "system", Content: []core.ContentBlock{core.NewTextBlock("coding instructions")}},
		{Role: "assistant", ToolCalls: []core.ToolCall{{ID: "previous-call", Name: "ls", Arguments: map[string]any{}}}},
		{Role: "tool", ToolResult: &core.ChatToolResult{ToolCallID: "previous-call", Name: "ls", Content: []core.ContentBlock{core.NewTextBlock("README.md")}}},
		{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("Read it")}},
	}
	result, err := llm.GenerateWithTools(context.Background(), messages, []map[string]any{{
		"name": "read", "description": "read a file", "parameters": map[string]any{"type": "object"},
	}})
	require.NoError(t, err)
	calls := result["tool_calls"].([]core.ToolCall)
	require.Len(t, calls, 1)
	assert.Equal(t, "call_1", calls[0].ID)
	assert.Equal(t, "README.md", calls[0].Arguments["path"])
	assert.Equal(t, &core.TokenInfo{PromptTokens: 10, CompletionTokens: 4, TotalTokens: 14}, result["_usage"])
	assert.Equal(t, int32(1), requests.Load())
	assert.Equal(t, int32(1), resolves.Load())

	providerData := result["provider_data"].(map[string]any)
	_, err = llm.GenerateWithTools(context.Background(), []core.ChatMessage{
		{Role: "assistant", ToolCalls: calls, ProviderData: providerData},
		{Role: "tool", ToolResult: &core.ChatToolResult{ToolCallID: "call_1", Name: "read", Content: []core.ContentBlock{core.NewTextBlock("contents")}}},
		{Role: "user", Content: []core.ContentBlock{core.NewTextBlock("continue")}},
	}, []map[string]any{{"name": "read", "parameters": map[string]any{"type": "object"}}})
	require.NoError(t, err)
	assert.Equal(t, int32(2), resolves.Load(), "credentials must resolve for every request")
}

func TestOpenAICodexRetriesUnauthorizedWithForcedRefresh(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request := requests.Add(1)
		if request == 1 {
			assert.Equal(t, "Bearer stale", r.Header.Get("Authorization"))
			http.Error(w, "expired", http.StatusUnauthorized)
			return
		}
		assert.Equal(t, "Bearer fresh", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()

	var resolutions []string
	llm, err := NewOpenAICodexLLM("gpt-5.2-codex",
		WithOpenAICodexBaseURL(server.URL), WithOpenAICodexHTTPClient(server.Client()),
		WithOpenAICodexCredentials(func(_ context.Context, rejectedAccessToken string) (OpenAICodexCredentials, error) {
			resolutions = append(resolutions, rejectedAccessToken)
			token := "stale"
			if rejectedAccessToken != "" {
				token = "fresh"
			}
			return OpenAICodexCredentials{AccessToken: token, AccountID: "account"}, nil
		}),
	)
	require.NoError(t, err)
	response, err := llm.Generate(context.Background(), "hello")
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Content)
	assert.Equal(t, []string{"", "stale"}, resolutions)
	assert.Equal(t, int32(2), requests.Load())
}

func TestOpenAICodexRejectsMalformedContinuation(t *testing.T) {
	llm, err := NewOpenAICodexLLM("gpt-5.4", WithOpenAICodexCredentials(
		func(context.Context, string) (OpenAICodexCredentials, error) {
			return OpenAICodexCredentials{AccessToken: "token", AccountID: "account"}, nil
		},
	))
	require.NoError(t, err)
	cycle := map[string]any{"type": "reasoning"}
	cycle["self"] = cycle
	for _, providerData := range []map[string]any{
		{"openai-codex": []any{"not-an-object"}},
		{"openai-codex": []any{}},
		{"openai-codex": []any{cycle}},
	} {
		_, err = llm.requestPayload([]core.ChatMessage{{Role: "assistant", ProviderData: providerData}}, nil)
		require.Error(t, err)
	}
}

func TestParseOpenAICodexSSERequiresCompletedTerminal(t *testing.T) {
	tests := []struct {
		name string
		sse  string
	}{
		{name: "clean eof", sse: `data: {"type":"response.output_text.delta","delta":"partial"}\n\n`},
		{name: "done marker only", sse: "data: [DONE]\n\n"},
		{name: "incomplete", sse: `data: {"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}\n\n`},
		{name: "unfinished call", sse: `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read"}}\n\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n`},
		{name: "missing call id", sse: `data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","name":"read","arguments":"{}"}}\n\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n`},
		{name: "terminal failed status", sse: `data: {"type":"response.completed","response":{"status":"failed"}}\n\n`},
		{name: "terminal missing response", sse: `data: {"type":"response.completed"}\n\n`},
		{name: "terminal missing status", sse: `data: {"type":"response.completed","response":{}}\n\n`},
		{name: "duplicate index conflicting call id", sse: `data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"first","arguments":"{}"}}\n\ndata: {"type":"response.output_item.completed","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_2","name":"second","arguments":"{}"}}\n\n`},
		{name: "duplicate index malformed type", sse: `data: {"type":"response.output_item.done","output_index":0,"item":{"type":["message"],"id":"msg_1"}}\n\ndata: {"type":"response.output_item.completed","output_index":0,"item":{"type":["message"],"id":"msg_1"}}\n\n`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := parseOpenAICodexSSE(strings.NewReader(strings.ReplaceAll(test.sse, `\n`, "\n")))
			require.Error(t, err)
			assert.Nil(t, result)
		})
	}
}

func TestParseOpenAICodexSSEOrdersToolCallsByOutputIndex(t *testing.T) {
	sse := `data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"second","arguments":"{}"}}\n\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"first","arguments":"{}"}}\n\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n`
	result, err := parseOpenAICodexSSE(strings.NewReader(strings.ReplaceAll(sse, `\n`, "\n")))
	require.NoError(t, err)
	require.Len(t, result.toolCalls, 2)
	assert.Equal(t, "call_1", result.toolCalls[0].ID)
	assert.Equal(t, "call_2", result.toolCalls[1].ID)
}

func TestOpenAICodexAccountID(t *testing.T) {
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-42"}}`))
	got, err := OpenAICodexAccountID("header." + payload + ".signature")
	require.NoError(t, err)
	assert.Equal(t, "acct-42", got)
}

func TestOpenAICodexFactoryUsesAccountIDFromToken(t *testing.T) {
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-42"}}`))
	llm, err := NewOpenAICodexLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "openai-codex", APIKey: "opaque-access-token",
		Params: map[string]any{"id_token": "header." + payload + ".signature"},
	}, "gpt-5.2-codex")
	require.NoError(t, err)
	assert.Equal(t, "openai-codex", llm.ProviderName())
}
