package llms_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAICodexRunsThroughSharedAgentLoop(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request := requests.Add(1)
		var payload map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		input := payload["input"].([]any)
		w.Header().Set("Content-Type", "text/event-stream")
		if request == 1 {
			fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"opaque\",\"summary\":[]}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"{\\\"value\\\":\\\"pong\\\"}\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
			return
		}
		require.GreaterOrEqual(t, len(input), 4)
		assert.Equal(t, "reasoning", input[1].(map[string]any)["type"])
		assert.Equal(t, "function_call", input[2].(map[string]any)["type"])
		assert.Equal(t, "function_call_output", input[3].(map[string]any)["type"])
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"pong\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"pong\",\"annotations\":[]}]}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()

	llm, err := llms.NewOpenAICodexLLM("gpt-5.4",
		llms.WithOpenAICodexBaseURL(server.URL),
		llms.WithOpenAICodexHTTPClient(server.Client()),
		llms.WithOpenAICodexCredentials(func(context.Context, string) (llms.OpenAICodexCredentials, error) {
			return llms.OpenAICodexCredentials{AccessToken: "token", AccountID: "account"}, nil
		}),
	)
	require.NoError(t, err)
	adapter, err := agents.NewLLMAdapter(llm)
	require.NoError(t, err)
	result, err := agents.RunLoop(context.Background(), adapter, []core.Tool{echoTool{}},
		[]agents.Message{agents.NewTextMessage(agents.RoleUser, "echo pong")}, agents.LoopConfig{
			RunID: "codex-loop", Task: "echo pong", MaxTurns: 3, Completion: agents.TextCompletion(),
		})
	require.NoError(t, err)
	assert.Equal(t, agents.StopReasonText, result.StopReason)
	assert.Equal(t, "pong", result.FinalAnswer)
	assert.Equal(t, 1, result.ToolCalls)
	assert.Equal(t, int32(2), requests.Load())
	require.NotEmpty(t, result.Messages[1].ProviderData)
}

type echoTool struct{}

func (echoTool) Name() string        { return "echo" }
func (echoTool) Description() string { return "return a value" }
func (t echoTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{Name: t.Name(), Description: t.Description(), InputSchema: t.InputSchema()}
}
func (echoTool) CanHandle(context.Context, string) bool { return true }
func (echoTool) Validate(map[string]any) error          { return nil }
func (echoTool) Execute(_ context.Context, arguments map[string]any) (core.ToolResult, error) {
	return core.ToolResult{Data: arguments["value"]}, nil
}
func (echoTool) InputSchema() models.InputSchema {
	return models.InputSchema{Type: "object", Properties: map[string]models.ParameterSchema{"value": {Type: "string"}}}
}
