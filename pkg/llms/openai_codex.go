package llms

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

const defaultOpenAICodexBaseURL = "https://chatgpt.com/backend-api"

// OpenAICodexCredentials are resolved before every request so callers can
// refresh short-lived ChatGPT subscription credentials without rebuilding the LLM.
type OpenAICodexCredentials struct {
	AccessToken string
	AccountID   string
}

// OpenAICodexCredentialResolver returns current ChatGPT subscription credentials.
// rejectedAccessToken is non-empty after an authentication failure. The owner
// should refresh only when storage still contains that token; another concurrent
// request may already have rotated it.
type OpenAICodexCredentialResolver func(ctx context.Context, rejectedAccessToken string) (OpenAICodexCredentials, error)

type OpenAICodexOption func(*openAICodexConfig)

type openAICodexConfig struct {
	baseURL            string
	timeout            time.Duration
	headers            map[string]string
	httpClient         *http.Client
	credentialResolver OpenAICodexCredentialResolver
	originator         string
	reasoningEffort    string
}

// OpenAICodexLLM implements the ChatGPT subscription Codex Responses protocol.
type OpenAICodexLLM struct {
	*core.BaseLLM
	resolver        OpenAICodexCredentialResolver
	headers         map[string]string
	originator      string
	reasoningEffort string
	httpClient      *http.Client
}

func WithOpenAICodexCredentials(resolver OpenAICodexCredentialResolver) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.credentialResolver = resolver }
}

func WithOpenAICodexBaseURL(baseURL string) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.baseURL = strings.TrimRight(baseURL, "/") }
}

func WithOpenAICodexHTTPClient(client *http.Client) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.httpClient = client }
}

func WithOpenAICodexTimeout(timeout time.Duration) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.timeout = timeout }
}

func WithOpenAICodexHeader(name, value string) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.headers[name] = value }
}

func WithOpenAICodexOriginator(originator string) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.originator = originator }
}

func WithOpenAICodexReasoningEffort(effort string) OpenAICodexOption {
	return func(c *openAICodexConfig) { c.reasoningEffort = effort }
}

func NewOpenAICodexLLM(modelID core.ModelID, opts ...OpenAICodexOption) (*OpenAICodexLLM, error) {
	config := openAICodexConfig{
		baseURL:    defaultOpenAICodexBaseURL,
		timeout:    120 * time.Second,
		headers:    map[string]string{},
		originator: "dspy-go",
	}
	for _, opt := range opts {
		opt(&config)
	}
	if config.credentialResolver == nil {
		return nil, errors.New(errors.InvalidInput, "OpenAI Codex credential resolver is required")
	}
	endpoint := &core.EndpointConfig{
		BaseURL:    config.baseURL,
		Path:       "/codex/responses",
		TimeoutSec: int(config.timeout.Seconds()),
	}
	base := core.NewBaseLLM("openai-codex", modelID, []core.Capability{
		core.CapabilityCompletion, core.CapabilityChat, core.CapabilityJSON, core.CapabilityToolCalling,
	}, endpoint)
	client := config.httpClient
	if client == nil {
		client = base.GetHTTPClient()
	}
	return &OpenAICodexLLM{
		BaseLLM:         base,
		resolver:        config.credentialResolver,
		headers:         cloneStringMap(config.headers),
		originator:      config.originator,
		reasoningEffort: config.reasoningEffort,
		httpClient:      client,
	}, nil
}

func NewOpenAICodexLLMFromConfig(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (*OpenAICodexLLM, error) {
	token := strings.TrimSpace(config.APIKey)
	if token == "" {
		token = strings.TrimSpace(os.Getenv("OPENAI_OAUTH_TOKEN"))
	}
	if token == "" {
		return nil, errors.New(errors.InvalidInput, "OpenAI Codex OAuth access token is required")
	}
	accountID := ""
	if config.Endpoint != nil {
		accountID = headerValue(config.Endpoint.Headers, "ChatGPT-Account-ID")
	}
	if accountID == "" {
		accountID, _ = config.Params["account_id"].(string)
		accountID = strings.TrimSpace(accountID)
	}
	if accountID == "" {
		idToken, _ := config.Params["id_token"].(string)
		if idToken == "" {
			idToken = strings.TrimSpace(os.Getenv("OPENAI_ID_TOKEN"))
		}
		if idToken != "" {
			accountID, _ = OpenAICodexAccountIDFromToken(idToken)
		}
	}
	if accountID == "" {
		var err error
		accountID, err = OpenAICodexAccountID(token)
		if err != nil {
			return nil, errors.Wrap(err, errors.InvalidInput, "resolve OpenAI Codex account id from account_id, id_token, or access token")
		}
	}
	credentials := OpenAICodexCredentials{AccessToken: token, AccountID: accountID}
	opts := []OpenAICodexOption{WithOpenAICodexCredentials(func(context.Context, string) (OpenAICodexCredentials, error) {
		return credentials, nil
	})}
	baseURL := config.BaseURL
	if config.Endpoint != nil && config.Endpoint.BaseURL != "" {
		baseURL = config.Endpoint.BaseURL
	}
	if baseURL != "" {
		opts = append(opts, WithOpenAICodexBaseURL(baseURL))
	}
	if config.Endpoint != nil {
		if config.Endpoint.TimeoutSec > 0 {
			opts = append(opts, WithOpenAICodexTimeout(time.Duration(config.Endpoint.TimeoutSec)*time.Second))
		}
		for name, value := range config.Endpoint.Headers {
			if !strings.EqualFold(name, "Authorization") && !strings.EqualFold(name, "ChatGPT-Account-ID") {
				opts = append(opts, WithOpenAICodexHeader(name, value))
			}
		}
	}
	return NewOpenAICodexLLM(modelID, opts...)
}

func OpenAICodexProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOpenAICodexLLMFromConfig(ctx, config, modelID)
}

// OpenAICodexAccountID extracts the ChatGPT account identifier from an OAuth access JWT.
// The token signature is verified by the server when the token is used; this only reads claims.
func OpenAICodexAccountID(accessToken string) (string, error) {
	return OpenAICodexAccountIDFromToken(accessToken)
}

// OpenAICodexAccountIDFromToken extracts the ChatGPT account identifier from
// an OpenAI ID or access JWT without verifying its signature.
func OpenAICodexAccountIDFromToken(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", errors.New(errors.InvalidInput, "OpenAI Codex access token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", errors.Wrap(err, errors.InvalidInput, "decode OpenAI Codex access token")
	}
	var claims struct {
		Auth struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", errors.Wrap(err, errors.InvalidInput, "decode OpenAI Codex access token claims")
	}
	if strings.TrimSpace(claims.Auth.AccountID) == "" {
		return "", errors.New(errors.InvalidInput, "OpenAI Codex access token is missing chatgpt_account_id")
	}
	return strings.TrimSpace(claims.Auth.AccountID), nil
}

func (o *OpenAICodexLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	result, err := o.generate(ctx, []core.ChatMessage{{Role: "user", Content: []core.ContentBlock{core.NewTextBlock(prompt)}}}, nil, options...)
	if err != nil {
		return nil, err
	}
	if len(result.providerData) > 0 {
		result.metadata["provider_data"] = result.providerData
	}
	return &core.LLMResponse{Content: result.content, Usage: result.usage, Metadata: result.metadata}, nil
}

func (o *OpenAICodexLLM) GenerateWithTools(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	if len(tools) == 0 {
		return nil, errors.New(errors.InvalidInput, "at least one tool schema is required")
	}
	result, err := o.generate(ctx, messages, tools, options...)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"_usage": result.usage}
	if len(result.providerData) > 0 {
		out["provider_data"] = result.providerData
	}
	if result.content != "" {
		out["content"] = result.content
	}
	if len(result.toolCalls) > 0 {
		out["tool_calls"] = result.toolCalls
		out["function_call"] = map[string]any{"name": result.toolCalls[0].Name, "arguments": result.toolCalls[0].Arguments}
	}
	if result.content == "" && len(result.toolCalls) == 0 {
		out["content"] = "No content or function call received from model"
	}
	return out, nil
}

func (o *OpenAICodexLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	return o.GenerateWithTools(ctx, []core.ChatMessage{{Role: "user", Content: []core.ContentBlock{core.NewTextBlock(prompt)}}}, functions, options...)
}

func (o *OpenAICodexLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	response, err := o.Generate(ctx, prompt+"\n\nReturn only a valid JSON object.", options...)
	if err != nil {
		return nil, err
	}
	return utils.ParseJSONResponse(response.Content)
}

func (o *OpenAICodexLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.LLMResponse, error) {
	return o.Generate(ctx, flattenCoreChatMessageContent(content), options...)
}

func (o *OpenAICodexLLM) StreamGenerate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.StreamResponse, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	chunks := make(chan core.StreamChunk, 1)
	go func() {
		defer close(chunks)
		response, err := o.Generate(streamCtx, prompt, options...)
		if err != nil {
			chunks <- core.StreamChunk{Done: true, Error: err}
			return
		}
		chunks <- core.StreamChunk{Content: response.Content, Done: true, Usage: response.Usage}
	}()
	return &core.StreamResponse{ChunkChannel: chunks, Cancel: cancel}, nil
}

func (o *OpenAICodexLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, options ...core.GenerateOption) (*core.StreamResponse, error) {
	return o.StreamGenerate(ctx, flattenCoreChatMessageContent(content), options...)
}

func (o *OpenAICodexLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, errors.New(errors.UnsupportedOperation, "OpenAI Codex subscriptions do not provide embeddings")
}

func (o *OpenAICodexLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, errors.New(errors.UnsupportedOperation, "OpenAI Codex subscriptions do not provide embeddings")
}

type openAICodexResult struct {
	content      string
	toolCalls    []core.ToolCall
	usage        *core.TokenInfo
	metadata     map[string]any
	providerData map[string]any
}

type codexToolBuilder struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

func (o *OpenAICodexLLM) generate(ctx context.Context, messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (*openAICodexResult, error) {
	payload, err := o.requestPayload(messages, tools, options...)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "encode OpenAI Codex request")
	}

	rejectedAccessToken := ""
	for attempt := 0; attempt < 2; attempt++ {
		credentials, err := o.resolver(ctx, rejectedAccessToken)
		if err != nil {
			return nil, errors.Wrap(err, errors.Unknown, "resolve OpenAI Codex credentials")
		}
		if strings.TrimSpace(credentials.AccessToken) == "" || strings.TrimSpace(credentials.AccountID) == "" {
			return nil, errors.New(errors.InvalidInput, "OpenAI Codex credentials require access token and account id")
		}
		resp, err := o.sendRequest(ctx, body, credentials)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusUnauthorized && attempt == 0 {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
			resp.Body.Close()
			rejectedAccessToken = credentials.AccessToken
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			limited, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
			return nil, errors.WithFields(errors.New(errors.Unknown, fmt.Sprintf("OpenAI Codex request failed: %s", resp.Status)), errors.Fields{"status_code": resp.StatusCode, "body": strings.TrimSpace(string(limited))})
		}
		return parseOpenAICodexSSE(resp.Body)
	}
	return nil, errors.New(errors.Unknown, "OpenAI Codex authentication failed")
}

func (o *OpenAICodexLLM) sendRequest(ctx context.Context, body []byte, credentials OpenAICodexCredentials) (*http.Response, error) {
	endpoint := o.GetEndpointConfig()
	url := resolveOpenAICodexResponsesURL(endpoint.BaseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "create OpenAI Codex request")
	}
	for name, value := range o.headers {
		req.Header.Set(name, value)
	}
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(credentials.AccessToken))
	req.Header.Set("ChatGPT-Account-ID", strings.TrimSpace(credentials.AccountID))
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	req.Header.Set("Originator", o.originator)
	req.Header.Set("User-Agent", "dspy-go")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "send OpenAI Codex request")
	}
	return resp, nil
}

func resolveOpenAICodexResponsesURL(baseURL string) string {
	normalized := strings.TrimRight(baseURL, "/")
	switch {
	case strings.HasSuffix(normalized, "/codex/responses"):
		return normalized
	case strings.HasSuffix(normalized, "/codex"):
		return normalized + "/responses"
	default:
		return normalized + "/codex/responses"
	}
}

func (o *OpenAICodexLLM) requestPayload(messages []core.ChatMessage, tools []map[string]any, options ...core.GenerateOption) (map[string]any, error) {
	// The subscription Codex endpoint currently rejects public Responses
	// sampling and max-output parameters. Keep options in the interface for
	// compatibility, but do not forward unsupported fields.
	_ = options
	instructions := make([]string, 0, 1)
	input := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		text := flattenCoreChatMessageContent(message.Content)
		switch message.Role {
		case "system", "developer":
			if text != "" {
				instructions = append(instructions, text)
			}
		case "user":
			input = append(input, map[string]any{"role": "user", "content": []map[string]any{{"type": "input_text", "text": text}}})
		case "assistant":
			var replayed bool
			var err error
			input, replayed, err = appendOpenAICodexProviderData(input, message.ProviderData)
			if err != nil {
				return nil, err
			}
			if replayed {
				continue
			}
			if text != "" {
				input = append(input, map[string]any{"type": "message", "role": "assistant", "status": "completed", "content": []map[string]any{{"type": "output_text", "text": text, "annotations": []any{}}}})
			}
			for _, call := range message.ToolCalls {
				arguments, err := json.Marshal(call.Arguments)
				if err != nil {
					return nil, err
				}
				input = append(input, map[string]any{"type": "function_call", "call_id": call.ID, "name": call.Name, "arguments": string(arguments)})
			}
		case "tool":
			if message.ToolResult != nil {
				input = append(input, map[string]any{"type": "function_call_output", "call_id": message.ToolResult.ToolCallID, "output": flattenCoreChatMessageContent(message.ToolResult.Content)})
			}
		}
	}
	if len(instructions) == 0 {
		instructions = append(instructions, "You are a helpful assistant.")
	}
	payload := map[string]any{
		"model": o.ModelID(), "store": false, "stream": true,
		"instructions": strings.Join(instructions, "\n\n"), "input": input,
		"text": map[string]any{"verbosity": "low"}, "include": []string{"reasoning.encrypted_content"},
		"tool_choice": "auto", "parallel_tool_calls": true,
	}
	if o.reasoningEffort != "" {
		payload["reasoning"] = map[string]any{"effort": o.reasoningEffort, "summary": "auto"}
	}
	if len(tools) > 0 {
		converted := make([]map[string]any, 0, len(tools))
		for _, tool := range tools {
			name, _ := tool["name"].(string)
			if strings.TrimSpace(name) == "" {
				return nil, errors.New(errors.InvalidInput, "tool schema missing non-empty name")
			}
			parameters := tool["parameters"]
			if parameters == nil {
				parameters = map[string]any{"type": "object"}
			}
			strict, _ := tool["strict"].(bool)
			converted = append(converted, map[string]any{"type": "function", "name": name, "description": tool["description"], "parameters": parameters, "strict": strict})
		}
		payload["tools"] = converted
	}
	return payload, nil
}

func appendOpenAICodexProviderData(input []map[string]any, providerData map[string]any) ([]map[string]any, bool, error) {
	raw, ok := providerData["openai-codex"]
	if !ok {
		return input, false, nil
	}
	start := len(input)
	switch items := raw.(type) {
	case []map[string]any:
		for _, item := range items {
			cloned, err := cloneOpenAICodexContinuationItem(item)
			if err != nil {
				return nil, false, err
			}
			input = append(input, cloned)
		}
	case []any:
		for _, rawItem := range items {
			item, ok := rawItem.(map[string]any)
			if !ok {
				return nil, false, errors.New(errors.InvalidInput, "OpenAI Codex provider continuation item must be an object")
			}
			cloned, err := cloneOpenAICodexContinuationItem(item)
			if err != nil {
				return nil, false, err
			}
			input = append(input, cloned)
		}
	default:
		return nil, false, errors.New(errors.InvalidInput, "OpenAI Codex provider continuation must be an array")
	}
	if len(input) == start {
		return nil, false, errors.New(errors.InvalidInput, "OpenAI Codex provider continuation must not be empty")
	}
	return input, true, nil
}

func cloneOpenAICodexContinuationItem(item map[string]any) (map[string]any, error) {
	if item == nil {
		return nil, errors.New(errors.InvalidInput, "OpenAI Codex provider continuation item must not be null")
	}
	encoded, err := json.Marshal(item)
	if err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "encode OpenAI Codex provider continuation item")
	}
	var cloned map[string]any
	if err := json.Unmarshal(encoded, &cloned); err != nil {
		return nil, errors.Wrap(err, errors.InvalidInput, "decode OpenAI Codex provider continuation item")
	}
	return cloned, nil
}

func parseOpenAICodexSSE(reader io.Reader) (*openAICodexResult, error) {
	result := &openAICodexResult{usage: &core.TokenInfo{}, metadata: map[string]any{}}
	builders := map[string]*codexToolBuilder{}
	outputItems := map[int]map[string]any{}
	terminal := false
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), 32<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return nil, errors.Wrap(err, errors.InvalidResponse, "decode OpenAI Codex SSE event")
		}
		typeName, _ := event["type"].(string)
		switch typeName {
		case "error", "response.failed":
			return nil, errors.WithFields(errors.New(errors.InvalidResponse, "OpenAI Codex response failed"), errors.Fields{"event": event})
		case "response.output_text.delta":
			if delta, ok := event["delta"].(string); ok {
				result.content += delta
			}
		case "response.output_item.added", "response.output_item.done", "response.output_item.completed":
			item, _ := event["item"].(map[string]any)
			if item == nil {
				return nil, errors.New(errors.InvalidResponse, "OpenAI Codex output item event is missing item")
			}
			finished := typeName != "response.output_item.added"
			if item["type"] == "function_call" {
				key := codexToolKey(item, event)
				builder := builders[key]
				if builder == nil {
					builder = &codexToolBuilder{}
					builders[key] = builder
				}
				updateCodexToolBuilder(builder, item)
				if finished {
					item = cloneCodexMap(item)
					item["call_id"] = builder.ID
					item["name"] = builder.Name
					item["arguments"] = builder.Arguments.String()
					delete(builders, key)
				}
			} else if item["type"] == "message" && result.content == "" && finished {
				result.content = codexMessageText(item)
			}
			if finished {
				index, ok := codexOutputIndex(event)
				if !ok {
					return nil, errors.New(errors.InvalidResponse, "OpenAI Codex completed output item is missing output_index")
				}
				if existing, exists := outputItems[index]; exists && !sameCodexOutputIdentity(existing, item) {
					return nil, errors.WithFields(errors.New(errors.InvalidResponse, "OpenAI Codex reused output_index for different items"), errors.Fields{"output_index": index})
				}
				outputItems[index] = cloneCodexMap(item)
			}
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			key := codexToolKey(event, event)
			builder := builders[key]
			if builder == nil {
				builder = &codexToolBuilder{}
				builders[key] = builder
			}
			if value, ok := event["delta"].(string); ok {
				builder.Arguments.WriteString(value)
			}
			if value, ok := event["arguments"].(string); ok {
				builder.Arguments.Reset()
				builder.Arguments.WriteString(value)
			}
		case "response.incomplete":
			return nil, errors.WithFields(errors.New(errors.InvalidResponse, "OpenAI Codex response was incomplete"), errors.Fields{"event": event})
		case "response.completed", "response.done":
			response, ok := event["response"].(map[string]any)
			if !ok {
				return nil, errors.New(errors.InvalidResponse, "OpenAI Codex terminal event is missing response")
			}
			status, _ := response["status"].(string)
			if status != "completed" {
				return nil, errors.WithFields(errors.New(errors.InvalidResponse, "OpenAI Codex terminal response was not completed"), errors.Fields{"status": status})
			}
			result.metadata["id"] = response["id"]
			result.metadata["status"] = status
			setCodexUsage(result.usage, response["usage"])
			terminal = true
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, errors.Wrap(err, errors.Unknown, "read OpenAI Codex SSE response")
	}
	if !terminal {
		return nil, errors.New(errors.InvalidResponse, "OpenAI Codex SSE ended before response.completed")
	}
	if len(builders) > 0 {
		return nil, errors.New(errors.InvalidResponse, "OpenAI Codex SSE ended with unfinished tool calls")
	}
	if len(outputItems) > 0 {
		indices := make([]int, 0, len(outputItems))
		for index := range outputItems {
			indices = append(indices, index)
		}
		sort.Ints(indices)
		items := make([]map[string]any, 0, len(indices))
		seenCallIDs := make(map[string]struct{})
		for _, index := range indices {
			item := outputItems[index]
			items = append(items, item)
			if item["type"] == "function_call" {
				builder := &codexToolBuilder{}
				updateCodexToolBuilder(builder, item)
				if _, exists := seenCallIDs[builder.ID]; exists {
					return nil, errors.WithFields(errors.New(errors.InvalidResponse, "OpenAI Codex reused function call id"), errors.Fields{"call_id": builder.ID})
				}
				if err := appendCodexToolCall(result, builder); err != nil {
					return nil, err
				}
				seenCallIDs[builder.ID] = struct{}{}
			}
		}
		result.providerData = map[string]any{"openai-codex": items}
	}
	return result, nil
}

func sameCodexOutputIdentity(left, right map[string]any) bool {
	leftType, leftTypeOK := left["type"].(string)
	rightType, rightTypeOK := right["type"].(string)
	if !leftTypeOK || !rightTypeOK || leftType == "" || leftType != rightType {
		return false
	}
	for _, field := range []string{"id", "call_id"} {
		leftRaw, leftExists := left[field]
		rightRaw, rightExists := right[field]
		if !leftExists && !rightExists {
			continue
		}
		leftValue, leftOK := leftRaw.(string)
		rightValue, rightOK := rightRaw.(string)
		if !leftOK || !rightOK || leftValue == "" || leftValue != rightValue {
			return false
		}
	}
	return true
}

func codexOutputIndex(event map[string]any) (int, bool) {
	switch value := event["output_index"].(type) {
	case float64:
		return int(value), value >= 0 && value == float64(int(value))
	case int:
		return value, value >= 0
	default:
		return 0, false
	}
}

func codexToolKey(item, event map[string]any) string {
	for _, source := range []map[string]any{item, event} {
		for _, field := range []string{"id", "item_id", "call_id"} {
			if value, ok := source[field].(string); ok && value != "" {
				return value
			}
		}
	}
	return fmt.Sprintf("output_%v", event["output_index"])
}
func updateCodexToolBuilder(builder *codexToolBuilder, item map[string]any) {
	if value, ok := item["call_id"].(string); ok {
		builder.ID = value
	}
	if value, ok := item["name"].(string); ok {
		builder.Name = value
	}
	if value, ok := item["arguments"].(string); ok {
		builder.Arguments.Reset()
		builder.Arguments.WriteString(value)
	}
}
func appendCodexToolCall(result *openAICodexResult, builder *codexToolBuilder) error {
	if strings.TrimSpace(builder.ID) == "" || strings.TrimSpace(builder.Name) == "" {
		return errors.New(errors.InvalidResponse, "OpenAI Codex function call is missing call_id or name")
	}
	var arguments map[string]any
	raw := builder.Arguments.String()
	if raw == "" {
		arguments = map[string]any{}
	} else if err := json.Unmarshal([]byte(raw), &arguments); err != nil {
		return errors.Wrap(err, errors.InvalidResponse, "decode OpenAI Codex tool arguments")
	}
	result.toolCalls = append(result.toolCalls, core.ToolCall{ID: builder.ID, Name: builder.Name, Arguments: arguments})
	return nil
}
func codexMessageText(item map[string]any) string {
	content, _ := item["content"].([]any)
	var parts []string
	for _, raw := range content {
		block, _ := raw.(map[string]any)
		if text, ok := block["text"].(string); ok {
			parts = append(parts, text)
		} else if refusal, ok := block["refusal"].(string); ok {
			parts = append(parts, refusal)
		}
	}
	return strings.Join(parts, "")
}
func setCodexUsage(target *core.TokenInfo, raw any) {
	usage, _ := raw.(map[string]any)
	target.PromptTokens = jsonInt(usage["input_tokens"])
	target.CompletionTokens = jsonInt(usage["output_tokens"])
	target.TotalTokens = jsonInt(usage["total_tokens"])
	if target.TotalTokens == 0 {
		target.TotalTokens = target.PromptTokens + target.CompletionTokens
	}
}
func jsonInt(value any) int { number, _ := value.(float64); return int(number) }
func headerValue(headers map[string]string, name string) string {
	for key, value := range headers {
		if strings.EqualFold(key, name) {
			return value
		}
	}
	return ""
}
func cloneCodexMap(source map[string]any) map[string]any {
	target := make(map[string]any, len(source))
	for key, value := range source {
		switch typed := value.(type) {
		case map[string]any:
			target[key] = cloneCodexMap(typed)
		case []any:
			items := make([]any, len(typed))
			for i, item := range typed {
				if child, ok := item.(map[string]any); ok {
					items[i] = cloneCodexMap(child)
				} else {
					items[i] = item
				}
			}
			target[key] = items
		default:
			target[key] = value
		}
	}
	return target
}

func cloneStringMap(source map[string]string) map[string]string {
	target := make(map[string]string, len(source))
	for key, value := range source {
		target[key] = value
	}
	return target
}
