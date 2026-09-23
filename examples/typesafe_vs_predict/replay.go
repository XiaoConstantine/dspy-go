package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
)

//go:embed testdata/category_responses.json
var comparisonFixture []byte

type recordedCategory struct {
	Ticket   string          `json:"ticket"`
	Response json.RawMessage `json:"response"`
}

type systemOneReplayFixture struct {
	request   typesafe.SystemOneRequest
	response  []byte
	requestID string
}

type recordedLLMResponse struct {
	match   string
	content string
	usage   core.TokenInfo
}

type llmCallStats struct {
	Calls      int
	UsageCalls int
	Prompt     int
	Completion int
}

type recordingLLM struct {
	core.LLM
	mu         sync.Mutex
	calls      int
	usageCalls int
	prompt     int
	completion int
}

func newComparisonClient(replay bool, model string) (decide.SystemOneClient, func(), error) {
	if !replay {
		options := make([]typesafe.ClientOption, 0, 1)
		if strings.TrimSpace(model) != "" {
			options = append(options, typesafe.WithDefaultModel(model))
		}
		client, err := typesafe.NewClient(options...)
		return client, func() {}, err
	}

	var recorded []recordedCategory
	if err := json.Unmarshal(comparisonFixture, &recorded); err != nil {
		return nil, func() {}, fmt.Errorf("decode comparison replay fixture: %w", err)
	}
	if len(recorded) != len(comparisonTickets) {
		return nil, func() {}, fmt.Errorf("comparison replay fixture has %d records, want %d", len(recorded), len(comparisonTickets))
	}
	fixtures := make([]systemOneReplayFixture, len(recorded))
	for index, item := range recorded {
		if item.Ticket != comparisonTickets[index].Text {
			return nil, func() {}, fmt.Errorf("comparison replay fixture %d does not match the labeled ticket", index)
		}
		fixtures[index] = systemOneReplayFixture{
			request:   expectedSystemOneReplayRequest(item.Ticket),
			response:  item.Response,
			requestID: fmt.Sprintf("req_comparison_replay_%d", index+1),
		}
	}
	return startSystemOneReplay(fixtures, model)
}

func expectedSystemOneReplayRequest(ticket string) typesafe.SystemOneRequest {
	return typesafe.SystemOneRequest{
		State: map[string]any{"ticket": ticket},
		Model: "jev-replay",
		Questions: map[string]typesafe.Question{
			"category": typesafe.ChoiceQuestion{
				Instructions: map[string]any{
					"question": "Exactly one of: billing, technical, account, product",
					"task":     "Classify into exactly one category. Use billing for charges, invoices, payments, or refunds; technical for failures, bugs, outages, or unexpected behavior; account for login, identity, access, permissions, or account settings; and product for product usage, features, exports, or how-to questions.",
					"inputs": []map[string]string{{
						"name": "ticket", "description": "Support ticket text",
					}},
				},
				Criteria: map[string]any{
					"billing":   "Charges, invoices, payments, or refunds",
					"technical": "Failures, bugs, outages, or unexpected behavior",
					"account":   "Login, identity, access, permissions, or account settings",
					"product":   "Product usage, features, exports, or how-to questions",
				},
			},
		},
	}
}

func startSystemOneReplay(fixtures []systemOneReplayFixture, model string) (decide.SystemOneClient, func(), error) {
	responses := make(map[string]systemOneReplayFixture, len(fixtures))
	for index, fixture := range fixtures {
		requestKey, err := canonicalReplayJSON(fixture.request)
		if err != nil {
			return nil, func() {}, fmt.Errorf("encode replay fixture %d request: %w", index, err)
		}
		if _, duplicate := responses[requestKey]; duplicate {
			return nil, func() {}, fmt.Errorf("replay fixture %d duplicates a request", index)
		}
		var response typesafe.SystemOneResponse
		if err := json.Unmarshal(fixture.response, &response); err != nil {
			return nil, func() {}, fmt.Errorf("decode replay fixture %d: %w", index, err)
		}
		fixture.response = append([]byte(nil), fixture.response...)
		responses[requestKey] = fixture
	}

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/systemone" {
			http.Error(writer, "replay only serves POST /v1/systemone", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "could not read replay request", http.StatusBadRequest)
			return
		}
		requestKey, err := canonicalReplayBody(body)
		if err != nil {
			http.Error(writer, "replay request is not valid JSON", http.StatusBadRequest)
			return
		}
		fixture, found := responses[requestKey]
		if !found {
			http.Error(writer, "no recorded response for exact request", http.StatusNotFound)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-typesafe-request-id", fixture.requestID)
		_, _ = writer.Write(fixture.response)
	}))

	policy := typesafe.DefaultRetryPolicy()
	policy.MaxRetries = 0
	replayModel := strings.TrimSpace(model)
	if replayModel == "" {
		replayModel = "jev-replay"
	}
	client, err := typesafe.NewClient(
		typesafe.WithAPIKey("local-replay-key"),
		typesafe.WithBaseURL(server.URL),
		typesafe.WithDefaultModel(replayModel),
		typesafe.WithRetryPolicy(policy),
	)
	if err != nil {
		server.Close()
		return nil, func() {}, fmt.Errorf("create replay client: %w", err)
	}
	return client, server.Close, nil
}

func canonicalReplayBody(body []byte) (string, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return "", fmt.Errorf("request must contain one JSON value")
	}
	return canonicalReplayJSON(value)
}

func canonicalReplayJSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var normalized any
	if err := decoder.Decode(&normalized); err != nil {
		return "", err
	}
	canonical, err := json.Marshal(normalized)
	if err != nil {
		return "", err
	}
	return string(canonical), nil
}

func newComparisonReplayLLM() (core.LLM, error) {
	predictions := []supportCategory{
		categoryBilling,
		categoryTechnical,
		categoryAccount,
		categoryProduct,
		categoryAccount,
		categoryTechnical,
		categoryProduct,
		categoryProduct,
	}
	responses := make([]recordedLLMResponse, len(comparisonTickets))
	for index, ticket := range comparisonTickets {
		promptTokens := 108 + index
		responses[index] = recordedLLMResponse{
			match:   ticket.Text,
			content: "category:\n" + string(predictions[index]),
			usage: core.TokenInfo{
				PromptTokens:     promptTokens,
				CompletionTokens: 4,
				TotalTokens:      promptTokens + 4,
			},
		}
	}
	return newTextReplayLLM("comparison-replay-llm", responses)
}

type textReplayLLM struct {
	*core.BaseLLM
	responses []recordedLLMResponse
}

var _ core.LLM = (*textReplayLLM)(nil)

func newTextReplayLLM(model string, responses []recordedLLMResponse) (core.LLM, error) {
	if strings.TrimSpace(model) == "" || len(responses) == 0 {
		return nil, fmt.Errorf("replay LLM requires a model and responses")
	}
	for index, response := range responses {
		if response.match == "" || response.content == "" {
			return nil, fmt.Errorf("replay LLM response %d is incomplete", index)
		}
	}
	return &textReplayLLM{
		BaseLLM:   core.NewBaseLLM("replay", core.ModelID(model), []core.Capability{core.CapabilityCompletion}, nil),
		responses: append([]recordedLLMResponse(nil), responses...),
	}, nil
}

func (l *textReplayLLM) Generate(ctx context.Context, prompt string, _ ...core.GenerateOption) (*core.LLMResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, response := range l.responses {
		if strings.Contains(prompt, response.match) {
			usage := response.usage
			return &core.LLMResponse{Content: response.content, Usage: &usage}, nil
		}
	}
	return nil, fmt.Errorf("replay LLM: no recorded response matches the prompt")
}

func (*textReplayLLM) GenerateWithJSON(context.Context, string, ...core.GenerateOption) (map[string]any, error) {
	return nil, fmt.Errorf("replay LLM: JSON generation is not recorded")
}

func (*textReplayLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, fmt.Errorf("replay LLM: function generation is not recorded")
}

func (*textReplayLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("replay LLM: embeddings are not recorded")
}

func (*textReplayLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("replay LLM: embeddings are not recorded")
}

func (*textReplayLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("replay LLM: streaming is not recorded")
}

func newRecordingLLM(llm core.LLM) *recordingLLM {
	return &recordingLLM{LLM: llm}
}

func (l *recordingLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	response, err := l.LLM.Generate(ctx, prompt, options...)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.calls++
	if response != nil && response.Usage != nil {
		l.usageCalls++
		l.prompt += response.Usage.PromptTokens
		l.completion += response.Usage.CompletionTokens
	}
	return response, err
}

func (l *recordingLLM) Snapshot() llmCallStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	return llmCallStats{
		Calls:      l.calls,
		UsageCalls: l.usageCalls,
		Prompt:     l.prompt,
		Completion: l.completion,
	}
}

func newLiveLLM(apiKey string, model core.ModelID) (core.LLM, error) {
	llms.EnsureFactory()
	created, err := llms.NewLLM(apiKey, model)
	if err != nil {
		return nil, err
	}
	// A head-to-head measurement must reach the provider rather than count a
	// transparent dspy-go response-cache hit as a live Predict call.
	if cached, ok := created.(*cache.CachedLLM); ok {
		cached.SetCacheEnabled(false)
	}
	return created, nil
}

func percentile(values []time.Duration, quantile float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]time.Duration(nil), values...)
	slices.Sort(ordered)
	quantile = math.Max(0, math.Min(1, quantile))
	index := int(math.Ceil(quantile*float64(len(ordered)))) - 1
	if index < 0 {
		index = 0
	}
	return ordered[index]
}
