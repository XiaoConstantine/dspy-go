package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
)

//go:embed testdata/*.json
var replayFixtureFS embed.FS

type systemOneReplayFixture struct {
	request   typesafe.SystemOneRequest
	response  []byte
	requestID string
}

type countingSystemOneClient struct {
	inner decide.SystemOneClient
	calls atomic.Int64
}

func (c *countingSystemOneClient) SystemOne(ctx context.Context, request typesafe.SystemOneRequest) (*typesafe.SystemOneResponse, error) {
	c.calls.Add(1)
	return c.inner.SystemOne(ctx, request)
}

func (c *countingSystemOneClient) Calls() int64 { return c.calls.Load() }

func newSystemOneClient(replay bool, model string) (decide.SystemOneClient, func(), error) {
	if !replay {
		options := make([]typesafe.ClientOption, 0, 1)
		if strings.TrimSpace(model) != "" {
			options = append(options, typesafe.WithDefaultModel(model))
		}
		client, err := typesafe.NewClient(options...)
		return client, func() {}, err
	}

	fixtureFiles := []string{
		"testdata/account_question_response.json",
		"testdata/incident_response.json",
		"testdata/refund_response.json",
		"testdata/product_question_response.json",
	}
	fixtures := make([]systemOneReplayFixture, len(replayTickets))
	for index, ticket := range replayTickets {
		body, err := replayFixtureFS.ReadFile(fixtureFiles[index])
		if err != nil {
			return nil, func() {}, fmt.Errorf("read replay fixture: %w", err)
		}
		fixtures[index] = systemOneReplayFixture{
			request:   expectedSystemOneReplayRequest(ticket.Text),
			response:  body,
			requestID: fmt.Sprintf("req_support_replay_%d", index+1),
		}
	}
	return startSystemOneReplay(fixtures, model)
}

func expectedSystemOneReplayRequest(ticket string) typesafe.SystemOneRequest {
	instructions := func(question string) map[string]any {
		return map[string]any{
			"question": question,
			"task":     "Triage the ticket. Treat account-specific changes and transactions as requiring tools, and broad outages or security-sensitive requests as requiring human review.",
			"inputs": []map[string]string{{
				"name": "ticket", "description": "Support ticket text",
			}},
		}
	}
	return typesafe.SystemOneRequest{
		State: map[string]any{"ticket": ticket},
		Model: "jev-replay",
		Questions: map[string]typesafe.Question{
			"answerable":  typesafe.NoulQuestion{Instructions: instructions("Can a bounded acknowledgment be drafted from the ticket alone, without private account data, external tools, product-specific instructions, or human judgment?")},
			"needs_human": typesafe.NoulQuestion{Instructions: instructions("Does policy, security, ambiguity, or incident impact require human review before replying?")},
			"category": typesafe.ChoiceQuestion{
				Instructions: instructions("Which support category best matches this ticket?"),
				Criteria: map[string]any{
					"billing":   "Charges, invoices, payments, or refunds",
					"technical": "Product failures, bugs, or service outages",
					"account":   "Login, identity, permissions, or account settings",
					"product":   "Product usage, features, or how-to questions",
					"other":     "Anything not covered by the other categories",
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

// replayLLM is the deterministic generative stand-in used by -replay. It only
// implements the JSON path exercised by ChainOfThought in this example.
type replayLLM struct {
	*core.BaseLLM
}

var _ core.LLM = (*replayLLM)(nil)

func newReplayLLM() core.LLM {
	return &replayLLM{BaseLLM: core.NewBaseLLM(
		"replay",
		core.ModelID("support-replay-llm"),
		[]core.Capability{core.CapabilityCompletion, core.CapabilityJSON},
		nil,
	)}
}

func (l *replayLLM) GenerateWithJSON(ctx context.Context, prompt string, _ ...core.GenerateOption) (map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	responses := map[string]map[string]any{
		replayTickets[0].Text: {
			"reasoning": "The only releasable text is the locally approved acknowledgment.",
			"reply":     approvedAcknowledgment,
		},
		replayTickets[3].Text: {
			"reasoning": "The only releasable text is the locally approved acknowledgment.",
			"reply":     approvedAcknowledgment,
		},
	}
	for ticket, response := range responses {
		if strings.Contains(prompt, ticket) {
			return maps.Clone(response), nil
		}
	}
	return nil, fmt.Errorf("replay LLM: no recorded draft for prompt")
}

func (l *replayLLM) Generate(context.Context, string, ...core.GenerateOption) (*core.LLMResponse, error) {
	return nil, fmt.Errorf("replay LLM: text generation is not recorded")
}

func (l *replayLLM) GenerateWithFunctions(context.Context, string, []map[string]any, ...core.GenerateOption) (map[string]any, error) {
	return nil, fmt.Errorf("replay LLM: function generation is not recorded")
}

func (l *replayLLM) CreateEmbedding(context.Context, string, ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("replay LLM: embeddings are not recorded")
}

func (l *replayLLM) CreateEmbeddings(context.Context, []string, ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("replay LLM: embeddings are not recorded")
}

func (l *replayLLM) StreamGenerate(context.Context, string, ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("replay LLM: streaming is not recorded")
}

// countingLLM counts the JSON call and any text fallback used by this example.
type countingLLM struct {
	core.LLM
	calls atomic.Int64
}

func (l *countingLLM) Calls() int64 { return l.calls.Load() }

func (l *countingLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	l.calls.Add(1)
	return l.LLM.Generate(ctx, prompt, options...)
}

func (l *countingLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]any, error) {
	l.calls.Add(1)
	return l.LLM.GenerateWithJSON(ctx, prompt, options...)
}

func newLiveLLM(apiKey, model string) (core.LLM, error) {
	if strings.TrimSpace(apiKey) == "" {
		apiKey = firstEnvironmentValue("DSPY_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
	}
	llms.EnsureFactory()
	return llms.NewLLM(apiKey, core.ModelID(model))
}

func firstEnvironmentValue(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}
