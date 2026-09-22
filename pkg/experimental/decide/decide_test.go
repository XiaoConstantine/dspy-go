package decide_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

type category string

const (
	billing   category = "billing"
	technical category = "technical"
)

func TestDecideFixtureVerticalSlice(t *testing.T) {
	wantRequest := readFixture(t, "testdata/decide_request.json")
	responseFixture := readFixture(t, "testdata/decide_response.json")
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls++
		body, err := io.ReadAll(request.Body)
		require.NoError(t, err)
		assert.JSONEq(t, string(wantRequest), string(body))
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-typesafe-request-id", "req_decide_fixture")
		_, err = writer.Write(responseFixture)
		require.NoError(t, err)
	}))
	defer server.Close()

	client := newHTTPFixtureClient(t, server.URL)
	module := newFixtureModule(t, client)
	module.SetLLM(nil) // A program-wide text LLM must not replace the client.

	result, err := module.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	assert.Equal(t, 1, calls, "all outputs should be batched into one request")
	assert.Equal(t, "jev-test-2026-09-01", result.Model)
	assert.Equal(t, "req_decide_fixture", result.RequestID)
	assert.Equal(t, typesafe.Usage{InputTokens: 120, OutputTokens: 12}, result.Usage)
	assert.Equal(t, true, result.Outputs["urgent"])
	assert.InDelta(t, 7.4, result.Outputs["severity"], 1e-12)
	assert.Equal(t, technical, result.Outputs["category"])
	assert.IsType(t, category(""), result.Outputs["category"])

	noul := result.Decisions["urgent"].(decide.NoulDecision)
	assert.True(t, noul.Value)
	assert.InDelta(t, 0.82, noul.Probability, 1e-12)
	assert.InDelta(t, 0.64, noul.Confidence, 1e-12)

	score := result.Decisions["severity"].(decide.ScoreDecision)
	assert.InDelta(t, 7.4, score.Value, 1e-12)
	assert.InDelta(t, 1.7, score.ProviderScore, 1e-12)
	assert.InDelta(t, 0.61, score.ProviderConfidence, 1e-12)
	assert.Equal(t, []float64{0, 2, 10}, score.Anchors())
	assert.Equal(t, map[int]float64{0: 0.1, 1: 0.2, 2: 0.7}, score.Probabilities())
	probabilities := score.Probabilities()
	probabilities[0] = 1
	assert.Equal(t, 0.1, score.Probabilities()[0], "evidence accessors must return copies")

	choice := result.Decisions["category"].(decide.ChoiceDecision[category])
	assert.Equal(t, technical, choice.Value)
	assert.Equal(t, technical, choice.ProviderValue)
	assert.Equal(t, "technical", choice.ProviderLabel)
	assert.InDelta(t, 0.73, choice.ProviderConfidence, 1e-12)

	outputs, err := module.Process(context.Background(), fixtureInputs())
	require.NoError(t, err)
	assert.Equal(t, true, outputs["urgent"])
	assert.InDelta(t, 7.4, outputs["severity"], 1e-12)
	assert.Equal(t, technical, outputs["category"])
	assert.Equal(t, 2, calls)

	_, isDemoProvider := any(module).(core.DemoProvider)
	_, isDemoConsumer := any(module).(core.DemoConsumer)
	assert.False(t, isDemoProvider)
	assert.False(t, isDemoConsumer)
	assert.Equal(t, "Decide", module.GetModuleType())
}

func TestDecideLocalParametersPreserveProviderEvidence(t *testing.T) {
	client := newRecordingClient(t)
	module := newFixtureModule(t, client)

	before, err := module.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	require.NoError(t, module.SetThreshold("urgent", 0.9))
	require.NoError(t, module.SetScoreAnchors("severity", []float64{0, 4, 10}))
	require.NoError(t, module.SetChoiceMultipliers("category", map[string]float64{"technical": 0.1}))
	after, err := module.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)

	assert.True(t, before.Outputs["urgent"].(bool))
	assert.False(t, after.Outputs["urgent"].(bool))
	assert.InDelta(t, 7.8, after.Outputs["severity"], 1e-12)
	assert.Equal(t, billing, after.Outputs["category"])

	beforeChoice := before.Decisions["category"].(decide.ChoiceDecision[category])
	afterChoice := after.Decisions["category"].(decide.ChoiceDecision[category])
	assert.Equal(t, technical, afterChoice.ProviderValue)
	assert.Equal(t, "technical", afterChoice.ProviderLabel)
	assert.Equal(t, beforeChoice.ProviderConfidence, afterChoice.ProviderConfidence)
	assert.Equal(t, beforeChoice.Probabilities(), afterChoice.Probabilities())
	assert.Equal(t, billing, afterChoice.Value)
	assert.Equal(t, "billing", afterChoice.LocalLabel)

	beforeScore := before.Decisions["severity"].(decide.ScoreDecision)
	afterScore := after.Decisions["severity"].(decide.ScoreDecision)
	assert.Equal(t, beforeScore.Probabilities(), afterScore.Probabilities())
	assert.Equal(t, beforeScore.ProviderConfidence, afterScore.ProviderConfidence)
	assert.Equal(t, beforeScore.ProviderScore, afterScore.ProviderScore)

	require.Len(t, client.requests(), 2)
	assert.True(t, reflect.DeepEqual(client.requests()[0], client.requests()[1]), "local parameters must not alter provider requests")
}

func TestDecideCloneAndVersionedStateRoundTrip(t *testing.T) {
	client := newRecordingClient(t)
	module := newFixtureModule(t, client)
	require.NoError(t, module.SetThreshold("urgent", 0.9))
	require.NoError(t, module.SetScoreAnchors("severity", []float64{0, 4, 10}))
	require.NoError(t, module.SetChoiceMultipliers("category", map[string]float64{"technical": 0.1}))

	parameters := module.GetTunedParameters()
	encoded, err := json.Marshal(parameters)
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), client.secret)
	assert.Contains(t, string(encoded), `"schema_version":1`)

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(encoded, &decoded))
	restored := newFixtureModule(t, client)
	require.NoError(t, restored.SetTunedParameters(decoded))

	originalResult, err := module.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	restoredResult, err := restored.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	assert.Equal(t, originalResult.Outputs, restoredResult.Outputs)

	clone := module.Clone().(*decide.Decide)
	require.NoError(t, clone.SetThreshold("urgent", 0.2))
	require.NoError(t, clone.SetScoreAnchors("severity", []float64{0, 6, 10}))
	require.NoError(t, clone.SetChoiceMultipliers("category", nil))
	cloneResult, err := clone.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	assert.True(t, cloneResult.Outputs["urgent"].(bool))
	assert.InDelta(t, 8.2, cloneResult.Outputs["severity"], 1e-12)
	assert.Equal(t, technical, cloneResult.Outputs["category"])

	again, err := module.ProcessDecision(context.Background(), fixtureInputs())
	require.NoError(t, err)
	assert.False(t, again.Outputs["urgent"].(bool))
	assert.InDelta(t, 7.8, again.Outputs["severity"], 1e-12)
	assert.Equal(t, billing, again.Outputs["category"])
}

func TestDecideProgramStateDoesNotPersistClientCredential(t *testing.T) {
	client := newRecordingClient(t)
	module := newFixtureModule(t, client)
	require.NoError(t, module.SetThreshold("urgent", 0.77))
	program := core.NewProgram(map[string]core.Module{"decision": module}, nil)
	path := t.TempDir() + "/program.json"
	require.NoError(t, core.SaveProgram(&program, path))

	contents, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.NotContains(t, string(contents), client.secret)

	restored := newFixtureModule(t, client)
	restoredProgram := core.NewProgram(map[string]core.Module{"decision": restored}, nil)
	require.NoError(t, core.LoadProgram(&restoredProgram, path))
	parameters := restored.GetTunedParameters()
	thresholds := parameters["thresholds"].(map[string]float64)
	assert.InDelta(t, 0.77, thresholds["urgent"], 1e-12)
}

func TestDecideRejectsIncompatibleSignatureBeforeRequest(t *testing.T) {
	client := newRecordingClient(t)
	module := newFixtureModule(t, client)
	incompatible := fixtureSignature()
	incompatible.Outputs[0].Name = "renamed"
	module.SetSignature(incompatible)

	_, err := module.Process(context.Background(), fixtureInputs())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "incompatible signature")
	assert.Empty(t, client.requests())

	compatible := fixtureSignature()
	compatible.Instruction = "Assess very carefully."
	compatible.Outputs[0].Description = "Is immediate action needed?"
	module.SetSignature(compatible)
	_, err = module.Process(context.Background(), fixtureInputs())
	require.NoError(t, err)
	require.Len(t, client.requests(), 1)
}

func TestDecideRejectsMalformedEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*typesafe.SystemOneResponse)
		match  string
	}{
		{
			name: "noul out of range",
			mutate: func(response *typesafe.SystemOneResponse) {
				response.Answers["urgent"] = typesafe.NoulAnswer{Probability: 1.1}
			},
			match: "noul probability",
		},
		{
			name: "score missing level",
			mutate: func(response *typesafe.SystemOneResponse) {
				answer := response.Answers["severity"].(typesafe.ScoreAnswer)
				answer.Probabilities = map[int]float64{0: 0.2, 1: 0.8}
				response.Answers["severity"] = answer
			},
			match: "Score distribution",
		},
		{
			name: "choice unknown label",
			mutate: func(response *typesafe.SystemOneResponse) {
				answer := response.Answers["category"].(typesafe.ChoiceAnswer)
				answer.Choice = "unknown"
				response.Answers["category"] = answer
			},
			match: "unknown label",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := newRecordingClient(t)
			test.mutate(client.response)
			module := newFixtureModule(t, client)
			_, err := module.ProcessDecision(context.Background(), fixtureInputs())
			require.Error(t, err)
			assert.Contains(t, err.Error(), test.match)
		})
	}
}

func TestDecidePropagatesCancellation(t *testing.T) {
	client := &cancelClient{}
	module := newFixtureModule(t, client)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := module.Process(ctx, fixtureInputs())
	require.ErrorIs(t, err, context.Canceled)
}

func TestDecideValidatesAnswerSpaces(t *testing.T) {
	client := newRecordingClient(t)
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("input")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	_, err := decide.New(client, signature, decide.Choice[any]("answer",
		decide.Option[any](int(1), "integer"),
		decide.Option[any]("1", "string"),
	))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "colliding")

	_, err = decide.New(client, signature, decide.Score("answer", decide.Level(0, "only")))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "between 2 and 10")
}

func newFixtureModule(t *testing.T, client decide.SystemOneClient) *decide.Decide {
	t.Helper()
	module, err := decide.New(
		client,
		fixtureSignature(),
		decide.Noul("urgent"),
		decide.Score("severity",
			decide.Level(0, "Minor"),
			decide.Level(2, "Disruptive"),
			decide.Level(10, "Blocking"),
		),
		decide.Choice[category]("category",
			decide.Option(billing, "Payment issue"),
			decide.Option(technical, "Product malfunction"),
		),
	)
	require.NoError(t, err)
	return module
}

func fixtureSignature() core.Signature {
	return core.NewSignature(
		[]core.InputField{{Field: core.NewField("ticket", core.WithDescription("Support ticket text"))}},
		[]core.OutputField{
			{Field: core.NewField("urgent", core.WithDescription("Does this require immediate attention?"))},
			{Field: core.NewField("severity", core.WithDescription("Rate the impact."))},
			{Field: core.NewField("category", core.WithDescription("Classify the issue."))},
		},
	).WithInstruction("Assess the support ticket.")
}

func fixtureInputs() map[string]any {
	return map[string]any{"ticket": "Payment failed and checkout is unavailable."}
}

func newHTTPFixtureClient(t *testing.T, baseURL string) *typesafe.Client {
	t.Helper()
	policy := typesafe.DefaultRetryPolicy()
	policy.MaxRetries = 0
	client, err := typesafe.NewClient(
		typesafe.WithAPIKey("fixture-key"),
		typesafe.WithBaseURL(baseURL),
		typesafe.WithDefaultModel("jev-test-pinned"),
		typesafe.WithRetryPolicy(policy),
	)
	require.NoError(t, err)
	return client
}

type recordingClient struct {
	mu       sync.Mutex
	response *typesafe.SystemOneResponse
	record   []typesafe.SystemOneRequest
	secret   string
}

func newRecordingClient(t *testing.T) *recordingClient {
	t.Helper()
	var response typesafe.SystemOneResponse
	require.NoError(t, json.Unmarshal(readFixture(t, "testdata/decide_response.json"), &response))
	return &recordingClient{response: &response, secret: "fixture-secret-must-not-persist"}
}

func (c *recordingClient) SystemOne(ctx context.Context, request typesafe.SystemOneRequest) (*typesafe.SystemOneResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.record = append(c.record, request)
	return c.response, nil
}

func (c *recordingClient) requests() []typesafe.SystemOneRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]typesafe.SystemOneRequest(nil), c.record...)
}

type cancelClient struct{}

func (*cancelClient) SystemOne(ctx context.Context, _ typesafe.SystemOneRequest) (*typesafe.SystemOneResponse, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	contents, err := os.ReadFile(path)
	require.NoError(t, err)
	return contents
}
