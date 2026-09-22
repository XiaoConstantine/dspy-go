package typesafe_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

func TestClientSystemOneFixture(t *testing.T) {
	wantRequest := readFixture(t, "testdata/system_one_request.json")
	responseFixture := readFixture(t, "testdata/system_one_response.json")

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		assert.Equal(t, http.MethodPost, request.Method)
		assert.Equal(t, "/v1/systemone", request.URL.Path)
		assert.Equal(t, "Bearer fixture-key", request.Header.Get("Authorization"))
		assert.Equal(t, "application/json", request.Header.Get("Accept"))
		assert.Equal(t, "application/json", request.Header.Get("Content-Type"))
		assert.NotEmpty(t, request.Header.Get("X-TypeSafe-SDK"))
		assert.NotEmpty(t, request.Header.Get("X-TypeSafe-Runtime"))
		assert.Empty(t, request.Header.Get("X-TypeSafe-Retry-Count"))

		body, err := io.ReadAll(request.Body)
		require.NoError(t, err)
		assert.JSONEq(t, string(wantRequest), string(body))

		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-typesafe-request-id", "req_fixture")
		_, err = writer.Write(responseFixture)
		require.NoError(t, err)
	}))
	defer server.Close()

	client := newFixtureClient(t, server.URL, "jev-test-pinned", noRetryPolicy())
	response, err := client.SystemOne(context.Background(), fixtureRequest())
	require.NoError(t, err)

	assert.Equal(t, "jev-test-2026-09-01", response.Model)
	assert.Equal(t, "req_fixture", response.RequestID)
	assert.Equal(t, typesafe.Usage{InputTokens: 120, OutputTokens: 12}, response.Usage)
	require.Len(t, response.Answers, 3)

	noul, ok := response.Answers["urgent"].(typesafe.NoulAnswer)
	require.True(t, ok)
	assert.InDelta(t, 0.82, noul.Probability, 1e-12)

	score, ok := response.Answers["severity"].(typesafe.ScoreAnswer)
	require.True(t, ok)
	assert.Equal(t, map[int]float64{0: 0.1, 1: 0.2, 2: 0.7}, score.Probabilities)
	assert.Equal(t, "Blocking", score.Legend[2])

	choice, ok := response.Answers["category"].(typesafe.ChoiceAnswer)
	require.True(t, ok)
	assert.Equal(t, "technical", choice.Choice)
	assert.Equal(t, map[string]float64{"billing": 0.2, "technical": 0.8}, choice.Probabilities)
}

func TestClientRequestModelOverride(t *testing.T) {
	var model string
	responseFixture := readFixture(t, "testdata/system_one_response.json")
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(request.Body).Decode(&body))
		model, _ = body["model"].(string)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write(responseFixture)
	}))
	defer server.Close()

	client := newFixtureClient(t, server.URL, "default-model", noRetryPolicy())
	request := fixtureRequest()
	request.Model = "pinned-override"
	_, err := client.SystemOne(context.Background(), request)
	require.NoError(t, err)
	assert.Equal(t, "pinned-override", model)
	assert.Equal(t, "default-model", client.DefaultModelName())
}

func TestClientClassifiesAPIErrorFixture(t *testing.T) {
	fixture := readFixture(t, "testdata/validation_error.json")
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-typesafe-request-id", "req_invalid")
		writer.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = writer.Write(fixture)
	}))
	defer server.Close()

	client := newFixtureClient(t, server.URL, "jev-test", noRetryPolicy())
	_, err := client.SystemOne(context.Background(), fixtureRequest())
	require.Error(t, err)
	var apiError *typesafe.APIError
	require.ErrorAs(t, err, &apiError)
	assert.Equal(t, http.StatusUnprocessableEntity, apiError.StatusCode)
	assert.Equal(t, "req_invalid", apiError.RequestID)
	assert.Contains(t, apiError.Error(), "questions.severity.score.criteria: Field required")
	assert.False(t, apiError.Retryable())
}

func TestClientRetriesTransientFixtureResponse(t *testing.T) {
	responseFixture := readFixture(t, "testdata/system_one_response.json")
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		attempt := attempts.Add(1)
		if attempt == 1 {
			assert.Empty(t, request.Header.Get("X-TypeSafe-Retry-Count"))
			writer.Header().Set("retry-after-ms", "0")
			writer.WriteHeader(http.StatusServiceUnavailable)
			_, _ = writer.Write([]byte(`{"error":"try again"}`))
			return
		}
		assert.Equal(t, "1", request.Header.Get("X-TypeSafe-Retry-Count"))
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write(responseFixture)
	}))
	defer server.Close()

	policy := noRetryPolicy()
	policy.MaxRetries = 1
	policy.RespectRetryAfter = true
	policy.MaxRetryAfter = time.Second
	client := newFixtureClient(t, server.URL, "jev-test", policy)
	_, err := client.SystemOne(context.Background(), fixtureRequest())
	require.NoError(t, err)
	assert.EqualValues(t, 2, attempts.Load())
}

func TestClientClassifiesConfiguredTimeout(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		<-request.Context().Done()
		return nil, request.Context().Err()
	})}
	client, err := typesafe.NewClient(
		typesafe.WithAPIKey("fixture-key"),
		typesafe.WithHTTPClient(httpClient),
		typesafe.WithTimeout(5*time.Millisecond),
		typesafe.WithRetryPolicy(noRetryPolicy()),
	)
	require.NoError(t, err)

	_, err = client.SystemOne(context.Background(), fixtureRequest())
	require.Error(t, err)
	var timeoutError *typesafe.TimeoutError
	require.ErrorAs(t, err, &timeoutError)
	assert.Equal(t, 5*time.Millisecond, timeoutError.Duration)
	assert.True(t, timeoutError.Timeout())
}

func TestClientCancellationPropagates(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
	}))
	defer server.Close()
	defer close(release)

	client := newFixtureClient(t, server.URL, "jev-test", noRetryPolicy())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := client.SystemOne(ctx, fixtureRequest())
		done <- err
	}()
	<-started
	cancel()

	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("SystemOne did not return after context cancellation")
	}
}

func TestClientValidatesBeforeTransport(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	client := newFixtureClient(t, server.URL, "jev-test", noRetryPolicy())

	_, err := client.SystemOne(context.Background(), typesafe.SystemOneRequest{
		State:     "content",
		Questions: map[string]typesafe.Question{},
	})
	require.Error(t, err)
	var validation *typesafe.ValidationError
	require.ErrorAs(t, err, &validation)
	assert.Equal(t, "questions", validation.Field)
	assert.Zero(t, calls.Load())

	_, err = client.SystemOne(context.Background(), typesafe.SystemOneRequest{
		State: "content",
		Questions: map[string]typesafe.Question{
			"score": typesafe.ScoreQuestion{},
		},
	})
	require.ErrorAs(t, err, &validation)
	assert.Zero(t, calls.Load())
}

func TestClientRejectsMalformedSuccessFixture(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"model":"jev-test",
			"answers":{"urgent":{"type":"noul"}},
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer server.Close()
	client := newFixtureClient(t, server.URL, "jev-test", noRetryPolicy())

	_, err := client.SystemOne(context.Background(), fixtureRequest())
	require.Error(t, err)
	var validation *typesafe.ResponseValidationError
	require.ErrorAs(t, err, &validation)
	assert.Equal(t, "answers.urgent.noul", validation.Field)
}

func TestNewClientRequiresCredential(t *testing.T) {
	t.Setenv(typesafe.APIKeyEnv, "")
	_, err := typesafe.NewClient()
	require.Error(t, err)
	var validation *typesafe.ValidationError
	require.ErrorAs(t, err, &validation)
	assert.Equal(t, "api_key", validation.Field)
}

func TestClientFormattingRedactsCredential(t *testing.T) {
	t.Setenv(typesafe.BaseURLEnv, "")
	t.Setenv(typesafe.DefaultModelEnv, "")
	const secret = "fixture-secret"
	client, err := typesafe.NewClient(typesafe.WithAPIKey(secret))
	require.NoError(t, err)
	assert.NotContains(t, fmt.Sprintf("%v", client), secret)
	assert.NotContains(t, fmt.Sprintf("%+v", client), secret)
	assert.NotContains(t, fmt.Sprintf("%#v", client), secret)
}

func fixtureRequest() typesafe.SystemOneRequest {
	return typesafe.SystemOneRequest{
		State: map[string]any{"ticket": "Payment failed and checkout is unavailable."},
		Questions: map[string]typesafe.Question{
			"urgent": typesafe.NoulQuestion{Instructions: "Does this require immediate attention?"},
			"severity": typesafe.ScoreQuestion{
				Instructions: "How severe is the issue?",
				Criteria:     []any{"Minor", "Disruptive", "Blocking"},
			},
			"category": typesafe.ChoiceQuestion{
				Instructions: "What is this ticket about?",
				Criteria: map[string]any{
					"billing":   "Payment issue",
					"technical": "Product malfunction",
				},
			},
		},
	}
}

func newFixtureClient(t *testing.T, baseURL, model string, policy typesafe.RetryPolicy) *typesafe.Client {
	t.Helper()
	client, err := typesafe.NewClient(
		typesafe.WithAPIKey("fixture-key"),
		typesafe.WithBaseURL(baseURL),
		typesafe.WithDefaultModel(model),
		typesafe.WithRetryPolicy(policy),
	)
	require.NoError(t, err)
	return client
}

func noRetryPolicy() typesafe.RetryPolicy {
	policy := typesafe.DefaultRetryPolicy()
	policy.MaxRetries = 0
	policy.BackoffInitial = 0
	policy.BackoffMax = 0
	policy.BackoffJitter = 0
	return policy
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func readFixture(t *testing.T, path string) []byte {
	t.Helper()
	contents, err := os.ReadFile(path)
	require.NoError(t, err)
	return contents
}
