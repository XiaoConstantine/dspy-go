package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"

	"github.com/XiaoConstantine/dspy-go/pkg/experimental/decide"
	"github.com/XiaoConstantine/dspy-go/pkg/experimental/typesafe"
)

//go:embed testdata/*.json
var tuningFixtureFS embed.FS

type recordedUrgency struct {
	Ticket   string          `json:"ticket"`
	Response json.RawMessage `json:"response"`
}

type systemOneReplayFixture struct {
	request   typesafe.SystemOneRequest
	response  []byte
	requestID string
}

type countingClient struct {
	inner decide.SystemOneClient
	calls atomic.Int64
}

func (c *countingClient) SystemOne(ctx context.Context, request typesafe.SystemOneRequest) (*typesafe.SystemOneResponse, error) {
	c.calls.Add(1)
	return c.inner.SystemOne(ctx, request)
}

func (c *countingClient) Calls() int64 { return c.calls.Load() }

func newTuningClient(replay bool, model string) (decide.SystemOneClient, func(), error) {
	if !replay {
		options := make([]typesafe.ClientOption, 0, 1)
		if strings.TrimSpace(model) != "" {
			options = append(options, typesafe.WithDefaultModel(model))
		}
		client, err := typesafe.NewClient(options...)
		return client, func() {}, err
	}

	calibration, err := readUrgencyFixtures("testdata/urgent_responses.json", "calibration", calibrationTickets)
	if err != nil {
		return nil, func() {}, err
	}
	heldOut, err := readUrgencyFixtures("testdata/heldout_responses.json", "heldout", heldOutTickets)
	if err != nil {
		return nil, func() {}, err
	}
	return startSystemOneReplay(append(calibration, heldOut...), model)
}

func readUrgencyFixtures(path, split string, tickets []labeledTicket) ([]systemOneReplayFixture, error) {
	contents, err := tuningFixtureFS.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read urgency replay fixture %q: %w", path, err)
	}
	var recorded []recordedUrgency
	if err := json.Unmarshal(contents, &recorded); err != nil {
		return nil, fmt.Errorf("decode urgency replay fixture %q: %w", path, err)
	}
	if len(recorded) != len(tickets) {
		return nil, fmt.Errorf("urgency replay fixture %q has %d records, want %d", path, len(recorded), len(tickets))
	}
	fixtures := make([]systemOneReplayFixture, len(recorded))
	for index, item := range recorded {
		if item.Ticket != tickets[index].Text {
			return nil, fmt.Errorf("urgency replay fixture %q record %d does not match the labeled ticket", path, index)
		}
		fixtures[index] = systemOneReplayFixture{
			request:   expectedSystemOneReplayRequest(item.Ticket),
			response:  item.Response,
			requestID: fmt.Sprintf("req_tuning_%s_replay_%d", split, index+1),
		}
	}
	return fixtures, nil
}

func expectedSystemOneReplayRequest(ticket string) typesafe.SystemOneRequest {
	return typesafe.SystemOneRequest{
		State: map[string]any{"ticket": ticket},
		Model: "jev-replay",
		Questions: map[string]typesafe.Question{
			"urgent": typesafe.NoulQuestion{Instructions: map[string]any{
				"question": "Does this require immediate operational or security escalation?",
				"task":     "Judge operational urgency from the supplied ticket. Routine account and billing requests are not urgent merely because the customer uses time-sensitive language.",
				"inputs": []map[string]string{{
					"name": "ticket", "description": "Support ticket text",
				}},
			}},
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
