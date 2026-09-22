package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	// Environment variables understood by NewClient.
	APIKeyEnv       = "TYPESAFE_API_KEY"
	BaseURLEnv      = "TYPESAFE_BASE_URL"
	DefaultModelEnv = "TYPESAFE_DEFAULT_MODEL"

	DefaultBaseURL = "https://api.typesafe.ai"
	DefaultModel   = "jev-latest"
	DefaultTimeout = 10 * time.Second

	systemOnePath   = "/v1/systemone"
	maxResponseSize = 4 << 20
	sdkIdentifier   = "dspy-go/experimental"
)

// RetryPolicy controls retries of transient HTTP and connection failures.
// It has no independent total budget; use the call context for a total deadline.
type RetryPolicy struct {
	// MaxRetries is the number of retries after the initial attempt.
	MaxRetries int
	// BackoffInitial is the delay before the first retry.
	BackoffInitial time.Duration
	// BackoffMax caps exponential backoff.
	BackoffMax time.Duration
	// BackoffJitter is the fraction randomly subtracted from a backoff delay.
	BackoffJitter float64
	// RespectRetryAfter enables Retry-After and retry-after-ms handling.
	RespectRetryAfter bool
	// MaxRetryAfter limits accepted server-requested delays.
	MaxRetryAfter time.Duration
}

// DefaultRetryPolicy returns the defaults used by NewClient.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxRetries:        2,
		BackoffInitial:    500 * time.Millisecond,
		BackoffMax:        5 * time.Second,
		BackoffJitter:     0.25,
		RespectRetryAfter: true,
		MaxRetryAfter:     time.Minute,
	}
}

func (p RetryPolicy) validate() error {
	if p.MaxRetries < 0 {
		return validationError("retry.max_retries", "must be non-negative")
	}
	if p.BackoffInitial < 0 {
		return validationError("retry.backoff_initial", "must be non-negative")
	}
	if p.BackoffMax < 0 {
		return validationError("retry.backoff_max", "must be non-negative")
	}
	if math.IsNaN(p.BackoffJitter) || math.IsInf(p.BackoffJitter, 0) || p.BackoffJitter < 0 || p.BackoffJitter > 1 {
		return validationError("retry.backoff_jitter", "must be between zero and one")
	}
	if p.MaxRetryAfter < 0 {
		return validationError("retry.max_retry_after", "must be non-negative")
	}
	return nil
}

type clientConfig struct {
	apiKey    string
	baseURL   string
	model     string
	timeout   time.Duration
	http      *http.Client
	retry     RetryPolicy
	userAgent string
}

// ClientOption configures a Client.
type ClientOption func(*clientConfig) error

// WithAPIKey sets the API credential. It takes precedence over TYPESAFE_API_KEY.
func WithAPIKey(apiKey string) ClientOption {
	return func(config *clientConfig) error {
		config.apiKey = apiKey
		return nil
	}
}

// WithBaseURL sets the API root. It takes precedence over TYPESAFE_BASE_URL.
func WithBaseURL(baseURL string) ClientOption {
	return func(config *clientConfig) error {
		config.baseURL = baseURL
		return nil
	}
}

// WithDefaultModel sets the model used when a request omits Model.
func WithDefaultModel(model string) ClientOption {
	return func(config *clientConfig) error {
		config.model = model
		return nil
	}
}

// WithTimeout sets the per-attempt timeout.
func WithTimeout(timeout time.Duration) ClientOption {
	return func(config *clientConfig) error {
		config.timeout = timeout
		return nil
	}
}

// WithHTTPClient supplies the transport used for requests. The Client does not
// mutate or close it.
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(config *clientConfig) error {
		if httpClient == nil {
			return validationError("http_client", "must not be nil")
		}
		config.http = httpClient
		return nil
	}
}

// WithRetryPolicy replaces the default retry policy.
func WithRetryPolicy(policy RetryPolicy) ClientOption {
	return func(config *clientConfig) error {
		if err := policy.validate(); err != nil {
			return err
		}
		config.retry = policy
		return nil
	}
}

// Client calls TypeSafe's System One HTTP API. It is safe for concurrent use
// when its configured http.Client is safe for concurrent use.
type Client struct {
	apiKey    string
	baseURL   string
	model     string
	timeout   time.Duration
	http      *http.Client
	retry     RetryPolicy
	userAgent string
}

// NewClient constructs a client from environment defaults and explicit options.
// Explicit options take precedence over environment variables.
func NewClient(options ...ClientOption) (*Client, error) {
	config := clientConfig{
		apiKey:    strings.TrimSpace(os.Getenv(APIKeyEnv)),
		baseURL:   environmentOrDefault(BaseURLEnv, DefaultBaseURL),
		model:     environmentOrDefault(DefaultModelEnv, DefaultModel),
		timeout:   DefaultTimeout,
		http:      http.DefaultClient,
		retry:     DefaultRetryPolicy(),
		userAgent: sdkIdentifier,
	}
	for _, option := range options {
		if option == nil {
			return nil, validationError("client_option", "must not be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}

	config.apiKey = strings.TrimSpace(config.apiKey)
	if err := validateAPIKey(config.apiKey); err != nil {
		return nil, err
	}
	baseURL, err := normalizeBaseURL(config.baseURL)
	if err != nil {
		return nil, err
	}
	config.model = strings.TrimSpace(config.model)
	if config.model == "" {
		return nil, validationError("default_model", "must not be empty")
	}
	if config.timeout <= 0 {
		return nil, validationError("timeout", "must be positive")
	}
	if config.http == nil {
		return nil, validationError("http_client", "must not be nil")
	}
	if err := config.retry.validate(); err != nil {
		return nil, err
	}

	return &Client{
		apiKey: config.apiKey, baseURL: baseURL, model: config.model,
		timeout: config.timeout, http: config.http, retry: config.retry,
		userAgent: config.userAgent,
	}, nil
}

// DefaultModelName returns the configured request default without exposing
// credentials or transport state.
func (c *Client) DefaultModelName() string { return c.model }

// String describes non-secret client configuration. The API key is always
// redacted, including when Client is formatted with the fmt package.
func (c *Client) String() string {
	if c == nil {
		return "typesafe.Client<nil>"
	}
	return fmt.Sprintf("typesafe.Client{base_url:%q, default_model:%q, timeout:%s, api_key:<redacted>}", c.baseURL, c.model, c.timeout)
}

// GoString is the %#v-safe counterpart to String.
func (c *Client) GoString() string { return c.String() }

// SystemOne answers named questions about the request state.
func (c *Client) SystemOne(ctx context.Context, request SystemOneRequest) (*SystemOneResponse, error) {
	if ctx == nil {
		return nil, validationError("context", "must not be nil")
	}
	if err := request.validate(c.model); err != nil {
		return nil, err
	}
	if request.Model == "" {
		request.Model = c.model
	} else {
		request.Model = strings.TrimSpace(request.Model)
	}

	body, err := json.Marshal(request)
	if err != nil {
		return nil, validationError("request", "could not be encoded as JSON: "+err.Error())
	}

	for attempt := 0; ; attempt++ {
		response, err := c.systemOneAttempt(ctx, body, request, attempt)
		if err == nil {
			return response, nil
		}
		if attempt >= c.retry.MaxRetries || !isRetryable(err) {
			return nil, err
		}
		delay := c.retryDelay(attempt, err)
		if err := waitForRetry(ctx, delay); err != nil {
			return nil, err
		}
	}
}

func (c *Client) systemOneAttempt(ctx context.Context, body []byte, expected SystemOneRequest, retryCount int) (*SystemOneResponse, error) {
	attemptContext, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	request, err := http.NewRequestWithContext(attemptContext, http.MethodPost, c.baseURL+systemOnePath, bytes.NewReader(body))
	if err != nil {
		return nil, validationError("base_url", "could not construct request: "+err.Error())
	}
	request.Header.Set("Authorization", "Bearer "+c.apiKey)
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", c.userAgent)
	request.Header.Set("X-TypeSafe-SDK", c.userAgent)
	request.Header.Set("X-TypeSafe-Runtime", fmt.Sprintf("go/%s (%s; %s)", strings.TrimPrefix(runtime.Version(), "go"), runtime.GOOS, runtime.GOARCH))
	if retryCount > 0 {
		request.Header.Set("X-TypeSafe-Retry-Count", strconv.Itoa(retryCount))
	}

	httpResponse, err := c.http.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if errors.Is(attemptContext.Err(), context.DeadlineExceeded) || isNetworkTimeout(err) {
			return nil, &TimeoutError{Duration: c.timeout, Err: err}
		}
		return nil, &ConnectionError{Err: err}
	}
	defer httpResponse.Body.Close()
	requestID := httpResponse.Header.Get("x-typesafe-request-id")

	responseBody, err := io.ReadAll(io.LimitReader(httpResponse.Body, maxResponseSize+1))
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, &ConnectionError{Err: fmt.Errorf("read response body: %w", err)}
	}
	if len(responseBody) > maxResponseSize {
		validation := responseValidationError("", fmt.Sprintf("body exceeds %d bytes", maxResponseSize))
		validation.StatusCode = httpResponse.StatusCode
		validation.RequestID = requestID
		return nil, validation
	}

	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		return nil, apiErrorFromResponse(httpResponse, responseBody, requestID)
	}

	var result SystemOneResponse
	if err := json.Unmarshal(responseBody, &result); err != nil {
		var validation *ResponseValidationError
		if !errors.As(err, &validation) {
			validation = responseValidationError("", err.Error())
		}
		validation.StatusCode = httpResponse.StatusCode
		validation.RequestID = requestID
		return nil, validation
	}
	if validation := validateResponseForRequest(&result, expected); validation != nil {
		validation.StatusCode = httpResponse.StatusCode
		validation.RequestID = requestID
		return nil, validation
	}
	result.RequestID = requestID
	return &result, nil
}

func (c *Client) retryDelay(attempt int, err error) time.Duration {
	var apiError *APIError
	if c.retry.RespectRetryAfter && errors.As(err, &apiError) && apiError.RetryAfter >= 0 && apiError.RetryAfter <= c.retry.MaxRetryAfter {
		return apiError.RetryAfter
	}
	if c.retry.BackoffInitial == 0 || c.retry.BackoffMax == 0 {
		return 0
	}
	delay := c.retry.BackoffInitial
	for i := 0; i < attempt && delay < c.retry.BackoffMax; i++ {
		if delay > c.retry.BackoffMax/2 {
			delay = c.retry.BackoffMax
			break
		}
		delay *= 2
	}
	if delay > c.retry.BackoffMax {
		delay = c.retry.BackoffMax
	}
	if c.retry.BackoffJitter > 0 {
		delay = time.Duration(float64(delay) * (1 - rand.Float64()*c.retry.BackoffJitter))
	}
	return delay
}

func apiErrorFromResponse(response *http.Response, body []byte, requestID string) *APIError {
	var decoded any
	if len(body) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.UseNumber()
		if err := decoder.Decode(&decoded); err != nil {
			decoded = string(body)
		}
	}
	return &APIError{
		StatusCode: response.StatusCode,
		RequestID:  requestID,
		Body:       decoded,
		RetryAfter: parseRetryAfter(response.Header, time.Now()),
		Message:    extractAPIMessage(decoded),
	}
}

func parseRetryAfter(headers http.Header, now time.Time) time.Duration {
	if raw := strings.TrimSpace(headers.Get("retry-after-ms")); raw != "" {
		if milliseconds, err := strconv.ParseFloat(raw, 64); err == nil && milliseconds >= 0 && !math.IsInf(milliseconds, 0) && !math.IsNaN(milliseconds) {
			return time.Duration(milliseconds * float64(time.Millisecond))
		}
	}
	if raw := strings.TrimSpace(headers.Get("Retry-After")); raw != "" {
		if seconds, err := strconv.ParseFloat(raw, 64); err == nil && seconds >= 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds) {
			return time.Duration(seconds * float64(time.Second))
		}
		if date, err := http.ParseTime(raw); err == nil {
			return max(0, date.Sub(now))
		}
	}
	return -1
}

func isRetryable(err error) bool {
	var apiError *APIError
	if errors.As(err, &apiError) {
		return apiError.Retryable()
	}
	var timeoutError *TimeoutError
	if errors.As(err, &timeoutError) {
		return true
	}
	var connectionError *ConnectionError
	return errors.As(err, &connectionError)
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func environmentOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func validateAPIKey(apiKey string) error {
	if apiKey == "" {
		return validationError("api_key", "missing; pass WithAPIKey or set "+APIKeyEnv)
	}
	for _, character := range apiKey {
		if character < 0x21 || character > 0x7e {
			return validationError("api_key", "must contain only printable ASCII characters without whitespace")
		}
	}
	return nil
}

func normalizeBaseURL(value string) (string, error) {
	value = strings.TrimRight(strings.TrimSpace(value), "/")
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", validationError("base_url", "must be an absolute HTTP or HTTPS URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", validationError("base_url", "must use HTTP or HTTPS")
	}
	if parsed.User != nil {
		return "", validationError("base_url", "must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", validationError("base_url", "must not contain a query or fragment")
	}
	return value, nil
}

func isNetworkTimeout(err error) bool {
	var networkError net.Error
	return errors.As(err, &networkError) && networkError.Timeout()
}

func isNilInterface(value any) bool {
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
