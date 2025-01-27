package llms

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/utils"
)

// LlamacppLLM implements the core.LLM interface for Llamacpp-hosted models.
type LlamacppLLM struct {
	*core.BaseLLM
}

// NewLlamacppLLM creates a new LlamacppLLM instance.
func NewLlamacppLLM(endpoint string) (*LlamacppLLM, error) {
	if endpoint == "" {
		endpoint = "http://localhost:8080" // Default llamacpp endpoint
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}
	endpointCfg := &core.EndpointConfig{
		BaseURL: endpoint,
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 10 * 60, // Default timeout
	}

	return &LlamacppLLM{
		BaseLLM: core.NewBaseLLM("llamacpp", "", capabilities, endpointCfg),
	}, nil
}

type llamacppRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type llamacppResponse struct {
	Index           int    `json:"index"`            // Response index number
	Content         string `json:"content"`          // The actual generated text response
	Tokens          []any  `json:"tokens"`           // Token information (if requested)
	IDSlot          int    `json:"id_slot"`          // Slot ID in the server
	Stop            bool   `json:"stop"`             // Whether generation stopped naturally
	Model           string `json:"model"`            // Model identifier
	TokensPredicted int    `json:"tokens_predicted"` // Number of tokens generated
	TokensEvaluated int    `json:"tokens_evaluated"` // Number of tokens processed

	// Input and processing metadata
	Prompt       string `json:"prompt"`        // Original input prompt
	HasNewLine   bool   `json:"has_new_line"`  // Whether response has a newline
	Truncated    bool   `json:"truncated"`     // Whether response was truncated
	StopType     string `json:"stop_type"`     // Type of stop condition met
	StoppingWord string `json:"stopping_word"` // Word that triggered stopping
	TokensCached int    `json:"tokens_cached"` // Number of tokens cached
	// Performance timing information
	Timings struct {
		PromptN             int     `json:"prompt_n"`               // Number of prompt tokens
		PromptMS            float64 `json:"prompt_ms"`              // Time spent on prompt processing
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`    // Average time per prompt token
		PromptPerSecond     float64 `json:"prompt_per_second"`      // Tokens per second for prompt
		PredictedN          int     `json:"predicted_n"`            // Number of predicted tokens
		PredictedMS         float64 `json:"predicted_ms"`           // Time spent on prediction
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"` // Average time per predicted token
		PredictedPerSecond  float64 `json:"predicted_per_second"`   // Tokens per second for prediction
	} `json:"timings"`
}

// Generate implements the core.LLM interface.
func (o *LlamacppLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := llamacppRequest{
		Prompt:      prompt,
		Stream:      false,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/completion", bytes.NewBuffer(jsonData))
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range o.GetEndpointConfig().Headers {
		req.Header.Set(key, value)
	}

	resp, err := o.GetHTTPClient().Do(req)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return &core.LLMResponse{}, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}
	var llamacppResp llamacppResponse
	err = json.Unmarshal(body, &llamacppResp)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	// Create token info if available
	tokenInfo := &core.TokenInfo{
		PromptTokens:     llamacppResp.TokensEvaluated,
		CompletionTokens: llamacppResp.TokensPredicted,
		TotalTokens:      llamacppResp.TokensEvaluated + llamacppResp.TokensPredicted,
	}
	return &core.LLMResponse{Content: llamacppResp.Content, Usage: tokenInfo}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *LlamacppLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}
