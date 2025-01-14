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

// OllamaLLM implements the core.LLM interface for Ollama-hosted models.
type OllamaLLM struct {
	*core.BaseLLM
}

// NewOllamaLLM creates a new OllamaLLM instance.
func NewOllamaLLM(endpoint, model string) (*OllamaLLM, error) {
	if endpoint == "" {
		endpoint = "http://localhost:11434" // Default Ollama endpoint
	}
	capabilities := []core.Capability{
		core.CapabilityCompletion,
		core.CapabilityChat,
		core.CapabilityJSON,
	}
	endpointCfg := &core.EndpointConfig{
		BaseURL: endpoint,
		Path:    "api/generate", // Ollama's generation endpoint
		Headers: map[string]string{
			"Content-Type": "application/json",
		},
		TimeoutSec: 30, // Default timeout
	}

	return &OllamaLLM{
		BaseLLM: core.NewBaseLLM("ollama", core.ModelID(model), capabilities, endpointCfg),
	}, nil
}

type ollamaRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

type ollamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
}

// Generate implements the core.LLM interface.
func (o *OllamaLLM) Generate(ctx context.Context, prompt string, options ...core.GenerateOption) (*core.LLMResponse, error) {
	opts := core.NewGenerateOptions()
	for _, opt := range options {
		opt(opts)
	}

	reqBody := ollamaRequest{
		Model:       o.ModelID(),
		Prompt:      prompt,
		Stream:      false,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.GetEndpointConfig().BaseURL+"/api/generate", bytes.NewBuffer(jsonData))
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

	var ollamaResp ollamaResponse
	err = json.Unmarshal(body, &ollamaResp)
	if err != nil {
		return &core.LLMResponse{}, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// TODO: add token usage
	return &core.LLMResponse{Content: ollamaResp.Response}, nil
}

// GenerateWithJSON implements the core.LLM interface.
func (o *OllamaLLM) GenerateWithJSON(ctx context.Context, prompt string, options ...core.GenerateOption) (map[string]interface{}, error) {
	response, err := o.Generate(ctx, prompt, options...)
	if err != nil {
		return nil, err
	}

	return utils.ParseJSONResponse(response.Content)
}
