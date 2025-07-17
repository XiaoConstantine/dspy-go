package openai

// ChatCompletionRequest represents a request to the OpenAI Chat Completions API.
type ChatCompletionRequest struct {
	Model            string                  `json:"model"`
	Messages         []ChatCompletionMessage `json:"messages"`
	Temperature      *float64                `json:"temperature,omitempty"`
	MaxTokens        *int                    `json:"max_tokens,omitempty"`
	Stream           bool                    `json:"stream,omitempty"`
	ResponseFormat   *ResponseFormat         `json:"response_format,omitempty"`
	TopP             *float64                `json:"top_p,omitempty"`
	FrequencyPenalty *float64                `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64                `json:"presence_penalty,omitempty"`
	Stop             []string                `json:"stop,omitempty"`
	User             string                  `json:"user,omitempty"`
}

// ChatCompletionMessage represents a message in the conversation.
type ChatCompletionMessage struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// ChatCompletionResponse represents a response from the Chat Completions API.
type ChatCompletionResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []ChatChoice    `json:"choices"`
	Usage   CompletionUsage `json:"usage"`
}

// ChatChoice represents a choice in the completion response.
type ChatChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

// CompletionUsage represents token usage information.
type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// ChatCompletionStreamResponse represents a streaming response chunk.
type ChatCompletionStreamResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []ChatChoiceStream `json:"choices"`
}

// ChatChoiceStream represents a choice in a streaming response.
type ChatChoiceStream struct {
	Index        int                   `json:"index"`
	Delta        ChatCompletionMessage `json:"delta"`
	FinishReason *string               `json:"finish_reason"`
}

// EmbeddingRequest represents a request to the Embeddings API.
type EmbeddingRequest struct {
	Input          interface{} `json:"input"` // string or []string
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"`
	Dimensions     *int        `json:"dimensions,omitempty"`
	User           string      `json:"user,omitempty"`
}

// EmbeddingResponse represents a response from the Embeddings API.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  CompletionUsage `json:"usage"`
}

// EmbeddingData represents a single embedding result.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// APIError represents an error returned by the OpenAI API.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// ErrorResponse represents an error response from the OpenAI API.
type ErrorResponse struct {
	Error APIError `json:"error"`
}
