package llms

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	llm "github.com/XiaoConstantine/llm-go"
	llmmodels "github.com/XiaoConstantine/llm-go/models"
	llmopenai "github.com/XiaoConstantine/llm-go/openai"
	llmcodex "github.com/XiaoConstantine/llm-go/openai/codex"
	"github.com/anthropics/anthropic-sdk-go"
)

// Provider-specific names remain aliases for source compatibility. All of them
// now use the same provider-neutral llm-go adapter.
type (
	AnthropicLLM   = GeneratorLLM
	GeminiLLM      = GeneratorLLM
	OpenAILLM      = GeneratorLLM
	OpenAICodexLLM = GeneratorLLM
	OllamaLLM      = GeneratorLLM
	LlamacppLLM    = GeneratorLLM
)

func NewAnthropicLLM(apiKey string, model anthropic.Model) (*AnthropicLLM, error) {
	return NewAnthropicLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "anthropic", APIKey: apiKey,
	}, core.ModelID(model))
}

func NewAnthropicLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*AnthropicLLM, error) {
	if normalized, ok := core.ResolveAnthropicModelID(modelID); ok {
		modelID = normalized
	}
	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_OAUTH_TOKEN"))
	if apiKey == "" {
		apiKey = strings.TrimSpace(config.APIKey)
	}
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	}
	if apiKey == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "Anthropic API key or OAuth token is required")
	}
	config.Name = "anthropic"
	config.APIKey = apiKey
	return newProviderLLM(ctx, config, modelID, llmmodels.AnthropicMessages)
}

func AnthropicProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewAnthropicLLMFromConfig(ctx, config, modelID)
}

func NewGeminiLLM(apiKey string, modelID core.ModelID) (*GeminiLLM, error) {
	return NewGeminiLLMFromConfig(context.Background(), core.ProviderConfig{
		Name: "google", APIKey: apiKey,
	}, modelID)
}

func NewGeminiLLMFromConfig(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (*GeminiLLM, error) {
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
	}
	if apiKey == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "Gemini API key is required")
	}
	if modelID == "" {
		modelID = core.ModelGoogleGeminiFlash
	}
	config.Name = "google"
	config.APIKey = apiKey
	return newProviderLLM(ctx, config, modelID, llmmodels.GeminiGenerateContent)
}

func GeminiProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewGeminiLLMFromConfig(ctx, config, modelID)
}

// OpenAIOption configures the legacy Chat Completions constructor. The default
// dspy-go factory uses llm-go's Responses API for official OpenAI models.
type OpenAIOption func(*OpenAIConfig)

type OpenAIConfig struct {
	baseURL    string
	path       string
	apiKey     string
	headers    map[string]string
	timeout    time.Duration
	httpClient *http.Client
}

func WithAPIKey(apiKey string) OpenAIOption {
	return func(config *OpenAIConfig) { config.apiKey = apiKey }
}

func WithOpenAIBaseURL(baseURL string) OpenAIOption {
	return func(config *OpenAIConfig) { config.baseURL = baseURL }
}

func WithOpenAIPath(path string) OpenAIOption {
	return func(config *OpenAIConfig) { config.path = path }
}

func WithOpenAITimeout(timeout time.Duration) OpenAIOption {
	return func(config *OpenAIConfig) { config.timeout = timeout }
}

func WithHeader(name, value string) OpenAIOption {
	return func(config *OpenAIConfig) {
		if config.headers == nil {
			config.headers = make(map[string]string)
		}
		config.headers[name] = value
	}
}

func WithHTTPClient(client *http.Client) OpenAIOption {
	return func(config *OpenAIConfig) { config.httpClient = client }
}

func NewOpenAILLM(modelID core.ModelID, options ...OpenAIOption) (*OpenAILLM, error) {
	config := OpenAIConfig{
		baseURL: "https://api.openai.com",
		path:    "/v1/chat/completions",
		timeout: 60 * time.Second,
		headers: make(map[string]string),
	}
	for _, option := range options {
		option(&config)
	}
	if config.apiKey == "" {
		config.apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	return newOpenAICompatible("openai", modelID, config)
}

func NewOpenAI(modelID core.ModelID, apiKey string) (*OpenAILLM, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI API key is required")
	}
	return NewOpenAILLM(modelID, WithAPIKey(apiKey))
}

func NewOpenAILLMFromConfig(_ context.Context, providerConfig core.ProviderConfig, modelID core.ModelID) (*OpenAILLM, error) {
	config := openAIConfigFromProvider(providerConfig, "https://api.openai.com", "/v1/chat/completions", 60*time.Second)
	if config.apiKey == "" {
		config.apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	provider := strings.TrimSpace(providerConfig.Name)
	if provider == "" {
		provider = "openai"
	}
	return newOpenAICompatible(provider, modelID, config)
}

// OpenAIProviderFactory uses the current OpenAI Responses API selected by the
// llm-go model catalog rather than the legacy constructor above.
func OpenAIProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	if baseURL := strings.TrimRight(configuredBaseURL(config), "/"); baseURL != "" && baseURL != "https://api.openai.com" && baseURL != "https://api.openai.com/v1" {
		return NewOpenAILLMFromConfig(ctx, config, modelID)
	}
	if config.Endpoint != nil && config.Endpoint.Path != "" {
		path := "/" + strings.Trim(strings.TrimSpace(config.Endpoint.Path), "/")
		if path != "/v1/responses" && path != "/responses" {
			return NewOpenAILLMFromConfig(ctx, config, modelID)
		}
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	if apiKey == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI API key is required")
	}
	config.Name = "openai"
	config.APIKey = apiKey
	return newProviderLLM(ctx, config, modelID, llmmodels.OpenAIResponses)
}

func NewOpenAICompatible(
	provider string,
	modelID core.ModelID,
	baseURL string,
	options ...OpenAIOption,
) (*OpenAILLM, error) {
	defaults := []OpenAIOption{
		WithOpenAIBaseURL(baseURL),
		WithOpenAIPath("/v1/chat/completions"),
	}
	config := OpenAIConfig{timeout: 60 * time.Second, headers: make(map[string]string)}
	for _, option := range append(defaults, options...) {
		option(&config)
	}
	return newOpenAICompatible(provider, modelID, config)
}

func NewLiteLLM(modelID core.ModelID, apiKey string, options ...OpenAIOption) (*OpenAILLM, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "LiteLLM API key is required")
	}
	defaults := []OpenAIOption{WithAPIKey(apiKey), WithOpenAIPath("/chat/completions")}
	return NewOpenAICompatible("litellm", modelID, "http://localhost:4000", append(defaults, options...)...)
}

func NewLocalAI(modelID core.ModelID, baseURL string, options ...OpenAIOption) (*OpenAILLM, error) {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = "http://localhost:8080"
	}
	return NewOpenAICompatible("localai", modelID, baseURL, options...)
}

func NewFastChat(modelID core.ModelID, baseURL string, options ...OpenAIOption) (*OpenAILLM, error) {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = "http://localhost:8000"
	}
	return NewOpenAICompatible("fastchat", modelID, baseURL, options...)
}

func LiteLLMProviderFactory(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	baseURL := firstNonempty(configuredBaseURL(config), "http://localhost:4000")
	options := append([]OpenAIOption{WithOpenAIBaseURL(baseURL)}, providerOpenAIOptions(config)...)
	return NewLiteLLM(modelID, config.APIKey, options...)
}

func LocalAIProviderFactory(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewLocalAI(modelID, firstNonempty(configuredBaseURL(config), "http://localhost:8080"), providerOpenAIOptions(config)...)
}

func FastChatProviderFactory(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewFastChat(modelID, firstNonempty(configuredBaseURL(config), "http://localhost:8000"), providerOpenAIOptions(config)...)
}

type OllamaConfig struct {
	UseOpenAIAPI bool   `yaml:"use_openai_api" json:"use_openai_api"`
	BaseURL      string `yaml:"base_url" json:"base_url"`
	APIKey       string `yaml:"api_key" json:"api_key"`
	Timeout      int    `yaml:"timeout" json:"timeout"`
}

type OllamaOption func(*OllamaConfig)

func WithNativeAPI() OllamaOption {
	return func(config *OllamaConfig) { config.UseOpenAIAPI = false }
}

func WithOpenAIAPI() OllamaOption {
	return func(config *OllamaConfig) { config.UseOpenAIAPI = true }
}

func WithBaseURL(baseURL string) OllamaOption {
	return func(config *OllamaConfig) { config.BaseURL = baseURL }
}

func WithAuth(apiKey string) OllamaOption {
	return func(config *OllamaConfig) { config.APIKey = apiKey }
}

func WithTimeout(timeout int) OllamaOption {
	return func(config *OllamaConfig) { config.Timeout = timeout }
}

func NewOllamaLLM(modelID core.ModelID, options ...OllamaOption) (*OllamaLLM, error) {
	config := OllamaConfig{UseOpenAIAPI: true, BaseURL: "http://localhost:11434", Timeout: 60}
	for _, option := range options {
		option(&config)
	}
	if !config.UseOpenAIAPI {
		return nil, dspyerrors.New(dspyerrors.UnsupportedOperation, "llm-go requires Ollama's OpenAI-compatible API")
	}
	modelID = core.ModelID(strings.TrimPrefix(string(modelID), "ollama:"))
	if strings.TrimSpace(string(modelID)) == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "Ollama model name is required")
	}
	return NewOpenAICompatible(
		"ollama",
		modelID,
		config.BaseURL,
		WithAPIKey(config.APIKey),
		WithOpenAITimeout(time.Duration(config.Timeout)*time.Second),
	)
}

func NewOllamaLLMFromConfig(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (*OllamaLLM, error) {
	options := []OllamaOption{
		WithBaseURL(firstNonempty(configuredBaseURL(config), "http://localhost:11434")),
		WithAuth(config.APIKey),
	}
	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		options = append(options, WithTimeout(config.Endpoint.TimeoutSec))
	}
	if timeout, ok := config.Params["timeout"].(int); ok {
		options = append(options, WithTimeout(timeout))
	}
	if useOpenAI, ok := config.Params["use_openai_api"].(bool); ok && !useOpenAI {
		options = append(options, WithNativeAPI())
	}
	return NewOllamaLLM(modelID, options...)
}

func OllamaProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOllamaLLMFromConfig(ctx, config, modelID)
}

func NewLlamacppLLM(endpoint string) (*LlamacppLLM, error) {
	return NewOpenAICompatible(
		"llamacpp",
		"default",
		firstNonempty(endpoint, "http://localhost:8080"),
		WithOpenAITimeout(10*time.Minute),
	)
}

func NewLlamacppLLMFromConfig(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (*LlamacppLLM, error) {
	modelID = core.ModelID(strings.TrimPrefix(string(modelID), "llamacpp:"))
	if modelID == "" {
		modelID = "default"
	}
	return NewOpenAICompatible(
		"llamacpp",
		modelID,
		firstNonempty(configuredBaseURL(config), "http://localhost:8080"),
		append([]OpenAIOption{WithOpenAITimeout(10 * time.Minute)}, providerOpenAIOptions(config)...)...,
	)
}

func LlamacppProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewLlamacppLLMFromConfig(ctx, config, modelID)
}

type OpenAICodexCredentials struct {
	AccessToken string
	AccountID   string
}

type OpenAICodexCredentialResolver func(context.Context, string) (OpenAICodexCredentials, error)
type OpenAICodexOption func(*openAICodexConfig)

type openAICodexConfig struct {
	baseURL            string
	timeout            time.Duration
	headers            map[string]string
	httpClient         *http.Client
	credentialResolver OpenAICodexCredentialResolver
	originator         string
	reasoningEffort    llm.ReasoningEffort
}

func WithOpenAICodexCredentials(resolver OpenAICodexCredentialResolver) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.credentialResolver = resolver }
}

func WithOpenAICodexBaseURL(baseURL string) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.baseURL = baseURL }
}

func WithOpenAICodexHTTPClient(client *http.Client) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.httpClient = client }
}

func WithOpenAICodexTimeout(timeout time.Duration) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.timeout = timeout }
}

func WithOpenAICodexHeader(name, value string) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.headers[name] = value }
}

func WithOpenAICodexOriginator(originator string) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.originator = originator }
}

func WithOpenAICodexReasoningEffort(effort string) OpenAICodexOption {
	return func(config *openAICodexConfig) { config.reasoningEffort = llm.ReasoningEffort(effort) }
}

func NewOpenAICodexLLM(modelID core.ModelID, options ...OpenAICodexOption) (*OpenAICodexLLM, error) {
	modelID = core.ModelID(strings.TrimPrefix(string(modelID), "openai-codex:"))
	if strings.TrimSpace(string(modelID)) == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI Codex model name is required")
	}
	config := openAICodexConfig{
		baseURL: "https://chatgpt.com/backend-api", timeout: 120 * time.Second,
		headers: make(map[string]string), originator: "dspy-go",
	}
	for _, option := range options {
		option(&config)
	}
	if config.credentialResolver == nil {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI Codex credential resolver is required")
	}
	client := config.httpClient
	if client == nil {
		client = newHTTPClient(config.timeout)
	}
	headers := stringMapHeader(config.headers)
	codexGenerator, err := llmcodex.New(llmcodex.Config{
		Provider: "openai-codex", Model: string(modelID),
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision},
		ResolveCredentials: func(ctx context.Context, rejected string) (llmcodex.Credentials, error) {
			credentials, err := config.credentialResolver(ctx, rejected)
			return llmcodex.Credentials{AccessToken: credentials.AccessToken, AccountID: credentials.AccountID}, err
		},
		BaseURL: config.baseURL, HTTPClient: client, Headers: headers, Originator: config.originator,
	})
	if err != nil {
		return nil, err
	}
	var generator llm.Generator = codexGenerator
	if config.reasoningEffort != "" {
		generator = &reasoningGenerator{Generator: generator, effort: config.reasoningEffort}
	}
	endpoint := &core.EndpointConfig{
		BaseURL: config.baseURL, Path: "/codex/responses",
		Headers: cloneStringMap(config.headers), TimeoutSec: int(config.timeout.Seconds()),
	}
	return adapt(generator, endpoint, client, nil)
}

func NewOpenAICodexLLMFromConfig(_ context.Context, config core.ProviderConfig, modelID core.ModelID) (*OpenAICodexLLM, error) {
	token := firstNonempty(strings.TrimSpace(config.APIKey), strings.TrimSpace(os.Getenv("OPENAI_OAUTH_TOKEN")))
	if token == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI Codex OAuth access token is required")
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
		idToken = firstNonempty(idToken, os.Getenv("OPENAI_ID_TOKEN"))
		if idToken != "" {
			accountID, _ = OpenAICodexAccountIDFromToken(idToken)
		}
	}
	if accountID == "" {
		var err error
		accountID, err = OpenAICodexAccountID(token)
		if err != nil {
			return nil, dspyerrors.Wrap(err, dspyerrors.InvalidInput, "resolve OpenAI Codex account id from account_id, id_token, or access token")
		}
	}
	credentials := OpenAICodexCredentials{AccessToken: token, AccountID: strings.TrimSpace(accountID)}
	options := []OpenAICodexOption{WithOpenAICodexCredentials(func(context.Context, string) (OpenAICodexCredentials, error) {
		return credentials, nil
	})}
	if baseURL := configuredBaseURL(config); baseURL != "" {
		options = append(options, WithOpenAICodexBaseURL(baseURL))
	}
	if config.Endpoint != nil {
		if config.Endpoint.TimeoutSec > 0 {
			options = append(options, WithOpenAICodexTimeout(time.Duration(config.Endpoint.TimeoutSec)*time.Second))
		}
		for name, value := range config.Endpoint.Headers {
			if !strings.EqualFold(name, "Authorization") && !strings.EqualFold(name, "ChatGPT-Account-ID") {
				options = append(options, WithOpenAICodexHeader(name, value))
			}
		}
	}
	return NewOpenAICodexLLM(modelID, options...)
}

func OpenAICodexProviderFactory(ctx context.Context, config core.ProviderConfig, modelID core.ModelID) (core.LLM, error) {
	return NewOpenAICodexLLMFromConfig(ctx, config, modelID)
}

func OpenAICodexAccountID(token string) (string, error) {
	return llmcodex.AccountIDFromToken(token)
}

func OpenAICodexAccountIDFromToken(token string) (string, error) {
	return llmcodex.AccountIDFromToken(token)
}

type reasoningGenerator struct {
	llm.Generator
	effort llm.ReasoningEffort
}

func (g *reasoningGenerator) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
	if request.ReasoningEffort == "" {
		request.ReasoningEffort = g.effort
	}
	return g.Generator.Generate(ctx, request)
}

func (g *reasoningGenerator) Stream(ctx context.Context, request llm.Request) (llm.Stream, error) {
	if request.ReasoningEffort == "" {
		request.ReasoningEffort = g.effort
	}
	return g.Generator.Stream(ctx, request)
}

func newProviderLLM(ctx context.Context, config core.ProviderConfig, modelID core.ModelID, api llmmodels.API) (*GeneratorLLM, error) {
	provider := strings.TrimSpace(config.Name)
	model := strings.TrimSpace(string(modelID))
	if provider == "" || model == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "provider and model are required")
	}
	model = strings.TrimPrefix(model, provider+":")

	baseURL := normalizeProviderBaseURL(provider, api, configuredBaseURL(config))
	headers := providerHeaders(provider, config.APIKey, endpointHeaders(config))
	client := configuredHTTPClient(config, providerTimeout(api))
	collection, err := llmmodels.New(llmmodels.ProviderConfig{
		ID: provider, API: api, APIKey: providerAPIKey(provider, config.APIKey),
		BaseURL: baseURL, HTTPClient: client, Headers: headers,
	})
	if err != nil {
		return nil, err
	}

	info := defaultModelInfo(provider, model, api)
	if catalogModel, ok := llmmodels.BuiltinCatalog().Model(provider, model); ok {
		info = catalogModel.Info()
	}
	if modelConfig, ok := config.Models[model]; ok && len(modelConfig.Capabilities) != 0 {
		info.Capabilities = llmCapabilities(modelConfig.Capabilities)
	}
	generator, err := collection.Generator(info)
	if err != nil {
		return nil, err
	}
	endpoint := providerEndpoint(config, baseURL, api, headers)
	var embedder embeddingClient
	switch api {
	case llmmodels.GeminiGenerateContent:
		embedder, err = newGeminiEmbeddingClient(ctx, config.APIKey, baseURL, headerStringMap(headers), client)
	case llmmodels.OpenAIResponses:
		embeddingHeaders := headerStringMap(headers)
		embeddingHeaders["Authorization"] = "Bearer " + config.APIKey
		embedder = newOpenAIEmbeddingClient(baseURL, embeddingHeaders, client, "")
	}
	if err != nil {
		return nil, err
	}
	return adapt(generator, endpoint, client, embedder)
}

func newOpenAICompatible(provider string, modelID core.ModelID, config OpenAIConfig) (*OpenAILLM, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "provider is required")
	}
	apiBaseURL, err := chatCompletionsBaseURL(config.baseURL, config.path)
	if err != nil {
		return nil, err
	}
	if provider == "openai" && config.apiKey == "" && apiBaseURL == "https://api.openai.com/v1" {
		return nil, dspyerrors.New(dspyerrors.InvalidInput, "OpenAI API key is required")
	}
	client := config.httpClient
	if client == nil {
		client = newHTTPClient(config.timeout)
	}
	generator, err := llmopenai.New(llmopenai.Config{
		Provider: provider, Model: string(modelID), APIKey: config.apiKey,
		BaseURL: apiBaseURL, HTTPClient: client, Headers: stringMapHeader(config.headers),
		Capabilities: []llm.Capability{
			llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityJSON, llm.CapabilityVision,
		},
	})
	if err != nil {
		return nil, err
	}
	endpointHeaders := cloneStringMap(config.headers)
	if config.apiKey != "" {
		endpointHeaders["Authorization"] = "Bearer " + config.apiKey
	}
	endpoint := &core.EndpointConfig{
		BaseURL: config.baseURL, Path: config.path, Headers: endpointHeaders,
		TimeoutSec: int(config.timeout.Seconds()),
	}
	defaultEmbeddingModel := ""
	if provider == "ollama" || provider == "llamacpp" {
		defaultEmbeddingModel = string(modelID)
	}
	embedder := newOpenAIEmbeddingClient(config.baseURL, endpointHeaders, client, defaultEmbeddingModel)
	return adapt(generator, endpoint, client, embedder)
}

func openAIConfigFromProvider(config core.ProviderConfig, baseURL, path string, timeout time.Duration) OpenAIConfig {
	result := OpenAIConfig{
		baseURL: firstNonempty(configuredBaseURL(config), baseURL), path: path,
		apiKey: config.APIKey, timeout: timeout, headers: endpointHeaderMap(config),
	}
	if config.Endpoint != nil {
		if config.Endpoint.Path != "" {
			result.path = config.Endpoint.Path
		}
		if config.Endpoint.TimeoutSec > 0 {
			result.timeout = time.Duration(config.Endpoint.TimeoutSec) * time.Second
		}
	}
	if client, ok := config.Params["http_client"].(*http.Client); ok {
		result.httpClient = client
	}
	return result
}

func providerOpenAIOptions(config core.ProviderConfig) []OpenAIOption {
	options := []OpenAIOption{WithAPIKey(config.APIKey)}
	if config.Endpoint != nil {
		if config.Endpoint.Path != "" {
			options = append(options, WithOpenAIPath(config.Endpoint.Path))
		}
		if config.Endpoint.TimeoutSec > 0 {
			options = append(options, WithOpenAITimeout(time.Duration(config.Endpoint.TimeoutSec)*time.Second))
		}
		for name, value := range config.Endpoint.Headers {
			options = append(options, WithHeader(name, value))
		}
	}
	if client, ok := config.Params["http_client"].(*http.Client); ok {
		options = append(options, WithHTTPClient(client))
	}
	return options
}

func defaultModelInfo(provider, model string, api llmmodels.API) llm.ModelInfo {
	capabilities := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools}
	switch api {
	case llmmodels.GeminiGenerateContent:
		capabilities = append(capabilities, llm.CapabilityJSON, llm.CapabilityVision, llm.CapabilityAudio)
	case llmmodels.OpenAIChatCompletions:
		capabilities = append(capabilities, llm.CapabilityJSON, llm.CapabilityVision)
	case llmmodels.OpenAIResponses, llmmodels.OpenAICodexResponses:
		capabilities = append(capabilities, llm.CapabilityVision)
	}
	return llm.ModelInfo{Provider: provider, Model: model, Capabilities: capabilities}
}

func llmCapabilities(capabilities []string) []llm.Capability {
	converted := []llm.Capability{llm.CapabilityGeneration}
	for _, capability := range capabilities {
		switch core.Capability(capability) {
		case core.CapabilityStreaming:
			converted = appendLLMCapability(converted, llm.CapabilityStreaming)
		case core.CapabilityToolCalling:
			converted = appendLLMCapability(converted, llm.CapabilityTools)
		case core.CapabilityJSON:
			converted = appendLLMCapability(converted, llm.CapabilityJSON)
		case core.CapabilityMultimodal, core.CapabilityVision:
			converted = appendLLMCapability(converted, llm.CapabilityVision)
		case core.CapabilityAudio:
			converted = appendLLMCapability(converted, llm.CapabilityAudio)
		}
	}
	return converted
}

func appendLLMCapability(capabilities []llm.Capability, capability llm.Capability) []llm.Capability {
	if hasCapability(capabilities, capability) {
		return capabilities
	}
	return append(capabilities, capability)
}

func chatCompletionsBaseURL(baseURL, path string) (string, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return "", dspyerrors.New(dspyerrors.InvalidInput, "OpenAI-compatible base URL is required")
	}
	path = "/" + strings.TrimLeft(strings.TrimSpace(path), "/")
	const suffix = "/chat/completions"
	if !strings.HasSuffix(path, suffix) {
		return "", dspyerrors.New(dspyerrors.InvalidInput, "llm-go requires a Chat Completions endpoint path")
	}
	return baseURL + strings.TrimSuffix(path, suffix), nil
}

func configuredBaseURL(config core.ProviderConfig) string {
	if config.Endpoint != nil && strings.TrimSpace(config.Endpoint.BaseURL) != "" {
		return strings.TrimSpace(config.Endpoint.BaseURL)
	}
	return strings.TrimSpace(config.BaseURL)
}

func normalizeProviderBaseURL(provider string, api llmmodels.API, baseURL string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return ""
	}
	switch api {
	case llmmodels.AnthropicMessages:
		if baseURL == "https://api.anthropic.com" {
			return baseURL + "/v1"
		}
	case llmmodels.GeminiGenerateContent:
		baseURL = strings.TrimSuffix(strings.TrimSuffix(baseURL, "/v1beta"), "/v1")
	case llmmodels.OpenAIResponses, llmmodels.OpenAIChatCompletions:
		if provider == "openai" && baseURL == "https://api.openai.com" {
			return baseURL + "/v1"
		}
	}
	return baseURL
}

func configuredHTTPClient(config core.ProviderConfig, fallback time.Duration) *http.Client {
	if client, ok := config.Params["http_client"].(*http.Client); ok && client != nil {
		return client
	}
	timeout := fallback
	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		timeout = time.Duration(config.Endpoint.TimeoutSec) * time.Second
	}
	return newHTTPClient(timeout)
}

func newHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout, Transport: core.DefaultTransportConfig().ToTransport()}
}

func endpointHeaders(config core.ProviderConfig) http.Header {
	return stringMapHeader(endpointHeaderMap(config))
}

func endpointHeaderMap(config core.ProviderConfig) map[string]string {
	if config.Endpoint == nil {
		return make(map[string]string)
	}
	return cloneStringMap(config.Endpoint.Headers)
}

func providerHeaders(provider, apiKey string, headers http.Header) http.Header {
	if provider == "anthropic" && strings.HasPrefix(apiKey, "sk-ant-oat") {
		headers.Set("Authorization", "Bearer "+apiKey)
		headers.Set("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
		headers.Set("anthropic-dangerous-direct-browser-access", "true")
		headers.Set("user-agent", "claude-cli/2.1.2 (external, cli)")
		headers.Set("x-app", "cli")
	}
	return headers
}

func providerAPIKey(provider, apiKey string) string {
	if provider == "anthropic" && strings.HasPrefix(apiKey, "sk-ant-oat") {
		return ""
	}
	return apiKey
}

func providerEndpoint(config core.ProviderConfig, baseURL string, api llmmodels.API, headers http.Header) *core.EndpointConfig {
	timeout := int(providerTimeout(api).Seconds())
	if config.Endpoint != nil && config.Endpoint.TimeoutSec > 0 {
		timeout = config.Endpoint.TimeoutSec
	}
	path := map[llmmodels.API]string{
		llmmodels.AnthropicMessages:     "/messages",
		llmmodels.GeminiGenerateContent: "/models/:generateContent",
		llmmodels.OpenAIResponses:       "/responses",
		llmmodels.OpenAIChatCompletions: "/chat/completions",
		llmmodels.OpenAICodexResponses:  "/codex/responses",
	}[api]
	if config.Endpoint != nil && config.Endpoint.Path != "" {
		path = config.Endpoint.Path
	}
	return &core.EndpointConfig{
		BaseURL: baseURL, Path: path, Headers: headerStringMap(headers), TimeoutSec: timeout,
	}
}

func providerTimeout(api llmmodels.API) time.Duration {
	if api == llmmodels.GeminiGenerateContent {
		return 10 * time.Minute
	}
	return 60 * time.Second
}

func stringMapHeader(values map[string]string) http.Header {
	header := make(http.Header, len(values))
	for name, value := range values {
		header.Set(name, value)
	}
	return header
}

func headerStringMap(header http.Header) map[string]string {
	values := make(map[string]string, len(header))
	for name := range header {
		values[name] = header.Get(name)
	}
	return values
}

func headerValue(headers map[string]string, name string) string {
	for key, value := range headers {
		if strings.EqualFold(key, name) {
			return value
		}
	}
	return ""
}

func cloneStringMap(values map[string]string) map[string]string {
	clone := make(map[string]string, len(values))
	for key, value := range values {
		clone[key] = value
	}
	return clone
}

func firstNonempty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

var _ llm.Generator = (*reasoningGenerator)(nil)
