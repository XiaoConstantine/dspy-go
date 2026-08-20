package core

import "strings"

var anthropicModelAliases = map[ModelID]ModelID{
	"haiku-4.5":         ModelAnthropicClaude45Haiku,
	"haiku-4-5":         ModelAnthropicClaude45Haiku,
	"claude-haiku-4.5":  ModelAnthropicClaude45Haiku,
	"sonnet-4.5":        ModelAnthropicClaude45Sonnet,
	"sonnet-4-5":        ModelAnthropicClaude45Sonnet,
	"claude-sonnet-4.5": ModelAnthropicClaude45Sonnet,
	"opus-4.5":          ModelAnthropicClaude45Opus,
	"opus-4-5":          ModelAnthropicClaude45Opus,
	"claude-opus-4.5":   ModelAnthropicClaude45Opus,
	"sonnet-4.6":        ModelAnthropicClaude46Sonnet,
	"sonnet-4-6":        ModelAnthropicClaude46Sonnet,
	"claude-sonnet-4.6": ModelAnthropicClaude46Sonnet,
	"opus-4.6":          ModelAnthropicClaude46Opus,
	"opus-4-6":          ModelAnthropicClaude46Opus,
	"claude-opus-4.6":   ModelAnthropicClaude46Opus,
	"opus-4.7":          ModelAnthropicClaude47Opus,
	"opus-4-7":          ModelAnthropicClaude47Opus,
	"claude-opus-4.7":   ModelAnthropicClaude47Opus,
	"opus-4.8":          ModelAnthropicClaude48Opus,
	"opus-4-8":          ModelAnthropicClaude48Opus,
	"claude-opus-4.8":   ModelAnthropicClaude48Opus,
	"sonnet-5":          ModelAnthropicClaude5Sonnet,
	"fable-5":           ModelAnthropicClaude5Fable,
	"mythos-5":          ModelAnthropicClaude5Mythos,
	"opus-5":            ModelAnthropicClaude5Opus,
}

// ResolveAnthropicModelID resolves a supported Anthropic model ID or convenience
// alias to its canonical API model ID.
func ResolveAnthropicModelID(modelID ModelID) (ModelID, bool) {
	if modelInList(modelID, ProviderModels["anthropic"]) {
		return modelID, true
	}
	resolved, ok := anthropicModelAliases[modelID]
	return resolved, ok
}

// Clone returns a deep copy of the provider configuration.
func (config ProviderConfig) Clone() ProviderConfig {
	cloned := config
	if config.Models != nil {
		cloned.Models = make(map[string]ModelConfig, len(config.Models))
		for key, model := range config.Models {
			cloned.Models[key] = model.Clone()
		}
	}
	if config.Params != nil {
		cloned.Params = make(map[string]any, len(config.Params))
		for key, value := range config.Params {
			cloned.Params[key] = value
		}
	}
	if config.Endpoint != nil {
		endpoint := *config.Endpoint
		cloned.Endpoint = &endpoint
	}
	return cloned
}

// Clone returns a deep copy of the model configuration.
func (config ModelConfig) Clone() ModelConfig {
	cloned := config
	if config.Capabilities != nil {
		cloned.Capabilities = append([]string(nil), config.Capabilities...)
	}
	if config.Params != nil {
		cloned.Params = make(map[string]any, len(config.Params))
		for key, value := range config.Params {
			cloned.Params[key] = value
		}
	}
	if config.DefaultOptions != nil {
		options := *config.DefaultOptions
		if config.DefaultOptions.Stop != nil {
			options.Stop = append([]string(nil), config.DefaultOptions.Stop...)
		}
		cloned.DefaultOptions = &options
	}
	return cloned
}

// InferProviderFromModelID infers the provider name from a model identifier for
// backward-compatible model resolution.
func InferProviderFromModelID(modelID ModelID) string {
	modelStr := string(modelID)

	if _, ok := ResolveAnthropicModelID(modelID); ok {
		return "anthropic"
	}
	if modelInList(modelID, ProviderModels["google"]) {
		return "google"
	}
	if modelInList(modelID, ProviderModels["openai"]) {
		return "openai"
	}
	if modelInList(modelID, ProviderModels["ollama"]) {
		return "ollama"
	}

	if strings.HasPrefix(modelStr, "gpt-") {
		return "openai"
	}
	if strings.HasPrefix(modelStr, "o1") || strings.HasPrefix(modelStr, "o3") {
		return "openai"
	}
	if strings.HasPrefix(modelStr, "ollama:") {
		return "ollama"
	}
	if strings.HasPrefix(modelStr, "llamacpp:") {
		return "llamacpp"
	}

	return ""
}

func modelInList(modelID ModelID, models []ModelID) bool {
	for _, candidate := range models {
		if candidate == modelID {
			return true
		}
	}
	return false
}
