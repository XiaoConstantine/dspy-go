package core

import "strings"

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

	if modelInList(modelID, ProviderModels["anthropic"]) {
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

	if strings.HasPrefix(modelStr, "claude-") {
		return "anthropic"
	}
	if strings.HasPrefix(modelStr, "opus-") {
		return "anthropic"
	}
	if strings.HasPrefix(modelStr, "sonnet-") {
		return "anthropic"
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
