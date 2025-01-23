package core

// LLMFactory defines a simple interface for creating LLM instances.
// This maintains compatibility with existing code while allowing for configuration.
type LLMFactory interface {
	// CreateLLM creates a new LLM instance. It uses the global configuration
	// from core.GlobalConfig for client settings.
	CreateLLM(apiKey string, modelID ModelID) (LLM, error)
}

// DefaultFactory is the global factory instance used by the configuration system.
var DefaultFactory LLMFactory
