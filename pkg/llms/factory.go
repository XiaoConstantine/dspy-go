package llm

import (
	"fmt"
	"strings"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// NewLLM creates a new LLM instance based on the provided model ID.
func NewLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	switch {
	case modelID == core.ModelAnthropicHaiku || modelID == core.ModelAnthropicSonnet || modelID == core.ModelAnthropicOpus:
		return NewAnthropicLLM(apiKey, anthropic.ModelID(modelID))
	case strings.HasPrefix(string(modelID), "ollama:"):
		parts := strings.SplitN(string(modelID), ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid Ollama model ID format. Use 'ollama:<model_name>'")
		}
		return NewOllamaLLM("http://localhost:11434", parts[1])
	default:
		return nil, fmt.Errorf("unsupported model ID: %s", modelID)
	}
}
