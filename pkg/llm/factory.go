package llm

import (
	"fmt"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// NewLLM creates a new LLM instance based on the provided model ID.
func NewLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {
	switch modelID {
	case core.ModelAnthropicHaiku, core.ModelAnthropicSonnet, core.ModelAnthropicOpus:
		anthropicModel, ok := anthropic.GetModelID(string(modelID))
		if !ok {
			return nil, fmt.Errorf("invalid Anthropic model ID: %s", modelID)
		}
		return NewAnthropicLLM(apiKey, anthropicModel)
	default:
		return nil, fmt.Errorf("unsupported model ID: %s", modelID)
	}
}
