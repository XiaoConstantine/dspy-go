package llms

import (
	"fmt"
	"strings"
	"sync"

	"github.com/XiaoConstantine/anthropic-go/anthropic"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type DefaultLLMFactory struct{}

var (
	defaultFactory     *DefaultLLMFactory
	defaultFactoryOnce sync.Once
)

func ensureFactory() {
	defaultFactoryOnce.Do(func() {
		defaultFactory = &DefaultLLMFactory{}
		core.DefaultFactory = defaultFactory
	})
}

// NewLLM creates a new LLM instance based on the provided model ID.
func NewLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {

	ensureFactory()
	var llm core.LLM
	var err error
	switch {
	case modelID == core.ModelAnthropicHaiku || modelID == core.ModelAnthropicSonnet || modelID == core.ModelAnthropicOpus:
		llm, err = NewAnthropicLLM(apiKey, anthropic.ModelID(modelID))
	case modelID == core.ModelGoogleGeminiFlash || modelID == core.ModelGoogleGeminiPro || modelID == core.ModelGoogleGeminiFlashThinking:
		llm, err = NewGeminiLLM(apiKey, modelID)
	case strings.HasPrefix(string(modelID), "ollama:"):
		parts := strings.SplitN(string(modelID), ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid Ollama model ID format. Use 'ollama:<model_name>'")
		}
		llm, err = NewOllamaLLM("http://localhost:11434", parts[1])
	case strings.HasPrefix(string(modelID), "llamacpp:"):
		return NewLlamacppLLM("http://localhost:8080")
	default:
		return nil, fmt.Errorf("unsupported model ID: %s", modelID)
	}
	if err != nil {
		return nil, err
	}
	return core.Chain(llm,
		func(l core.LLM) core.LLM { return core.NewModelContextDecorator(l) },
	), nil
}

// Implement the LLMFactory interface.
func (f *DefaultLLMFactory) CreateLLM(apiKey string, modelID core.ModelID) (core.LLM, error) {

	return NewLLM(apiKey, modelID)
}

func init() {
	ensureFactory()
}

func EnsureFactory() {
	ensureFactory()
}
