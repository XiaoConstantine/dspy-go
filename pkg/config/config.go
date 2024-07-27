package config

import (
	"fmt"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llm"
)

type Config struct {
    DefaultLLM core.LLM
    TeacherLLM core.LLM
}

var GlobalConfig = &Config{}

// ConfigureDefaultLLM sets up the default LLM to be used across the package.
func ConfigureDefaultLLM(apiKey string, modelID core.ModelID) error {
    llmInstance, err := llm.NewLLM(apiKey, modelID)
    if err != nil {
        return fmt.Errorf("failed to configure default LLM: %w", err)
    }
    GlobalConfig.DefaultLLM = llmInstance
    return nil
}

// ConfigureTeacherLLM sets up the teacher LLM.
func ConfigureTeacherLLM(apiKey string, modelID core.ModelID) error {
    llmInstance, err := llm.NewLLM(apiKey, modelID)
    if err != nil {
        return fmt.Errorf("failed to configure teacher LLM: %w", err)
    }
    GlobalConfig.TeacherLLM = llmInstance
    return nil
}

// GetDefaultLLM returns the default LLM.
func GetDefaultLLM() core.LLM {
    return GlobalConfig.DefaultLLM
}

// GetTeacherLLM returns the teacher LLM.
func GetTeacherLLM() core.LLM {
    return GlobalConfig.TeacherLLM
}
