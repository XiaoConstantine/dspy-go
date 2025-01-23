package core

import (
	"fmt"
)

type Config struct {
	DefaultLLM       LLM
	TeacherLLM       LLM
	ConcurrencyLevel int
}

var GlobalConfig = &Config{
	// default concurrency 1
	ConcurrencyLevel: 1,
}

// ConfigureDefaultLLM sets up the default LLM to be used across the package.
func ConfigureDefaultLLM(apiKey string, modelID ModelID) error {
	llmInstance, err := DefaultFactory.CreateLLM(apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure default LLM: %w", err)
	}
	GlobalConfig.DefaultLLM = llmInstance
	return nil
}

// ConfigureTeacherLLM sets up the teacher LLM.
func ConfigureTeacherLLM(apiKey string, modelID ModelID) error {
	llmInstance, err := DefaultFactory.CreateLLM(apiKey, modelID)
	if err != nil {
		return fmt.Errorf("failed to configure teacher LLM: %w", err)
	}
	GlobalConfig.TeacherLLM = llmInstance
	return nil
}

// GetDefaultLLM returns the default LLM.
func GetDefaultLLM() LLM {
	return GlobalConfig.DefaultLLM
}

// GetTeacherLLM returns the teacher LLM.
func GetTeacherLLM() LLM {
	return GlobalConfig.TeacherLLM
}

func SetConcurrencyOptions(level int) {
	if level > 0 {
		GlobalConfig.ConcurrencyLevel = level
	} else {
		GlobalConfig.ConcurrencyLevel = 1 // Reset to default value for invalid inputs
	}
}
