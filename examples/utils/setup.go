package utils

import (
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func SetupLLM(apiKey string, modelID core.ModelID) {
	err := core.ConfigureDefaultLLM(apiKey, modelID)
	if err != nil {
		log.Fatalf("Failed to configure default LLM: %v", err)
	}
}
