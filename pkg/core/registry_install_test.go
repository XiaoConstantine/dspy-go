package core_test

import (
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/registry"
)

func init() {
	core.SetRegistryConstructor(func() core.LLMRegistry {
		return registry.NewLLMRegistry()
	})
}
