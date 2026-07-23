package modules

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	"github.com/stretchr/testify/assert"
)

func TestReActNativeConfigurationRetainsLegacyModuleCompatibility(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("task")}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)
	module := NewReAct(signature, tools.NewInMemoryToolRegistry(), 3)
	module.WithNativeFunctionCalling()
	assert.NotEmpty(t, module.Predict.GetInterceptors())
	module.WithNativeFunctionCallingConfig(interceptors.FunctionCallingConfig{IncludeFinishTool: true})
	assert.NotEmpty(t, module.Predict.GetInterceptors())
}
