package react

import (
	"testing"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/stretchr/testify/assert"
)

func TestPlannerCompatibleInputAliasesSignaturePrimaryForReWOO(t *testing.T) {
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("query")}},
		[]core.OutputField{{Field: core.NewField("result")}},
	)
	input := map[string]any{"query": "investigate"}
	adapted := plannerCompatibleInput(input, ModeReWOO, signature)
	assert.Equal(t, "investigate", adapted["task"])
	assert.Equal(t, "investigate", adapted["query"])
	assert.NotContains(t, input, "task")

	hybrid := plannerCompatibleInput(input, ModeHybrid, signature)
	assert.Equal(t, "investigate", hybrid["task"])

	react := plannerCompatibleInput(input, ModeReAct, signature)
	assert.NotContains(t, react, "task")
}
