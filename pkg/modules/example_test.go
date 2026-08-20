package modules_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

type uppercaseModule struct {
	core.BaseModule
}

func newUppercaseModule() *uppercaseModule {
	signature := core.Signature{
		Inputs:  []core.InputField{{Name: "text"}},
		Outputs: []core.OutputField{{Name: "text"}},
	}
	return &uppercaseModule{BaseModule: *core.NewModule(signature)}
}

func (m *uppercaseModule) Process(
	ctx context.Context,
	inputs map[string]any,
	opts ...core.Option,
) (map[string]any, error) {
	text, ok := inputs["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text must be a string")
	}
	return map[string]any{"text": strings.ToUpper(text)}, nil
}

func (m *uppercaseModule) Clone() core.Module {
	return &uppercaseModule{BaseModule: *m.BaseModule.Clone().(*core.BaseModule)}
}

func ExampleNewParallel() {
	parallel := modules.NewParallel(newUppercaseModule(), modules.WithMaxWorkers(2))
	result, err := parallel.Process(context.Background(), map[string]any{
		"batch_inputs": []map[string]any{
			{"text": "alpha"},
			{"text": "beta"},
		},
	})
	if err != nil {
		fmt.Println("parallel error:", err)
		return
	}

	for _, item := range result["results"].([]map[string]any) {
		fmt.Println(item["text"])
	}

	// Output:
	// ALPHA
	// BETA
}
