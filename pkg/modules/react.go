// modules/react.go

package modules

import (
	"context"
	"errors"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type Tool interface {
	CanHandle(action string) bool
	Execute(ctx context.Context, action string) (string, error)
}

type ReAct struct {
	core.BaseModule
	Predict  *Predict
	Tools    []Tool
	MaxIters int
}

func NewReAct(signature core.Signature, tools []Tool, maxIters int) *ReAct {
	modifiedSignature := appendReActFields(signature)
	predict := NewPredict(modifiedSignature)

	return &ReAct{
		BaseModule: *core.NewModule(modifiedSignature),
		Predict:    predict,
		Tools:      tools,
		MaxIters:   maxIters,
	}
}

func (r *ReAct) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.Predict.SetLLM(llm)
}

func (r *ReAct) Process(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	for i := 0; i < r.MaxIters; i++ {
		prediction, err := r.Predict.Process(ctx, inputs)
		if err != nil {
			return nil, err
		}

		action, ok := prediction["action"].(string)
		if !ok {
			return nil, errors.New("invalid action in prediction")
		}

		if action == "Finish" {
			return prediction, nil
		}

		for _, tool := range r.Tools {
			if tool.CanHandle(action) {
				observation, err := tool.Execute(ctx, action)
				if err != nil {
					return nil, err
				}
				inputs["observation"] = observation
				break
			}
		}
	}

	return nil, errors.New("max iterations reached")
}

func (r *ReAct) Clone() core.Module {
	return &ReAct{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		Predict:    r.Predict.Clone().(*Predict),
		Tools:      r.Tools, // Note: This is a shallow copy of the tools
		MaxIters:   r.MaxIters,
	}
}

func appendReActFields(signature core.Signature) core.Signature {
	newSignature := signature
	newFields := []core.OutputField{
		{Field: core.Field{Name: "thought", Prefix: "Thought:"}},
		{Field: core.Field{Name: "action", Prefix: "Action:"}},
		{Field: core.Field{Name: "observation", Prefix: "Observation:"}},
	}
	newSignature.Outputs = append(newFields, newSignature.Outputs...)
	return newSignature
}
