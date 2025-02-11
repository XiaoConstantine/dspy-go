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

// WithDefaultOptions sets default options by configuring the underlying Predict module.
func (r *ReAct) WithDefaultOptions(opts ...core.Option) *ReAct {
	// Simply delegate to the Predict module's WithDefaultOptions
	r.Predict.WithDefaultOptions(opts...)
	return r
}

func (r *ReAct) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.Predict.SetLLM(llm)
}

func (r *ReAct) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	ctx, span := core.StartSpan(ctx, "ReAct")
	defer core.EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)

	for i := 0; i < r.MaxIters; i++ {
		prediction, err := r.Predict.Process(ctx, inputs, opts...)
		if err != nil {
			return nil, err
		}

		action, ok := prediction["action"].(string)
		if !ok {
			err := errors.New("invalid action in prediction")
			span.WithError(err)
			return nil, err
		}

		if action == "Finish" {
			span.WithAnnotation("prediction", prediction)
			return prediction, nil
		}

		for _, tool := range r.Tools {
			if tool.CanHandle(action) {
				observation, err := tool.Execute(ctx, action)
				if err != nil {
					span.WithError(err)

					return nil, err
				}
				inputs["observation"] = observation
				break
			}
		}
	}
	err := errors.New("max iterations reached")
	span.WithError(err)

	return nil, err
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
		{Field: core.NewField("thought")},
		{Field: core.NewField("action")},
		{Field: core.NewField("observation")},
	}
	newSignature.Outputs = append(newFields, newSignature.Outputs...)
	return newSignature
}
