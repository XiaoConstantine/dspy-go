package modules

import (
	"context"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type ChainOfThought struct {
	Predict *Predict
}

func NewChainOfThought(signature core.Signature) *ChainOfThought {
	modifiedSignature := appendRationaleField(signature)
	return &ChainOfThought{
		Predict: NewPredict(modifiedSignature),
	}
}

func (c *ChainOfThought) Process(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	tm := core.GetTraceManager(ctx)
	trace := tm.StartTrace("ChainOfThought", "ChainOfThought")
	defer tm.EndTrace()

	trace.SetInputs(inputs)
	outputs, err := c.Predict.Process(ctx, inputs)
	if err != nil {
		trace.SetError(err)
		return nil, err
	}
	trace.SetOutputs(outputs)

	return outputs, nil
}

func (c *ChainOfThought) GetSignature() core.Signature {
	return c.Predict.GetSignature()
}

func (c *ChainOfThought) SetLLM(llm core.LLM) {
	c.Predict.SetLLM(llm)
}

func (c *ChainOfThought) Clone() core.Module {
	return &ChainOfThought{
		Predict: c.Predict.Clone().(*Predict),
	}
}

func appendRationaleField(signature core.Signature) core.Signature {
	newSignature := signature
	newSignature.Outputs = append([]core.OutputField{{Field: core.Field{Name: "rationale", Prefix: "Reasoning: Let's think step by step."}}}, newSignature.Outputs...)
	return newSignature
}
