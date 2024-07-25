// modules/predict.go

package modules

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

type Predict struct {
	core.BaseModule
	Demos []core.Example
}

func NewPredict(signature core.Signature) *Predict {
	return &Predict{
		BaseModule: *core.NewModule(signature),
		Demos:      []core.Example{},
	}
}

func (p *Predict) Process(ctx context.Context, inputs map[string]any) (map[string]any, error) {
	if err := p.ValidateInputs(inputs); err != nil {
		return nil, err
	}

	prompt := formatPrompt(p.Signature, p.Demos, inputs)
	completion, err := p.LLM.Generate(ctx, prompt)
	if err != nil {
		return nil, err
	}

	outputs := parseCompletion(completion, p.Signature)
	return p.FormatOutputs(outputs), nil
}

func (p *Predict) Clone() core.Module {
	return &Predict{
		BaseModule: *p.BaseModule.Clone().(*core.BaseModule),
		Demos:      append([]core.Example{}, p.Demos...),
	}
}

// Helper functions (to be implemented).
func formatPrompt(signature core.Signature, demos []core.Example, inputs map[string]any) string {
	var sb strings.Builder

	// Write the instruction
	sb.WriteString(fmt.Sprintf("Given the fields '%s', produce the fields '%s'.\n\n",
		joinFieldNames(inputFieldsToFields(signature.Inputs)),
		joinFieldNames(outputFieldsToFields(signature.Outputs)),
	))

	// Write the demonstrations
	for _, demo := range demos {
		sb.WriteString("---\n\n")
		for _, field := range signature.Inputs {
			sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, demo.Inputs[field.Name]))
		}
		for _, field := range signature.Outputs {
			sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, demo.Outputs[field.Name]))
		}
		sb.WriteString("\n")
	}

	// Write the current input
	sb.WriteString("---\n\n")
	for _, field := range signature.Inputs {
		sb.WriteString(fmt.Sprintf("%s: %v\n", field.Name, inputs[field.Name]))
	}

	return sb.String()
}

func parseCompletion(completion string, signature core.Signature) map[string]any {
	outputs := make(map[string]any)
	lines := strings.Split(strings.TrimSpace(completion), "\n")

	for _, line := range lines {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			for _, field := range signature.Outputs {
				if field.Name == key {
					outputs[key] = value
					break
				}
			}
		}
	}

	return outputs
}

func joinFieldNames(fields []core.Field) string {
	names := make([]string, len(fields))
	for i, field := range fields {
		names[i] = field.Name
	}
	return strings.Join(names, ", ")
}

func inputFieldsToFields(inputs []core.InputField) []core.Field {
	fields := make([]core.Field, len(inputs))
	for i, input := range inputs {
		fields[i] = input.Field
	}
	return fields
}

func outputFieldsToFields(outputs []core.OutputField) []core.Field {
	fields := make([]core.Field, len(outputs))
	for i, output := range outputs {
		fields[i] = output.Field
	}
	return fields
}
