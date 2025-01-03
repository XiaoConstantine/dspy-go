package modules

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

type Predict struct {
	core.BaseModule
	Demos []core.Example
	LLM   core.LLM
}

// Ensure Predict implements core.Module.
var _ core.Module = (*Predict)(nil)

func NewPredict(signature core.Signature) *Predict {
	return &Predict{
		BaseModule: *core.NewModule(signature),
		Demos:      []core.Example{},
		LLM:        config.GetDefaultLLM(),
	}
}

func (p *Predict) Process(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()

	ctx, span := core.StartSpan(ctx, "Predict")
	defer core.EndSpan(ctx)
	logger.Debug(ctx, "Processing inputs: %v", inputs)
	span.WithAnnotation("inputs", inputs)

	if err := p.ValidateInputs(inputs); err != nil {
		span.WithError(err)
		return nil, err
	}

	signature := p.GetSignature()
	prompt := formatPrompt(signature, p.Demos, inputs)

	logger.Debug(ctx, "Generated prompt with prompt: %v", prompt)

	resp, err := p.LLM.Generate(ctx, prompt)
	if err != nil {
		span.WithError(err)

		return nil, err
	}

	logger.Debug(ctx, "LLM Completion: %v", resp.Content)

	if resp.Usage != nil {
		if state := core.GetExecutionState(ctx); state != nil {
			state.WithTokenUsage(&core.TokenUsage{
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
			})
		}
		logger.Debug(ctx, "LLM Completion total token usage: %d, %d, %d", resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}

	outputs := parseCompletion(resp.Content, signature)
	formattedOutputs := p.FormatOutputs(outputs)
	logger.Debug(ctx, "Formatted LLM Completion: %v", outputs)

	span.WithAnnotation("outputs", formattedOutputs)

	return formattedOutputs, nil
}

func (p *Predict) Clone() core.Module {
	return &Predict{
		BaseModule: *p.BaseModule.Clone().(*core.BaseModule),
		Demos:      append([]core.Example{}, p.Demos...),
		LLM:        p.LLM,
	}
}

func (p *Predict) GetDemos() []core.Example {
	return p.Demos
}

func formatPrompt(signature core.Signature, demos []core.Example, inputs map[string]any) string {
	var sb strings.Builder

	// Write the instruction
	sb.WriteString(fmt.Sprintf("Given the fields '%s', produce the fields '%s'.\n\n",
		joinFieldNames(inputFieldsToFields(signature.Inputs)),
		joinFieldNames(outputFieldsToFields(signature.Outputs)),
	))

	for _, field := range signature.Outputs {
		if field.Prefix != "" {
			sb.WriteString(fmt.Sprintf("The %s field should start with '%s' followed by the content on new lines.\n",
				field.Name, field.Prefix))
		}
	}
	sb.WriteString("\n")

	// Add the instruction if present
	if signature.Instruction != "" {
		sb.WriteString(signature.Instruction + "\n\n")
	}
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

// func parseCompletion(completion string, signature core.Signature) map[string]any {
// 	outputs := make(map[string]any)
// 	lines := strings.Split(strings.TrimSpace(completion), "\n")
//
// 	var currentField *core.OutputField
// 	var currentValue strings.Builder
//
// 	for _, line := range lines {
// 		line = strings.TrimSpace(line)
// 		if line == "" {
// 			continue
// 		}
//
// 		for _, field := range signature.Outputs {
// 			if field.Prefix != "" && strings.HasPrefix(line, field.Prefix) {
// 				currentField = &field
// 				continue
// 			}
// 		}
// 		if currentField != nil {
// 			// Skip the line with just the prefix
// 			if strings.HasPrefix(line, currentField.Prefix) {
// 				continue
// 			}
// 			currentValue.WriteString(line)
// 			currentValue.WriteString("\n")
// 		}
// 	}
// 	if currentField != nil && currentValue.Len() > 0 {
// 		outputs[currentField.Name] = strings.TrimSpace(currentValue.String())
// 	}
//
// 	return outputs
// }

func parseCompletion(completion string, signature core.Signature) map[string]any {
	outputs := make(map[string]any)
	lines := strings.Split(strings.TrimSpace(completion), "\n")

	var currentField *core.OutputField
	var contentLines []string

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Try to match a field prefix
		for _, field := range signature.Outputs {
			prefix := strings.TrimSpace(field.Prefix)
			if prefix != "" && strings.HasPrefix(strings.ToLower(line), strings.ToLower(prefix)) {
				// If we were collecting content for a previous field, save it
				if currentField != nil && len(contentLines) > 0 {
					outputs[currentField.Name] = strings.TrimSpace(strings.Join(contentLines, "\n"))
				}

				// Start collecting content for this new field
				currentField = &field
				contentLines = nil
				continue
			}
		}

		// If we have a current field and this isn't a prefix line, collect the content
		if currentField != nil {
			// Don't add the prefix line itself to the content
			if !strings.HasPrefix(strings.ToLower(line), strings.ToLower(currentField.Prefix)) {
				contentLines = append(contentLines, line)
			}
		}

		// If this is the last line, save any remaining content
		if i == len(lines)-1 && currentField != nil && len(contentLines) > 0 {
			outputs[currentField.Name] = strings.TrimSpace(strings.Join(contentLines, "\n"))
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

func (p *Predict) FormatOutputs(outputs map[string]interface{}) map[string]interface{} {
	formattedOutputs := make(map[string]interface{})
	for _, field := range p.GetSignature().Outputs {
		if value, ok := outputs[field.Name]; ok {
			formattedOutputs[field.Name] = value
		}
	}
	return formattedOutputs
}

func (p *Predict) GetSignature() core.Signature {
	return p.BaseModule.GetSignature()
}

func (p *Predict) ValidateInputs(inputs map[string]interface{}) error {
	signature := p.GetSignature()
	for _, field := range signature.Inputs {
		if _, ok := inputs[field.Name]; !ok {
			return fmt.Errorf("missing required input: %s", field.Name)
		}
	}
	return nil
}

func (p *Predict) SetDemos(demos []core.Example) {
	p.Demos = demos
}

func (p *Predict) SetLLM(llm core.LLM) {
	p.LLM = llm
}
