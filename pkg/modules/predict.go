package modules

import (
	"context"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
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
	fmt.Printf("Predict Process inputs: %+v\n", inputs)

	tm := core.GetTraceManager(ctx)
	trace := tm.StartTrace("Predict", "Predict")
	defer tm.EndTrace()

	trace.SetInputs(inputs)

	if err := p.ValidateInputs(inputs); err != nil {
		trace.SetError(err)
		return nil, err
	}

	signature := p.GetSignature()
	prompt := formatPrompt(signature, p.Demos, inputs)
	fmt.Printf("Generated prompt: %s\n", prompt)

	completion, err := p.LLM.Generate(ctx, prompt)
	if err != nil {
		trace.SetError(err)

		return nil, err
	}
	fmt.Printf("LLM completion: %s\n", completion)

	outputs := parseCompletion(completion, signature)
	formattedOutputs := p.FormatOutputs(outputs)
	fmt.Printf("Parsed outputs: %+v\n", outputs)

	trace.SetOutputs(formattedOutputs)

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

//	func parseCompletion(completion string, signature core.Signature) map[string]any {
//		outputs := make(map[string]any)
//		lines := strings.Split(strings.TrimSpace(completion), "\n")
//
//		var currentField string
//		var currentContent []string
//
//		// Helper function to store accumulated content
//		storeContent := func() {
//			if currentField != "" && len(currentContent) > 0 {
//				outputs[currentField] = strings.Join(currentContent, "\n")
//			}
//		}
//
//		for _, line := range lines {
//			line = strings.TrimSpace(line)
//			if line == "" {
//				continue
//			}
//
//			// Check for field headers
//			foundNewField := false
//			for _, field := range signature.Outputs {
//				prefix := field.Name + ":"
//				if strings.HasPrefix(line, prefix) {
//					// Store any existing content before switching fields
//					storeContent()
//
//					// Start new field
//					currentField = field.Name
//					currentContent = nil
//					foundNewField = true
//					break
//				}
//			}
//
//			// If this wasn't a field header and we have a current field
//			if !foundNewField && currentField != "" {
//				// Handle both bullet points and regular lines
//				content := line
//				if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "•") {
//					content = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "-"), "•"))
//				}
//				if content != "" {
//					currentContent = append(currentContent, content)
//				}
//			}
//		}
//
//		// Don't forget to store the last field's content
//		storeContent()
//
//		return outputs
//	}
func parseCompletion(completion string, signature core.Signature) map[string]any {
	outputs := make(map[string]any)
	lines := strings.Split(strings.TrimSpace(completion), "\n")
	// Track which field we're currently collecting content for

	var currentField *core.OutputField
	var currentValue strings.Builder

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		for _, field := range signature.Outputs {
			if field.Prefix != "" && strings.HasPrefix(line, field.Prefix) {
				currentField = &field
				continue
			}
		}
		if currentField != nil {
			// Skip the line with just the prefix
			if strings.HasPrefix(line, currentField.Prefix) {
				continue
			}
			currentValue.WriteString(line)
			currentValue.WriteString("\n")
		}
	}
	if currentField != nil && currentValue.Len() > 0 {
		outputs[currentField.Name] = strings.TrimSpace(currentValue.String())
	}

	return outputs
}

// func parseCompletion(completion string, signature core.Signature) map[string]interface{} {
// 	// Initialize empty map to store results
// 	outputs := make(map[string]interface{})
//
// 	// Split completion into lines, trimming whitespace
// 	lines := strings.Split(strings.TrimSpace(completion), "\n")
//
// 	// Iterate through each line looking for key:value pairs
// 	for _, line := range lines {
// 		// Split line at first colon
// 		parts := strings.SplitN(line, ":", 2)
// 		if len(parts) == 2 {
// 			key := strings.TrimSpace(parts[0])
// 			value := strings.TrimSpace(parts[1])
//
// 			// Check if key matches any output field names
// 			for _, field := range signature.Outputs {
// 				if field.Name == key {
// 					outputs[key] = value
// 					break
// 				}
// 			}
// 		}
// 	}
//
// 	return outputs
// }

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
