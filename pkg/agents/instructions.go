package agents

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

// InstructionProvider is the minimal DSPy contract accepted by
// WithInstructions. Signatures, modules, and programs satisfy it structurally.
type InstructionProvider interface {
	GetSignature() core.Signature
}

// InstructionArtifactProvider explicitly certifies that a module's optimized
// instruction state can be represented by a signature and demonstrations.
type InstructionArtifactProvider interface {
	InstructionArtifacts() (core.Signature, []core.Example, error)
}

// InstructionSource supplies an ownership-safe snapshot of canonical initial
// messages. Implementations must not execute model or module inference.
type InstructionSource interface {
	InstructionMessages() ([]Message, error)
}

type staticInstructionSource struct {
	messages []Message
}

func (s staticInstructionSource) InstructionMessages() ([]Message, error) {
	return CloneMessages(s.messages), nil
}

func staticInstructions(text string) InstructionSource {
	text = strings.TrimSpace(text)
	if text == "" {
		return staticInstructionSource{}
	}
	return staticInstructionSource{messages: []Message{NewTextMessage(RoleSystem, text)}}
}

type dspyInstructionSource struct {
	signature core.Signature
	demos     []core.Example
}

func (s dspyInstructionSource) InstructionMessages() ([]Message, error) {
	messages := make([]Message, 0, 1+len(s.demos)*2)
	if system := renderSignatureInstructions(s.signature); system != "" {
		messages = append(messages, NewTextMessage(RoleSystem, system))
	}
	for _, demo := range s.demos {
		input, err := renderExampleMessage(RoleUser, s.signature.Inputs, demo.Inputs)
		if err != nil {
			return nil, fmt.Errorf("render demonstration input: %w", err)
		}
		output, err := renderExampleMessage(RoleAssistant, s.signature.Outputs, demo.Outputs)
		if err != nil {
			return nil, fmt.Errorf("render demonstration output: %w", err)
		}
		messages = append(messages, input, output)
	}
	return CloneMessages(messages), nil
}

func instructionSourceFromProvider(provider InstructionProvider) (InstructionSource, error) {
	if isNilInstructionValue(provider) {
		return nil, fmt.Errorf("instruction provider is required")
	}

	switch value := provider.(type) {
	case core.Signature:
		return dspyInstructionSource{signature: cloneSignature(value)}, nil
	case *core.Signature:
		return dspyInstructionSource{signature: cloneSignature(*value)}, nil
	case core.Program:
		return instructionSourceFromProgram(value)
	case *core.Program:
		return instructionSourceFromProgram(*value)
	case core.Module:
		artifacts, ok := value.(InstructionArtifactProvider)
		if !ok {
			return nil, fmt.Errorf("module %T cannot be represented as instruction artifacts: pass its signature explicitly or implement agents.InstructionArtifactProvider", value)
		}
		signature, demos, err := artifacts.InstructionArtifacts()
		if err != nil {
			return nil, fmt.Errorf("module %T instruction artifacts: %w", value, err)
		}
		return dspyInstructionSource{
			signature: cloneSignature(signature),
			demos:     cloneExamples(demos),
		}, nil
	default:
		return dspyInstructionSource{signature: cloneSignature(provider.GetSignature())}, nil
	}
}

func instructionSourceFromProgram(program core.Program) (InstructionSource, error) {
	if program.Forward != nil {
		return nil, fmt.Errorf("program with executable Forward behavior cannot be represented as instruction artifacts; pass the intended compatible module explicitly")
	}
	if len(program.Modules) != 1 {
		return nil, fmt.Errorf("program instruction input must contain exactly one module, got %d; pass the intended module explicitly", len(program.Modules))
	}
	for _, module := range program.Modules {
		return instructionSourceFromProvider(module)
	}
	return nil, fmt.Errorf("program instruction input has no module")
}

func isNilInstructionValue(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func cloneSignature(signature core.Signature) core.Signature {
	cloned := signature
	cloned.Inputs = append([]core.InputField(nil), signature.Inputs...)
	cloned.Outputs = append([]core.OutputField(nil), signature.Outputs...)
	return cloned
}

func cloneExamples(examples []core.Example) []core.Example {
	if examples == nil {
		return nil
	}
	cloned := make([]core.Example, len(examples))
	for i, example := range examples {
		cloned[i] = core.Example{
			Inputs:  cloneAnyMap(example.Inputs),
			Outputs: cloneAnyMap(example.Outputs),
		}
	}
	return cloned
}

func renderSignatureInstructions(signature core.Signature) string {
	var builder strings.Builder
	if instruction := strings.TrimSpace(signature.Instruction); instruction != "" {
		builder.WriteString(instruction)
	}
	if len(signature.Inputs) > 0 {
		if builder.Len() > 0 {
			builder.WriteString("\n\n")
		}
		builder.WriteString("Inputs:\n")
		renderSignatureFields(&builder, signature.Inputs)
	}
	if len(signature.Outputs) > 0 {
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString("Outputs:\n")
		renderSignatureFields(&builder, signature.Outputs)
	}
	return strings.TrimSpace(builder.String())
}

func renderSignatureFields[T interface {
	core.InputField | core.OutputField
}](builder *strings.Builder, fields []T) {
	for _, raw := range fields {
		var field core.Field
		switch value := any(raw).(type) {
		case core.InputField:
			field = value.Field
		case core.OutputField:
			field = value.Field
		}
		fmt.Fprintf(builder, "- %s [%s]", field.Name, effectiveFieldType(field))
		if field.Prefix != "" {
			fmt.Fprintf(builder, " prefix %q", field.Prefix)
		} else {
			builder.WriteString(" without a prefix")
		}
		if field.Description != "" {
			fmt.Fprintf(builder, ": %s", field.Description)
		}
		builder.WriteByte('\n')
	}
}

func renderExampleMessage[T interface {
	core.InputField | core.OutputField
}](role MessageRole, fields []T, values map[string]any) (Message, error) {
	message := Message{Role: role}
	for _, raw := range fields {
		var field core.Field
		switch value := any(raw).(type) {
		case core.InputField:
			field = value.Field
		case core.OutputField:
			field = value.Field
		}
		blocks, err := renderExampleValue(field, values[field.Name])
		if err != nil {
			return Message{}, err
		}
		message.Content = append(message.Content, blocks...)
	}
	if len(fields) == 0 && len(values) > 0 {
		keys := make([]string, 0, len(values))
		for key := range values {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			message.Content = append(message.Content, core.NewTextBlock(fmt.Sprintf("%s: %v", key, values[key])))
		}
	}
	return message, nil
}

func renderExampleValue(field core.Field, value any) ([]core.ContentBlock, error) {
	fieldType := effectiveFieldType(field)
	if block, ok := value.(core.ContentBlock); ok {
		if block.Type != fieldType {
			return nil, fmt.Errorf("field %q expects %s content, got %s", field.Name, fieldType, block.Type)
		}
		cloned := cloneContentBlocks([]core.ContentBlock{block})[0]
		if cloned.Type == core.FieldTypeText && field.Prefix != "" {
			cloned.Text = strings.TrimSpace(field.Prefix + " " + cloned.Text)
		}
		if cloned.Metadata == nil {
			cloned.Metadata = map[string]any{}
		}
		cloned.Metadata["dspy_field"] = field.Name
		return []core.ContentBlock{cloned}, nil
	}
	if blocks, ok := value.([]core.ContentBlock); ok {
		result := make([]core.ContentBlock, 0, len(blocks))
		for _, block := range blocks {
			rendered, err := renderExampleValue(field, block)
			if err != nil {
				return nil, err
			}
			result = append(result, rendered...)
		}
		return result, nil
	}
	if fieldType != core.FieldTypeText {
		return nil, fmt.Errorf("field %q expects a core.ContentBlock for %s content, got %T", field.Name, fieldType, value)
	}
	text := fmt.Sprint(value)
	if field.Prefix != "" {
		text = field.Prefix + " " + text
	}
	return []core.ContentBlock{core.NewTextBlock(strings.TrimSpace(text))}, nil
}

func effectiveFieldType(field core.Field) core.FieldType {
	if field.Type == "" {
		return core.FieldTypeText
	}
	return field.Type
}
