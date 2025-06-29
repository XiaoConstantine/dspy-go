package core

import (
	"fmt"
	"strings"
)

// Field represents a single field in a signature.
type Field struct {
	Name        string
	Description string
	Prefix      string
}

// NewField creates a new Field with smart defaults.
func NewField(name string, opts ...FieldOption) Field {
	// Start with sensible defaults
	f := Field{
		Name:   name,
		Prefix: name + ":", // Default prefix is the field name with colon
	}

	// Apply any custom options
	for _, opt := range opts {
		opt(&f)
	}

	return f
}

// FieldOption allows customization of Field creation.
type FieldOption func(*Field)

// WithDescription sets a custom description.
func WithDescription(desc string) FieldOption {
	return func(f *Field) {
		f.Description = desc
	}
}

// WithCustomPrefix overrides the default prefix.
func WithCustomPrefix(prefix string) FieldOption {
	return func(f *Field) {
		f.Prefix = prefix
	}
}

// WithNoPrefix removes the prefix entirely.
func WithNoPrefix() FieldOption {
	return func(f *Field) {
		f.Prefix = ""
	}
}

// InputField represents an input field.
type InputField struct {
	Field
}

// OutputField represents an output field.
type OutputField struct {
	Field
}

// Signature represents the input and output specification of a module.
type Signature struct {
	Inputs      []InputField
	Outputs     []OutputField
	Instruction string
}

// NewSignature creates a new Signature with the given inputs and outputs.
func NewSignature(inputs []InputField, outputs []OutputField) Signature {
	return Signature{
		Inputs:  inputs,
		Outputs: outputs,
	}
}

// WithInstruction adds an instruction to the Signature.
func (s Signature) WithInstruction(instruction string) Signature {
	s.Instruction = instruction
	return s
}

// String returns a string representation of the Signature.
func (s Signature) String() string {
	var sb strings.Builder
	sb.WriteString("Inputs:\n")
	for _, input := range s.Inputs {
		sb.WriteString(fmt.Sprintf("  - %s (%s)\n", input.Name, input.Description))
	}
	sb.WriteString("Outputs:\n")
	for _, output := range s.Outputs {
		sb.WriteString(fmt.Sprintf("  - %s (%s)\n", output.Name, output.Description))
	}
	if s.Instruction != "" {
		sb.WriteString(fmt.Sprintf("Instruction: %s\n", s.Instruction))
	}
	return sb.String()
}

// ParseSignature parses a signature string into a Signature struct.
func ParseSignature(signatureStr string) (Signature, error) {
	parts := strings.Split(signatureStr, "->")
	if len(parts) != 2 {
		return Signature{}, fmt.Errorf("invalid signature format: %s", signatureStr)
	}

	inputs := parseInputFields(strings.TrimSpace(parts[0]))
	outputs := parseOutputFields(strings.TrimSpace(parts[1]))

	return NewSignature(inputs, outputs), nil
}

func parseInputFields(fieldsStr string) []InputField {
	fieldStrs := strings.Split(fieldsStr, ",")
	fields := make([]InputField, len(fieldStrs))
	for i, fieldStr := range fieldStrs {
		fieldStr = strings.TrimSpace(fieldStr)
		fields[i] = InputField{Field: Field{Name: fieldStr}}
	}
	return fields
}

func parseOutputFields(fieldsStr string) []OutputField {
	fieldStrs := strings.Split(fieldsStr, ",")
	fields := make([]OutputField, len(fieldStrs))
	for i, fieldStr := range fieldStrs {
		fieldStr = strings.TrimSpace(fieldStr)
		fields[i] = OutputField{Field: Field{Name: fieldStr}}
	}
	return fields
}

// ShorthandNotation creates a Signature from a shorthand notation string.
func ShorthandNotation(notation string) (Signature, error) {
	return ParseSignature(notation)
}

// AppendInput adds an input field to the signature.
func (s Signature) AppendInput(name string, prefix string, description string) Signature {
	newInput := InputField{
		Field: Field{
			Name:        name,
			Prefix:      prefix,
			Description: description,
		},
	}
	s.Inputs = append(s.Inputs, newInput)
	return s
}

// PrependOutput adds an output field to the beginning of the outputs.
func (s Signature) PrependOutput(name string, prefix string, description string) Signature {
	newOutput := OutputField{
		Field: Field{
			Name:        name,
			Prefix:      prefix,
			Description: description,
		},
	}
	s.Outputs = append([]OutputField{newOutput}, s.Outputs...)
	return s
}
