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
