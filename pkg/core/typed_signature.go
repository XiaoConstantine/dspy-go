package core

import (
	"fmt"
	"reflect"
	"strings"
)

// TypedSignature provides compile-time type safety for module inputs and outputs.
type TypedSignature[TInput, TOutput any] interface {
	// GetInputType returns the reflect.Type for the input struct
	GetInputType() reflect.Type

	// GetOutputType returns the reflect.Type for the output struct
	GetOutputType() reflect.Type

	// ValidateInput performs compile-time and runtime validation of input
	ValidateInput(input TInput) error

	// ValidateOutput performs compile-time and runtime validation of output
	ValidateOutput(output TOutput) error

	// GetFieldMetadata returns parsed struct tag metadata
	GetFieldMetadata() SignatureMetadata

	// ToLegacySignature converts to the existing Signature interface for backward compatibility
	ToLegacySignature() Signature
}

// SignatureMetadata contains parsed information from struct tags.
type SignatureMetadata struct {
	Inputs      []FieldMetadata
	Outputs     []FieldMetadata
	Instruction string
}

// FieldMetadata represents parsed struct tag information.
type FieldMetadata struct {
	Name        string       // Field name from struct tag or field name
	Required    bool         // Whether field is required
	Description string       // Field description
	Prefix      string       // Output prefix for LLM generation
	Type        FieldType    // Field type (text, image, audio)
	GoType      reflect.Type // The actual Go type
}

// typedSignatureImpl implements TypedSignature.
type typedSignatureImpl[TInput, TOutput any] struct {
	inputType   reflect.Type
	outputType  reflect.Type
	metadata    SignatureMetadata
	instruction string
}

// NewTypedSignature creates a new typed signature for the given input/output types.
func NewTypedSignature[TInput, TOutput any]() TypedSignature[TInput, TOutput] {
	var input TInput
	var output TOutput

	inputType := reflect.TypeOf(input)
	outputType := reflect.TypeOf(output)

	// Handle pointer types
	if inputType != nil && inputType.Kind() == reflect.Ptr {
		inputType = inputType.Elem()
	}
	if outputType != nil && outputType.Kind() == reflect.Ptr {
		outputType = outputType.Elem()
	}

	metadata := SignatureMetadata{
		Inputs:  parseStructFields(inputType, true),
		Outputs: parseStructFields(outputType, false),
	}

	return &typedSignatureImpl[TInput, TOutput]{
		inputType:  inputType,
		outputType: outputType,
		metadata:   metadata,
	}
}

// WithInstruction sets an instruction for the typed signature.
func WithInstruction[TInput, TOutput any](instruction string) func(TypedSignature[TInput, TOutput]) TypedSignature[TInput, TOutput] {
	return func(ts TypedSignature[TInput, TOutput]) TypedSignature[TInput, TOutput] {
		if impl, ok := ts.(*typedSignatureImpl[TInput, TOutput]); ok {
			impl.instruction = instruction
			impl.metadata.Instruction = instruction
		}
		return ts
	}
}

func (ts *typedSignatureImpl[TInput, TOutput]) GetInputType() reflect.Type {
	return ts.inputType
}

func (ts *typedSignatureImpl[TInput, TOutput]) GetOutputType() reflect.Type {
	return ts.outputType
}

func (ts *typedSignatureImpl[TInput, TOutput]) ValidateInput(input TInput) error {
	return validateStruct(input, ts.metadata.Inputs, "input")
}

func (ts *typedSignatureImpl[TInput, TOutput]) ValidateOutput(output TOutput) error {
	return validateStruct(output, ts.metadata.Outputs, "output")
}

func (ts *typedSignatureImpl[TInput, TOutput]) GetFieldMetadata() SignatureMetadata {
	return ts.metadata
}

func (ts *typedSignatureImpl[TInput, TOutput]) ToLegacySignature() Signature {
	// Convert typed signature to legacy format
	inputs := make([]InputField, len(ts.metadata.Inputs))
	for i, field := range ts.metadata.Inputs {
		inputs[i] = InputField{
			Field: Field{
				Name:        field.Name,
				Description: field.Description,
				Prefix:      field.Prefix,
				Type:        field.Type,
			},
		}
	}

	outputs := make([]OutputField, len(ts.metadata.Outputs))
	for i, field := range ts.metadata.Outputs {
		outputs[i] = OutputField{
			Field: Field{
				Name:        field.Name,
				Description: field.Description,
				Prefix:      field.Prefix,
				Type:        field.Type,
			},
		}
	}

	signature := NewSignature(inputs, outputs)
	if ts.instruction != "" {
		signature = signature.WithInstruction(ts.instruction)
	}

	return signature
}

// parseStructFields extracts field metadata from struct tags.
func parseStructFields(t reflect.Type, isInput bool) []FieldMetadata {
	if t == nil {
		return nil
	}

	// Handle non-struct types
	if t.Kind() != reflect.Struct {
		return nil
	}

	var fields []FieldMetadata

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		metadata := parseFieldMetadata(field, isInput)
		fields = append(fields, metadata)
	}

	return fields
}

// parseFieldMetadata parses struct tag information for a single field.
func parseFieldMetadata(field reflect.StructField, isInput bool) FieldMetadata {
	metadata := FieldMetadata{
		Name:   field.Name,
		GoType: field.Type,
		Type:   FieldTypeText, // Default to text
	}

	// Parse dspy struct tag: `dspy:"fieldname,required"`
	if dspyTag := field.Tag.Get("dspy"); dspyTag != "" {
		parts := strings.Split(dspyTag, ",")
		if len(parts) > 0 && parts[0] != "" {
			metadata.Name = parts[0]
		}

		// Check for required flag
		for _, part := range parts[1:] {
			switch strings.TrimSpace(part) {
			case "required":
				metadata.Required = true
			}
		}
	}

	// Parse description tag
	if desc := field.Tag.Get("description"); desc != "" {
		metadata.Description = desc
	}

	// Set default prefix (field name with colon)
	if isInput {
		metadata.Prefix = ""
	} else {
		metadata.Prefix = metadata.Name + ":"
	}

	// Parse prefix tag for outputs
	if prefix := field.Tag.Get("prefix"); prefix != "" {
		metadata.Prefix = prefix
	}

	// Determine field type based on Go type
	metadata.Type = inferFieldType(field.Type)

	return metadata
}

// inferFieldType determines DSPy field type from Go type.
func inferFieldType(t reflect.Type) FieldType {
	// Handle pointer types
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	switch t {
	case reflect.TypeOf([]byte{}):
		return FieldTypeImage // Assume byte slices are images
	default:
		return FieldTypeText
	}
}

// validateStruct performs runtime validation of struct fields.
func validateStruct(value any, expectedFields []FieldMetadata, fieldType string) error {
	if value == nil {
		return fmt.Errorf("%s cannot be nil", fieldType)
	}

	v := reflect.ValueOf(value)

	// Handle pointer types
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return fmt.Errorf("%s cannot be nil", fieldType)
		}
		v = v.Elem()
	}

	if v.Kind() != reflect.Struct {
		return fmt.Errorf("%s must be a struct, got %s", fieldType, v.Kind())
	}

	t := v.Type()

	// Validate required fields by matching Go field names
	for _, expected := range expectedFields {
		if !expected.Required {
			continue
		}

		// Find the Go field name that corresponds to this metadata
		var goFieldName string
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if !field.IsExported() {
				continue
			}

			// Get the dspy tag name or use field name
			fieldName := field.Name
			if dspyTag := field.Tag.Get("dspy"); dspyTag != "" {
				parts := strings.Split(dspyTag, ",")
				if len(parts) > 0 && parts[0] != "" {
					fieldName = parts[0]
				}
			}

			if fieldName == expected.Name {
				goFieldName = field.Name
				break
			}
		}

		if goFieldName == "" {
			return fmt.Errorf("required %s field '%s' is missing", fieldType, expected.Name)
		}

		field := v.FieldByName(goFieldName)
		if !field.IsValid() {
			return fmt.Errorf("required %s field '%s' is missing", fieldType, expected.Name)
		}

		// Check if field is zero value
		if field.IsZero() {
			return fmt.Errorf("required %s field '%s' cannot be empty", fieldType, expected.Name)
		}
	}

	return nil
}

// Backward compatibility: convert legacy signature to typed.
func FromLegacySignature(sig Signature) TypedSignature[map[string]any, map[string]any] {
	metadata := SignatureMetadata{
		Instruction: sig.Instruction,
	}

	// Convert input fields
	for _, input := range sig.Inputs {
		metadata.Inputs = append(metadata.Inputs, FieldMetadata{
			Name:        input.Name,
			Description: input.Description,
			Prefix:      input.Prefix,
			Type:        input.Type,
			Required:    true,               // Assume legacy fields are required
			GoType:      reflect.TypeOf(""), // Default to string
		})
	}

	// Convert output fields
	for _, output := range sig.Outputs {
		metadata.Outputs = append(metadata.Outputs, FieldMetadata{
			Name:        output.Name,
			Description: output.Description,
			Prefix:      output.Prefix,
			Type:        output.Type,
			Required:    false,              // Outputs are generally not "required"
			GoType:      reflect.TypeOf(""), // Default to string
		})
	}

	return &typedSignatureImpl[map[string]any, map[string]any]{
		inputType:   reflect.TypeOf(map[string]any{}),
		outputType:  reflect.TypeOf(map[string]any{}),
		metadata:    metadata,
		instruction: sig.Instruction,
	}
}
