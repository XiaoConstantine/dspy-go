package core

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
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

	// WithInstruction returns a new TypedSignature with the specified instruction
	WithInstruction(instruction string) TypedSignature[TInput, TOutput]
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
	GoFieldName string       // Original Go struct field name for direct lookup
	Required    bool         // Whether field is required
	Description string       // Field description
	Prefix      string       // Output prefix for LLM generation
	Type        FieldType    // Field type (text, image, audio)
	GoType      reflect.Type // The actual Go type
}

// typedSignatureImpl implements TypedSignature.
type typedSignatureImpl[TInput, TOutput any] struct {
	inputType  reflect.Type
	outputType reflect.Type
	metadata   SignatureMetadata
}

// createTypedSignatureImpl is a helper function that creates a TypedSignature implementation.
// This reduces code duplication between cached and non-cached versions.
func createTypedSignatureImpl[TInput, TOutput any](inputType, outputType reflect.Type) *typedSignatureImpl[TInput, TOutput] {
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

// getReflectTypes extracts and normalizes reflect.Type information for the given generic types.
// It handles pointer types by extracting the underlying element type.
func getReflectTypes[TInput, TOutput any]() (reflect.Type, reflect.Type) {
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

	return inputType, outputType
}

// NewTypedSignature creates a new typed signature for the given input/output types.
func NewTypedSignature[TInput, TOutput any]() TypedSignature[TInput, TOutput] {
	inputType, outputType := getReflectTypes[TInput, TOutput]()
	return createTypedSignatureImpl[TInput, TOutput](inputType, outputType)
}

// Global cache for TypedSignature instances to improve performance.
var typedSignatureCache sync.Map

// signatureCacheKey represents a composite key for caching TypedSignatures.
type signatureCacheKey struct {
	inputType  reflect.Type
	outputType reflect.Type
}

// NewTypedSignatureCached creates a cached typed signature for the given input/output types.
// This function provides better performance for repeated calls with the same types.
func NewTypedSignatureCached[TInput, TOutput any]() TypedSignature[TInput, TOutput] {
	inputType, outputType := getReflectTypes[TInput, TOutput]()

	// Create cache key
	key := signatureCacheKey{
		inputType:  inputType,
		outputType: outputType,
	}

	// Try to get from cache first
	if cached, ok := typedSignatureCache.Load(key); ok {
		return cached.(TypedSignature[TInput, TOutput])
	}

	// Not in cache, create new signature using the helper
	signature := createTypedSignatureImpl[TInput, TOutput](inputType, outputType)

	// Store in cache for future use
	typedSignatureCache.Store(key, signature)

	return signature
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
	if ts.metadata.Instruction != "" {
		signature = signature.WithInstruction(ts.metadata.Instruction)
	}

	return signature
}

func (ts *typedSignatureImpl[TInput, TOutput]) WithInstruction(instruction string) TypedSignature[TInput, TOutput] {
	// Create a deep copy with the new instruction to avoid shallow copy issues
	newMetadata := SignatureMetadata{
		Instruction: instruction,
		Inputs:      make([]FieldMetadata, len(ts.metadata.Inputs)),
		Outputs:     make([]FieldMetadata, len(ts.metadata.Outputs)),
	}

	// Deep copy the input and output field metadata
	copy(newMetadata.Inputs, ts.metadata.Inputs)
	copy(newMetadata.Outputs, ts.metadata.Outputs)

	return &typedSignatureImpl[TInput, TOutput]{
		inputType:  ts.inputType,
		outputType: ts.outputType,
		metadata:   newMetadata,
	}
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
		Name:        field.Name,
		GoFieldName: field.Name, // Cache the Go field name for efficient lookup
		GoType:      field.Type,
		Type:        FieldTypeText, // Default to text
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

// validateStruct performs runtime validation of struct fields and maps.
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

	// Handle map validation (for legacy signatures)
	if v.Kind() == reflect.Map {
		for _, expected := range expectedFields {
			if !expected.Required {
				continue
			}
			key := reflect.ValueOf(expected.Name)
			mapValue := v.MapIndex(key)
			if !mapValue.IsValid() || mapValue.IsZero() {
				return fmt.Errorf("required %s field '%s' cannot be empty", fieldType, expected.Name)
			}
		}
		return nil
	}

	// Handle struct validation
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("%s must be a struct or map, got %s", fieldType, v.Kind())
	}

	// Validate required fields using cached Go field names
	for _, expected := range expectedFields {
		if !expected.Required {
			continue
		}

		// Use cached GoFieldName for efficient lookup
		var field reflect.Value
		if expected.GoFieldName != "" {
			field = v.FieldByName(expected.GoFieldName)
		} else {
			// Fallback for legacy metadata without GoFieldName
			field = v.FieldByName(expected.Name)
		}

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
			GoFieldName: input.Name, // For maps, GoFieldName is the same as Name
			Description: input.Description,
			Prefix:      input.Prefix,
			Type:        input.Type,
			Required:    false,              // Default to optional for backward compatibility
			GoType:      reflect.TypeOf(""), // Default to string
		})
	}

	// Convert output fields
	for _, output := range sig.Outputs {
		metadata.Outputs = append(metadata.Outputs, FieldMetadata{
			Name:        output.Name,
			GoFieldName: output.Name, // For maps, GoFieldName is the same as Name
			Description: output.Description,
			Prefix:      output.Prefix,
			Type:        output.Type,
			Required:    false,              // Outputs are generally not "required"
			GoType:      reflect.TypeOf(""), // Default to string
		})
	}

	return &typedSignatureImpl[map[string]any, map[string]any]{
		inputType:  reflect.TypeOf(map[string]any{}),
		outputType: reflect.TypeOf(map[string]any{}),
		metadata:   metadata,
	}
}
