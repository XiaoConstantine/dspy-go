package utils

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// ParseJSONResponse attempts to parse a string response as JSON.
// It handles common LLM response formats including:
// - Raw JSON.
// - JSON wrapped in markdown code blocks (```json ... ```).
func ParseJSONResponse(response string) (map[string]interface{}, error) {
	// Strip markdown code blocks if present
	cleanedResponse := stripMarkdownCodeBlock(response)

	var result map[string]interface{}
	err := json.Unmarshal([]byte(cleanedResponse), &result)
	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.InvalidResponse, "failed to parse JSON"),
			errors.Fields{
				"error_type":   "json_parse_error",
				"data_preview": truncateString(response, 100),
				"data_length":  len(response),
			})
	}
	return result, nil
}

// stripMarkdownCodeBlock removes markdown code block wrappers from a string.
// Handles formats like ```json\n{...}\n``` or ```\n{...}\n```.
func stripMarkdownCodeBlock(s string) string {
	s = strings.TrimSpace(s)

	// Check for ```json or ``` prefix
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
	} else {
		// No markdown wrapper, return as-is
		return s
	}

	// Remove closing ```
	if idx := strings.LastIndex(s, "```"); idx != -1 {
		s = s[:idx]
	}

	return strings.TrimSpace(s)
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Max returns the maximum of two integers.
func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CloneParams creates a deep copy of a parameter map.
func CloneParams(params map[string]interface{}) map[string]interface{} {
	clone := make(map[string]interface{})
	for k, v := range params {
		clone[k] = v
	}
	return clone
}

// Type conversion utilities for typed module support

// ConvertTypedInputsToLegacy converts a typed struct to map[string]any.
func ConvertTypedInputsToLegacy(inputs any) (map[string]any, error) {
	if inputs == nil {
		return make(map[string]any), nil
	}

	v := reflect.ValueOf(inputs)
	t := reflect.TypeOf(inputs)

	// Handle pointer types
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return make(map[string]any), nil
		}
		v = v.Elem()
		t = t.Elem()
	}

	// Handle non-struct types (like map[string]any already)
	if v.Kind() != reflect.Struct {
		if mapValue, ok := inputs.(map[string]any); ok {
			return mapValue, nil
		}
		return nil, fmt.Errorf("input must be a struct or map[string]any, got %T", inputs)
	}

	result := make(map[string]any)

	// Extract field values using reflection
	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		value := v.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		fieldName := strings.ToLower(field.Name) // Question → question
		if dspyTag := field.Tag.Get("dspy"); dspyTag != "" {
			parts := strings.Split(dspyTag, ",")
			if len(parts) > 0 && parts[0] != "" {
				fieldName = parts[0] // Override lowercase default if specified
			}
		}

		// Convert field value to interface{}
		result[fieldName] = value.Interface()
	}

	return result, nil
}

// ConvertLegacyOutputsToTyped converts map[string]any to a typed struct.
func ConvertLegacyOutputsToTyped[T any](outputs map[string]any) (T, error) {
	var zero T

	if outputs == nil {
		return zero, fmt.Errorf("outputs cannot be nil")
	}

	// Get the type information for T
	outputType := reflect.TypeOf(zero)

	// Handle interface types like `any` or `map[string]any`
	if outputType == nil || outputType.Kind() == reflect.Map {
		if mapValue, ok := any(outputs).(T); ok {
			return mapValue, nil
		}
		return zero, fmt.Errorf("failed to convert map[string]any to type %T: type assertion failed (ensure the target type is compatible with map types)", zero)
	}

	// Handle pointer to struct types
	if outputType.Kind() == reflect.Ptr {
		elemType := outputType.Elem()
		if elemType.Kind() != reflect.Struct {
			return zero, fmt.Errorf("unsupported output type %T: pointers are only supported for struct types, not %s", zero, elemType.Kind())
		}
		newValue := reflect.New(elemType)
		if err := PopulateStructFromMap(newValue.Elem(), elemType, outputs); err != nil {
			return zero, err
		}
		return newValue.Interface().(T), nil
	}

	// Handle struct types
	if outputType.Kind() != reflect.Struct {
		return zero, fmt.Errorf("unsupported output type %T (kind: %s): only struct, map[string]any, and interface{} types are supported for conversion", zero, outputType.Kind())
	}

	// Create a new instance of the struct
	newValue := reflect.New(outputType).Elem()
	if err := PopulateStructFromMap(newValue, outputType, outputs); err != nil {
		return zero, err
	}

	return newValue.Interface().(T), nil
}

// PopulateStructFromMap fills a struct with values from a map.
func PopulateStructFromMap(structValue reflect.Value, structType reflect.Type, data map[string]any) error {
	for i := 0; i < structType.NumField(); i++ {
		field := structType.Field(i)
		fieldValue := structValue.Field(i)

		// Skip unexported fields
		if !field.IsExported() || !fieldValue.CanSet() {
			continue
		}

		// Get field name from dspy tag or use struct field name
		fieldName := field.Name
		if dspyTag := field.Tag.Get("dspy"); dspyTag != "" {
			parts := strings.Split(dspyTag, ",")
			if len(parts) > 0 && parts[0] != "" {
				fieldName = parts[0]
			}
		}

		// Get value from map
		value, exists := data[fieldName]
		if !exists {
			// Field not found in map - leave as zero value
			continue
		}

		// Convert and set the value
		if err := SetFieldValue(fieldValue, value); err != nil {
			return fmt.Errorf("failed to set field %s: %w", field.Name, err)
		}
	}

	return nil
}

// SetFieldValue sets a struct field value with type conversion.
func SetFieldValue(fieldValue reflect.Value, value any) error {
	if value == nil {
		// Set to zero value
		fieldValue.Set(reflect.Zero(fieldValue.Type()))
		return nil
	}

	valueReflect := reflect.ValueOf(value)
	fieldType := fieldValue.Type()

	// Direct assignment if types match
	if valueReflect.Type().AssignableTo(fieldType) {
		fieldValue.Set(valueReflect)
		return nil
	}

	// Handle special cases with overflow checks BEFORE general conversion
	switch fieldType.Kind() {
	case reflect.String:
		fieldValue.SetString(fmt.Sprintf("%v", value))
		return nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if intVal, ok := convertToInt(value); ok {
			if fieldValue.OverflowInt(intVal) {
				return fmt.Errorf("value %v overflows field of type %s", value, fieldType)
			}
			fieldValue.SetInt(intVal)
			return nil
		}
	case reflect.Float32, reflect.Float64:
		if floatVal, ok := convertToFloat(value); ok {
			if fieldValue.OverflowFloat(floatVal) {
				return fmt.Errorf("value %v overflows field of type %s", value, fieldType)
			}
			fieldValue.SetFloat(floatVal)
			return nil
		}
	case reflect.Bool:
		if boolVal, ok := convertToBool(value); ok {
			fieldValue.SetBool(boolVal)
			return nil
		}
	}

	// Type conversion if possible (fallback for other types)
	if valueReflect.Type().ConvertibleTo(fieldType) {
		fieldValue.Set(valueReflect.Convert(fieldType))
		return nil
	}

	return fmt.Errorf("cannot convert %T to %s", value, fieldType)
}

// Helper conversion functions.
func convertToInt(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint:
		if uint64(v) > math.MaxInt64 {
			return 0, false
		}
		return int64(v), true
	case uint8:
		return int64(v), true
	case uint16:
		return int64(v), true
	case uint32:
		return int64(v), true
	case uint64:
		if v > math.MaxInt64 {
			return 0, false
		}
		return int64(v), true
	case float32:
		if v > math.MaxInt64 || v < math.MinInt64 {
			return 0, false
		}
		return int64(v), true
	case float64:
		if v > math.MaxInt64 || v < math.MinInt64 {
			return 0, false
		}
		return int64(v), true
	case string:
		// Use strconv for robust and idiomatic integer parsing
		if i, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
			return i, true
		}
	}
	return 0, false
}

func convertToFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	case string:
		// Use strconv for robust and idiomatic float parsing
		if f, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

func convertToBool(value any) (bool, bool) {
	switch v := value.(type) {
	case bool:
		return v, true
	case string:
		// Use strconv.ParseBool for robust and idiomatic bool parsing
		// Handles "true", "TRUE", "True", "1", "false", "FALSE", "False", "0", etc.
		if b, err := strconv.ParseBool(strings.TrimSpace(v)); err == nil {
			return b, true
		}
	case int, int8, int16, int32, int64:
		return reflect.ValueOf(v).Int() != 0, true
	case uint, uint8, uint16, uint32, uint64:
		return reflect.ValueOf(v).Uint() != 0, true
	case float32, float64:
		return reflect.ValueOf(v).Float() != 0, true
	}
	return false, false
}
