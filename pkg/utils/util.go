package utils

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// ParseJSONResponse attempts to parse a string response as JSON.
func ParseJSONResponse(response string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := json.Unmarshal([]byte(response), &result)
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

		// Get field name from dspy tag or use struct field name
		fieldName := field.Name
		if dspyTag := field.Tag.Get("dspy"); dspyTag != "" {
			parts := strings.Split(dspyTag, ",")
			if len(parts) > 0 && parts[0] != "" {
				fieldName = parts[0]
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

	// Handle pointer types
	if outputType.Kind() == reflect.Ptr {
		outputType = outputType.Elem()
		// Create a new instance of the pointed-to type
		newValue := reflect.New(outputType)
		err := PopulateStructFromMap(newValue.Elem(), outputType, outputs)
		if err != nil {
			return zero, err
		}
		return newValue.Interface().(T), nil
	}

	// Handle direct struct types
	if outputType.Kind() != reflect.Struct {
		// Handle map[string]any case
		if mapValue, ok := any(outputs).(T); ok {
			return mapValue, nil
		}
		return zero, fmt.Errorf("output type must be a struct or map[string]any, got %T", zero)
	}

	// Create a new instance of the struct
	newValue := reflect.New(outputType).Elem()
	err := PopulateStructFromMap(newValue, outputType, outputs)
	if err != nil {
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

	// Type conversion if possible
	if valueReflect.Type().ConvertibleTo(fieldType) {
		fieldValue.Set(valueReflect.Convert(fieldType))
		return nil
	}

	// Handle special cases
	switch fieldType.Kind() {
	case reflect.String:
		fieldValue.SetString(fmt.Sprintf("%v", value))
		return nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if intVal, ok := convertToInt(value); ok {
			fieldValue.SetInt(intVal)
			return nil
		}
	case reflect.Float32, reflect.Float64:
		if floatVal, ok := convertToFloat(value); ok {
			fieldValue.SetFloat(floatVal)
			return nil
		}
	case reflect.Bool:
		if boolVal, ok := convertToBool(value); ok {
			fieldValue.SetBool(boolVal)
			return nil
		}
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
	case float32:
		return int64(v), true
	case float64:
		return int64(v), true
	case string:
		if parsed, err := fmt.Sscanf(v, "%d", new(int64)); err == nil && parsed == 1 {
			var result int64
			if _, err := fmt.Sscanf(v, "%d", &result); err == nil {
				return result, true
			}
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
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case string:
		if parsed, err := fmt.Sscanf(v, "%f", new(float64)); err == nil && parsed == 1 {
			var result float64
			if _, err := fmt.Sscanf(v, "%f", &result); err == nil {
				return result, true
			}
		}
	}
	return 0, false
}

func convertToBool(value any) (bool, bool) {
	switch v := value.(type) {
	case bool:
		return v, true
	case string:
		if v == "true" || v == "1" {
			return true, true
		}
		if v == "false" || v == "0" {
			return false, true
		}
	case int:
		return v != 0, true
	case float64:
		return v != 0, true
	}
	return false, false
}
