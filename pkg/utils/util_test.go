package utils

import (
	"reflect"
	"testing"
)

func TestParseJSONResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name:     "Valid JSON object",
			input:    `{"key": "value", "number": 42}`,
			expected: map[string]interface{}{"key": "value", "number": float64(42)},
			wantErr:  false,
		},
		{
			name:     "Empty JSON object",
			input:    `{}`,
			expected: map[string]interface{}{},
			wantErr:  false,
		},
		{
			name:     "JSON with nested object",
			input:    `{"outer": {"inner": "value"}}`,
			expected: map[string]interface{}{"outer": map[string]interface{}{"inner": "value"}},
			wantErr:  false,
		},
		{
			name:     "JSON with array",
			input:    `{"array": [1, 2, 3]}`,
			expected: map[string]interface{}{"array": []interface{}{float64(1), float64(2), float64(3)}},
			wantErr:  false,
		},
		{
			name:     "Invalid JSON",
			input:    `{"key": "value"`,
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Empty string",
			input:    "",
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "Non-object JSON",
			input:    `["array", "items"]`,
			expected: nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ParseJSONResponse(tt.input)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseJSONResponse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("ParseJSONResponse() = %v, want %v", result, tt.expected)
			}
		})
	}
}
