package utils

import (
	"encoding/json"

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
