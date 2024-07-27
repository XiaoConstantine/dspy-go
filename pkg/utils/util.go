package utils

import (
	"encoding/json"
	"fmt"
)

// ParseJSONResponse attempts to parse a string response as JSON.
func ParseJSONResponse(response string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := json.Unmarshal([]byte(response), &result)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response as JSON: %w", err)
	}
	return result, nil
}
