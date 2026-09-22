package typesafe

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ValidationError reports invalid client configuration or request data.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	if e.Field == "" {
		return e.Message
	}
	return e.Field + ": " + e.Message
}

func validationError(field, message string) error {
	return &ValidationError{Field: field, Message: message}
}

// ResponseValidationError reports a successful response whose JSON shape does
// not match the System One contract.
type ResponseValidationError struct {
	Field      string
	Message    string
	StatusCode int
	RequestID  string
}

func (e *ResponseValidationError) Error() string {
	message := "invalid System One response"
	if e.Field != "" {
		message += fmt.Sprintf(" at %q", e.Field)
	}
	message += ": " + e.Message
	if e.RequestID != "" {
		message += " (request_id=" + e.RequestID + ")"
	}
	return message
}

func responseValidationError(field, message string) *ResponseValidationError {
	return &ResponseValidationError{Field: field, Message: message}
}

// APIError is a non-2xx response from the TypeSafe API.
type APIError struct {
	StatusCode int
	RequestID  string
	Body       any
	RetryAfter time.Duration
	Message    string
}

func (e *APIError) Error() string {
	message := e.Message
	if message == "" {
		message = http.StatusText(e.StatusCode)
	}
	result := fmt.Sprintf("TypeSafe API returned %d", e.StatusCode)
	if message != "" {
		result += ": " + message
	}
	if e.RequestID != "" {
		result += " (request_id=" + e.RequestID + ")"
	}
	return result
}

// Retryable reports whether the default policy may retry this status.
func (e *APIError) Retryable() bool {
	return e.StatusCode == http.StatusRequestTimeout || e.StatusCode == http.StatusTooManyRequests ||
		e.StatusCode >= 500 && e.StatusCode <= 599
}

// ConnectionError reports a failure before a complete HTTP response arrived.
type ConnectionError struct {
	Err error
}

func (e *ConnectionError) Error() string {
	if e.Err == nil {
		return "TypeSafe API connection error"
	}
	return "TypeSafe API connection error: " + e.Err.Error()
}

func (e *ConnectionError) Unwrap() error { return e.Err }

// TimeoutError is a connection error caused by the configured per-attempt
// timeout. Caller cancellation and caller deadlines are returned as context
// errors instead.
type TimeoutError struct {
	Duration time.Duration
	Err      error
}

func (e *TimeoutError) Error() string {
	return fmt.Sprintf("TypeSafe API request timed out after %s", e.Duration)
}

func (e *TimeoutError) Unwrap() error   { return e.Err }
func (e *TimeoutError) Timeout() bool   { return true }
func (e *TimeoutError) Temporary() bool { return true }

func extractAPIMessage(body any) string {
	if text, ok := body.(string); ok {
		return text
	}
	object, ok := body.(map[string]any)
	if !ok {
		return ""
	}
	if value, ok := object["error"].(string); ok {
		return value
	}
	if nested, ok := object["error"].(map[string]any); ok {
		if value, ok := nested["message"].(string); ok {
			return value
		}
	}
	if value, ok := object["message"].(string); ok {
		return value
	}
	if value, ok := object["detail"].(string); ok {
		return value
	}
	if nested, ok := object["detail"].(map[string]any); ok {
		if value, ok := nested["message"].(string); ok {
			return value
		}
	}
	if details, ok := object["detail"].([]any); ok {
		parts := make([]string, 0, len(details))
		for _, item := range details {
			entry, ok := item.(map[string]any)
			if !ok {
				continue
			}
			message, ok := entry["msg"].(string)
			if !ok {
				continue
			}
			path := validationPath(entry["loc"])
			if path != "" {
				message = path + ": " + message
			}
			parts = append(parts, message)
		}
		return strings.Join(parts, "; ")
	}
	return ""
}

func validationPath(value any) string {
	items, ok := value.([]any)
	if !ok {
		return ""
	}
	parts := make([]string, 0, len(items))
	for _, item := range items {
		if item == "body" {
			continue
		}
		switch item := item.(type) {
		case string:
			parts = append(parts, item)
		case json.Number:
			parts = append(parts, item.String())
		case float64:
			parts = append(parts, fmt.Sprintf("%g", item))
		}
	}
	return strings.Join(parts, ".")
}
