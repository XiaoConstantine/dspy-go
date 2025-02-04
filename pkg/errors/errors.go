package errors

import (
	"fmt"
	"strings"
)

// ErrorCode defines known error types in the system.
type ErrorCode int

const (
	// Core error codes.
	Unknown ErrorCode = iota
	InvalidInput
	ValidationFailed
	ResourceNotFound
	Timeout
	RateLimitExceeded
	Canceled

	// LLM specific errors.
	LLMGenerationFailed
	TokenLimitExceeded
	InvalidResponse

	// Workflow errors.
	WorkflowExecutionFailed
	StepExecutionFailed
	InvalidWorkflowState
)

// Error represents a structured error with context.
type Error struct {
	code     ErrorCode // Type of error
	message  string    // Human-readable message
	original error     // Original/wrapped error
	fields   Fields    // Additional context
}

// Fields carries structured data about the error.
type Fields map[string]interface{}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString(e.message)

	if e.original != nil {
		b.WriteString(": ")
		b.WriteString(e.original.Error())
	}

	if len(e.fields) > 0 {
		b.WriteString(" [")
		for k, v := range e.fields {
			fmt.Fprintf(&b, "%s=%v ", k, v)
		}
		b.WriteString("]")
	}

	return strings.TrimSpace(b.String())
}

func (e *Error) Unwrap() error {
	return e.original
}

func (e *Error) Code() ErrorCode {
	return e.code
}

// New creates a new error with a code and message.
func New(code ErrorCode, message string) error {
	return &Error{
		code:    code,
		message: message,
	}
}

// Wrap wraps an existing error with additional context.
func Wrap(err error, code ErrorCode, message string) error {
	if err == nil {
		return nil
	}
	return &Error{
		code:     code,
		message:  message,
		original: err,
	}
}

// WithFields adds structured context to an error.
func WithFields(err error, fields Fields) error {
	if err == nil {
		return nil
	}

	// If it's already our error type, add fields
	if e, ok := err.(*Error); ok {
		newFields := make(Fields)
		for k, v := range e.fields {
			newFields[k] = v
		}
		for k, v := range fields {
			newFields[k] = v
		}

		return &Error{
			code:     e.code,
			message:  e.message,
			original: e.original,
			fields:   newFields,
		}
	}

	// Otherwise, create new error
	return &Error{
		code:     Unknown,
		message:  err.Error(),
		original: err,
		fields:   fields,
	}
}

// Is implements error matching.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok {
		return false
	}
	return e.code == t.code
}

// As implements error type casting for errors.As.
func (e *Error) As(target interface{}) bool {
	// Check if target is a pointer to *Error
	errorPtr, ok := target.(**Error)
	if !ok {
		return false
	}
	// Set the target pointer to our error
	*errorPtr = e
	return true
}

func (e *Error) Fields() Fields {
	if e.fields == nil {
		return Fields{}
	}
	// Create a copy of the fields map
	fields := make(Fields, len(e.fields))
	for k, v := range e.fields {
		fields[k] = v
	}
	return fields
}
