package errors

import (
	stderrors "errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewError tests the basic creation of errors.
func TestNewError(t *testing.T) {
	tests := []struct {
		name    string
		code    ErrorCode
		message string
	}{
		{
			name:    "ValidationFailed",
			code:    ValidationFailed,
			message: "validation failed",
		},
		{
			name:    "ResourceNotFound",
			code:    ResourceNotFound,
			message: "resource not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := New(tt.code, tt.message)

			customErr, ok := err.(*Error)

			assert.True(t, ok, "should be a custom *Error")
			// Test that error was created correctly
			assert.Equal(t, tt.code, customErr.Code())
			assert.Equal(t, tt.message, customErr.Error())

			// Test nil original error for new errors
			assert.Nil(t, customErr.Unwrap())
		})
	}
}

// TestWrapError tests error wrapping functionality.
func TestWrapError(t *testing.T) {
	// Original error to wrap
	originalErr := stderrors.New("original error")

	tests := []struct {
		name       string
		err        error
		code       ErrorCode
		wrapMsg    string
		expectNil  bool
		expectCode ErrorCode
	}{
		{
			name:       "Wrap normal error",
			err:        originalErr,
			code:       ValidationFailed,
			wrapMsg:    "validation context",
			expectNil:  false,
			expectCode: ValidationFailed,
		},
		{
			name:      "Wrap nil error",
			err:       nil,
			code:      ValidationFailed,
			wrapMsg:   "validation context",
			expectNil: true,
		},
		{
			name:       "Wrap custom error",
			err:        New(ResourceNotFound, "not found"),
			code:       ValidationFailed,
			wrapMsg:    "validation context",
			expectNil:  false,
			expectCode: ValidationFailed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := Wrap(tt.err, tt.code, tt.wrapMsg)

			if tt.expectNil {
				assert.Nil(t, wrapped)
				return
			}

			assert.NotNil(t, wrapped)

			// Check proper wrapping
			ourErr := wrapped.(*Error)
			assert.Equal(t, tt.expectCode, ourErr.Code())
			assert.Contains(t, ourErr.Error(), tt.wrapMsg)

			// Verify original error is preserved
			unwrapped := ourErr.Unwrap()
			if tt.err != nil {
				assert.Equal(t, tt.err.Error(), unwrapped.Error())
			}
		})
	}
}

// TestErrorInterfaces tests compliance with Go error interfaces.
func TestErrorInterfaces(t *testing.T) {
	t.Run("errors.Is support", func(t *testing.T) {
		// Create two errors of same type
		err1 := New(ValidationFailed, "first")
		err2 := New(ValidationFailed, "second")

		// Create error of different type
		err3 := New(ResourceNotFound, "third")

		// Test Is behavior
		assert.True(t, stderrors.Is(err1, err2),
			"Errors with same code should match with Is")
		assert.False(t, stderrors.Is(err1, err3),
			"Errors with different codes should not match with Is")
	})

	t.Run("errors.As support", func(t *testing.T) {
		originalErr := New(ValidationFailed, "original")
		wrappedErr := Wrap(originalErr, ResourceNotFound, "wrapped")

		var customErr *Error
		assert.True(t, stderrors.As(wrappedErr, &customErr),
			"Should be able to extract custom error type")
		assert.Equal(t, ResourceNotFound, customErr.Code())
	})

	t.Run("error unwrapping", func(t *testing.T) {
		baseErr := stderrors.New("base error")
		wrapped := Wrap(baseErr, ValidationFailed, "wrapped error")

		unwrapped := stderrors.Unwrap(wrapped)
		assert.Equal(t, baseErr.Error(), unwrapped.Error())
	})
}

// TestErrorString tests the string representation of errors.
func TestErrorString(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		contains []string // Strings that should be in error message
	}{
		{
			name:     "Simple error",
			err:      New(ValidationFailed, "validation failed"),
			contains: []string{"validation failed"},
		},
		{
			name: "Wrapped error",
			err: Wrap(
				stderrors.New("original problem"),
				ValidationFailed,
				"validation context",
			),
			contains: []string{
				"validation context",
				"original problem",
			},
		},
		{
			name: "Multiple wraps",
			err: Wrap(
				Wrap(
					stderrors.New("root cause"),
					ResourceNotFound,
					"not found",
				),
				ValidationFailed,
				"validation failed",
			),
			contains: []string{
				"validation failed",
				"not found",
				"root cause",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errString := tt.err.Error()
			for _, str := range tt.contains {
				assert.Contains(t, errString, str,
					"Error string should contain expected message")
			}
		})
	}
}

func TestErrorFields(t *testing.T) {
	t.Run("Empty fields", func(t *testing.T) {
		err := New(ValidationFailed, "error")
		customErr := err.(*Error)
		assert.Empty(t, customErr.Fields())
	})

	t.Run("Add fields", func(t *testing.T) {
		fields := Fields{
			"string": "value",
			"int":    42,
			"bool":   true,
		}
		err := WithFields(New(ValidationFailed, "error"), fields)
		customErr := err.(*Error)
		assert.Equal(t, fields, customErr.Fields())
	})

	t.Run("Merge fields", func(t *testing.T) {
		err := WithFields(New(ValidationFailed, "error"), Fields{"a": 1})
		err = WithFields(err, Fields{"b": 2})
		customErr := err.(*Error)
		assert.Len(t, customErr.Fields(), 2)
		assert.Equal(t, 1, customErr.Fields()["a"])
		assert.Equal(t, 2, customErr.Fields()["b"])
	})
}

func TestErrorCodeString(t *testing.T) {
	tests := []struct {
		code    ErrorCode
		message string
	}{
		{Unknown, "Unknown"},
		{InvalidInput, "InvalidInput"},
		{ValidationFailed, "ValidationFailed"},
		{ResourceNotFound, "ResourceNotFound"},
		{Timeout, "Timeout"},
		{RateLimitExceeded, "RateLimitExceeded"},
		{LLMGenerationFailed, "LLMGenerationFailed"},
		{TokenLimitExceeded, "TokenLimitExceeded"},
		{InvalidResponse, "InvalidResponse"},
		{WorkflowExecutionFailed, "WorkflowExecutionFailed"},
		{StepExecutionFailed, "StepExecutionFailed"},
		{InvalidWorkflowState, "InvalidWorkflowState"},
		{ErrorCode(999), "ErrorCode(999)"}, // Test unknown code
	}

	for _, tt := range tests {
		t.Run(tt.message, func(t *testing.T) {
			err := New(tt.code, tt.message)
			customErr, ok := err.(*Error)
			require.True(t, ok)
			assert.Equal(t, tt.code, customErr.Code())
			assert.Equal(t, tt.message, customErr.Error())
		})
	}
}

func TestErrorCreation(t *testing.T) {
	t.Run("New error", func(t *testing.T) {
		err := New(ValidationFailed, "validation error")
		customErr, ok := err.(*Error)
		require.True(t, ok)
		assert.Equal(t, ValidationFailed, customErr.Code())
		assert.Equal(t, "validation error", customErr.Error())
		assert.Nil(t, customErr.Unwrap())
	})

	t.Run("With fields", func(t *testing.T) {
		fields := Fields{"key": "value"}
		err := WithFields(
			New(ValidationFailed, "validation error"),
			fields,
		)
		customErr, ok := err.(*Error)
		require.True(t, ok)
		assert.Equal(t, fields["key"], customErr.Fields()["key"])
	})
}
