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

// CustomError is a test error type that's not our Error type.
type CustomError struct {
	msg string
}

func (c *CustomError) Error() string {
	return c.msg
}

// TestErrorAsMethod tests the As() method which currently has 0% coverage.
func TestErrorAsMethod(t *testing.T) {
	t.Run("As method with correct target type", func(t *testing.T) {
		err := New(ValidationFailed, "validation error")
		var customErr *Error

		// Test successful As() call
		assert.True(t, stderrors.As(err, &customErr))
		assert.NotNil(t, customErr)
		assert.Equal(t, ValidationFailed, customErr.Code())
		assert.Equal(t, "validation error", customErr.Error())
	})

	t.Run("As method with incorrect target type", func(t *testing.T) {
		err := New(ValidationFailed, "validation error")
		var wrongType *CustomError

		// Test failed As() call with wrong type
		assert.False(t, stderrors.As(err, &wrongType))
		assert.Nil(t, wrongType)
	})

	t.Run("As method with non-pointer target", func(t *testing.T) {
		err := New(ValidationFailed, "validation error")
		customErr := err.(*Error)

		// Test As() directly on Error instance with non-pointer
		var wrongType string
		assert.False(t, customErr.As(wrongType))
	})

	t.Run("As method with wrapped error", func(t *testing.T) {
		baseErr := stderrors.New("base error")
		wrappedErr := Wrap(baseErr, ValidationFailed, "wrapped")

		var customErr *Error
		assert.True(t, stderrors.As(wrappedErr, &customErr))
		assert.Equal(t, ValidationFailed, customErr.Code())
		assert.Equal(t, "wrapped", customErr.message)
	})
}

// TestErrorStringEdgeCases tests edge cases in the Error() method to improve coverage.
func TestErrorStringEdgeCases(t *testing.T) {
	t.Run("Error with empty fields map", func(t *testing.T) {
		err := &Error{
			code:     ValidationFailed,
			message:  "test message",
			original: nil,
			fields:   Fields{}, // Empty but not nil
		}

		result := err.Error()
		assert.Equal(t, "test message", result)
		assert.NotContains(t, result, "[")
		assert.NotContains(t, result, "]")
	})

	t.Run("Error with nil fields", func(t *testing.T) {
		err := &Error{
			code:     ValidationFailed,
			message:  "test message",
			original: nil,
			fields:   nil, // Nil fields
		}

		result := err.Error()
		assert.Equal(t, "test message", result)
	})

	t.Run("Error with fields and no original error", func(t *testing.T) {
		err := &Error{
			code:    ValidationFailed,
			message: "test message",
			fields: Fields{
				"key1": "value1",
				"key2": 42,
			},
		}

		result := err.Error()
		assert.Contains(t, result, "test message")
		assert.Contains(t, result, "[")
		assert.Contains(t, result, "]")
		assert.Contains(t, result, "key1=value1")
		assert.Contains(t, result, "key2=42")
	})

	t.Run("Error with fields and original error", func(t *testing.T) {
		originalErr := stderrors.New("original error")
		err := &Error{
			code:     ValidationFailed,
			message:  "test message",
			original: originalErr,
			fields: Fields{
				"context": "test context",
			},
		}

		result := err.Error()
		assert.Contains(t, result, "test message")
		assert.Contains(t, result, ": original error")
		assert.Contains(t, result, "[")
		assert.Contains(t, result, "context=test context")
	})

	t.Run("Error with multiple fields formatting", func(t *testing.T) {
		err := &Error{
			code:    ValidationFailed,
			message: "test",
			fields: Fields{
				"string": "value",
				"int":    123,
				"bool":   true,
				"float":  3.14,
			},
		}

		result := err.Error()
		assert.Contains(t, result, "test")
		assert.Contains(t, result, "string=value")
		assert.Contains(t, result, "int=123")
		assert.Contains(t, result, "bool=true")
		assert.Contains(t, result, "float=3.14")
	})
}

// TestWithFieldsEdgeCases tests edge cases in WithFields to improve coverage.
func TestWithFieldsEdgeCases(t *testing.T) {
	t.Run("WithFields on nil error", func(t *testing.T) {
		result := WithFields(nil, Fields{"key": "value"})
		assert.Nil(t, result)
	})

	t.Run("WithFields on non-Error type", func(t *testing.T) {
		baseErr := stderrors.New("base error")
		fields := Fields{"context": "test"}

		result := WithFields(baseErr, fields)
		assert.NotNil(t, result)

		customErr, ok := result.(*Error)
		require.True(t, ok)
		assert.Equal(t, Unknown, customErr.Code())
		assert.Equal(t, "base error", customErr.message)
		assert.Equal(t, baseErr, customErr.original)
		assert.Equal(t, "test", customErr.Fields()["context"])
	})

	t.Run("WithFields on Error with nil fields", func(t *testing.T) {
		err := &Error{
			code:    ValidationFailed,
			message: "test",
			fields:  nil,
		}

		newFields := Fields{"new": "value"}
		result := WithFields(err, newFields)

		customErr, ok := result.(*Error)
		require.True(t, ok)
		assert.Equal(t, "value", customErr.Fields()["new"])
	})

	t.Run("WithFields field overwriting", func(t *testing.T) {
		err := WithFields(
			New(ValidationFailed, "test"),
			Fields{"key": "original", "other": "value"},
		)

		// Add fields with overlapping key
		result := WithFields(err, Fields{"key": "overwritten", "new": "added"})

		customErr, ok := result.(*Error)
		require.True(t, ok)
		fields := customErr.Fields()
		assert.Equal(t, "overwritten", fields["key"])
		assert.Equal(t, "value", fields["other"])
		assert.Equal(t, "added", fields["new"])
	})
}

// TestErrorIsEdgeCases tests edge cases in the Is() method to improve coverage.
func TestErrorIsEdgeCases(t *testing.T) {
	t.Run("Is method with non-Error target", func(t *testing.T) {
		err := New(ValidationFailed, "test")
		baseErr := stderrors.New("base error")

		customErr := err.(*Error)
		assert.False(t, customErr.Is(baseErr))
	})

	t.Run("Is method with nil target", func(t *testing.T) {
		err := New(ValidationFailed, "test")
		customErr := err.(*Error)
		assert.False(t, customErr.Is(nil))
	})

	t.Run("Is method with same instance", func(t *testing.T) {
		err := New(ValidationFailed, "test")
		customErr := err.(*Error)
		assert.True(t, customErr.Is(customErr))
	})
}

// TestAllErrorCodes tests error codes that might not be covered.
func TestAllErrorCodes(t *testing.T) {
	// Test all error codes to ensure they're covered
	testCases := []struct {
		code ErrorCode
		name string
	}{
		{Unknown, "Unknown"},
		{InvalidInput, "InvalidInput"},
		{ValidationFailed, "ValidationFailed"},
		{ResourceNotFound, "ResourceNotFound"},
		{Timeout, "Timeout"},
		{RateLimitExceeded, "RateLimitExceeded"},
		{Canceled, "Canceled"},
		{ResourceExhausted, "ResourceExhausted"},
		{LLMGenerationFailed, "LLMGenerationFailed"},
		{TokenLimitExceeded, "TokenLimitExceeded"},
		{InvalidResponse, "InvalidResponse"},
		{ModelNotSupported, "ModelNotSupported"},
		{ProviderNotFound, "ProviderNotFound"},
		{ConfigurationError, "ConfigurationError"},
		{WorkflowExecutionFailed, "WorkflowExecutionFailed"},
		{StepExecutionFailed, "StepExecutionFailed"},
		{InvalidWorkflowState, "InvalidWorkflowState"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := New(tc.code, "test error")
			customErr, ok := err.(*Error)
			require.True(t, ok)
			assert.Equal(t, tc.code, customErr.Code())
		})
	}
}

// TestFieldsMethodEdgeCases tests edge cases in Fields() method.
func TestFieldsMethodEdgeCases(t *testing.T) {
	t.Run("Fields method with nil fields", func(t *testing.T) {
		err := &Error{
			code:    ValidationFailed,
			message: "test",
			fields:  nil,
		}

		fields := err.Fields()
		assert.NotNil(t, fields)
		assert.Empty(t, fields)
	})

	t.Run("Fields method returns copy not reference", func(t *testing.T) {
		originalFields := Fields{"key": "original"}
		err := &Error{
			code:    ValidationFailed,
			message: "test",
			fields:  originalFields,
		}

		returnedFields := err.Fields()
		returnedFields["key"] = "modified"

		// Original should not be modified
		assert.Equal(t, "original", originalFields["key"])
		assert.Equal(t, "original", err.fields["key"])
	})
}

// TestErrorChainIntegration tests complex error chains.
func TestErrorChainIntegration(t *testing.T) {
	t.Run("Deep error chain with fields", func(t *testing.T) {
		// Create a deep chain of errors
		baseErr := stderrors.New("database connection failed")

		level1 := Wrap(baseErr, ResourceNotFound, "user not found")
		level1 = WithFields(level1, Fields{"user_id": 123})

		level2 := Wrap(level1, ValidationFailed, "validation failed")
		level2 = WithFields(level2, Fields{"field": "email"})

		level3 := Wrap(level2, InvalidInput, "invalid request")
		level3 = WithFields(level3, Fields{"request_id": "abc123"})

		// Test the final error
		finalErr := level3.(*Error)
		assert.Equal(t, InvalidInput, finalErr.Code())
		assert.Contains(t, finalErr.Error(), "invalid request")
		assert.Contains(t, finalErr.Error(), "validation failed")
		assert.Contains(t, finalErr.Error(), "user not found")
		assert.Contains(t, finalErr.Error(), "database connection failed")
		assert.Contains(t, finalErr.Error(), "request_id=abc123")

		// Test unwrapping
		unwrapped := finalErr.Unwrap().(*Error)
		assert.Equal(t, ValidationFailed, unwrapped.Code())
		assert.Contains(t, unwrapped.Error(), "field=email")
		assert.Contains(t, unwrapped.Fields()["field"], "email")
	})
}
