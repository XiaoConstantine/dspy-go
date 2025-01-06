package workflows

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestErrors(t *testing.T) {
	// Test error variables are properly defined
	t.Run("Error definitions", func(t *testing.T) {
		assert.NotNil(t, ErrStepConditionFailed)
		assert.NotNil(t, ErrStepNotFound)
		assert.NotNil(t, ErrInvalidInput)
		assert.NotNil(t, ErrDuplicateStepID)
		assert.NotNil(t, ErrCyclicDependency)
	})

	// Test error messages are as expected
	t.Run("Error messages", func(t *testing.T) {
		assert.Equal(t, "step condition check failed", ErrStepConditionFailed.Error())
		assert.Equal(t, "step not found in workflow", ErrStepNotFound.Error())
		assert.Equal(t, "invalid input parameters", ErrInvalidInput.Error())
		assert.Equal(t, "duplicate step ID", ErrDuplicateStepID.Error())
		assert.Equal(t, "cyclic dependency detected in workflow", ErrCyclicDependency.Error())
	})

	// Test errors can be used with errors.Is
	t.Run("Error comparison", func(t *testing.T) {
		err := ErrStepConditionFailed
		assert.True(t, errors.Is(err, ErrStepConditionFailed))
		assert.False(t, errors.Is(err, ErrStepNotFound))
	})

	// Test error wrapping
	t.Run("Error wrapping", func(t *testing.T) {
		wrapped := fmt.Errorf("failed to execute step: %w", ErrStepConditionFailed)
		assert.True(t, errors.Is(wrapped, ErrStepConditionFailed))
		assert.Contains(t, wrapped.Error(), "failed to execute step")
	})
}
