package workflows

import "github.com/XiaoConstantine/dspy-go/pkg/errors"

var (
	// ErrStepConditionFailed indicates a step's condition check failed.
	ErrStepConditionFailed = errors.New(errors.InvalidWorkflowState, "step condition check failed")

	// ErrStepNotFound indicates a referenced step doesn't exist in workflow.
	ErrStepNotFound = errors.New(errors.ResourceNotFound, "step not found in workflow")

	// ErrInvalidInput indicates missing or invalid input parameters.
	ErrInvalidInput = errors.New(errors.InvalidInput, "invalid input parameters")

	// ErrDuplicateStepID indicates attempt to add step with existing ID.
	ErrDuplicateStepID = errors.New(errors.ValidationFailed, "duplicate step ID")

	// ErrCyclicDependency indicates circular dependencies between steps.
	ErrCyclicDependency = errors.New(errors.WorkflowExecutionFailed, "cyclic dependency detected in workflow")
)

func WrapWorkflowError(err error, fields map[string]interface{}) error {
	if err == nil {
		return nil
	}
	return errors.WithFields(err, fields)
}
