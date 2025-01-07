package workflows

import stderror "errors"
import "github.com/XiaoConstantine/dspy-go/pkg/errors"

var (
	// ErrStepConditionFailed indicates a step's condition check failed.
	ErrStepConditionFailed = stderror.New("step condition check failed")

	// ErrStepNotFound indicates a referenced step doesn't exist in workflow.
	ErrStepNotFound = stderror.New("step not found in workflow")

	// ErrInvalidInput indicates missing or invalid input parameters.
	ErrInvalidInput = stderror.New("invalid input parameters")

	// ErrDuplicateStepID indicates attempt to add step with existing ID.
	ErrDuplicateStepID = stderror.New("duplicate step ID")

	// ErrCyclicDependency indicates circular dependencies between steps.
	ErrCyclicDependency = stderror.New("cyclic dependency detected in workflow")
)

func WrapWorkflowError(err error, fields map[string]interface{}) error {
	if err == nil {
		return nil
	}

	// Map standard workflow errors to specific ErrorCodes (if needed)
	var code errors.ErrorCode
	switch err {
	case ErrStepConditionFailed:
		code = errors.InvalidWorkflowState
	case ErrStepNotFound:
		code = errors.ResourceNotFound
	case ErrInvalidInput:
		code = errors.InvalidInput
	case ErrDuplicateStepID:
		code = errors.ValidationFailed
	case ErrCyclicDependency:
		code = errors.WorkflowExecutionFailed
	default:
		code = errors.Unknown
	}

	return errors.WithFields(errors.Wrap(err, code, err.Error()), fields)
}
