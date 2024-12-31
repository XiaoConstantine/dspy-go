package workflows

import "errors"

var (
	// ErrStepConditionFailed indicates a step's condition check failed.
	ErrStepConditionFailed = errors.New("step condition check failed")

	// ErrStepNotFound indicates a referenced step doesn't exist in workflow.
	ErrStepNotFound = errors.New("step not found in workflow")

	// ErrInvalidInput indicates missing or invalid input parameters.
	ErrInvalidInput = errors.New("invalid input parameters")

	// ErrDuplicateStepID indicates attempt to add step with existing ID.
	ErrDuplicateStepID = errors.New("duplicate step ID")

	// ErrCyclicDependency indicates circular dependencies between steps.
	ErrCyclicDependency = errors.New("cyclic dependency detected in workflow")
)
