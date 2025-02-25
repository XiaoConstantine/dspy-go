package workflows

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// Step represents a single unit of computation in a workflow. Each step wraps a DSPy
// module and adds workflow-specific metadata and control logic.
type Step struct {
	// ID uniquely identifies this step within the workflow
	ID string

	// Module is the underlying DSPy module that performs the actual computation
	// This could be a Predict, ChainOfThought, or any other DSPy module type
	Module core.Module

	// NextSteps contains the IDs of steps that should execute after this one
	// This allows us to define branching and conditional execution paths
	NextSteps []string

	// Condition is an optional function that determines if this step should execute
	// It can examine the current workflow state to make this decision
	Condition func(state map[string]interface{}) bool

	// RetryConfig specifies how to handle failures of this step
	RetryConfig *RetryConfig
}

// RetryConfig defines how to handle step failures.
type RetryConfig struct {
	// MaxAttempts is the maximum number of retry attempts
	MaxAttempts int

	// BackoffMultiplier determines how long to wait between retries
	BackoffMultiplier float64
}

// StepResult holds the outputs and metadata from executing a step.
type StepResult struct {
	// StepID identifies which step produced this result
	StepID string

	// Outputs contains the data produced by this step
	Outputs map[string]interface{}

	// Metadata contains additional information about the execution
	Metadata map[string]interface{}

	// NextSteps indicates which steps should run next (may be modified by step execution)
	NextSteps []string
}

// Execute runs the step's DSPy module with the provided inputs.
func (s *Step) Execute(ctx context.Context, inputs map[string]interface{}) (*StepResult, error) {
	signature := s.Module.GetSignature()

	// First validate that we have all required inputs
	if err := s.validateInputs(inputs, signature); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "input validation failed"),
			errors.Fields{
				"step_id":    s.ID,
				"input_keys": getKeysAsArray(inputs),
			})
	}
	// Check if step's condition allows execution
	if s.Condition != nil && !s.Condition(inputs) {
		return nil, errors.WithFields(
			ErrStepConditionFailed,
			errors.Fields{
				"step_id": s.ID,
			})
	}

	var result *StepResult
	var err error

	// Apply retry logic if configured
	if s.RetryConfig != nil {
		result, err = s.executeWithRetry(ctx, inputs)
	} else {
		result, err = s.executeOnce(ctx, inputs)
	}

	if err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.StepExecutionFailed, fmt.Sprintf("step %s execution failed", s.ID)),
			errors.Fields{
				"step_id":          s.ID,
				"retry_configured": s.RetryConfig != nil,
			})
	}

	// Validate outputs match expected fields
	if err := s.validateOutputs(result.Outputs, signature); err != nil {
		return nil, errors.WithFields(
			errors.Wrap(err, errors.ValidationFailed, "output validation failed"),
			errors.Fields{
				"step_id":     s.ID,
				"output_keys": getKeysAsArray(result.Outputs),
			})
	}
	return result, nil
}

// executeOnce performs a single execution attempt of the step.
func (s *Step) executeOnce(ctx context.Context, inputs map[string]interface{}) (*StepResult, error) {
	if err := checkCtxErr(ctx); err != nil {
		return nil, err
	}
	// Execute the underlying DSPy module
	outputs, err := s.Module.Process(ctx, inputs)

	if err != nil {
		return nil, err
	}
	if err := checkCtxErr(ctx); err != nil {
		return nil, err
	}
	return &StepResult{
		StepID:    s.ID,
		Outputs:   outputs,
		NextSteps: s.NextSteps,
		Metadata:  make(map[string]interface{}),
	}, nil
}

// executeWithRetry implements retry logic for the step.
func (s *Step) executeWithRetry(ctx context.Context, inputs map[string]interface{}) (*StepResult, error) {
	var lastErr error
	for attempt := 0; attempt < s.RetryConfig.MaxAttempts; attempt++ {

		if err := checkCtxErr(ctx); err != nil {
			return nil, errors.Wrap(err, errors.Canceled, "context canceled before executing step")
		}
		result, err := s.executeOnce(ctx, inputs)
		if err == nil {
			return result, nil
		}
		lastErr = err

		// Wait before retrying, with exponential backoff
		backoffDuration := time.Duration(float64(time.Second) *
			math.Pow(s.RetryConfig.BackoffMultiplier, float64(attempt)))
		select {
		case <-ctx.Done():
			return nil, errors.Wrap(ctx.Err(), errors.Canceled, "context canceled during retry backoff")
		case <-time.After(backoffDuration):
			continue
		}
	}
	return nil, errors.WithFields(
		errors.Wrap(lastErr, errors.StepExecutionFailed, "max retry attempts reached"),
		errors.Fields{
			"step_id":      s.ID,
			"max_attempts": s.RetryConfig.MaxAttempts,
		})
}

// validateInputs checks if all required input fields are present.
func (s *Step) validateInputs(inputs map[string]interface{}, signature core.Signature) error {
	for _, field := range signature.Inputs {
		if _, ok := inputs[field.Name]; !ok {
			return errors.WithFields(
				errors.New(errors.InvalidInput, fmt.Sprintf("missing required input field: %s", field.Name)),
				errors.Fields{
					"field_name": field.Name,
					"step_id":    s.ID,
				})
		}
	}
	return nil
}

// validateOutputs checks if all expected output fields are present.
func (s *Step) validateOutputs(outputs map[string]interface{}, signature core.Signature) error {
	for _, field := range signature.Outputs {
		if _, ok := outputs[field.Name]; !ok {
			return errors.WithFields(
				errors.New(errors.InvalidResponse, fmt.Sprintf("missing required output field: %s", field.Name)),
				errors.Fields{
					"field_name":        field.Name,
					"step_id":           s.ID,
					"available_outputs": getKeysAsArray(outputs),
				})
		}
	}
	return nil
}

func checkCtxErr(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func getKeysAsArray(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
