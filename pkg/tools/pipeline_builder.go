package tools

import (
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/errors"
)

// PipelineBuilder provides a fluent API for constructing tool pipelines.
type PipelineBuilder struct {
	name     string
	steps    []PipelineStep
	options  PipelineOptions
	registry core.ToolRegistry
	lastErr  error
}

// NewPipelineBuilder creates a new pipeline builder.
func NewPipelineBuilder(name string, registry core.ToolRegistry) *PipelineBuilder {
	return &PipelineBuilder{
		name:     name,
		steps:    make([]PipelineStep, 0),
		registry: registry,
		options: PipelineOptions{
			Timeout:         5 * time.Minute, // Default 5 minute timeout
			FailureStrategy: FailFast,
			Parallel:        false,
			CacheResults:    true,
		},
	}
}

// Step adds a basic step to the pipeline.
func (pb *PipelineBuilder) Step(toolName string) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName: toolName,
		Timeout:  30 * time.Second,
		Retries:  0,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// StepWithTimeout adds a step with a custom timeout.
func (pb *PipelineBuilder) StepWithTimeout(toolName string, timeout time.Duration) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName: toolName,
		Timeout:  timeout,
		Retries:  0,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// StepWithRetries adds a step with retry configuration.
func (pb *PipelineBuilder) StepWithRetries(toolName string, retries int) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName: toolName,
		Timeout:  30 * time.Second,
		Retries:  retries,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// StepWithTransformer adds a step with a data transformer.
func (pb *PipelineBuilder) StepWithTransformer(toolName string, transformer DataTransformer) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName:    toolName,
		Transformer: transformer,
		Timeout:     30 * time.Second,
		Retries:     0,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// ConditionalStep adds a step that executes only if conditions are met.
func (pb *PipelineBuilder) ConditionalStep(toolName string, conditions ...Condition) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName:   toolName,
		Timeout:    30 * time.Second,
		Retries:    0,
		Conditions: conditions,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// AdvancedStep adds a fully configured step.
func (pb *PipelineBuilder) AdvancedStep(toolName string, timeout time.Duration, retries int, transformer DataTransformer, conditions ...Condition) *PipelineBuilder {
	if pb.lastErr != nil {
		return pb
	}

	step := PipelineStep{
		ToolName:    toolName,
		Transformer: transformer,
		Timeout:     timeout,
		Retries:     retries,
		Conditions:  conditions,
	}

	pb.steps = append(pb.steps, step)
	return pb
}

// Timeout sets the overall pipeline timeout.
func (pb *PipelineBuilder) Timeout(timeout time.Duration) *PipelineBuilder {
	pb.options.Timeout = timeout
	return pb
}

// FailFast sets the pipeline to fail on the first error.
func (pb *PipelineBuilder) FailFast() *PipelineBuilder {
	pb.options.FailureStrategy = FailFast
	return pb
}

// ContinueOnError sets the pipeline to continue executing even after errors.
func (pb *PipelineBuilder) ContinueOnError() *PipelineBuilder {
	pb.options.FailureStrategy = ContinueOnError
	return pb
}

// Parallel enables parallel execution of independent steps.
func (pb *PipelineBuilder) Parallel() *PipelineBuilder {
	pb.options.Parallel = true
	return pb
}

// Sequential sets execution to sequential mode.
func (pb *PipelineBuilder) Sequential() *PipelineBuilder {
	pb.options.Parallel = false
	return pb
}

// EnableCaching enables result caching for pipeline steps.
func (pb *PipelineBuilder) EnableCaching() *PipelineBuilder {
	pb.options.CacheResults = true
	return pb
}

// DisableCaching disables result caching for pipeline steps.
func (pb *PipelineBuilder) DisableCaching() *PipelineBuilder {
	pb.options.CacheResults = false
	return pb
}

// Build creates the final pipeline.
func (pb *PipelineBuilder) Build() (*ToolPipeline, error) {
	if pb.lastErr != nil {
		return nil, pb.lastErr
	}

	if len(pb.steps) == 0 {
		return nil, errors.New(errors.InvalidInput, "pipeline must have at least one step")
	}

	// Validate all tools exist in registry
	for _, step := range pb.steps {
		if _, err := pb.registry.Get(step.ToolName); err != nil {
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "tool not found in registry"),
				errors.Fields{"tool_name": step.ToolName},
			)
		}
	}

	pipeline := NewToolPipeline(pb.name, pb.registry, pb.options)

	// Add all steps to the pipeline
	for _, step := range pb.steps {
		if err := pipeline.AddStep(step); err != nil {
			return nil, err
		}
	}

	return pipeline, nil
}

// Common data transformers

// TransformExtractField creates a transformer that extracts a specific field.
func TransformExtractField(fieldName string) DataTransformer {
	return func(input interface{}) (map[string]interface{}, error) {
		if inputMap, ok := input.(map[string]interface{}); ok {
			if value, exists := inputMap[fieldName]; exists {
				return map[string]interface{}{
					"value": value,
				}, nil
			}
			return nil, errors.WithFields(
				errors.New(errors.InvalidInput, "field not found in input"),
				errors.Fields{"field": fieldName},
			)
		}
		return nil, errors.New(errors.InvalidInput, "input is not a map")
	}
}

// TransformRename creates a transformer that renames fields.
func TransformRename(fieldMappings map[string]string) DataTransformer {
	return func(input interface{}) (map[string]interface{}, error) {
		inputMap, ok := input.(map[string]interface{})
		if !ok {
			return nil, errors.New(errors.InvalidInput, "input is not a map")
		}

		result := make(map[string]interface{})

		// Copy all fields, renaming as specified
		for oldName, value := range inputMap {
			if newName, shouldRename := fieldMappings[oldName]; shouldRename {
				result[newName] = value
			} else {
				result[oldName] = value
			}
		}

		return result, nil
	}
}

// TransformFilter creates a transformer that filters fields.
func TransformFilter(allowedFields []string) DataTransformer {
	allowedMap := make(map[string]bool)
	for _, field := range allowedFields {
		allowedMap[field] = true
	}

	return func(input interface{}) (map[string]interface{}, error) {
		inputMap, ok := input.(map[string]interface{})
		if !ok {
			return nil, errors.New(errors.InvalidInput, "input is not a map")
		}

		result := make(map[string]interface{})
		for field, value := range inputMap {
			if allowedMap[field] {
				result[field] = value
			}
		}

		return result, nil
	}
}

// TransformAddConstant creates a transformer that adds constant fields.
func TransformAddConstant(constantFields map[string]interface{}) DataTransformer {
	return func(input interface{}) (map[string]interface{}, error) {
		inputMap, ok := input.(map[string]interface{})
		if !ok {
			return nil, errors.New(errors.InvalidInput, "input is not a map")
		}

		result := make(map[string]interface{})

		// Copy input fields
		for key, value := range inputMap {
			result[key] = value
		}

		// Add constant fields
		for key, value := range constantFields {
			result[key] = value
		}

		return result, nil
	}
}

// TransformChain creates a transformer that applies multiple transformers in sequence.
func TransformChain(transformers ...DataTransformer) DataTransformer {
	return func(input interface{}) (map[string]interface{}, error) {
		current := input
		var err error

		for _, transformer := range transformers {
			current, err = transformer(current)
			if err != nil {
				return nil, err
			}
		}

		if result, ok := current.(map[string]interface{}); ok {
			return result, nil
		}

		return nil, errors.New(errors.Unknown, "transformer chain did not produce map result")
	}
}

// Common condition builders

// ConditionExists creates a condition that checks if a field exists.
func ConditionExists(field string) Condition {
	return Condition{
		Field:    field,
		Operator: "exists",
	}
}

// ConditionEquals creates a condition that checks if a field equals a value.
func ConditionEquals(field string, value interface{}) Condition {
	return Condition{
		Field:    field,
		Operator: "eq",
		Value:    value,
	}
}

// ConditionNotEquals creates a condition that checks if a field does not equal a value.
func ConditionNotEquals(field string, value interface{}) Condition {
	return Condition{
		Field:    field,
		Operator: "ne",
		Value:    value,
	}
}

// ConditionContains creates a condition that checks if a field contains a substring.
func ConditionContains(field string, substring string) Condition {
	return Condition{
		Field:    field,
		Operator: "contains",
		Value:    substring,
	}
}
