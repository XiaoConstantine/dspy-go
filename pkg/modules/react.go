package modules

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

type ReAct struct {
	core.BaseModule
	Predict  *Predict
	Tools    []core.Tool
	MaxIters int
}

func NewReAct(signature core.Signature, tools []core.Tool, maxIters int) *ReAct {
	modifiedSignature := appendReActFields(signature)
	predict := NewPredict(modifiedSignature)

	return &ReAct{
		BaseModule: *core.NewModule(modifiedSignature),
		Predict:    predict,
		Tools:      tools,
		MaxIters:   maxIters,
	}
}

// WithDefaultOptions sets default options by configuring the underlying Predict module.
func (r *ReAct) WithDefaultOptions(opts ...core.Option) *ReAct {
	// Simply delegate to the Predict module's WithDefaultOptions
	r.Predict.WithDefaultOptions(opts...)
	return r
}

func (r *ReAct) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.Predict.SetLLM(llm)
}

func (r *ReAct) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	ctx, span := core.StartSpan(ctx, "ReAct")
	defer core.EndSpan(ctx)

	span.WithAnnotation("inputs", inputs)

	for i := 0; i < r.MaxIters; i++ {
		prediction, err := r.Predict.Process(ctx, inputs, opts...)
		if err != nil {
			return nil, err
		}

		action, ok := prediction["action"].(string)
		if !ok {
			err := errors.New("invalid action in prediction")
			span.WithError(err)
			return nil, err
		}

		if action == "Finish" {
			span.WithAnnotation("prediction", prediction)
			return prediction, nil
		}

		toolResult, err := r.executeMatchingTool(ctx, action, inputs)
		if err != nil {
			span.WithError(err)
			return nil, err
		}
		if toolResult != nil {
			// Convert tool result to string observation for the model
			observation := formatToolResult(*toolResult)
			inputs["observation"] = observation
		}
	}
	err := errors.New("max iterations reached")
	span.WithError(err)

	return nil, err
}

// executeMatchingTool finds and executes the appropriate tool for the given action.
func (r *ReAct) executeMatchingTool(ctx context.Context, action string, inputs map[string]interface{}) (*core.ToolResult, error) {
	logger := logging.GetLogger()
	logger.Debug(ctx, "action str: %s", action)

	// Find matching tool
	var matchingTool core.Tool
	var matchScore float64
	for _, tool := range r.Tools {

		metadata := tool.Metadata()
		if tool.CanHandle(ctx, action) {
			currentScore := calculateToolMatchScore(metadata, action)
			logger.Debug(ctx, "get score: %f for tool: %v", currentScore, tool)

			if currentScore > matchScore {
				matchingTool = tool
				matchScore = currentScore
			}
		}
	}

	if matchingTool == nil {
		return nil, fmt.Errorf("no tool found for action: %s", action)
	}

	// Prepare tool parameters
	params := extractToolParams(action, inputs, matchingTool.Metadata())

	// Validate parameters
	if err := matchingTool.Validate(params); err != nil {
		return nil, fmt.Errorf("invalid tool parameters: %w", err)
	}

	// Execute tool
	result, err := matchingTool.Execute(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}

	return &result, nil
}

func (r *ReAct) Clone() core.Module {
	return &ReAct{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		Predict:    r.Predict.Clone().(*Predict),
		Tools:      r.Tools, // Note: This is a shallow copy of the tools
		MaxIters:   r.MaxIters,
	}
}

func appendReActFields(signature core.Signature) core.Signature {
	newSignature := signature
	newFields := []core.OutputField{
		{Field: core.NewField("thought")},
		{Field: core.NewField("action")},
		{Field: core.NewField("observation")},
	}
	newSignature.Outputs = append(newFields, newSignature.Outputs...)
	return newSignature
}

// formatToolResult converts a ToolResult into a string observation for the model.
func formatToolResult(result core.ToolResult) string {
	// Format the result data
	observation := fmt.Sprintf("Result: %v", result.Data)

	// Add relevant metadata and annotations
	if len(result.Metadata) > 0 {
		observation += fmt.Sprintf("\nMetadata: %v", result.Metadata)
	}
	if len(result.Annotations) > 0 {
		observation += fmt.Sprintf("\nAnnotations: %v", result.Annotations)
	}

	return observation
}

// extractToolParams extracts tool parameters from the action string and current inputs.
func extractToolParams(action string, inputs map[string]interface{}, toolMetadata *core.ToolMetadata) map[string]interface{} {
	params := make(map[string]interface{})

	// Always include the action
	params["action"] = action

	// Copy inputs that match the tool's input schema
	if toolMetadata != nil && toolMetadata.InputSchema != nil {
		for paramName := range toolMetadata.InputSchema {
			if value, exists := inputs[paramName]; exists {
				// We could add type validation here based on paramType
				// For now, just copy the value if it exists
				params[paramName] = value
			}
		}
	} else {
		// If no schema is available, include all inputs
		for key, value := range inputs {
			params[key] = value
		}
	}

	return params
}

// calculateToolMatchScore determines how well a tool matches an action.
func calculateToolMatchScore(metadata *core.ToolMetadata, action string) float64 {
	score := 0.1

	// Check if action matches tool name
	if strings.Contains(strings.ToLower(action), strings.ToLower(metadata.Name)) {
		score += 0.5
	}

	// Check capabilities
	for _, capability := range metadata.Capabilities {
		if strings.Contains(strings.ToLower(action), strings.ToLower(capability)) {
			score += 0.3
		}
	}

	return score
}
