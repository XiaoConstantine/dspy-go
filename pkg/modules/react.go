package modules

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/logging"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
)

// ReAct implements the ReAct agent loop (Reason, Action, Observation).
// It uses a Predict module to generate thoughts and actions, and executes tools.
type ReAct struct {
	core.BaseModule
	Predict  *Predict
	Tools    []core.Tool
	MaxIters int
	// Internal map for quick tool lookup by name
	toolMap map[string]core.Tool
}

// NewReAct creates a new ReAct module.
// It takes a signature (which it modifies), a list of tools, and max iterations.
func NewReAct(signature core.Signature, tools []core.Tool, maxIters int) *ReAct {
	modifiedSignature := appendReActFields(signature)

	predict := NewPredict(modifiedSignature)

	// Build the tool map for efficient lookup
	toolMap := make(map[string]core.Tool)
	for _, tool := range tools {
		if tool != nil && tool.Metadata() != nil {
			toolMap[tool.Metadata().Name] = tool
		}
	}

	return &ReAct{
		BaseModule: *core.NewModule(modifiedSignature),
		Predict:    predict,
		Tools:      tools,
		MaxIters:   maxIters,
		toolMap:    toolMap,
	}
}

// WithDefaultOptions sets default options by configuring the underlying Predict module.
func (r *ReAct) WithDefaultOptions(opts ...core.Option) *ReAct {
	r.Predict.WithDefaultOptions(opts...)
	return r
}

// SetLLM sets the language model for both the base module and the internal Predict module.
func (r *ReAct) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.Predict.SetLLM(llm)
}

// Process executes the ReAct loop.
func (r *ReAct) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	ctx, span := core.StartSpan(ctx, "ReAct")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()
	span.WithAnnotation("initial_inputs", inputs)

	// Create working state that we'll update through iterations
	state := make(map[string]interface{})
	for k, v := range inputs {
		state[k] = v
	}

	// Ensure observation field exists
	if _, exists := state["observation"]; !exists {
		state["observation"] = ""
	}

	// Track our conversation context
	conversationContext := ""

	for i := 0; i < r.MaxIters; i++ {
		// Debug logging to track iterations explicitly
		logger.Debug(ctx, "*** STARTING REACT ITERATION %d/%d ***", i+1, r.MaxIters)
		logger.Debug(ctx, "Current state: %v", state)

		// Add conversation context to state for this iteration
		if conversationContext != "" {
			state["conversation_context"] = conversationContext
		}

		// --- Predict Step ---
		logger.Debug(ctx, "Sending prediction request in iteration %d", i+1)
		prediction, err := r.Predict.Process(ctx, state, opts...)
		if err != nil {
			logger.Error(ctx, "Prediction error in iteration %d: %v", i+1, err)
			span.WithError(err)
			return nil, fmt.Errorf("error in Predict step (iteration %d): %w", i, err)
		}

		logger.Debug(ctx, "Received prediction in iteration %d: %v", i+1, prediction)

		// Extract and verify action field
		actionField, ok := prediction["action"]
		if !ok {
			logger.Error(ctx, "Missing action field in iteration %d", i+1)
			state["observation"] = "Error: Prediction was missing the 'action' field."

			// Update conversation context for next iteration
			conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: MISSING\nObservation: Error: missing action field\n",
				i+1, prediction["thought"])

			continue
		}

		// Process action based on type
		var parsedToolName string
		var parsedArgsMap map[string]interface{}

		// Parse the action (simplifying for clarity)
		actionStr, isString := actionField.(string)
		if isString {
			// Try to parse as XML action (simplifying error handling)
			var xmlAction tools.XMLAction
			if err := xml.Unmarshal([]byte(actionStr), &xmlAction); err == nil {
				// Check if this is a finish action
				if strings.ToLower(xmlAction.ToolName) == "finish" ||
					strings.ToLower(xmlAction.Content) == "finish" {
					logger.Debug(ctx, "Received FINISH action in iteration %d", i+1)
					return prediction, nil // End with success
				}
				parsedToolName = xmlAction.ToolName
				parsedArgsMap = xmlAction.GetArgumentsMap()
				logger.Debug(ctx, "Parsed tool action in iteration %d: %s with args %v",
					i+1, parsedToolName, parsedArgsMap)
			} else {
				// XML parsing failed
				logger.Error(ctx, "Invalid action format in iteration %d: %s, with error: %v", i+1, actionStr, err)
				state["observation"] = fmt.Sprintf("Error: Invalid action format: %s", actionStr)

				// Update conversation context
				conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %s\nObservation: Error: invalid action format\n",
					i+1, prediction["thought"], actionStr)

				continue
			}
		} else {
			// Not a string action
			logger.Error(ctx, "Action has invalid type in iteration %d: %T", i+1, actionField)
			state["observation"] = fmt.Sprintf("Error: Action has invalid type: %T", actionField)

			// Update conversation context
			conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: [invalid type %T]\nObservation: Error: invalid action type\n",
				i+1, prediction["thought"], actionField)

			continue
		}

		// Execute the tool
		logger.Info(ctx, "Executing tool '%s' in iteration %d", parsedToolName, i+1)
		toolResult, err := r.executeToolByName(ctx, parsedToolName, parsedArgsMap)
		if err != nil {
			logger.Error(ctx, "Tool execution failed in iteration %d: %v", i+1, err)
			errorObservation := fmt.Sprintf("Error executing tool '%s': %s", parsedToolName, err.Error())
			state["observation"] = errorObservation

			// Update conversation context
			conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %s\nObservation: %s\n",
				i+1, prediction["thought"], actionStr, errorObservation)

			continue
		}

		// Tool executed successfully
		observation := formatToolResult(toolResult)
		state["observation"] = observation
		logger.Info(ctx, "Tool executed successfully in iteration %d, observation set", i+1)

		// Update conversation context for next iteration
		conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %s\nObservation: %s\n",
			i+1, prediction["thought"], actionStr, observation)
	}

	// If we get here, we've hit max iterations without finishing
	logger.Warn(ctx, "Max iterations (%d) reached without 'Finish' action", r.MaxIters)
	return state, fmt.Errorf("max iterations reached without 'Finish' action")
}

// executeToolByName finds a tool by its name and executes it.
// Accepts arguments as map[string]interface{} for broad compatibility.
func (r *ReAct) executeToolByName(ctx context.Context, toolName string, arguments map[string]interface{}) (core.ToolResult, error) {
	ctx, span := core.StartSpan(ctx, "executeToolByName")
	defer core.EndSpan(ctx)
	logger := logging.GetLogger()

	span.WithAnnotation("tool_name", toolName)
	span.WithAnnotation("arguments", arguments)
	logger.Debug(ctx, "Attempting to execute tool '%s'", toolName)

	// Find the tool in the map
	matchingTool, found := r.toolMap[toolName]
	if !found {
		err := fmt.Errorf("no tool registered with name: '%s'", toolName)
		span.WithError(err)
		logger.Error(ctx, "%s", err.Error())
		return core.ToolResult{}, err
	}

	// Ensure arguments map is not nil for validation/execution
	argsForExec := arguments
	if argsForExec == nil {
		argsForExec = make(map[string]interface{})
		logger.Debug(ctx, "Arguments map was nil, initialized to empty map for execution.")
	} else {
		// Log args only if not nil initially and potentially filter/truncate sensitive values
		logger.Debug(ctx, "Executing tool '%s' with args: %v", toolName, argsForExec)
	}

	// Validate parameters
	if err := matchingTool.Validate(argsForExec); err != nil {
		// Wrap error for more context
		err = fmt.Errorf("invalid parameters provided for tool '%s': %w. Provided args: %v", toolName, err, argsForExec)
		span.WithError(err)
		logger.Warn(ctx, "%s", err.Error())
		return core.ToolResult{}, err
	}
	logger.Debug(ctx, "Arguments validated for tool '%s'", toolName)

	// Execute tool
	result, err := matchingTool.Execute(ctx, argsForExec)
	if err != nil {
		// Wrap error for more context
		err = fmt.Errorf("tool '%s' execution failed: %w", toolName, err)
		span.WithError(err)
		logger.Error(ctx, "%s", err.Error()) // Log execution error
		return core.ToolResult{}, err
	}
	logger.Info(ctx, "Tool '%s' execution successful.", toolName) // Use Info for successful execution
	span.WithAnnotation("execution_result", result)

	return result, nil
}

// Clone creates a copy of the ReAct module.
// Note: Tools and the Predict module are cloned, but the LLM instance is shared.
func (r *ReAct) Clone() core.Module {
	// Rebuild tool map for the clone
	toolMapClone := make(map[string]core.Tool)
	clonedTools := make([]core.Tool, len(r.Tools))
	for i, tool := range r.Tools {
		// If tools have internal state, they should ideally implement a Clone method.
		// Assuming shallow copy is sufficient for now.
		clonedTools[i] = tool // Shallow copy
		if tool != nil && tool.Metadata() != nil {
			toolMapClone[tool.Metadata().Name] = tool // Point to the same tool instance
		}
	}

	// Ensure Predict module is cloned
	clonedPredict, ok := r.Predict.Clone().(*Predict)
	if !ok {
		panic("Failed to clone Predict module")
	}

	return &ReAct{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		Predict:    clonedPredict,
		Tools:      clonedTools,
		MaxIters:   r.MaxIters,
		toolMap:    toolMapClone,
	}
}

// appendReActFields adds the standard ReAct fields (thought, action, observation)
// to the beginning of a signature's output fields.
func appendReActFields(signature core.Signature) core.Signature {
	const reactFormattingInstructions = `
  CRITICAL FORMATTING RULES:
  1. Format your response with these EXACT field headers, each on a new line:
     thought: [your reasoning]
     action: [your action]
     observation: [result from previous action, if any]
     answer: [your final answer when complete]
  
  2. ALWAYS include both 'thought' and EXACTLY ONE valid 'action' field (formatted as an XML block described in the action field description) in EVERY response. Do NOT output multiple <action> blocks.
  `
	newSignature := signature

	newSignature.Instruction = reactFormattingInstructions + "\n" + signature.Instruction
	// Define standard ReAct output fields
	reactFields := []core.OutputField{
		{Field: core.NewField("thought")},
		{Field: core.NewField("action", core.WithDescription("The action to take. MUST be an XML block like '<action><tool_name>...</tool_name><arguments><arg key=\"...\">...</arg></arguments></action>'. To finish, use '<action><tool_name>Finish</tool_name></action>'. MUST INCLUDE and RETURN ONE ACTION at a time"))},
		{Field: core.NewField("observation", core.WithDescription("The result of the previous action (tool output or error message). Leave empty on first step."))},
	}

	// Prepend ReAct fields to existing output fields
	newSignature.Outputs = append(reactFields, newSignature.Outputs...)

	// Ensure the original signature has an 'answer' field for the final result
	foundAnswer := false
	for _, field := range newSignature.Outputs {
		if field.Name == "answer" {
			foundAnswer = true
			break
		}
	}
	if !foundAnswer {
		// Add a default 'answer' field if not present in the original signature
		newSignature.Outputs = append(newSignature.Outputs, core.OutputField{Field: core.NewField("answer", core.WithDescription("The  final answer to the query."))})
	}

	return newSignature
}

// formatToolResult converts a ToolResult into a string observation suitable for the LLM.
func formatToolResult(result core.ToolResult) string {
	var dataStr string
	if result.Data != nil {
		switch v := result.Data.(type) {
		case []byte:
			// Attempt to unmarshal if it looks like JSON, otherwise return as string
			var jsonData interface{}
			if json.Unmarshal(v, &jsonData) == nil {
				prettyJSON, err := json.MarshalIndent(jsonData, "", "  ")
				if err == nil {
					dataStr = string(prettyJSON)
				} else {
					dataStr = string(v) // Fallback to raw string if re-marshalling fails
				}
			} else {
				dataStr = string(v) // Not JSON, return as string
			}
		case string:
			dataStr = v
		default:
			// Attempt to JSON marshal other types for a structured string representation
			prettyJSON, err := json.MarshalIndent(v, "", "  ")
			if err == nil {
				dataStr = string(prettyJSON)
			} else {
				dataStr = fmt.Sprintf("%+v", v) // Fallback to detailed Go syntax
			}
		}
	} else {
		dataStr = "<empty result>"
	}

	// Basic observation format - Ensure LLM knows this prefix means observation
	observation := fmt.Sprintf("Observation:\n%s", dataStr)

	// Limit observation size? Could truncate dataStr if too long.
	const maxObservationLength = 2000 // Example limit
	if len(observation) > maxObservationLength {
		suffix := "... (truncated)"
		trimmedDataLen := maxObservationLength - len("Observation:\n") - len(suffix) - 1
		if trimmedDataLen < 0 {
			trimmedDataLen = 0
		}
		// Format with precise control over the output
		observation = fmt.Sprintf("Observation:\n%s\n%s",
			dataStr[:trimmedDataLen],
			suffix)
	}

	return observation // Return just the observation string
}
