package modules

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"sort"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
)

// ReAct implements the ReAct agent loop (Reason, Action, Observation).
// It uses a Predict module to generate thoughts and actions, and executes tools.
type ReAct struct {
	core.BaseModule
	Predict   *Predict
	Registry  *tools.InMemoryToolRegistry
	MaxIters  int
	XMLConfig *interceptors.XMLConfig // Optional XML config for enhanced parsing
}

// NewReAct creates a new ReAct module.
// It takes a signature (which it modifies), a tool registry pointer, and max iterations.
func NewReAct(signature core.Signature, registry *tools.InMemoryToolRegistry, maxIters int) *ReAct {
	// Create the ReAct instance first so we can call the instance method
	react := &ReAct{
		Registry: registry,
		MaxIters: maxIters,
	}

	// Now modify the signature using the instance method
	modifiedSignature := react.appendReActFields(signature)

	predict := NewPredict(modifiedSignature).WithTextOutput() // Explicitly use text output for ReAct

	react.BaseModule = *core.NewModule(modifiedSignature)
	react.Predict = predict

	return react
}

// WithDefaultOptions sets default options by configuring the underlying Predict module.
func (r *ReAct) WithDefaultOptions(opts ...core.Option) *ReAct {
	r.Predict.WithDefaultOptions(opts...)
	return r
}

// WithXMLParsing enables XML interceptor-based parsing for tool actions.
// This replaces the hardcoded XML parsing with configurable XML interceptors.
func (r *ReAct) WithXMLParsing(config interceptors.XMLConfig) *ReAct {
	r.XMLConfig = &config
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

		// Process action based on type - handle both structured and string formats
		var parsedToolName string
		var parsedArgsMap map[string]interface{}

		// With XML-by-default, action field can be structured or string
		parsedToolName, parsedArgsMap, err = r.parseActionField(ctx, actionField)
		if err != nil {
			logger.Error(ctx, "Action parsing failed in iteration %d: %v", i+1, err)
			state["observation"] = fmt.Sprintf("Error: Action parsing failed: %s", err.Error())

			// Update conversation context
			conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %v\nObservation: Error: action parsing failed\n",
				i+1, prediction["thought"], actionField)

			continue
		}

		// Check if this is a finish action
		if strings.ToLower(parsedToolName) == "finish" {
			logger.Debug(ctx, "Received FINISH action in iteration %d", i+1)
			return prediction, nil // End with success
		}

		// Execute the tool
		logger.Info(ctx, "Executing tool '%s' in iteration %d", parsedToolName, i+1)
		toolResult, err := r.executeToolByName(ctx, parsedToolName, parsedArgsMap)
		if err != nil {
			logger.Error(ctx, "Tool execution failed in iteration %d: %v", i+1, err)
			errorObservation := fmt.Sprintf("Error executing tool '%s': %s", parsedToolName, err.Error())
			state["observation"] = errorObservation

			// Update conversation context
			conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %v\nObservation: %s\n",
				i+1, prediction["thought"], actionField, errorObservation)

			continue
		}

		// Tool executed successfully
		observation := formatToolResult(toolResult)
		state["observation"] = observation
		logger.Info(ctx, "Tool executed successfully in iteration %d, observation set", i+1)

		// Update conversation context for next iteration
		conversationContext += fmt.Sprintf("\nIteration %d:\nThought: %s\nAction: %v\nObservation: %s\n",
			i+1, prediction["thought"], actionField, observation)
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

	// Find the tool in the registry
	matchingTool, err := r.Registry.Get(toolName)
	if err != nil {
		// Handle error, e.g., tool not found
		span.WithError(err)
		logger.Error(ctx, "Failed to get tool '%s' from registry: %v", toolName, err)
		// Return the registry error directly or wrap it
		return core.ToolResult{}, fmt.Errorf("tool '%s' not found in registry: %w", toolName, err)
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
// Note: Predict module is cloned, but the LLM instance and ToolRegistry are shared.
// Cloning the registry itself might be complex and depends on the registry implementation.
// Sharing the registry is usually acceptable.
func (r *ReAct) Clone() core.Module {
	// Ensure Predict module is cloned
	clonedPredict, ok := r.Predict.Clone().(*Predict)
	if !ok {
		panic("Failed to clone Predict module")
	}

	return &ReAct{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		Predict:    clonedPredict,
		Registry:   r.Registry,
		MaxIters:   r.MaxIters,
		XMLConfig:  r.XMLConfig, // Share the XML config (it's read-only)
	}
}

// appendReActFields adds the standard ReAct fields (thought, action, observation)
// to the beginning of a signature's output fields.
func (r *ReAct) appendReActFields(signature core.Signature) core.Signature {
	const reactFormattingInstructions = `
  CRITICAL REASONING RULES:
  1. Provide your reasoning in the 'thought' field
  2. Specify exactly ONE action in the 'action' field (formatted as described in the action field description)
  3. The 'observation' field contains results from previous actions (leave empty on first step)
  4. Use 'answer' field for your final response when complete
  5. ALWAYS include both 'thought' and 'action' fields in every response
  `
	newSignature := signature

	newSignature.Instruction = reactFormattingInstructions + "\n" + signature.Instruction

	// Build dynamic action description with available tools
	actionDescription := r.buildActionDescription()

	// Define standard ReAct output fields
	reactFields := []core.OutputField{
		{Field: core.NewField("thought")},
		{Field: core.NewField("action", core.WithDescription(actionDescription))},
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

// buildActionDescription creates a dynamic action description including available tools and their parameters.
func (r *ReAct) buildActionDescription() string {
	baseDescription := "The action to take. MUST be an XML block like '<action><tool_name>...</tool_name><arguments><arg key=\"param_name\">value</arg></arguments></action>'. To finish, use '<action><tool_name>Finish</tool_name></action>'. MUST INCLUDE and RETURN ONE ACTION at a time.\n\nAvailable tools:"

	if r.Registry == nil {
		return baseDescription + "\n- No tools available"
	}

	tools := r.Registry.List()
	if len(tools) == 0 {
		return baseDescription + "\n- No tools available"
	}

	// Sort tools by name for consistent output
	sort.Slice(tools, func(i, j int) bool {
		return tools[i].Name() < tools[j].Name()
	})

	var toolDescriptions strings.Builder
	for _, tool := range tools {
		toolDescriptions.WriteString(fmt.Sprintf("\n- %s: %s", tool.Name(), tool.Description()))

		// Get parameter information from InputSchema
		inputSchema := tool.InputSchema()
		if len(inputSchema.Properties) > 0 {
			toolDescriptions.WriteString("\n  Parameters:")

			// Sort parameter names for consistent output
			paramNames := make([]string, 0, len(inputSchema.Properties))
			for paramName := range inputSchema.Properties {
				paramNames = append(paramNames, paramName)
			}
			sort.Strings(paramNames)

			for _, paramName := range paramNames {
				propSchema := inputSchema.Properties[paramName]
				required := ""
				if propSchema.Required {
					required = " (required)"
				}
				description := "No description available"
				if propSchema.Description != "" {
					description = propSchema.Description
				}
				toolDescriptions.WriteString(fmt.Sprintf("\n    - %s: %s%s", paramName, description, required))
			}
		}
	}

	return baseDescription + toolDescriptions.String()
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

// parseActionField handles action parsing for both structured XML data and string data.
// This supports XML-by-default parsing where the action field might be structured.
func (r *ReAct) parseActionField(ctx context.Context, actionField interface{}) (string, map[string]interface{}, error) {
	// Handle different action field types based on XML parsing results
	switch action := actionField.(type) {
	case string:
		// String format - parse as XML string (legacy format)
		return r.parseActionString(ctx, action)

	case map[string]interface{}:
		// Structured format from XML parsing - extract tool info directly
		// Note: This case is currently not reached in default usage since ReAct uses WithTextOutput().
		// It's preserved for future extensibility when users might enable XML output after initialization.
		return r.parseActionStruct(ctx, action)

	default:
		return "", nil, fmt.Errorf("unsupported action field type: %T", actionField)
	}
}

// parseActionString handles string-based action parsing (legacy format).
func (r *ReAct) parseActionString(ctx context.Context, actionStr string) (string, map[string]interface{}, error) {
	// Check for simple finish command first (XML-by-default case)
	actionStr = strings.TrimSpace(actionStr)
	if strings.ToLower(actionStr) == "finish" {
		return "finish", make(map[string]interface{}), nil
	}

	if r.XMLConfig != nil {
		// Use XML interceptors for enhanced parsing
		return r.parseActionWithInterceptors(ctx, actionStr)
	}

	// Fallback to original hardcoded XML parsing for backward compatibility
	var xmlAction tools.XMLAction
	if err := xml.Unmarshal([]byte(actionStr), &xmlAction); err != nil {
		return "", nil, fmt.Errorf("XML parsing failed: %w", err)
	}

	// Handle finish actions that might use Content field (original behavior)
	toolName := xmlAction.ToolName
	if strings.ToLower(xmlAction.ToolName) == "finish" ||
		strings.ToLower(strings.TrimSpace(xmlAction.Content)) == "finish" {
		toolName = "finish"
	}

	argsMap := xmlAction.GetArgumentsMap()
	return toolName, argsMap, nil
}

// parseActionStruct handles structured action parsing from XML-by-default parsing.
func (r *ReAct) parseActionStruct(ctx context.Context, actionStruct map[string]interface{}) (string, map[string]interface{}, error) {
	// Look for tool_name field in the structured data
	toolNameRaw, hasToolName := actionStruct["tool_name"]
	if hasToolName {
		toolName, ok := toolNameRaw.(string)
		if !ok {
			return "", nil, fmt.Errorf("tool_name field is not a string: %T", toolNameRaw)
		}

		// Extract arguments if present
		argsMap := make(map[string]interface{})
		if argsRaw, hasArgs := actionStruct["arguments"]; hasArgs {
			if argsStruct, ok := argsRaw.(map[string]interface{}); ok {
				// Handle nested argument structure
				if argsList, hasArg := argsStruct["arg"]; hasArg {
					// Handle both single arg and multiple args
					switch args := argsList.(type) {
					case map[string]interface{}:
						// Single argument
						if key, hasKey := args["key"]; hasKey {
							if keyStr, ok := key.(string); ok {
								if content, hasContent := args["content"]; hasContent {
									argsMap[keyStr] = content
								}
							}
						}
					case []interface{}:
						// Multiple arguments
						for _, argItem := range args {
							if argMap, ok := argItem.(map[string]interface{}); ok {
								if key, hasKey := argMap["key"]; hasKey {
									if keyStr, ok := key.(string); ok {
										if content, hasContent := argMap["content"]; hasContent {
											argsMap[keyStr] = content
										}
									}
								}
							}
						}
					}
				} else {
					// Direct argument mapping
					argsMap = argsStruct
				}
			}
		}

		return toolName, argsMap, nil
	}

	// Check if this is a simple finish action or direct tool name
	if content, hasContent := actionStruct["content"]; hasContent {
		if contentStr, ok := content.(string); ok {
			if strings.ToLower(strings.TrimSpace(contentStr)) == "finish" {
				return "finish", make(map[string]interface{}), nil
			}
		}
	}

	return "", nil, fmt.Errorf("unable to extract tool information from structured action: %+v", actionStruct)
}

// parseActionWithInterceptors uses the centralized XML parser to parse action strings.
// This provides enhanced error handling, validation, security features, and consistency
// with the main XML interceptor implementation.
func (r *ReAct) parseActionWithInterceptors(ctx context.Context, actionStr string) (string, map[string]interface{}, error) {
	// Use the centralized ParseXMLAction function from the interceptors package
	// This ensures consistency with the main XML parsing logic and avoids duplication
	return interceptors.ParseXMLAction(actionStr, *r.XMLConfig)
}

// NewTypedReAct creates a new type-safe ReAct module from a typed signature.
// Typed modules use text-based parsing by default since they typically rely on prefixes.
func NewTypedReAct[TInput, TOutput any](registry *tools.InMemoryToolRegistry, maxIters int) *ReAct {
	typedSig := core.NewTypedSignatureCached[TInput, TOutput]()
	legacySig := typedSig.ToLegacySignature()

	react := NewReAct(legacySig, registry, maxIters)
	// Use clearer variable names for type display
	var i TInput
	var o TOutput
	react.DisplayName = fmt.Sprintf("TypedReAct[%T,%T]", i, o)

	return react
}
