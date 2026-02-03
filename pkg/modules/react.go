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
// If the loop ends without a "Finish" action (e.g., max iterations reached),
// a fallback Extract module attempts to produce an answer from the gathered trajectory.
type ReAct struct {
	core.BaseModule
	Predict   *Predict
	Extract   *ChainOfThought                // Fallback extraction module for when loop ends without Finish
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

	// Create fallback extraction module
	// This module takes the original inputs + trajectory and produces the original outputs
	extractSignature := createExtractSignature(signature)
	extract := NewChainOfThought(extractSignature)

	react.BaseModule = *core.NewModule(modifiedSignature)
	react.Predict = predict
	react.Extract = extract

	return react
}

// createExtractSignature builds a signature for the fallback extraction module.
// It includes the original input fields plus a trajectory field, and produces the original outputs.
func createExtractSignature(originalSignature core.Signature) core.Signature {
	// Copy original inputs and add trajectory field
	inputs := make([]core.InputField, len(originalSignature.Inputs)+1)
	copy(inputs, originalSignature.Inputs)
	inputs[len(originalSignature.Inputs)] = core.InputField{
		Field: core.NewField("trajectory",
			core.WithDescription("The complete history of thoughts, actions, and observations from the ReAct loop")),
	}

	// Use original outputs (without ReAct's thought/action/observation fields)
	return core.NewSignature(inputs, originalSignature.Outputs).
		WithInstruction("Based on the trajectory of thoughts, actions, and observations above, provide the final answer. " +
			"The trajectory contains all the information gathered during the reasoning process. " +
			"Synthesize this information to produce a complete and accurate response.")
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

// WithNativeFunctionCalling enables native LLM function calling for action selection.
// This bypasses text-based XML parsing entirely by using the LLM's built-in
// function/tool calling capabilities (e.g., OpenAI function calling, Gemini tools).
//
// Benefits:
//   - Eliminates parsing errors and hallucinated observations
//   - Strongly typed tool arguments from the LLM
//   - More reliable tool selection
//
// Requirements:
//   - The LLM must support CapabilityToolCalling
//   - Falls back to text-based parsing if not supported
//
// Usage:
//
//	react := modules.NewReAct(signature, registry, maxIters)
//	react.WithNativeFunctionCalling() // Enable native function calling
func (r *ReAct) WithNativeFunctionCalling() *ReAct {
	config := interceptors.FunctionCallingConfig{
		ToolRegistry:      r.Registry,
		IncludeFinishTool: true,
		FinishToolDescription: "Call this tool when you have completed the task and have the final answer. " +
			"Pass the final answer as the 'answer' argument.",
	}
	interceptor := interceptors.NativeFunctionCallingInterceptor(config)
	r.Predict.SetInterceptors([]core.ModuleInterceptor{interceptor})
	return r
}

// WithNativeFunctionCallingConfig enables native function calling with custom configuration.
func (r *ReAct) WithNativeFunctionCallingConfig(config interceptors.FunctionCallingConfig) *ReAct {
	// Ensure the registry is set
	if config.ToolRegistry == nil {
		config.ToolRegistry = r.Registry
	}
	interceptor := interceptors.NativeFunctionCallingInterceptor(config)
	r.Predict.SetInterceptors([]core.ModuleInterceptor{interceptor})
	return r
}

// SetLLM sets the language model for the base module and all internal modules (Predict and Extract).
func (r *ReAct) SetLLM(llm core.LLM) {
	r.BaseModule.SetLLM(llm)
	r.Predict.SetLLM(llm)
	r.Extract.SetLLM(llm)
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

		// Add conversation context to state for this iteration (always set, even if empty)
		state["conversation_context"] = conversationContext

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
	// Use fallback extraction to produce an answer from the gathered trajectory
	logger.Warn(ctx, "Max iterations (%d) reached without 'Finish' action, using fallback extraction", r.MaxIters)

	return r.extractFromTrajectory(ctx, inputs, conversationContext, opts...)
}

// extractFromTrajectory uses the Extract module to produce an answer from the accumulated trajectory.
// This is called when the ReAct loop ends without a "Finish" action (e.g., max iterations reached).
func (r *ReAct) extractFromTrajectory(ctx context.Context, originalInputs map[string]any, trajectory string, opts ...core.Option) (map[string]any, error) {
	logger := logging.GetLogger()
	ctx, span := core.StartSpan(ctx, "extractFromTrajectory")
	defer core.EndSpan(ctx)

	// Build extraction inputs: original inputs + trajectory
	extractInputs := make(map[string]any)
	for k, v := range originalInputs {
		extractInputs[k] = v
	}
	extractInputs["trajectory"] = trajectory

	logger.Debug(ctx, "Running fallback extraction with trajectory length: %d", len(trajectory))
	span.WithAnnotation("trajectory_length", len(trajectory))

	result, err := r.Extract.Process(ctx, extractInputs, opts...)
	if err != nil {
		span.WithError(err)
		logger.Error(ctx, "Fallback extraction failed: %v", err)
		return nil, fmt.Errorf("fallback extraction failed after max iterations: %w", err)
	}

	logger.Info(ctx, "Fallback extraction completed successfully")
	span.WithAnnotation("extraction_result", result)

	return result, nil
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
// Note: Predict and Extract modules are cloned, but the LLM instance and ToolRegistry are shared.
// Cloning the registry itself might be complex and depends on the registry implementation.
// Sharing the registry is usually acceptable.
func (r *ReAct) Clone() core.Module {
	// Clone Predict module. The type assertion panic is defensive - Clone() always returns
	// the same concrete type, so this should never fail in practice.
	clonedPredict, ok := r.Predict.Clone().(*Predict)
	if !ok {
		panic("Failed to clone Predict module: unexpected type returned from Clone()")
	}

	// Clone Extract module. Same defensive check as above.
	clonedExtract, ok := r.Extract.Clone().(*ChainOfThought)
	if !ok {
		panic("Failed to clone Extract module: unexpected type returned from Clone()")
	}

	return &ReAct{
		BaseModule: *r.BaseModule.Clone().(*core.BaseModule),
		Predict:    clonedPredict,
		Extract:    clonedExtract,
		Registry:   r.Registry,
		MaxIters:   r.MaxIters,
		XMLConfig:  r.XMLConfig, // Share the XML config (it's read-only)
	}
}

// appendReActFields adds the standard ReAct fields (thought, action, observation)
// to the beginning of a signature's output fields, and adds conversation_context as an input field.
func (r *ReAct) appendReActFields(signature core.Signature) core.Signature {
	const reactFormattingInstructions = `
  CRITICAL REASONING RULES:
  1. Provide your reasoning in the 'thought' field
  2. Specify exactly ONE action in the 'action' field (formatted as described in the action field description)
  3. The 'observation' field contains results from previous actions (leave empty on first step)
  4. Use 'answer' field for your final response when complete
  5. ALWAYS include both 'thought' and 'action' fields in every response

  IMPORTANT: Output ONE action at a time and STOP. Wait for the real observation before continuing.
  Do NOT simulate, predict, or hallucinate observations - they will be provided to you after each action.
  Do NOT output multiple thought/action cycles in a single response.
  `
	newSignature := signature

	newSignature.Instruction = reactFormattingInstructions + "\n" + signature.Instruction

	// Add conversation_context as an INPUT field so the LLM can see the history
	historyField := core.InputField{
		Field: core.NewField("conversation_context",
			core.WithDescription("Complete history of previous thoughts, actions, and observations in this conversation. Use this to avoid repeating actions and to build on what you've already learned.")),
	}
	newSignature.Inputs = append(newSignature.Inputs, historyField)

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

	// Extract only the first action block to handle LLM "simulation" behavior
	// where the model outputs multiple thought/action/observation cycles
	actionStr = extractFirstActionBlock(actionStr)

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
	// Check for simple finish action first (early return)
	if isFinishAction(actionStruct) {
		return "finish", make(map[string]interface{}), nil
	}

	// Look for tool_name field in the structured data
	toolNameRaw, hasToolName := actionStruct["tool_name"]
	if !hasToolName {
		return "", nil, fmt.Errorf("unable to extract tool information from structured action: %+v", actionStruct)
	}

	toolName, ok := toolNameRaw.(string)
	if !ok {
		return "", nil, fmt.Errorf("tool_name field is not a string: %T", toolNameRaw)
	}

	argsMap := extractArguments(actionStruct)
	return toolName, argsMap, nil
}

// isFinishAction checks if the action struct represents a finish action.
func isFinishAction(actionStruct map[string]interface{}) bool {
	content, hasContent := actionStruct["content"]
	if !hasContent {
		return false
	}

	contentStr, ok := content.(string)
	if !ok {
		return false
	}

	return strings.ToLower(strings.TrimSpace(contentStr)) == "finish"
}

// extractArguments extracts arguments from the action struct.
func extractArguments(actionStruct map[string]interface{}) map[string]interface{} {
	argsRaw, hasArgs := actionStruct["arguments"]
	if !hasArgs {
		return make(map[string]interface{})
	}

	argsStruct, ok := argsRaw.(map[string]interface{})
	if !ok {
		return make(map[string]interface{})
	}

	// Check for nested "arg" structure
	argsList, hasArg := argsStruct["arg"]
	if !hasArg {
		// Direct argument mapping
		return argsStruct
	}

	// Handle nested argument structure
	switch args := argsList.(type) {
	case map[string]interface{}:
		return extractSingleArg(args)
	case []interface{}:
		return extractArgsFromList(args)
	default:
		return make(map[string]interface{})
	}
}

// extractSingleArg extracts a single argument from a map with "key" and "content" fields.
func extractSingleArg(args map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	key, hasKey := args["key"]
	if !hasKey {
		return result
	}

	keyStr, ok := key.(string)
	if !ok {
		return result
	}

	if content, hasContent := args["content"]; hasContent {
		result[keyStr] = content
	}

	return result
}

// extractArgsFromList extracts multiple arguments from a list of arg maps.
func extractArgsFromList(args []interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	for _, argItem := range args {
		argMap, ok := argItem.(map[string]interface{})
		if !ok {
			continue
		}

		key, hasKey := argMap["key"]
		if !hasKey {
			continue
		}

		keyStr, ok := key.(string)
		if !ok {
			continue
		}

		if content, hasContent := argMap["content"]; hasContent {
			result[keyStr] = content
		}
	}

	return result
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

// extractFirstActionBlock extracts only the first <action>...</action> block from a string.
// This handles the case where LLMs "simulate" multiple thought/action/observation cycles
// in a single response, including hallucinated observations.
//
// The function:
// 1. Truncates at hallucinated observation markers (## observation, Observation:)
// 2. Extracts only the first complete <action>...</action> block
// 3. Returns the original string if no action block is found (for backward compatibility).
func extractFirstActionBlock(input string) string {
	// First, truncate at hallucinated observation markers
	// These indicate the LLM is simulating the conversation
	observationMarkers := []string{
		"## observation",
		"## Observation",
		"Observation:",
		"\nobservation:",
		"\nObservation:",
	}

	truncated := input
	for _, marker := range observationMarkers {
		if idx := strings.Index(strings.ToLower(truncated), strings.ToLower(marker)); idx > 0 {
			// Only truncate if marker appears after some content
			truncated = truncated[:idx]
		}
	}

	// Now extract just the first <action>...</action> block
	actionStart := strings.Index(truncated, "<action")
	if actionStart == -1 {
		// No action tag found, return truncated string as-is
		return strings.TrimSpace(truncated)
	}

	// Find the closing </action> tag
	actionEnd := strings.Index(truncated[actionStart:], "</action>")
	if actionEnd == -1 {
		// No closing tag found, try to find a self-closing tag or return from start
		selfClose := strings.Index(truncated[actionStart:], "/>")
		if selfClose != -1 && selfClose < 100 { // Reasonable limit for self-closing tag
			return strings.TrimSpace(truncated[actionStart : actionStart+selfClose+2])
		}
		// Return from action start to end of truncated string
		return strings.TrimSpace(truncated[actionStart:])
	}

	// Extract the complete first action block
	endPos := actionStart + actionEnd + len("</action>")
	return strings.TrimSpace(truncated[actionStart:endPos])
}
