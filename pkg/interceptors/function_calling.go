// Package interceptors provides middleware components for dspy-go modules and agents.
package interceptors

import (
	"context"
	"fmt"
	"slices"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
)

// FunctionCallingConfig configures the native function calling interceptor.
type FunctionCallingConfig struct {
	// ToolRegistry provides the available tools for function calling.
	ToolRegistry *tools.InMemoryToolRegistry

	// StrictMode requires the LLM to always call a function (no free-form text).
	// When false, the LLM may respond with text instead of a function call.
	StrictMode bool

	// IncludeFinishTool adds a special "Finish" tool that signals task completion.
	// This is essential for ReAct loops to know when to stop.
	IncludeFinishTool bool

	// FinishToolDescription customizes the Finish tool's description.
	// Defaults to a standard description if empty.
	FinishToolDescription string
}

// DefaultFunctionCallingConfig returns sensible defaults for ReAct usage.
func DefaultFunctionCallingConfig() FunctionCallingConfig {
	return FunctionCallingConfig{
		StrictMode:            false,
		IncludeFinishTool:     true,
		FinishToolDescription: "Call this tool when you have completed the task and have the final answer. Pass the final answer as the 'answer' argument.",
	}
}

// NativeFunctionCallingInterceptor creates a module interceptor that uses native LLM
// function calling instead of text-based action parsing.
//
// This interceptor transforms the ReAct pattern by:
// 1. Converting tools from the registry into LLM function schemas
// 2. Using GenerateWithFunctions instead of Generate
// 3. Returning structured tool calls that bypass XML/text parsing
//
// Benefits over text-based parsing:
// - No hallucinated observations (LLM only returns one tool call)
// - Strongly typed arguments from the LLM
// - Provider-optimized tool selection
// - Eliminates parsing errors and ambiguity
//
// Usage:
//
//	config := interceptors.FunctionCallingConfig{
//	    ToolRegistry:      registry,
//	    IncludeFinishTool: true,
//	}
//	interceptor := interceptors.NativeFunctionCallingInterceptor(config)
//	react.Predict.SetInterceptors([]core.ModuleInterceptor{interceptor})
func NativeFunctionCallingInterceptor(config FunctionCallingConfig) core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		logger := logging.GetLogger()

		// Check if the LLM supports function calling
		llm := core.GlobalConfig.DefaultLLM
		if llm == nil {
			// Fall back to normal handler if no LLM configured
			logger.Debug(ctx, "No LLM configured, falling back to text-based parsing")
			return handler(ctx, inputs, opts...)
		}

		if !supportsToolCalling(llm) {
			// Fall back to normal handler if LLM doesn't support function calling
			logger.Debug(ctx, "LLM %s does not support function calling, falling back to text-based parsing", llm.ProviderName())
			return handler(ctx, inputs, opts...)
		}

		// Build function schemas from tool registry
		functions, err := buildFunctionSchemas(config)
		if err != nil {
			logger.Warn(ctx, "Failed to build function schemas: %v, falling back to text-based parsing", err)
			return handler(ctx, inputs, opts...)
		}

		if len(functions) == 0 {
			logger.Debug(ctx, "No functions available, falling back to text-based parsing")
			return handler(ctx, inputs, opts...)
		}

		// Build the prompt from inputs
		prompt := buildPromptFromInputs(inputs, info)

		logger.Debug(ctx, "Using native function calling with %d functions", len(functions))

		// Call LLM with function calling
		result, err := llm.GenerateWithFunctions(ctx, prompt, functions)
		if err != nil {
			logger.Error(ctx, "GenerateWithFunctions failed: %v", err)
			return nil, fmt.Errorf("native function calling failed: %w", err)
		}

		// Transform the result into the expected ReAct format
		return transformFunctionCallResult(result, inputs)
	}
}

// supportsToolCalling checks if the LLM supports native function/tool calling.
func supportsToolCalling(llm core.LLM) bool {
	capabilities := llm.Capabilities()
	return slices.Contains(capabilities, core.CapabilityToolCalling)
}

// buildFunctionSchemas converts tools from the registry into function schemas
// that can be passed to GenerateWithFunctions.
func buildFunctionSchemas(config FunctionCallingConfig) ([]map[string]interface{}, error) {
	var functions []map[string]interface{}

	// Add tools from registry
	if config.ToolRegistry != nil {
		registeredTools := config.ToolRegistry.List()
		for _, tool := range registeredTools {
			schema := tool.InputSchema()

			// Extract required fields from properties
			var required []string
			properties := make(map[string]interface{})
			for name, paramSchema := range schema.Properties {
				properties[name] = map[string]interface{}{
					"type":        paramSchema.Type,
					"description": paramSchema.Description,
				}
				if paramSchema.Required {
					required = append(required, name)
				}
			}

			function := map[string]interface{}{
				"name":        tool.Name(),
				"description": tool.Description(),
				"parameters": map[string]interface{}{
					"type":       schema.Type,
					"properties": properties,
					"required":   required,
				},
			}
			functions = append(functions, function)
		}
	}

	// Add Finish tool if configured
	if config.IncludeFinishTool {
		description := config.FinishToolDescription
		if description == "" {
			description = "Call this tool when you have completed the task and have the final answer."
		}

		finishTool := map[string]interface{}{
			"name":        "Finish",
			"description": description,
			"parameters": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"answer": map[string]interface{}{
						"type":        "string",
						"description": "The final answer or result of the task",
					},
					"reasoning": map[string]interface{}{
						"type":        "string",
						"description": "Brief explanation of how the answer was derived",
					},
				},
				"required": []string{"answer"},
			},
		}
		functions = append(functions, finishTool)
	}

	return functions, nil
}

// buildPromptFromInputs constructs a prompt string from the module inputs.
// This extracts the relevant fields and formats them for the LLM.
func buildPromptFromInputs(inputs map[string]any, info *core.ModuleInfo) string {
	var prompt string

	// Add signature instruction if available
	if info != nil && info.Signature.Instruction != "" {
		prompt += info.Signature.Instruction + "\n\n"
	}

	// Add conversation context if present (for multi-turn ReAct)
	if context, ok := inputs["conversation_context"].(string); ok && context != "" {
		prompt += "Previous conversation:\n" + context + "\n\n"
	}

	// Add task/question
	if task, ok := inputs["task"].(string); ok {
		prompt += "Task: " + task + "\n"
	}
	if question, ok := inputs["question"].(string); ok {
		prompt += "Question: " + question + "\n"
	}

	// Add observation from previous iteration
	if observation, ok := inputs["observation"].(string); ok && observation != "" {
		prompt += "\nObservation from previous action: " + observation + "\n"
	}

	// Add any other relevant inputs in deterministic order
	var otherKeys []string
	for key := range inputs {
		switch key {
		case "task", "question", "observation", "conversation_context", "thought", "action":
			// Already handled or internal fields
			continue
		default:
			otherKeys = append(otherKeys, key)
		}
	}
	slices.Sort(otherKeys) // Sort keys for deterministic prompt order

	for _, key := range otherKeys {
		if strVal, ok := inputs[key].(string); ok && strVal != "" {
			prompt += fmt.Sprintf("\n%s: %s", key, strVal)
		}
	}

	prompt += "\n\nBased on the above, decide which tool to use next. If you have enough information to answer, use the Finish tool."

	return prompt
}

// transformFunctionCallResult converts the LLM's function call response
// into the format expected by the ReAct module.
func transformFunctionCallResult(result map[string]interface{}, originalInputs map[string]any) (map[string]any, error) {
	output := make(map[string]any)

	// Copy through original inputs that should be preserved
	for key, value := range originalInputs {
		output[key] = value
	}

	// Extract function call from result
	functionCall, hasFunctionCall := result["function_call"].(map[string]interface{})

	if hasFunctionCall {
		toolName, _ := functionCall["name"].(string)
		arguments, _ := functionCall["arguments"].(map[string]interface{})

		// Set the action field as a structured map (ReAct can handle both string and map)
		output["action"] = map[string]interface{}{
			"tool_name": toolName,
			"arguments": arguments,
		}

		// Extract thought if the LLM provided one (some providers include reasoning)
		if thought, ok := result["content"].(string); ok && thought != "" {
			output["thought"] = thought
		} else {
			// Generate a default thought based on the action
			output["thought"] = fmt.Sprintf("I will use the %s tool to proceed.", toolName)
		}

		// For Finish tool, also set the answer field
		if toolName == "Finish" {
			if answer, ok := arguments["answer"].(string); ok {
				output["answer"] = answer
			}
			if reasoning, ok := arguments["reasoning"].(string); ok {
				output["reasoning"] = reasoning
			}
		}
	} else if content, ok := result["content"].(string); ok {
		// LLM responded with text instead of a function call
		// This shouldn't happen in strict mode, but handle gracefully
		output["thought"] = content
		output["action"] = map[string]interface{}{
			"tool_name": "Finish",
			"arguments": map[string]interface{}{
				"answer": content,
			},
		}
	}

	return output, nil
}

// FunctionCallingReActAdapter wraps the ReAct module to use native function calling.
// This is a higher-level helper that configures everything automatically.
type FunctionCallingReActAdapter struct {
	config      FunctionCallingConfig
	interceptor core.ModuleInterceptor
}

// NewFunctionCallingReActAdapter creates an adapter that enables native function calling for ReAct.
func NewFunctionCallingReActAdapter(registry *tools.InMemoryToolRegistry) *FunctionCallingReActAdapter {
	config := DefaultFunctionCallingConfig()
	config.ToolRegistry = registry

	return &FunctionCallingReActAdapter{
		config:      config,
		interceptor: NativeFunctionCallingInterceptor(config),
	}
}

// WithStrictMode enables strict mode where the LLM must always call a function.
func (a *FunctionCallingReActAdapter) WithStrictMode() *FunctionCallingReActAdapter {
	a.config.StrictMode = true
	a.interceptor = NativeFunctionCallingInterceptor(a.config)
	return a
}

// WithCustomFinishDescription sets a custom description for the Finish tool.
func (a *FunctionCallingReActAdapter) WithCustomFinishDescription(description string) *FunctionCallingReActAdapter {
	a.config.FinishToolDescription = description
	a.interceptor = NativeFunctionCallingInterceptor(a.config)
	return a
}

// GetInterceptor returns the configured module interceptor.
func (a *FunctionCallingReActAdapter) GetInterceptor() core.ModuleInterceptor {
	return a.interceptor
}
