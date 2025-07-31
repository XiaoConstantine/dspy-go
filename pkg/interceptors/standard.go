package interceptors

import (
	"context"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
)

// LoggingModuleInterceptor creates an interceptor that logs module execution.
// It logs before and after module execution with timing information.
func LoggingModuleInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		logger := logging.GetLogger()
		start := time.Now()

		logger.Info(ctx, "Module starting: %s (type: %s)", info.ModuleName, info.ModuleType)

		result, err := handler(ctx, inputs, opts...)

		duration := time.Since(start)
		if err != nil {
			logger.Error(ctx, "Module failed: %s, duration: %v, error: %v", info.ModuleName, duration, err)
		} else {
			logger.Info(ctx, "Module completed: %s, duration: %v", info.ModuleName, duration)
		}

		return result, err
	}
}

// LoggingAgentInterceptor creates an interceptor that logs agent execution.
// It logs before and after agent execution with timing information.
func LoggingAgentInterceptor() core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		logger := logging.GetLogger()
		start := time.Now()

		logger.Info(ctx, "Agent starting: %s (type: %s)", info.AgentID, info.AgentType)

		result, err := handler(ctx, input)

		duration := time.Since(start)
		if err != nil {
			logger.Error(ctx, "Agent failed: %s, duration: %v, error: %v", info.AgentID, duration, err)
		} else {
			logger.Info(ctx, "Agent completed: %s, duration: %v", info.AgentID, duration)
		}

		return result, err
	}
}

// LoggingToolInterceptor creates an interceptor that logs tool execution.
// It logs before and after tool execution with timing and argument information.
func LoggingToolInterceptor() core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		logger := logging.GetLogger()
		start := time.Now()

		logger.Info(ctx, "Tool starting: %s (type: %s), args: %v", info.Name, info.ToolType, args)

		result, err := handler(ctx, args)

		duration := time.Since(start)
		if err != nil {
			logger.Error(ctx, "Tool failed: %s, duration: %v, error: %v", info.Name, duration, err)
		} else {
			logger.Info(ctx, "Tool completed: %s, duration: %v", info.Name, duration)
		}

		return result, err
	}
}

// TracingModuleInterceptor creates an interceptor that adds distributed tracing to module execution.
// It integrates with the existing span system in core.ExecutionState.
func TracingModuleInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		// Start a new span for this module execution
		ctx, span := core.StartSpanWithContext(ctx, "module.process", info.ModuleName, map[string]interface{}{
			"module_type": info.ModuleType,
			"version":     info.Version,
		})
		defer core.EndSpan(ctx)

		// Add input information to span
		if span != nil {
			span.WithAnnotation("input_fields", getInputFieldNames(inputs))
			span.WithAnnotation("signature", info.Signature)
		}

		result, err := handler(ctx, inputs, opts...)

		// Record error in span if execution failed
		if err != nil && span != nil {
			span.WithError(err)
		}

		// Add output information to span
		if span != nil && result != nil {
			span.WithAnnotation("output_fields", getOutputFieldNames(result))
		}

		return result, err
	}
}

// TracingAgentInterceptor creates an interceptor that adds distributed tracing to agent execution.
func TracingAgentInterceptor() core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		// Start a new span for this agent execution
		ctx, span := core.StartSpanWithContext(ctx, "agent.execute", info.AgentID, map[string]interface{}{
			"agent_type":    info.AgentType,
			"version":       info.Version,
			"capabilities":  len(info.Capabilities),
		})
		defer core.EndSpan(ctx)

		// Add input information to span
		if span != nil {
			span.WithAnnotation("input_keys", getMapKeys(input))
		}

		result, err := handler(ctx, input)

		// Record error in span if execution failed
		if err != nil && span != nil {
			span.WithError(err)
		}

		// Add output information to span
		if span != nil && result != nil {
			span.WithAnnotation("output_keys", getMapKeys(result))
		}

		return result, err
	}
}

// TracingToolInterceptor creates an interceptor that adds distributed tracing to tool execution.
func TracingToolInterceptor() core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		// Start a new span for this tool execution
		ctx, span := core.StartSpanWithContext(ctx, "tool.call", info.Name, map[string]interface{}{
			"tool_type":    info.ToolType,
			"version":      info.Version,
			"description":  info.Description,
		})
		defer core.EndSpan(ctx)

		// Add argument information to span
		if span != nil {
			span.WithAnnotation("args", args)
			span.WithAnnotation("capabilities", info.Capabilities)
		}

		result, err := handler(ctx, args)

		// Record error in span if execution failed
		if err != nil && span != nil {
			span.WithError(err)
		}

		// Add result information to span
		if span != nil {
			span.WithAnnotation("result_data", result.Data)
		}

		return result, err
	}
}

// MetricsModuleInterceptor creates an interceptor that collects performance metrics for modules.
// It tracks execution duration, success/failure rates, and token usage.
func MetricsModuleInterceptor() core.ModuleInterceptor {
	return func(ctx context.Context, inputs map[string]any, info *core.ModuleInfo, handler core.ModuleHandler, opts ...core.Option) (map[string]any, error) {
		start := time.Now()

		result, err := handler(ctx, inputs, opts...)

		duration := time.Since(start)

		// Record metrics in execution state for later collection
		if state := core.GetExecutionState(ctx); state != nil {
			// Create metrics annotation for this module execution
			metrics := map[string]interface{}{
				"module_name":     info.ModuleName,
				"module_type":     info.ModuleType,
				"duration_ms":     duration.Milliseconds(),
				"success":         err == nil,
				"timestamp":       start.Unix(),
				"input_count":     len(inputs),
				"output_count":    len(result),
			}

			// Add token usage if available
			if tokenUsage := state.GetTokenUsage(); tokenUsage != nil {
				metrics["prompt_tokens"] = tokenUsage.PromptTokens
				metrics["completion_tokens"] = tokenUsage.CompletionTokens
				metrics["total_tokens"] = tokenUsage.TotalTokens
				metrics["cost"] = tokenUsage.Cost
			}

			// Store metrics in current span
			if span := state.GetCurrentSpan(); span != nil {
				span.WithAnnotation("metrics", metrics)
			}
		}

		return result, err
	}
}

// MetricsAgentInterceptor creates an interceptor that collects performance metrics for agents.
func MetricsAgentInterceptor() core.AgentInterceptor {
	return func(ctx context.Context, input map[string]interface{}, info *core.AgentInfo, handler core.AgentHandler) (map[string]interface{}, error) {
		start := time.Now()

		result, err := handler(ctx, input)

		duration := time.Since(start)

		// Record metrics in execution state for later collection
		if state := core.GetExecutionState(ctx); state != nil {
			metrics := map[string]interface{}{
				"agent_id":          info.AgentID,
				"agent_type":        info.AgentType,
				"duration_ms":       duration.Milliseconds(),
				"success":           err == nil,
				"timestamp":         start.Unix(),
				"input_count":       len(input),
				"output_count":      len(result),
				"capabilities_used": len(info.Capabilities),
			}

			// Add token usage if available
			if tokenUsage := state.GetTokenUsage(); tokenUsage != nil {
				metrics["prompt_tokens"] = tokenUsage.PromptTokens
				metrics["completion_tokens"] = tokenUsage.CompletionTokens
				metrics["total_tokens"] = tokenUsage.TotalTokens
				metrics["cost"] = tokenUsage.Cost
			}

			// Store metrics in current span
			if span := state.GetCurrentSpan(); span != nil {
				span.WithAnnotation("metrics", metrics)
			}
		}

		return result, err
	}
}

// MetricsToolInterceptor creates an interceptor that collects performance metrics for tools.
func MetricsToolInterceptor() core.ToolInterceptor {
	return func(ctx context.Context, args map[string]interface{}, info *core.ToolInfo, handler core.ToolHandler) (core.ToolResult, error) {
		start := time.Now()

		result, err := handler(ctx, args)

		duration := time.Since(start)

		// Record metrics in execution state for later collection
		if state := core.GetExecutionState(ctx); state != nil {
			metrics := map[string]interface{}{
				"tool_name":    info.Name,
				"tool_type":    info.ToolType,
				"duration_ms":  duration.Milliseconds(),
				"success":      err == nil,
				"timestamp":    start.Unix(),
				"args_count":   len(args),
			}

			// Add result information if available
			if result.Data != nil {
				metrics["has_data"] = true
			}

			// Store metrics in current span
			if span := state.GetCurrentSpan(); span != nil {
				span.WithAnnotation("metrics", metrics)
			}
		}

		return result, err
	}
}

// Helper functions

// getInputFieldNames extracts field names from module inputs.
func getInputFieldNames(inputs map[string]any) []string {
	keys := make([]string, 0, len(inputs))
	for k := range inputs {
		keys = append(keys, k)
	}
	return keys
}

// getOutputFieldNames extracts field names from module outputs.
func getOutputFieldNames(outputs map[string]any) []string {
	keys := make([]string, 0, len(outputs))
	for k := range outputs {
		keys = append(keys, k)
	}
	return keys
}

// getMapKeys extracts keys from a generic map.
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
