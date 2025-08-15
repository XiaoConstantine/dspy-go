package main

import (
	"context"
	"fmt"
	"os"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// WeatherTool demonstrates a simple weather checking tool.
type WeatherTool struct{}

func (w *WeatherTool) Name() string {
	return "weather_check"
}

func (w *WeatherTool) Description() string {
	return "Check the weather for a given location"
}

func (w *WeatherTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"location": {
				Type:        "string",
				Description: "The location to check weather for",
				Required:    true,
			},
		},
	}
}

func (w *WeatherTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         w.Name(),
		Description:  w.Description(),
		InputSchema:  w.InputSchema(),
		Capabilities: []string{"weather"},
		Version:      "1.0.0",
	}
}

func (w *WeatherTool) CanHandle(ctx context.Context, intent string) bool {
	// This tool can handle weather-related queries
	return true // For simplicity, assume it can handle any intent
}

func (w *WeatherTool) Validate(params map[string]interface{}) error {
	if _, ok := params["location"]; !ok {
		return fmt.Errorf("location parameter is required")
	}
	return nil
}

func (w *WeatherTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	logger := logging.GetLogger()
	location, ok := params["location"].(string)
	if !ok {
		return core.ToolResult{}, fmt.Errorf("location parameter is required")
	}

	logger.Info(ctx, "WeatherTool executed for location: %s", location)

	// Simulate weather data
	weatherData := fmt.Sprintf("Weather in %s: Sunny, 72°F with light breeze", location)

	return core.ToolResult{
		Data: weatherData,
		Metadata: map[string]interface{}{
			"tool_name": "weather_check",
			"location":  location,
		},
		Annotations: map[string]interface{}{
			"simulated": true,
		},
	}, nil
}

func main() {
	logger := logging.GetLogger()
	ctx := context.Background()

	logger.Info(ctx, "=== ReAct with XML Interceptors Example ===")
	logger.Info(ctx, "Starting ReAct XML interceptors demonstration")

	// Check for required environment variable
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		logger.Error(ctx, "GEMINI_API_KEY environment variable is required")
		logger.Info(ctx, "Please set GEMINI_API_KEY to run this example")
		return
	}

	// Initialize LLM factory and create Gemini LLM
	llms.EnsureFactory()
	llm, err := llms.NewGeminiLLM(apiKey, core.ModelID("gemini-2.0-flash"))
	if err != nil {
		logger.Error(ctx, "Failed to create Gemini LLM: %v", err)
		return
	}
	logger.Info(ctx, "Gemini LLM initialized successfully")

	// Create tool registry and add our weather tool
	registry := tools.NewInMemoryToolRegistry()
	weatherTool := &WeatherTool{}
	if err := registry.Register(weatherTool); err != nil {
		logger.Error(ctx, "Failed to register weather tool: %v", err)
		return
	}
	logger.Info(ctx, "Weather tool registered successfully")

	// Create signature for the ReAct module
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Example 1: ReAct without XML interceptors (original behavior)
	logger.Info(ctx, "--- Example 1: Traditional ReAct (hardcoded XML parsing) ---")
	logger.Info(ctx, "Running traditional ReAct example")

	react1 := modules.NewReAct(signature, registry, 5)
	react1.SetLLM(llm)

	inputs := map[string]any{"question": "What's the weather like in San Francisco?"}

	outputs1, err := react1.Process(ctx, inputs)
	if err != nil {
		logger.Error(ctx, "Error in traditional ReAct: %v", err)
	} else {
		logger.Info(ctx, "Traditional ReAct Result: %v", outputs1["answer"])
		logger.Info(ctx, "Traditional ReAct completed successfully")
	}

	// Example 2: ReAct with XML interceptors (enhanced parsing)
	logger.Info(ctx, "--- Example 2: ReAct with XML Interceptors (enhanced parsing) ---")
	logger.Info(ctx, "Running ReAct with XML interceptors")

	react2 := modules.NewReAct(signature, registry, 5)

	// Configure XML interceptors with enhanced features
	xmlConfig := interceptors.DefaultXMLConfig().
		WithMaxSize(4096).       // 4KB limit for security
		WithStrictParsing(true). // Require all fields
		WithValidation(true).    // Validate XML syntax
		WithFallback(true)       // Enable fallback to text on errors

	logger.Debug(ctx, "Configured XML interceptors with max size: %d bytes", 4096)
	react2.WithXMLParsing(xmlConfig)
	react2.SetLLM(llm)

	outputs2, err := react2.Process(ctx, inputs)
	if err != nil {
		logger.Error(ctx, "Error in XML interceptor ReAct: %v", err)
	} else {
		logger.Info(ctx, "XML Interceptor ReAct Result: %v", outputs2["answer"])
		logger.Info(ctx, "XML interceptor ReAct completed successfully")
	}

	logger.Info(ctx, "=== XML Interceptor Features Demonstrated ===")
	logger.Info(ctx, "✅ Backward compatibility maintained")
	logger.Info(ctx, "✅ Enhanced XML validation and security")
	logger.Info(ctx, "✅ Configurable size limits and timeouts")
	logger.Info(ctx, "✅ Graceful error handling and fallback")
	logger.Info(ctx, "✅ Opt-in enhancement (no breaking changes)")

	logger.Info(ctx, "ReAct XML interceptors demonstration completed")
}
