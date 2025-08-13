package main

import (
	"context"
	"os"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
	// Set debug logging
	debugLogger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs: []logging.Output{
			logging.NewConsoleOutput(false),
		},
	})
	logging.SetLogger(debugLogger)
	logger := logging.GetLogger()
	ctx := context.Background()

	logger.Info(ctx, "=== Predict with XML Output Example ===")
	logger.Info(ctx, "Starting Predict XML output demonstration")

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

	// Example 1: Traditional Predict (original behavior)
	logger.Info(ctx, "--- Example 1: Traditional Predict (prefix-based output) ---")
	logger.Info(ctx, "Running traditional Predict example")

	signature1 := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	predict1 := modules.NewPredict(signature1).WithName("Traditional Predict")
	predict1.SetLLM(llm)

	inputs := map[string]any{"question": "What are the main benefits of renewable energy?"}

	outputs1, err := predict1.Process(ctx, inputs)
	if err != nil {
		logger.Error(ctx, "Error in traditional Predict: %v", err)
	} else {
		logger.Info(ctx, "Traditional Predict Result: %v", outputs1["answer"])
		logger.Info(ctx, "Traditional Predict completed successfully")
	}

	// Example 2: Predict with XML Output (enhanced structured output)
	logger.Info(ctx, "--- Example 2: Predict with XML Output (structured output) ---")
	logger.Info(ctx, "Running Predict with XML output")

	// XML Mode: Can use simple core.Field{} instead of core.NewField()
	// because XML interceptors handle structure, no prefixes needed!
	signature2 := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "topic"}}},
		[]core.OutputField{
			{Field: core.Field{Name: "summary"}},
			{Field: core.Field{Name: "key_points"}},
			{Field: core.Field{Name: "conclusion"}},
		},
	)

	// Configure XML output with enhanced features
	xmlConfig := interceptors.DefaultXMLConfig().
		WithMaxSize(4096).       // 4KB limit for security
		WithStrictParsing(true). // Require all fields
		WithValidation(true).    // Validate XML syntax
		WithFallback(true)       // Enable fallback to text on errors

	logger.Debug(ctx, "Configured XML output with max size: %d bytes", 4096)

	predict2 := modules.NewPredict(signature2).
		WithName("XML Predict").
		WithXMLOutput(xmlConfig)
	predict2.SetLLM(llm)

	inputs2 := map[string]any{"topic": "The impact of artificial intelligence on modern healthcare"}

	outputs2, err := predict2.Process(ctx, inputs2)
	if err != nil {
		logger.Error(ctx, "Error in XML output Predict: %v", err)
	} else {
		logger.Info(ctx, "XML Output Predict Results:")
		logger.Info(ctx, "Summary: %v", outputs2["summary"])
		logger.Info(ctx, "Key Points: %v", outputs2["key_points"])
		logger.Info(ctx, "Conclusion: %v", outputs2["conclusion"])
		logger.Info(ctx, "XML output Predict completed successfully")
	}

	// Example 3: Advanced XML Configuration (performance-optimized)
	logger.Info(ctx, "--- Example 3: Performance-Optimized XML Predict ---")
	logger.Info(ctx, "Running Predict with performance-optimized XML configuration")

	signature3 := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "scenario"}}},
		[]core.OutputField{
			{Field: core.Field{Name: "analysis"}},
			{Field: core.Field{Name: "recommendations"}},
		},
	)

	// Performance-optimized XML configuration
	perfConfig := interceptors.PerformantXMLConfig().
		WithMaxSize(10240). // 10KB to accommodate longer responses
		WithCustomTag("analysis", "analysis_section").
		WithCustomTag("recommendations", "recommendations_section")

	predict3 := modules.NewPredict(signature3).
		WithName("Performance Predict").
		WithXMLOutput(perfConfig)
	predict3.SetLLM(llm)

	inputs3 := map[string]any{
		"scenario": "A startup company wants to implement AI-powered customer service automation",
	}

	outputs3, err := predict3.Process(ctx, inputs3)
	if err != nil {
		logger.Error(ctx, "Error in performance-optimized Predict: %v", err)
	} else {
		logger.Info(ctx, "Performance-Optimized Results:")
		logger.Info(ctx, "Analysis: %v", outputs3["analysis"])
		logger.Info(ctx, "Recommendations: %v", outputs3["recommendations"])
		logger.Info(ctx, "Performance-optimized Predict completed successfully")
	}

	// Example 4: Secure XML Configuration (with enhanced security)
	logger.Info(ctx, "--- Example 4: Secure XML Predict ---")
	logger.Info(ctx, "Running Predict with secure XML configuration")

	signature4 := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "data"}}},
		[]core.OutputField{
			{Field: core.Field{Name: "processed_result"}},
			{Field: core.Field{Name: "security_assessment"}},
		},
	)

	// Secure XML configuration with strict limits
	secureConfig := interceptors.SecureXMLConfig().
		WithMaxSize(1024).      // Smaller limit for security
		WithMaxDepth(3).        // Limit XML nesting
		WithTimeout(5 * time.Second)       // 5 second timeout

	predict4 := modules.NewPredict(signature4).
		WithName("Secure Predict").
		WithXMLOutput(secureConfig)
	predict4.SetLLM(llm)

	inputs4 := map[string]any{
		"data": "Analyze this user input for potential security vulnerabilities: <script>alert('xss')</script>",
	}

	outputs4, err := predict4.Process(ctx, inputs4)
	if err != nil {
		logger.Error(ctx, "Error in secure Predict: %v", err)
	} else {
		logger.Info(ctx, "Secure XML Results:")
		logger.Info(ctx, "Processed Result: %v", outputs4["processed_result"])
		logger.Info(ctx, "Security Assessment: %v", outputs4["security_assessment"])
		logger.Info(ctx, "Secure Predict completed successfully")
	}

	logger.Info(ctx, "=== XML Output Features Demonstrated ===")
	logger.Info(ctx, "✅ Structured multi-field output")
	logger.Info(ctx, "✅ Enhanced XML validation and security")
	logger.Info(ctx, "✅ Configurable size limits and timeouts")
	logger.Info(ctx, "✅ Custom XML tags and performance optimization")
	logger.Info(ctx, "✅ Graceful error handling and fallback")
	logger.Info(ctx, "✅ Backward compatibility maintained")
	logger.Info(ctx, "✅ Bypassed traditional parsing for better performance")

	logger.Info(ctx, "Predict XML output demonstration completed")
}
