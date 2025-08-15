package main

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

// ComposableXMLExample demonstrates using XML interceptors with other interceptors.
func main() {
	fmt.Println("=== Composable XML Interceptor Example ===")

	// Create a signature
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("text", core.WithDescription("Input text to analyze"))},
		},
		[]core.OutputField{
			{Field: core.NewField("sentiment", core.WithDescription("Sentiment analysis result"))},
			{Field: core.NewField("score", core.WithDescription("Sentiment score"))},
		},
	)

	// Create module
	module := &SentimentModule{
		BaseModule: core.BaseModule{
			Signature:  signature,
			ModuleType: "SentimentModule",
		},
	}

	// Create XML interceptor configuration
	xmlConfig := interceptors.DefaultXMLConfig()

	// Create interceptor chain with XML, logging, and metrics
	interceptorChain := []core.ModuleInterceptor{
		interceptors.LoggingModuleInterceptor(),            // Log execution
		interceptors.XMLFormatModuleInterceptor(xmlConfig), // Format request with XML instructions
		interceptors.MetricsModuleInterceptor(),            // Collect metrics
		interceptors.XMLParseModuleInterceptor(xmlConfig),  // Parse XML response
	}

	// Apply composed interceptors
	module.SetInterceptors(interceptorChain)
	fmt.Println("Composable interceptor chain applied successfully")

	// Use the module with full interceptor chain
	ctx := context.Background()
	inputs := map[string]any{
		"text": "I love this new XML interceptor system!",
	}

	// Use ProcessWithInterceptors to ensure interceptor chain is applied
	outputs, err := module.ProcessWithInterceptors(ctx, inputs, module.GetInterceptors())
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Final outputs: %+v\n", outputs)
}

// SentimentModule is a mock sentiment analysis module.
type SentimentModule struct {
	core.BaseModule
}

// Process implements sentiment analysis with XML output.
func (m *SentimentModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Check if XML instructions were injected
	if text, ok := inputs["text"].(string); ok {
		fmt.Printf("Processing text with XML instructions: %s\n", text)
	}

	// Simulate sentiment analysis - return XML response
	return map[string]any{
		"response": `<response>
			<sentiment>positive</sentiment>
			<score>0.85</score>
		</response>`,
	}, nil
}

// Clone creates a copy of the sentiment module.
func (m *SentimentModule) Clone() core.Module {
	return &SentimentModule{
		BaseModule: core.BaseModule{
			Signature:  m.Signature,
			ModuleType: m.ModuleType,
		},
	}
}

// Implement InterceptableModule interface for SentimentModule.
func (m *SentimentModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *SentimentModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *SentimentModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *SentimentModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}
