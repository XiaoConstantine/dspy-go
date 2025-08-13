package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

// CustomConfigExample demonstrates creating custom XML configurations.
func main() {
	fmt.Println("=== Custom XML Configuration Example ===")

	// Example 1: Strict Configuration
	strictExample()

	// Example 2: Performance Configuration
	performanceExample()

	// Example 3: Custom Tags Configuration
	customTagsExample()
}

func strictExample() {
	fmt.Println("\n--- Strict XML Configuration ---")

	// Create strict config - requires all fields, no fallback
	strictConfig := interceptors.StrictXMLConfig()

	fmt.Printf("Strict config: StrictParsing=%v, Fallback=%v, Validation=%v\n",
		strictConfig.StrictParsing, strictConfig.FallbackToText, strictConfig.ValidateXML)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("query", core.WithDescription("User query"))},
		},
		[]core.OutputField{
			{Field: core.NewField("response", core.WithDescription("Response"))},
			{Field: core.NewField("metadata", core.WithDescription("Response metadata"))},
		},
	)

	module := &StrictModule{
		BaseModule: core.BaseModule{
			Signature:  signature,
			ModuleType: "StrictModule",
		},
	}

	// Apply strict XML interceptor
	xmlInterceptor := interceptors.XMLModuleInterceptor(strictConfig)
	module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})

	ctx := context.Background()
	inputs := map[string]any{"query": "What is AI?"}

	outputs, err := module.ProcessWithInterceptors(ctx, inputs, module.GetInterceptors())
	if err != nil {
		log.Printf("Strict processing error: %v\n", err)
	} else {
		fmt.Printf("Strict outputs: %+v\n", outputs)
	}
}

func performanceExample() {
	fmt.Println("\n--- Performance XML Configuration ---")

	// Create performance-optimized config
	perfConfig := interceptors.PerformantXMLConfig()

	fmt.Printf("Performance config: MaxDepth=%d, MaxSize=%d, Validation=%v\n",
		perfConfig.MaxDepth, perfConfig.MaxSize, perfConfig.ValidateXML)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("data", core.WithDescription("Data to process"))},
		},
		[]core.OutputField{
			{Field: core.NewField("result", core.WithDescription("Processing result"))},
		},
	)

	module := &PerformanceModule{
		BaseModule: core.BaseModule{
			Signature:  signature,
			ModuleType: "PerformanceModule",
		},
	}

	// Apply performance XML interceptor
	xmlInterceptor := interceptors.XMLModuleInterceptor(perfConfig)
	module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})

	ctx := context.Background()
	inputs := map[string]any{"data": "Large dataset for processing"}

	outputs, err := module.ProcessWithInterceptors(ctx, inputs, module.GetInterceptors())
	if err != nil {
		log.Printf("Performance processing error: %v\n", err)
	} else {
		fmt.Printf("Performance outputs: %+v\n", outputs)
	}
}

func customTagsExample() {
	fmt.Println("\n--- Custom Tags XML Configuration ---")

	// Create custom config with custom tags and type hints
	customConfig := interceptors.DefaultXMLConfig().
		WithCustomTag("user_input", "input").
		WithCustomTag("ai_response", "output").
		WithTypeHints(true).
		WithPreserveWhitespace(true).
		WithTimeout(10 * time.Second)

	fmt.Printf("Custom tags: user_input->%s, ai_response->%s\n",
		customConfig.GetTagName("user_input"),
		customConfig.GetTagName("ai_response"))

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("user_input", core.WithDescription("User's question"))},
		},
		[]core.OutputField{
			{Field: core.NewField("ai_response", core.WithDescription("AI's response"))},
		},
	)

	module := &CustomTagsModule{
		BaseModule: core.BaseModule{
			Signature:  signature,
			ModuleType: "CustomTagsModule",
		},
	}

	// Apply custom XML interceptor
	xmlInterceptor := interceptors.XMLModuleInterceptor(customConfig)
	module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})

	ctx := context.Background()
	inputs := map[string]any{"user_input": "Explain quantum computing"}

	outputs, err := module.ProcessWithInterceptors(ctx, inputs, module.GetInterceptors())
	if err != nil {
		log.Printf("Custom tags processing error: %v\n", err)
	} else {
		fmt.Printf("Custom tags outputs: %+v\n", outputs)
	}
}

// Module implementations for examples

type StrictModule struct {
	core.BaseModule
}

func (m *StrictModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{
		"response": `<response>
			<response>AI is artificial intelligence technology.</response>
			<metadata>Source: knowledge base</metadata>
		</response>`,
	}, nil
}

func (m *StrictModule) Clone() core.Module {
	return &StrictModule{BaseModule: m.BaseModule}
}

func (m *StrictModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *StrictModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *StrictModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *StrictModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}

type PerformanceModule struct {
	core.BaseModule
}

func (m *PerformanceModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{
		"response": `<response><result>Processed successfully</result></response>`,
	}, nil
}

func (m *PerformanceModule) Clone() core.Module {
	return &PerformanceModule{BaseModule: m.BaseModule}
}

func (m *PerformanceModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *PerformanceModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *PerformanceModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *PerformanceModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}

type CustomTagsModule struct {
	core.BaseModule
}

func (m *CustomTagsModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Use custom tag names in response
	return map[string]any{
		"response": `<response>
			<output>Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in ways classical computers cannot.</output>
		</response>`,
	}, nil
}

func (m *CustomTagsModule) Clone() core.Module {
	return &CustomTagsModule{BaseModule: m.BaseModule}
}

func (m *CustomTagsModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *CustomTagsModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *CustomTagsModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *CustomTagsModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}
