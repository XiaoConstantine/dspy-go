package main

import (
	"context"
	"fmt"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
)

// BasicXMLExample demonstrates basic XMLAdapter usage.
func main() {
	fmt.Println("=== Basic XML Interceptor Example ===")

	// Create a signature for a QA module
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question", core.WithDescription("The question to answer"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The answer to the question"))},
			{Field: core.NewField("confidence", core.WithDescription("Confidence score"))},
		},
	)

	// Create a mock module (in real usage, this would be your actual module)
	module := &MockModule{
		BaseModule: core.BaseModule{
			Signature:  signature,
			ModuleType: "QAModule",
		},
	}

	// Create XML interceptor with default configuration
	xmlConfig := interceptors.DefaultXMLConfig()
	xmlInterceptor := interceptors.XMLModuleInterceptor(xmlConfig)

	// Apply XML interceptor to the module
	module.SetInterceptors([]core.ModuleInterceptor{xmlInterceptor})
	fmt.Println("XML interceptor applied successfully")

	// Use the module with XML interceptor
	ctx := context.Background()
	inputs := map[string]any{
		"question": "What is the capital of France?",
	}

	// Use ProcessWithInterceptors to ensure interceptors are applied
	outputs, err := module.ProcessWithInterceptors(ctx, inputs, module.GetInterceptors())
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Outputs: %+v\n", outputs)
}

// MockModule is a simple module implementation for testing.
type MockModule struct {
	core.BaseModule
}

// Process implements a mock processing function.
func (m *MockModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	// Check if XML instructions were injected (by the format interceptor)
	if question, ok := inputs["question"].(string); ok {
		fmt.Printf("Received question with XML instructions: %s\n", question)
	}

	// Simulate module processing - return XML response that will be parsed
	return map[string]any{
		"response": `<response>
			<answer>Paris is the capital of France.</answer>
			<confidence>0.95</confidence>
		</response>`,
	}, nil
}

// Clone creates a copy of the mock module.
func (m *MockModule) Clone() core.Module {
	return &MockModule{
		BaseModule: core.BaseModule{
			Signature:  m.Signature,
			ModuleType: m.ModuleType,
		},
	}
}

// Implement InterceptableModule interface for MockModule.
func (m *MockModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *MockModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *MockModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *MockModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}
