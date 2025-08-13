package interceptors

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
)

func TestXMLFormatModuleInterceptor(t *testing.T) {
	config := DefaultXMLConfig()
	interceptor := XMLFormatModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("prompt")},
		},
		[]core.OutputField{
			{Field: core.NewField("result")},
		},
	)

	info := core.NewModuleInfo("TestModule", "Test", signature)
	inputs := map[string]any{
		"prompt": "Analyze this text",
	}

	// Mock handler that captures the modified inputs
	var capturedInputs map[string]any
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		capturedInputs = inputs
		return map[string]any{"result": "test"}, nil
	}

	// Execute interceptor
	_, err := interceptor(context.Background(), inputs, info, handler)
	if err != nil {
		t.Fatalf("Interceptor failed: %v", err)
	}

	// Verify XML instructions were injected
	modifiedPrompt, ok := capturedInputs["prompt"].(string)
	if !ok {
		t.Fatal("Prompt should be a string")
	}

	if !strings.Contains(modifiedPrompt, "<result>") {
		t.Error("XML instructions should contain result tags")
	}

	if !strings.Contains(modifiedPrompt, "XML structure") {
		t.Error("XML instructions should contain formatting guidance")
	}
}

func TestXMLParseModuleInterceptor(t *testing.T) {
	config := DefaultXMLConfig()
	interceptor := XMLParseModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("input")},
		},
		[]core.OutputField{
			{Field: core.NewField("name")},
			{Field: core.NewField("age")},
		},
	)

	info := core.NewModuleInfo("TestModule", "Test", signature)
	inputs := map[string]any{
		"input": "test",
	}

	// Mock handler that returns XML response
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{
			"response": `<response>
				<name>John Doe</name>
				<age>30</age>
			</response>`,
		}, nil
	}

	// Execute interceptor
	outputs, err := interceptor(context.Background(), inputs, info, handler)
	if err != nil {
		t.Fatalf("Interceptor failed: %v", err)
	}

	// Verify parsed fields
	if name, ok := outputs["name"]; !ok || name != "John Doe" {
		t.Errorf("Expected name 'John Doe', got %v", name)
	}

	if age, ok := outputs["age"]; !ok || age != "30" {
		t.Errorf("Expected age '30', got %v", age)
	}
}

func TestXMLParseModuleInterceptor_MalformedXML(t *testing.T) {
	config := DefaultXMLConfig().WithFallback(true)
	interceptor := XMLParseModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("input")},
		},
		[]core.OutputField{
			{Field: core.NewField("result")},
		},
	)

	info := core.NewModuleInfo("TestModule", "Test", signature)
	inputs := map[string]any{"input": "test"}

	// Mock handler with malformed XML
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{
			"response": `<response><result>test</invalid>`, // Malformed XML
		}, nil
	}

	// Should fallback gracefully
	outputs, err := interceptor(context.Background(), inputs, info, handler)
	if err != nil {
		t.Fatalf("Should not fail with fallback enabled: %v", err)
	}

	// Should return original response
	if response, ok := outputs["response"]; !ok {
		t.Error("Should preserve original response on fallback")
	} else if !strings.Contains(response.(string), "test") {
		t.Error("Fallback response should contain original content")
	}
}

func TestXMLParseModuleInterceptor_StrictMode(t *testing.T) {
	config := StrictXMLConfig()
	interceptor := XMLParseModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("input")},
		},
		[]core.OutputField{
			{Field: core.NewField("result")},
			{Field: core.NewField("score")}, // This field will be missing
		},
	)

	info := core.NewModuleInfo("TestModule", "Test", signature)
	inputs := map[string]any{"input": "test"}

	// Mock handler with incomplete XML
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{
			"response": `<response><result>test</result></response>`, // Missing score field
		}, nil
	}

	// Should fail in strict mode
	_, err := interceptor(context.Background(), inputs, info, handler)
	if err == nil {
		t.Error("Should fail in strict mode with missing required field")
	}

	if !strings.Contains(err.Error(), "score") {
		t.Errorf("Error should mention missing field 'score', got: %v", err)
	}
}

func TestXMLModuleInterceptor_FullPipeline(t *testing.T) {
	config := DefaultXMLConfig()
	interceptor := XMLModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question")},
		},
		[]core.OutputField{
			{Field: core.NewField("answer")},
			{Field: core.NewField("confidence")},
		},
	)

	info := core.NewModuleInfo("TestModule", "Test", signature)
	inputs := map[string]any{
		"question": "What is the capital of France?",
	}

	// Mock handler that simulates LLM with XML instructions and response
	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		// Verify XML instructions were injected
		if question, ok := inputs["question"].(string); ok {
			if !strings.Contains(question, "XML structure") {
				t.Error("XML instructions should be injected in question field")
			}
		}

		// Return XML response
		return map[string]any{
			"response": `<response>
				<answer>Paris is the capital of France</answer>
				<confidence>0.95</confidence>
			</response>`,
		}, nil
	}

	// Execute full pipeline
	outputs, err := interceptor(context.Background(), inputs, info, handler)
	if err != nil {
		t.Fatalf("Full pipeline failed: %v", err)
	}

	// Verify structured outputs
	if answer, ok := outputs["answer"]; !ok || answer != "Paris is the capital of France" {
		t.Errorf("Expected answer 'Paris is the capital of France', got %v", answer)
	}

	if confidence, ok := outputs["confidence"]; !ok || confidence != "0.95" {
		t.Errorf("Expected confidence '0.95', got %v", confidence)
	}
}

func TestXMLConfig_Builders(t *testing.T) {
	config := DefaultXMLConfig().
		WithStrictParsing(false).
		WithFallback(false).
		WithValidation(false).
		WithMaxDepth(15).
		WithMaxSize(2048).
		WithTimeout(60*time.Second).
		WithCustomTag("question", "q").
		WithTypeHints(true).
		WithPreserveWhitespace(true)

	if config.StrictParsing {
		t.Error("StrictParsing should be false")
	}
	if config.FallbackToText {
		t.Error("FallbackToText should be false")
	}
	if config.ValidateXML {
		t.Error("ValidateXML should be false")
	}
	if config.MaxDepth != 15 {
		t.Errorf("Expected MaxDepth=15, got %d", config.MaxDepth)
	}
	if config.MaxSize != 2048 {
		t.Errorf("Expected MaxSize=2048, got %d", config.MaxSize)
	}
	if config.ParseTimeout != 60*time.Second {
		t.Errorf("Expected timeout=60s, got %v", config.ParseTimeout)
	}
	if config.GetTagName("question") != "q" {
		t.Errorf("Expected custom tag 'q', got %s", config.GetTagName("question"))
	}
	if !config.IncludeTypeHints {
		t.Error("IncludeTypeHints should be true")
	}
	if !config.PreserveWhitespace {
		t.Error("PreserveWhitespace should be true")
	}
}

func TestPresetConfigurations(t *testing.T) {
	tests := []struct {
		name   string
		config XMLConfig
		checks func(t *testing.T, config XMLConfig)
	}{
		{
			name:   "StrictXMLConfig",
			config: StrictXMLConfig(),
			checks: func(t *testing.T, config XMLConfig) {
				if !config.StrictParsing {
					t.Error("Strict config should have strict parsing enabled")
				}
				if config.FallbackToText {
					t.Error("Strict config should not have fallback enabled")
				}
			},
		},
		{
			name:   "FlexibleXMLConfig",
			config: FlexibleXMLConfig(),
			checks: func(t *testing.T, config XMLConfig) {
				if config.StrictParsing {
					t.Error("Flexible config should not have strict parsing")
				}
				if !config.FallbackToText {
					t.Error("Flexible config should have fallback enabled")
				}
			},
		},
		{
			name:   "PerformantXMLConfig",
			config: PerformantXMLConfig(),
			checks: func(t *testing.T, config XMLConfig) {
				if config.ValidateXML {
					t.Error("Performant config should not validate XML")
				}
				if config.MaxDepth != 5 {
					t.Errorf("Expected MaxDepth=5, got %d", config.MaxDepth)
				}
			},
		},
		{
			name:   "SecureXMLConfig",
			config: SecureXMLConfig(),
			checks: func(t *testing.T, config XMLConfig) {
				if config.MaxDepth != 3 {
					t.Errorf("Expected MaxDepth=3, got %d", config.MaxDepth)
				}
				if !config.ValidateXML {
					t.Error("Secure config should validate XML")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.checks(t, tt.config)
		})
	}
}

func TestXMLParser_SecurityLimits(t *testing.T) {
	config := DefaultXMLConfig().
		WithMaxDepth(2).
		WithMaxSize(100)

	parser := &XMLParser{
		config: config,
		cache:  make(map[string]*ParsedSignature),
	}

	signature := core.NewSignature(
		[]core.InputField{},
		[]core.OutputField{
			{Field: core.NewField("result")},
		},
	)

	// Test depth limit
	deepXML := `<response><level1><level2><level3><result>test</result></level3></level2></level1></response>`
	_, err := parser.parseXML(deepXML, signature)
	if err == nil {
		t.Error("Should fail with depth limit exceeded")
	}

	// Test size limit
	largeXML := strings.Repeat("a", 200) // Exceeds 100 byte limit
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err = parser.ParseXMLOutputs(ctx, map[string]any{"response": largeXML}, signature)
	if err == nil {
		t.Error("Should fail with size limit exceeded")
	}
}

func TestApplyXMLInterceptors(t *testing.T) {
	config := DefaultXMLConfig()

	// Create test module that implements InterceptableModule
	module := &TestInterceptableModule{
		BaseModule: core.BaseModule{
			Signature: core.NewSignature(
				[]core.InputField{{Field: core.NewField("input")}},
				[]core.OutputField{{Field: core.NewField("output")}},
			),
			ModuleType: "TestModule",
		},
	}

	// Apply XML interceptors
	err := ApplyXMLInterceptors(module, config)
	if err != nil {
		t.Fatalf("ApplyXMLInterceptors failed: %v", err)
	}

	// Verify interceptors were applied
	interceptors := module.GetInterceptors()
	if len(interceptors) == 0 {
		t.Error("Expected interceptors to be applied")
	}
}

func TestCreateXMLInterceptorChain(t *testing.T) {
	config := DefaultXMLConfig()

	// Create chain with additional interceptors
	chain := CreateXMLInterceptorChain(config,
		LoggingModuleInterceptor(),
		MetricsModuleInterceptor(),
	)

	// Should have format + 2 additional + parse = 4 interceptors
	expectedCount := 4
	if len(chain) != expectedCount {
		t.Errorf("Expected %d interceptors in chain, got %d", expectedCount, len(chain))
	}
}

// TestInterceptableModule is a test implementation of InterceptableModule
type TestInterceptableModule struct {
	core.BaseModule
}

func (m *TestInterceptableModule) Process(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
	return map[string]any{
		"response": `<response><output>test result</output></response>`,
	}, nil
}

func (m *TestInterceptableModule) Clone() core.Module {
	return &TestInterceptableModule{BaseModule: m.BaseModule}
}

func (m *TestInterceptableModule) ProcessWithInterceptors(ctx context.Context, inputs map[string]any, interceptors []core.ModuleInterceptor, opts ...core.Option) (map[string]any, error) {
	return m.ProcessWithInterceptorsImpl(ctx, inputs, interceptors, m.Process, opts...)
}

func (m *TestInterceptableModule) SetInterceptors(interceptors []core.ModuleInterceptor) {
	m.BaseModule.SetInterceptors(interceptors)
}

func (m *TestInterceptableModule) GetInterceptors() []core.ModuleInterceptor {
	return m.BaseModule.GetInterceptors()
}

func (m *TestInterceptableModule) ClearInterceptors() {
	m.BaseModule.ClearInterceptors()
}

// Benchmark tests
func BenchmarkXMLParsing(b *testing.B) {
	config := DefaultXMLConfig()
	parser := &XMLParser{
		config: config,
		cache:  make(map[string]*ParsedSignature),
	}

	signature := core.NewSignature(
		[]core.InputField{},
		[]core.OutputField{
			{Field: core.NewField("name")},
			{Field: core.NewField("age")},
			{Field: core.NewField("email")},
		},
	)

	xmlResponse := `<response>
		<name>John Doe</name>
		<age>30</age>
		<email>john@example.com</email>
	</response>`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := parser.parseXML(xmlResponse, signature)
		if err != nil {
			b.Fatalf("Parsing failed: %v", err)
		}
	}
}

func BenchmarkXMLModuleInterceptor_FullPipeline(b *testing.B) {
	config := DefaultXMLConfig()
	interceptor := XMLModuleInterceptor(config)

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question")},
		},
		[]core.OutputField{
			{Field: core.NewField("answer")},
		},
	)

	info := core.NewModuleInfo("BenchModule", "Test", signature)
	inputs := map[string]any{"question": "test"}

	handler := func(ctx context.Context, inputs map[string]any, opts ...core.Option) (map[string]any, error) {
		return map[string]any{
			"response": `<response><answer>Test answer</answer></response>`,
		}, nil
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := interceptor(context.Background(), inputs, info, handler)
		if err != nil {
			b.Fatalf("Interceptor failed: %v", err)
		}
	}
}
