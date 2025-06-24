package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"
	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// CompositeTool demonstrates tool composition by combining multiple tools into a single reusable unit.
type CompositeTool struct {
	name     string
	pipeline *tools.ToolPipeline
	registry core.ToolRegistry
}

func (c *CompositeTool) Name() string {
	return c.name
}

func (c *CompositeTool) Description() string {
	return fmt.Sprintf("Composite tool that combines multiple operations: %s", c.name)
}

func (c *CompositeTool) Metadata() *core.ToolMetadata {
	steps := c.pipeline.GetSteps()
	capabilities := make([]string, 0, len(steps))
	for _, step := range steps {
		capabilities = append(capabilities, step.ToolName)
	}

	return &core.ToolMetadata{
		Name:         c.name,
		Description:  c.Description(),
		Capabilities: capabilities,
		Version:      "1.0.0",
	}
}

func (c *CompositeTool) CanHandle(ctx context.Context, intent string) bool {
	return true
}

func (c *CompositeTool) Validate(params map[string]interface{}) error {
	return nil
}

func (c *CompositeTool) InputSchema() models.InputSchema {
	return models.InputSchema{}
}

func (c *CompositeTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := c.pipeline.Execute(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}

	// Return the final result from the pipeline
	if len(result.Results) > 0 {
		finalResult := result.Results[len(result.Results)-1]
		return finalResult, nil
	}

	return core.ToolResult{
		Data: map[string]interface{}{
			"composite_result": "completed",
			"steps_executed":   len(result.Results),
		},
	}, nil
}

// Helper function to create composite tools.
func NewCompositeTool(name string, registry core.ToolRegistry, builder func(*tools.PipelineBuilder) *tools.PipelineBuilder) (*CompositeTool, error) {
	pipeline, err := builder(tools.NewPipelineBuilder(name+"_pipeline", registry)).Build()
	if err != nil {
		return nil, err
	}

	return &CompositeTool{
		name:     name,
		pipeline: pipeline,
		registry: registry,
	}, nil
}

// Simple tools for composition examples.
type TextProcessorTool struct {
	operation string
}

func (t *TextProcessorTool) Name() string        { return "text_" + t.operation }
func (t *TextProcessorTool) Description() string { return "Processes text with " + t.operation }
func (t *TextProcessorTool) Metadata() *core.ToolMetadata {
	return &core.ToolMetadata{
		Name:         t.Name(),
		Description:  t.Description(),
		Capabilities: []string{"text_processing", t.operation},
		Version:      "1.0.0",
	}
}
func (t *TextProcessorTool) CanHandle(ctx context.Context, intent string) bool { return true }
func (t *TextProcessorTool) Validate(params map[string]interface{}) error      { return nil }
func (t *TextProcessorTool) InputSchema() models.InputSchema                   { return models.InputSchema{} }

func (t *TextProcessorTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	text, ok := params["text"].(string)
	if !ok {
		text = "sample text"
	}

	// Simulate processing time
	time.Sleep(20 * time.Millisecond)

	var processedText string
	switch t.operation {
	case "uppercase":
		processedText = fmt.Sprintf("%s (UPPERCASED)", text)
	case "lowercase":
		processedText = fmt.Sprintf("%s (lowercased)", text)
	case "reverse":
		runes := []rune(text)
		for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
			runes[i], runes[j] = runes[j], runes[i]
		}
		processedText = string(runes) + " (reversed)"
	case "length":
		processedText = fmt.Sprintf("%s (length: %d)", text, len(text))
	default:
		processedText = text + " (processed)"
	}

	return core.ToolResult{
		Data: map[string]interface{}{
			"text":      processedText,
			"operation": t.operation,
			"timestamp": time.Now().Unix(),
		},
		Metadata: map[string]interface{}{
			"tool":      t.Name(),
			"operation": t.operation,
		},
	}, nil
}

func setupRegistry() core.ToolRegistry {
	registry := tools.NewInMemoryToolRegistry()

	// Register basic text processing tools
	basicTools := []core.Tool{
		&TextProcessorTool{operation: "uppercase"},
		&TextProcessorTool{operation: "lowercase"},
		&TextProcessorTool{operation: "reverse"},
		&TextProcessorTool{operation: "length"},
	}

	for _, tool := range basicTools {
		if err := registry.Register(tool); err != nil {
			log.Fatalf("Failed to register tool %s: %v", tool.Name(), err)
		}
	}

	return registry
}

func demonstrateBasicComposition(registry core.ToolRegistry) {
	fmt.Println("\n🔧 Basic Tool Composition")
	fmt.Println("=========================")

	// Create a composite tool that combines text processing operations
	textProcessor, err := NewCompositeTool("text_processor", registry, func(builder *tools.PipelineBuilder) *tools.PipelineBuilder {
		return builder.
			Step("text_uppercase").
			Step("text_reverse").
			Step("text_length")
	})

	if err != nil {
		log.Fatalf("Failed to create composite tool: %v", err)
	}

	// Register the composite tool
	if err := registry.Register(textProcessor); err != nil {
		log.Fatalf("Failed to register composite tool: %v", err)
	}

	fmt.Printf("✅ Created composite tool: %s\n", textProcessor.Name())
	fmt.Printf("📋 Capabilities: %v\n", textProcessor.Metadata().Capabilities)

	// Use the composite tool
	ctx := context.Background()
	input := map[string]interface{}{
		"text": "Hello World",
	}

	fmt.Printf("🔄 Processing input: %s\n", input["text"])

	result, err := textProcessor.Execute(ctx, input)
	if err != nil {
		log.Fatalf("Composite tool execution failed: %v", err)
	}

	fmt.Printf("✅ Composite processing completed\n")
	if data, ok := result.Data.(map[string]interface{}); ok {
		if finalText, ok := data["text"].(string); ok {
			fmt.Printf("🎯 Final result: %s\n", finalText)
		}
	}
}

func main() {
	fmt.Println("🧩 Tool Composition Examples")
	fmt.Println("=============================")

	// Setup registry with basic tools
	registry := setupRegistry()
	fmt.Printf("✅ Registered %d basic tools\n", len(registry.List()))

	// Run basic composition demonstration
	demonstrateBasicComposition(registry)

	fmt.Println("\n🎉 Tool composition demonstration completed!")
	fmt.Println("\nKey Composition Features Demonstrated:")
	fmt.Println("  ✅ Basic tool composition into reusable units")
	fmt.Println("  ✅ Composite tools as building blocks")
	fmt.Printf("\n📊 Final registry size: %d tools (including composites)\n", len(registry.List()))
}
