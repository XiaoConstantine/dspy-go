# DSPy-Go Examples

This directory contains comprehensive examples demonstrating the various features and capabilities of DSPy-Go, a Go implementation of the DSPy framework for declarative, composable AI systems.

## 📁 Example Categories

### 🧠 Core DSPy Patterns
- **[basic/](basic/)** - Fundamental DSPy concepts and simple workflows
- **[advanced_patterns/](advanced_patterns/)** - Complex workflow patterns and compositions
- **[chain_of_thought/](chain_of_thought/)** - Chain-of-thought reasoning implementations
- **[optimization/](optimization/)** - Module optimization and fine-tuning examples

### 🔧 Tool Management
- **[smart_tool_registry/](smart_tool_registry/)** - Intelligent tool selection and management system

### 🌐 Integrations
- **[mcp/](mcp/)** - Model Context Protocol (MCP) integrations
- **[others/](others/)** - Additional integrations and utilities

## 🚀 Quick Start

### Smart Tool Registry

The Smart Tool Registry provides intelligent tool selection using Bayesian inference, performance tracking, and automatic discovery from MCP servers.

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/tools"
)

func main() {
    // Create registry with intelligent features
    config := &tools.SmartToolRegistryConfig{
        AutoDiscoveryEnabled:       true,
        PerformanceTrackingEnabled: true,
        FallbackEnabled:           true,
    }
    registry := tools.NewSmartToolRegistry(config)
    
    // Register tools
    searchTool := &MySearchTool{}
    registry.Register(searchTool)
    
    // Intelligent selection based on intent
    ctx := context.Background()
    tool, err := registry.SelectBest(ctx, "find user information")
    if err != nil {
        panic(err)
    }
    
    // Execute with performance tracking
    result, err := registry.ExecuteWithTracking(ctx, tool.Name(), params)
}
```

### Basic DSPy Workflow

```go
package main

import (
    "context"
    "github.com/XiaoConstantine/dspy-go/pkg/dspy"
    "github.com/XiaoConstantine/dspy-go/pkg/core"
)

func main() {
    // Create a language model
    lm := dspy.NewOpenAI(openai.DefaultConfig("your-api-key"))
    
    // Create a signature for classification
    signature := dspy.Signature{
        Input:  []string{"text"},
        Output: []string{"sentiment"},
        Instructions: "Classify the sentiment of the given text as positive, negative, or neutral.",
    }
    
    // Create a module
    classifier := dspy.NewPredict(lm, signature)
    
    // Use the module
    ctx := context.Background()
    result, err := classifier.Forward(ctx, map[string]interface{}{
        "text": "I love this product!",
    })
    
    fmt.Printf("Sentiment: %s\n", result["sentiment"])
}
```

## 📋 Example Index

### Smart Tool Registry Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [main.go](smart_tool_registry/main.go) | Basic Smart Tool Registry usage | Tool registration, intelligent selection, performance tracking |
| [advanced_example.go](smart_tool_registry/advanced_example.go) | Advanced features demonstration | Custom selectors, fallbacks, capability analysis |

**Features Demonstrated:**
- 🧠 **Bayesian Tool Selection**: Multi-factor scoring with configurable weights
- 📊 **Performance Tracking**: Real-time metrics and reliability scoring
- 🔍 **Capability Analysis**: Automatic capability extraction and matching
- 🔄 **Auto-Discovery**: MCP server integration for dynamic tool registration
- 🛡️ **Fallback Mechanisms**: Intelligent fallback selection when tools fail
- ⚙️ **Custom Configuration**: Configurable selection algorithms and weights

### Running the Examples

```bash
# Basic Smart Tool Registry example
cd examples/smart_tool_registry
go run main.go

# Advanced features demonstration
cd examples/smart_tool_registry
# Edit advanced_example.go to uncomment main() function
go run advanced_example.go
```

## 🏗️ Integrating Smart Tool Registry with DSPy-Go

The Smart Tool Registry seamlessly integrates with DSPy-Go workflows to provide intelligent tool management:

### 1. In DSPy Modules

```go
type IntelligentModule struct {
    registry *tools.SmartToolRegistry
    signature dspy.Signature
}

func (m *IntelligentModule) Forward(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
    // Use registry to select best tool for current task
    intent := fmt.Sprintf("process %v", inputs["task_type"])
    tool, err := m.registry.SelectBest(ctx, intent)
    if err != nil {
        return nil, err
    }
    
    // Execute with performance tracking
    result, err := m.registry.ExecuteWithTracking(ctx, tool.Name(), inputs)
    if err != nil {
        return nil, err
    }
    
    return result.Data.(map[string]interface{}), nil
}
```

### 2. In Workflow Builders

```go
import "github.com/XiaoConstantine/dspy-go/pkg/agents/workflows"

// Create workflow with intelligent tool selection
builder := workflows.NewWorkflowBuilder()
builder.SetToolRegistry(smartRegistry) // Use Smart Tool Registry

workflow := builder.
    AddStep("analyze", "analyze the input data").
    AddStep("process", "process the analysis results").
    AddStep("generate", "generate final output").
    Build()
```

### 3. With MCP Servers

```go
// Auto-discover tools from MCP servers
mcpDiscovery := &tools.MCPDiscoveryService{
    ServerURL: "http://localhost:8080/mcp",
    PollInterval: 30 * time.Second,
}

config := &tools.SmartToolRegistryConfig{
    AutoDiscoveryEnabled: true,
    MCPDiscovery: mcpDiscovery,
}

registry := tools.NewSmartToolRegistry(config)
// Tools will be automatically discovered and registered
```

## 🔧 Creating Custom Tools

Tools in DSPy-Go implement the `core.Tool` interface:

```go
type MyCustomTool struct {
    name string
}

func (t *MyCustomTool) Name() string {
    return t.name
}

func (t *MyCustomTool) Description() string {
    return "My custom tool description"
}

func (t *MyCustomTool) Metadata() *core.ToolMetadata {
    return &core.ToolMetadata{
        Name:         t.name,
        Description:  t.Description(),
        Capabilities: []string{"custom", "processing"},
        Version:      "1.0.0",
    }
}

func (t *MyCustomTool) CanHandle(ctx context.Context, intent string) bool {
    return strings.Contains(strings.ToLower(intent), "custom")
}

func (t *MyCustomTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
    // Your tool logic here
    return core.ToolResult{
        Data: map[string]interface{}{"result": "success"},
    }, nil
}

func (t *MyCustomTool) Validate(params map[string]interface{}) error {
    return nil // Implement validation logic
}

func (t *MyCustomTool) InputSchema() models.InputSchema {
    return models.InputSchema{
        Type: "object",
        Properties: map[string]models.ParameterSchema{
            "input": {
                Type:        "string",
                Description: "Input parameter",
                Required:    true,
            },
        },
    }
}
```

## 📖 Additional Resources

- **[Smart Tool Registry Documentation](smart_tool_registry/README.md)** - Detailed documentation for the Smart Tool Registry
- **[DSPy-Go Main Documentation](../README.md)** - Complete DSPy-Go framework documentation
- **[API Reference](../pkg/)** - Generated API documentation

## 🤝 Contributing

When adding new examples:

1. Create a dedicated directory for your example
2. Include a README.md with clear usage instructions
3. Add comprehensive comments explaining key concepts
4. Include both basic and advanced usage patterns
5. Ensure examples are self-contained and runnable

## 📄 License

All examples are provided under the same license as DSPy-Go. See the main repository LICENSE file for details.