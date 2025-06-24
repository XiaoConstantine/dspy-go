// Package dspy is a Go implementation of the DSPy framework for using language models
// to solve complex tasks through composable steps and prompting techniques.
//
// DSPy-Go provides a collection of modules, optimizers, and tools for building
// reliable LLM-powered applications. It focuses on making it easy to:
//   - Break down complex tasks into modular steps
//   - Optimize prompts and chain-of-thought reasoning
//   - Build flexible agent-based systems
//   - Handle common LLM interaction patterns
//   - Evaluate and improve system performance
//
// Key Components:
//
//   - Core: Fundamental abstractions like Module, Signature, LLM and Program
//     for defining and executing LLM-based workflows.
//
//   - Modules: Building blocks for composing LLM workflows:
//     * Predict: Basic prediction module for simple LLM interactions
//     * ChainOfThought: Implements step-by-step reasoning with rationale tracking
//     * ReAct: Implements Reasoning and Acting with tool integration
//     * Refine: Quality improvement through multiple attempts with reward functions and temperature variation
//     * Parallel: Concurrent execution wrapper for batch processing with any module
//     * MultiChainComparison: Compares multiple reasoning attempts and synthesizes holistic evaluation
//
//   - Optimizers: Tools for improving prompt effectiveness:
//     * BootstrapFewShot: Automatically selects high-quality examples for few-shot learning
//     * MIPRO: Multi-step interactive prompt optimization
//     * Copro: Collaborative prompt optimization
//     * SIMBA: Stochastic Introspective Mini-Batch Ascent with self-analysis
//     * TPE: Tree-structured Parzen Estimator for Bayesian optimization
//
//   - Agents: Advanced patterns for building sophisticated AI systems:
//     * Memory: Different memory implementations for tracking conversation history
//     * Tools: Integration with external tools and APIs, including:
//       - Smart Tool Registry: Intelligent tool selection using Bayesian inference
//       - Performance Tracking: Real-time metrics and reliability scoring
//       - Auto-Discovery: Dynamic tool registration from MCP servers
//       - MCP (Model Context Protocol) support for seamless integrations
//       - Tool Chaining: Sequential execution of tools in pipelines with data transformation
//       - Tool Composition: Combining multiple tools into reusable composite units
//       - Parallel Execution: Advanced parallel tool execution with intelligent scheduling
//       - Dependency Resolution: Automatic execution planning based on tool dependencies
//     * Workflows:
//       - Chain: Sequential execution of steps
//       - Parallel: Concurrent execution of multiple workflow steps
//       - Router: Dynamic routing based on classification
//       - Advanced Patterns: ForEach, While, Until loops with conditional execution
//     * Orchestrator: Flexible task decomposition and execution
//
//   - Integration with multiple LLM providers:
//     * Anthropic Claude
//     * Google Gemini
//     * Ollama
//     * LlamaCPP
//
// Simple Example:
//
//	import (
//	    "context"
//	    "fmt"
//	    "log"
//
//	    "github.com/XiaoConstantine/dspy-go/pkg/core"
//	    "github.com/XiaoConstantine/dspy-go/pkg/llms"
//	    "github.com/XiaoConstantine/dspy-go/pkg/modules"
//	    "github.com/XiaoConstantine/dspy-go/pkg/config"
//	)
//
//	func main() {
//	    // Configure the default LLM
//	    llms.EnsureFactory()
//	    err := config.ConfigureDefaultLLM("your-api-key", core.ModelAnthropicSonnet)
//	    if err != nil {
//	        log.Fatalf("Failed to configure LLM: %v", err)
//	    }
//
//	    // Create a signature for question answering
//	    signature := core.NewSignature(
//	        []core.InputField{{Field: core.Field{Name: "question"}}},
//	        []core.OutputField{{Field: core.Field{Name: "answer"}}},
//	    )
//
//	    // Create a ChainOfThought module
//	    cot := modules.NewChainOfThought(signature)
//
//	    // Create a program
//	    program := core.NewProgram(
//	        map[string]core.Module{"cot": cot},
//	        func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
//	            return cot.Process(ctx, inputs)
//	        },
//	    )
//
//	    // Execute the program
//	    result, err := program.Execute(context.Background(), map[string]interface{}{
//	        "question": "What is the capital of France?",
//	    })
//	    if err != nil {
//	        log.Fatalf("Error executing program: %v", err)
//	    }
//
//	    fmt.Printf("Answer: %s\n", result["answer"])
//	}
//
// Advanced Features:
//
//   - Tracing and Logging: Detailed tracing and structured logging for debugging and optimization
//     Execution context is tracked and passed through the pipeline for debugging and analysis.
//
//   - Error Handling: Comprehensive error management with custom error types and centralized handling
//
//   - Metric-Based Optimization: Improve module performance based on custom evaluation metrics
//
//   - Smart Tool Management: Intelligent tool selection, performance tracking, auto-discovery, 
//     chaining, and composition for building complex tool workflows
//
//   - Custom Tool Integration: Extend ReAct modules with domain-specific tools
//
//   - Workflow Retry Logic: Resilient execution with configurable retry mechanisms and backoff strategies
//
//   - Streaming Support: Process LLM outputs incrementally as they're generated
//
//   - Data Storage: Integration with various storage backends for persistence of examples and results
//
//   - Dataset Management: Built-in support for downloading and managing datasets like GSM8K and HotPotQA
//
//   - Arrow Support: Integration with Apache Arrow for efficient data handling and processing
//
// Working with Smart Tool Registry:
//
//	import "github.com/XiaoConstantine/dspy-go/pkg/tools"
//
//	// Create intelligent tool registry
//	config := &tools.SmartToolRegistryConfig{
//	    AutoDiscoveryEnabled:       true,
//	    PerformanceTrackingEnabled: true,
//	    FallbackEnabled:           true,
//	}
//	registry := tools.NewSmartToolRegistry(config)
//
//	// Register tools
//	registry.Register(mySearchTool)
//	registry.Register(myAnalysisTool)
//
//	// Intelligent tool selection based on intent
//	tool, err := registry.SelectBest(ctx, "find user information")
//	result, err := registry.ExecuteWithTracking(ctx, tool.Name(), params)
//
//Working with Tool Chaining and Composition:
//
//	// Tool Chaining - Sequential execution with data transformation
//	pipeline, err := tools.NewPipelineBuilder("data_processing", registry).
//	    Step("data_extractor").
//	    StepWithTransformer("data_validator", tools.TransformExtractField("result")).
//	    ConditionalStep("data_enricher", tools.ConditionExists("validated")).
//	    StepWithRetries("data_transformer", 3).
//	    FailFast().
//	    EnableCaching().
//	    Build()
//
//	result, err := pipeline.Execute(ctx, input)
//
//	// Dependency-aware execution with automatic parallelization
//	graph := tools.NewDependencyGraph()
//	graph.AddNode(&tools.DependencyNode{
//	    ToolName: "extractor",
//	    Dependencies: []string{},
//	    Outputs: []string{"raw_data"},
//	})
//	graph.AddNode(&tools.DependencyNode{
//	    ToolName: "validator", 
//	    Dependencies: []string{"extractor"},
//	    Inputs: []string{"raw_data"},
//	})
//
//	depPipeline, err := tools.NewDependencyPipeline("smart_pipeline", registry, graph, options)
//	result, err := depPipeline.ExecuteWithDependencies(ctx, input)
//
//	// Parallel execution with advanced scheduling
//	executor := tools.NewParallelExecutor(registry, 4)
//	tasks := []*tools.ParallelTask{
//	    {ID: "task1", ToolName: "analyzer", Input: data1, Priority: 1},
//	    {ID: "task2", ToolName: "processor", Input: data2, Priority: 2},
//	}
//	results, err := executor.ExecuteParallel(ctx, tasks, &tools.PriorityScheduler{})
//
//	// Tool Composition - Create reusable composite tools
//	type CompositeTool struct {
//	    name     string
//	    pipeline *tools.ToolPipeline
//	}
//	
//	textProcessor, err := NewCompositeTool("text_processor", registry, 
//	    func(builder *tools.PipelineBuilder) *tools.PipelineBuilder {
//	        return builder.Step("uppercase").Step("reverse").Step("length")
//	    })
//	
//	// Register and use composite tool like any other tool
//	registry.Register(textProcessor)
//	result, err := textProcessor.Execute(ctx, input)
//
// Working with Workflows:
//
//	// Chain workflow example
//	workflow := workflows.NewChainWorkflow(store)
//	workflow.AddStep(&workflows.Step{
//	    ID: "step1",
//	    Module: modules.NewPredict(signature1),
//	})
//	workflow.AddStep(&workflows.Step{
//	    ID: "step2", 
//	    Module: modules.NewPredict(signature2),
//	    // Configurable retry logic
//	    RetryConfig: &workflows.RetryConfig{
//	        MaxAttempts: 3,
//	        BackoffMultiplier: 2.0,
//	    },
//	    // Conditional execution
//	    Condition: func(state map[string]interface{}) bool {
//	        return someCondition(state)
//	    },
//	})
//
// For more examples and detailed documentation, visit:
// https://github.com/XiaoConstantine/dspy-go
//
// DSPy-Go is released under the MIT License.
package dspy
