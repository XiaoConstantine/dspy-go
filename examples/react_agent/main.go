package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/react"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/tools"

	models "github.com/XiaoConstantine/mcp-go/pkg/model"
)

// ReActInput defines the input structure for our ReAct agent.
type ReActInput struct {
	Task string `json:"task" dspy:"task,required" description:"The task to be completed by the agent"`
}

// ReActOutput defines the output structure for our ReAct agent.
type ReActOutput struct {
	Result string `json:"result" dspy:"result" description:"The final result or answer from completing the task"`
}

// SearchTool simulates a search tool.
type SearchTool struct {
	name        string
	description string
}

func NewSearchTool() *SearchTool {
	return &SearchTool{
		name:        "search",
		description: "Search for information on the internet or in databases",
	}
}

func (st *SearchTool) Name() string {
	return st.name
}

func (st *SearchTool) Description() string {
	return st.description
}

func (st *SearchTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"query": {
				Type:        "string",
				Description: "The search query",
				Required:    true,
			},
		},
	}
}

func (st *SearchTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query parameter is required")
	}

	// Simulate search results
	result := fmt.Sprintf("Search results for '%s': Found relevant information about the topic.", query)

	return &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: result,
			},
		},
		IsError: false,
	}, nil
}

func (st *SearchTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := st.Call(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}
	return core.ToolResult{Data: result}, nil
}

func (st *SearchTool) Validate(params map[string]interface{}) error {
	if _, ok := params["query"]; !ok {
		return fmt.Errorf("query parameter is required")
	}
	return nil
}

// CalculatorTool simulates a calculator tool.
type CalculatorTool struct {
	name        string
	description string
}

func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{
		name:        "calculator",
		description: "Perform mathematical calculations",
	}
}

func (ct *CalculatorTool) Name() string {
	return ct.name
}

func (ct *CalculatorTool) Description() string {
	return ct.description
}

func (ct *CalculatorTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"expression": {
				Type:        "string",
				Description: "The mathematical expression to calculate",
				Required:    true,
			},
		},
	}
}

func (ct *CalculatorTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	expression, ok := args["expression"].(string)
	if !ok {
		return nil, fmt.Errorf("expression parameter is required")
	}

	// Simple calculation simulation
	result := fmt.Sprintf("Calculated '%s' = 42", expression)

	return &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: result,
			},
		},
		IsError: false,
	}, nil
}

func (ct *CalculatorTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := ct.Call(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}
	return core.ToolResult{Data: result}, nil
}

func (ct *CalculatorTool) Validate(params map[string]interface{}) error {
	if _, ok := params["expression"]; !ok {
		return fmt.Errorf("expression parameter is required")
	}
	return nil
}

// SummarizerTool simulates a text summarization tool.
type SummarizerTool struct {
	name        string
	description string
}

func NewSummarizerTool() *SummarizerTool {
	return &SummarizerTool{
		name:        "summarize",
		description: "Summarize long text content",
	}
}

func (st *SummarizerTool) Name() string {
	return st.name
}

func (st *SummarizerTool) Description() string {
	return st.description
}

func (st *SummarizerTool) InputSchema() models.InputSchema {
	return models.InputSchema{
		Type: "object",
		Properties: map[string]models.ParameterSchema{
			"text": {
				Type:        "string",
				Description: "The text to summarize",
				Required:    true,
			},
		},
	}
}

func (st *SummarizerTool) Call(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
	text, ok := args["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text parameter is required")
	}

	// Simulate summarization
	result := fmt.Sprintf("Summary of text (%d chars): Key points extracted and condensed.", len(text))

	return &models.CallToolResult{
		Content: []models.Content{
			models.TextContent{
				Type: "text",
				Text: result,
			},
		},
		IsError: false,
	}, nil
}

func (st *SummarizerTool) Execute(ctx context.Context, params map[string]interface{}) (core.ToolResult, error) {
	result, err := st.Call(ctx, params)
	if err != nil {
		return core.ToolResult{}, err
	}
	return core.ToolResult{Data: result}, nil
}

func (st *SummarizerTool) Validate(params map[string]interface{}) error {
	if _, ok := params["text"]; !ok {
		return fmt.Errorf("text parameter is required")
	}
	return nil
}

func main() {
	// Parse command line flags
	apiKey := flag.String("api-key", "", "API key for the LLM provider")
	model := flag.String("model", "gemini-flash", "Model to use (gemini-flash, gemini-pro)")
	mode := flag.String("mode", "react", "Execution mode (react, rewoo, hybrid)")
	reflectionFlag := flag.Bool("reflection", true, "Enable self-reflection")
	planningFlag := flag.Bool("planning", true, "Enable task planning")
	memoryFlag := flag.Bool("memory", true, "Enable memory optimization")
	flag.Parse()

	if *apiKey == "" {
		fmt.Println("Please provide an API key with -api-key flag")
		os.Exit(1)
	}

	// Initialize LLM
	llms.EnsureFactory()
	var modelName string
	switch *model {
	case "gemini-flash":
		modelName = "gemini-2.0-flash"
	case "gemini-pro":
		modelName = "gemini-2.0-pro"
	default:
		fmt.Printf("Unsupported model: %s\n", *model)
		os.Exit(1)
	}

	llm, err := llms.NewGeminiLLM(*apiKey, core.ModelID(modelName))
	if err != nil {
		fmt.Printf("Failed to configure LLM: %v\n", err)
		os.Exit(1)
	}

	// Create execution context
	ctx := core.WithExecutionState(context.Background())

	// Determine execution mode
	var executionMode react.ExecutionMode
	switch *mode {
	case "react":
		executionMode = react.ModeReAct
	case "rewoo":
		executionMode = react.ModeReWOO
	case "hybrid":
		executionMode = react.ModeHybrid
	default:
		fmt.Printf("Invalid execution mode: %s\n", *mode)
		os.Exit(1)
	}

	// Create ReAct agent with configuration
	var opts []react.Option
	opts = append(opts, react.WithExecutionMode(executionMode))
	opts = append(opts, react.WithMaxIterations(10))
	opts = append(opts, react.WithTimeout(5*time.Minute))

	if *reflectionFlag {
		opts = append(opts, react.WithReflection(true, 3))
	}

	if *planningFlag {
		opts = append(opts, react.WithPlanning(react.Interleaved, 5))
	}

	if *memoryFlag {
		opts = append(opts, react.WithMemoryOptimization(24*time.Hour, 0.3))
	}

	agent := react.NewReActAgent("demo-agent", "Research Assistant", opts...)

	// Create typed signature for the agent
	typedSignature := core.NewTypedSignature[ReActInput, ReActOutput]().WithInstruction(`You are a helpful research assistant with access to these tools:
- search: Search for information (use exact name: "search")
- calculator: Perform mathematical calculations (use exact name: "calculator")
- summarize: Summarize text content (use exact name: "summarize")

Always use the exact tool names as specified above. Use XML format for actions: <action><tool_name>toolname</tool_name><parameters>{"param": "value"}</parameters></action>`)

	// Convert to legacy signature for compatibility
	signature := typedSignature.ToLegacySignature()

	// Initialize the agent with LLM and signature
	err = agent.Initialize(llm, signature)
	if err != nil {
		fmt.Printf("Failed to initialize agent: %v\n", err)
		os.Exit(1)
	}

	// Create tool instances once for efficiency
	searchToolInstance := NewSearchTool()
	searchTool := tools.NewFuncTool("search", "Search for information",
		searchToolInstance.InputSchema(),
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return searchToolInstance.Call(ctx, args)
		})

	calculatorToolInstance := NewCalculatorTool()
	calculatorTool := tools.NewFuncTool("calculator", "Perform calculations",
		calculatorToolInstance.InputSchema(),
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return calculatorToolInstance.Call(ctx, args)
		})

	summarizerToolInstance := NewSummarizerTool()
	summarizerTool := tools.NewFuncTool("summarize", "Summarize text",
		summarizerToolInstance.InputSchema(),
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			return summarizerToolInstance.Call(ctx, args)
		})

	// Register tools using a loop to reduce duplication
	toolsToRegister := []core.Tool{searchTool, calculatorTool, summarizerTool}
	for _, tool := range toolsToRegister {
		if err := agent.RegisterTool(tool); err != nil {
			fmt.Printf("Failed to register tool '%s': %v\n", tool.Name(), err)
			os.Exit(1)
		}
	}

	// Demo tasks
	tasks := []string{
		"Search for information about artificial intelligence and summarize the key findings.",
		"Calculate the result of 15 * 23 + 47 and then search for historical facts about that number.",
		"Find information about climate change and create a summary of the main impacts.",
		"Research the latest developments in quantum computing and provide a brief overview.",
	}

	fmt.Printf("🤖 ReAct Agent Demo\n")
	fmt.Printf("==================\n")
	fmt.Printf("Mode: %s\n", *mode)
	fmt.Printf("Reflection: %v\n", *reflectionFlag)
	fmt.Printf("Planning: %v\n", *planningFlag)
	fmt.Printf("Memory Optimization: %v\n", *memoryFlag)
	fmt.Printf("\n")

	// Execute tasks
	for i, task := range tasks {
		fmt.Printf("📋 Task %d: %s\n", i+1, task)
		fmt.Printf("----------------------------------------\n")

		input := map[string]interface{}{
			"task": task,
		}

		startTime := time.Now()
		result, err := agent.Execute(ctx, input)
		duration := time.Since(startTime)

		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
		} else {
			// The result might be the full prediction map or just the result value
			// The result is expected to contain a "result" key as per the ReActOutput schema.
			if resultValue, ok := result["result"]; ok {
				fmt.Printf("✅ Result: %v\n", resultValue)
			} else {
				// If the expected key is not found, display the raw result for debugging.
				fmt.Printf("✅ Result (raw): %v\n", result)
			}
		}

		fmt.Printf("⏱️  Duration: %v\n", duration)

		// Show execution history for this task
		history := agent.GetExecutionHistory()
		if len(history) > 0 {
			lastExecution := history[len(history)-1]
			fmt.Printf("🔄 Actions taken: %d\n", len(lastExecution.Actions))
			if len(lastExecution.Reflections) > 0 {
				fmt.Printf("💭 Reflections: %d\n", len(lastExecution.Reflections))
			}
		}

		fmt.Printf("\n")

		// Wait between tasks
		if i < len(tasks)-1 {
			time.Sleep(1 * time.Second)
		}
	}

	// Show agent statistics
	fmt.Printf("📊 Agent Statistics\n")
	fmt.Printf("==================\n")

	history := agent.GetExecutionHistory()
	fmt.Printf("Total executions: %d\n", len(history))

	// Only calculate stats manually if reflection is disabled
	if !*reflectionFlag && len(history) > 0 {
		successful := 0
		totalActions := 0
		for _, exec := range history {
			if exec.Success {
				successful++
			}
			totalActions += len(exec.Actions)
		}

		fmt.Printf("Success rate: %.1f%%\n", float64(successful)/float64(len(history))*100)
		fmt.Printf("Average actions per task: %.1f\n", float64(totalActions)/float64(len(history)))
	}

	// Show reflection insights if enabled
	if *reflectionFlag && agent.Reflector != nil {
		reflections := agent.Reflector.GetTopReflections(3)
		if len(reflections) > 0 {
			fmt.Printf("\n🧠 Top Insights:\n")
			for i, ref := range reflections {
				fmt.Printf("%d. %s (confidence: %.2f)\n", i+1, ref.Insight, ref.Confidence)
			}
		}

		metrics := agent.Reflector.GetMetrics()
		if metrics.TotalExecutions > 0 {
			fmt.Printf("\n📈 Performance Metrics:\n")
			fmt.Printf("Total executions: %d\n", metrics.TotalExecutions)
			fmt.Printf("Success rate: %.1f%%\n", float64(metrics.SuccessfulRuns)/float64(metrics.TotalExecutions)*100)
			fmt.Printf("Average iterations: %.2f\n", metrics.AverageIterations)
		}
	}

	// Show memory statistics if enabled
	if *memoryFlag && agent.Optimizer != nil {
		memStats := agent.Optimizer.GetStatistics()
		fmt.Printf("\n🧠 Memory Statistics:\n")
		fmt.Printf("Total items: %v\n", memStats["total_items"])
		fmt.Printf("Categories: %v\n", memStats["categories"])
		if avgImportance, ok := memStats["avg_importance"]; ok {
			fmt.Printf("Average importance: %.3f\n", avgImportance)
		}
	}

	fmt.Printf("\n🎉 Demo completed successfully!\n")
}

// Helper methods for accessing internal components
