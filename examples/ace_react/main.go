// Package main demonstrates ACE (Agentic Context Engineering) integrated with
// the ReAct agent. This shows how ACE enables self-improving agents that learn
// from their execution trajectories and apply those learnings to future tasks.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/agents/ace"
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

func main() {
	// Parse command line flags
	apiKey := flag.String("api-key", "", "API key for the LLM provider (defaults to GEMINI_API_KEY env)")
	model := flag.String("model", "gemini-flash", "Model to use (gemini-flash, gemini-pro)")
	learningsDir := flag.String("learnings-dir", "", "Directory to store learnings (temp dir if empty)")
	flag.Parse()

	// Get API key from flag or environment
	key := *apiKey
	if key == "" {
		key = os.Getenv("GEMINI_API_KEY")
	}
	if key == "" {
		fmt.Println("Please provide an API key with -api-key flag or GEMINI_API_KEY environment variable")
		os.Exit(1)
	}

	// Set up learnings directory
	var learningsPath string
	var cleanup func()
	if *learningsDir == "" {
		tmpDir, err := os.MkdirTemp("", "ace-react-demo-*")
		if err != nil {
			fmt.Printf("Failed to create temp dir: %v\n", err)
			os.Exit(1)
		}
		cleanup = func() { os.RemoveAll(tmpDir) }
		learningsPath = filepath.Join(tmpDir, "learnings.md")
	} else {
		if err := os.MkdirAll(*learningsDir, 0755); err != nil {
			fmt.Printf("Failed to create learnings dir: %v\n", err)
			os.Exit(1)
		}
		learningsPath = filepath.Join(*learningsDir, "learnings.md")
		cleanup = func() {}
	}
	defer cleanup()

	fmt.Println("===============================================")
	fmt.Println("  ACE + ReAct Agent Demo")
	fmt.Println("  Self-Improving Agent with Learning Memory")
	fmt.Println("===============================================")
	fmt.Printf("Learnings file: %s\n\n", learningsPath)

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

	llm, err := llms.NewGeminiLLM(key, core.ModelID(modelName))
	if err != nil {
		fmt.Printf("Failed to configure LLM: %v\n", err)
		os.Exit(1)
	}

	// Configure ACE with lower thresholds for demo visibility
	aceConfig := ace.Config{
		Enabled:             true,
		LearningsPath:       learningsPath,
		AsyncReflection:     false, // Sync for demo visibility
		CurationFrequency:   1,     // Process immediately for demo
		MinConfidence:       0.4,   // Lower threshold to capture more patterns
		MaxTokens:           80000,
		PruneMinRatio:       0.2,
		PruneMinUsage:       2,
		SimilarityThreshold: 0.85,
	}

	// Create ReAct agent with ACE enabled
	agent := react.NewReActAgent(
		"ace-demo-agent",
		"Self-Improving Research Assistant",
		react.WithExecutionMode(react.ModeReAct),
		react.WithMaxIterations(5), // Lower iterations to keep demos short
		react.WithTimeout(2*time.Minute),
		react.WithReflection(true, 3), // Enable reflection for richer insights
		react.WithACE(aceConfig),      // Enable ACE!
	)

	// Create typed signature for the agent
	typedSignature := core.NewTypedSignature[ReActInput, ReActOutput]().WithInstruction(`You are a self-improving research assistant with access to these tools:
- search: Search for information (use exact name: "search")
- calculator: Perform mathematical calculations (use exact name: "calculator")

IMPORTANT: Be efficient! Complete tasks in as few steps as possible.
- For simple questions, use one tool call then provide the answer
- For calculations, use calculator once then respond
- Do NOT make unnecessary repeat searches

When you cite learnings from past experiences, reference them by their ID (e.g., [L001] or [M001]).

Use XML format for actions: <action><tool_name>toolname</tool_name><parameters>{"param": "value"}</parameters></action>`)

	signature := typedSignature.ToLegacySignature()

	// Initialize the agent with LLM and signature
	if err := agent.Initialize(llm, signature); err != nil {
		fmt.Printf("Failed to initialize agent: %v\n", err)
		os.Exit(1)
	}

	// Register tools (including one that can fail)
	registerTools(agent)

	// Create execution context
	ctx := core.WithExecutionState(context.Background())

	// ================================================================
	// Part 1: Seed initial learnings to demonstrate the full cycle
	// ================================================================
	fmt.Println("PHASE 1: Seeding Initial Learnings")
	fmt.Println("-----------------------------------")
	fmt.Println("In production, these would come from past executions.")
	fmt.Println("Here we seed some to show the injection mechanism.")
	fmt.Println()

	seedLearnings(learningsPath)

	// Show seeded learnings
	file := ace.NewLearningsFile(learningsPath)
	learnings, _ := file.Load()
	fmt.Printf("Seeded %d learnings:\n", len(learnings))
	for _, l := range learnings {
		fmt.Printf("  [%s] %s\n", l.ShortCode(), l.Content)
	}

	// ================================================================
	// Part 2: Run tasks that will generate new learnings
	// ================================================================
	fmt.Println("\n\nPHASE 2: Running Tasks (building new learnings)")
	fmt.Println("------------------------------------------------")

	tasks := []struct {
		name        string
		description string
		expectFail  bool
	}{
		{"simple_calc", "Calculate 10 + 5", false},
		{"weather", "What is the weather in NYC?", false},
		{"fail_task", "Search for data in the broken_database", true}, // Will fail!
		{"math_chain", "Calculate 100 divided by 4", false},
	}

	for i, task := range tasks {
		icon := "📋"
		if task.expectFail {
			icon = "⚠️"
		}
		fmt.Printf("\n%s [Task %d/%d] %s\n", icon, i+1, len(tasks), task.name)
		fmt.Printf("   Query: %s\n", task.description)

		input := map[string]interface{}{
			"task": task.description,
		}

		startTime := time.Now()
		result, execErr := agent.Execute(ctx, input)
		duration := time.Since(startTime)

		if execErr != nil {
			fmt.Printf("   ❌ ERROR: %v\n", execErr)
		} else {
			if resultValue, ok := result["result"]; ok {
				resultStr := fmt.Sprintf("%v", resultValue)
				if len(resultStr) > 80 {
					resultStr = resultStr[:80] + "..."
				}
				fmt.Printf("   ✅ Result: %s\n", resultStr)
			}
		}
		fmt.Printf("   ⏱️  Duration: %v\n", duration)

		// Brief pause between tasks
		if i < len(tasks)-1 {
			time.Sleep(300 * time.Millisecond)
		}
	}

	// ================================================================
	// Part 3: Display final learnings
	// ================================================================
	fmt.Println("\n\n===============================================")
	fmt.Println("  PHASE 3: Final Learnings State")
	fmt.Println("===============================================")

	// Reload learnings
	learnings, err = file.Load()
	if err != nil {
		fmt.Printf("Error loading learnings: %v\n", err)
	} else {
		fmt.Printf("\nTotal learnings: %d\n\n", len(learnings))

		// Group by category
		strategies := []ace.Learning{}
		mistakes := []ace.Learning{}
		other := []ace.Learning{}

		for _, l := range learnings {
			switch l.Category {
			case "strategies", "patterns":
				strategies = append(strategies, l)
			case "mistakes":
				mistakes = append(mistakes, l)
			default:
				other = append(other, l)
			}
		}

		if len(strategies) > 0 {
			fmt.Println("✅ STRATEGIES (what works well):")
			for _, l := range strategies {
				fmt.Printf("   [%s] %s (%.0f%% success, %d uses)\n",
					l.ShortCode(), l.Content, l.SuccessRate()*100, l.TotalUses())
			}
		}

		if len(mistakes) > 0 {
			fmt.Println("\n❌ MISTAKES (what to avoid):")
			for _, l := range mistakes {
				fmt.Printf("   [%s] %s (%.0f%% success, %d uses)\n",
					l.ShortCode(), l.Content, l.SuccessRate()*100, l.TotalUses())
			}
		}

		if len(other) > 0 {
			fmt.Println("\n📝 OTHER:")
			for _, l := range other {
				fmt.Printf("   [%s] %s (%.0f%% success, %d uses)\n",
					l.ShortCode(), l.Content, l.SuccessRate()*100, l.TotalUses())
			}
		}
	}

	// ================================================================
	// Part 4: Show context injection preview
	// ================================================================
	fmt.Println("\n===============================================")
	fmt.Println("  Context Injection Preview")
	fmt.Println("===============================================")
	fmt.Println("This is what gets added to future agent prompts:")
	fmt.Println()

	contextContent := ace.FormatForInjection(learnings)
	if contextContent == "" {
		fmt.Println("(No learnings to inject)")
	} else {
		// Add indentation for display
		lines := strings.Split(contextContent, "\n")
		for _, line := range lines {
			fmt.Printf("  %s\n", line)
		}
	}

	// ================================================================
	// Part 5: Show reflection insights
	// ================================================================
	if agent.Reflector != nil {
		reflections := agent.Reflector.GetTopReflections(5)
		if len(reflections) > 0 {
			fmt.Println("\n===============================================")
			fmt.Println("  Self-Reflector Insights")
			fmt.Println("===============================================")
			for i, ref := range reflections {
				fmt.Printf("%d. [%.0f%%] %s\n", i+1, ref.Confidence*100, ref.Insight)
			}
		}
	}

	fmt.Println("\n===============================================")
	fmt.Println("  Demo Complete!")
	fmt.Println("===============================================")
	fmt.Println("Key concepts demonstrated:")
	fmt.Println("  1. Learnings persist across agent executions")
	fmt.Println("  2. Successful patterns become 'strategies'")
	fmt.Println("  3. Failures become 'mistakes to avoid'")
	fmt.Println("  4. Learnings are injected into future prompts")
	fmt.Println("  5. Citation tracking updates learning success rates")
	fmt.Println("\nRun with --learnings-dir=./my_learnings to persist between runs!")
}

// seedLearnings creates initial learnings to demonstrate the full cycle.
func seedLearnings(path string) {
	file := ace.NewLearningsFile(path)

	// Create some seed learnings
	learnings := []ace.Learning{
		{
			ID:       "strategies-00001",
			Category: "strategies",
			Content:  "Use calculator tool for any arithmetic operations",
			Helpful:  3,
			Harmful:  0,
		},
		{
			ID:       "strategies-00002",
			Category: "strategies",
			Content:  "For weather queries, search once and provide the answer immediately",
			Helpful:  2,
			Harmful:  0,
		},
		{
			ID:       "mistakes-00001",
			Category: "mistakes",
			Content:  "Avoid broken_database tool - it always times out",
			Helpful:  0,
			Harmful:  2,
		},
	}

	if err := file.Save(learnings); err != nil {
		fmt.Printf("Warning: Failed to seed learnings: %v\n", err)
	}
}

// registerTools adds the available tools to the agent.
func registerTools(agent *react.ReActAgent) {
	// Search tool
	searchTool := tools.NewFuncTool(
		"search",
		"Search for information on the internet",
		models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"query": {
					Type:        "string",
					Description: "The search query",
					Required:    true,
				},
			},
		},
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			query, _ := args["query"].(string)

			// Check for broken database query - simulates a failure
			if strings.Contains(strings.ToLower(query), "broken_database") {
				return &models.CallToolResult{
					Content: []models.Content{
						models.TextContent{Type: "text", Text: "ERROR: Connection timeout - database unreachable"},
					},
					IsError: true,
				}, nil
			}

			// Simulated search results
			results := map[string]string{
				"weather": "NYC Weather: Sunny, 72F (22C), clear skies",
				"nyc":     "NYC Weather: Sunny, 72F (22C), clear skies",
				"tokyo":   "Tokyo population: approximately 13.96 million (2024)",
				"ml":      "Machine learning: AI technique for pattern recognition",
			}

			result := fmt.Sprintf("Search results for '%s': Found relevant information.", query)
			queryLower := strings.ToLower(query)
			for key, val := range results {
				if strings.Contains(queryLower, key) {
					result = val
					break
				}
			}

			return &models.CallToolResult{
				Content: []models.Content{
					models.TextContent{Type: "text", Text: result},
				},
			}, nil
		},
	)

	// Calculator tool
	calculatorTool := tools.NewFuncTool(
		"calculator",
		"Perform mathematical calculations",
		models.InputSchema{
			Type: "object",
			Properties: map[string]models.ParameterSchema{
				"expression": {
					Type:        "string",
					Description: "The mathematical expression to calculate",
					Required:    true,
				},
			},
		},
		func(ctx context.Context, args map[string]interface{}) (*models.CallToolResult, error) {
			expression, _ := args["expression"].(string)

			// Simple expression evaluation for demo
			result := "Result: "
			exprLower := strings.ToLower(expression)

			switch {
			case strings.Contains(exprLower, "10") && strings.Contains(exprLower, "5"):
				result += "15"
			case strings.Contains(exprLower, "100") && strings.Contains(exprLower, "4"):
				result += "25"
			case strings.Contains(exprLower, "25") && strings.Contains(exprLower, "4"):
				result += "100"
			default:
				result += "42 (simulated)"
			}

			return &models.CallToolResult{
				Content: []models.Content{
					models.TextContent{Type: "text", Text: result},
				},
			}, nil
		},
	)

	toolsToRegister := []core.Tool{searchTool, calculatorTool}
	for _, tool := range toolsToRegister {
		if err := agent.RegisterTool(tool); err != nil {
			fmt.Printf("Warning: Failed to register tool '%s': %v\n", tool.Name(), err)
		}
	}
}
