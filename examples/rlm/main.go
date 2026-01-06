// Package main demonstrates the RLM (Recursive Language Model) module in dspy-go.
// RLM enables LLMs to explore large contexts programmatically through a Go REPL,
// making iterative queries to sub-LLMs until a final answer is reached.
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

func main() {
	// Setup logging
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logging.DEBUG,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	ctx := core.WithExecutionState(context.Background())

	// Get API key from environment
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Initialize the LLM factory
	llms.EnsureFactory()

	// Create the Anthropic LLM using dspy-go's built-in LLM
	llm, err := llms.NewAnthropicLLM(apiKey, "claude-haiku-4-5")
	if err != nil {
		fmt.Printf("Failed to create LLM: %v\n", err)
		os.Exit(1)
	}

	// Create a sample document with product reviews
	document := `
Review 1: This product is amazing, best purchase ever! Rating: 5 stars
Review 2: Terrible quality, broke after one day. Rating: 1 star
Review 3: It's okay, nothing special. Rating: 3 stars
Review 4: Absolutely love it, highly recommend! Rating: 5 stars
Review 5: Waste of money, very disappointed. Rating: 1 star
Review 6: Good value for the price. Rating: 4 stars
Review 7: Does exactly what it says. Rating: 4 stars
Review 8: Completely unusable, returned it. Rating: 1 star
Review 9: Best in class, exceeded expectations. Rating: 5 stars
Review 10: Average product, works fine. Rating: 3 stars
`
	// Repeat to make it longer (simulating a large context)
	document = strings.Repeat(document, 100)

	query := "What percentage of reviews are positive (4-5 stars) vs negative (1-2 stars)?"

	fmt.Println("=" + strings.Repeat("=", 59))
	fmt.Println("RLM (Recursive Language Model) Example")
	fmt.Println("=" + strings.Repeat("=", 59))
	fmt.Printf("Context size: %d characters\n", len(document))
	fmt.Printf("Query: %s\n", query)
	fmt.Println(strings.Repeat("-", 60))

	// Create RLM module - just pass the LLM, no adapter needed!
	rlmModule := rlm.NewFromLLM(
		llm,
		rlm.WithMaxIterations(10),
		rlm.WithVerbose(true),
		rlm.WithTimeout(5*time.Minute),
	)

	fmt.Println("\nStarting RLM completion...")
	start := time.Now()

	// Use Complete() for the RLM-specific interface
	result, err := rlmModule.Complete(ctx, document, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Final Answer: %s\n", result.Response)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Duration: %v\n", result.Duration)
	fmt.Printf("Total Duration: %v\n", time.Since(start))

	// Token usage
	fmt.Println("\nToken Usage:")
	fmt.Printf("  Prompt Tokens: %d\n", result.Usage.PromptTokens)
	fmt.Printf("  Completion Tokens: %d\n", result.Usage.CompletionTokens)
	fmt.Printf("  Total Tokens: %d\n", result.Usage.TotalTokens)

	// Show detailed token tracking
	tracker := rlmModule.GetTokenTracker()
	rootUsage := tracker.GetRootUsage()
	subUsage := tracker.GetSubUsage()
	subCalls := tracker.GetSubCalls()

	fmt.Println("\nDetailed Token Tracking:")
	fmt.Printf("  Root LLM: %d prompt, %d completion\n",
		rootUsage.PromptTokens, rootUsage.CompletionTokens)
	fmt.Printf("  Sub-LLM Calls: %d\n", len(subCalls))
	fmt.Printf("  Sub-LLM Tokens: %d prompt, %d completion\n",
		subUsage.PromptTokens, subUsage.CompletionTokens)
}
