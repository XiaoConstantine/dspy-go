package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

func main() {
	apiKey := flag.String("api-key", "", "API Key for the LLM provider")
	flag.Parse()

	// Initialize context and LLM
	ctx := core.WithExecutionState(context.Background())

	// Configure LLM
	llms.EnsureFactory()

	err := core.ConfigureDefaultLLM(*apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Run different multi-chain comparison examples
	fmt.Println("=== DSPy-Go MultiChainComparison Module Examples ===")

	// Example 1: Mathematical reasoning comparison
	runMathReasoningExample(ctx)

	// Example 2: Text analysis with multiple perspectives
	runTextAnalysisExample(ctx)

	// Example 3: Problem solving with different approaches
	runProblemSolvingExample(ctx)
}

// Example 1: Mathematical reasoning comparison.
func runMathReasoningExample(ctx context.Context) {
	fmt.Println("1. Mathematical Reasoning Comparison")
	fmt.Println("===================================")

	// Create a signature for mathematical reasoning
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("problem", core.WithDescription("The mathematical problem to solve"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The final numerical answer"))},
		},
	).WithInstruction("Solve the mathematical problem step by step and provide the final answer.")

	// Create MultiChainComparison with 3 reasoning attempts
	multiChain := modules.NewMultiChainComparison(signature, 3, 0.7)

	// Simulate three different reasoning attempts for the same problem
	completions := []map[string]interface{}{
		{
			"rationale": "use algebraic manipulation to solve for x",
			"answer":    "12",
		},
		{
			"reasoning": "apply the quadratic formula step by step",
			"answer":    "12",
		},
		{
			"rationale": "factor the equation and find the roots",
			"answer":    "10", // Intentionally different answer to show comparison
		},
	}

	start := time.Now()

	// Process the multi-chain comparison
	result, err := multiChain.Process(ctx, map[string]interface{}{
		"problem":     "Solve for x: x² - 7x - 30 = 0",
		"completions": completions,
	})
	if err != nil {
		log.Printf("Multi-chain comparison failed: %v", err)
		return
	}

	duration := time.Since(start)

	fmt.Printf("Completed comparison in %v\n\n", duration)
	fmt.Printf("Holistic Reasoning: %s\n", result["rationale"])
	fmt.Printf("Final Answer: %s\n\n", result["answer"])
}

// Example 2: Text analysis with multiple perspectives.
func runTextAnalysisExample(ctx context.Context) {
	fmt.Println("2. Text Analysis with Multiple Perspectives")
	fmt.Println("==========================================")

	// Create signature for sentiment analysis
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("text", core.WithDescription("Text to analyze"))},
		},
		[]core.OutputField{
			{Field: core.NewField("sentiment", core.WithDescription("Overall sentiment: positive, negative, or neutral"))},
			{Field: core.NewField("confidence", core.WithDescription("Confidence level: high, medium, or low"))},
		},
	).WithInstruction("Analyze the sentiment of the given text and provide a confidence assessment.")

	// Create MultiChainComparison with 4 reasoning attempts
	multiChain := modules.NewMultiChainComparison(signature, 4, 0.8)

	// Simulate four different analysis perspectives
	completions := []map[string]interface{}{
		{
			"rationale":  "focus on positive keywords and emotional tone",
			"sentiment":  "positive",
			"confidence": "high",
		},
		{
			"reasoning":  "analyze grammatical structure and context clues",
			"sentiment":  "positive",
			"confidence": "medium",
		},
		{
			"rationale":  "consider sarcasm and implicit meaning",
			"sentiment":  "neutral",
			"confidence": "low",
		},
		{
			"reasoning":  "evaluate overall message and intent",
			"sentiment":  "positive",
			"confidence": "high",
		},
	}

	start := time.Now()

	result, err := multiChain.Process(ctx, map[string]interface{}{
		"text":        "This product exceeded my expectations! The quality is fantastic and delivery was super quick. Highly recommend!",
		"completions": completions,
	})
	if err != nil {
		log.Printf("Text analysis comparison failed: %v", err)
		return
	}

	duration := time.Since(start)

	fmt.Printf("Completed analysis in %v\n\n", duration)
	fmt.Printf("Comprehensive Analysis: %s\n", result["rationale"])
	fmt.Printf("Final Sentiment: %s\n", result["sentiment"])
	fmt.Printf("Final Confidence: %s\n\n", result["confidence"])
}

// Example 3: Problem solving with different approaches.
func runProblemSolvingExample(ctx context.Context) {
	fmt.Println("3. Problem Solving with Different Approaches")
	fmt.Println("===========================================")

	// Create signature for strategic problem solving
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("scenario", core.WithDescription("The problem scenario to solve"))},
		},
		[]core.OutputField{
			{Field: core.NewField("strategy", core.WithDescription("The recommended strategy"))},
			{Field: core.NewField("reasoning", core.WithDescription("Explanation of the reasoning"))},
		},
	).WithInstruction("Analyze the scenario and recommend the best strategy with clear reasoning.")

	// Create MultiChainComparison with 3 reasoning attempts
	multiChain := modules.NewMultiChainComparison(signature, 3, 0.6)

	// Simulate three different problem-solving approaches
	completions := []map[string]interface{}{
		{
			"rationale": "prioritize immediate cost reduction and efficiency",
			"strategy":  "Implement automated systems to reduce labor costs",
			"reasoning": "Automation provides immediate cost savings and long-term efficiency gains",
		},
		{
			"rationale": "focus on customer satisfaction and retention strategies",
			"strategy":  "Invest in customer service improvements and loyalty programs",
			"reasoning": "Happy customers drive sustainable revenue growth",
		},
		{
			"rationale": "balance short-term and long-term business objectives",
			"strategy":  "Gradual optimization with phased implementation",
			"reasoning": "Balanced approach minimizes risk while ensuring steady progress",
		},
	}

	start := time.Now()

	result, err := multiChain.Process(ctx, map[string]interface{}{
		"scenario":    "A small business is struggling with declining profits and needs to decide between cutting costs, investing in growth, or maintaining status quo.",
		"completions": completions,
	})
	if err != nil {
		log.Printf("Problem solving comparison failed: %v", err)
		return
	}

	duration := time.Since(start)

	fmt.Printf("Completed strategic analysis in %v\n\n", duration)
	fmt.Printf("Holistic Assessment: %s\n", result["rationale"])
	fmt.Printf("Recommended Strategy: %s\n", result["strategy"])
	fmt.Printf("Comprehensive Reasoning: %s\n\n", result["reasoning"])
}
