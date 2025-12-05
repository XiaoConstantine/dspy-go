package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// Example demonstrating the Refine module for improving prediction quality.
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

	// Example 1: Math Problem Solving with Quality Refinement
	fmt.Println("=== Example 1: Math Problem Refinement ===")
	runMathExample(ctx)

	fmt.Println("\n=== Example 2: Creative Writing Refinement ===")
	runCreativeWritingExample(ctx)

	fmt.Println("\n=== Example 3: Question Answering with Accuracy Refinement ===")
	runQAExample(ctx)

	// Example 4: OfferFeedback demonstration
	demonstrateFeedback(ctx)

	fmt.Println("\n=== Example 5: Advanced Refinement Features ===")
	runAdvancedExample(ctx)
}

func runMathExample(ctx context.Context) {
	// Create signature for math problem solving
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("problem", core.WithDescription("A math word problem to solve"))},
		},
		[]core.OutputField{
			{Field: core.NewField("reasoning", core.WithDescription("Step-by-step reasoning"))},
			{Field: core.NewField("answer", core.WithDescription("The final numerical answer"))},
		},
	).WithInstruction("Solve the math problem with clear reasoning and provide the final answer.")

	// Create base prediction module with native JSON structured output
	// This uses GenerateWithJSON for reliable multi-field extraction (reasoning + answer)
	mathSolver := modules.NewPredict(signature).WithStructuredOutput()

	// Define reward function that checks for correct mathematical reasoning
	mathRewardFn := func(inputs, outputs map[string]interface{}) float64 {
		reasoning, hasReasoning := outputs["reasoning"].(string)
		answer, hasAnswer := outputs["answer"].(string)

		if !hasReasoning || !hasAnswer {
			return 0.0
		}

		score := 0.0

		// Reward clear reasoning steps
		if strings.Contains(strings.ToLower(reasoning), "step") {
			score += 0.3
		}

		// Reward mathematical operations mentioned
		mathWords := []string{"multiply", "divide", "add", "subtract", "equals", "total"}
		for _, word := range mathWords {
			if strings.Contains(strings.ToLower(reasoning), word) {
				score += 0.1
				break
			}
		}

		// Reward if answer contains a number
		if strings.ContainsAny(answer, "0123456789") {
			score += 0.4
		}

		// Bonus for complete solution
		if len(reasoning) > 50 && len(answer) > 0 {
			score += 0.2
		}

		return score
	}

	// Create refine module
	config := modules.RefineConfig{
		N:         4, // Try up to 4 attempts
		RewardFn:  mathRewardFn,
		Threshold: 0.8, // High threshold for math accuracy
	}

	refiner := modules.NewRefine(mathSolver, config)

	// Test problem
	problem := "Sarah has 24 apples. She gives 1/3 of them to her friend and eats 2 of the remaining apples. How many apples does she have left?"

	inputs := map[string]interface{}{
		"problem": problem,
	}

	fmt.Printf("Problem: %s\n", problem)

	// Get refined solution
	outputs, err := refiner.Process(ctx, inputs)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Reasoning: %s\n", outputs["reasoning"])
	fmt.Printf("Answer: %s\n", outputs["answer"])
}

func runCreativeWritingExample(ctx context.Context) {
	// Create signature for creative writing
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("prompt", core.WithDescription("A creative writing prompt"))},
			{Field: core.NewField("style", core.WithDescription("The desired writing style"))},
		},
		[]core.OutputField{
			{Field: core.NewField("story", core.WithDescription("A creative short story"))},
		},
	).WithInstruction("Write a creative and engaging short story based on the prompt in the specified style.")

	// Create base prediction module with native JSON structured output
	// for reliable story extraction
	writer := modules.NewPredict(signature).WithStructuredOutput()

	// Define reward function for creative writing quality
	creativityRewardFn := func(inputs, outputs map[string]interface{}) float64 {
		story, hasStory := outputs["story"].(string)
		if !hasStory || len(story) < 50 {
			return 0.0
		}

		score := 0.0

		// Reward length (more creative content)
		if len(story) > 200 {
			score += 0.3
		}

		// Reward descriptive language
		descriptiveWords := []string{"vivid", "mysterious", "brilliant", "enchanting", "dramatic"}
		for _, word := range descriptiveWords {
			if strings.Contains(strings.ToLower(story), word) {
				score += 0.1
			}
		}

		// Reward dialogue (character interaction)
		if strings.Contains(story, "\"") {
			score += 0.2
		}

		// Reward varied sentence structure
		sentences := strings.Split(story, ".")
		if len(sentences) > 3 {
			score += 0.2
		}

		// Reward narrative elements
		narrativeElements := []string{"suddenly", "meanwhile", "however", "finally"}
		for _, element := range narrativeElements {
			if strings.Contains(strings.ToLower(story), element) {
				score += 0.1
				break
			}
		}

		return score
	}

	// Create refine module with higher attempt count for creativity
	config := modules.RefineConfig{
		N:         5, // More attempts for creative exploration
		RewardFn:  creativityRewardFn,
		Threshold: 0.7, // Moderate threshold for creativity
	}

	refiner := modules.NewRefine(writer, config)

	// Test inputs
	inputs := map[string]interface{}{
		"prompt": "A time traveler discovers they can only travel to moments of great historical significance, but each trip changes something small that has big consequences.",
		"style":  "suspenseful and thought-provoking",
	}

	fmt.Printf("Prompt: %s\n", inputs["prompt"])
	fmt.Printf("Style: %s\n", inputs["style"])

	// Get refined story
	outputs, err := refiner.Process(ctx, inputs)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	story := outputs["story"].(string)
	fmt.Printf("Refined Story (%d characters):\n%s\n", len(story), story)
}

func runQAExample(ctx context.Context) {
	// Create signature for question answering
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context", core.WithDescription("Background information"))},
			{Field: core.NewField("question", core.WithDescription("The question to answer"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("A comprehensive answer"))},
			{Field: core.NewField("confidence", core.WithDescription("Confidence level (high/medium/low)"))},
		},
	).WithInstruction("Answer the question based on the context provided. Include your confidence level.")

	// Create base prediction module with native JSON structured output
	// for reliable multi-field extraction (answer + confidence)
	qa := modules.NewPredict(signature).WithStructuredOutput()

	// Define reward function for answer quality
	qaRewardFn := func(inputs, outputs map[string]interface{}) float64 {
		answer, hasAnswer := outputs["answer"].(string)
		confidence, hasConfidence := outputs["confidence"].(string)
		context := inputs["context"].(string)

		if !hasAnswer || !hasConfidence {
			return 0.0
		}

		score := 0.0

		// Reward comprehensive answers
		if len(answer) > 100 {
			score += 0.3
		}

		// Reward answers that reference the context
		contextWords := strings.Fields(strings.ToLower(context))
		answerLower := strings.ToLower(answer)
		contextMatches := 0
		for _, word := range contextWords {
			if len(word) > 4 && strings.Contains(answerLower, word) {
				contextMatches++
			}
		}
		if contextMatches > 0 {
			score += 0.3
		}

		// Reward explicit confidence indicators
		confidenceLower := strings.ToLower(confidence)
		if strings.Contains(confidenceLower, "high") ||
			strings.Contains(confidenceLower, "medium") ||
			strings.Contains(confidenceLower, "low") {
			score += 0.2
		}

		// Reward structured answers
		if strings.Contains(answer, ":") || strings.Contains(answer, "-") {
			score += 0.2
		}

		return score
	}

	// Create refine module
	config := modules.RefineConfig{
		N:         3, // Moderate attempts for QA
		RewardFn:  qaRewardFn,
		Threshold: 0.8, // High threshold for accuracy
	}

	refiner := modules.NewRefine(qa, config)

	// Test context and question
	context := `The Python programming language was created by Guido van Rossum and first released in 1991.
	Python is known for its simplicity and readability, making it popular for beginners and experts alike.
	It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
	Python has a large standard library and extensive third-party package ecosystem through PyPI (Python Package Index).`

	question := "What makes Python popular among programmers?"

	inputs := map[string]interface{}{
		"context":  context,
		"question": question,
	}

	fmt.Printf("Context: %s\n", context)
	fmt.Printf("Question: %s\n", question)

	// Get refined answer
	outputs, err := refiner.Process(ctx, inputs)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Answer: %s\n", outputs["answer"])
	fmt.Printf("Confidence: %s\n", outputs["confidence"])
}

// Example of using OfferFeedback for module improvement advice.
func demonstrateFeedback(ctx context.Context) {
	fmt.Println("\n=== Example 4: OfferFeedback Demonstration ===")

	feedback := modules.NewOfferFeedback()

	inputs := map[string]interface{}{
		"program_inputs":   "What is 2+2?",
		"program_outputs":  "The answer is four",
		"reward_value":     "0.3",
		"target_threshold": "0.8",
	}

	fmt.Printf("Low-performing prediction:\n")
	fmt.Printf("Input: %s\n", inputs["program_inputs"])
	fmt.Printf("Output: %s\n", inputs["program_outputs"])
	fmt.Printf("Reward: %s (below threshold of %s)\n", inputs["reward_value"], inputs["target_threshold"])

	outputs, err := feedback.Process(ctx, inputs)
	if err != nil {
		log.Printf("Feedback error: %v", err)
		return
	}

	fmt.Printf("\nAI-Generated Feedback:\n")
	fmt.Printf("Discussion: %s\n", outputs["discussion"])
	fmt.Printf("Improvement Advice: %s\n", outputs["advice"])
}

// Advanced example showing dynamic configuration and complex reward functions.
func runAdvancedExample(ctx context.Context) {
	// Create signature for code generation
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("task", core.WithDescription("Programming task description"))},
			{Field: core.NewField("language", core.WithDescription("Programming language"))},
		},
		[]core.OutputField{
			{Field: core.NewField("code", core.WithDescription("Generated code solution"))},
			{Field: core.NewField("explanation", core.WithDescription("Code explanation"))},
		},
	).WithInstruction("Write clean, well-documented code to solve the given task.")

	// Create base prediction module with native JSON structured output
	// for reliable multi-field extraction (code + explanation)
	coder := modules.NewPredict(signature).WithStructuredOutput()

	// Complex reward function that evaluates multiple aspects
	codeQualityRewardFn := func(inputs, outputs map[string]interface{}) float64 {
		code, hasCode := outputs["code"].(string)
		explanation, hasExplanation := outputs["explanation"].(string)

		if !hasCode || !hasExplanation {
			return 0.0
		}

		score := 0.0
		codeLower := strings.ToLower(code)

		// Code structure and syntax (0-0.4)
		if strings.Contains(code, "{") && strings.Contains(code, "}") {
			score += 0.2 // Has proper structure
		}
		if strings.Contains(codeLower, "function") || strings.Contains(codeLower, "def") {
			score += 0.2 // Has function definition
		}

		// Documentation and comments (0-0.3)
		commentCount := strings.Count(code, "//") + strings.Count(code, "#")
		if commentCount > 0 {
			score += 0.15
		}
		if len(explanation) > 50 {
			score += 0.15 // Good explanation
		}

		// Code quality indicators (0-0.3)
		qualityIndicators := []string{"const", "let", "var", "return", "if", "for", "while"}
		for _, indicator := range qualityIndicators {
			if strings.Contains(codeLower, indicator) {
				score += 0.05
				if score >= 0.3 { // Cap this section at 0.3
					break
				}
			}
		}

		return score
	}

	// Start with initial configuration
	config := modules.RefineConfig{
		N:         2, // Start with few attempts
		RewardFn:  codeQualityRewardFn,
		Threshold: 0.6, // Moderate threshold
	}

	refiner := modules.NewRefine(coder, config)

	// Test inputs
	inputs := map[string]interface{}{
		"task":     "Create a function that calculates the factorial of a number",
		"language": "JavaScript",
	}

	fmt.Printf("Task: %s\n", inputs["task"])
	fmt.Printf("Language: %s\n", inputs["language"])

	// First attempt with basic configuration
	fmt.Printf("\n--- Attempt 1: Basic Configuration (N=2, threshold=0.6) ---\n")
	outputs, err := refiner.Process(ctx, inputs)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	firstScore := codeQualityRewardFn(inputs, outputs)
	fmt.Printf("First attempt score: %.2f\n", firstScore)
	fmt.Printf("Code:\n%s\n", outputs["code"])
	fmt.Printf("Explanation: %s\n", outputs["explanation"])

	// If score is low, try with higher settings
	if firstScore < 0.8 {
		fmt.Printf("\n--- Attempt 2: Enhanced Configuration (N=4, threshold=0.8) ---\n")

		// Update configuration for better results
		enhancedConfig := modules.RefineConfig{
			N:         4, // More attempts
			RewardFn:  codeQualityRewardFn,
			Threshold: 0.8, // Higher threshold
		}

		refiner.UpdateConfig(enhancedConfig)

		outputs, err = refiner.Process(ctx, inputs)
		if err != nil {
			log.Printf("Error: %v", err)
			return
		}

		secondScore := codeQualityRewardFn(inputs, outputs)
		fmt.Printf("Enhanced attempt score: %.2f\n", secondScore)
		fmt.Printf("Code:\n%s\n", outputs["code"])
		fmt.Printf("Explanation: %s\n", outputs["explanation"])

		improvement := secondScore - firstScore
		if improvement > 0 {
			fmt.Printf("\nImprovement: +%.2f score points\n", improvement)
		}
	}

	// Demonstrate configuration inspection
	currentConfig := refiner.GetConfig()
	fmt.Printf("\nCurrent Configuration:\n")
	fmt.Printf("- Max Attempts: %d\n", currentConfig.N)
	fmt.Printf("- Threshold: %.2f\n", currentConfig.Threshold)
	fmt.Printf("- Wrapped Module: %T\n", refiner.GetWrappedModule())

	// Show temperature sequence used
	fmt.Printf("\nTemperature sequence for %d attempts: ", currentConfig.N)
	// Create a temporary refiner to show temperature generation
	_ = modules.NewRefine(coder, currentConfig)
	// Note: We can't access generateTemperatureSequence directly as it's private,
	// but we can document what it would look like
	fmt.Printf("[0.3, 0.7, ~0.9, ~0.8] (approximate values)\n")
}
