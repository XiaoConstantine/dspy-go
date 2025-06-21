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

	// Run different parallel processing examples
	fmt.Println("=== DSPy-Go Parallel Module Examples ===")

	// Example 1: Basic batch processing
	runBasicBatchExample(ctx)

	// Example 2: Sentiment analysis with error handling
	runSentimentAnalysisExample(ctx)

	// Example 3: Translation with custom workers
	runTranslationExample(ctx)

	// Example 4: Question answering with failure handling
	runQAWithFailuresExample(ctx)
}

// Example 1: Basic batch text summarization.
func runBasicBatchExample(ctx context.Context) {
	fmt.Println("1. Basic Batch Text Summarization")
	fmt.Println("==================================")

	// Create a signature for text summarization
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("text", core.WithDescription("The text to summarize"))},
		},
		[]core.OutputField{
			{Field: core.NewField("summary", core.WithDescription("A concise summary of the text"))},
		},
	).WithInstruction("Provide a brief, clear summary of the given text in 1-2 sentences.")

	// Create a basic predict module
	predict := modules.NewPredict(signature)

	// Wrap it with parallel execution (default workers = CPU count)
	parallel := modules.NewParallel(predict)

	// Prepare batch inputs
	batchInputs := []map[string]interface{}{
		{"text": "Artificial intelligence is transforming healthcare by enabling more accurate diagnoses, personalized treatments, and efficient drug discovery. Machine learning algorithms can analyze medical images, predict patient outcomes, and assist doctors in making better decisions."},
		{"text": "Climate change is causing rising sea levels, extreme weather events, and ecosystem disruption worldwide. Scientists emphasize the urgent need for renewable energy adoption and carbon emission reduction to mitigate these effects."},
		{"text": "The stock market experienced significant volatility this week due to inflation concerns and geopolitical tensions. Investors are closely watching central bank policies and their potential impact on interest rates."},
	}

	start := time.Now()

	// Process all inputs in parallel
	result, err := parallel.Process(ctx, map[string]interface{}{
		"batch_inputs": batchInputs,
	})
	if err != nil {
		log.Printf("Parallel processing failed: %v", err)
		return
	}

	duration := time.Since(start)
	results := result["results"].([]map[string]interface{})

	fmt.Printf("Processed %d texts in %v\n\n", len(results), duration)

	for i, res := range results {
		fmt.Printf("Text %d Summary: %s\n", i+1, res["summary"])
	}
	fmt.Println()
}

// Example 2: Sentiment analysis with error handling.
func runSentimentAnalysisExample(ctx context.Context) {
	fmt.Println("2. Sentiment Analysis with Error Handling")
	fmt.Println("========================================")

	// Create signature for sentiment analysis
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("review", core.WithDescription("Customer review text"))},
		},
		[]core.OutputField{
			{Field: core.NewField("sentiment", core.WithDescription("Sentiment: positive, negative, or neutral"))},
			{Field: core.NewField("confidence", core.WithDescription("Confidence score from 0.0 to 1.0"))},
		},
	).WithInstruction("Analyze the sentiment of the customer review and provide a confidence score.")

	predict := modules.NewPredict(signature)

	// Configure parallel with custom settings
	parallel := modules.NewParallel(predict,
		modules.WithMaxWorkers(2),        // Limit to 2 concurrent workers
		modules.WithReturnFailures(true), // Include failed results in output
	)

	// Prepare reviews including some that might cause issues
	reviews := []map[string]interface{}{
		{"review": "This product is absolutely amazing! Best purchase I've ever made."},
		{"review": "Terrible quality, broke after one day. Very disappointed."},
		{"review": "It's okay, nothing special but does the job."},
		{"review": "Outstanding customer service and fast shipping!"},
		{"review": ""}, // Empty review - might cause issues
	}

	start := time.Now()

	result, err := parallel.Process(ctx, map[string]interface{}{
		"batch_inputs": reviews,
	})
	if err != nil {
		log.Printf("Parallel processing failed: %v", err)
		return
	}

	duration := time.Since(start)
	results := result["results"].([]map[string]interface{})
	failures, hasFailures := result["failures"].([]map[string]interface{})

	fmt.Printf("Processed %d reviews in %v\n", len(reviews), duration)
	fmt.Printf("Successful: %d, Failed: %d\n\n", len(results), len(failures))

	for i, res := range results {
	if res != nil {
	fmt.Printf("Review %d: %s (confidence: %s)\n", 
	   i+1, res["sentiment"], res["confidence"])
		} else {
			fmt.Printf("Review %d: [FAILED]\n", i+1)
		}
	}

	if hasFailures && len(failures) > 0 {
		fmt.Println("\nFailures:")
		for _, failure := range failures {
			fmt.Printf("Index %v: %s\n", failure["index"], failure["error"])
		}
	}
	fmt.Println()
}

// Example 3: Translation with custom worker count.
func runTranslationExample(ctx context.Context) {
	fmt.Println("3. Multi-language Translation")
	fmt.Println("=============================")

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("text", core.WithDescription("Text to translate"))},
			{Field: core.NewField("target_language", core.WithDescription("Target language"))},
		},
		[]core.OutputField{
			{Field: core.NewField("translation", core.WithDescription("Translated text"))},
		},
	).WithInstruction("Translate the given text to the specified target language.")

	predict := modules.NewPredict(signature)

	// Use 3 workers for translation tasks
	parallel := modules.NewParallel(predict, modules.WithMaxWorkers(3))

	// Prepare translation tasks
	translations := []map[string]interface{}{
		{"text": "Hello, how are you today?", "target_language": "Spanish"},
		{"text": "The weather is beautiful", "target_language": "French"},
		{"text": "Thank you for your help", "target_language": "German"},
		{"text": "Good morning everyone", "target_language": "Italian"},
		{"text": "Have a great day!", "target_language": "Japanese"},
	}

	start := time.Now()

	result, err := parallel.Process(ctx, map[string]interface{}{
		"batch_inputs": translations,
	})
	if err != nil {
		log.Printf("Translation failed: %v", err)
		return
	}

	duration := time.Since(start)
	results := result["results"].([]map[string]interface{})

	fmt.Printf("Completed %d translations in %v\n\n", len(results), duration)

	for i, res := range results {
	original := translations[i]
	if res != nil {
	fmt.Printf("%s → %s: %s\n", 
	   original["text"], original["target_language"], res["translation"])
		} else {
			fmt.Printf("%s → %s: [FAILED]\n", 
				original["text"], original["target_language"])
		}
	}
	fmt.Println()
}

// Example 4: Question answering with stop-on-first-error.
func runQAWithFailuresExample(ctx context.Context) {
	fmt.Println("4. Question Answering with Stop-on-First-Error")
	fmt.Println("==============================================")

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question", core.WithDescription("Question to answer"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("Clear, factual answer"))},
		},
	).WithInstruction("Provide a clear, factual answer to the question.")

	predict := modules.NewPredict(signature)

	// Configure to stop on first error
	parallel := modules.NewParallel(predict,
		modules.WithMaxWorkers(4),
		modules.WithStopOnFirstError(true), // Stop if any question fails
	)

	questions := []map[string]interface{}{
		{"question": "What is the capital of France?"},
		{"question": "How many continents are there?"},
		{"question": "What is 2 + 2?"},
		{"question": "Who wrote Romeo and Juliet?"},
	}

	start := time.Now()

	result, err := parallel.Process(ctx, map[string]interface{}{
		"batch_inputs": questions,
	})

	duration := time.Since(start)

	if err != nil {
		fmt.Printf("Processing stopped due to error after %v: %v\n", duration, err)
		return
	}

	results := result["results"].([]map[string]interface{})
	fmt.Printf("Answered %d questions in %v\n\n", len(results), duration)

	for i, res := range results {
		if res != nil {
			fmt.Printf("Q: %s\nA: %s\n\n", questions[i]["question"], res["answer"])
		} else {
			fmt.Printf("Q: %s\nA: [FAILED]\n\n", questions[i]["question"])
		}
	}
}
