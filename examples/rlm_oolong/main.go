// Package main demonstrates the RLM module on OOLONG benchmark tasks.
// OOLONG (Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities)
// is a benchmark for testing long-context reasoning requiring multi-hop analysis
// and aggregation over large text contexts.
//
// Reference: https://arxiv.org/abs/2511.02817
// Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

// OolongTask represents a single OOLONG benchmark task.
type OolongTask struct {
	ID                string `json:"id"`
	ContextLen        int    `json:"context_len"`
	Dataset           string `json:"dataset"`
	ContextWindowText string `json:"context_window_text"`
	Question          string `json:"question"`
	TaskGroup         string `json:"task_group"`
	Task              string `json:"task"`
	Answer            string `json:"answer"`
	AnswerType        string `json:"answer_type"`
}

// BenchmarkResult stores the result of running a single task.
type BenchmarkResult struct {
	TaskID         string
	Question       string
	ExpectedAnswer string
	RLMAnswer      string
	BaselineAnswer string
	RLMCorrect     bool
	BaselineCorrect bool
	RLMIterations  int
	RLMDuration    time.Duration
	BaselineDuration time.Duration
}

// getSampleOolongTasks returns embedded sample OOLONG-style tasks for testing.
// These are simplified examples that demonstrate the types of reasoning required.
func getSampleOolongTasks() []OolongTask {
	// Sample context simulating product reviews with user IDs and timestamps
	reviewContext := `
User: alice | Date: 2024-01-15 | Product: Widget A | Rating: 5 | Review: Excellent product, highly recommend!
User: bob | Date: 2024-01-16 | Product: Widget B | Rating: 2 | Review: Poor quality, broke after a week.
User: alice | Date: 2024-01-17 | Product: Widget C | Rating: 4 | Review: Good value for money.
User: charlie | Date: 2024-01-18 | Product: Widget A | Rating: 5 | Review: Best purchase ever!
User: bob | Date: 2024-01-19 | Product: Widget A | Rating: 1 | Review: Terrible, completely unusable.
User: diana | Date: 2024-01-20 | Product: Widget B | Rating: 3 | Review: It's okay, nothing special.
User: alice | Date: 2024-02-01 | Product: Widget A | Rating: 5 | Review: Still works great after weeks!
User: charlie | Date: 2024-02-02 | Product: Widget C | Rating: 4 | Review: Solid choice.
User: eve | Date: 2024-02-03 | Product: Widget A | Rating: 5 | Review: Amazing quality.
User: bob | Date: 2024-02-04 | Product: Widget C | Rating: 2 | Review: Not worth the price.
User: diana | Date: 2024-02-05 | Product: Widget A | Rating: 4 | Review: Pretty good overall.
User: eve | Date: 2024-02-06 | Product: Widget B | Rating: 3 | Review: Average product.
User: charlie | Date: 2024-02-07 | Product: Widget B | Rating: 4 | Review: Better than expected.
User: alice | Date: 2024-02-08 | Product: Widget B | Rating: 5 | Review: Love it!
User: diana | Date: 2024-02-09 | Product: Widget C | Rating: 4 | Review: Would buy again.
`

	// Repeat context to make it longer (simulating larger OOLONG contexts)
	longContext := strings.Repeat(reviewContext, 50)

	return []OolongTask{
		{
			ID:                "counting_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "How many reviews are there in total?",
			TaskGroup:         "counting",
			Task:              "count_total",
			Answer:            "750",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "counting_2",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "How many reviews have a rating of 5 stars?",
			TaskGroup:         "counting",
			Task:              "count_by_label",
			Answer:            "250",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "user_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "What is the average rating given by user 'alice'?",
			TaskGroup:         "user",
			Task:              "user_average",
			Answer:            "4.75",
			AnswerType:        "NUMERIC",
		},
		{
			ID:                "comparison_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "Which product has the most 5-star reviews: Widget A, Widget B, or Widget C?",
			TaskGroup:         "comparison",
			Task:              "most_frequent",
			Answer:            "Widget A",
			AnswerType:        "LABEL",
		},
		{
			ID:                "temporal_1",
			ContextLen:        len(longContext),
			Dataset:           "sample_reviews",
			ContextWindowText: longContext,
			Question:          "Are there more reviews in January or February?",
			TaskGroup:         "temporal",
			Task:              "temporal_comparison",
			Answer:            "February",
			AnswerType:        "LABEL",
		},
	}
}

// fetchOolongFromHuggingFace attempts to fetch OOLONG tasks from HuggingFace.
// Returns sample tasks if fetch fails.
func fetchOolongFromHuggingFace(limit int) ([]OolongTask, error) {
	// HuggingFace datasets API endpoint for oolong-synth
	url := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=oolongbench/oolong-synth&config=default&split=validation&offset=0&length=%d", limit)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch from HuggingFace: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HuggingFace API returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse HuggingFace response format
	var hfResp struct {
		Rows []struct {
			Row OolongTask `json:"row"`
		} `json:"rows"`
	}

	if err := json.Unmarshal(body, &hfResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	tasks := make([]OolongTask, len(hfResp.Rows))
	for i, row := range hfResp.Rows {
		tasks[i] = row.Row
	}

	return tasks, nil
}

// runBaseline runs the baseline (direct prompting) on a task.
func runBaseline(ctx context.Context, llm core.LLM, task OolongTask) (string, time.Duration, error) {
	start := time.Now()

	// Create a simple signature for direct Q&A
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "context", Description: "The context to analyze"}},
			{Field: core.Field{Name: "question", Description: "The question to answer"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "answer", Description: "The answer to the question"}},
		},
	).WithInstruction("Answer the question based on the provided context. Be precise and concise.")

	predict := modules.NewPredict(signature)
	predict.SetLLM(llm)

	// Truncate context if too long for baseline (simulate typical LLM context limits)
	contextText := task.ContextWindowText
	maxContextLen := 100000 // ~100k chars, reasonable for most models
	if len(contextText) > maxContextLen {
		contextText = contextText[:maxContextLen] + "\n... [truncated]"
	}

	result, err := predict.Process(ctx, map[string]any{
		"context":  contextText,
		"question": task.Question,
	})

	if err != nil {
		return "", time.Since(start), err
	}

	answer, _ := result["answer"].(string)
	return strings.TrimSpace(answer), time.Since(start), nil
}

// runRLM runs the RLM module on a task.
func runRLM(ctx context.Context, llm core.LLM, task OolongTask, verbose bool) (string, int, time.Duration, error) {
	rlmModule := rlm.NewFromLLM(
		llm,
		rlm.WithMaxIterations(15),
		rlm.WithVerbose(verbose),
		rlm.WithTimeout(5*time.Minute),
	)

	start := time.Now()
	result, err := rlmModule.Complete(ctx, task.ContextWindowText, task.Question)
	if err != nil {
		return "", 0, time.Since(start), err
	}

	return strings.TrimSpace(result.Response), result.Iterations, result.Duration, nil
}

// checkAnswer performs flexible answer matching.
func checkAnswer(expected, actual string) bool {
	// Normalize strings for comparison
	expected = strings.ToLower(strings.TrimSpace(expected))
	actual = strings.ToLower(strings.TrimSpace(actual))

	// Direct match
	if expected == actual {
		return true
	}

	// Check if actual contains expected (for numeric answers that might have extra text)
	if strings.Contains(actual, expected) {
		return true
	}

	// Check if expected contains actual (for partial matches)
	if strings.Contains(expected, actual) && len(actual) > 2 {
		return true
	}

	return false
}

// printResults displays benchmark results in a formatted table.
func printResults(results []BenchmarkResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("BENCHMARK RESULTS")
	fmt.Println(strings.Repeat("=", 100))

	rlmCorrect := 0
	baselineCorrect := 0
	totalRLMTime := time.Duration(0)
	totalBaselineTime := time.Duration(0)

	for _, r := range results {
		if r.RLMCorrect {
			rlmCorrect++
		}
		if r.BaselineCorrect {
			baselineCorrect++
		}
		totalRLMTime += r.RLMDuration
		totalBaselineTime += r.BaselineDuration
	}

	fmt.Printf("\n%-15s %-10s %-10s\n", "Metric", "RLM", "Baseline")
	fmt.Println(strings.Repeat("-", 40))
	fmt.Printf("%-15s %-10d %-10d\n", "Correct", rlmCorrect, baselineCorrect)
	fmt.Printf("%-15s %-10d %-10d\n", "Total", len(results), len(results))
	fmt.Printf("%-15s %-10.1f%% %-10.1f%%\n", "Accuracy",
		float64(rlmCorrect)/float64(len(results))*100,
		float64(baselineCorrect)/float64(len(results))*100)
	fmt.Printf("%-15s %-10s %-10s\n", "Total Time", totalRLMTime.Round(time.Millisecond), totalBaselineTime.Round(time.Millisecond))

	fmt.Println("\n" + strings.Repeat("-", 100))
	fmt.Println("DETAILED RESULTS")
	fmt.Println(strings.Repeat("-", 100))

	for i, r := range results {
		fmt.Printf("\nTask %d: %s\n", i+1, r.TaskID)
		fmt.Printf("  Question: %s\n", r.Question)
		fmt.Printf("  Expected: %s\n", r.ExpectedAnswer)
		fmt.Printf("  RLM Answer: %s (correct: %v, iterations: %d, time: %v)\n",
			truncate(r.RLMAnswer, 80), r.RLMCorrect, r.RLMIterations, r.RLMDuration.Round(time.Millisecond))
		fmt.Printf("  Baseline Answer: %s (correct: %v, time: %v)\n",
			truncate(r.BaselineAnswer, 80), r.BaselineCorrect, r.BaselineDuration.Round(time.Millisecond))
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func main() {
	// Command line flags
	provider := flag.String("provider", "anthropic", "LLM provider (anthropic, openai, gemini)")
	model := flag.String("model", "", "Model to use (default: claude-haiku-4-5 for anthropic, gpt-4o-mini for openai, gemini-2.5-flash for gemini)")
	numTasks := flag.Int("tasks", 3, "Number of tasks to run (max 5 for embedded samples)")
	useHuggingFace := flag.Bool("hf", false, "Fetch tasks from HuggingFace (requires internet)")
	verbose := flag.Bool("verbose", false, "Enable verbose RLM output")
	skipBaseline := flag.Bool("skip-baseline", false, "Skip baseline comparison")
	flag.Parse()

	// Setup logging
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logLevel := logging.INFO
	if *verbose {
		logLevel = logging.DEBUG
	}
	logger := logging.NewLogger(logging.Config{
		Severity: logLevel,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	ctx := core.WithExecutionState(context.Background())

	// Initialize LLM factory
	llms.EnsureFactory()

	// Setup LLM based on provider
	var llm core.LLM
	var err error

	switch *provider {
	case "anthropic":
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			logger.Fatal(ctx, "ANTHROPIC_API_KEY environment variable not set")
		}
		modelName := *model
		if modelName == "" {
			modelName = "claude-haiku-4-5"
		}
		llm, err = llms.NewAnthropicLLM(apiKey, anthropic.Model(modelName))
		if err != nil {
			logger.Fatalf(ctx, "Failed to create Anthropic LLM: %v", err)
		}
		logger.Info(ctx, "Using Anthropic model: %s", modelName)

	case "openai":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			logger.Fatal(ctx, "OPENAI_API_KEY environment variable not set")
		}
		modelID := core.ModelOpenAIGPT4oMini
		if *model != "" {
			modelID = core.ModelID(*model)
		}
		llm, err = llms.NewOpenAI(modelID, apiKey)
		if err != nil {
			logger.Fatalf(ctx, "Failed to create OpenAI LLM: %v", err)
		}
		logger.Info(ctx, "Using OpenAI model: %s", modelID)

	case "gemini":
		apiKey := os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			logger.Fatal(ctx, "GOOGLE_API_KEY environment variable not set")
		}
		modelID := core.ModelGoogleGeminiFlash // gemini-2.5-flash
		if *model != "" {
			modelID = core.ModelID(*model)
		}
		llm, err = llms.NewGeminiLLM(apiKey, modelID)
		if err != nil {
			logger.Fatalf(ctx, "Failed to create Gemini LLM: %v", err)
		}
		logger.Info(ctx, "Using Gemini model: %s", modelID)

	default:
		logger.Fatalf(ctx, "Unsupported provider: %s (use 'anthropic', 'openai', or 'gemini')", *provider)
	}

	// Load tasks
	var tasks []OolongTask
	if *useHuggingFace {
		logger.Info(ctx, "Fetching tasks from HuggingFace...")
		tasks, err = fetchOolongFromHuggingFace(*numTasks)
		if err != nil {
			logger.Warn(ctx, "Failed to fetch from HuggingFace: %v. Using embedded samples.", err)
			tasks = getSampleOolongTasks()
		}
	} else {
		tasks = getSampleOolongTasks()
	}

	// Limit tasks
	if *numTasks < len(tasks) {
		tasks = tasks[:*numTasks]
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("OOLONG Benchmark - RLM vs Baseline Comparison")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Tasks: %d\n", len(tasks))
	fmt.Printf("Verbose: %v\n", *verbose)
	fmt.Println(strings.Repeat("-", 60))

	// Run benchmark
	var results []BenchmarkResult

	for i, task := range tasks {
		fmt.Printf("\n[Task %d/%d] %s\n", i+1, len(tasks), task.ID)
		fmt.Printf("  Question: %s\n", task.Question)
		fmt.Printf("  Context length: %d chars\n", len(task.ContextWindowText))

		result := BenchmarkResult{
			TaskID:         task.ID,
			Question:       task.Question,
			ExpectedAnswer: task.Answer,
		}

		// Run RLM
		fmt.Println("  Running RLM...")
		rlmAnswer, iterations, rlmDuration, err := runRLM(ctx, llm, task, *verbose)
		if err != nil {
			logger.Error(ctx, "RLM failed: %v", err)
			result.RLMAnswer = fmt.Sprintf("ERROR: %v", err)
		} else {
			result.RLMAnswer = rlmAnswer
			result.RLMIterations = iterations
			result.RLMDuration = rlmDuration
			result.RLMCorrect = checkAnswer(task.Answer, rlmAnswer)
		}
		fmt.Printf("  RLM: %s (iterations: %d, time: %v)\n",
			truncate(result.RLMAnswer, 50), result.RLMIterations, result.RLMDuration.Round(time.Millisecond))

		// Run baseline (unless skipped)
		if !*skipBaseline {
			fmt.Println("  Running baseline...")
			baselineAnswer, baselineDuration, err := runBaseline(ctx, llm, task)
			if err != nil {
				logger.Error(ctx, "Baseline failed: %v", err)
				result.BaselineAnswer = fmt.Sprintf("ERROR: %v", err)
			} else {
				result.BaselineAnswer = baselineAnswer
				result.BaselineDuration = baselineDuration
				result.BaselineCorrect = checkAnswer(task.Answer, baselineAnswer)
			}
			fmt.Printf("  Baseline: %s (time: %v)\n",
				truncate(result.BaselineAnswer, 50), result.BaselineDuration.Round(time.Millisecond))
		}

		results = append(results, result)
	}

	// Print summary
	printResults(results)
}
