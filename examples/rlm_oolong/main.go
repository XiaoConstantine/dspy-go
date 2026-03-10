// Package main demonstrates the RLM module on OOLONG benchmark tasks.
// OOLONG (Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities)
// is a benchmark for testing long-context reasoning requiring multi-hop analysis
// and aggregation over large text contexts.
//
// Reference: https://arxiv.org/abs/2511.02817
// Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth
//
// Usage:
//
//	# Run with embedded sample tasks
//	go run main.go -tasks 3
//
//	# Run with tasks from JSON file (rlm-go format)
//	go run main.go -file /path/to/oolong_tasks_500.json -tasks 10
//
//	# Run with different models
//	go run main.go -provider anthropic -model claude-sonnet-4-5-20250929
//	go run main.go -provider gemini -model gemini-2.5-pro
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

type OolongTask = datasets.OolongTask

// BenchmarkResult stores the result of running a single task.
type BenchmarkResult struct {
	TaskID           string        `json:"task_id"`
	Question         string        `json:"question"`
	ExpectedAnswer   string        `json:"expected_answer"`
	RLMAnswer        string        `json:"rlm_answer"`
	BaselineAnswer   string        `json:"baseline_answer"`
	RLMCorrect       bool          `json:"rlm_correct"`
	BaselineCorrect  bool          `json:"baseline_correct"`
	RLMIterations    int           `json:"rlm_iterations"`
	RLMDuration      time.Duration `json:"rlm_duration"`
	BaselineDuration time.Duration `json:"baseline_duration"`
	RLMInputTokens   int           `json:"rlm_input_tokens"`
	RLMOutputTokens  int           `json:"rlm_output_tokens"`
	Error            string        `json:"error,omitempty"`
}

// BenchmarkSummary provides aggregate statistics.
type BenchmarkSummary struct {
	TotalTasks           int     `json:"total_tasks"`
	RLMCorrect           int     `json:"rlm_correct"`
	BaselineCorrect      int     `json:"baseline_correct"`
	RLMAccuracy          float64 `json:"rlm_accuracy"`
	BaselineAccuracy     float64 `json:"baseline_accuracy"`
	TotalRLMTime         float64 `json:"total_rlm_time_secs"`
	TotalBaselineTime    float64 `json:"total_baseline_time_secs"`
	AvgRLMTime           float64 `json:"avg_rlm_time_secs"`
	AvgBaselineTime      float64 `json:"avg_baseline_time_secs"`
	TotalRLMInputTokens  int     `json:"total_rlm_input_tokens"`
	TotalRLMOutputTokens int     `json:"total_rlm_output_tokens"`
}

func loadTasksFromFile(path string) ([]OolongTask, error) {
	return datasets.LoadOolongTasksFromFile(path)
}

func getSampleOolongTasks() []OolongTask {
	return datasets.SampleOolongTasks()
}

// runBaseline runs the baseline (direct prompting) on a task.
func runBaseline(ctx context.Context, llm core.LLM, task OolongTask) (string, time.Duration, error) {
	start := time.Now()

	// Create a simple signature for direct Q&A
	// Use core.NewField to ensure proper Prefix is set for text parsing
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("context", core.WithDescription("The context to analyze"))},
			{Field: core.NewField("question", core.WithDescription("The question to answer"))},
		},
		[]core.OutputField{
			{Field: core.NewField("answer", core.WithDescription("The answer to the question"))},
		},
	).WithInstruction("Answer the question based on the provided context. Be precise and concise. Provide only the numerical answer.")

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

// RLMOptions holds configuration for RLM runs.
type RLMOptions struct {
	Verbose  bool
	MaxIters int
	TraceDir string
}

// RLMResult holds the result of an RLM run including token usage.
type RLMResult struct {
	Answer       string
	Iterations   int
	Duration     time.Duration
	InputTokens  int
	OutputTokens int
}

// runRLM runs the RLM module on a task.
func runRLM(ctx context.Context, llm core.LLM, task OolongTask, opts RLMOptions) (*RLMResult, error) {
	rlmOpts := []rlm.Option{
		rlm.WithMaxIterations(opts.MaxIters),
		rlm.WithVerbose(opts.Verbose),
		rlm.WithTimeout(5 * time.Minute),
	}

	if opts.TraceDir != "" {
		rlmOpts = append(rlmOpts, rlm.WithTraceDir(opts.TraceDir))
	}

	rlmModule := rlm.NewFromLLM(llm, rlmOpts...)

	start := time.Now()
	result, err := rlmModule.Complete(ctx, task.ContextWindowText, task.Question)
	if err != nil {
		return &RLMResult{Duration: time.Since(start)}, err
	}

	return &RLMResult{
		Answer:       strings.TrimSpace(result.Response),
		Iterations:   result.Iterations,
		Duration:     result.Duration,
		InputTokens:  result.Usage.PromptTokens,
		OutputTokens: result.Usage.CompletionTokens,
	}, nil
}

func checkAnswer(expected, actual string) bool {
	return datasets.CheckOolongAnswer(expected, actual)
}

// computeSummary calculates aggregate statistics from benchmark results.
func computeSummary(results []BenchmarkResult) BenchmarkSummary {
	if len(results) == 0 {
		return BenchmarkSummary{}
	}

	var rlmCorrect, baselineCorrect int
	var totalRLMTime, totalBaselineTime float64
	var totalRLMInput, totalRLMOutput int

	for _, r := range results {
		if r.RLMCorrect {
			rlmCorrect++
		}
		if r.BaselineCorrect {
			baselineCorrect++
		}
		totalRLMTime += r.RLMDuration.Seconds()
		totalBaselineTime += r.BaselineDuration.Seconds()
		totalRLMInput += r.RLMInputTokens
		totalRLMOutput += r.RLMOutputTokens
	}

	return BenchmarkSummary{
		TotalTasks:           len(results),
		RLMCorrect:           rlmCorrect,
		BaselineCorrect:      baselineCorrect,
		RLMAccuracy:          float64(rlmCorrect) / float64(len(results)),
		BaselineAccuracy:     float64(baselineCorrect) / float64(len(results)),
		TotalRLMTime:         totalRLMTime,
		TotalBaselineTime:    totalBaselineTime,
		AvgRLMTime:           totalRLMTime / float64(len(results)),
		AvgBaselineTime:      totalBaselineTime / float64(len(results)),
		TotalRLMInputTokens:  totalRLMInput,
		TotalRLMOutputTokens: totalRLMOutput,
	}
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
	taskOffset := flag.Int("task-offset", 0, "Deterministic offset into the selected OOLONG task source")
	taskFile := flag.String("file", "", "Path to OOLONG tasks JSON file (rlm-go format)")
	useHuggingFace := flag.Bool("hf", false, "Fetch tasks from HuggingFace (requires internet)")
	verbose := flag.Bool("verbose", false, "Enable verbose RLM output")
	skipBaseline := flag.Bool("skip-baseline", false, "Skip baseline comparison")
	maxIters := flag.Int("max-iters", 15, "Maximum iterations for RLM")
	outputFile := flag.String("output", "", "Output JSON file for results")
	traceDir := flag.String("trace-dir", "", "Directory for RLM trace logs (JSONL format)")
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
	if *taskFile != "" {
		logger.Info(ctx, "Loading tasks from file: %s", *taskFile)
		tasks, err = loadTasksFromFile(*taskFile)
		if err != nil {
			logger.Fatalf(ctx, "Failed to load tasks from file: %v", err)
		}
		tasks = datasets.SliceOolongTasks(tasks, *taskOffset, *numTasks)
		logger.Info(ctx, "Loaded %d tasks from file", len(tasks))
	} else if *useHuggingFace {
		logger.Info(ctx, "Fetching tasks from HuggingFace...")
		tasks, err = datasets.FetchOolongTasksFromHuggingFaceRange(*taskOffset, *numTasks)
		if err != nil {
			logger.Warn(ctx, "Failed to fetch from HuggingFace: %v. Using embedded samples.", err)
			tasks = getSampleOolongTasks()
			tasks = datasets.SliceOolongTasks(tasks, *taskOffset, *numTasks)
		}
	} else {
		tasks = getSampleOolongTasks()
		tasks = datasets.SliceOolongTasks(tasks, *taskOffset, *numTasks)
	}

	if len(tasks) == 0 {
		logger.Fatalf(ctx, "No OOLONG tasks found (offset=%d, tasks=%d)", *taskOffset, *numTasks)
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("OOLONG Benchmark - RLM vs Baseline Comparison")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Tasks: %d\n", len(tasks))
	fmt.Printf("Task Offset: %d\n", *taskOffset)
	fmt.Printf("Max Iterations: %d\n", *maxIters)
	fmt.Printf("Verbose: %v\n", *verbose)
	if *traceDir != "" {
		fmt.Printf("Trace Dir: %s\n", *traceDir)
	}
	fmt.Println(strings.Repeat("-", 60))

	// RLM options
	rlmOpts := RLMOptions{
		Verbose:  *verbose,
		MaxIters: *maxIters,
		TraceDir: *traceDir,
	}

	// Run benchmark
	var results []BenchmarkResult

	for i, task := range tasks {
		fmt.Printf("\n[Task %d/%d] %s\n", i+1, len(tasks), task.ID)
		fmt.Printf("  Question: %s\n", truncate(task.Question, 80))
		fmt.Printf("  Context length: %d chars\n", len(task.ContextWindowText))

		result := BenchmarkResult{
			TaskID:         task.ID,
			Question:       task.Question,
			ExpectedAnswer: task.Answer,
		}

		// Run RLM
		fmt.Println("  Running RLM...")
		rlmResult, err := runRLM(ctx, llm, task, rlmOpts)
		if err != nil {
			logger.Error(ctx, "RLM failed: %v", err)
			result.RLMAnswer = fmt.Sprintf("ERROR: %v", err)
			result.Error = err.Error()
			result.RLMDuration = rlmResult.Duration
		} else {
			result.RLMAnswer = rlmResult.Answer
			result.RLMIterations = rlmResult.Iterations
			result.RLMDuration = rlmResult.Duration
			result.RLMInputTokens = rlmResult.InputTokens
			result.RLMOutputTokens = rlmResult.OutputTokens
			result.RLMCorrect = checkAnswer(task.Answer, rlmResult.Answer)
		}
		fmt.Printf("  RLM: %s (correct: %v, iterations: %d, time: %v)\n",
			truncate(result.RLMAnswer, 50), result.RLMCorrect, result.RLMIterations, result.RLMDuration.Round(time.Millisecond))
		fmt.Printf("       Expected: %s\n", task.Answer)

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
			fmt.Printf("  Baseline: %s (correct: %v, time: %v)\n",
				truncate(result.BaselineAnswer, 50), result.BaselineCorrect, result.BaselineDuration.Round(time.Millisecond))
		}

		results = append(results, result)
	}

	// Print summary
	printResults(results)

	// Save results if output file specified
	if *outputFile != "" {
		summary := computeSummary(results)
		output := map[string]any{
			"provider":  *provider,
			"model":     *model,
			"max_iters": *maxIters,
			"results":   results,
			"summary":   summary,
		}

		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			logger.Error(ctx, "Error marshaling output: %v", err)
		} else if err := os.WriteFile(*outputFile, data, 0644); err != nil {
			logger.Error(ctx, "Error writing output: %v", err)
		} else {
			fmt.Printf("\nResults saved to: %s\n", *outputFile)
		}
	}

	fmt.Println("\nBenchmark complete!")
}
