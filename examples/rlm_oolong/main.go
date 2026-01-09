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
	"io"
	"net/http"
	"os"
	"regexp"
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
// Supports both HuggingFace format and rlm-go format.
type OolongTask struct {
	// HuggingFace format fields
	ID                string `json:"id"`
	ContextLen        int    `json:"context_len"`
	Dataset           string `json:"dataset"`
	ContextWindowText string `json:"context_window_text"`
	Question          string `json:"question"`
	TaskGroup         string `json:"task_group"`
	Task              string `json:"task"`
	Answer            string `json:"answer"`
	AnswerType        string `json:"answer_type"`

	// rlm-go format fields (task_id, context, question, answer)
	TaskID  string `json:"task_id"`
	Context string `json:"context"`
}

// Normalize returns a normalized version of the task (handles both formats).
func (t OolongTask) Normalize() OolongTask {
	// Use rlm-go format fields if HuggingFace fields are empty
	if t.ID == "" && t.TaskID != "" {
		t.ID = t.TaskID
	}
	if t.ContextWindowText == "" && t.Context != "" {
		t.ContextWindowText = t.Context
	}
	return t
}

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
	TotalTasks       int     `json:"total_tasks"`
	RLMCorrect       int     `json:"rlm_correct"`
	BaselineCorrect  int     `json:"baseline_correct"`
	RLMAccuracy      float64 `json:"rlm_accuracy"`
	BaselineAccuracy float64 `json:"baseline_accuracy"`
	TotalRLMTime     float64 `json:"total_rlm_time_secs"`
	TotalBaselineTime float64 `json:"total_baseline_time_secs"`
	AvgRLMTime       float64 `json:"avg_rlm_time_secs"`
	AvgBaselineTime  float64 `json:"avg_baseline_time_secs"`
	TotalRLMInputTokens  int `json:"total_rlm_input_tokens"`
	TotalRLMOutputTokens int `json:"total_rlm_output_tokens"`
}

// loadTasksFromFile loads OOLONG tasks from a JSON file.
// Supports both rlm-go format (task_id, context, question, answer)
// and HuggingFace format.
func loadTasksFromFile(path string) ([]OolongTask, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var tasks []OolongTask
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	// Normalize all tasks
	for i := range tasks {
		tasks[i] = tasks[i].Normalize()
	}

	return tasks, nil
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
	Verbose   bool
	MaxIters  int
	TraceDir  string
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

// checkAnswer determines if the model's answer matches the expected answer.
// Uses the same logic as rlm-go's check_answer for consistency.
func checkAnswer(expected, actual string) bool {
	// Normalize expected - handle list notation like "['incorrect']"
	expectedNorm := strings.ToLower(strings.TrimSpace(expected))
	if strings.HasPrefix(expectedNorm, "[") && strings.HasSuffix(expectedNorm, "]") {
		inner := strings.TrimSpace(expectedNorm[1 : len(expectedNorm)-1])
		if (strings.HasPrefix(inner, "'") && strings.HasSuffix(inner, "'")) ||
			(strings.HasPrefix(inner, "\"") && strings.HasSuffix(inner, "\"")) {
			inner = inner[1 : len(inner)-1]
		}
		expectedNorm = inner
	}

	actualNorm := strings.ToLower(strings.TrimSpace(actual))
	responseLen := len(actualNorm)

	// Exact match
	if expectedNorm == actualNorm {
		return true
	}

	// For SHORT responses (< 50 chars), allow flexible matching
	if responseLen < 50 {
		// Word boundary match
		pattern := `(?:^|[\s'":=-])` + regexp.QuoteMeta(expectedNorm) + `(?:$|[\s'".,;:=-])`
		if matched, _ := regexp.MatchString(pattern, actualNorm); matched {
			return true
		}

		// Numeric match
		if isNumeric(expectedNorm) {
			cleaned := regexp.MustCompile(`[^\d]`).ReplaceAllString(actualNorm, "")
			if cleaned == expectedNorm {
				return true
			}
		}

		return false
	}

	// For LONG responses (>= 50 chars), only check the LAST LINE
	lines := strings.Split(strings.TrimSpace(actualNorm), "\n")
	lastLine := ""
	if len(lines) > 0 {
		lastLine = strings.TrimSpace(lines[len(lines)-1])
	}

	// Match structured formats: "Label: X", "Answer: X", "User: X"
	structuredPattern := `^\s*(?:the\s+)?(?:answer|label|result|user)\s*(?:is)?[:=]\s*["']?([^"'\n,]+)["']?\s*$`
	re := regexp.MustCompile(structuredPattern)
	if match := re.FindStringSubmatch(lastLine); len(match) > 1 {
		extracted := strings.Trim(strings.TrimSpace(match[1]), ".,;:")
		if expectedNorm == extracted {
			return true
		}
	}

	// If last line is short, check if it equals the answer
	if len(lastLine) < 30 {
		cleaned := strings.Trim(lastLine, ".,;:\"'")
		if expectedNorm == cleaned {
			return true
		}
	}

	// Numeric in structured format
	if isNumeric(expectedNorm) {
		numPattern := `^\s*(?:answer|result|user)?[:=]?\s*(\d+)\s*$`
		numRe := regexp.MustCompile(numPattern)
		if match := numRe.FindStringSubmatch(lastLine); len(match) > 1 && match[1] == expectedNorm {
			return true
		}
	}

	return false
}

// isNumeric checks if a string contains only digits.
func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
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
		logger.Info(ctx, "Loaded %d tasks from file", len(tasks))
	} else if *useHuggingFace {
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
			"provider":    *provider,
			"model":       *model,
			"max_iters":   *maxIters,
			"results":     results,
			"summary":     summary,
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
