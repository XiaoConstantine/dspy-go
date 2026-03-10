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

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/agents/optimize"
	agentrlm "github.com/XiaoConstantine/dspy-go/pkg/agents/rlm"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

const (
	passThreshold     = 1.0
	defaultMaxIters   = 8
	defaultValidation = 0.25
)

type evalConfig struct {
	Provider        string  `json:"provider"`
	Model           string  `json:"model"`
	TaskSource      string  `json:"task_source"`
	TaskOffset      int     `json:"task_offset"`
	TaskCount       int     `json:"task_count"`
	Population      int     `json:"population"`
	Generations     int     `json:"generations"`
	MaxIterations   int     `json:"max_iterations"`
	ValidationSplit float64 `json:"validation_split"`
}

type runSummary struct {
	AverageScore      float64        `json:"average_score"`
	PassedExamples    int            `json:"passed_examples"`
	FailedExamples    int            `json:"failed_examples"`
	CompletedExamples int            `json:"completed_examples"`
	EvaluationErrors  int            `json:"evaluation_errors"`
	AverageSteps      float64        `json:"average_steps"`
	AverageSubLLM     float64        `json:"average_sub_llm_calls"`
	AverageSubRLM     float64        `json:"average_sub_rlm_calls"`
	TerminationCounts map[string]int `json:"termination_counts,omitempty"`
	ExampleIDs        []string       `json:"example_ids,omitempty"`
}

type evalReport struct {
	GeneratedAt                   time.Time          `json:"generated_at"`
	Configuration                 evalConfig         `json:"configuration"`
	TrainingExampleCount          int                `json:"training_example_count"`
	ValidationExampleCount        int                `json:"validation_example_count"`
	TaskIDs                       []string           `json:"task_ids"`
	SeedIterationPrompt           string             `json:"seed_iteration_prompt"`
	OptimizedIterationPrompt      string             `json:"optimized_iteration_prompt,omitempty"`
	Baseline                      runSummary         `json:"baseline"`
	Optimized                     runSummary         `json:"optimized"`
	BestValidationWeightedFitness float64            `json:"best_validation_weighted_fitness,omitempty"`
	BestValidationAverageScore    float64            `json:"best_validation_average_score,omitempty"`
	BestValidationObjectives      map[string]float64 `json:"best_validation_objectives,omitempty"`
}

func main() {
	provider := flag.String("provider", "anthropic", "LLM provider to use: anthropic, openai, or gemini")
	model := flag.String("model", "", "Model to use (provider default if empty)")
	apiKey := flag.String("api-key", "", "API key override (otherwise uses provider environment variable)")
	population := flag.Int("population", 4, "GEPA population size")
	generations := flag.Int("generations", 2, "GEPA generations")
	taskFile := flag.String("file", "", "Path to OOLONG tasks JSON file")
	useHuggingFace := flag.Bool("hf", false, "Fetch tasks from HuggingFace instead of using embedded samples")
	taskCount := flag.Int("tasks", 5, "Number of OOLONG tasks to evaluate")
	taskOffset := flag.Int("task-offset", 0, "Deterministic offset into the selected OOLONG task source")
	maxIters := flag.Int("max-iters", defaultMaxIters, "Maximum RLM iterations per task")
	outputFile := flag.String("output", "", "Optional path to write a JSON evaluation report")
	verbose := flag.Bool("verbose", false, "Enable verbose RLM logging")
	flag.Parse()

	ctx := core.WithExecutionState(context.Background())

	logLevel := logging.INFO
	if *verbose {
		logLevel = logging.DEBUG
	}
	logger := logging.NewLogger(logging.Config{
		Severity: logLevel,
		Outputs: []logging.Output{
			logging.NewConsoleOutput(true, logging.WithColor(true)),
		},
	})
	logging.SetLogger(logger)

	llms.EnsureFactory()

	llm, modelName, err := buildLLM(ctx, logger, *provider, *model, *apiKey)
	if err != nil {
		logger.Fatalf(ctx, "Failed to create LLM: %v", err)
	}

	originalDefault := core.GetDefaultLLM()
	originalTeacher := core.GetTeacherLLM()
	core.SetDefaultLLM(llm)
	core.GlobalConfig.TeacherLLM = llm
	defer func() {
		core.SetDefaultLLM(originalDefault)
		core.GlobalConfig.TeacherLLM = originalTeacher
	}()

	tasks, taskSource, err := loadTasks(*taskFile, *useHuggingFace, *taskOffset, *taskCount)
	if err != nil {
		logger.Fatalf(ctx, "Failed to load OOLONG tasks: %v", err)
	}

	examples := toAgentExamples(tasks)
	trainExamples, validationExamples := splitExamples(examples, defaultValidation)

	harness := &optimize.Harness{
		Evaluator:     oolongEvaluator{},
		PassThreshold: passThreshold,
	}

	seedAgent := newAdaptiveAgent(llm, weakSeedPrompt(), *maxIters, *verbose)
	baselineRun, err := harness.Run(ctx, seedAgent, examples)
	if err != nil {
		logger.Fatalf(ctx, "Baseline run failed: %v", err)
	}

	fmt.Println("=== Live RLM GEPA OOLONG Example ===")
	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Model: %s\n", modelName)
	fmt.Printf("Tasks: %d total (%d train / %d validation)\n", len(examples), len(trainExamples), len(validationExamples))
	fmt.Printf("Max RLM iterations: %d\n", *maxIters)
	fmt.Println("This example uses real model calls on OOLONG-style long-context tasks.")
	fmt.Println()
	printRunSummary("Baseline", baselineRun)

	seedPrompt := weakSeedPrompt()
	optimizer := optimize.NewGEPAAgentOptimizer(
		newAdaptiveAgent(llm, seedPrompt, *maxIters, *verbose),
		oolongEvaluator{},
		optimize.GEPAAdapterConfig{
			PopulationSize:  *population,
			MaxGenerations:  *generations,
			ReflectionFreq:  1,
			EvalConcurrency: 1,
			PassThreshold:   passThreshold,
			PrimaryArtifact: optimize.ArtifactRLMIterationPrompt,
			ValidationSplit: defaultValidation,
		},
	)

	result, err := optimizer.Optimize(ctx, optimize.GEPAOptimizeRequest{
		SeedArtifacts:      seedAgent.GetArtifacts(),
		TrainingExamples:   trainExamples,
		ValidationExamples: validationExamples,
	})
	if err != nil {
		logger.Fatalf(ctx, "GEPA optimization failed: %v", err)
	}

	optimizedAgent := newAdaptiveAgent(llm, weakSeedPrompt(), *maxIters, *verbose)
	if err := optimizedAgent.SetArtifacts(result.BestArtifacts); err != nil {
		logger.Fatalf(ctx, "Applying optimized artifacts failed: %v", err)
	}

	optimizedRun, err := harness.Run(ctx, optimizedAgent, examples)
	if err != nil {
		logger.Fatalf(ctx, "Optimized run failed: %v", err)
	}

	optimizedPrompt := strings.TrimSpace(result.BestArtifacts.Text[optimize.ArtifactRLMIterationPrompt])
	fmt.Println()
	fmt.Println("Optimized iteration prompt:")
	fmt.Println(optimizedPrompt)
	if result.BestValidationEvaluation != nil && result.BestValidationEvaluation.Fitness != nil {
		fmt.Println()
		fmt.Printf("Best validation weighted fitness: %.2f\n", result.BestValidationEvaluation.Fitness.WeightedScore)
		fmt.Printf("Best validation average score: %.2f\n", result.BestValidationEvaluation.AverageScore)
	}
	fmt.Println()
	printRunSummary("Optimized", optimizedRun)

	report := buildEvalReport(evalReportInput{
		Config: evalConfig{
			Provider:        *provider,
			Model:           modelName,
			TaskSource:      taskSource,
			TaskOffset:      *taskOffset,
			TaskCount:       len(tasks),
			Population:      *population,
			Generations:     *generations,
			MaxIterations:   *maxIters,
			ValidationSplit: defaultValidation,
		},
		Tasks:               tasks,
		TrainingCount:       len(trainExamples),
		ValidationCount:     len(validationExamples),
		SeedIterationPrompt: strings.TrimSpace(seedPrompt),
		OptimizedPrompt:     optimizedPrompt,
		Baseline:            baselineRun,
		Optimized:           optimizedRun,
		BestValidation:      result.BestValidationEvaluation,
	})

	if *outputFile != "" {
		if err := writeReport(*outputFile, report); err != nil {
			logger.Fatalf(ctx, "Writing eval report failed: %v", err)
		}
		fmt.Printf("\nWrote eval report to %s\n", *outputFile)
	}
}

func buildLLM(ctx context.Context, logger *logging.Logger, provider, model, apiKey string) (core.LLM, string, error) {
	switch provider {
	case "anthropic":
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return nil, "", fmt.Errorf("ANTHROPIC_API_KEY environment variable not set")
		}
		modelName := model
		if modelName == "" {
			modelName = "claude-haiku-4-5"
		}
		llm, err := llms.NewAnthropicLLM(apiKey, anthropic.Model(modelName))
		if err != nil {
			return nil, "", err
		}
		logger.Info(ctx, "Using Anthropic model: %s", modelName)
		return llm, modelName, nil

	case "openai":
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		if apiKey == "" {
			return nil, "", fmt.Errorf("OPENAI_API_KEY environment variable not set")
		}
		modelID := core.ModelOpenAIGPT4oMini
		if model != "" {
			modelID = core.ModelID(model)
		}
		llm, err := llms.NewOpenAI(modelID, apiKey)
		if err != nil {
			return nil, "", err
		}
		logger.Info(ctx, "Using OpenAI model: %s", modelID)
		return llm, string(modelID), nil

	case "gemini":
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
			if apiKey == "" {
				apiKey = os.Getenv("GEMINI_API_KEY")
			}
		}
		if apiKey == "" {
			return nil, "", fmt.Errorf("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
		}
		modelID := core.ModelGoogleGeminiFlash
		if model != "" {
			modelID = core.ModelID(model)
		}
		llm, err := llms.NewGeminiLLM(apiKey, modelID)
		if err != nil {
			return nil, "", err
		}
		logger.Info(ctx, "Using Gemini model: %s", modelID)
		return llm, string(modelID), nil
	}

	return nil, "", fmt.Errorf("unsupported provider %q", provider)
}

func loadTasks(taskFile string, useHuggingFace bool, offset, count int) ([]datasets.OolongTask, string, error) {
	var tasks []datasets.OolongTask
	var err error
	source := "embedded"

	switch {
	case taskFile != "":
		tasks, err = datasets.LoadOolongTasksFromFile(taskFile)
		source = "file"
	case useHuggingFace:
		tasks, err = datasets.FetchOolongTasksFromHuggingFaceRange(offset, count)
		source = "huggingface"
	default:
		tasks = datasets.SampleOolongTasks()
	}
	if err != nil {
		return nil, "", err
	}

	if !useHuggingFace {
		tasks = datasets.SliceOolongTasks(tasks, offset, count)
	}
	if len(tasks) == 0 {
		return nil, "", fmt.Errorf("no OOLONG tasks found for source %s (offset=%d, count=%d)", source, offset, count)
	}
	return tasks, source, nil
}

func toAgentExamples(tasks []datasets.OolongTask) []optimize.AgentExample {
	examples := make([]optimize.AgentExample, 0, len(tasks))
	for _, task := range tasks {
		task = task.Normalize()
		examples = append(examples, optimize.AgentExample{
			ID: task.ID,
			Inputs: map[string]interface{}{
				"context": task.ContextWindowText,
				"query":   task.Question,
			},
			Outputs: map[string]interface{}{
				"answer": task.Answer,
			},
			Metadata: map[string]interface{}{
				"answer_type": task.AnswerType,
				"task_group":  task.TaskGroup,
				"context_len": task.ContextLen,
				"dataset":     task.Dataset,
			},
		})
	}
	return examples
}

func splitExamples(examples []optimize.AgentExample, validationSplit float64) ([]optimize.AgentExample, []optimize.AgentExample) {
	if len(examples) <= 1 || validationSplit <= 0 {
		return examples, nil
	}

	validationCount := int(float64(len(examples)) * validationSplit)
	if validationCount <= 0 {
		validationCount = 1
	}
	if validationCount >= len(examples) {
		validationCount = len(examples) - 1
	}

	split := len(examples) - validationCount
	train := append([]optimize.AgentExample(nil), examples[:split]...)
	validation := append([]optimize.AgentExample(nil), examples[split:]...)
	return train, validation
}

func printRunSummary(label string, run *optimize.HarnessRunResult) {
	if run == nil {
		return
	}

	fmt.Printf("%s average score: %.2f\n", label, run.AverageScore)
	fmt.Printf("%s passed examples: %d/%d\n", label, run.PassedExamples, run.CompletedExamples)
	if len(run.Results) == 0 || run.Results[0].Result == nil || run.Results[0].Result.SideInfo == nil {
		return
	}

	trace := run.Results[0].Result.SideInfo.Trace
	if trace == nil {
		return
	}
	fmt.Printf("%s first trace termination: %s\n", label, trace.TerminationCause)
	fmt.Printf("%s first trace steps: %d\n", label, len(trace.Steps))
	fmt.Printf("%s first trace sub-LLM calls: %d\n", label, intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubLLMCallCount))
}

type oolongEvaluator struct{}

func (oolongEvaluator) Evaluate(ctx context.Context, agent optimize.OptimizableAgent, ex optimize.AgentExample) (*optimize.EvalResult, error) {
	startedAt := time.Now()
	output, execErr := agent.Execute(ctx, core.ShallowCopyMap(ex.Inputs))
	trace := latestTrace(agent)
	latency := time.Since(startedAt)
	if trace != nil && trace.ProcessingTime > 0 {
		latency = trace.ProcessingTime
	}

	sideInfo := &optimize.SideInfo{
		Trace:       trace,
		Diagnostics: map[string]interface{}{},
		Scores:      map[string]float64{},
		LatencyMS:   float64(latency) / float64(time.Millisecond),
	}

	if ex.Metadata != nil {
		sideInfo.Diagnostics["task_group"] = ex.Metadata["task_group"]
		sideInfo.Diagnostics["answer_type"] = ex.Metadata["answer_type"]
		sideInfo.Diagnostics["context_len"] = ex.Metadata["context_len"]
	}

	if execErr != nil {
		sideInfo.Diagnostics["execution_error"] = execErr.Error()
		return &optimize.EvalResult{Score: 0, SideInfo: sideInfo}, nil
	}

	expectedAnswer, _ := ex.Outputs["answer"].(string)
	actualAnswer, _ := output["answer"].(string)
	answerScore := 0.0
	if datasets.CheckOolongAnswer(expectedAnswer, actualAnswer) {
		answerScore = 1.0
	} else {
		sideInfo.Diagnostics["expected_answer"] = expectedAnswer
		sideInfo.Diagnostics["actual_answer"] = actualAnswer
	}

	contextInteraction := 0.0
	if hasUsefulContextInteraction(trace) {
		contextInteraction = 1.0
	}

	terminationScore := 0.0
	if trace != nil && (trace.TerminationCause == "final_answer" || trace.TerminationCause == "state_final" || trace.TerminationCause == "regex_final") {
		terminationScore = 1.0
	}

	sideInfo.Scores["answer_match"] = answerScore
	sideInfo.Scores["context_interaction"] = contextInteraction
	sideInfo.Scores["termination"] = terminationScore
	if trace != nil {
		sideInfo.Diagnostics["termination_cause"] = trace.TerminationCause
		sideInfo.Diagnostics["step_count"] = len(trace.Steps)
	}

	if answerScore < 1 {
		sideInfo.FailedTests = append(sideInfo.FailedTests, "answer_match")
	} else {
		sideInfo.PassedTests = append(sideInfo.PassedTests, "answer_match")
	}

	return &optimize.EvalResult{
		Score:    answerScore,
		SideInfo: sideInfo,
	}, nil
}

func latestTrace(agent optimize.OptimizableAgent) *agents.ExecutionTrace {
	provider, ok := agent.(interface{ LastExecutionTrace() *agents.ExecutionTrace })
	if !ok {
		return nil
	}
	return provider.LastExecutionTrace()
}

func hasUsefulContextInteraction(trace *agents.ExecutionTrace) bool {
	if trace == nil {
		return false
	}
	for tool, count := range trace.ToolUsageCount {
		if tool != "" && count > 0 {
			return true
		}
	}
	for _, step := range trace.Steps {
		switch step.Tool {
		case "explore", "query", "compute", "subrlm":
			return true
		}
	}
	return intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubLLMCallCount) > 0 ||
		intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubRLMCallCount) > 0
}

func intMetric(metadata map[string]interface{}, key string) int {
	if metadata == nil {
		return 0
	}
	switch value := metadata[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	default:
		return 0
	}
}

func newAdaptiveAgent(llm core.LLM, iterationPrompt string, maxIterations int, verbose bool) *agentrlm.Agent {
	module := modrlm.NewFromLLM(
		llm,
		modrlm.WithAdaptiveIterationConfig(modrlm.AdaptiveIterationConfig{
			Enabled:                true,
			BaseIterations:         3,
			MaxIterations:          maxIterations,
			ContextScaleFactor:     1 << 19,
			EnableEarlyTermination: true,
			ConfidenceThreshold:    2,
		}),
		modrlm.WithMaxIterations(maxIterations),
		modrlm.WithTimeout(5*time.Minute),
		modrlm.WithVerbose(verbose),
	)

	agent := agentrlm.NewAgent("oolong-rlm-gepa", module)
	artifacts := agent.GetArtifacts()
	artifacts.Text[optimize.ArtifactRLMIterationPrompt] = iterationPrompt
	if err := agent.SetArtifacts(artifacts); err != nil {
		panic(err)
	}
	return agent
}

func weakSeedPrompt() string {
	return strings.TrimSpace(`
Review the context methodically and answer the question once you can support it from the evidence.
Prefer one strong exploration or query step over many small loops.
If the answer is numeric or categorical, extract it exactly before finalizing.
`)
}

type evalReportInput struct {
	Config              evalConfig
	Tasks               []datasets.OolongTask
	TrainingCount       int
	ValidationCount     int
	SeedIterationPrompt string
	OptimizedPrompt     string
	Baseline            *optimize.HarnessRunResult
	Optimized           *optimize.HarnessRunResult
	BestValidation      *optimize.GEPACandidateEvaluation
}

func buildEvalReport(input evalReportInput) evalReport {
	taskIDs := make([]string, 0, len(input.Tasks))
	for _, task := range input.Tasks {
		normalized := task.Normalize()
		if normalized.ID != "" {
			taskIDs = append(taskIDs, normalized.ID)
		}
	}

	report := evalReport{
		GeneratedAt:              time.Now().UTC(),
		Configuration:            input.Config,
		TrainingExampleCount:     input.TrainingCount,
		ValidationExampleCount:   input.ValidationCount,
		TaskIDs:                  taskIDs,
		SeedIterationPrompt:      input.SeedIterationPrompt,
		OptimizedIterationPrompt: input.OptimizedPrompt,
		Baseline:                 summarizeRun(input.Baseline),
		Optimized:                summarizeRun(input.Optimized),
	}
	if input.BestValidation != nil {
		report.BestValidationAverageScore = input.BestValidation.AverageScore
		if input.BestValidation.Fitness != nil {
			report.BestValidationWeightedFitness = input.BestValidation.Fitness.WeightedScore
			report.BestValidationObjectives = map[string]float64{
				"success_rate":   input.BestValidation.Fitness.SuccessRate,
				"output_quality": input.BestValidation.Fitness.OutputQuality,
				"efficiency":     input.BestValidation.Fitness.Efficiency,
				"robustness":     input.BestValidation.Fitness.Robustness,
				"generalization": input.BestValidation.Fitness.Generalization,
				"diversity":      input.BestValidation.Fitness.Diversity,
				"innovation":     input.BestValidation.Fitness.Innovation,
			}
		}
	}
	return report
}

func summarizeRun(run *optimize.HarnessRunResult) runSummary {
	summary := runSummary{
		TerminationCounts: map[string]int{},
	}
	if run == nil {
		return summary
	}

	summary.AverageScore = run.AverageScore
	summary.PassedExamples = run.PassedExamples
	summary.FailedExamples = run.FailedExamples
	summary.CompletedExamples = run.CompletedExamples
	summary.EvaluationErrors = run.EvaluationErrors
	summary.ExampleIDs = make([]string, 0, len(run.Results))

	var totalSteps int
	var totalSubLLM int
	var totalSubRLM int
	var traced int
	for _, example := range run.Results {
		summary.ExampleIDs = append(summary.ExampleIDs, example.ExampleID)
		if example.Result == nil || example.Result.SideInfo == nil || example.Result.SideInfo.Trace == nil {
			continue
		}
		trace := example.Result.SideInfo.Trace
		traced++
		totalSteps += len(trace.Steps)
		totalSubLLM += intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubLLMCallCount)
		totalSubRLM += intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubRLMCallCount)
		termination := trace.TerminationCause
		if termination == "" {
			termination = "unknown"
		}
		summary.TerminationCounts[termination]++
	}
	if traced > 0 {
		summary.AverageSteps = float64(totalSteps) / float64(traced)
		summary.AverageSubLLM = float64(totalSubLLM) / float64(traced)
		summary.AverageSubRLM = float64(totalSubRLM) / float64(traced)
	}
	if len(summary.TerminationCounts) == 0 {
		summary.TerminationCounts = nil
	}
	return summary
}

func writeReport(path string, report evalReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}
