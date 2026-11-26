package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/adapters"
	"github.com/XiaoConstantine/dspy-go/pkg/cache"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	dspyerrors "github.com/XiaoConstantine/dspy-go/pkg/errors"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

// ComparisonMetrics tracks metrics for comparison.
type ComparisonMetrics struct {
	Scores         []float64 `json:"scores"`
	ExecutionTimes []float64 `json:"execution_times"`
	TokenUsage     []int     `json:"token_usage"`
}

func (cm *ComparisonMetrics) AddScore(score float64) {
	cm.Scores = append(cm.Scores, score)
}

func (cm *ComparisonMetrics) AddExecutionTime(time float64) {
	cm.ExecutionTimes = append(cm.ExecutionTimes, time)
}

func (cm *ComparisonMetrics) AddTokenUsage(tokens int) {
	cm.TokenUsage = append(cm.TokenUsage, tokens)
}

func (cm *ComparisonMetrics) GetAverageScore() float64 {
	if len(cm.Scores) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, score := range cm.Scores {
		sum += score
	}
	return sum / float64(len(cm.Scores))
}

func (cm *ComparisonMetrics) GetTotalTokens() int {
	sum := 0
	for _, tokens := range cm.TokenUsage {
		sum += tokens
	}
	return sum
}

func (cm *ComparisonMetrics) GetTotalTime() float64 {
	sum := 0.0
	for _, time := range cm.ExecutionTimes {
		sum += time
	}
	return sum
}

// OptimizerComparison handles the comparison between Go and Python implementations.
type OptimizerComparison struct {
	ModelName string
	LLM       core.LLM
	// Adapter is "none", "xml" or "json"; see newPredictor.
	Adapter string

	BootstrapMetrics *ComparisonMetrics
	MIPROMetrics     *ComparisonMetrics
	SIMBAMetrics     *ComparisonMetrics
	CoProMetrics     *ComparisonMetrics
	GEPAMetrics      *ComparisonMetrics
}

// resolveModel returns the model to run, its label in the results, and its
// credential. An empty model is Gemini Flash, as before the -model flag
// existed. A model with the "openai-codex:" prefix runs on a ChatGPT
// subscription and reads its OAuth access token from OPENAI_OAUTH_TOKEN.
func resolveModel(model string) (core.ModelID, string, string) {
	modelID, label, keyVar := core.ModelID(model), model, "GEMINI_API_KEY"
	switch {
	case model == "":
		modelID, label = core.ModelGoogleGeminiFlash, "gemini-2.0-flash"
	case strings.HasPrefix(model, "openai-codex:"):
		keyVar = "OPENAI_OAUTH_TOKEN"
	case strings.HasPrefix(model, "gpt-"), strings.HasPrefix(model, "o1"), strings.HasPrefix(model, "o3"):
		keyVar = "OPENAI_API_KEY"
	}
	apiKey := os.Getenv(keyVar)
	if apiKey == "" {
		log.Fatalf("%s environment variable is required", keyVar)
	}
	return modelID, label, apiKey
}

func NewOptimizerComparison(model string) *OptimizerComparison {
	modelID, modelName, apiKey := resolveModel(model)

	// Initialize factory
	llms.EnsureFactory()

	// Configure default LLM
	err := core.ConfigureDefaultLLM(apiKey, modelID)
	if err != nil {
		log.Fatalf("Failed to configure default LLM: %v", err)
	}

	// Configure teacher LLM for optimization
	err = core.ConfigureTeacherLLM(apiKey, modelID)
	if err != nil {
		log.Fatalf("Failed to configure teacher LLM: %v", err)
	}

	// Get the configured LLM for reference
	llm, err := llms.NewLLM(apiKey, modelID)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	return &OptimizerComparison{
		ModelName:        modelName,
		LLM:              llm,
		BootstrapMetrics: &ComparisonMetrics{},
		MIPROMetrics:     &ComparisonMetrics{},
		SIMBAMetrics:     &ComparisonMetrics{},
		CoProMetrics:     &ComparisonMetrics{},
		GEPAMetrics:      &ComparisonMetrics{},
	}
}

func (oc *OptimizerComparison) CreateSampleDataset(size int) []core.Example {
	sampleData := []struct {
		question string
		answer   string
	}{
		{"What is the capital of France?", "Paris"},
		{"What is 2 + 2?", "4"},
		{"What color is the sky?", "Blue"},
		{"What is the largest planet?", "Jupiter"},
		{"What is the smallest prime number?", "2"},
		{"What is the chemical symbol for water?", "H2O"},
		{"What is the speed of light?", "299,792,458 m/s"},
		{"What year did World War II end?", "1945"},
		{"What is the square root of 16?", "4"},
		{"What is the boiling point of water?", "100°C"},
	}

	examples := make([]core.Example, size)
	for i := 0; i < size; i++ {
		data := sampleData[i%len(sampleData)]
		examples[i] = core.Example{
			Inputs: map[string]interface{}{
				"question": data.question,
				"prompt":   data.question, // Add prompt field for MIPRO compatibility
			},
			Outputs: map[string]interface{}{
				"answer": data.answer,
			},
		}
	}

	return examples
}

// scorer adapts AccuracyMetric for CompileWithScorer. AccuracyMetric keeps no
// state, so optimizers may call it concurrently.
func (oc *OptimizerComparison) scorer(counts *outputCounts, signature core.Signature) core.Scorer {
	return core.ScorerFunc(func(ctx context.Context, example core.Example, prediction map[string]interface{}) (float64, error) {
		counts.recordScored(signature, prediction)
		return oc.AccuracyMetric(example.Outputs, prediction, ctx), nil
	})
}

// newPredictor builds the program's predictor with the adapter chosen by the
// -adapter flag. With "none" it is a plain Predict, as before the flag existed.
func (oc *OptimizerComparison) newPredictor(signature core.Signature) *modules.Predict {
	predictor := modules.NewPredict(signature)
	switch oc.Adapter {
	case "xml":
		return predictor.WithAdapter(adapters.NewXMLAdapter())
	case "json":
		return predictor.WithAdapter(adapters.NewJSONAdapter())
	}
	return predictor
}

// replyRecorder keeps the text of the model's latest Generate reply.
type replyRecorder struct {
	core.LLM
	mu    sync.Mutex
	reply string
}

func (r *replyRecorder) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	response, err := r.LLM.Generate(ctx, prompt, opts...)
	if err == nil && response != nil {
		r.mu.Lock()
		r.reply = response.Content
		r.mu.Unlock()
	}
	return response, err
}

func (r *replyRecorder) last() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.reply
}

// outputCounts records how one optimizer's program calls ended. A silent
// failure is a call that returned no error and lacks a declared output.
type outputCounts struct {
	mu sync.Mutex

	// Held-out evaluation calls.
	calls, silentFailures, blankOutputs, outputErrors, otherErrors int
	// Predictions the optimizer scored while compiling. Calls that failed
	// while compiling never reach the scorer, so they are not counted.
	compileScored, compileSilentFailures int
	// Baseline runs only: successful calls that used the adapter's fallback
	// request, and the tokens the model reported.
	fallbacks, promptTokens, completionTokens int
}

func (c *outputCounts) recordFallback(used bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if used {
		c.fallbacks++
	}
}

func (c *outputCounts) recordTokens(usage *core.TokenUsage) {
	if usage == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.promptTokens += usage.PromptTokens
	c.completionTokens += usage.CompletionTokens
}

// recordCall counts one held-out call and returns how it ended: "ok",
// "blank" (every output present, a text one empty), "silent_failure",
// "output_error" or "error".
func (c *outputCounts) recordCall(signature core.Signature, prediction map[string]interface{}, err error) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	switch {
	case dspyerrors.IsOutputError(err):
		c.outputErrors++
		return "output_error"
	case err != nil:
		c.otherErrors++
		return "error"
	case len(signature.MissingOutputs(prediction)) > 0:
		c.silentFailures++
		return "silent_failure"
	}
	// A blank answer is not a silent failure, but without an adapter the
	// parser returns nil for it and with one it returns "". Counting it keeps
	// the two comparable.
	for _, field := range signature.Outputs {
		if text, ok := prediction[field.Name].(string); ok && strings.TrimSpace(text) == "" {
			c.blankOutputs++
			return "blank"
		}
	}
	return "ok"
}

func (c *outputCounts) recordScored(signature core.Signature, prediction map[string]interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.compileScored++
	if len(signature.MissingOutputs(prediction)) > 0 {
		c.compileSilentFailures++
	}
}

func (c *outputCounts) report() map[string]interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	rate := func(part, whole int) float64 {
		if whole == 0 {
			return 0
		}
		return float64(part) / float64(whole)
	}
	return map[string]interface{}{
		"calls":                       c.calls,
		"silent_failures":             c.silentFailures,
		"silent_failure_rate":         rate(c.silentFailures, c.calls),
		"blank_outputs":               c.blankOutputs,
		"output_errors":               c.outputErrors,
		"other_errors":                c.otherErrors,
		"compile_scored":              c.compileScored,
		"compile_silent_failures":     c.compileSilentFailures,
		"compile_silent_failure_rate": rate(c.compileSilentFailures, c.compileScored),
		"fallbacks":                   c.fallbacks,
		"prompt_tokens":               c.promptTokens,
		"completion_tokens":           c.completionTokens,
	}
}

func (oc *OptimizerComparison) AccuracyMetric(example, prediction map[string]interface{}, ctx context.Context) float64 {
	expected, ok := example["answer"].(string)
	if !ok {
		return 0.0
	}

	predicted, ok := prediction["answer"].(string)
	if !ok {
		return 0.0
	}

	// Simple substring matching for demo purposes - match Python DSPy behavior
	if len(expected) > 0 && len(predicted) > 0 {
		expectedLower := strings.ToLower(strings.TrimSpace(expected))
		predictedLower := strings.ToLower(strings.TrimSpace(predicted))

		// Bidirectional substring matching: expected in predicted OR predicted in expected
		if strings.Contains(expectedLower, predictedLower) || strings.Contains(predictedLower, expectedLower) {
			return 1.0
		}
	}

	return 0.0
}

// CreateHotpotQADataset returns the first size HotpotQA validation questions.
// It downloads the dataset on first use.
func (oc *OptimizerComparison) CreateHotpotQADataset(size int) ([]core.Example, error) {
	rows, err := datasets.LoadHotpotQA()
	if err != nil {
		return nil, err
	}
	if size > len(rows) {
		return nil, fmt.Errorf("HotpotQA has %d questions, fewer than the %d requested", len(rows), size)
	}
	examples := make([]core.Example, size)
	for i, row := range rows[:size] {
		examples[i] = core.Example{
			Inputs: map[string]interface{}{
				"question": row.Question,
				"prompt":   row.Question, // MIPRO reads the prompt field
			},
			Outputs: map[string]interface{}{"answer": row.Answer},
		}
	}
	return examples, nil
}

// TestBaseline runs the unoptimized predictor over the whole dataset. It makes
// one model call per example, so it is the cheapest way to measure the output
// contract on a live model.
func (oc *OptimizerComparison) TestBaseline(ctx context.Context, dataset []core.Example) map[string]interface{} {
	log.Println("Testing unoptimized Predict")

	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}
	// Keep each raw reply, to log the ones that yield no answer.
	recorder := &replyRecorder{LLM: core.GetDefaultLLM()}
	if oc.Adapter == "none" {
		predictor.SetLLM(recorder)
	}

	// Count the calls that needed the adapter's fallback request.
	ctx = modules.WithPredictionTraceObserver(ctx, func(_ *modules.Predict, _, _ map[string]any, meta modules.PredictionTraceMeta) {
		counts.recordFallback(meta.FallbackUsed)
	})

	startTime := time.Now()
	totalScore := 0.0
	examples := make([]map[string]interface{}, 0, len(dataset))
	for _, example := range dataset {
		// A fresh execution state per call, so its token usage is this call's.
		callCtx := core.WithExecutionState(ctx)
		prediction, err := predictor.Process(callCtx, example.Inputs)
		outcome := counts.recordCall(signature, prediction, err)
		counts.recordTokens(core.GetExecutionState(callCtx).GetTokenUsage())
		score := 0.0
		switch outcome {
		case "output_error", "error":
			log.Printf("Error during evaluation: %v", err)
			var parseErr *dspyerrors.ParseError
			if errors.As(err, &parseErr) {
				log.Printf("Rejected reply: %q", parseErr.Raw)
			}
		case "silent_failure", "blank":
			log.Printf("No answer (%s) from reply %q; outputs %v", outcome, recorder.last(), prediction)
		default:
			score = oc.AccuracyMetric(example.Outputs, prediction, ctx)
		}
		totalScore += score
		examples = append(examples, map[string]interface{}{"outcome": outcome, "score": score})
	}

	results := map[string]interface{}{
		"optimizer":       "Baseline",
		"evaluation_time": time.Since(startTime).Seconds(),
		"average_score":   totalScore / float64(len(dataset)),
		"output_contract": counts.report(),
		"total_examples":  len(dataset),
		"examples":        examples,
	}
	log.Printf("Baseline results: %+v", results)
	return results
}

func (oc *OptimizerComparison) TestBootstrapFewShot(ctx context.Context, dataset []core.Example, maxBootstrappedDemos int) (core.Program, map[string]interface{}) {
	log.Println("Testing BootstrapFewShot optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(bound map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			predictor := bound["predictor"]
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return predictor.Process(ctx, inputs)
			}
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Convert examples to map format expected by optimizer
	trainset := make([]map[string]interface{}, len(trainExamples))
	for i, example := range trainExamples {
		trainset[i] = map[string]interface{}{
			"question": example.Inputs["question"],
			"prompt":   example.Inputs["prompt"], // Include prompt for MIPRO
			"answer":   example.Outputs["answer"],
		}
	}

	// Create optimizer; predictions the scorer rates above zero become demos.
	optimizer := optimizers.NewBootstrapFewShot(nil, maxBootstrappedDemos)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.CompileWithScorer(ctx, program, trainExamples, oc.scorer(counts, signature))
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		counts.recordCall(signature, prediction, err)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.BootstrapMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":              "BootstrapFewShot",
		"compilation_time":       compilationTime,
		"average_score":          avgScore,
		"output_contract":        counts.report(),
		"total_examples":         len(valset),
		"max_bootstrapped_demos": maxBootstrappedDemos,
		"demonstrations":         oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("BootstrapFewShot results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestMIPRO(ctx context.Context, dataset []core.Example, numTrials int, maxBootstrappedDemos int) (core.Program, map[string]interface{}) {
	log.Println("Testing MIPRO optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(bound map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			predictor := bound["predictor"]
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return predictor.Process(ctx, inputs)
			}
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create optimizer
	optimizer := optimizers.NewMIPRO(
		nil, // the scorer is passed to CompileWithScorer
		optimizers.WithMode(optimizers.LightMode),
		optimizers.WithNumTrials(numTrials),
		optimizers.WithMaxLabeledDemos(maxBootstrappedDemos),
	)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.CompileWithScorer(ctx, program, trainExamples, oc.scorer(counts, signature))
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		counts.recordCall(signature, prediction, err)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.MIPROMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":              "MIPRO",
		"compilation_time":       compilationTime,
		"average_score":          avgScore,
		"output_contract":        counts.report(),
		"total_examples":         len(valset),
		"num_trials":             numTrials,
		"max_bootstrapped_demos": maxBootstrappedDemos,
		"demonstrations":         oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("MIPRO results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestCOPRO(ctx context.Context, dataset []core.Example, breadth int, depth int) (core.Program, map[string]interface{}) {
	log.Println("Testing COPRO optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(bound map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			predictor := bound["predictor"]
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return predictor.Process(ctx, inputs)
			}
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create COPRO optimizer (prompt optimization algorithm)
	coproOptimizer := optimizers.NewCOPRO(
		nil,                                 // the scorer is passed to CompileWithScorer
		optimizers.WithBreadth(breadth),     // Number of prompt candidates
		optimizers.WithDepth(depth),         // Refinement iterations
		optimizers.WithInitTemperature(1.2), // Exploration randomness
		optimizers.WithTrackStats(false),
	)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := coproOptimizer.CompileWithScorer(ctx, program, trainExamples, oc.scorer(counts, signature))
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		counts.recordCall(signature, prediction, err)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.CoProMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":        "COPRO",
		"compilation_time": compilationTime,
		"average_score":    avgScore,
		"output_contract":  counts.report(),
		"total_examples":   len(valset),
		"breadth":          breadth,
		"depth":            depth,
		"demonstrations":   oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("COPRO results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestSIMBA(ctx context.Context, dataset []core.Example, batchSize int, maxSteps int) (core.Program, map[string]interface{}) {
	log.Println("Testing SIMBA optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(bound map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			predictor := bound["predictor"]
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return predictor.Process(ctx, inputs)
			}
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create optimizer with fast mode for better compatibility test performance
	optimizer := optimizers.NewSIMBA(
		optimizers.WithSIMBABatchSize(batchSize),
		optimizers.WithSIMBAMaxSteps(maxSteps),
		optimizers.WithSIMBANumCandidates(4),
		optimizers.WithSamplingTemperature(0.2),
		optimizers.WithFastMode(true), // Enable fast mode for compatibility tests
	)

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := optimizer.CompileWithScorer(ctx, program, trainExamples, oc.scorer(counts, signature))
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		counts.recordCall(signature, prediction, err)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.SIMBAMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":        "SIMBA",
		"compilation_time": compilationTime,
		"average_score":    avgScore,
		"output_contract":  counts.report(),
		"total_examples":   len(valset),
		"batch_size":       batchSize,
		"max_steps":        maxSteps,
		"demonstrations":   oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("SIMBA results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) TestGEPA(ctx context.Context, dataset []core.Example, maxGenerations int) (core.Program, map[string]interface{}) {
	log.Println("Testing GEPA optimizer")

	// Create a simple program for testing
	signature := core.NewSignature(
		[]core.InputField{{Field: core.NewField("question", core.WithDescription("The question to answer"))}},
		[]core.OutputField{{Field: core.NewField("answer", core.WithDescription("The answer to the question"))}},
	)
	predictor := oc.newPredictor(signature)
	counts := &outputCounts{}

	program := core.NewProgramWithForwardFactory(
		map[string]core.Module{
			"predictor": predictor,
		},
		func(bound map[string]core.Module) func(context.Context, map[string]interface{}) (map[string]interface{}, error) {
			predictor := bound["predictor"]
			return func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
				return predictor.Process(ctx, inputs)
			}
		},
	)

	// Split dataset - handle small datasets properly
	datasetSize := len(dataset)
	trainSize := min(datasetSize*3/4, datasetSize-1) // Use 3/4 for training, leave at least 1 for validation
	if trainSize < 1 {
		trainSize = 1
	}

	trainExamples := dataset[:trainSize]
	valset := dataset[trainSize:]

	// Create GEPA optimizer with smaller parameters for compatibility testing
	config := optimizers.DefaultGEPAConfig()
	config.MaxGenerations = maxGenerations
	config.EvaluationBatchSize = 3 // Smaller batch for testing
	config.ConcurrencyLevel = 2    // Reduced concurrency for testing

	gepaOptimizer, err := optimizers.NewGEPA(config)
	if err != nil {
		log.Printf("Error creating GEPA optimizer: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Compile program
	startTime := time.Now()
	optimizedProgram, err := gepaOptimizer.CompileWithScorer(ctx, program, trainExamples, oc.scorer(counts, signature))
	compilationTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Error during compilation: %v", err)
		return program, map[string]interface{}{
			"error": err.Error(),
		}
	}

	// Evaluate on validation set
	totalScore := 0.0
	for _, example := range valset {
		prediction, err := optimizedProgram.Execute(ctx, example.Inputs)
		counts.recordCall(signature, prediction, err)
		if err != nil {
			log.Printf("Error during evaluation: %v", err)
			continue
		}

		score := oc.AccuracyMetric(example.Outputs, prediction, ctx)
		totalScore += score
		oc.GEPAMetrics.AddScore(score)
	}

	avgScore := totalScore / float64(len(valset))

	results := map[string]interface{}{
		"optimizer":        "GEPA",
		"compilation_time": compilationTime,
		"average_score":    avgScore,
		"output_contract":  counts.report(),
		"total_examples":   len(valset),
		"max_generations":  maxGenerations,
		"demonstrations":   oc.extractDemonstrations(optimizedProgram),
	}

	log.Printf("GEPA results: %+v", results)
	return optimizedProgram, results
}

func (oc *OptimizerComparison) extractDemonstrations(program core.Program) []map[string]interface{} {
	demonstrations := []map[string]interface{}{}

	// Try to extract demos from predictors
	for _, module := range program.Modules {
		if predictor, ok := module.(*modules.Predict); ok {
			demos := predictor.GetDemos()
			for _, demo := range demos {
				demonstrations = append(demonstrations, map[string]interface{}{
					"inputs":  demo.Inputs,
					"outputs": demo.Outputs,
				})
			}
		}
	}

	return demonstrations
}

func (oc *OptimizerComparison) RunComparison(ctx context.Context, datasetSize int) map[string]interface{} {
	log.Println("Starting optimizer comparison")

	// Create dataset
	dataset := oc.CreateSampleDataset(datasetSize)

	// Test optimizers
	_, bootstrapResults := oc.TestBootstrapFewShot(ctx, dataset, 4)
	_, miproResults := oc.TestMIPRO(ctx, dataset, 5, 3)
	_, simbaResults := oc.TestSIMBA(ctx, dataset, 4, 6)

	// Compare results
	bootstrapScore := bootstrapResults["average_score"].(float64)
	miproScore := miproResults["average_score"].(float64)
	simbaScore := simbaResults["average_score"].(float64)
	bootstrapTime := bootstrapResults["compilation_time"].(float64)
	miproTime := miproResults["compilation_time"].(float64)
	simbaTime := simbaResults["compilation_time"].(float64)

	// Find best optimizer
	bestOptimizer := "BootstrapFewShot"
	bestScore := bootstrapScore
	if miproScore > bestScore {
		bestOptimizer = "MIPRO"
		bestScore = miproScore
	}
	if simbaScore > bestScore {
		bestOptimizer = "SIMBA"
		bestScore = simbaScore
	}

	comparisonResults := map[string]interface{}{
		"dataset_size":      datasetSize,
		"model":             oc.ModelName,
		"bootstrap_fewshot": bootstrapResults,
		"mipro":             miproResults,
		"simba":             simbaResults,
		"comparison": map[string]interface{}{
			"bootstrap_vs_mipro_score_diff": miproScore - bootstrapScore,
			"bootstrap_vs_simba_score_diff": simbaScore - bootstrapScore,
			"mipro_vs_simba_score_diff":     simbaScore - miproScore,
			"bootstrap_vs_mipro_time_diff":  miproTime - bootstrapTime,
			"bootstrap_vs_simba_time_diff":  simbaTime - bootstrapTime,
			"mipro_vs_simba_time_diff":      simbaTime - miproTime,
			"best_optimizer":                bestOptimizer,
			"best_score":                    bestScore,
		},
	}

	return comparisonResults
}

func (oc *OptimizerComparison) SaveResults(results map[string]interface{}, filename string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return err
	}

	log.Printf("Results saved to %s", filename)
	return nil
}

func main() {
	var optimizer = flag.String("optimizer", "all", "Optimizer to test: bootstrap, mipro, simba, copro, gepa, or all; baseline runs the unoptimized predictor")
	var datasetSize = flag.Int("dataset-size", 20, "Dataset size for testing")
	var enableCache = flag.Bool("enable-cache", false, "Enable caching for performance comparison")
	var cacheType = flag.String("cache-type", "memory", "Cache type: memory or sqlite")
	var withoutCache = flag.Bool("without-cache", false, "Run without cache for baseline comparison")
	var adapter = flag.String("adapter", "none", "Output adapter for the program's predictor: none, xml, or json")
	var model = flag.String("model", "", "Model to run; empty is Gemini Flash. Use the openai-codex: prefix for a ChatGPT subscription, such as openai-codex:gpt-6-luna")
	var datasetName = flag.String("dataset", "sample", "Dataset: sample (10 built-in questions, repeated) or hotpotqa")
	flag.Parse()
	if *adapter != "none" && *adapter != "xml" && *adapter != "json" {
		log.Fatalf("unknown -adapter %q: use none, xml, or json", *adapter)
	}

	ctx := context.Background()

	// Configure caching based on flags
	if *enableCache && !*withoutCache {
		fmt.Printf("Enabling %s cache for performance testing...\n", *cacheType)
		// Set environment variables to enable caching
		os.Setenv("DSPY_CACHE_ENABLED", "true")
		os.Setenv("DSPY_CACHE_TYPE", *cacheType)
		os.Setenv("DSPY_CACHE_TTL", "1h")

		if *cacheType == "sqlite" {
			homeDir, _ := os.UserHomeDir()
			cacheDir := filepath.Join(homeDir, ".dspy-go")
			if err := os.MkdirAll(cacheDir, 0755); err != nil {
				log.Printf("Warning: could not create cache directory: %v", err)
			}
			os.Setenv("DSPY_CACHE_PATH", filepath.Join(cacheDir, "cache.db"))
		} else {
			os.Setenv("DSPY_CACHE_MAX_SIZE", "100MB")
		}
	} else if *withoutCache {
		fmt.Println("Running without cache for baseline comparison...")
		os.Setenv("DSPY_CACHE_ENABLED", "false")
	}

	// Initialize comparison
	comparison := NewOptimizerComparison(*model)
	comparison.Adapter = *adapter

	// Log cache status
	if *enableCache && !*withoutCache {
		fmt.Printf("Cache enabled: %s\n", *cacheType)
		// Clear cache at start to ensure clean test
		if err := cache.ClearGlobalCache(ctx); err != nil {
			log.Printf("Warning: Could not clear cache: %v", err)
		}
	}

	// Create dataset
	var dataset []core.Example
	switch *datasetName {
	case "sample":
		dataset = comparison.CreateSampleDataset(*datasetSize)
	case "hotpotqa":
		var err error
		if dataset, err = comparison.CreateHotpotQADataset(*datasetSize); err != nil {
			log.Fatalf("Failed to load HotpotQA: %v", err)
		}
	default:
		log.Fatalf("unknown -dataset %q: use sample or hotpotqa", *datasetName)
	}

	results := map[string]interface{}{
		"dataset":          *datasetName,
		"dataset_size":     *datasetSize,
		"model":            comparison.ModelName,
		"optimizer_tested": *optimizer,
		"adapter":          *adapter,
	}

	if *optimizer == "baseline" {
		fmt.Println("Testing unoptimized Predict...")
		results["baseline"] = comparison.TestBaseline(ctx, dataset)
	}

	// Test individual optimizers based on parameter
	if *optimizer == "bootstrap" || *optimizer == "all" {
		fmt.Println("Testing BootstrapFewShot...")
		_, bootstrapResults := comparison.TestBootstrapFewShot(ctx, dataset, 4)
		results["bootstrap_fewshot"] = bootstrapResults
	}

	if *optimizer == "mipro" || *optimizer == "all" {
		fmt.Println("Testing MIPRO...")
		_, miproResults := comparison.TestMIPRO(ctx, dataset, 5, 3)
		results["mipro"] = miproResults
	}

	if *optimizer == "simba" || *optimizer == "all" {
		fmt.Println("Testing SIMBA...")
		_, simbaResults := comparison.TestSIMBA(ctx, dataset, 4, 6)
		results["simba"] = simbaResults
	}

	if *optimizer == "gepa" || *optimizer == "all" {
		fmt.Println("Testing GEPA...")
		_, gepaResults := comparison.TestGEPA(ctx, dataset, 3)
		results["gepa"] = gepaResults
	}

	if *optimizer == "copro" || *optimizer == "all" {
		fmt.Println("Testing COPRO...")
		_, coproResults := comparison.TestCOPRO(ctx, dataset, 5, 2) // breadth=5, depth=2
		results["copro"] = coproResults
	}

	// Add comparison section if testing multiple optimizers
	if *optimizer == "all" {
		bootstrapResults := results["bootstrap_fewshot"].(map[string]interface{})
		miproResults := results["mipro"].(map[string]interface{})
		simbaResults := results["simba"].(map[string]interface{})
		coproResults := results["copro"].(map[string]interface{})
		gepaResults := results["gepa"].(map[string]interface{})

		bootstrapScore := bootstrapResults["average_score"].(float64)
		miproScore := miproResults["average_score"].(float64)
		simbaScore := simbaResults["average_score"].(float64)
		coproScore := coproResults["average_score"].(float64)
		gepaScore := gepaResults["average_score"].(float64)
		bootstrapTime := bootstrapResults["compilation_time"].(float64)
		miproTime := miproResults["compilation_time"].(float64)
		simbaTime := simbaResults["compilation_time"].(float64)
		coproTime := coproResults["compilation_time"].(float64)
		gepaTime := gepaResults["compilation_time"].(float64)

		// Find best optimizer
		bestOptimizer := "BootstrapFewShot"
		bestScore := bootstrapScore
		if miproScore > bestScore {
			bestOptimizer = "MIPRO"
			bestScore = miproScore
		}
		if simbaScore > bestScore {
			bestOptimizer = "SIMBA"
			bestScore = simbaScore
		}
		if coproScore > bestScore {
			bestOptimizer = "COPRO"
			bestScore = coproScore
		}
		if gepaScore > bestScore {
			bestOptimizer = "GEPA"
			bestScore = gepaScore
		}

		results["comparison"] = map[string]interface{}{
			"bootstrap_vs_mipro_score_diff": miproScore - bootstrapScore,
			"bootstrap_vs_simba_score_diff": simbaScore - bootstrapScore,
			"bootstrap_vs_copro_score_diff": coproScore - bootstrapScore,
			"bootstrap_vs_gepa_score_diff":  gepaScore - bootstrapScore,
			"mipro_vs_simba_score_diff":     simbaScore - miproScore,
			"mipro_vs_copro_score_diff":     coproScore - miproScore,
			"mipro_vs_gepa_score_diff":      gepaScore - miproScore,
			"simba_vs_copro_score_diff":     coproScore - simbaScore,
			"simba_vs_gepa_score_diff":      gepaScore - simbaScore,
			"copro_vs_gepa_score_diff":      gepaScore - coproScore,
			"bootstrap_vs_mipro_time_diff":  miproTime - bootstrapTime,
			"bootstrap_vs_simba_time_diff":  simbaTime - bootstrapTime,
			"bootstrap_vs_copro_time_diff":  coproTime - bootstrapTime,
			"bootstrap_vs_gepa_time_diff":   gepaTime - bootstrapTime,
			"mipro_vs_simba_time_diff":      simbaTime - miproTime,
			"mipro_vs_copro_time_diff":      coproTime - miproTime,
			"mipro_vs_gepa_time_diff":       gepaTime - miproTime,
			"simba_vs_copro_time_diff":      coproTime - simbaTime,
			"simba_vs_gepa_time_diff":       gepaTime - simbaTime,
			"copro_vs_gepa_time_diff":       gepaTime - coproTime,
			"best_optimizer":                bestOptimizer,
			"best_score":                    bestScore,
		}
	}

	// Add cache statistics if caching was enabled
	if *enableCache && !*withoutCache {
		cacheStats := cache.GetGlobalCacheStats()
		results["cache_stats"] = map[string]interface{}{
			"hits":     cacheStats.Hits,
			"misses":   cacheStats.Misses,
			"sets":     cacheStats.Sets,
			"size":     cacheStats.Size,
			"hit_rate": float64(cacheStats.Hits) / float64(cacheStats.Hits+cacheStats.Misses),
		}
		fmt.Printf("\nCache Statistics:\n")
		fmt.Printf("  - Hits: %d\n", cacheStats.Hits)
		fmt.Printf("  - Misses: %d\n", cacheStats.Misses)
		fmt.Printf("  - Hit Rate: %.2f%%\n", float64(cacheStats.Hits)/float64(cacheStats.Hits+cacheStats.Misses)*100)
		fmt.Printf("  - Cache Size: %d bytes\n", cacheStats.Size)
	}

	// Save results
	filename := "go_comparison_results.json"
	if *enableCache && !*withoutCache {
		filename = "go_comparison_results_cached.json"
	} else if *withoutCache {
		filename = "go_comparison_results_nocache.json"
	}

	err := comparison.SaveResults(results, filename)
	if err != nil {
		log.Printf("Error saving results: %v", err)
	}

	// Print summary
	fmt.Println("\n=== dspy-go Optimizer Comparison Results ===")
	fmt.Printf("Dataset size: %v\n", results["dataset_size"])
	fmt.Printf("Model: %v\n", results["model"])
	fmt.Printf("Optimizer tested: %v\n", results["optimizer_tested"])

	if bootstrapResults, ok := results["bootstrap_fewshot"].(map[string]interface{}); ok {
		fmt.Printf("\nBootstrapFewShot:\n")
		fmt.Printf("  - Average score: %.3f\n", bootstrapResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", bootstrapResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(bootstrapResults["demonstrations"].([]map[string]interface{})))
	}

	if miproResults, ok := results["mipro"].(map[string]interface{}); ok {
		fmt.Printf("\nMIPRO:\n")
		fmt.Printf("  - Average score: %.3f\n", miproResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", miproResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(miproResults["demonstrations"].([]map[string]interface{})))
	}

	if simbaResults, ok := results["simba"].(map[string]interface{}); ok {
		fmt.Printf("\nSIMBA:\n")
		fmt.Printf("  - Average score: %.3f\n", simbaResults["average_score"])
		fmt.Printf("  - Compilation time: %.2fs\n", simbaResults["compilation_time"])
		fmt.Printf("  - Demonstrations: %d\n", len(simbaResults["demonstrations"].([]map[string]interface{})))
	}

	if coproResults, ok := results["copro"].(map[string]interface{}); ok {
		fmt.Printf("\nCOPRO:\n")
		if avgScore, exists := coproResults["average_score"]; exists && avgScore != nil {
			fmt.Printf("  - Average score: %.3f\n", avgScore)
		} else {
			fmt.Printf("  - Average score: N/A (compilation failed)\n")
		}
		if compTime, exists := coproResults["compilation_time"]; exists && compTime != nil {
			fmt.Printf("  - Compilation time: %.2fs\n", compTime)
		} else {
			fmt.Printf("  - Compilation time: N/A\n")
		}
		if breadth, exists := coproResults["breadth"]; exists && breadth != nil {
			fmt.Printf("  - Breadth: %v\n", breadth)
		}
		if depth, exists := coproResults["depth"]; exists && depth != nil {
			fmt.Printf("  - Depth: %v\n", depth)
		}
		if demos, exists := coproResults["demonstrations"]; exists && demos != nil {
			if demoSlice, ok := demos.([]map[string]interface{}); ok {
				fmt.Printf("  - Demonstrations: %d\n", len(demoSlice))
			} else {
				fmt.Printf("  - Demonstrations: 0 (error extracting)\n")
			}
		} else {
			fmt.Printf("  - Demonstrations: 0\n")
		}
		if err, exists := coproResults["error"]; exists {
			fmt.Printf("  - Error: %s\n", err)
		}
	}

	if gepaResults, ok := results["gepa"].(map[string]interface{}); ok {
		fmt.Printf("\nGEPA:\n")
		if avgScore, exists := gepaResults["average_score"]; exists && avgScore != nil {
			fmt.Printf("  - Average score: %.3f\n", avgScore)
		} else {
			fmt.Printf("  - Average score: N/A (compilation failed)\n")
		}
		if compTime, exists := gepaResults["compilation_time"]; exists && compTime != nil {
			fmt.Printf("  - Compilation time: %.2fs\n", compTime)
		} else {
			fmt.Printf("  - Compilation time: N/A\n")
		}
		if maxGen, exists := gepaResults["max_generations"]; exists && maxGen != nil {
			fmt.Printf("  - Max generations: %v\n", maxGen)
		}
		if demos, exists := gepaResults["demonstrations"]; exists && demos != nil {
			if demoSlice, ok := demos.([]map[string]interface{}); ok {
				fmt.Printf("  - Demonstrations: %d\n", len(demoSlice))
			} else {
				fmt.Printf("  - Demonstrations: 0 (error extracting)\n")
			}
		} else {
			fmt.Printf("  - Demonstrations: 0\n")
		}
		if err, exists := gepaResults["error"]; exists {
			fmt.Printf("  - Error: %s\n", err)
		}
	}

	if comparisonResults, ok := results["comparison"].(map[string]interface{}); ok {
		fmt.Printf("\nBest optimizer: %v (%.3f)\n", comparisonResults["best_optimizer"], comparisonResults["best_score"])
	}
}

// Helper function for min.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
