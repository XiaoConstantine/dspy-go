package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"slices"
	"strings"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/sourcegraph/conc/pool"
)

var goroutineMonitor *GoroutineMonitor

// answerTokens lowercases an answer and drops punctuation and symbols before
// splitting it into words, so "Yes." and "yes" compare equal.
func answerTokens(s string) []string {
	return strings.Fields(strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			return -1
		}
		return unicode.ToLower(r)
	}, s))
}

func computeF1(prediction, ground_truth string) float64 {
	pred_tokens := answerTokens(prediction)
	truth_tokens := answerTokens(ground_truth)

	common := make(map[string]bool)
	for _, token := range pred_tokens {
		for _, truth_token := range truth_tokens {
			if token == truth_token {
				common[token] = true
				break
			}
		}
	}

	if len(pred_tokens) == 0 || len(truth_tokens) == 0 {
		return 0.0
	}

	precision := float64(len(common)) / float64(len(pred_tokens))
	recall := float64(len(common)) / float64(len(truth_tokens))

	if precision+recall == 0 {
		return 0.0
	}

	return 2 * precision * recall / (precision + recall)
}

// Helper function to get map keys for debugging.
func keys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func evaluateModel(ctx context.Context, program core.Program, examples []datasets.HotPotQAExample) (float64, float64) {
	logger := logging.GetLogger()
	var totalF1, exactMatch float64
	var validExamples int32

	results := make(chan struct {
		f1         float64
		exactMatch float64
	}, len(examples))

	total := len(examples)
	var processed int32 = 0

	// Start a goroutine to periodically report progress
	go func() {
		ticker := time.NewTicker(time.Second * 10)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				current := atomic.LoadInt32(&processed)
				logger.Info(ctx, "Progress: %d/%d (%.2f%%)\n", current, total, float64(current)/float64(total)*100)
			}
		}
	}()

	p := pool.New().WithMaxGoroutines(core.GetConcurrencyLevel())
	for _, ex := range examples {
		example := ex
		p.Go(func() {
			logger.Debug(ctx, "Starting new corotine")

			goroutineMonitor.TrackGoroutine()

			defer goroutineMonitor.ReleaseGoroutine()
			result, err := program.Execute(context.Background(), map[string]any{"question": ex.Question})
			if err != nil {
				logger.Error(ctx, "Error executing program: %v", err)
				return
			}

			predictedAnswer, ok := result["answer"].(string)
			if !ok {
				// Check if we have rationale field (indicates parsing worked partially)
				if _, hasRationale := result["rationale"]; hasRationale {
					logger.Error(ctx, "Error: XML parsed rationale but missing answer field. Result: %v", result)
				} else {
					logger.Error(ctx, "Error: XML parsing failed, falling back to text. Keys: %v", keys(result))
				}
				return
			}

			f1 := computeF1(fmt.Sprintf("%v", predictedAnswer), example.Answer)
			exactMatch := 0.0
			if slices.Equal(answerTokens(predictedAnswer), answerTokens(example.Answer)) {
				exactMatch = 1.0
			}
			results <- struct {
				f1         float64
				exactMatch float64
			}{f1: f1, exactMatch: exactMatch}
			atomic.AddInt32(&processed, 1)
		})
	}
	go func() {
		p.Wait()
		close(results)
	}()
	for result := range results {
		totalF1 += result.f1
		exactMatch += result.exactMatch
		atomic.AddInt32(&validExamples, 1)
	}
	if validExamples == 0 {
		log.Printf("Warning: No valid examples processed")
		return 0, 0
	}

	avgF1 := totalF1 / float64(validExamples)
	exactMatchAccuracy := exactMatch / float64(validExamples)

	return avgF1, exactMatchAccuracy
}

// RunHotPotQAExample optimizes and evaluates a question-answering program on
// HotpotQA. A positive maxExamples limits the run to that many questions; the
// full validation split has 7,405, and each one costs a model call.
func RunHotPotQAExample(apiKey string, maxExamples int) {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)
	ctx := core.WithExecutionState(context.Background())
	goroutineMonitor = NewGoroutineMonitor(5*time.Second, ctx)
	goroutineMonitor.Start()
	defer goroutineMonitor.Stop()

	// Setup LLM
	llms.EnsureFactory()
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to setup llm")
	}

	// Set concurrency level
	core.SetConcurrencyOptions(10)
	examples, err := datasets.LoadHotpotQA()
	if err != nil {
		logger.Fatalf(ctx, "Failed to load HotPotQA dataset: %v", err)
	}

	rand.Shuffle(len(examples), func(i, j int) { examples[i], examples[j] = examples[j], examples[i] })
	if maxExamples > 0 && len(examples) > maxExamples {
		examples = examples[:maxExamples]
	}

	trainSize := int(0.025 * float64(len(examples)))
	valSize := int(0.1 * float64(len(examples)))
	trainExamples := examples[:trainSize]
	valExamples := examples[trainSize : trainSize+valSize]
	testExamples := examples[trainSize+valSize:]

	signature := core.NewSignature(
		[]core.InputField{{Name: "question"}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Use native JSON structured output for more reliable parsing
	cot := modules.NewChainOfThought(signature).WithStructuredOutput()

	// The forward function is built from the program's own modules, so the
	// compiled program runs the module that holds the bootstrapped demos.
	program := core.NewProgramWithForwardFactory(map[string]core.Module{"cot": cot}, func(mods map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			return mods["cot"].Process(ctx, inputs)
		}
	})

	// Keep predictions whose F1 against the reference answer is at least 0.5.
	optimizer := optimizers.NewBootstrapFewShot(nil, 5)
	optimizer.MetricThreshold = 0.5

	// Convert to core.Examples
	trainCoreExamples := make([]core.Example, len(trainExamples))
	for i, ex := range trainExamples {
		trainCoreExamples[i] = core.Example{
			Inputs: map[string]any{
				"question": ex.Question,
			},
			Outputs: map[string]any{
				"answer": ex.Answer,
			},
		}
	}

	// A prediction without a string answer cannot be scored, which the
	// optimizer treats as a prediction it does not keep.
	scorer := core.ScorerFunc(func(_ context.Context, example core.Example, prediction map[string]any) (float64, error) {
		predicted, ok := prediction["answer"].(string)
		if !ok {
			return 0, fmt.Errorf("prediction has no string answer")
		}
		expected, ok := example.Outputs["answer"].(string)
		if !ok {
			return 0, fmt.Errorf("example has no string answer")
		}
		return computeF1(predicted, expected), nil
	})

	compiledProgram, err := optimizer.CompileWithScorer(ctx, program, trainCoreExamples, scorer)
	if err != nil {
		logger.Fatalf(ctx, "Failed to compile program: %v", err)
	}

	valF1, valExactMatch := evaluateModel(ctx, compiledProgram, valExamples)
	logger.Info(ctx, "Validation Results - F1: %.4f, Exact Match: %.4f\n", valF1, valExactMatch)

	testF1, testExactMatch := evaluateModel(ctx, compiledProgram, testExamples)
	logger.Info(ctx, "Test Results - F1: %.4f, Exact Match: %.4f\n", testF1, testExactMatch)

	// Example predictions
	for _, ex := range testExamples[:5] {
		result, err := compiledProgram.Execute(ctx, map[string]any{"question": ex.Question})
		if err != nil {
			logger.Error(ctx, "Error executing program: %v", err)
			continue
		}
		logger.Info(ctx, "Question: %s\n", ex.Question)
		logger.Info(ctx, "Predicted Answer: %s\n", result["answer"])
		logger.Info(ctx, "Actual Answer: %s\n", ex.Answer)
		logger.Info(ctx, "F1 Score: %.4f\n\n", computeF1(result["answer"].(string), ex.Answer))
	}
}

func main() {
	apiKey := flag.String("api-key", "", "Anthropic API Key")
	maxExamples := flag.Int("max-examples", 0, "Limit the run to this many questions (0 uses the whole dataset)")
	flag.Parse()

	RunHotPotQAExample(*apiKey, *maxExamples)
}
