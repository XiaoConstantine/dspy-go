package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
	"github.com/sourcegraph/conc/pool"
)

var goroutineMonitor *GoroutineMonitor

func computeF1(prediction, ground_truth string) float64 {
	pred_tokens := strings.Fields(strings.ToLower(prediction))
	truth_tokens := strings.Fields(strings.ToLower(ground_truth))

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

	p := pool.New().WithMaxGoroutines(core.GlobalConfig.ConcurrencyLevel)
	for _, ex := range examples {
		example := ex
		p.Go(func() {
			logger.Debug(ctx, "Starting new corotine")

			goroutineMonitor.TrackGoroutine()

			defer goroutineMonitor.ReleaseGoroutine()
			result, err := program.Execute(context.Background(), map[string]interface{}{"question": ex.Question})
			if err != nil {
				logger.Error(ctx, "Error executing program: %v", err)
				return
			}

			predictedAnswer, ok := result["answer"].(string)
			if !ok {
				logger.Error(ctx, "Error: Could not find answer in result: %v", result)
				return
			}

			f1 := computeF1(fmt.Sprintf("%v", predictedAnswer), example.Answer)
			exactMatch := 0.0
			if predictedAnswer == example.Answer {
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

func RunHotPotQAExample(apiKey string) {
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

	trainSize := int(0.025 * float64(len(examples)))
	valSize := int(0.1 * float64(len(examples)))
	trainExamples := examples[:trainSize]
	valExamples := examples[trainSize : trainSize+valSize]
	testExamples := examples[trainSize+valSize:]

	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	cot := modules.NewChainOfThought(signature)

	program := core.NewProgram(map[string]core.Module{"cot": cot}, func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		return cot.Process(ctx, inputs)
	})

	metric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
		return computeF1(prediction["answer"].(string), example["answer"].(string)) > 0.5
	}

	optimizer := optimizers.NewBootstrapFewShot(metric, 5)

	trainset := make([]map[string]interface{}, len(trainExamples))
	for i, ex := range trainExamples {
		trainset[i] = map[string]interface{}{
			"question": ex.Question,
			"answer":   ex.Answer,
		}
	}

	compiledProgram, err := optimizer.Compile(ctx, program, program, trainset)
	if err != nil {
		logger.Fatalf(ctx, "Failed to compile program: %v", err)
	}

	valF1, valExactMatch := evaluateModel(ctx, compiledProgram, valExamples)
	logger.Info(ctx, "Validation Results - F1: %.4f, Exact Match: %.4f\n", valF1, valExactMatch)

	testF1, testExactMatch := evaluateModel(ctx, compiledProgram, testExamples)
	logger.Info(ctx, "Test Results - F1: %.4f, Exact Match: %.4f\n", testF1, testExactMatch)

	// Example predictions
	for _, ex := range testExamples[:5] {
		result, err := compiledProgram.Execute(ctx, map[string]interface{}{"question": ex.Question})
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
	flag.Parse()

	RunHotPotQAExample(*apiKey)
}
