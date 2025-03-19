package main

import (
	"context"
	"flag"
	"log"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func RunGSM8KExample(apiKey string) {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	ctx := core.WithExecutionState(context.Background())
	// Setup LLM
	llms.EnsureFactory()
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to setup llm")
	}

	// Load GSM8K dataset
	examples, err := datasets.LoadGSM8K()
	if err != nil {
		log.Fatalf("Failed to load GSM8K dataset: %v", err)
	}

	// Create signature for ChainOfThought
	signature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "question"}}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ChainOfThought module
	cot := modules.NewChainOfThought(signature)

	// Create program
	program := core.NewProgram(map[string]core.Module{"cot": cot}, func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		return cot.Process(ctx, inputs, core.WithGenerateOptions(
			core.WithTemperature(0.7),
			core.WithMaxTokens(8192),
		))
	})

	// Create optimizer
	optimizer := optimizers.NewBootstrapFewShot(func(example, prediction map[string]interface{}, ctx context.Context) bool {
		return example["answer"] == prediction["answer"]
	}, 5)

	// Prepare training set
	trainset := make([]map[string]interface{}, len(examples[:10]))
	for i, ex := range examples[:10] {
		trainset[i] = map[string]interface{}{
			"question": ex.Question,
			"answer":   ex.Answer,
		}
	}

	// Compile the program
	compiledProgram, err := optimizer.Compile(ctx, program, program, trainset)
	if err != nil {
		logger.Fatalf(ctx, "Failed to compile program: %v", err)
	}

	// Test the compiled program
	for _, ex := range examples[10:15] {
		result, err := compiledProgram.Execute(ctx, map[string]interface{}{"question": ex.Question})
		if err != nil {
			log.Printf("Error executing program: %v", err)
			continue
		}

		logger.Info(ctx, "Question: %s\n", ex.Question)
		logger.Info(ctx, "Predicted Answer: %s\n", result["answer"])
		logger.Info(ctx, "Actual Answer: %s\n\n", ex.Answer)
	}
}

func main() {
	apiKey := flag.String("api-key", "", "Anthropic API Key")
	flag.Parse()

	RunGSM8KExample(*apiKey)
}
