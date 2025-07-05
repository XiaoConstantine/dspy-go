package main

import (
	"context"
	"flag"
	"log"
	"os"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

func RunGSM8KExample(configPath string, apiKey string) {
	ctx := core.WithExecutionState(context.Background())

	// Setup logging
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	// Load configuration if provided
	var cfg *config.Config
	if configPath != "" {
		configManager, err := config.NewManager(config.WithConfigPath(configPath))
		if err != nil {
			log.Fatalf("Failed to create config manager: %v", err)
		}
		if err := configManager.Load(); err != nil {
			log.Fatalf("Failed to load config from file: %v", err)
		}
		cfg = configManager.Get()
	}
	// Note: Don't use default config if not provided since it may have incompatible settings

	// Setup LLM
	llms.EnsureFactory()
	
	// Use API key from parameter or environment
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GOOGLE_API_KEY")
		}
		if apiKey == "" {
			apiKey = os.Getenv("DSPY_API_KEY")
		}
	}
	
	// Configure LLM - use config if available, otherwise fallback to default
	if apiKey != "" {
		modelID := core.ModelGoogleGeminiFlash // Default
		if cfg != nil && cfg.LLM.Default.ModelID != "" {
			modelID = core.ModelID(cfg.LLM.Default.ModelID)
		}
		
		err := core.ConfigureDefaultLLM(apiKey, modelID)
		if err != nil {
			logger.Fatalf(ctx, "Failed to setup LLM: %v", err)
		}
	} else {
		logger.Fatalf(ctx, "API key is required. Provide via --api-key flag or GEMINI_API_KEY/GOOGLE_API_KEY/DSPY_API_KEY environment variable")
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

	// Create program with generation options from configuration
	program := core.NewProgram(map[string]core.Module{"cot": cot}, func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		temperature := 0.7
		maxTokens := 8192
		
		// Use config values if available
		if cfg != nil && cfg.LLM.Default.Generation.Temperature > 0 {
			temperature = cfg.LLM.Default.Generation.Temperature
		}
		if cfg != nil && cfg.LLM.Default.Generation.MaxTokens > 0 {
			maxTokens = cfg.LLM.Default.Generation.MaxTokens
		}
		
		return cot.Process(ctx, inputs, core.WithGenerateOptions(
			core.WithTemperature(temperature),
			core.WithMaxTokens(maxTokens),
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
	configPath := flag.String("config", "", "Path to configuration file (optional)")
	apiKey := flag.String("api-key", "", "API Key (optional, can use DSPY_API_KEY env var)")
	flag.Parse()

	RunGSM8KExample(*configPath, *apiKey)
}
