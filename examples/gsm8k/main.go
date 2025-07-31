package main

import (
	"context"
	"flag"
	"log"
	"os"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/config"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/interceptors"
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

	// Setup interceptors to showcase functionality
	logger.Info(ctx, "Setting up module interceptors for enhanced observability and reliability...")

	// Create memory cache for caching module results
	cache := interceptors.NewMemoryCache()
	defer cache.Stop() // Clean up background goroutines

	// Configure interceptors for the ChainOfThought module
	moduleInterceptors := []core.ModuleInterceptor{
		// Logging interceptor - logs module execution start/completion
		interceptors.LoggingModuleInterceptor(),

		// Metrics interceptor - tracks performance metrics
		interceptors.MetricsModuleInterceptor(),

		// Tracing interceptor - adds distributed tracing spans
		interceptors.TracingModuleInterceptor(),

		// Input validation interceptor - validates inputs for safety
		interceptors.ValidationModuleInterceptor(interceptors.DefaultValidationConfig()),

		// Caching interceptor - caches results for identical inputs (5 minute TTL)
		interceptors.CachingModuleInterceptor(cache, 5*time.Minute),

		// Timeout interceptor - prevents modules from running too long
		interceptors.TimeoutModuleInterceptor(30*time.Second),

		// Retry interceptor - retries failed executions with exponential backoff
		interceptors.RetryModuleInterceptor(interceptors.RetryConfig{
			MaxAttempts: 3,
			Delay:       1*time.Second,
			Backoff:     2.0,
		}),
	}

	// Set interceptors on the module
	// ChainOfThought implements InterceptableModule interface
	cot.SetInterceptors(moduleInterceptors)
	logger.Info(ctx, "Successfully configured %d interceptors for ChainOfThought module", len(moduleInterceptors))

	// Create program with generation options from configuration
	program := core.NewProgram(map[string]core.Module{"cot": cot}, func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
		temperature := 0.7
		maxTokens := 8192

		// Use config values if available
		if cfg != nil {
			// The config manager handles defaults, so we can use the values directly.
			temperature = cfg.LLM.Default.Generation.Temperature
			maxTokens = cfg.LLM.Default.Generation.MaxTokens
		}

		// Use ProcessWithInterceptors with the configured interceptors
		// ChainOfThought implements InterceptableModule interface
		return cot.ProcessWithInterceptors(ctx, inputs, nil, core.WithGenerateOptions(
			core.WithTemperature(temperature),
			core.WithMaxTokens(maxTokens),
		))
	})

	// Create optimizer
	optimizer := optimizers.NewBootstrapFewShot(func(example, prediction map[string]interface{}, ctx context.Context) bool {
		return example["answer"] == prediction["answer"]
	}, 5)

	// Prepare training set as core.Examples
	trainExamples := make([]core.Example, len(examples[:10]))
	for i, ex := range examples[:10] {
		trainExamples[i] = core.Example{
			Inputs: map[string]interface{}{
				"question": ex.Question,
			},
			Outputs: map[string]interface{}{
				"answer": ex.Answer,
			},
		}
	}

	// Create dataset
	trainDataset := datasets.NewSimpleDataset(trainExamples)

	// Define metric function
	metricFunc := func(expected, actual map[string]interface{}) float64 {
		if expected["answer"] == actual["answer"] {
			return 1.0
		}
		return 0.0
	}

	// Compile the program
	compiledProgram, err := optimizer.Compile(ctx, program, trainDataset, metricFunc)
	if err != nil {
		logger.Fatalf(ctx, "Failed to compile program: %v", err)
	}

	// Test the compiled program
	logger.Info(ctx, "\n=== Testing compiled program with interceptors ===")
	logger.Info(ctx, "The following execution will demonstrate interceptor benefits:")
	logger.Info(ctx, "- Logging: Start/completion of each module execution")
	logger.Info(ctx, "- Metrics: Performance timing and success/failure tracking")
	logger.Info(ctx, "- Tracing: Distributed tracing spans for observability")
	logger.Info(ctx, "- Validation: Input safety checks")
	logger.Info(ctx, "- Caching: Duplicate questions will be served from cache")
	logger.Info(ctx, "- Timeout: Protection against long-running operations")
	logger.Info(ctx, "- Retry: Automatic retry on failures with exponential backoff\n")

	for i, ex := range examples[10:15] {
		logger.Info(ctx, "--- Processing question %d ---", i+1)
		result, err := compiledProgram.Execute(ctx, map[string]interface{}{"question": ex.Question})
		if err != nil {
			log.Printf("Error executing program: %v", err)
			continue
		}

		logger.Info(ctx, "Question: %s", ex.Question)
		logger.Info(ctx, "Predicted Answer: %s", result["answer"])
		logger.Info(ctx, "Actual Answer: %s\n", ex.Answer)
	}

	// Demonstrate caching by asking the same question twice
	logger.Info(ctx, "\n=== Demonstrating caching interceptor ===")
	logger.Info(ctx, "Running the same question twice - second execution should be cached:")

	testQuestion := map[string]interface{}{"question": examples[10].Question}

	logger.Info(ctx, "\nFirst execution (will be cached):")
	start := time.Now()
	result1, _ := compiledProgram.Execute(ctx, testQuestion)
	duration1 := time.Since(start)
	logger.Info(ctx, "First execution took: %v", duration1)

	logger.Info(ctx, "\nSecond execution (should use cache):")
	start = time.Now()
	result2, _ := compiledProgram.Execute(ctx, testQuestion)
	duration2 := time.Since(start)
	logger.Info(ctx, "Second execution took: %v (should be much faster due to caching)", duration2)

	// Verify results are identical
	if result1["answer"] == result2["answer"] {
		logger.Info(ctx, "✅ Cache working correctly - identical results in %v vs %v", duration1, duration2)
	} else {
		logger.Warn(ctx, "⚠️  Cache may not be working - results differ")
	}
}

func main() {
	configPath := flag.String("config", "", "Path to configuration file (optional)")
	apiKey := flag.String("api-key", "", "API Key (optional, can use DSPY_API_KEY env var)")
	flag.Parse()

	RunGSM8KExample(*configPath, *apiKey)
}
