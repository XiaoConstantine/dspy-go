package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
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
		[]core.InputField{{Name: "question"}},
		[]core.OutputField{{Field: core.NewField("answer")}},
	)

	// Create ChainOfThought module with native JSON structured output
	// This uses GenerateWithJSON instead of text parsing, which is more robust
	// and eliminates parsing errors from malformed LLM responses
	cot := modules.NewChainOfThought(signature).WithStructuredOutput()

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
		// interceptors.ValidationModuleInterceptor(interceptors.DefaultValidationConfig()),

		// Caching interceptor - caches results for identical inputs (5 minute TTL)
		interceptors.CachingModuleInterceptor(cache, 5*time.Minute),

		// Timeout interceptor - prevents modules from running too long
		interceptors.TimeoutModuleInterceptor(30 * time.Second),

		// Retry interceptor - retries failed executions with exponential backoff
		interceptors.RetryModuleInterceptor(interceptors.RetryConfig{
			MaxAttempts: 3,
			Delay:       1 * time.Second,
			Backoff:     2.0,
		}),
	}

	// Set interceptors on the module
	// ChainOfThought implements InterceptableModule interface
	// Note: Append to existing interceptors to preserve structured output functionality
	existingInterceptors := cot.GetInterceptors()
	allInterceptors := append(existingInterceptors, moduleInterceptors...)
	cot.SetInterceptors(allInterceptors)
	logger.Info(ctx, "Successfully configured %d total interceptors for ChainOfThought module (including structured output interceptor)", len(allInterceptors))

	// Create program with generation options from configuration
	// The forward function is built from the program's own modules, so the
	// compiled program runs the module that holds the bootstrapped demos.
	program := core.NewProgramWithForwardFactory(map[string]core.Module{"cot": cot}, func(mods map[string]core.Module) func(context.Context, map[string]any) (map[string]any, error) {
		cot := mods["cot"].(*modules.ChainOfThought)
		return func(ctx context.Context, inputs map[string]any) (map[string]any, error) {
			// Generation options come only from an explicit config file. Without
			// one the model's defaults apply, because some providers (such as
			// openai-codex) reject an explicit temperature.
			var opts []core.Option
			if cfg != nil {
				opts = append(opts, core.WithGenerateOptions(
					core.WithTemperature(cfg.LLM.Default.Generation.Temperature),
					core.WithMaxTokens(cfg.LLM.Default.Generation.MaxTokens),
				))
			}

			// Use ProcessWithInterceptors with the configured interceptors
			// ChainOfThought implements InterceptableModule interface
			return cot.ProcessWithInterceptors(ctx, inputs, nil, opts...)
		}
	})

	// Create optimizer
	optimizer := optimizers.NewBootstrapFewShot(nil, 5)

	// Prepare training set as core.Examples
	trainExamples := make([]core.Example, len(examples[:10]))
	for i, ex := range examples[:10] {
		trainExamples[i] = core.Example{
			Inputs: map[string]any{
				"question": ex.Question,
			},
			Outputs: map[string]any{
				"answer": ex.Answer,
			},
		}
	}

	// A prediction becomes a demonstration when its final number matches the
	// reference, which GSM8K writes after "####" at the end of its solution.
	scorer := core.ScorerFunc(func(_ context.Context, example core.Example, prediction map[string]any) (float64, error) {
		expected := finalNumber(fmt.Sprint(example.Outputs["answer"]))
		if expected != "" && expected == finalNumber(fmt.Sprint(prediction["answer"])) {
			return 1, nil
		}
		return 0, nil
	})

	// Compile the program
	compiledProgram, err := optimizer.CompileWithScorer(ctx, program, trainExamples, scorer)
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
		result, err := compiledProgram.Execute(ctx, map[string]any{"question": ex.Question})
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

	testQuestion := map[string]any{"question": examples[10].Question}

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

var numberPattern = regexp.MustCompile(`-?\d+(\.\d+)?`)

// finalNumber returns the last number in text with separators removed, or ""
// when there is none.
func finalNumber(text string) string {
	numbers := numberPattern.FindAllString(strings.ReplaceAll(text, ",", ""), -1)
	if len(numbers) == 0 {
		return ""
	}
	return strings.TrimSuffix(strings.TrimSuffix(numbers[len(numbers)-1], ".00"), ".0")
}

func main() {
	configPath := flag.String("config", "", "Path to configuration file (optional)")
	apiKey := flag.String("api-key", "", "API Key (optional, can use DSPY_API_KEY env var)")
	flag.Parse()

	RunGSM8KExample(*configPath, *apiKey)
}
