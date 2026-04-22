package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	agentrlm "github.com/XiaoConstantine/dspy-go/pkg/agents/rlm"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

type policySpec struct {
	Name        string
	Description string
	Options     func(adaptiveThreshold int) []modrlm.Option
}

type policySummary struct {
	Name                 string
	Description          string
	Answer               string
	Termination          string
	Steps                int
	HistoryCompressions  int
	SubLLMCalls          int
	RootPromptMeanTokens int
	RootPromptMaxTokens  int
	Duration             time.Duration
	Error                string
}

func main() {
	provider := flag.String("provider", "anthropic", "LLM provider to use: anthropic, openai, or gemini")
	model := flag.String("model", "", "Model to use (provider default if empty)")
	apiKey := flag.String("api-key", "", "API key override (otherwise uses provider environment variable)")
	maxIters := flag.Int("max-iters", 6, "Maximum RLM iterations per policy run")
	adaptiveThreshold := flag.Int("adaptive-threshold", 2, "History-entry threshold where adaptive replay switches to checkpointed replay")
	repeats := flag.Int("repeats", 120, "How many times to repeat the sample review document")
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

	document := reviewDocument(*repeats)
	query := "What percentage of reviews are positive (4-5 stars) versus negative (1-2 stars)?"

	policies := []policySpec{
		{
			Name:        string(modrlm.ContextPolicyFull),
			Description: "Replay every iteration verbatim.",
			Options: func(_ int) []modrlm.Option {
				return []modrlm.Option{
					modrlm.WithContextPolicyPreset(modrlm.ContextPolicyFull),
				}
			},
		},
		{
			Name:        string(modrlm.ContextPolicyCheckpointed),
			Description: "Summarize older iterations and keep only the most recent one verbatim.",
			Options: func(_ int) []modrlm.Option {
				return []modrlm.Option{
					modrlm.WithContextPolicyPreset(modrlm.ContextPolicyCheckpointed),
					modrlm.WithHistoryCompression(1, 200),
				}
			},
		},
		{
			Name:        string(modrlm.ContextPolicyAdaptive),
			Description: "Start with full replay, then checkpoint once the history reaches the threshold.",
			Options: func(threshold int) []modrlm.Option {
				return []modrlm.Option{
					modrlm.WithContextPolicyPreset(modrlm.ContextPolicyAdaptive),
					modrlm.WithAdaptiveCheckpointThreshold(threshold),
				}
			},
		},
	}

	fmt.Println("=== RLM Context Policy Comparison ===")
	fmt.Printf("Provider: %s\n", *provider)
	fmt.Printf("Model: %s\n", modelName)
	fmt.Printf("Context size: %d characters\n", len(document))
	fmt.Printf("Adaptive threshold: %d\n", *adaptiveThreshold)
	fmt.Printf("Max iterations: %d\n", *maxIters)
	fmt.Println()
	fmt.Printf("Query: %s\n", query)
	fmt.Println()

	for _, policy := range policies {
		summary := runPolicy(ctx, llm, policy, *adaptiveThreshold, *maxIters, *verbose, document, query)
		printSummary(summary)
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
			modelName = string(anthropic.ModelClaude_3_Haiku_20240307)
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

func reviewDocument(repeats int) string {
	base := `
Review 1: This product is amazing, best purchase ever! Rating: 5 stars
Review 2: Terrible quality, broke after one day. Rating: 1 star
Review 3: It's okay, nothing special. Rating: 3 stars
Review 4: Absolutely love it, highly recommend! Rating: 5 stars
Review 5: Waste of money, very disappointed. Rating: 1 star
Review 6: Good value for the price. Rating: 4 stars
Review 7: Does exactly what it says. Rating: 4 stars
Review 8: Completely unusable, returned it. Rating: 1 star
Review 9: Best in class, exceeded expectations. Rating: 5 stars
Review 10: Average product, works fine. Rating: 3 stars
`
	if repeats <= 0 {
		repeats = 1
	}
	return strings.Repeat(base, repeats)
}

func runPolicy(
	ctx context.Context,
	llm core.LLM,
	policy policySpec,
	adaptiveThreshold int,
	maxIters int,
	verbose bool,
	document string,
	query string,
) policySummary {
	options := []modrlm.Option{
		modrlm.WithMaxIterations(maxIters),
		modrlm.WithTimeout(3 * time.Minute),
		modrlm.WithVerbose(verbose),
	}
	options = append(options, policy.Options(adaptiveThreshold)...)

	module := modrlm.NewFromLLM(llm, options...)
	agent := agentrlm.NewAgent("rlm-context-"+policy.Name, module)

	startedAt := time.Now()
	output, err := agent.Execute(ctx, map[string]interface{}{
		"context": document,
		"query":   query,
	})
	duration := time.Since(startedAt)

	return summarizePolicyRun(policy, agent.LastExecutionTrace(), output, err, duration)
}

func summarizePolicyRun(
	policy policySpec,
	trace *agents.ExecutionTrace,
	output map[string]interface{},
	err error,
	duration time.Duration,
) policySummary {
	summary := policySummary{
		Name:        policy.Name,
		Description: policy.Description,
		Duration:    duration,
	}
	if output != nil {
		if answer, ok := output["answer"].(string); ok {
			summary.Answer = strings.TrimSpace(answer)
		}
	}
	if err != nil {
		summary.Error = err.Error()
	}
	if trace == nil {
		if summary.Termination == "" {
			summary.Termination = "unknown"
		}
		return summary
	}

	summary.Termination = trace.TerminationCause
	summary.Steps = len(trace.Steps)
	summary.HistoryCompressions = intMetric(trace.ContextMetadata, modrlm.TraceMetadataHistoryCompressions)
	summary.SubLLMCalls = intMetric(trace.ContextMetadata, modrlm.TraceMetadataSubLLMCallCount)
	summary.RootPromptMeanTokens = intMetric(trace.ContextMetadata, modrlm.TraceMetadataRootPromptMeanTokens)
	summary.RootPromptMaxTokens = intMetric(trace.ContextMetadata, modrlm.TraceMetadataRootPromptMaxTokens)
	if summary.Duration <= 0 {
		summary.Duration = trace.ProcessingTime
	}
	if summary.Termination == "" {
		summary.Termination = "unknown"
	}
	return summary
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

func printSummary(summary policySummary) {
	fmt.Printf("[%s] %s\n", summary.Name, summary.Description)
	if summary.Error != "" {
		fmt.Printf("error: %s\n\n", summary.Error)
		return
	}

	fmt.Printf("termination: %s\n", summary.Termination)
	fmt.Printf("steps: %d\n", summary.Steps)
	fmt.Printf("history compressions: %d\n", summary.HistoryCompressions)
	fmt.Printf("sub-LLM calls: %d\n", summary.SubLLMCalls)
	fmt.Printf("root prompt mean tokens: %d\n", summary.RootPromptMeanTokens)
	fmt.Printf("root prompt max tokens: %d\n", summary.RootPromptMaxTokens)
	fmt.Printf("duration: %s\n", summary.Duration.Round(time.Millisecond))
	fmt.Printf("answer: %s\n\n", oneLine(summary.Answer, 160))
}

func oneLine(text string, maxLen int) string {
	text = strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
	if text == "" {
		return "(empty)"
	}
	if maxLen > 0 && len(text) > maxLen {
		return text[:maxLen] + "..."
	}
	return text
}
