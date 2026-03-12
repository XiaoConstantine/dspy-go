package commands

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/benchmarks/tblite"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/spf13/cobra"
)

type terminalTaskCommandConfig struct {
	Provider        string
	Model           string
	APIKey          string
	MaxTurns        int
	MaxTokens       int
	Temperature     float64
	ToolOutputLimit int
	SystemPrompt    string
}

// NewAgentCommand exposes benchmark-oriented agent entry points.
func NewAgentCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "agent",
		Short: "Run agent-oriented tasks and benchmark adapters",
	}
	cmd.AddCommand(newRunTerminalTaskCommand(defaultTerminalTaskAgentFactory))
	return cmd
}

func newRunTerminalTaskCommand(factory func(*terminalTaskCommandConfig) (tblite.Agent, error)) *cobra.Command {
	cfg := &terminalTaskCommandConfig{}

	cmd := &cobra.Command{
		Use:   "run-terminal-task",
		Short: "Run one TBLite-style terminal task from JSON stdin",
		Long: `Read a tblite.TerminalTaskRequest from stdin, execute it with the native
tool-calling benchmark agent, and emit tblite.TerminalTaskResult JSON on stdout.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			req, err := decodeTerminalTaskRequest(cmd.InOrStdin())
			if err != nil {
				return err
			}
			if cfg.MaxTurns > 0 {
				req.MaxTurns = cfg.MaxTurns
			}

			agent, err := factory(cfg)
			if err != nil {
				return err
			}
			return runTerminalTaskCommand(cmd.Context(), agent, req, cmd.OutOrStdout())
		},
	}

	cmd.Flags().StringVar(&cfg.Provider, "provider", "google", "LLM provider used for default model/API key resolution")
	cmd.Flags().StringVar(&cfg.Model, "model", "", "Model ID to use (defaults from provider)")
	cmd.Flags().StringVar(&cfg.APIKey, "api-key", "", "Explicit API key (otherwise provider-specific environment variables are used)")
	cmd.Flags().IntVar(&cfg.MaxTurns, "max-turns", 0, "Override the task request max turns")
	cmd.Flags().IntVar(&cfg.MaxTokens, "max-tokens", 2048, "Max tokens per model response")
	cmd.Flags().Float64Var(&cfg.Temperature, "temperature", 0, "Sampling temperature for task execution")
	cmd.Flags().IntVar(&cfg.ToolOutputLimit, "tool-output-limit", 16384, "Max characters returned per tool observation")

	return cmd
}

func runTerminalTaskCommand(ctx context.Context, agent tblite.Agent, req tblite.TerminalTaskRequest, out io.Writer) error {
	if agent == nil {
		return fmt.Errorf("terminal task agent is required")
	}

	result, err := agent.RunTask(ctx, req)
	if err != nil {
		return err
	}

	encoder := json.NewEncoder(out)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(result)
}

func decodeTerminalTaskRequest(r io.Reader) (tblite.TerminalTaskRequest, error) {
	var req tblite.TerminalTaskRequest
	if err := json.NewDecoder(r).Decode(&req); err != nil {
		return tblite.TerminalTaskRequest{}, fmt.Errorf("decode terminal task request: %w", err)
	}
	if strings.TrimSpace(req.TaskID) == "" {
		return tblite.TerminalTaskRequest{}, fmt.Errorf("task_id is required")
	}
	if strings.TrimSpace(req.TaskDir) == "" {
		return tblite.TerminalTaskRequest{}, fmt.Errorf("task_dir is required")
	}
	if strings.TrimSpace(req.EnvironmentDir) == "" {
		return tblite.TerminalTaskRequest{}, fmt.Errorf("environment_dir is required")
	}
	return req, nil
}

func defaultTerminalTaskAgentFactory(cfg *terminalTaskCommandConfig) (tblite.Agent, error) {
	return newToolCallingBenchmarkAgent(cfg)
}

func newToolCallingBenchmarkAgent(cfg *terminalTaskCommandConfig) (*tblite.ToolCallingAgent, error) {
	llm, err := newTerminalTaskLLM(cfg)
	if err != nil {
		return nil, err
	}
	return newToolCallingBenchmarkAgentWithLLM(llm, cfg)
}

func newToolCallingBenchmarkAgentWithLLM(llm core.LLM, cfg *terminalTaskCommandConfig) (*tblite.ToolCallingAgent, error) {
	return tblite.NewToolCallingAgent(llm, tblite.ToolCallingAgentConfig{
		MaxTurns:    cfg.MaxTurns,
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		Toolset: tblite.ToolsetConfig{
			OutputLimit: cfg.ToolOutputLimit,
		},
		SystemPrompt: cfg.SystemPrompt,
	})
}

func newTerminalTaskLLM(cfg *terminalTaskCommandConfig) (core.LLM, error) {
	llms.EnsureFactory()

	provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
	if provider == "" {
		provider = "google"
	}

	modelID := strings.TrimSpace(cfg.Model)
	if modelID == "" {
		modelID = string(defaultModelForProvider(provider))
	}

	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		apiKey = resolveProviderAPIKey(provider)
	}
	if apiKey == "" {
		return nil, fmt.Errorf("api key required for provider %q", provider)
	}

	return llms.NewLLM(apiKey, core.ModelID(modelID))
}

func defaultModelForProvider(provider string) core.ModelID {
	switch provider {
	case "anthropic":
		return core.ModelAnthropicSonnet
	case "openai":
		return core.ModelOpenAIGPT4oMini
	case "google", "gemini":
		return core.ModelGoogleGeminiFlash
	default:
		return core.ModelGoogleGeminiFlash
	}
}

func resolveProviderAPIKey(provider string) string {
	switch provider {
	case "anthropic":
		return firstNonEmpty(os.Getenv("ANTHROPIC_API_KEY"), os.Getenv("DSPY_API_KEY"))
	case "openai":
		return firstNonEmpty(os.Getenv("OPENAI_API_KEY"), os.Getenv("DSPY_API_KEY"))
	case "google", "gemini":
		return firstNonEmpty(os.Getenv("GEMINI_API_KEY"), os.Getenv("GOOGLE_API_KEY"), os.Getenv("DSPY_API_KEY"))
	default:
		return firstNonEmpty(
			os.Getenv("DSPY_API_KEY"),
			os.Getenv("GEMINI_API_KEY"),
			os.Getenv("GOOGLE_API_KEY"),
			os.Getenv("OPENAI_API_KEY"),
			os.Getenv("ANTHROPIC_API_KEY"),
		)
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func (c *terminalTaskCommandConfig) ProviderOrDefault() string {
	if strings.TrimSpace(c.Provider) == "" {
		return "google"
	}
	return c.Provider
}

func (c *terminalTaskCommandConfig) ModelOrDefault() string {
	if strings.TrimSpace(c.Model) != "" {
		return c.Model
	}
	return string(defaultModelForProvider(c.ProviderOrDefault()))
}
