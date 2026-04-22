package main

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	modrlm "github.com/XiaoConstantine/dspy-go/pkg/modules/rlm"
)

type scenarioSummary struct {
	Name         string
	Termination  string
	Steps        int
	SubRLMCalls  int
	BudgetError  string
	FinalAnswer  string
	ExecutionErr string
}

type scriptedLLM struct {
	responses []string
	callCount int
}

func main() {
	query := flag.String("query", "Demonstrate how sub-RLM budgets block recursive delegation.", "Query to pass into the scripted budget demo")
	flag.Parse()

	scenarios := []struct {
		name string
		cfg  modrlm.SubRLMConfig
	}{
		{
			name: "direct-budget",
			cfg: modrlm.SubRLMConfig{
				MaxDepth:               3,
				MaxIterationsPerSubRLM: 2,
				MaxDirectSubRLMCalls:   1,
				MaxTotalSubRLMCalls:    4,
			},
		},
		{
			name: "total-budget",
			cfg: modrlm.SubRLMConfig{
				MaxDepth:               3,
				MaxIterationsPerSubRLM: 2,
				MaxDirectSubRLMCalls:   4,
				MaxTotalSubRLMCalls:    1,
			},
		},
	}

	fmt.Println("=== RLM Sub-RLM Budget Demo ===")
	fmt.Println("This example is scripted and deterministic.")
	fmt.Printf("Query: %s\n\n", strings.TrimSpace(*query))

	for _, scenario := range scenarios {
		summary := runBudgetScenario(context.Background(), *query, scenario.name, scenario.cfg)
		printScenario(summary)
	}
}

func runBudgetScenario(ctx context.Context, query, name string, cfg modrlm.SubRLMConfig) scenarioSummary {
	module := modrlm.New(
		&scriptedLLM{responses: scriptedBudgetResponses()},
		staticSubLLMClient{},
		modrlm.WithMaxIterations(3),
		modrlm.WithSubRLMConfig(cfg),
	)

	result, trace, err := module.CompleteWithTrace(ctx, "demo context", query)

	summary := scenarioSummary{
		Name: name,
	}
	if trace != nil {
		summary.Termination = trace.TerminationCause
		summary.Steps = len(trace.Steps)
		summary.SubRLMCalls = trace.SubRLMCallCount
		summary.BudgetError = firstBudgetError(trace)
	}
	if result != nil {
		summary.FinalAnswer = strings.TrimSpace(result.Response)
	}
	if err != nil {
		summary.ExecutionErr = err.Error()
	}
	return summary
}

func firstBudgetError(trace *modrlm.RLMTrace) string {
	if trace == nil {
		return ""
	}
	for _, step := range trace.Steps {
		if strings.Contains(step.Error, "budget exceeded") {
			return step.Error
		}
		if strings.Contains(step.Observation, "budget exceeded") {
			return step.Observation
		}
	}
	if strings.Contains(trace.Error, "budget exceeded") {
		return trace.Error
	}
	return ""
}

func printScenario(summary scenarioSummary) {
	fmt.Printf("[%s]\n", summary.Name)
	if summary.ExecutionErr != "" {
		fmt.Printf("execution error: %s\n\n", summary.ExecutionErr)
		return
	}
	fmt.Printf("termination: %s\n", summary.Termination)
	fmt.Printf("steps: %d\n", summary.Steps)
	fmt.Printf("successful sub-RLM calls: %d\n", summary.SubRLMCalls)
	fmt.Printf("budget error: %s\n", oneLine(summary.BudgetError, 120))
	fmt.Printf("final answer: %s\n\n", oneLine(summary.FinalAnswer, 120))
}

func scriptedBudgetResponses() []string {
	return []string{
		formatAction("Delegate the first sub-task.", "subrlm", "", "inspect the first chunk", ""),
		formatAction("Return the sub-result to the parent.", "final", "", "", "first chunk inspected"),
		formatAction("Delegate the second sub-task.", "subrlm", "", "inspect the second chunk", ""),
		formatAction("Report that the budget guardrail triggered.", "final", "", "", "budget guardrail demonstrated"),
	}
}

func formatAction(reasoning, action, code, subquery, answer string) string {
	return fmt.Sprintf(
		"Reasoning:\n%s\n\nAction:\n%s\n\nCode:\n%s\n\nSubQuery:\n%s\n\nAnswer:\n%s\n",
		reasoning,
		action,
		code,
		subquery,
		answer,
	)
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

func (m *scriptedLLM) Generate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	if m.callCount >= len(m.responses) {
		return nil, fmt.Errorf("no more scripted responses")
	}

	response := m.responses[m.callCount]
	m.callCount++
	usage := core.TokenInfo{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
	}
	return &core.LLMResponse{Content: response, Usage: &usage}, nil
}

func (m *scriptedLLM) GenerateWithJSON(ctx context.Context, prompt string, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("JSON generation not implemented in scripted budget demo")
}

func (m *scriptedLLM) GenerateWithFunctions(ctx context.Context, prompt string, functions []map[string]interface{}, opts ...core.GenerateOption) (map[string]interface{}, error) {
	return nil, fmt.Errorf("function calling not implemented in scripted budget demo")
}

func (m *scriptedLLM) GenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.LLMResponse, error) {
	return m.Generate(ctx, "", opts...)
}

func (m *scriptedLLM) StreamGenerate(ctx context.Context, prompt string, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in scripted budget demo")
}

func (m *scriptedLLM) StreamGenerateWithContent(ctx context.Context, content []core.ContentBlock, opts ...core.GenerateOption) (*core.StreamResponse, error) {
	return nil, fmt.Errorf("streaming not implemented in scripted budget demo")
}

func (m *scriptedLLM) CreateEmbedding(ctx context.Context, input string, opts ...core.EmbeddingOption) (*core.EmbeddingResult, error) {
	return nil, fmt.Errorf("embeddings not implemented in scripted budget demo")
}

func (m *scriptedLLM) CreateEmbeddings(ctx context.Context, inputs []string, opts ...core.EmbeddingOption) (*core.BatchEmbeddingResult, error) {
	return nil, fmt.Errorf("batch embeddings not implemented in scripted budget demo")
}

func (m *scriptedLLM) ProviderName() string { return "scripted" }
func (m *scriptedLLM) ModelID() string      { return "scripted-budget-demo" }
func (m *scriptedLLM) Capabilities() []core.Capability {
	return []core.Capability{core.CapabilityCompletion}
}

type staticSubLLMClient struct{}

func (staticSubLLMClient) Query(ctx context.Context, prompt string) (modrlm.QueryResponse, error) {
	return modrlm.QueryResponse{Response: "sub-llm placeholder", PromptTokens: 10, CompletionTokens: 5}, nil
}

func (staticSubLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]modrlm.QueryResponse, error) {
	results := make([]modrlm.QueryResponse, len(prompts))
	for i := range prompts {
		results[i] = modrlm.QueryResponse{Response: "sub-llm placeholder", PromptTokens: 10, CompletionTokens: 5}
	}
	return results, nil
}
