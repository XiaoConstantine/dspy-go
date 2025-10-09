package main

import (
	"context"
	"flag"
	"fmt"
	"strings"

	a2a "github.com/XiaoConstantine/dspy-go/pkg/agents/communication"
	"github.com/XiaoConstantine/dspy-go/pkg/agents"
	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
)

// ============================================================================
// Search Agent - Performs web searches and query refinement
// ============================================================================

type SearchAgent struct {
	searchModule core.Module
}

func NewSearchAgent() (*SearchAgent, error) {
	// Create signature for search query generation and results
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "topic", Description: "The research topic or question"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "search_queries", Description: "List of 3-5 specific search queries to gather comprehensive information", Prefix: "search queries:"}},
			{Field: core.Field{Name: "search_results", Description: "Simulated search results with relevant information", Prefix: "results:"}},
		},
	).WithInstruction(`You are a skilled research assistant specializing in information gathering.
Your task is to:
1. Generate 3-5 specific, targeted search queries based on the topic
2. Simulate finding high-quality, diverse search results for each query
3. Include key facts, statistics, and different perspectives
4. Organize results by query for clarity`)

	return &SearchAgent{
		searchModule: modules.NewPredict(signature),
	}, nil
}

func (s *SearchAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	logger.Info(ctx, "🔍 SearchAgent: Gathering information...")

	result, err := s.searchModule.Process(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	return result, nil
}

func (s *SearchAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *SearchAgent) GetMemory() agents.Memory {
	return nil
}

// ============================================================================
// Analysis Agent - Analyzes and extracts insights from search results
// ============================================================================

type AnalysisAgent struct {
	analysisModule core.Module
}

func NewAnalysisAgent() (*AnalysisAgent, error) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "topic", Description: "The research topic"}},
			{Field: core.Field{Name: "search_results", Description: "Search results to analyze"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "key_findings", Description: "5-7 key findings extracted from the search results", Prefix: "key findings:"}},
			{Field: core.Field{Name: "patterns", Description: "Common patterns or themes identified", Prefix: "patterns:"}},
			{Field: core.Field{Name: "contradictions", Description: "Any contradictions or disagreements found", Prefix: "contradictions:"}},
			{Field: core.Field{Name: "gaps", Description: "Information gaps that need further research", Prefix: "gaps:"}},
		},
	).WithInstruction(`You are an expert analytical researcher.
Analyze the provided search results and:
1. Extract the most important and relevant findings
2. Identify common patterns, themes, or trends
3. Note any contradictions or conflicting information
4. Highlight gaps in the current information
Be critical, thorough, and evidence-based.`)

	return &AnalysisAgent{
		analysisModule: modules.NewPredict(signature),
	}, nil
}

func (a *AnalysisAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	logger.Info(ctx, "📊 AnalysisAgent: Analyzing search results...")

	result, err := a.analysisModule.Process(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("analysis failed: %w", err)
	}

	return result, nil
}

func (a *AnalysisAgent) GetCapabilities() []core.Tool {
	return nil
}

func (a *AnalysisAgent) GetMemory() agents.Memory {
	return nil
}

// ============================================================================
// Synthesis Agent - Creates coherent reports from analyzed information
// ============================================================================

type SynthesisAgent struct {
	synthesisModule core.Module
}

func NewSynthesisAgent() (*SynthesisAgent, error) {
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "topic", Description: "The research topic"}},
			{Field: core.Field{Name: "key_findings", Description: "Key findings from analysis"}},
			{Field: core.Field{Name: "patterns", Description: "Identified patterns"}},
			{Field: core.Field{Name: "contradictions", Description: "Contradictions found"}},
			{Field: core.Field{Name: "gaps", Description: "Information gaps"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "executive_summary", Description: "2-3 paragraph executive summary", Prefix: "executive summary:"}},
			{Field: core.Field{Name: "detailed_report", Description: "Comprehensive research report with sections", Prefix: "detailed report:"}},
			{Field: core.Field{Name: "conclusions", Description: "Evidence-based conclusions", Prefix: "conclusions:"}},
			{Field: core.Field{Name: "recommendations", Description: "Recommendations for further research or action", Prefix: "recommendations:"}},
		},
	).WithInstruction(`You are a senior research analyst specializing in synthesizing complex information.
Create a comprehensive research report that:
1. Provides a clear executive summary of the key points
2. Presents a detailed, well-structured report with clear sections
3. Draws evidence-based conclusions
4. Offers actionable recommendations
Use clear, professional language. Structure the content logically.`)

	return &SynthesisAgent{
		synthesisModule: modules.NewPredict(signature),
	}, nil
}

func (s *SynthesisAgent) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	logger.Info(ctx, "📝 SynthesisAgent: Creating research report...")

	result, err := s.synthesisModule.Process(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("synthesis failed: %w", err)
	}

	return result, nil
}

func (s *SynthesisAgent) GetCapabilities() []core.Tool {
	return nil
}

func (s *SynthesisAgent) GetMemory() agents.Memory {
	return nil
}

// ============================================================================
// Research Orchestrator - Coordinates the multi-agent research workflow
// ============================================================================

type ResearchOrchestrator struct {
	executor *a2a.A2AExecutor
}

func NewResearchOrchestrator() (*ResearchOrchestrator, *a2a.A2AExecutor) {
	agent := &ResearchOrchestrator{}
	executor := a2a.NewExecutorWithConfig(agent, a2a.ExecutorConfig{
		Name: "ResearchOrchestrator",
	})
	agent.executor = executor
	return agent, executor
}

func (r *ResearchOrchestrator) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	logger := logging.GetLogger()
	topic, ok := input["topic"].(string)
	if !ok {
		return nil, fmt.Errorf("input 'topic' is required and must be a string")
	}

	logger.Info(ctx, "\n🎯 Research Orchestrator: Starting deep research on: %s\n", topic)
	logger.Info(ctx, "%s", strings.Repeat("=", 80))

	// Step 1: Search for information
	logger.Info(ctx, "\nStep 1/3: Information Gathering")
	logger.Info(ctx, "%s", strings.Repeat("-", 80))

	searchResult, err := r.executor.CallSubAgent(ctx, "search", a2a.NewUserMessage(topic))
	if err != nil {
		return nil, fmt.Errorf("search phase failed: %w", err)
	}

	searchQueries := ""
	if sq, ok := searchResult.Parts[0].Metadata["field"].(string); ok && sq == "search_queries" {
		searchQueries = searchResult.Parts[0].Text
	}
	searchResults := ""
	if len(searchResult.Parts) > 1 {
		searchResults = searchResult.Parts[1].Text
	}

	logger.Info(ctx, "✓ Search completed. Found %d result sets.", len(searchResult.Parts))

	// Step 2: Analyze the search results
	logger.Info(ctx, "\nStep 2/3: Information Analysis")
	logger.Info(ctx, "%s", strings.Repeat("-", 80))

	analysisInput := a2a.NewMessage(a2a.RoleUser,
		a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
		a2a.NewTextPartWithMetadata(searchResults, map[string]interface{}{"field": "search_results"}),
	)

	analysisResult, err := r.executor.CallSubAgent(ctx, "analysis", analysisInput)
	if err != nil {
		return nil, fmt.Errorf("analysis phase failed: %w", err)
	}

	logger.Info(ctx, "✓ Analysis completed. Identified key findings and patterns.")

	// Extract analysis results
	keyFindings := ""
	patterns := ""
	contradictions := ""
	gaps := ""

	for _, part := range analysisResult.Parts {
		if field, ok := part.Metadata["field"].(string); ok {
			switch field {
			case "key_findings":
				keyFindings = part.Text
			case "patterns":
				patterns = part.Text
			case "contradictions":
				contradictions = part.Text
			case "gaps":
				gaps = part.Text
			}
		}
	}

	// Step 3: Synthesize the final report
	logger.Info(ctx, "\nStep 3/3: Report Synthesis")
	logger.Info(ctx, "%s", strings.Repeat("-", 80))

	synthesisInput := a2a.NewMessage(a2a.RoleUser,
		a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
		a2a.NewTextPartWithMetadata(keyFindings, map[string]interface{}{"field": "key_findings"}),
		a2a.NewTextPartWithMetadata(patterns, map[string]interface{}{"field": "patterns"}),
		a2a.NewTextPartWithMetadata(contradictions, map[string]interface{}{"field": "contradictions"}),
		a2a.NewTextPartWithMetadata(gaps, map[string]interface{}{"field": "gaps"}),
	)

	synthesisResult, err := r.executor.CallSubAgent(ctx, "synthesis", synthesisInput)
	if err != nil {
		return nil, fmt.Errorf("synthesis phase failed: %w", err)
	}

	logger.Info(ctx, "✓ Report synthesis completed.")
	logger.Info(ctx, "\n%s", strings.Repeat("=", 80))

	// Compile final output
	output := map[string]interface{}{
		"topic":           topic,
		"search_queries":  searchQueries,
		"search_results":  searchResults,
		"key_findings":    keyFindings,
		"patterns":        patterns,
		"contradictions":  contradictions,
		"gaps":            gaps,
	}

	// Add synthesis results
	for _, part := range synthesisResult.Parts {
		if field, ok := part.Metadata["field"].(string); ok {
			output[field] = part.Text
		}
	}

	return output, nil
}

func (r *ResearchOrchestrator) GetCapabilities() []core.Tool {
	return nil
}

func (r *ResearchOrchestrator) GetMemory() agents.Memory {
	return nil
}

// ============================================================================
// Helper function to print research report
// ============================================================================

func printResearchReport(ctx context.Context, result map[string]interface{}) {
	logger := logging.GetLogger()

	logger.Info(ctx, "\n╔════════════════════════════════════════════════════════════════╗")
	logger.Info(ctx, "║                    RESEARCH REPORT                             ║")
	logger.Info(ctx, "╚════════════════════════════════════════════════════════════════╝\n")

	if topic, ok := result["topic"].(string); ok {
		logger.Info(ctx, "📌 Topic: %s\n", topic)
	}

	if summary, ok := result["executive_summary"].(string); ok {
		logger.Info(ctx, "📋 EXECUTIVE SUMMARY")
		logger.Info(ctx, "%s", strings.Repeat("-", 80))
		logger.Info(ctx, "%s\n", summary)
	}

	if report, ok := result["detailed_report"].(string); ok {
		logger.Info(ctx, "📄 DETAILED REPORT")
		logger.Info(ctx, "%s", strings.Repeat("-", 80))
		logger.Info(ctx, "%s\n", report)
	}

	if conclusions, ok := result["conclusions"].(string); ok {
		logger.Info(ctx, "💡 CONCLUSIONS")
		logger.Info(ctx, "%s", strings.Repeat("-", 80))
		logger.Info(ctx, "%s\n", conclusions)
	}

	if recommendations, ok := result["recommendations"].(string); ok {
		logger.Info(ctx, "🎯 RECOMMENDATIONS")
		logger.Info(ctx, "%s", strings.Repeat("-", 80))
		logger.Info(ctx, "%s\n", recommendations)
	}
}

// ============================================================================
// Main Example
// ============================================================================

func main() {
	// Parse command-line flags
	apiKey := flag.String("api-key", "", "LLM API Key (required)")
	model := flag.String("model", "gemini-2.0-flash-exp", "Model to use")
	flag.Parse()

	if *apiKey == "" {
		fmt.Println("Error: --api-key is required")
		fmt.Println("Usage: go run main.go --api-key YOUR_API_KEY [--model MODEL_NAME]")
		return
	}

	// Set up context and logger
	ctx := context.Background()
	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs: []logging.Output{
			logging.NewConsoleOutput(false),
		},
	})
	logging.SetLogger(logger)
	log := logging.GetLogger()

	log.Info(ctx, "╔════════════════════════════════════════════════════════════════╗")
	log.Info(ctx, "║       A2A Deep Research Agent - Multi-Agent Composition        ║")
	log.Info(ctx, "╚════════════════════════════════════════════════════════════════╝\n")

	// Initialize LLM factory and configure default LLM
	log.Info(ctx, "⚙️  Configuring LLM: %s", *model)
	llms.EnsureFactory() // Initialize the LLM factory
	modelID := core.ModelID(*model)
	if err := core.ConfigureDefaultLLM(*apiKey, modelID); err != nil {
		log.Error(ctx, "Failed to configure LLM: %v", err)
		return
	}
	ctx = core.WithExecutionState(ctx)

	// Create specialized sub-agents
	log.Info(ctx, "🔧 Initializing research agents...")

	searchAgent, err := NewSearchAgent()
	if err != nil {
		log.Error(ctx, "Failed to create SearchAgent: %v", err)
		return
	}

	analysisAgent, err := NewAnalysisAgent()
	if err != nil {
		log.Error(ctx, "Failed to create AnalysisAgent: %v", err)
		return
	}

	synthesisAgent, err := NewSynthesisAgent()
	if err != nil {
		log.Error(ctx, "Failed to create SynthesisAgent: %v", err)
		return
	}

	// Wrap agents with A2A executors
	searchExec := a2a.NewExecutorWithConfig(searchAgent, a2a.ExecutorConfig{
		Name: "SearchAgent",
	})

	analysisExec := a2a.NewExecutorWithConfig(analysisAgent, a2a.ExecutorConfig{
		Name: "AnalysisAgent",
	})

	synthesisExec := a2a.NewExecutorWithConfig(synthesisAgent, a2a.ExecutorConfig{
		Name: "SynthesisAgent",
	})

	// Create research orchestrator and register sub-agents
	_, orchestratorExec := NewResearchOrchestrator()
	orchestratorExec.WithSubAgent("search", searchExec).
		WithSubAgent("analysis", analysisExec).
		WithSubAgent("synthesis", synthesisExec)

	log.Info(ctx, "✓ Research system ready with 3 specialized agents\n")

	// Example research topics
	topics := []string{
		"What are the latest advancements in quantum computing and their potential impact on cryptography?",
		"How is artificial intelligence being used to combat climate change?",
	}

	// Conduct research on each topic
	for i, topic := range topics {
		log.Info(ctx, "\n\n%s", strings.Repeat("█", 80))
		log.Info(ctx, "RESEARCH PROJECT %d/%d", i+1, len(topics))
		log.Info(ctx, "%s\n", strings.Repeat("█", 80))

		// Create message with metadata specifying the field name as "topic"
		msg := a2a.NewMessage(a2a.RoleUser,
			a2a.NewTextPartWithMetadata(topic, map[string]interface{}{"field": "topic"}),
		)
		artifact, err := orchestratorExec.Execute(ctx, msg)
		if err != nil {
			log.Error(ctx, "Research failed: %v", err)
			continue
		}

		// Convert artifact to result map
		result := make(map[string]interface{})
		result["topic"] = topic

		for _, part := range artifact.Parts {
			if field, ok := part.Metadata["field"].(string); ok {
				result[field] = part.Text
			}
		}

		printResearchReport(ctx, result)
	}

	log.Info(ctx, "\n%s", strings.Repeat("=", 80))
	log.Info(ctx, "✓ All research projects completed!")
	log.Info(ctx, "%s\n", strings.Repeat("=", 80))

	log.Info(ctx, "\n🎓 A2A Protocol Features Demonstrated:")
	log.Info(ctx, "  ✅ Multi-agent hierarchical composition")
	log.Info(ctx, "  ✅ In-process agent communication (no HTTP overhead)")
	log.Info(ctx, "  ✅ Agent capability registration and discovery")
	log.Info(ctx, "  ✅ Complex multi-step workflows with real LLMs")
	log.Info(ctx, "  ✅ Standardized message/artifact protocol")
}
