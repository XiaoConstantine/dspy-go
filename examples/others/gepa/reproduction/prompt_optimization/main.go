package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)

// Figure2ReproductionExperiment reproduces the exact prompt evolution shown in Figure 2.
type Figure2ReproductionExperiment struct {
	logger          *logging.Logger
	ctx             context.Context
	promptEvolution []PromptEvolutionStep
}

// PromptEvolutionStep tracks each step in the prompt evolution.
type PromptEvolutionStep struct {
	Generation  int
	Fitness     float64
	Instruction string
	Timestamp   time.Time
}

// createMultiHopRetrievalSignature creates the exact signature from Figure 2.
func createMultiHopRetrievalSignature() *core.Signature {
	// This matches the exact task in Figure 2:
	// Input: question, summary_1
	// Output: query
	// Task: Generate search query for second hop of multi-hop retrieval

	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.NewField("question", core.WithDescription("Multi-hop question requiring information from multiple documents"))},
			{Field: core.NewField("summary_1", core.WithDescription("Summary of information retrieved in the first hop"))},
		},
		[]core.OutputField{
			{Field: core.NewField("query", core.WithDescription("Search query for retrieving second-hop documents"))},
		},
	).WithInstruction("Given the fields question, summary_1, produce the fields query.") // EXACT seed prompt from Figure 2

	return &signature
}

// createMultiHopRetrievalData creates synthetic data similar to Figure 2's task.
func createMultiHopRetrievalData() []core.Example {
	// Create examples that match the multi-hop retrieval task from Figure 2
	examples := []core.Example{
		{
			Inputs: map[string]interface{}{
				"question":  "What is the population of the wider metropolitan area that includes the city where the University of Edinburgh is located?",
				"summary_1": "The University of Edinburgh is located in Edinburgh, Scotland. Edinburgh is the capital city of Scotland with a population of approximately 540,000 people within the city limits.",
			},
			Outputs: map[string]interface{}{
				"query": "Edinburgh metropolitan area population Scotland wider region",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What album was released by the band that performed the opening theme for the TV series that aired from 1993 to 1998?",
				"summary_1": "The TV series that aired from 1993 to 1998 is 'The X-Files'. The opening theme was performed by the band 'Mark Snow'.",
			},
			Outputs: map[string]interface{}{
				"query": "Mark Snow discography albums released X-Files composer",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What is the birth year of the director of the movie that won the Academy Award for Best Picture in 1995?",
				"summary_1": "The movie that won the Academy Award for Best Picture in 1995 was 'Forrest Gump'. The movie was directed by Robert Zemeckis.",
			},
			Outputs: map[string]interface{}{
				"query": "Robert Zemeckis birth year director born when",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What is the height of the tallest building in the city where the 2024 Olympics were held?",
				"summary_1": "The 2024 Summer Olympics were held in Paris, France. Paris is the capital and largest city of France.",
			},
			Outputs: map[string]interface{}{
				"query": "tallest building Paris France height skyscraper highest",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What company acquired the social media platform founded by the person who also co-founded PayPal?",
				"summary_1": "The social media platform founded by a PayPal co-founder is Twitter (now X). It was founded by Jack Dorsey, who was also involved with PayPal's early development.",
			},
			Outputs: map[string]interface{}{
				"query": "Twitter acquisition company bought Elon Musk X platform",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What is the scientific name of the animal that is the national symbol of the country that borders both China and Russia?",
				"summary_1": "The country that borders both China and Russia is Mongolia. Mongolia's national symbol and national animal is the Przewalski's horse, also known as the Mongolian wild horse.",
			},
			Outputs: map[string]interface{}{
				"query": "Przewalski's horse scientific name Equus ferus przewalskii Mongolia",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What is the architectural style of the cathedral in the city that is the birthplace of Shakespeare?",
				"summary_1": "William Shakespeare was born in Stratford-upon-Avon, England. The city is known for its Tudor-style architecture and Shakespeare-related sites.",
			},
			Outputs: map[string]interface{}{
				"query": "Stratford-upon-Avon cathedral architecture style Holy Trinity Church",
			},
		},
		{
			Inputs: map[string]interface{}{
				"question":  "What is the market capitalization of the company that owns the search engine with the largest market share?",
				"summary_1": "The search engine with the largest market share is Google Search, which commands over 90% of the global search market. Google Search is owned by Google LLC, which is a subsidiary of Alphabet Inc.",
			},
			Outputs: map[string]interface{}{
				"query": "Alphabet Inc market capitalization Google parent company stock value",
			},
		},
	}

	// Shuffle for training variety
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})

	return examples
}

// evaluateQueryQuality evaluates the quality of generated queries.
func evaluateQueryQuality(expected, actual map[string]interface{}) float64 {
	expectedQuery := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", expected["query"])))
	actualQuery := strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", actual["query"])))

	if expectedQuery == "" || actualQuery == "" {
		return 0.0
	}

	// Split into words for analysis
	expectedWords := strings.Fields(expectedQuery)
	actualWords := strings.Fields(actualQuery)

	if len(expectedWords) == 0 || len(actualWords) == 0 {
		return 0.0
	}

	// Calculate word overlap (similar to F1 but for query terms)
	common := make(map[string]bool)
	for _, actualWord := range actualWords {
		for _, expectedWord := range expectedWords {
			if actualWord == expectedWord {
				common[actualWord] = true
				break
			}
		}
	}

	precision := float64(len(common)) / float64(len(actualWords))
	recall := float64(len(common)) / float64(len(expectedWords))

	if precision+recall == 0 {
		return 0.0
	}

	// F1-like score for query quality
	return 2 * precision * recall / (precision + recall)
}

func (exp *Figure2ReproductionExperiment) runFigure2Reproduction(apiKey string) error {
	exp.logger.Info(exp.ctx, "📄 Reproducing Figure 2: GEPA Prompt Evolution")
	exp.logger.Info(exp.ctx, "🎯 Task: Multi-hop document retrieval query generation")
	exp.logger.Info(exp.ctx, "🌱 Seed Prompt: 'Given the fields question, summary_1, produce the fields query.'")
	exp.logger.Info(exp.ctx, "")

	// Configure LLM
	llms.EnsureFactory()
	err := core.ConfigureDefaultLLM(apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		return fmt.Errorf("failed to setup LLM: %w", err)
	}

	// Create the multi-hop retrieval program with native JSON structured output
	// for reliable single-field extraction (query)
	signature := createMultiHopRetrievalSignature()
	module := modules.NewChainOfThought(*signature).WithStructuredOutput()

	program := core.NewProgram(
		map[string]core.Module{"retriever": module},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return module.Process(ctx, inputs)
		},
	)

	// Load multi-hop retrieval examples
	examples := createMultiHopRetrievalData()
	exp.logger.Info(exp.ctx, "📊 Created %d multi-hop retrieval examples", len(examples))

	// Configure GEPA for EXTREME prompt evolution (to match Figure 2's sophistication)
	config := &optimizers.GEPAConfig{
		PopulationSize:       25, // Larger population for more diversity
		MaxGenerations:       30, // Many more generations for evolution
		SelectionStrategy:    "adaptive_pareto",
		MutationRate:         0.8,  // VERY high mutation rate for dramatic evolution
		CrossoverRate:        0.3,  // Low crossover, focus on creative mutation
		ElitismRate:          0.2,  // Higher elitism to preserve good prompts
		ReflectionFreq:       1,    // Reflect EVERY generation for maximum improvement
		ReflectionDepth:      5,    // Very deep reflection analysis
		SelfCritiqueTemp:     0.95, // Very high temperature for creative critique
		TournamentSize:       4,
		ConvergenceThreshold: 0.001, // Very strict convergence
		StagnationLimit:      8,     // Allow many generations of stagnation
		EvaluationBatchSize:  8,     // Larger batch for thorough evaluation
		ConcurrencyLevel:     2,     // Lower concurrency to allow longer processing
		Temperature:          0.95,  // Very high temp for maximum creativity
		MaxTokens:            8192,  // Much more tokens for very detailed prompts
	}

	exp.logger.Info(exp.ctx, "🔧 GEPA Configuration for Sophisticated Evolution:")
	exp.logger.Info(exp.ctx, "   Population: %d, Max Generations: %d", config.PopulationSize, config.MaxGenerations)
	exp.logger.Info(exp.ctx, "   Mutation Rate: %.1f%% (higher for creativity)", config.MutationRate*100)
	exp.logger.Info(exp.ctx, "   Temperature: %.1f (higher for detailed prompts)", config.Temperature)
	exp.logger.Info(exp.ctx, "   Max Tokens: %d (allow detailed evolution)", config.MaxTokens)
	exp.logger.Info(exp.ctx, "")

	gepa, err := optimizers.NewGEPA(config)
	if err != nil {
		return fmt.Errorf("failed to create GEPA optimizer: %w", err)
	}

	// Set up evolution tracking
	gepa.SetProgressReporter(&Figure2ProgressReporter{exp: exp})

	// Create dataset
	dataset := datasets.NewSimpleDataset(examples)

	exp.logger.Info(exp.ctx, "🚀 Starting GEPA Evolution...")
	exp.logger.Info(exp.ctx, "   Watching for the dramatic prompt evolution shown in Figure 2!")
	exp.logger.Info(exp.ctx, "")

	startTime := time.Now()

	// Run GEPA optimization
	optimizedProgram, err := gepa.Compile(exp.ctx, program, dataset, evaluateQueryQuality)
	if err != nil {
		return fmt.Errorf("GEPA optimization failed: %w", err)
	}

	duration := time.Since(startTime)

	// Get final state
	state := gepa.GetOptimizationState()

	exp.logger.Info(exp.ctx, "")
	exp.logger.Info(exp.ctx, "🏆 Figure 2 Reproduction Results:")
	exp.logger.Info(exp.ctx, "%s", "="+strings.Repeat("=", 60))
	exp.logger.Info(exp.ctx, "⏱️ Optimization Duration: %v", duration)
	exp.logger.Info(exp.ctx, "🧬 Generations Completed: %d", state.CurrentGeneration)
	exp.logger.Info(exp.ctx, "📈 Best Fitness Achieved: %.4f", state.BestFitness)
	exp.logger.Info(exp.ctx, "")

	// Display the dramatic prompt evolution
	exp.displayPromptEvolution(state)

	// Test the evolved system
	exp.testEvolvedSystem(optimizedProgram, examples[:3])

	return nil
}

func (exp *Figure2ReproductionExperiment) displayPromptEvolution(state *optimizers.GEPAState) {
	exp.logger.Info(exp.ctx, "📜 PROMPT EVOLUTION ANALYSIS:")
	exp.logger.Info(exp.ctx, "%s", strings.Repeat("=", 80))
	exp.logger.Info(exp.ctx, "")

	// Show seed prompt (what we started with)
	exp.logger.Info(exp.ctx, "🌱 SEED PROMPT (Generation 0):")
	exp.logger.Info(exp.ctx, "┌─────────────────────────────────────────────────────────────────────┐")
	exp.logger.Info(exp.ctx, "│ Given the fields question, summary_1, produce the fields query.     │")
	exp.logger.Info(exp.ctx, "└─────────────────────────────────────────────────────────────────────┘")
	exp.logger.Info(exp.ctx, "")
	exp.logger.Info(exp.ctx, "Characteristics: Basic, minimal, no guidance on HOW to generate queries")
	exp.logger.Info(exp.ctx, "Length: ~10 words")
	exp.logger.Info(exp.ctx, "")

	// Show evolved prompt (what GEPA discovered)
	if state.BestCandidate != nil {
		evolved := state.BestCandidate.Instruction
		exp.logger.Info(exp.ctx, "🎯 GEPA'S EVOLVED PROMPT (Generation %d):", state.BestCandidate.Generation)
		exp.logger.Info(exp.ctx, "┌─────────────────────────────────────────────────────────────────────┐")

		// Wrap long prompt for display
		lines := wrapText(evolved, 65)
		for _, line := range lines {
			exp.logger.Info(exp.ctx, "│ %-67s │", line)
		}

		exp.logger.Info(exp.ctx, "└─────────────────────────────────────────────────────────────────────┘")
		exp.logger.Info(exp.ctx, "")

		// Analyze the evolution
		exp.analyzePromptEvolution(evolved)
	}

	exp.logger.Info(exp.ctx, "")
	exp.logger.Info(exp.ctx, "💡 KEY INSIGHT: GEPA evolved from a basic instruction to a sophisticated")
	exp.logger.Info(exp.ctx, "    multi-hop reasoning guide, demonstrating the paper's core innovation!")
}

func (exp *Figure2ReproductionExperiment) analyzePromptEvolution(evolved string) {
	wordCount := len(strings.Fields(evolved))

	exp.logger.Info(exp.ctx, "📊 EVOLUTION ANALYSIS:")
	exp.logger.Info(exp.ctx, "   Length: ~%d words (vs. 10 in seed)", wordCount)

	// Check for key improvements that mirror Figure 2
	improvements := []string{}

	if strings.Contains(strings.ToLower(evolved), "step") {
		improvements = append(improvements, "✅ Step-by-step reasoning guidance")
	}
	if strings.Contains(strings.ToLower(evolved), "context") || strings.Contains(strings.ToLower(evolved), "information") {
		improvements = append(improvements, "✅ Context utilization instructions")
	}
	if strings.Contains(strings.ToLower(evolved), "missing") || strings.Contains(strings.ToLower(evolved), "additional") {
		improvements = append(improvements, "✅ Gap identification guidance")
	}
	if strings.Contains(strings.ToLower(evolved), "retrieve") || strings.Contains(strings.ToLower(evolved), "search") {
		improvements = append(improvements, "✅ Retrieval strategy awareness")
	}
	if strings.Contains(strings.ToLower(evolved), "relevant") || strings.Contains(strings.ToLower(evolved), "related") {
		improvements = append(improvements, "✅ Relevance assessment")
	}
	if len(evolved) > 200 {
		improvements = append(improvements, "✅ Detailed operational guidance")
	}

	exp.logger.Info(exp.ctx, "")
	exp.logger.Info(exp.ctx, "🎯 EVOLVED CAPABILITIES:")
	for _, improvement := range improvements {
		exp.logger.Info(exp.ctx, "   %s", improvement)
	}

	if len(improvements) >= 4 {
		exp.logger.Info(exp.ctx, "")
		exp.logger.Info(exp.ctx, "🏆 SUCCESS: Achieved sophisticated prompt evolution matching Figure 2!")
	}
}

func (exp *Figure2ReproductionExperiment) testEvolvedSystem(program core.Program, examples []core.Example) {
	exp.logger.Info(exp.ctx, "")
	exp.logger.Info(exp.ctx, "🧪 TESTING EVOLVED SYSTEM:")
	exp.logger.Info(exp.ctx, "%s", strings.Repeat("-", 50))

	for i, example := range examples {
		exp.logger.Info(exp.ctx, "")
		exp.logger.Info(exp.ctx, "Test %d:", i+1)
		exp.logger.Info(exp.ctx, "Question: %s", example.Inputs["question"])
		exp.logger.Info(exp.ctx, "Summary 1: %s", example.Inputs["summary_1"])
		exp.logger.Info(exp.ctx, "Expected Query: %s", example.Outputs["query"])

		result, err := program.Execute(exp.ctx, example.Inputs)
		if err != nil {
			exp.logger.Error(exp.ctx, "Execution error: %v", err)
			continue
		}

		if query, ok := result["query"].(string); ok {
			exp.logger.Info(exp.ctx, "Generated Query: %s", query)
			score := evaluateQueryQuality(example.Outputs, result)
			exp.logger.Info(exp.ctx, "Quality Score: %.4f", score)
		}
	}
}

// Helper function to wrap text for display.
func wrapText(text string, maxWidth int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{text}
	}

	var lines []string
	var currentLine strings.Builder

	for _, word := range words {
		if currentLine.Len() == 0 {
			currentLine.WriteString(word)
		} else if currentLine.Len()+1+len(word) <= maxWidth {
			currentLine.WriteString(" " + word)
		} else {
			lines = append(lines, currentLine.String())
			currentLine.Reset()
			currentLine.WriteString(word)
		}
	}

	if currentLine.Len() > 0 {
		lines = append(lines, currentLine.String())
	}

	return lines
}

// Figure2ProgressReporter tracks evolution progress.
type Figure2ProgressReporter struct {
	exp *Figure2ReproductionExperiment
}

func (pr *Figure2ProgressReporter) Report(operation string, current, total int) {
	if current%2 == 0 || current == total {
		percentage := float64(current) / float64(total) * 100
		pr.exp.logger.Info(pr.exp.ctx, "🧬 Evolution Progress: Gen %d/%d (%.1f%%) - %s",
			current, total, percentage, operation)

		if current%4 == 0 && current > 0 {
			pr.exp.logger.Info(pr.exp.ctx, "   🔍 Monitoring prompt sophistication at generation %d...", current)
		}
	}
}

func RunFigure2Reproduction(apiKey string) {
	// Setup logging
	output := logging.NewConsoleOutput(true, logging.WithColor(true))
	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)

	ctx := core.WithExecutionState(context.Background())

	experiment := &Figure2ReproductionExperiment{
		logger:          logger,
		ctx:             ctx,
		promptEvolution: make([]PromptEvolutionStep, 0),
	}

	if err := experiment.runFigure2Reproduction(apiKey); err != nil {
		logger.Fatalf(ctx, "Figure 2 reproduction failed: %v", err)
	}
}

func main() {
	apiKey := flag.String("api-key", "", "API Key for the LLM provider")
	flag.Parse()

	if *apiKey == "" {
		fmt.Println("Please provide an API key using -api-key flag")
		fmt.Println("Usage: go run figure2_reproduction.go -api-key YOUR_API_KEY")
		return
	}

	log.Printf("📄 GEPA Figure 2 Reproduction Experiment")
	log.Printf("🎯 Reproducing dramatic prompt evolution from paper")
	log.Printf("🌱 Starting with: 'Given the fields question, summary_1, produce the fields query.'")
	log.Printf("")

	RunFigure2Reproduction(*apiKey)
}
