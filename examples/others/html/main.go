package main

import (
	"context"
	"flag"
	"strings"

	"github.com/XiaoConstantine/dspy-go/pkg/core"
	"github.com/XiaoConstantine/dspy-go/pkg/datasets"
	"github.com/XiaoConstantine/dspy-go/pkg/llms"
	"github.com/XiaoConstantine/dspy-go/pkg/logging"
	"github.com/XiaoConstantine/dspy-go/pkg/modules"
	"github.com/XiaoConstantine/dspy-go/pkg/optimizers"
)


// Function to create the HTML parsing program structure.
func createHTMLParsingProgram() core.Program {
	// Create a signature for extracting structured data from HTML
	extractSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "document"}}},
		[]core.OutputField{
			{Field: core.NewField("title")},
			{Field: core.NewField("headings")},
			{Field: core.NewField("entity_info")},
		},
	)

	// Create a ChainOfThought module for extraction
	extract := modules.NewChainOfThought(extractSignature)

	// Create the program
	program := core.NewProgram(
		map[string]core.Module{"extract_metadata": extract},
		func(ctx context.Context, inputs map[string]interface{}) (map[string]interface{}, error) {
			return extract.Process(ctx, inputs)
		},
	)
	return program
}

func cleanMarkdownCodeBlocks(input string) string {
	// Remove markdown code fence markers
	input = strings.TrimPrefix(input, "```html")
	input = strings.TrimPrefix(input, "```")
	input = strings.TrimSuffix(input, "```")
	input = strings.TrimSpace(input)
	return input
}

func createMetricFunc() func(example, prediction map[string]interface{}, ctx context.Context) bool {
	// Create a signature for our judge module
	signature := core.NewSignature(
		[]core.InputField{
			{Field: core.Field{Name: "document"}},
			{Field: core.Field{Name: "predicted_metadata"}},
		},
		[]core.OutputField{
			{Field: core.Field{Name: "rationale", Prefix: "rationale"}},
			{Field: core.Field{Name: "is_correct", Prefix: "is_correct"}},
		},
	).WithInstruction(`You are evaluating whether the predicted metadata correctly extracts information from the HTML document.
    First, analyze the document and identify the key information.
    Then, compare this with the predicted metadata.
    Explain your reasoning step by step, and finally determine if the prediction is correct.`)

	// Create a ChainOfThought module with this signature
	judge := modules.NewChainOfThought(signature)

	// Return a metric function that uses this module
	return func(example, prediction map[string]interface{}, ctx context.Context) bool {
		// Create inputs for the judge module
		inputs := map[string]interface{}{
			"document":           example["document"],
			"predicted_metadata": prediction,
		}

		// Execute the judge module
		result, err := judge.Process(ctx, inputs)
		if err != nil {
			// Handle error - in a metric, we might want to log and return false
			logging.GetLogger().Error(ctx, "Error in metric evaluation: %v", err)
			return false
		}

		// Extract and convert the is_correct field to a boolean
		isCorrectStr, ok := result["is_correct"].(string)
		if !ok {
			logging.GetLogger().Error(ctx, "Invalid is_correct type in result: %T", result["is_correct"])
			return false
		}

		// Parse the boolean string - could be more robust with regex or trimming
		isCorrect := strings.Contains(strings.ToLower(isCorrectStr), "true") ||
			strings.Contains(strings.ToLower(isCorrectStr), "yes")

		return isCorrect
	}
}

func main() {
	output := logging.NewConsoleOutput(true, logging.WithColor(true))

	logger := logging.NewLogger(logging.Config{
		Severity: logging.INFO,
		Outputs:  []logging.Output{output},
	})
	logging.SetLogger(logger)
	apiKey := flag.String("api-key", "", "Anthropic API Key")
	flag.Parse()
	ctx := core.WithExecutionState(context.Background())

	llms.EnsureFactory()

	// Configure the default LLM
	err := core.ConfigureDefaultLLM(*apiKey, core.ModelGoogleGeminiFlash)
	if err != nil {
		logger.Fatalf(ctx, "Failed to configure default LLM: %v", err)
	}

	// Create the base program structure
	html_program := createHTMLParsingProgram()

	// Q2: But I don't have HTML documents to use for optimization.
	// Create modules to synthesize data
	synthesizeTopicsSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "count"}}},
		[]core.OutputField{{Field: core.NewField("random_technical_topics")}},
	).WithInstruction(`Generate a list of diverse technical topics.
Each topic should be something that could have a web page with headings and entities.
Format your response as a list, DO NOT wrap response in json format`)
	synthesize_topics := modules.NewPredict(synthesizeTopicsSignature)

	synthesizeDocSignature := core.NewSignature(
		[]core.InputField{{Field: core.Field{Name: "topic"}}},
		[]core.OutputField{{Field: core.NewField("html_document")}},
	).WithInstruction(`Generate a realistic HTML document about the given topic.
The HTML should include a title tag in the head, multiple heading tags (h1, h2, etc.),
several paragraphs of content, and references to entity names (people, companies, technologies).
Use proper HTML structure with html, head, and body tags.`)
	synthesize_doc := modules.NewPredict(synthesizeDocSignature)

	// Generate topics
	topicsResult, err := synthesize_topics.Process(ctx, map[string]interface{}{
		"count": 5,
	})
	if err != nil {
		logger.Fatalf(ctx, "Failed to generate topics: %v", err)
	}
	// Parse topics from the result
	var topics []string
	if topicsContent, ok := topicsResult["random_technical_topics"]; ok && topicsContent != nil {
		logger.Info(ctx, "\nTopic content: %v", topicsContent)
		if topicsString, ok := topicsContent.(string); ok {
			logger.Info(ctx, "Parsing topics from multiline string...")
			lines := strings.Split(topicsString, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line != "" && !strings.HasPrefix(strings.ToLower(line), "random_technical_topics") {
					topics = append(topics, line)
				}
			}
		} else if topicsList, ok := topicsContent.([]interface{}); ok {
			logger.Info(ctx, "Topics already in list format, extracting...")
			for _, t := range topicsList {
				if topic, ok := t.(string); ok {
					topics = append(topics, topic)
				}
			}
		}
	}
	// Generate HTML documents for each topic
	var trainingExamples []core.Example
	for _, t := range topics {
		docResult, err := synthesize_doc.Process(ctx, map[string]interface{}{
			"topic": t,
		})
		if err != nil {
			logger.Error(ctx, "Failed to generate document for topic %v: %v", t, err)
			continue
		}
		if htmlDoc, ok := docResult["html_document"]; ok {
			htmlString, ok := htmlDoc.(string)
			if ok && htmlString != "" {
				htmlString = cleanMarkdownCodeBlocks(htmlString)
				example := core.Example{
					Inputs: map[string]interface{}{
						"document": htmlString,
					},
					Outputs: make(map[string]interface{}), // Leave empty for generation
				}
				trainingExamples = append(trainingExamples, example)
				logger.Info(ctx, "Created training example from %s document (%d chars)", t, len(htmlString))
			} else {
				logger.Error(ctx, "Generated document was empty or not a string for %s", t)
			}
		} else {
			logger.Error(ctx, "No html_document field in result for %s: %v", t, docResult)
		}
	}
	if len(trainingExamples) == 0 {
		logger.Fatal(ctx, "Failed to generate any valid training examples")
	}

	logger.Info(ctx, "Created %d training examples\n", len(trainingExamples))

	metricFunc := createMetricFunc()

	// Q3: How can I teach LLM to be better at this task?
	metric := func(example, prediction map[string]interface{}, ctx context.Context) float64 {
		if metricFunc(example, prediction, ctx) {
			return 1.0
		}
		logger := logging.GetLogger()
		logger.Info(ctx, "======= METRIC FALLBACK EVALUATION STARTED =======")
		_, ok := example["document"].(string)
		if !ok {
			logger.Error(ctx, "Document not found in example")
			return 0.0
		}
		title, hasTitle := prediction["title"].(string)
		headings, hasHeadings := prediction["headings"].(string)
		entities, hasEntities := prediction["entity_info"].(string)
		if !hasTitle || !hasHeadings || !hasEntities {
			logger.Error(ctx, "Missing required field in prediction: Title=%v, Headings=%v, Entities=%v", hasTitle, hasHeadings, hasEntities)
			return 0.0
		}
		if len(title) > 0 && len(headings) > 0 && len(entities) > 0 {
			logger.Info(ctx, "Metric fallback: Prediction seems complete, returning 0.7")
			return 0.7
		}
		logger.Info(ctx, "Metric fallback: Prediction incomplete, returning 0.1")
		return 0.1
	}

	// Create a bool-returning metric for the BootstrapFewShot constructor
	optimizerMetric := func(example, prediction map[string]interface{}, ctx context.Context) bool {
		score := metric(example, prediction, ctx) // Use the float64 metric internally
		return score >= 0.7                       // Threshold for pass/fail
	}

	// Create optimizer (BootstrapFewShot)
	maxDemos := 3
	optimizer := optimizers.NewBootstrapFewShot(optimizerMetric, maxDemos)

	// Compile the program
	logger.Info(ctx, "Compiling HTML extraction program...")
	
	// Create dataset from training examples
	trainDataset := datasets.NewSimpleDataset(trainingExamples)

	// Define metric function for optimizer compile
	compileMetric := func(expected, actual map[string]interface{}) float64 {
		// Simple exact match for demonstration
		if expected["title"] == actual["title"] {
			return 1.0
		}
		return 0.0
	}

	compiled_program, err := optimizer.Compile(ctx, html_program, trainDataset, compileMetric)
	if err != nil {
		logger.Fatalf(ctx, "Compilation failed: %v", err)
	}
	logger.Info(ctx, "Compilation complete.")

	// --- Demonstrate Save/Load --- //
	saveFilePath := "optimized_html_parser_state.json"
	logger.Info(ctx, "Saving optimized program state to %s...", saveFilePath)
	err = core.SaveProgram(&compiled_program, saveFilePath)
	if err != nil {
		logger.Error(ctx, "Failed to save program state: %v", err)
	} else {
		logger.Info(ctx, "Program state saved successfully.")

		// Create a new program instance and load the state
		logger.Info(ctx, "Creating new program instance to load state...")
		loaded_program := createHTMLParsingProgram() // Must recreate the same structure

		logger.Info(ctx, "Loading optimized program state from %s...", saveFilePath)
		err = core.LoadProgram(&loaded_program, saveFilePath)
		if err != nil {
			logger.Error(ctx, "Failed to load program state: %v", err)
			logger.Warn(ctx, "Using original compiled program due to load error.")
		} else {
			logger.Info(ctx, "Program state loaded successfully. Using loaded program.")
			compiled_program = loaded_program
		}
	}
	// --- End Save/Load Demonstration --- //

	// Evaluate the optimized program on one of the training examples (or a new one)
	testExample := trainingExamples[0]
	logger.Info(ctx, "\nEvaluating program on test example...")
	logger.Info(ctx, "Input Document:\n%s\n", testExample.Inputs["document"])

	result, err := compiled_program.Execute(ctx, testExample.Inputs)
	if err != nil {
		logger.Error(ctx, "Error executing compiled program: %v", err)
	} else {
		logger.Info(ctx, "\n--- Extracted Metadata ---")
		logger.Info(ctx, "Title: %s", result["title"])
		logger.Info(ctx, "Headings:\n%s", result["headings"])
		logger.Info(ctx, "Entity Info:\n%s", result["entity_info"])
		logger.Info(ctx, "------------------------")

		// Evaluate using the metric
		score := metric(testExample.Inputs, result, ctx)
		logger.Info(ctx, "Metric Score on Test Example: %.2f", score)
	}
}
